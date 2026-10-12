"""Model compile time: residency policy first, then policy-constrained autotuning.

compile_model decides the residency policy from minimum footprints, tunes every
stage class inside that policy with weights already resident, selects one config
per class within the unit capacity, and compiles the layers without allocating
the device. The returned session allocates, binds and runs at runtime.
"""

from dataclasses import asdict, dataclass
import math
import statistics
import time

import torch

from triton.backends.pim.residency import LAYER
from triton.backends.pim.runtime import PIMSession

from ._autotuner import (
    DEFAULT_CACHE_PATH, ConstraintEngine, PIMBenchmarkResult, PIMConfig, PIMConfigCache,
    SearchBudget, build_config_space, target_spec, tune,
)
from .config import PIMTorchConfig
from .model_policy import (
    Footprint, PolicyBudget, analyze_stages, coefficients, decide_policy, max_stage_rounds,
    minimum_footprint, policy_bytes, stage_constraint,
)
from .template import TORCH_GEMM_TEMPLATE_VERSION


MIB = 1 << 20


class StageModule(torch.nn.Module):
    """One stage class alone: a weight GEMM, or its independent activation GEMM instances."""

    def __init__(self, stage_class):
        super().__init__()
        self.key = stage_class.key
        for operand, weight in enumerate(stage_class.weights):
            if weight is not None:
                self.register_buffer(f"weight_{operand}", weight)

    def _gemm(self, left, right):
        if self.key.op == "aten.bmm.default":
            return torch.bmm(left, right)
        if self.key.input_dtype == "int8":
            return torch._int_mm(left, right)
        return torch.mm(left, right)

    def forward(self, *inputs):
        if self.key.has_weights:
            activation, = inputs
            left = self.weight_0 if self.key.weight_mask[0] else activation
            right = self.weight_1 if self.key.weight_mask[1] else activation
            return self._gemm(left, right)
        if self.key.op == "aten.bmm.default":
            return self._gemm(*inputs)
        return tuple(self._gemm(inputs[index], inputs[index + 1])
                     for index in range(0, len(inputs), 2))


def stage_inputs(key):
    m, n, k = key.mnk
    dtype = getattr(torch, key.input_dtype)

    def tensor(*shape):
        if dtype == torch.int8:
            return torch.randint(-8, 8, shape, dtype=dtype)
        return torch.randn(*shape, dtype=dtype)

    if key.has_weights:
        return (tensor(m, k) if key.weight_mask[1] else tensor(k, n),)
    if key.op == "aten.bmm.default":
        return (tensor(key.instances, m, k), tensor(key.instances, k, n))
    return tuple(operand for _ in range(key.instances) for operand in (tensor(m, k), tensor(k, n)))


def measure_stage(stage_class, spec, config, *, warmup, repeat):
    """E with the weights already resident, U of one upload, and measured footprint."""
    module = StageModule(stage_class).eval()
    inputs = stage_inputs(stage_class.key)

    def configure(gemms, spec):
        return {gemm.node: PIMTorchConfig(spec=spec, config=config) for gemm in gemms}

    # 후보마다 새로 컴파일한다. dynamo가 앞 후보의 컴파일 결과를 돌려주지 않게 한다.
    torch._dynamo.reset()
    with torch.no_grad(), PIMSession(spec) as session:
        plan, = session.prepare([module], inputs, configure=configure)
        requirement, = session.backend.requirements([plan])
        start = time.perf_counter_ns()
        session.backend.model.enter_layer(plan)
        upload_ms = (time.perf_counter_ns() - start) / 1e6
        for _ in range(warmup):
            session.run_model(*inputs)
        samples = []
        for _ in range(repeat):
            start = time.perf_counter_ns()
            session.run_model(*inputs)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return PIMBenchmarkResult(
        total_ms=statistics.median(samples),
        correctness=None,
        timings=(("weight_upload", upload_ms),),
        physical_allocations=((requirement.weight_bytes, requirement.state_bytes,
                               requirement.input_bytes, requirement.output_bytes),),
    )


@dataclass(frozen=True)
class Candidate:
    config: PIMConfig
    execution_ms: float
    upload_ms: float
    footprint: Footprint


def cache_version(policy, key):
    # stage class가 다르면 같은 (M,N,K)여도 측정 대상이 다르다.
    mask = "".join(str(int(flag)) for flag in key.weight_mask)
    return (f"{TORCH_GEMM_TEMPLATE_VERSION}|model-{policy}|{key.op}|w{mask}"
            f"|x{key.instances}")


def tune_stage_class(stage_class, spec, budget, minimum, *, cache_path, search_budget,
                     warmup, repeat, seed):
    """Measured candidates of one class that satisfy the policy constraint."""
    key = stage_class.key
    problem = key.problem
    constraint = stage_constraint(key, max_stage_rounds(problem, spec), budget)
    overrides = {"initial_round_patterns": ((1, 1, 1),)} if key.has_weights else {}
    cache = PIMConfigCache(cache_path, template_version=cache_version(budget.policy, key))

    def benchmark_fn(problem, target, config):
        return measure_stage(stage_class, target, config, warmup=warmup, repeat=repeat)

    print(f"[model compile] tune {key.label()}", flush=True)
    tune(problem, spec, benchmark_fn, space=build_config_space(problem, spec, **overrides),
         cache=cache, budget=search_budget, seed=seed, verbose=True,
         constraint=constraint, initial=(minimum.config,))
    # cache에는 다른 예산으로 잰 기록도 있을 수 있으므로 지금 제약으로 다시 거른다.
    engine = ConstraintEngine(problem, spec, constraint=constraint)
    candidates = []
    for values, record in cache.measured_results(problem, spec).items():
        config = PIMConfig(*values)
        if not engine.valid(config):
            continue
        timings = dict(record["timings"])
        candidates.append(Candidate(config, record["total_ms"], timings["weight_upload"],
                                    Footprint(*record["physical_allocations"][0])))
    if not candidates:
        raise ValueError(f"{key.label()}: no measured config satisfies policy {budget.policy}")
    return candidates


def candidate_time(policy, stage_class, candidate):
    """Time one request spends in this class: every stage runs once, P-3 also uploads."""
    upload = candidate.upload_ms if policy == LAYER else 0.0
    return stage_class.total * (candidate.execution_ms + upload)


def needed_bytes(policy, classes, choice):
    resident = sum(policy_bytes(policy, stage_class, choice[stage_class.key].footprint)
                   for stage_class in classes)
    return (resident
            + max(choice[stage_class.key].footprint.input for stage_class in classes)
            + max(choice[stage_class.key].footprint.output for stage_class in classes))


def _knapsack(classes, candidates, policy, capacity_mib):
    """Multiple-choice knapsack: one candidate per class, MiB-rounded bytes."""
    best = [0.0] + [math.inf] * capacity_mib
    picks = []
    for stage_class in classes:
        items = [(math.ceil(policy_bytes(policy, stage_class, candidate.footprint) / MIB),
                  candidate_time(policy, stage_class, candidate), candidate)
                 for candidate in candidates[stage_class.key]]
        following = [math.inf] * (capacity_mib + 1)
        pick = [None] * (capacity_mib + 1)
        for used, total in enumerate(best):
            if total == math.inf:
                continue
            for cost, elapsed, candidate in items:
                if used + cost <= capacity_mib and total + elapsed < following[used + cost]:
                    following[used + cost] = total + elapsed
                    pick[used + cost] = (used, candidate)
        best = following
        picks.append(pick)
    end = min(range(capacity_mib + 1), key=best.__getitem__)
    if best[end] == math.inf:
        return None
    choice = {}
    used = end
    for stage_class, pick in zip(reversed(classes), reversed(picks)):
        used, candidate = pick[used]
        choice[stage_class.key] = candidate
    return best[end], choice


def select_configs(classes, candidates, policy, capacity):
    """Fastest config per class if it fits; otherwise DP within the capacity."""
    fastest = {stage_class.key: min(candidates[stage_class.key],
                                    key=lambda candidate: candidate_time(policy, stage_class, candidate))
               for stage_class in classes}
    if needed_bytes(policy, classes, fastest) <= capacity:
        return "fastest", fastest
    # scratch는 stage 사이의 max라 합이 아니다. 상한 (I, O)를 열거하고 그 안에서 DP를 푼다.
    inputs = sorted({candidate.footprint.input for group in candidates.values() for candidate in group})
    outputs = sorted({candidate.footprint.output for group in candidates.values() for candidate in group})
    best = None
    for input_limit in inputs:
        for output_limit in outputs:
            allowed = {key: [candidate for candidate in group
                             if candidate.footprint.input <= input_limit
                             and candidate.footprint.output <= output_limit]
                       for key, group in candidates.items()}
            if not all(allowed.values()) or capacity < input_limit + output_limit:
                continue
            solved = _knapsack(classes, allowed, policy,
                               (capacity - input_limit - output_limit) // MIB)
            if solved is not None and (best is None or solved[0] < best[0]):
                best = solved
    if best is None:
        raise ValueError(f"no combination of measured configs fits policy {policy} "
                         f"in {capacity} bytes per unit")
    return "dp", best[1]


@dataclass
class CompiledModel:
    session: PIMSession
    policy: str
    report: dict


def compile_model(layers, example_inputs, *, target, units, runtime_options=None,
                  cache_path=DEFAULT_CACHE_PATH, max_benchmarks=20, max_generations=10,
                  warmup=1, repeat=3, seed=0):
    """Compile time of the whole model; the returned session runs it at runtime."""
    layers = tuple(layers)
    spec = target_spec(target, num_units=units, **(runtime_options or {}))
    profile = spec.common_profile
    capacity = profile.banks_per_unit * profile.bank_capacity_limit_bytes

    start = time.perf_counter_ns()
    classes, assignments = analyze_stages(layers, example_inputs)
    minimum = {stage_class.key: minimum_footprint(stage_class, spec) for stage_class in classes}
    residency = decide_policy(classes, minimum, spec)
    policy = residency.policy
    print(f"[model compile] target={target} units={units} policy={policy} "
          f"lower_bound_bytes={residency.bytes_per_unit} capacity={capacity}", flush=True)
    analysis_ms = (time.perf_counter_ns() - start) / 1e6

    start = time.perf_counter_ns()
    budget = PolicyBudget(policy, classes, minimum, spec)
    search_budget = SearchBudget(max_generations=max_generations, max_benchmarks=max_benchmarks)
    candidates = {
        stage_class.key: tune_stage_class(
            stage_class, spec, budget, minimum[stage_class.key], cache_path=cache_path,
            search_budget=search_budget, warmup=warmup, repeat=repeat, seed=seed)
        for stage_class in classes
    }
    method, choice = select_configs(classes, candidates, policy, capacity)
    tuning_ms = (time.perf_counter_ns() - start) / 1e6

    start = time.perf_counter_ns()
    session = PIMSession(spec)
    # PIMSession은 layer 순서대로 configure를 한 번씩 부른다.
    layer_assignments = iter(assignments)

    def configure(gemms, spec):
        assignment = next(layer_assignments)
        return {gemm.node: PIMTorchConfig(spec=spec, config=choice[assignment[gemm.node]].config)
                for gemm in gemms}

    torch._dynamo.reset()
    session.compile(layers, example_inputs, configure=configure)
    if session.residency.policy != policy:
        raise ValueError(f"compiled layers need policy {session.residency.policy}; "
                         f"compile time decided {policy}")
    compile_ms = (time.perf_counter_ns() - start) / 1e6

    report = {
        "target": target, "units": units, "spec": asdict(spec), "capacity_bytes": capacity,
        "policy": policy, "policy_lower_bound_bytes": residency.bytes_per_unit,
        "selection": method,
        "needed_bytes": needed_bytes(policy, classes, choice),
        "compiled_bytes_per_unit": session.residency.bytes_per_unit,
        "estimated_request_ms": sum(candidate_time(policy, stage_class, choice[stage_class.key])
                                    for stage_class in classes),
        "analysis_ms": analysis_ms, "tuning_ms": tuning_ms, "compile_ms": compile_ms,
        "stage_classes": [{
            "label": stage_class.key.label(),
            "mnk": list(stage_class.key.mnk), "op": stage_class.key.op,
            "weight_mask": list(stage_class.key.weight_mask),
            "instances": stage_class.key.instances,
            "stages_per_layer": stage_class.counts,
            "weight_coefficient": coefficients(policy, stage_class)[0],
            "state_coefficient": coefficients(policy, stage_class)[1],
            "minimum": {"config": asdict(minimum[stage_class.key].config),
                        "footprint": asdict(minimum[stage_class.key].footprint)},
            "measured": len(candidates[stage_class.key]),
            "selected": {"config": asdict(choice[stage_class.key].config),
                         "execution_ms": choice[stage_class.key].execution_ms,
                         "upload_ms": choice[stage_class.key].upload_ms,
                         "footprint": asdict(choice[stage_class.key].footprint)},
        } for stage_class in classes],
    }
    return CompiledModel(session, policy, report)
