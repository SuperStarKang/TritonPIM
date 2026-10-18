"""Model compile time: tiling configs of one layer, then every layer compiled with them.

compile_model analyzes the first layer into stage classes (after GEMM packing),
measures each class with its weights already resident and keeps the fastest
config, and compiles every layer with those configs without allocating the
device. plan_memory then fits as many layer slots as one unit holds: every layer
keeps its own slot if all fit; otherwise the first slots keep one layer each and
the last slot is shared, so a layer change in it uploads only its weights. The
returned session allocates, binds and runs at runtime.
"""

from dataclasses import asdict, dataclass, replace
import math
import multiprocessing
import statistics
import time

import torch

from triton.backends.pim.runtime import PIMSession

from ._autotuner import (
    DEFAULT_CACHE_PATH, ConstraintEngine, PIMBenchmarkResult, PIMConfig, PIMConfigCache,
    SearchBudget, TuningRecord, build_config_space, pim_hierarchy, target_spec, tune,
)
from .config import PIMTorchConfig
from .model_policy import Footprint, analyze_layer, max_stage_rounds, stage_constraint
from .template import TORCH_GEMM_TEMPLATE_VERSION


class StageModule(torch.nn.Module):
    """One stage class alone: a weight GEMM, or its independent activation GEMM instances."""

    def __init__(self, key, weights):
        super().__init__()
        self.key = key
        for operand, weight in enumerate(weights):
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


def random_tensor(dtype_name, shape):
    # 측정값(E, U, footprint)은 값이 아니라 shape과 dtype으로 정해진다.
    dtype = getattr(torch, dtype_name)
    if dtype == torch.int8:
        return torch.randint(-8, 8, shape, dtype=dtype)
    return torch.randn(*shape, dtype=dtype)


def stage_inputs(key):
    m, n, k = key.mnk

    def tensor(*shape):
        return random_tensor(key.input_dtype, shape)

    if key.has_weights:
        return (tensor(m, k) if key.weight_mask[1] else tensor(k, n),)
    if key.op == "aten.bmm.default":
        return (tensor(key.instances, m, k), tensor(key.instances, k, n))
    return tuple(operand for _ in range(key.instances) for operand in (tensor(m, k), tensor(k, n)))


def measure_stage(key, weights, spec, config, *, warmup, repeat):
    """E with the weights already resident, U of one upload, and measured footprint."""
    module = StageModule(key, weights).eval()
    inputs = stage_inputs(key)

    def configure(gemms, spec):
        return {gemm.node: PIMTorchConfig(spec=spec, config=config) for gemm in gemms}

    # 후보마다 새로 컴파일한다. dynamo가 앞 후보의 컴파일 결과를 돌려주지 않게 한다.
    torch._dynamo.reset()
    with torch.no_grad(), PIMSession(spec) as session:
        plan, = session.prepare([module], inputs, configure=configure)
        stage, = plan.stages
        # packing이 측정할 stage의 shape를 바꾸지 않았는지 확인한다.
        if (tuple(stage.shape), stage.instance_count) != (key.mnk, key.instances):
            raise ValueError(f"{key.label()}: the measured module became stage "
                             f"{tuple(stage.shape)} x{stage.instance_count}")
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


def _measure_child(queue, key, weight_shapes, spec, config, warmup, repeat):
    weights = tuple(None if shape is None else random_tensor(key.input_dtype, shape)
                    for shape in weight_shapes)
    queue.put(measure_stage(key, weights, spec, config, warmup=warmup, repeat=repeat))


def measure_isolated(stage_class, spec, config, *, warmup, repeat):
    """Measure one candidate in a fresh process.

    Native runtime state (device contexts, stub libraries, host DMA buffers) dies
    with the process, so candidates do not accumulate it. A runtime failure ends
    only that process; the candidate is recorded as failed (correctness=False,
    exit code in timings) and the search continues without it.
    """
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    weight_shapes = tuple(None if weight is None else tuple(weight.shape)
                          for weight in stage_class.weights)
    process = context.Process(target=_measure_child, args=(
        queue, stage_class.key, weight_shapes, spec, config, warmup, repeat))
    process.start()
    process.join()
    if process.exitcode != 0:
        print(f"\n[model compile] candidate {config.key()} failed: measurement process "
              f"exit code {process.exitcode}", flush=True)
        return PIMBenchmarkResult(total_ms=math.inf, correctness=False,
                                  timings=(("exit_code", process.exitcode),))
    return queue.get()


@dataclass(frozen=True)
class Candidate:
    config: PIMConfig
    execution_ms: float
    upload_ms: float
    footprint: Footprint


def cache_version(key):
    # E, U, footprint는 배치와 무관하다. stage class가 다르면 같은 (M,N,K)여도 측정 대상이 다르다.
    mask = "".join(str(int(flag)) for flag in key.weight_mask)
    return f"{TORCH_GEMM_TEMPLATE_VERSION}|model|{key.op}|w{mask}|x{key.instances}"


def widened_configs(stage_class, spec, cache_path):
    """Cached best configs of the GEMMs a packed class concatenated, widened along its axis.

    Each keeps its blocks, groups, partitions and M/N/K rounds; only the tile on the
    packed axis grows until the Execution Tiles cover the packed extent.
    """
    if stage_class.origin is None:
        return ()
    axis, shapes, instances = stage_class.origin
    key = stage_class.key
    index = "mnk".index(axis)
    configs = []
    for mnk in dict.fromkeys(shapes):
        member = replace(key, mnk=mnk, instances=instances)
        config = PIMConfigCache(cache_path, template_version=cache_version(member)).lookup(member.problem, spec)
        # member class를 잰 적이 없으면 넓힐 config가 없다.
        if config is None:
            continue
        hierarchy = pim_hierarchy(member.problem, config)
        groups = math.ceil(key.mnk[index] / hierarchy.block_group_extent[index])
        partitions = (config.p_m, config.p_n, config.p_k)[index]
        tile = math.ceil(groups / (partitions * hierarchy.rounds[index]))
        configs.append(replace(config, **{f"tile_{axis}": tile}))
    return tuple(configs)


def tune_stage_class(stage_class, spec, *, cache_path, search_budget, warmup, repeat, seed):
    """Fastest measured config of one class and the number of measured candidates."""
    key = stage_class.key
    problem = key.problem
    constraint = stage_constraint(key, max_stage_rounds(problem, spec))
    overrides = {"initial_round_patterns": ((1, 1, 1),)} if key.has_weights else {}
    cache = PIMConfigCache(cache_path, template_version=cache_version(key))
    initial = widened_configs(stage_class, spec, cache_path)

    def benchmark_fn(problem, target, config):
        return measure_isolated(stage_class, target, config, warmup=warmup, repeat=repeat)

    print(f"[model compile] tune {key.label()} initial={[config.key() for config in initial]}", flush=True)
    tune(problem, spec, benchmark_fn, space=build_config_space(problem, spec, **overrides),
         cache=cache, budget=search_budget, seed=seed, verbose=True, constraint=constraint,
         initial=initial)
    # cache에는 다른 제약으로 잰 기록도 있을 수 있으므로 지금 제약으로 다시 거른다.
    engine = ConstraintEngine(problem, spec, constraint=constraint)
    # cache hit이면 tune은 탐색하지 않는다. 아직 재지 않은 initial 후보는 여기서 재서 기록에 더한다.
    measured = cache.measured_results(problem, spec)
    added = [TuningRecord(config, None, measure_isolated(stage_class, spec, config,
                                                        warmup=warmup, repeat=repeat))
             for config in initial if config.key() not in measured and engine.valid(config)]
    if added:
        config, result = cache.lookup_result(problem, spec)
        best = min([TuningRecord(config, None, result)] + [record for record in added if record.result.ok],
                   key=lambda record: record.result.total_ms)
        cache.store(problem, spec, best, added)
    candidates = []
    for values, record in cache.measured_results(problem, spec).items():
        config = PIMConfig(*values)
        # 측정 프로세스가 실패한 후보(correctness=False)는 고르지 않는다.
        if record["correctness"] is False or not engine.valid(config):
            continue
        timings = dict(record["timings"])
        candidates.append(Candidate(config, record["total_ms"], timings["weight_upload"],
                                    Footprint(*record["physical_allocations"][0])))
    if not candidates:
        raise ValueError(f"{key.label()}: no measured config satisfies the model-stage constraints")
    return min(candidates, key=lambda candidate: candidate.execution_ms), len(candidates)


@dataclass
class CompiledModel:
    session: PIMSession
    report: dict


def compile_model(layers, example_inputs, *, target, units, runtime_options=None,
                  cache_path=DEFAULT_CACHE_PATH, max_benchmarks=20, max_generations=10,
                  warmup=1, repeat=3, seed=0):
    """Compile time of a model whose layers share one structure; the session runs it at runtime.

    1. analyze the first layer into stage classes
    2. tune every class with its weights resident and keep the fastest measured config
    3. compile every layer with those configs; plan_memory places them in slots
    """
    layers = tuple(layers)
    spec = target_spec(target, num_units=units, **(runtime_options or {}))

    start = time.perf_counter_ns()
    classes, assignment = analyze_layer(layers[0], example_inputs, spec)
    analysis_ms = (time.perf_counter_ns() - start) / 1e6
    print(f"[model compile] target={target} units={units} layers={len(layers)} "
          f"stage_classes={len(classes)}", flush=True)

    start = time.perf_counter_ns()
    search_budget = SearchBudget(max_generations=max_generations, max_benchmarks=max_benchmarks)
    selected = {}
    measured = {}
    for stage_class in classes:
        selected[stage_class.key], measured[stage_class.key] = tune_stage_class(
            stage_class, spec, cache_path=cache_path, search_budget=search_budget,
            warmup=warmup, repeat=repeat, seed=seed)
    tuning_ms = (time.perf_counter_ns() - start) / 1e6

    start = time.perf_counter_ns()
    session = PIMSession(spec)

    # 모든 layer의 GEMM 이름이 분석한 layer와 같다. 없는 이름은 KeyError로 드러난다.
    def configure(gemms, spec):
        return {gemm.node: PIMTorchConfig(spec=spec, config=selected[assignment[gemm.node]].config)
                for gemm in gemms}

    torch._dynamo.reset()
    session.compile(layers, example_inputs, configure=configure)
    analyzed = [(tuple(stage.shape), stage.instance_count, tuple(stage.weight_mask))
                for stage in session.layers[0].stages]
    for plan in session.layers[1:]:
        stages = [(tuple(stage.shape), stage.instance_count, tuple(stage.weight_mask))
                  for stage in plan.stages]
        if stages != analyzed:
            raise ValueError(f"layer {plan.index} has stages {stages}; the analyzed layer has {analyzed}")
    compile_ms = (time.perf_counter_ns() - start) / 1e6
    memory = session.memory
    print(f"[model compile] slots={memory.slots} pinned={memory.pinned} "
          f"bytes_per_unit={memory.bytes_per_unit} capacity={memory.capacity_bytes} "
          f"analysis_ms={analysis_ms:.1f} tuning_ms={tuning_ms:.1f} compile_ms={compile_ms:.1f}", flush=True)

    report = {
        "target": target, "units": units, "spec": asdict(spec), "layers": len(layers),
        "memory": {"slots": memory.slots, "pinned": memory.pinned,
                   "slot_of_layer": memory.slot_of_layer, "slot_bytes": list(memory.slot_bytes),
                   "scratch_bytes": memory.scratch_bytes, "bytes_per_unit": memory.bytes_per_unit,
                   "capacity_bytes": memory.capacity_bytes},
        # weight 상주 실행 시간만 더한 값. weight 업로드와 CPU 시간은 들어가지 않는다.
        "estimated_execution_ms": len(layers) * sum(
            stage_class.count * selected[stage_class.key].execution_ms for stage_class in classes),
        "analysis_ms": analysis_ms, "tuning_ms": tuning_ms, "compile_ms": compile_ms,
        "stage_classes": [{
            "label": stage_class.key.label(),
            "mnk": list(stage_class.key.mnk), "op": stage_class.key.op,
            "weight_mask": list(stage_class.key.weight_mask),
            "instances": stage_class.key.instances,
            "stages_per_layer": stage_class.count,
            "measured": measured[stage_class.key],
            "selected": {"config": asdict(selected[stage_class.key].config),
                         "execution_ms": selected[stage_class.key].execution_ms,
                         # 격리 측정의 첫 업로드 (할당과 descriptor 생성 포함)
                         "first_upload_ms": selected[stage_class.key].upload_ms,
                         "footprint": asdict(selected[stage_class.key].footprint)},
        } for stage_class in classes],
    }
    return CompiledModel(session, report)
