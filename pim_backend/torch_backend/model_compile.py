"""Model compile time: tiling configs of one layer, then every layer compiled with them.

compile_model analyzes the first layer into stage classes (after GEMM packing),
measures each class with its weights already resident, chooses the packing and one
measured config per class together (select_configs), and compiles every layer with
those configs without allocating the device. plan_memory then fits as many layer slots as one unit holds: every layer
keeps its own slot if all fit; otherwise layer i takes slot i mod slots and the
layers of one group are uploaded together, overwriting only the weights. The
returned session allocates, binds and runs at runtime.
"""

from collections import Counter
from itertools import product
from dataclasses import asdict, dataclass, replace
import math
import multiprocessing
from multiprocessing.connection import wait
import statistics
import time

import torch

from triton.backends.pim.residency import kv_parts
from triton.backends.pim.runtime import PIMSession

from ._autotuner import (
    DEFAULT_CACHE_PATH, ConstraintEngine, PIMBenchmarkResult, PIMConfig, PIMConfigCache,
    SearchBudget, TuningRecord, build_config_space,
    pim_hierarchy, target_model, target_spec, tune,
)
from .config import PIMTorchConfig
from .model_policy import (
    Footprint, StageClass, analyze_layer, max_stage_rounds, stage_constraint, stage_key,
)
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


def measure_stage(key, weights, session, config, *, warmup, repeat):
    """E with the weights already resident, U of one upload, and measured footprint.

    session is open; the prepared stage is released afterwards and the units stay allocated.
    """
    module = StageModule(key, weights).eval()
    inputs = stage_inputs(key)

    def configure(gemms, spec):
        return {gemm.node: PIMTorchConfig(spec=spec, config=config) for gemm in gemms}

    # 후보마다 새로 컴파일한다. dynamo가 앞 후보의 컴파일 결과를 돌려주지 않게 한다.
    torch._dynamo.reset()
    with torch.no_grad():
        plan, = session.prepare({"stage": ([module], inputs, configure)})["stage"]
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
            session.run_model("stage", *inputs)
        samples = []
        for _ in range(repeat):
            start = time.perf_counter_ns()
            session.run_model("stage", *inputs)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    session.release()
    return PIMBenchmarkResult(
        total_ms=statistics.median(samples),
        correctness=None,
        timings=(("weight_upload", upload_ms),),
        physical_allocations=((requirement.weight_bytes, requirement.state_bytes,
                               requirement.input_bytes, requirement.output_bytes),),
    )


def _measure_worker(connection, key, weight_shapes, spec, warmup, repeat):
    weights = tuple(None if shape is None else random_tensor(key.input_dtype, shape)
                    for shape in weight_shapes)
    with PIMSession(spec) as session:
        while (config := connection.recv()) is not None:
            connection.send(measure_stage(key, weights, session, config,
                                          warmup=warmup, repeat=repeat))


class StageMeasurer:
    """Measure the candidates of one class in one process that keeps the units allocated.

    The device is opened once per process, not once per candidate. A runtime failure
    ends the process; that candidate is recorded as failed (correctness=False, exit
    code in timings) and the next candidate starts a new process.
    """

    def __init__(self, stage_class, spec, *, warmup, repeat):
        weight_shapes = tuple(None if weight is None else tuple(weight.shape)
                              for weight in stage_class.weights)
        self.args = (stage_class.key, weight_shapes, spec, warmup, repeat)
        self.process = None

    def _start(self):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_measure_worker, args=(child, *self.args))
        self.process.start()
        child.close()

    def measure(self, config):
        if self.process is None:
            self._start()
        self.connection.send(config)
        # 결과가 오거나 측정 프로세스가 끝날 때까지 기다린다.
        ready = wait([self.connection, self.process.sentinel])
        if self.connection in ready:
            return self.connection.recv()
        self.process.join()
        print(f"\n[model compile] candidate {config.key()} failed: measurement process "
              f"exit code {self.process.exitcode}", flush=True)
        result = PIMBenchmarkResult(total_ms=math.inf, correctness=False,
                                    timings=(("exit_code", self.process.exitcode),))
        self.process = None
        return result

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if self.process is not None:
            self.connection.send(None)
            self.process.join()
        return False


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


def select_configs(classes, candidates, capacity_bytes, *, layers, repeats):
    """One measured candidate per class with the smallest time of one layer in one request.

    classes maps a stage class key to its stage count in the layer; one request runs
    the phase repeats times (prefill once, decode once per generated token after the
    first). Two memory plans give two times per layer:
      reloaded  layer groups take turns in the slots, so every run uploads the weights
                again: repeats * (E + U); one layer has to fit one unit with the scratch
      resident  every layer keeps its own slot, so the weights are uploaded once:
                repeats * E + U; all layers have to fit one unit with the scratch
    With one run both times are E + U, and the reloaded plan, which fits more choices,
    is kept. The slot bytes add up over the stages, the scratch is the largest input
    plus the largest output, so the search keeps every partial choice that no other
    choice beats in slot bytes, input, output and time together (multiple-choice knapsack).
    Returns (time_ms, slot_bytes, scratch_bytes, {key: candidate}) or None.
    """
    best = None
    # (unit 하나에 함께 놓이는 layer 수, upload 횟수): reloaded, resident 순서
    for resident_layers, uploads in ((1, repeats), (layers, 1)):
        # (slot bytes, 최대 input, 최대 output, 시간, 고른 후보)
        frontier = [(0, 0, 0, 0.0, {})]
        for key, count in classes.items():
            extended = [(slot + count * (candidate.footprint.weight + candidate.footprint.state),
                         max(largest_input, candidate.footprint.input),
                         max(largest_output, candidate.footprint.output),
                         elapsed + count * (repeats * candidate.execution_ms
                                            + uploads * candidate.upload_ms),
                         {**chosen, key: candidate})
                        for slot, largest_input, largest_output, elapsed, chosen in frontier
                        for candidate in candidates[key]]
            extended.sort(key=lambda option: (option[3], option[0]))
            frontier = []
            for option in extended:
                # 시간이 더 짧은 선택이 bytes도 모두 같거나 작으면 이 선택은 쓸 일이 없다.
                if not any(all(kept[axis] <= option[axis] for axis in range(3)) for kept in frontier):
                    frontier.append(option)
        feasible = [option for option in frontier
                    if resident_layers * option[0] + option[1] + option[2] <= capacity_bytes]
        if feasible and (best is None or feasible[0][3] < best[3]):
            best = feasible[0]
    if best is None:
        return None
    slot, largest_input, largest_output, elapsed, chosen = best
    return elapsed, slot, largest_input + largest_output, chosen


def fitted_configs(key, spec, cache):
    """Cached best configs of similar shapes in the same class context, fitted to this shape.

    Each keeps its blocks, groups and partitions; every tile grows or shrinks so the
    Execution Tiles cover this problem in one M/N/K round.
    """
    problem = key.problem
    configs = []
    for config in cache.lookup_similar(problem, spec, limit=4):
        hierarchy = pim_hierarchy(problem, config)
        partitions = (config.p_m, config.p_n, config.p_k)
        tiles = [math.ceil(groups / partition)
                 for groups, partition in zip(hierarchy.total_groups, partitions)]
        configs.append(replace(config, tile_m=tiles[0], tile_n=tiles[1], tile_k=tiles[2]))
    return tuple(configs)


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


def member_classes(stage_class):
    """Classes the GEMMs of a packed class form when they are not packed, and which class
    each original GEMM then belongs to."""
    axis, shapes, instances = stage_class.origin
    members = {}
    assignment = {}
    for sources in stage_class.stage_sources:
        if axis == "n":
            # member마다 원래 GEMM 하나이고, 이어 붙인 weight의 열 구간 하나를 쓴다.
            parts = []
            start = 0
            for source, mnk in zip(sources, shapes, strict=True):
                weights = tuple(None if weight is None else weight[:, start:start + mnk[1]]
                                for weight in stage_class.weights)
                parts.append(((source,), mnk, weights))
                start += mnk[1]
        else:
            # 원래 GEMM들이 instance로 묶인 stage 하나가 된다.
            parts = [(sources, shapes[0], stage_class.weights)]
        for part_sources, mnk, weights in parts:
            key = replace(stage_class.key, mnk=mnk, instances=instances)
            if key not in members:
                members[key] = StageClass(key, 0, weights, None, [])
            members[key].count += 1
            members[key].stage_sources.append(part_sources)
            for source in part_sources:
                assignment[source] = key
    return list(members.values()), assignment


def stage_config_space(key, spec):
    problem = key.problem
    # target이 k를 device로 나눌 수 없으면(AiMX) seed도 k를 나누지 않는다. instance들이 unit을
    # 나눠 한 batch round에 함께 돌도록 instance 하나가 채울 unit 수를 정한다.
    split_k = target_model(spec).mapping_reason(problem, (1, 1, 2)) is None
    # instance가 unit보다 많으면(AiMX 2 FPGA의 head stage) instance 하나는 unit 하나를 쓴다.
    overrides = {"partition_axes": (True, True, split_k),
                 "instance_units": max(1, spec.common_profile.num_units // key.instances)}
    if key.has_weights:
        overrides["initial_round_patterns"] = ((1, 1, 1),)
    return build_config_space(problem, spec, **overrides)


def tune_stage_class(stage_class, spec, *, cache_path, search_budget, warmup, repeat, seed):
    """Measured candidates of one class that satisfy the model-stage constraints."""
    key = stage_class.key
    problem = key.problem
    constraint = stage_constraint(key, max_stage_rounds(problem, spec))
    cache = PIMConfigCache(cache_path, template_version=cache_version(key))
    # packing member의 config를 넓힌 후보, 같은 context의 비슷한 shape config를 이 shape에 맞춘 후보
    initial = tuple({config.key(): config for config in (
        widened_configs(stage_class, spec, cache_path) + fitted_configs(key, spec, cache))}.values())

    print(f"[model compile] tune {key.label()} initial={[config.key() for config in initial]}", flush=True)
    # cache에 있는 후보만 쓰면 측정 프로세스를 띄우지 않는다(measure가 처음 불릴 때 띄운다).
    with StageMeasurer(stage_class, spec, warmup=warmup, repeat=repeat) as measurer:
        tune(problem, spec, lambda problem, target, config: measurer.measure(config),
             space=stage_config_space(key, spec), cache=cache, budget=search_budget, seed=seed,
             verbose=True, constraint=constraint, initial=initial)
        # cache에는 다른 제약으로 잰 기록도 있을 수 있으므로 지금 제약으로 다시 거른다.
        engine = ConstraintEngine(problem, spec, constraint=constraint)
        # cache hit이면 tune은 탐색하지 않는다. 아직 재지 않은 initial 후보는 여기서 재서 기록에 더한다.
        measured = cache.measured_results(problem, spec)
        added = [TuningRecord(config, None, measurer.measure(config))
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
    return candidates


@dataclass
class CompiledModel:
    session: PIMSession
    report: dict


def choose_phase(phase, layers, example_inputs, spec, capacity_bytes, *, repeats, cache_path,
                 search_budget, warmup, repeat, seed):
    """Stage classes, packing and one config per class for the first layer of one phase.

    capacity_bytes is what one unit has for the phase area (after the key/value cache);
    one request runs the phase repeats times.
    """
    start = time.perf_counter_ns()
    classes, assignment = analyze_layer(layers[0], example_inputs, spec)
    analysis_ms = (time.perf_counter_ns() - start) / 1e6
    print(f"[model compile] {phase}: stage_classes={len(classes)}", flush=True)

    start = time.perf_counter_ns()
    candidates = {}

    def tune_once(stage_class):
        if stage_class.key not in candidates:
            candidates[stage_class.key] = tune_stage_class(
                stage_class, spec, cache_path=cache_path, search_budget=search_budget,
                warmup=warmup, repeat=repeat, seed=seed)

    members = {}
    for stage_class in classes:
        # member를 먼저 재야 packed class가 member config를 넓혀 탐색을 시작한다(widened_configs).
        if stage_class.origin is not None:
            members[stage_class.key] = member_classes(stage_class)
            for member in members[stage_class.key][0]:
                tune_once(member)
        tune_once(stage_class)

    packed_classes = [stage_class for stage_class in classes if stage_class.origin is not None]
    best = None
    packing = []
    for decision in product((True, False), repeat=len(packed_classes)):
        separated = {stage_class.key for stage_class, packed in zip(packed_classes, decision)
                     if not packed}
        # member key가 packing하지 않은 다른 class와 같으면(q/k/v member와 o) 한 class로 합친다.
        parts = {}
        for stage_class in classes:
            for part in (members[stage_class.key][0] if stage_class.key in separated
                         else [stage_class]):
                parts.setdefault(part.key, []).append(part)
        counts = {key: sum(part.count for part in group) for key, group in parts.items()}
        selection = select_configs(counts, candidates, capacity_bytes,
                                   layers=len(layers), repeats=repeats)
        packing.append({"separated": [key.label() for key in separated],
                        "layer_ms": None if selection is None else selection[0],
                        "slot_bytes": None if selection is None else selection[1],
                        "scratch_bytes": None if selection is None else selection[2]})
        print(f"[model compile] {phase}: separate={[key.label() for key in separated]} -> "
              + ("no layer fits one unit" if selection is None else
                 f"{repeats} runs {selection[0]:.1f} ms/layer, slot {selection[1]} B, "
                 f"scratch {selection[2]} B"),
              flush=True)
        if selection is not None and (best is None or selection[0] < best[0][0]):
            best = (selection, separated, parts)
    if best is None:
        raise ValueError(f"{phase}: no choice of measured configs fits one layer in {capacity_bytes} bytes")
    (_, _, _, selected), separated_keys, parts = best
    separate = set()
    for key in separated_keys:
        member_assignment = members[key][1]
        separate.update(member_assignment)
        assignment.update(member_assignment)
    classes = [StageClass(key, sum(part.count for part in group), group[0].weights, group[0].origin,
                          [sources for part in group for sources in part.stage_sources])
               for key, group in parts.items()]
    tuning_ms = (time.perf_counter_ns() - start) / 1e6
    return {"selected": selected, "assignment": assignment, "separate": separate,
            "classes": classes, "candidates": candidates, "packing": packing, "repeats": repeats,
            "analysis_ms": analysis_ms, "tuning_ms": tuning_ms}


def compile_model(phases, *, target, units, kv=None, repeats=None, runtime_options=None,
                  cache_path=DEFAULT_CACHE_PATH, max_benchmarks=20, max_generations=10,
                  warmup=1, repeat=3, seed=0):
    """Compile time of a model whose layers share one structure in every phase; the session runs it.

    phases maps each phase name, in execution order, to (layers, example_inputs); kv is the
    model's KVRequirement, or None when the key/value cache stays on the host. repeats maps
    every phase to the times one request runs it (decode: generated tokens - 1); None runs
    every phase once. Every phase overlays one area after the key/value cache, so its
    configs are chosen for what is left:
    1. analyze the first layer of the phase into stage classes
    2. tune every class with its weights resident; a packed class also tunes its members
    3. for every pack/separate choice of the packed classes, select_configs picks one
       candidate per class; the choice with the smallest time of one layer in one
       request wins (E and U weighed by the repeats)
    4. compile every layer of every phase with those configs; plan_memory places them
    """
    spec = target_spec(target, num_units=units, **(runtime_options or {}))
    kv_bytes = 0 if kv is None else 2 * kv.layers * kv_parts(kv, units)[2]
    capacity_bytes = (spec.common_profile.banks_per_unit * spec.common_profile.bank_capacity_limit_bytes
                      - kv_bytes)
    print(f"[model compile] target={target} units={units} phases={list(phases)} "
          f"kv_bytes={kv_bytes}", flush=True)
    search_budget = SearchBudget(max_generations=max_generations, max_benchmarks=max_benchmarks)
    choices = {phase: choose_phase(phase, tuple(layers), example_inputs, spec, capacity_bytes,
                                   repeats=1 if repeats is None else repeats[phase],
                                   cache_path=cache_path, search_budget=search_budget,
                                   warmup=warmup, repeat=repeat, seed=seed)
               for phase, (layers, example_inputs) in phases.items()}

    def configure_with(choice):
        # 모든 layer의 GEMM 이름이 분석한 layer와 같다. 없는 이름은 KeyError로 드러난다.
        def configure(gemms, spec):
            return {gemm.node: PIMTorchConfig(spec=spec,
                                              config=choice["selected"][choice["assignment"][gemm.node]].config,
                                              pack=gemm.node not in choice["separate"])
                    for gemm in gemms}
        return configure

    start = time.perf_counter_ns()
    session = PIMSession(spec)
    torch._dynamo.reset()
    session.compile({phase: (layers, example_inputs, configure_with(choices[phase]))
                     for phase, (layers, example_inputs) in phases.items()}, kv=kv)
    for phase, plans in session.plans.items():
        classes = choices[phase]["classes"]
        # packing 판단을 반영한 layer가 측정한 class와 같은 stage들로 나뉘었는지 확인한다.
        compiled = Counter(stage_key(stage) for stage in plans[0].stages)
        expected = Counter({stage_class.key: stage_class.count for stage_class in classes})
        if compiled != expected:
            raise ValueError(f"{phase}: layer 0 compiled into stages {dict(compiled)}; "
                             f"the tuned classes are {expected}")
        analyzed = [(tuple(stage.shape), stage.instance_count, tuple(stage.weight_mask))
                    for stage in plans[0].stages]
        for plan in plans[1:]:
            stages = [(tuple(stage.shape), stage.instance_count, tuple(stage.weight_mask))
                      for stage in plan.stages]
            if stages != analyzed:
                raise ValueError(f"{phase}: layer {plan.index} has stages {stages}; "
                                 f"the analyzed layer has {analyzed}")
    compile_ms = (time.perf_counter_ns() - start) / 1e6
    memory = session.memory
    for phase, layout in memory.phases.items():
        print(f"[model compile] {phase}: slots={layout.slots} groups={layout.groups} bytes={layout.bytes} "
              f"analysis_ms={choices[phase]['analysis_ms']:.1f} tuning_ms={choices[phase]['tuning_ms']:.1f}",
              flush=True)
    print(f"[model compile] kv_bytes={memory.kv_bytes} bytes_per_unit={memory.bytes_per_unit} "
          f"capacity={memory.capacity_bytes} compile_ms={compile_ms:.1f}", flush=True)

    def phase_report(phase, layers):
        choice = choices[phase]
        selected = choice["selected"]
        return {
            "layers": len(layers), "repeats": choice["repeats"],
            # 한 번 실행할 때 weight 상주 실행 시간만 더한 값. weight 업로드와 CPU 시간은 들어가지 않는다.
            "estimated_execution_ms": len(layers) * sum(
                stage_class.count * selected[stage_class.key].execution_ms
                for stage_class in choice["classes"]),
            "analysis_ms": choice["analysis_ms"], "tuning_ms": choice["tuning_ms"],
            "packing": choice["packing"],
            "stage_classes": [{
                "label": stage_class.key.label(),
                "mnk": list(stage_class.key.mnk), "op": stage_class.key.op,
                "weight_mask": list(stage_class.key.weight_mask),
                "instances": stage_class.key.instances,
                "stages_per_layer": stage_class.count,
                "measured": len(choice["candidates"][stage_class.key]),
                "selected": {"config": asdict(selected[stage_class.key].config),
                             "execution_ms": selected[stage_class.key].execution_ms,
                             # 격리 측정의 첫 업로드. group 진입 때 pack과 전송이 함께 일어나므로 둘 다 들어간다.
                             "first_upload_ms": selected[stage_class.key].upload_ms,
                             "footprint": asdict(selected[stage_class.key].footprint)},
            } for stage_class in choice["classes"]],
        }

    report = {
        "target": target, "units": units, "spec": asdict(spec),
        "kv": None if kv is None else asdict(kv),
        "memory": {"kv_split": memory.kv_split, "kv_tokens_per_unit": memory.kv_tokens_per_unit,
                   "kv_bytes": memory.kv_bytes, "bytes_per_unit": memory.bytes_per_unit,
                   "capacity_bytes": memory.capacity_bytes,
                   "phases": {phase: {"slots": layout.slots, "groups": layout.groups,
                                      "slot_of_layer": layout.slot_of_layer,
                                      "group_of_layer": layout.group_of_layer,
                                      "slot_bytes": list(layout.slot_bytes),
                                      "scratch_bytes": layout.scratch_bytes, "bytes": layout.bytes}
                              for phase, layout in memory.phases.items()}},
        "compile_ms": compile_ms,
        "phases": {phase: phase_report(phase, layers) for phase, (layers, _) in phases.items()},
    }
    return CompiledModel(session, report)
