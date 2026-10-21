"""Vendor-neutral lifecycle of compiled model layers and their device resources."""

from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import replace
from threading import RLock, Thread, get_ident
import time
from typing import Protocol

from .residency import plan_memory


current_invocation = ContextVar("pim_invocation", default=None)
current_session = ContextVar("pim_session", default=None)
# UPMEM DPU 집합은 프로세스 전역이다. session과 직접 launch가 동시에 쓰지 않게 한다.
resident_lock = RLock()


class RuntimeBackend(Protocol):
    """Synchronous adapter contract of the model path.

    begin_model creates the target model runtime. resources() reports the spec's
    unit count and capacity; requirements(plans) reports per-operand bytes of one
    phase's layers for the common memory table; prepare_model(plans, memory) places
    the layers of every phase (plan.phase) at the table's offsets and links native
    artifacts. None of these touch the device, so a model can be compiled before open().

    open() allocates the units and exposes num_units. bind_model(plans) creates
    the device handles of every slot, weight_table(plans) reports weight placement,
    memory_table(plans) the table rows with their device addresses, and the model runtime
    implements enter_layer/run/exit_layer. end_model releases the model runtime
    while the units stay allocated, so begin_model can follow. stats contains
    cumulative transfer bytes and launch counts. close releases resources.
    """

    def open(self): ...
    num_units: int
    def begin_model(self): ...
    def end_model(self): ...
    def resources(self): ...
    def requirements(self, plans): ...
    def prepare_model(self, plans, memory): ...
    def bind_model(self, plans): ...
    def weight_table(self, plans): ...
    def memory_table(self, plans): ...
    def stats(self) -> dict: ...
    def close(self): ...


# GroupLoader가 누적하는 통계 이름. adapter가 model 통계를 합칠 때도 같은 이름을 쓴다.
GROUP_LOAD_STATS = ("weight_uploads", "pack_wall_ms", "background_pack_ms", "upload_wall_ms",
                    "group0_pack_ms", "group0_upload_ms")


def group_order(memory):
    """(phase, group) keys in the order one request enters them."""
    return [(phase, group) for phase, layout in memory.phases.items()
            for group in range(layout.groups)]


class GroupLoader:
    """Upload the weights of one layer group when the group is entered, for one request.

    order lists the (phase, group) keys in the order a request enters them. pack(key)
    lays a group's weights out in host memory and returns them; it never touches the
    device, so it runs on a background thread while the previous group computes
    (computation and transfers to one device cannot overlap). push(packed) uploads them
    and release(packed) frees the host memory. Entering a key takes its packed weights
    (waiting for the background pack, or packing now), pushes and releases them, then
    starts packing the next key of order; the last group of a phase is followed by the
    first group of the next phase, whose push then overwrites the previous phase. The
    first key has nothing to hide behind, so its pack and push are also accumulated
    apart (group0_*), to be subtracted from a request time.
    """

    def __init__(self, order, pack, push, release):
        self.order = order
        self.pack = pack
        self.push = push
        self.release = release
        self.loaded = None
        # 다음 group을 pack하는 background 작업: (group, thread, 결과)
        self.prefetch = None
        # weight_uploads는 group 업로드 횟수다. pack_wall_ms는 요청 경로에서 pack에 쓴 시간
        # (직접 pack하거나 background pack을 기다린 시간), background_pack_ms는 실행과 겹쳐 돈
        # pack 시간, upload_wall_ms는 push 시간이다.
        self.totals = dict.fromkeys(GROUP_LOAD_STATS, 0)

    def _start_prefetch(self, key):
        result = {}

        def work():
            start = time.perf_counter_ns()
            result["packed"] = self.pack(key)
            result["ms"] = (time.perf_counter_ns() - start) / 1e6

        phase, group = key
        thread = Thread(target=work, name=f"pim-pack-{phase}-group{group}")
        thread.start()
        self.prefetch = (key, thread, result)

    def _join_prefetch(self):
        key, thread, result = self.prefetch
        thread.join()
        self.prefetch = None
        self.totals["background_pack_ms"] += result["ms"]
        return key, result["packed"]

    def enter(self, key):
        if key == self.loaded:
            return
        start = time.perf_counter_ns()
        packed = None
        if self.prefetch is not None:
            prefetched, packed = self._join_prefetch()
            if prefetched != key:
                self.release(packed)
                packed = None
        if packed is None:
            packed = self.pack(key)
        pack_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        self.push(packed)
        upload_ms = (time.perf_counter_ns() - start) / 1e6
        # 보낸 weight는 이번 요청에서 다시 쓰지 않는다.
        self.release(packed)
        self.totals["pack_wall_ms"] += pack_ms
        self.totals["upload_wall_ms"] += upload_ms
        self.totals["weight_uploads"] += 1
        if key == self.order[0]:
            self.totals["group0_pack_ms"] += pack_ms
            self.totals["group0_upload_ms"] += upload_ms
        self.loaded = key
        # 이 group이 실행되는 동안 실행 순서의 다음 group을 host에서 미리 pack한다.
        position = self.order.index(key)
        if position + 1 < len(self.order):
            self._start_prefetch(self.order[position + 1])

    def close(self):
        """Finish a background pack and free what it packed; call before freeing its stages."""
        if self.prefetch is not None:
            self.release(self._join_prefetch()[1])


def create_runtime(spec) -> RuntimeBackend:
    if spec.common_profile.target == "upmem":
        from .upmem_adapter import UPMEMAdapter

        return UPMEMAdapter(spec)
    if spec.common_profile.target == "aimx":
        from .aimx_adapter import AiMXAdapter

        return AiMXAdapter(spec)
    raise NotImplementedError(
        f"{spec.common_profile.target}: request-scoped weight residency has no adapter"
    )


class PIMSession:
    """Own model resources, layer weights and KV storage until close.

    compile() prepares the Torch blocks of every phase before open; entering the
    session then allocates and binds them. prepare() does both inside an open
    session. run_model(phase, ...) executes the layers of one phase in order.
    """

    def __init__(self, spec):
        self.spec = spec
        self.backend = None
        self.phase = "new"
        self.open_ms = 0.0
        self.close_ms = 0.0
        # phase -> 그 phase의 LayerPlan들 (실행 순서)
        self.plans = {}
        self.weight_table = {}
        self.kv_table = {}
        self.loaded_layer = None
        self.memory = None

    def _all_plans(self):
        return tuple(plan for plans in self.plans.values() for plan in plans)

    def __enter__(self):
        if self.phase not in ("new", "compiled") or current_session.get() is not None:
            raise RuntimeError("a PIMSession must be opened once and cannot be nested")
        start = time.perf_counter_ns()
        if self.backend is None:
            self.backend = create_runtime(self.spec)
        with ExitStack() as cleanup:
            self.backend.open()
            cleanup.callback(self.backend.close)
            if self.phase == "compiled":
                # compile time의 artifact와 배치는 그 unit 수를 전제로 만들었다.
                compiled_units = self.spec.common_profile.num_units
                if self.backend.num_units != compiled_units:
                    raise ValueError(
                        f"the model was compiled for {compiled_units} units; "
                        f"{self.backend.num_units} were allocated")
                self.backend.bind_model(self._all_plans())
                self.weight_table = self.backend.weight_table(self._all_plans())
            token = current_session.set(self)
            cleanup.callback(current_session.reset, token)
            self.cleanup = cleanup.pop_all()
        self.thread = get_ident()
        self.num_units = self.backend.num_units
        self.spec = replace(self.spec, common_profile=replace(
            self.spec.common_profile, num_units=self.num_units))
        self.backend.spec = self.spec
        self.phase = "idle"
        self.open_ms = (time.perf_counter_ns() - start) / 1e6
        return self

    def compile(self, phases, *, kv=None):
        """Compile time: prepare the Torch blocks of every phase without allocating device resources.

        phases maps each phase name, in execution order, to (layers, example_inputs,
        configure); kv is the model's KVRequirement or None. The session keeps the
        compiled plans; entering it allocates the units, checks their count against
        the spec and binds the compiled stages.
        """
        if self.phase != "new" or self.backend is not None:
            raise RuntimeError("compile a PIMSession once, before opening it")
        self.backend = create_runtime(self.spec)
        self._build(phases, kv)
        self.phase = "compiled"
        return self.plans

    def prepare(self, phases, *, kv=None):
        """Export and prepare the Torch blocks of every phase on the session's allocated resources.

        configure(gemms, spec) maps original FX GEMM names to PIMTorchConfig.
        CPU preparation outputs supply the next block's positional inputs.
        """
        self._check_owner()
        self._build(phases, kv)
        self.backend.bind_model(self._all_plans())
        self.weight_table = self.backend.weight_table(self._all_plans())
        return self.plans

    def _build(self, phases, kv):
        """Export, configure, compile and place the layers of every phase; no device access."""
        import torch

        from torch_backend.fx_analysis import analyze_fx_gemms
        from torch_backend.model import build_layer_plan

        self.backend.begin_model()
        self.plans = {}
        with torch.no_grad():
            for phase, (layers, example_inputs, configure) in phases.items():
                layers = tuple(layers)
                if not layers:
                    raise ValueError(f"{phase}: a model phase requires at least one layer")
                plans = []
                for index, layer in enumerate(layers):
                    exported = torch.export.export(layer, example_inputs)
                    plan = build_layer_plan(
                        exported, example_inputs,
                        configs=configure(analyze_fx_gemms(exported, allow_bmm=True), self.spec),
                        index=index, spec=self.spec,
                    )
                    plan.phase = phase
                    plan.plan_model(self.spec, self.backend.model)
                    outputs = plan.compile_cpu()
                    plans.append(plan)
                    example_inputs = outputs if isinstance(outputs, tuple) else (outputs,)
                self.plans[phase] = tuple(plans)

        self.memory = plan_memory(
            self.backend.resources(),
            {phase: self.backend.requirements(plans) for phase, plans in self.plans.items()}, kv)
        self.backend.prepare_model(self._all_plans(), self.memory)

    def run_model(self, phase, *inputs):
        """Chain the prepared blocks of one phase; tuple outputs become the next block's inputs."""
        self._check_owner()
        if phase not in self.plans:
            raise RuntimeError(f"phase {phase!r} is not prepared; prepared: {list(self.plans)}")
        for plan in self.plans[phase]:
            self.backend.model.enter_layer(plan)
            self.loaded_layer = (phase, plan.index)
            outputs = plan.run_model(*inputs)
            self.backend.model.exit_layer(plan)
            inputs = outputs if isinstance(outputs, tuple) else (outputs,)
        return outputs

    def release(self):
        """Release the prepared layers and keep the units allocated; prepare() can follow."""
        self._check_owner()
        self.backend.end_model()
        self.plans = {}
        self.weight_table = {}
        self.memory = None
        self.loaded_layer = None

    def reserve_kv(self, layer_id, shape, dtype):
        """Reserve CPU KV storage once; sequence capacity is the penultimate axis."""
        import torch

        self._check_owner()
        if len(shape) < 2 or min(shape) < 1:
            raise ValueError("KV storage requires positive shape and sequence/feature axes")
        if layer_id in self.kv_table:
            raise ValueError("KV storage is already reserved for this layer")
        self.kv_table[layer_id] = {
            "key": torch.empty(shape, dtype=dtype),
            "value": torch.empty(shape, dtype=dtype),
            "capacity": shape[-2],
            "length": 0,
        }

    def append_kv(self, layer_id, key, value):
        cache = self.kv_table[layer_id]
        end = cache["length"] + key.shape[-2]
        if end > cache["capacity"]:
            raise ValueError("KV append exceeds the reserved sequence capacity")
        cache["key"][..., cache["length"]:end, :].copy_(key)
        cache["value"][..., cache["length"]:end, :].copy_(value)
        cache["length"] = end

    def kv_view(self, layer_id):
        cache = self.kv_table[layer_id]
        return (cache["key"][..., :cache["length"], :],
                cache["value"][..., :cache["length"], :])

    def _check_owner(self):
        if current_session.get() is not self or get_ident() != self.thread:
            raise RuntimeError("PIMSession must be used by its owning thread and context")

    def memory_table(self):
        """Memory table rows with device addresses: UPMEM MRAM base + offset, AiMX slot region + offset per FPGA."""
        self._check_owner()
        return self.backend.memory_table(self._all_plans())

    def stats(self):
        if self.backend is None:
            raise RuntimeError("PIMSession has not been opened")
        if self.phase != "closed":
            self._check_owner()
        return self.backend.stats()

    def close(self):
        if self.phase == "closed":
            return
        self._check_owner()
        start = time.perf_counter_ns()
        self.cleanup.close()
        self.close_ms = (time.perf_counter_ns() - start) / 1e6
        self.phase = "closed"
        self.plans = {}
        self.weight_table.clear()
        self.kv_table.clear()

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False
