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
    unit count and capacity; requirements(plans) reports per-operand bytes for the
    common memory table; prepare_model(plans, memory) places the layers at the
    table's offsets and links native artifacts. None of these touch the device, so a model
    can be compiled before open().

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


class GroupLoader:
    """Upload the weights of one layer group when the group is entered, for one request.

    pack(group) lays a group's weights out in host memory and returns them; it never
    touches the device, so it runs on a background thread while the previous group
    computes (computation and transfers to one device cannot overlap). push(packed)
    uploads them and release(packed) frees the host memory. Entering group g takes its
    packed weights (waiting for the background pack, or packing now), pushes and releases
    them, then starts packing g + 1. Group 0 has nothing to hide behind, so its pack and
    push are also accumulated apart, to be subtracted from a request time.
    """

    def __init__(self, groups, pack, push, release):
        self.groups = groups
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

    def _start_prefetch(self, group):
        result = {}

        def work():
            start = time.perf_counter_ns()
            result["packed"] = self.pack(group)
            result["ms"] = (time.perf_counter_ns() - start) / 1e6

        thread = Thread(target=work, name=f"pim-pack-group{group}")
        thread.start()
        self.prefetch = (group, thread, result)

    def _join_prefetch(self):
        group, thread, result = self.prefetch
        thread.join()
        self.prefetch = None
        self.totals["background_pack_ms"] += result["ms"]
        return group, result["packed"]

    def enter(self, group):
        if group == self.loaded:
            return
        start = time.perf_counter_ns()
        packed = None
        if self.prefetch is not None:
            prefetched, packed = self._join_prefetch()
            if prefetched != group:
                self.release(packed)
                packed = None
        if packed is None:
            packed = self.pack(group)
        pack_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        self.push(packed)
        upload_ms = (time.perf_counter_ns() - start) / 1e6
        # 보낸 weight는 이번 요청에서 다시 쓰지 않는다.
        self.release(packed)
        self.totals["pack_wall_ms"] += pack_ms
        self.totals["upload_wall_ms"] += upload_ms
        self.totals["weight_uploads"] += 1
        if group == 0:
            self.totals["group0_pack_ms"] += pack_ms
            self.totals["group0_upload_ms"] += upload_ms
        self.loaded = group
        # 이 group이 실행되는 동안 다음 group을 host에서 미리 pack한다.
        if group + 1 < self.groups:
            self._start_prefetch(group + 1)

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

    compile() prepares Torch blocks before open; entering the session then
    allocates and binds them. prepare() does both inside an open session.
    run_model executes the prepared layers in order.
    """

    def __init__(self, spec):
        self.spec = spec
        self.backend = None
        self.phase = "new"
        self.open_ms = 0.0
        self.close_ms = 0.0
        self.layers = ()
        self.weight_table = {}
        self.kv_table = {}
        self.loaded_layer = None
        self.memory = None

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
                self.backend.bind_model(self.layers)
                self.weight_table = self.backend.weight_table(self.layers)
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

    def compile(self, layers, example_inputs, *, configure):
        """Compile time: prepare Torch blocks without allocating device resources.

        The session keeps the compiled layers; entering it allocates the units,
        checks their count against the spec and binds the compiled stages.
        """
        if self.phase != "new" or self.backend is not None:
            raise RuntimeError("compile a PIMSession once, before opening it")
        self.backend = create_runtime(self.spec)
        self._build(layers, example_inputs, configure)
        self.phase = "compiled"
        return self.layers

    def prepare(self, layers, example_inputs, *, configure):
        """Export and prepare Torch blocks on the session's allocated resources.

        configure(gemms, spec) maps original FX GEMM names to PIMTorchConfig.
        CPU preparation outputs supply the next block's positional inputs.
        """
        self._check_owner()
        self._build(layers, example_inputs, configure)
        self.backend.bind_model(self.layers)
        self.weight_table = self.backend.weight_table(self.layers)
        return self.layers

    def _build(self, layers, example_inputs, configure):
        """Export, configure, compile and place the layers; no device access."""
        import torch

        from torch_backend.fx_analysis import analyze_fx_gemms
        from torch_backend.model import build_layer_plan

        layers = tuple(layers)
        if not layers:
            raise ValueError("a model requires at least one layer")
        self.backend.begin_model()
        plans = []
        with torch.no_grad():
            for index, layer in enumerate(layers):
                exported = torch.export.export(layer, example_inputs)
                plan = build_layer_plan(
                    exported, example_inputs,
                    configs=configure(analyze_fx_gemms(exported, allow_bmm=True), self.spec),
                    index=index, spec=self.spec,
                )
                plan.plan_model(self.spec, self.backend.model)
                outputs = plan.compile_cpu()
                plans.append(plan)
                example_inputs = outputs if isinstance(outputs, tuple) else (outputs,)

        self.layers = tuple(plans)
        self.memory = plan_memory(self.backend.resources(), self.backend.requirements(self.layers))
        self.backend.prepare_model(self.layers, self.memory)

    def run_model(self, *inputs):
        """Chain prepared blocks; tuple outputs become the next block's inputs."""
        self._check_owner()
        if not self.layers:
            raise RuntimeError("prepare the model before running it")
        for plan in self.layers:
            self.backend.model.enter_layer(plan)
            self.loaded_layer = plan.index
            outputs = plan.run_model(*inputs)
            self.backend.model.exit_layer(plan)
            inputs = outputs if isinstance(outputs, tuple) else (outputs,)
        return outputs

    def release(self):
        """Release the prepared layers and keep the units allocated; prepare() can follow."""
        self._check_owner()
        self.backend.end_model()
        self.layers = ()
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
        return self.backend.memory_table(self.layers)

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
        self.layers = ()
        self.weight_table.clear()
        self.kv_table.clear()

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False
