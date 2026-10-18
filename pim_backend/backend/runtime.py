"""Vendor-neutral lifecycle of compiled model layers and their device resources."""

from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import replace
from threading import RLock, get_ident
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
    unit count and capacity; requirements(plans) reports per-stage bytes for the
    common memory plan; prepare_model(plans, memory) places the layers in their
    slots and links native artifacts. None of these touch the device, so a model
    can be compiled before open().

    open() allocates the units and exposes num_units. bind_model(plans) creates
    the device handles of every slot, weight_table(plans) reports weight placement,
    memory_table(plans) the slot buffers and their addresses, and the model runtime
    implements enter_layer/run/exit_layer. stats contains cumulative transfer bytes
    and launch counts. close releases resources.
    """

    def open(self): ...
    num_units: int
    def begin_model(self): ...
    def resources(self): ...
    def requirements(self, plans): ...
    def prepare_model(self, plans, memory): ...
    def bind_model(self, plans): ...
    def weight_table(self, plans): ...
    def memory_table(self, plans): ...
    def stats(self) -> dict: ...
    def close(self): ...


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
        """Memory management table rows: slot, its layers, stage, unit, column, PA and bytes."""
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
