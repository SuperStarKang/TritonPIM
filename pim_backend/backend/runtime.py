"""Vendor-neutral lifecycle for compiled layers and request-scoped resources."""

from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, replace
from threading import RLock, get_ident
import time
from typing import Protocol


current_invocation = ContextVar("pim_invocation", default=None)
current_layer = ContextVar("pim_layer", default=None)
current_session = ContextVar("pim_session", default=None)
resident_lock = RLock()


class RuntimeBackend(Protocol):
    """Synchronous adapter contract for model and legacy layer execution.

    open exposes the actual num_units. begin_model creates the target execution
    runtime. prepare_model places the planned layers on those resources and
    finalizes their shared weight/scratch layout and native artifacts.

    Legacy compile_invocation/finalize_plan must not allocate device resources or
    perform transfers. load_layer replaces the previous placement on the same
    resources and returns timing fields in milliseconds. invocation returns a
    context manager; stats contains cumulative transfer bytes and launch counts
    across layer replacements. begin_run validates the loaded plan before each
    execution. Activations are sent for every GEMM; weights remain resident.
    close releases resources.
    """

    def validate_config(self, config): ...
    def compile_invocation(self, kernel, physical_inputs): ...
    def finalize_plan(self, plan): ...
    def open(self): ...
    num_units: int
    def begin_model(self): ...
    def prepare_model(self, plans): ...
    def load_layer(self, plan): ...
    def begin_run(self, plan): ...
    def invocation(self, kernel, physical_inputs): ...
    def stats(self) -> dict: ...
    def close(self): ...


def create_runtime(spec) -> RuntimeBackend:
    if spec.common_profile.target == "upmem":
        from .upmem_adapter import UPMEMAdapter

        return UPMEMAdapter(spec)
    raise NotImplementedError(
        f"{spec.common_profile.target}: request-scoped weight residency has no adapter"
    )


@dataclass(frozen=True, eq=False)
class ExecutionToken:
    session: object
    generation: int
    completed: bool


class PIMSession:
    """Own model resources, layer weights and KV storage until close.

    prepare creates LayerPlans from Torch blocks after acquiring resources.
    run_model executes those plans without completion tokens.
    start/load_weights/run_layer retain the legacy compiled-layer API.
    """

    def __init__(self, spec):
        self.spec = spec
        self.backend = None
        self.phase = "new"
        self.plan = None
        self.token = None
        self.generation = 0
        self.records = []
        self.open_ms = 0.0
        self.close_ms = 0.0
        self.layers = ()
        self.weight_table = {}
        self.kv_table = {}
        self.loaded_layer = None

    def __enter__(self):
        if self.phase != "new" or current_session.get() is not None:
            raise RuntimeError("a PIMSession must be opened once and cannot be nested")
        start = time.perf_counter_ns()
        self.backend = create_runtime(self.spec)
        with ExitStack() as cleanup:
            self.backend.open()
            cleanup.callback(self.backend.close)
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

    def prepare(self, layers, example_inputs, *, configure):
        """Export and prepare Torch blocks on the session's allocated resources.

        configure(gemms, spec) maps original FX GEMM names to PIMTorchConfig.
        CPU preparation outputs supply the next block's positional inputs.
        """
        import torch

        from torch_backend.fx_analysis import analyze_fx_gemms
        from torch_backend.model import build_layer_plan

        from .resident import WEIGHT

        self._check_owner()
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
                    index=index,
                )
                plan.plan_model(self.spec, self.backend.model)
                outputs = plan.compile_cpu()
                plans.append(plan)
                example_inputs = outputs if isinstance(outputs, tuple) else (outputs,)

        self.layers = tuple(plans)
        self.backend.prepare_model(self.layers)
        for plan in self.layers:
            for stage in plan.stages:
                for binding in stage.bindings:
                    if binding.role == WEIGHT:
                        self.weight_table[(plan.index, stage.name, binding.operand)] = {
                            "tensor": stage.weights[binding.operand],
                            "address": stage.addresses[binding.symbol],
                            "bytes_per_unit": binding.size,
                            "mapping": stage.physical_mapping,
                        }
        return self.layers

    @property
    def loaded_program(self):
        return self.backend.model.loaded_program

    def run_model(self, *inputs):
        """Chain prepared blocks; tuple outputs become the next block's inputs."""
        self._check_owner()
        if not self.layers:
            raise RuntimeError("prepare the model before running it")
        for plan in self.layers:
            self.backend.model.load_weights(plan)
            self.loaded_layer = plan.index
            outputs = plan.run_model(*inputs)
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

    def start(self):
        self._check_owner()
        if self.phase != "idle" or self.token is not None:
            raise RuntimeError("the request has already started")
        self.token = ExecutionToken(self, 0, True)
        return self.token

    def load_weights(self, plan, previous):
        self._check_owner()
        if (self.phase != "idle" or self.token is None
                or previous is not self.token or not previous.completed):
            raise RuntimeError("weight replacement requires the previous completion token")
        if plan.spec != self.spec:
            raise ValueError("layer and session target specifications differ")
        plan.check_state()
        self.phase = "loading"
        before = self.backend.stats()
        start = time.perf_counter_ns()
        timings = self.backend.load_layer(plan)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        after = self.backend.stats()
        self.records.append({
            "layer": plan.index,
            "weight_load_ms": elapsed,
            **timings,
            "load_transfers": {name: after[name] - before[name] for name in before},
        })
        self.plan = plan
        self.generation += 1
        self.token = ExecutionToken(self, self.generation, False)
        self.phase = "ready"
        return self.token

    def run_layer(self, plan, inputs, ready):
        self._check_owner()
        if (self.phase not in ("ready", "idle") or self.plan is None
                or ready is not self.token or plan is not self.plan):
            raise RuntimeError("layer execution requires its latest token and resident plan")
        if current_layer.get() is not None:
            raise RuntimeError("nested layer execution is unsupported")
        self.phase = "running"
        before = self.backend.stats()
        start = time.perf_counter_ns()
        self.backend.begin_run(plan)
        with ExitStack() as scope:
            scope.callback(current_layer.reset, current_layer.set((self, plan, ready)))
            outputs = plan.run(*inputs)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        after = self.backend.stats()
        transfers = {name: after[name] - before[name] for name in before}
        if transfers["weight_bytes"] != 0:
            raise RuntimeError("a layer execution transferred weights after its load node")
        self.records[-1].update(run_ms=elapsed, run_transfers=transfers)
        self.token = ExecutionToken(self, self.generation, True)
        self.phase = "idle"
        return outputs, self.token

    def invocation(self, kernel, physical_inputs, ready):
        self._check_owner()
        if (self.phase != "running" or ready is not self.token
                or kernel.runtime is not self.plan.runtime):
            raise RuntimeError("kernel does not belong to the executing layer")
        return self.backend.invocation(kernel, physical_inputs)

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
        self.plan = None
        self.layers = ()
        self.weight_table.clear()
        self.kv_table.clear()

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False
