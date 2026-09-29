"""Vendor-neutral lifecycle for compiled layers and request-scoped resources."""

from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock, get_ident
import time
from typing import Protocol


current_invocation = ContextVar("pim_invocation", default=None)
current_layer = ContextVar("pim_layer", default=None)
current_session = ContextVar("pim_session", default=None)
resident_lock = RLock()


class RuntimeBackend(Protocol):
    """Synchronous adapter contract.

    compile_invocation/finalize_plan must not allocate device resources or
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
    """Own resources until close; tokens order weight loads and layer executions.

    Compilation is performed separately by compile_layer. The runtime FX graph
    orders load_weights and run_layer through explicit completion tokens.
    A completion token can also execute the same resident plan again.
    A session is synchronous and belongs to the thread that opened it.
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
        self.phase = "idle"
        self.open_ms = (time.perf_counter_ns() - start) / 1e6
        return self

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

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False
