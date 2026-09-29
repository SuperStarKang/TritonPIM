"""UPMEM implementation of the request runtime contract.

Compilation records immutable host/device artifacts without acquiring hardware.
Loading a layer captures its packed weights, uploads them, and releases their
host staging buffers. Only the current layer owns MRAM addresses.
"""

from contextlib import ExitStack
import ctypes
from dataclasses import dataclass

from .resident import (
    INPUT, MRAM_BASE, OUTPUT, WEIGHT, ResidentInvocation, ResidentRuntime,
    TransferStats, _artifact_bindings, _build_host, _build_program,
)
from .runtime import current_invocation, resident_lock


@dataclass(frozen=True)
class UPMEMExecutable:
    bundle: dict
    entry: str
    bindings: tuple
    library: object


@dataclass(frozen=True)
class UPMEMLayerMemory:
    addresses: dict
    programs: dict
    bytes_per_unit: int
    weight_bytes_per_unit: int


class CompileInvocation:
    def __init__(self, adapter, kernel, physical_inputs):
        self.adapter = adapter
        self.kernel = kernel
        self.physical_inputs = physical_inputs
        self.calls = 0

    def __enter__(self):
        if current_invocation.get() is not None:
            raise RuntimeError("nested PIM compilation invocation")
        self.token = current_invocation.set(self)
        return self

    def __exit__(self, exception_type, exception, traceback):
        current_invocation.reset(self.token)
        if exception_type is None and self.calls != 1:
            raise RuntimeError(f"{self.kernel.name}: expected one compiled PIM kernel")
        return False

    def launch(self, launcher, bundle, entry, grid, arguments):
        from .driver import UPMEM_TIMING_VERSION

        self.calls += 1
        if self.calls != 1 or self.kernel.artifact is not None:
            raise RuntimeError("a layer kernel must compile exactly one host artifact")
        if launcher.pim_target != "upmem":
            raise ValueError("UPMEM compiler invocation received a different target")
        bindings = _artifact_bindings(
            bundle["target_ir"], [arguments[index] for index in launcher.arg_indices],
            self.physical_inputs, self.kernel.weight_mask,
            self.adapter.spec.common_profile.num_units,
        )
        library = ctypes.CDLL(_build_host(bundle, self.adapter.path))
        version = ctypes.c_uint32.in_dll(library, "pim_timing_version").value
        if version != UPMEM_TIMING_VERSION:
            raise RuntimeError("compiled UPMEM host timing ABI differs from the runtime")
        self.kernel.artifact = UPMEMExecutable(bundle, entry, bindings, library)
        # Do not call the generated host function: its output is discarded here.


class BoundRuntime(ResidentRuntime):
    def __init__(self, plan):
        self.plan = plan
        self.executables = {kernel.name: kernel.artifact for kernel in plan.runtime.kernels}
        super().__init__(plan.spec.common_profile.num_units,
                         plan.spec.common_profile.bank_capacity_limit_bytes)

    def library_for(self, name, bundle):
        executable = self.executables[name]
        if executable.bundle != bundle:
            raise RuntimeError(f"{name}: compiled host artifact changed during the request")
        return executable.library

    def program_for(self, kernel, addresses):
        if addresses != self.plan.backend_plan.addresses[kernel.name]:
            raise RuntimeError(f"{kernel.name}: captured placement differs from the compiled plan")
        return self.plan.backend_plan.programs[kernel.name]


class UPMEMKernelBinding:
    def __init__(self, kernel, runtime):
        self.name = kernel.name
        self.weight_mask = kernel.weight_mask
        self.runtime = runtime
        self.artifact = None


class UPMEMAdapter:
    def __init__(self, spec):
        from .driver import _build_upmem_runtime

        self.spec = spec
        self.path = _build_upmem_runtime()
        self.library = ctypes.CDLL(self.path)
        self.library.upmem_session_open.argtypes = [ctypes.c_uint32]
        self.library.upmem_session_open.restype = None
        self.library.upmem_session_close.argtypes = []
        self.library.upmem_session_close.restype = None
        self.runtime = None
        self.bindings = {}
        self.totals = {name: 0 for name, _ in TransferStats._fields_}
        self.totals.update(resource_opens=0, resource_closes=0)
        self.cleanup = None

    def validate_config(self, config):
        ResidentRuntime.validate_config(config)
        hierarchy = config.torch_config.config
        if hierarchy.p_k != 1:
            raise ValueError("UPMEM layer streaming currently requires p_k=1")
        m, n, k = config.padded_shape
        if config.orientation == "dual":
            m, n = n, m
        extent_m, extent_n, extent_k = hierarchy.execution_extent
        if (m > extent_m * hierarchy.p_m or n > extent_n * hierarchy.p_n
                or k > extent_k):
            raise ValueError("UPMEM layer streaming requires one scheduling round per GEMM")

    def compile_invocation(self, kernel, physical_inputs):
        return CompileInvocation(self, kernel, physical_inputs)

    def finalize_plan(self, plan):
        kernels = plan.runtime.kernels
        addresses = {kernel.name: {} for kernel in kernels}
        offset = 0
        for kernel in kernels:
            for binding in kernel.artifact.bindings:
                if binding.role == WEIGHT:
                    addresses[kernel.name][binding.symbol] = MRAM_BASE + offset
                    offset += binding.size
        weight_bytes = offset
        input_bytes = max(sum(binding.size for binding in kernel.artifact.bindings
                              if binding.role == INPUT) for kernel in kernels)
        output_base = offset + input_bytes
        output_bytes = max(binding.size for kernel in kernels
                           for binding in kernel.artifact.bindings if binding.role == OUTPUT)
        footprint = output_base + output_bytes
        if footprint > self.spec.common_profile.bank_capacity_limit_bytes:
            raise ValueError(f"layer {plan.index} needs {footprint} MRAM bytes per DPU")
        programs = {}
        for kernel in kernels:
            cursor = weight_bytes
            for binding in kernel.artifact.bindings:
                if binding.role == INPUT:
                    addresses[kernel.name][binding.symbol] = MRAM_BASE + cursor
                    cursor += binding.size
                elif binding.role == OUTPUT:
                    addresses[kernel.name][binding.symbol] = MRAM_BASE + output_base
            programs[kernel.name] = _build_program(kernel.artifact.bundle, addresses[kernel.name])
        return UPMEMLayerMemory(addresses, programs, footprint, weight_bytes)

    def open(self):
        if self.cleanup is not None or self.totals["resource_opens"]:
            raise RuntimeError("UPMEM adapter cannot be opened twice")
        with ExitStack() as cleanup:
            cleanup.enter_context(resident_lock)
            self.library.upmem_session_open(self.spec.common_profile.num_units)
            self.totals["resource_opens"] += 1
            cleanup.callback(self._release_resources)
            cleanup.callback(self._close_layer)
            self.cleanup = cleanup.pop_all()

    def _release_resources(self):
        self.library.upmem_session_close()
        self.totals["resource_closes"] += 1

    def _close_layer(self):
        if self.runtime is not None:
            for name, value in self.runtime.stats().items():
                self.totals[name] += value
            self.runtime.close()
            self.runtime = None
            self.bindings.clear()

    def load_layer(self, plan):
        if self.cleanup is None:
            raise RuntimeError("UPMEM resources are not open")
        self._close_layer()
        self.runtime = BoundRuntime(plan)
        if self.runtime.path != self.path:
            raise RuntimeError("UPMEM runtime changed after compilation")
        for kernel in plan.runtime.kernels:
            binding = UPMEMKernelBinding(kernel, self.runtime)
            self.bindings[kernel.name] = binding
            physical = kernel.preparation_inputs()
            with ResidentInvocation(binding, physical, capture=True):
                kernel.compiled(*physical)
            if self.runtime.library.upmem_resident_rounds(binding.artifact.handle) != 1:
                raise ValueError(f"{kernel.name}: captured more than one scheduling round")
            for index, region in enumerate(binding.artifact.bindings):
                if (region.role == WEIGHT
                        and self.runtime.library.upmem_resident_windows(binding.artifact.handle, index) != 1):
                    raise ValueError(f"{kernel.name}: captured more than one weight window")
        self.runtime.finalize()
        self.runtime.reset_inputs()
        return {"capture_ms": self.runtime.capture_ms, "upload_ms": self.runtime.upload_ms}

    def invocation(self, kernel, physical_inputs):
        binding = self.bindings[kernel.name]
        if binding.runtime is not self.runtime or not self.runtime.ready:
            raise RuntimeError("UPMEM layer is not ready")
        return ResidentInvocation(binding, physical_inputs, capture=False)

    def stats(self):
        totals = dict(self.totals)
        if self.runtime is not None:
            for name, value in self.runtime.stats().items():
                totals[name] += value
        return totals

    def close(self):
        if self.cleanup is not None:
            self.cleanup.close()
            self.cleanup = None
