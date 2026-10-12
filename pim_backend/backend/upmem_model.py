"""Session-owned memory and native execution for compiled model layers."""

import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile

from triton.runtime.cache import get_cache_manager

from .compiler import _run_translate
from .residency import LAYER, StageRequirement
from .upmem_artifact import INPUT, MRAM_BASE, OUTPUT, WEIGHT, TransferStats, _artifact_bindings, _build_program
from .runtime import current_invocation


def _build_model_host(bundle, runtime_path):
    from .driver import _upmem_host_flags

    source = _run_translate(bundle["target_ir"], "--upmem-to-model-host")
    directory = Path(__file__).resolve().parent
    key = hashlib.sha256((source + runtime_path
                          + (directory / "upmem_model.h").read_text()).encode()).hexdigest()
    cache = get_cache_manager(key)
    path = cache.get_file("model_host.so")
    if path is not None:
        return path
    with tempfile.TemporaryDirectory() as build_dir:
        cpp = Path(build_dir, "model.cpp")
        library = Path(build_dir, "model_host.so")
        cpp.write_text(source)
        subprocess.check_call([
            "g++", "-O2", "-std=c++17", "-fPIC", "-shared", "-fopenmp",
            f"-I{directory}", str(cpp), *_upmem_host_flags(), runtime_path,
            "-Wl,-rpath," + str(Path(runtime_path).parent), "-o", str(library),
        ])
        return cache.put(library.read_bytes(), "model_host.so", binary=True)


class ModelCompileInvocation:
    def __init__(self, runtime, stage, physical_inputs):
        self.runtime = runtime
        self.stage = stage
        self.physical_inputs = physical_inputs

    def __enter__(self):
        self.token = current_invocation.set(self)
        return self

    def __exit__(self, exception_type, exception, traceback):
        current_invocation.reset(self.token)
        return False

    def launch(self, launcher, bundle, entry, grid, arguments):
        stage = self.stage
        host_arguments = [arguments[index] for index in launcher.arg_indices]
        config = stage.config
        stage.bindings = _artifact_bindings(
            bundle["target_ir"], host_arguments, self.physical_inputs,
            stage.weight_mask, config.p_m * config.p_n * config.p_k,
        )
        stage.bundle = bundle
        stage.library = ctypes.CDLL(_build_model_host(bundle, self.runtime.path))
        stage.function = getattr(stage.library, entry)
        stage.function.argtypes = [ctypes.c_void_p] + [
            ctypes.c_void_p if pointer else ctypes.c_int32
            for pointer in launcher.arg_is_pointer
        ] + [ctypes.c_int32] * 3
        stage.function.restype = None
        stage.scalar_arguments = []
        stage.pointer_operands = {}
        for index, (argument, pointer) in enumerate(zip(host_arguments, launcher.arg_is_pointer)):
            if pointer:
                matches = [operand for operand, tensor in enumerate(self.physical_inputs)
                           if tensor.data_ptr() == argument.data_ptr()]
                stage.pointer_operands[index] = matches[0] if matches else 2
                stage.scalar_arguments.append(ctypes.c_void_p())
            else:
                stage.scalar_arguments.append(ctypes.c_int32(int(argument)))
        stage.scalar_arguments.extend(ctypes.c_int32(size) for size in grid)


class UPMEMModelRuntime:
    def __init__(self, adapter):
        self.path = adapter.path
        self.library = adapter.library
        self.stages = []
        self.loaded_program = None
        self.policy = None
        self.loaded = set()
        self.totals = {"program_loads": 0, "device_compiles": 0}
        pointer = ctypes.c_void_p
        uint = ctypes.c_uint32
        uints = ctypes.POINTER(uint)
        signatures = {
            "create": ([uint, uint, uints, uints, uints, uints, uints,
                        ctypes.POINTER(ctypes.c_uint64), uint, uint, uint], pointer),
            "arguments": ([pointer, ctypes.POINTER(pointer), uint], None),
            "stats": ([pointer, ctypes.POINTER(TransferStats)], None),
            "mapping": ([pointer, uints], None),
            "close": ([pointer], None),
        }
        for name, (arguments, returns) in signatures.items():
            function = getattr(self.library, "upmem_model_" + name)
            function.argtypes = arguments
            function.restype = returns
        self.library.upmem_runtime_load.argtypes = [ctypes.c_char_p]
        self.library.upmem_runtime_load.restype = ctypes.c_int

    def compile_invocation(self, stage, physical_inputs):
        return ModelCompileInvocation(self, stage, physical_inputs)

    @staticmethod
    def _role_bytes(stage, role):
        return sum(binding.size for binding in stage.bindings if binding.role == role)

    def requirements(self, plans):
        # 프로그램은 IRAM에 적재되므로 MRAM state는 없다. 입력·출력 scratch는 모든
        # stage가 같은 영역을 쓰고, 출력은 stage마다 slot 하나다.
        return [
            StageRequirement(plan.index, stage.name, self._role_bytes(stage, WEIGHT), 0,
                             self._role_bytes(stage, INPUT),
                             max(binding.size for binding in stage.bindings
                                 if binding.role == OUTPUT))
            for plan in plans for stage in plan.stages
        ]

    def place(self, plans, residency):
        """Lay out MRAM for the chosen policy and link every stage program.

        P-3 (layer) reuses one weight region for every layer; P-1/P-2 give each
        layer its own weight addresses so nothing is uploaded twice.
        """
        self.policy = residency.policy
        weight_region = 0
        input_bytes = 0
        output_bytes = 0
        offset = 0
        for plan in plans:
            if self.policy == LAYER:
                offset = 0
            start = offset
            for stage in plan.stages:
                stage.addresses = {}
                for binding in stage.bindings:
                    if binding.role == WEIGHT:
                        stage.addresses[binding.symbol] = MRAM_BASE + offset
                        offset += binding.size
                input_bytes = max(input_bytes, self._role_bytes(stage, INPUT))
                output_bytes = max(output_bytes, *(binding.size for binding in stage.bindings
                                                   if binding.role == OUTPUT))
            plan.weight_bytes_per_unit = offset - start
            weight_region = max(weight_region, offset)
        footprint = weight_region + input_bytes + output_bytes
        if footprint != residency.bytes_per_unit:
            raise RuntimeError(
                f"MRAM layout needs {footprint} bytes per unit; the residency plan "
                f"computed {residency.bytes_per_unit}")
        for plan in plans:
            for stage in plan.stages:
                offset = weight_region
                for binding in stage.bindings:
                    if binding.role == INPUT:
                        stage.addresses[binding.symbol] = MRAM_BASE + offset
                        offset += binding.size
                    elif binding.role == OUTPUT:
                        stage.addresses[binding.symbol] = MRAM_BASE + weight_region + input_bytes
                stage.program = _build_program(stage.bundle, stage.addresses, stats=self.totals)
            plan.backend_plan = {
                "policy": self.policy,
                "weight_bytes": plan.weight_bytes_per_unit,
                "input_offset": weight_region,
                "output_offset": weight_region + input_bytes,
                "bytes_per_unit": footprint,
            }

    def bind(self, plans, units):
        """Runtime: create the native handle of every placed stage on the open DPU set."""
        for plan in plans:
            for stage in plan.stages:
                self._create_stage(stage, units)

    def _create_stage(self, stage, units):
        uint = ctypes.c_uint32
        pool_count = uint.in_dll(stage.library, "pim_model_pool_count").value
        pool_bytes = (ctypes.c_uint64 * pool_count).in_dll(stage.library, "pim_model_pool_bytes")
        mapping = (uint * 3).in_dll(stage.library, "pim_model_device_mapping")
        metadata_bytes = uint.in_dll(stage.library, "pim_model_metadata_bytes").value
        config = stage.config
        stage.handle = self.library.upmem_model_create(
            stage.instance_count, stage.p_b,
            (uint * 3)(config.p_m, config.p_n, config.p_k), mapping,
            (uint * 3)(*(binding.role for binding in stage.bindings)),
            (uint * 3)(*(stage.addresses[binding.symbol] for binding in stage.bindings)),
            (uint * 3)(*(binding.size for binding in stage.bindings)),
            pool_bytes, pool_count, metadata_bytes, len(stage.scalar_arguments),
        )
        self.stages.append(stage)
        stage.arguments = (ctypes.c_void_p * (stage.instance_count * len(stage.scalar_arguments)))()
        stage.physical_mapping = (uint * units)()
        self.library.upmem_model_mapping(stage.handle, stage.physical_mapping)

    def load_program(self, stage):
        self.totals["program_loads"] += self.library.upmem_runtime_load(stage.program.encode())
        self.loaded_program = stage.program

    def invoke(self, stage, matrices, output, *, upload_weights=False):
        argument_count = len(stage.scalar_arguments)
        for instance in range(stage.instance_count):
            tensors = (*matrices[instance], output[instance])
            for argument, operand in stage.pointer_operands.items():
                stage.arguments[instance * argument_count + argument] = tensors[operand].data_ptr()
        self.library.upmem_model_arguments(stage.handle, stage.arguments, int(upload_weights))
        stage.function(stage.handle, *stage.scalar_arguments)

    def load_weights(self, plan):
        if plan.stages:
            self.load_program(plan.stages[0])
        for stage in plan.stages:
            if any(stage.weight_mask):
                self.invoke(stage, stage.preparation_matrices, stage.preparation_output,
                            upload_weights=True)

    def enter_layer(self, plan):
        if plan.index in self.loaded:
            return
        self.load_weights(plan)
        self.loaded.add(plan.index)

    def exit_layer(self, plan):
        # P-3에서는 다음 layer가 같은 weight 영역을 덮어쓴다. 해제할 device 상태는 없다.
        if self.policy == LAYER:
            self.loaded.discard(plan.index)

    def weight_table(self, plans):
        table = {}
        for plan in plans:
            for stage in plan.stages:
                for binding in stage.bindings:
                    if binding.role == WEIGHT:
                        table[(plan.index, stage.name, binding.operand)] = {
                            "tensor": stage.weights[binding.operand],
                            "address": stage.addresses[binding.symbol],
                            "bytes_per_unit": binding.size,
                            "mapping": stage.physical_mapping,
                        }
        return table

    def run(self, stage, matrices, output):
        self.load_program(stage)
        self.invoke(stage, matrices, output)

    def stats(self):
        totals = {name: 0 for name, _ in TransferStats._fields_}
        for stage in self.stages:
            stats = TransferStats()
            self.library.upmem_model_stats(stage.handle, ctypes.byref(stats))
            for name in totals:
                totals[name] += getattr(stats, name)
        totals.update(self.totals)
        return totals

    def close(self):
        for stage in self.stages:
            self.library.upmem_model_close(stage.handle)
        self.stages.clear()
