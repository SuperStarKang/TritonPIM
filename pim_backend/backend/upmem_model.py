"""Session-owned memory and native execution for compiled model layers."""

import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile

from triton.runtime.cache import get_cache_manager

from .compiler import _run_translate
from .residency import INPUT as INPUT_KIND, OUTPUT as OUTPUT_KIND, WEIGHT as WEIGHT_KIND
from .residency import OperandRequirement, StageRequirement
from .upmem_artifact import INPUT, MRAM_BASE, OUTPUT, WEIGHT, TransferStats, _artifact_bindings, _build_program
from .runtime import current_invocation


# binding role -> memory table 종류
ROLE_KINDS = {WEIGHT: WEIGHT_KIND, INPUT: INPUT_KIND, OUTPUT: OUTPUT_KIND}

# upmem_model.h UPMEM_MODEL_PACK: weight를 pool에 배치만 하고 보내지 않는다.
PACK = 2


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
        self.memory = None
        # group index -> native block 표. bind가 만들고 enter_layer가 보낸다.
        self.groups = {}
        self.loaded_group = None
        self.totals = {"program_loads": 0, "device_compiles": 0, "weight_pushes": 0}
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
            "group_create": ([uint, uint], pointer),
            "group_add": ([pointer, pointer], None),
            "group_push": ([pointer], None),
            "group_stats": ([pointer, ctypes.POINTER(TransferStats)], None),
            "group_close": ([pointer], None),
        }
        for name, (arguments, returns) in signatures.items():
            function = getattr(self.library, "upmem_model_" + name)
            function.argtypes = arguments
            function.restype = returns
        self.library.upmem_runtime_load.argtypes = [ctypes.c_char_p]
        self.library.upmem_runtime_load.restype = ctypes.c_int

    def compile_invocation(self, stage, physical_inputs):
        return ModelCompileInvocation(self, stage, physical_inputs)

    def requirements(self, plans):
        # 프로그램은 IRAM에 적재되므로 MRAM state는 없다. 입력·출력 scratch는 모든
        # stage가 같은 영역을 처음부터 쓴다.
        return [
            StageRequirement(plan.index, stage.name, tuple(
                OperandRequirement(index, ROLE_KINDS[binding.role], binding.size)
                for index, binding in enumerate(stage.bindings)))
            for plan in plans for stage in plan.stages
        ]

    def place(self, plans, memory):
        """Link every stage program at the MRAM offsets of the memory table.

        The slots are contiguous, so one layer group fills the weight regions with
        one transfer. Layers in the same slot get the same weight addresses, so a
        group change overwrites only the weights.
        """
        self.memory = memory
        for plan in plans:
            for stage in plan.stages:
                stage.addresses = {
                    binding.symbol: MRAM_BASE + memory.entries[(plan.index, stage.name, index)].offset
                    for index, binding in enumerate(stage.bindings)}
                stage.program = _build_program(stage.bundle, stage.addresses, stats=self.totals)
            plan.weight_bytes_per_unit = sum(binding.size for stage in plan.stages
                                             for binding in stage.bindings if binding.role == WEIGHT)
            plan.backend_plan = {
                "slot": memory.slot_of_layer[plan.index],
                "group": memory.group_of_layer[plan.index],
                "weight_bytes": plan.weight_bytes_per_unit,
                "bytes_per_unit": memory.bytes_per_unit,
            }

    def memory_table(self, plans):
        """The memory table rows with MRAM addresses; every DPU has this layout."""
        return [{"kind": entry.kind, "layer": entry.layer, "stage": entry.stage,
                 "operand": entry.operand, "slot": entry.slot, "group": entry.group,
                 "pa": MRAM_BASE + entry.offset, "bytes": entry.bytes}
                for entry in self.memory.entries.values()]

    def bind(self, plans, units):
        """Runtime: create the native handle of every placed stage on the open DPU set,
        lay out every weight in its stage pool once, and record each group's block table."""
        for plan in plans:
            for stage in plan.stages:
                self._create_stage(stage, units)
        stages_of_group = {}
        for plan in plans:
            for stage in plan.stages:
                if any(stage.weight_mask):
                    stages_of_group.setdefault(self.memory.group_of_layer[plan.index], []).append(stage)
        for group, stages in stages_of_group.items():
            sizes = [binding.size for stage in stages for binding in stage.bindings
                     if binding.role == WEIGHT]
            self.groups[group] = self.library.upmem_model_group_create(len(sizes), max(sizes))
            for stage in stages:
                self.invoke(stage, stage.preparation_matrices, stage.preparation_output,
                            upload_weights=PACK)
                self.library.upmem_model_group_add(self.groups[group], stage.handle)

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

    def invoke(self, stage, matrices, output, *, upload_weights=0):
        argument_count = len(stage.scalar_arguments)
        for instance in range(stage.instance_count):
            tensors = (*matrices[instance], output[instance])
            for argument, operand in stage.pointer_operands.items():
                stage.arguments[instance * argument_count + argument] = tensors[operand].data_ptr()
        self.library.upmem_model_arguments(stage.handle, stage.arguments, int(upload_weights))
        stage.function(stage.handle, *stage.scalar_arguments)

    def enter_layer(self, plan):
        # group이 이미 올라와 있으면 보내지 않는다. 아니면 group 전체 weight를 한 번에 보낸다.
        group = self.memory.group_of_layer[plan.index]
        if group == self.loaded_group or group not in self.groups:
            return
        self.load_program(plan.stages[0])
        self.library.upmem_model_group_push(self.groups[group])
        self.totals["weight_pushes"] += 1
        self.loaded_group = group

    def exit_layer(self, plan):
        """다음 group이 같은 weight 영역을 덮어쓴다. 해제할 device 상태는 없다."""

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
        for group in self.groups.values():
            stats = TransferStats()
            self.library.upmem_model_group_stats(group, ctypes.byref(stats))
            for name in totals:
                totals[name] += getattr(stats, name)
        totals.update(self.totals)
        return totals

    def close(self):
        for stage in self.stages:
            self.library.upmem_model_close(stage.handle)
        self.stages.clear()
        for group in self.groups.values():
            self.library.upmem_model_group_close(group)
        self.groups.clear()
