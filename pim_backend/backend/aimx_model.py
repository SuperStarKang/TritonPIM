"""Session-owned AiMX contexts and stage execution for compiled model layers.

Each stage runs one model stub (--aimx-to-model-stub). Instances are batched over
FPGAs: job = batch slot * devices + device, and every job owns one context on
one FPGA. Weights are staged only by upload calls; run calls stage activations.
"""

import ctypes

from .compiler import _run_translate
from .residency import LAYER, MODEL, WEIGHTS, StageRequirement
from .runtime import current_invocation


class AiMXModelCall(ctypes.Structure):
    _fields_ = [
        ("contexts", ctypes.POINTER(ctypes.c_void_p)),
        ("fpgas", ctypes.POINTER(ctypes.c_int)),
        ("pointers", ctypes.POINTER(ctypes.c_void_p)),
        ("weights", ctypes.POINTER(ctypes.c_int)),
        ("instances", ctypes.c_uint32),
        ("batch_size", ctypes.c_uint32),
        ("arguments", ctypes.c_uint32),
        ("upload", ctypes.c_int),
        ("batch_base", ctypes.c_uint32),
    ]


class ModelCompileInvocation:
    def __init__(self, runtime, stage, physical_inputs):
        self.runtime = runtime
        self.stage = stage
        self.physical_inputs = physical_inputs

    def __enter__(self):
        if current_invocation.get() is not None:
            raise RuntimeError("nested PIM compilation invocation")
        self.token = current_invocation.set(self)
        return self

    def __exit__(self, exception_type, exception, traceback):
        current_invocation.reset(self.token)
        return False

    def launch(self, launcher, bundle, entry, grid, arguments):
        from .driver import _STAGE_NAMES, _build_aimx_model_so

        if launcher.pim_target != "aimx":
            raise ValueError("AiMX model compilation received a different target")
        stage = self.stage
        host_arguments = [arguments[index] for index in launcher.arg_indices]
        source = _run_translate(bundle["target_ir"], "--aimx-to-model-stub")
        stage.library = ctypes.CDLL(_build_aimx_model_so(source, bundle["descriptors"]))
        stage.function = getattr(stage.library, entry)
        # host 함수는 kernel 인자 뒤에 grid 세 값을 받는다 (driver의 일반 launch와 같다).
        stage.function.argtypes = [ctypes.POINTER(AiMXModelCall)] + [
            ctypes.c_void_p if pointer else ctypes.c_int32
            for pointer in launcher.arg_is_pointer
        ] + [ctypes.c_int32] * 3
        stage.function.restype = None
        stage.timing_names = _STAGE_NAMES["aimx"]
        stage.last_native_ms = dict.fromkeys(stage.timing_names, 0.0)
        stage.last_pim_exec = 0

        library = stage.library
        slots = ctypes.c_uint32.in_dll(library, "pim_model_slot_count").value
        stub_arguments = ctypes.c_uint32.in_dll(library, "pim_model_argument_count").value
        if stub_arguments != len(host_arguments) + 3:
            raise RuntimeError(
                f"{stage.name}: model stub takes {stub_arguments} arguments; the launcher "
                f"passes {len(host_arguments)} kernel arguments and 3 grid values")
        stage.devices = ctypes.c_uint32.in_dll(library, "pim_model_device_count").value
        config = stage.config
        if stage.devices != config.p_m * config.p_n * config.p_k:
            raise RuntimeError(f"{stage.name}: model stub devices differ from [p_m,p_n,p_k]")
        stage.slot_bytes = tuple((ctypes.c_int64 * slots).in_dll(library, "pim_model_slot_bytes"))
        slot_input = tuple((ctypes.c_int64 * slots).in_dll(library, "pim_model_slot_input"))

        # 포인터 인자가 어느 GEMM operand인지: 0/1은 입력, 2는 출력.
        stage.scalar_arguments = []
        stage.pointer_operands = {}
        for index, (argument, pointer) in enumerate(zip(host_arguments, launcher.arg_is_pointer)):
            if pointer:
                matches = [operand for operand, tensor in enumerate(self.physical_inputs)
                           if tensor.data_ptr() == argument.data_ptr()]
                stage.pointer_operands[index] = matches[0] if matches else 2
                stage.scalar_arguments.append(None)
            else:
                stage.scalar_arguments.append(ctypes.c_int32(int(argument)))
        stage.scalar_arguments.extend(ctypes.c_int32(size) for size in grid)
        stage.argument_weights = tuple(
            int(index in stage.pointer_operands and stage.pointer_operands[index] < 2
                and stage.weight_mask[stage.pointer_operands[index]])
            for index in range(len(stage.scalar_arguments)))
        stage.slot_operands = tuple(
            stage.pointer_operands.get(argument) if argument >= 0 else None
            for argument in slot_input)
        stage.slot_weights = tuple(
            argument >= 0 and stage.argument_weights[argument] == 1
            for argument in slot_input)


class AiMXModelRuntime:
    def __init__(self, adapter):
        self.library = adapter.library
        self.base_fpga = adapter.spec.runtime_config.base_fpga
        self.units = adapter.spec.common_profile.num_units
        self.stages = []
        self.policy = None
        self.loaded = set()
        self.loaded_program = None
        self.totals = {"weight_uploads": 0, "weight_bytes": 0,
                       "state_releases": 0, "invocations": 0}

    def compile_invocation(self, stage, physical_inputs):
        return ModelCompileInvocation(self, stage, physical_inputs)

    @staticmethod
    def _bytes(stage, weights):
        return sum(size for size, weight in zip(stage.slot_bytes, stage.slot_weights)
                   if weight == weights)

    def requirements(self, plans):
        # stage마다 자기 activation buffer를 가지므로 공용 scratch는 없고 state에 들어간다.
        return [
            StageRequirement(plan.index, stage.name,
                             self._bytes(stage, True), self._bytes(stage, False), 0, 0)
            for plan in plans for stage in plan.stages
        ]

    def place(self, plans, residency):
        self.policy = residency.policy
        for plan in plans:
            plan.weight_bytes_per_unit = sum(self._bytes(stage, True) for stage in plan.stages)
            plan.backend_plan = {
                "policy": residency.policy,
                "weight_bytes": plan.weight_bytes_per_unit,
                "state_bytes": sum(self._bytes(stage, False) for stage in plan.stages),
                "bytes_per_unit": residency.bytes_per_unit,
            }

    def bind(self, plans):
        """Runtime: create the FPGA contexts of every placed stage."""
        for plan in plans:
            for stage in plan.stages:
                self._create_stage(stage)

    def _create_stage(self, stage):
        jobs = stage.p_b * stage.devices
        if jobs > self.units:
            raise ValueError(f"{stage.name}: {jobs} jobs exceed {self.units} FPGAs")
        slots = len(stage.slot_bytes)
        arguments = len(stage.scalar_arguments)
        stage.fpgas = (ctypes.c_int * jobs)(*(self.base_fpga + job for job in range(jobs)))
        stage.contexts = (ctypes.c_void_p * jobs)(*(
            self.library.aimx_model_create(fpga, slots) for fpga in stage.fpgas))
        stage.pointers = (ctypes.c_void_p * (stage.instance_count * arguments))()
        stage.weight_flags = (ctypes.c_int * arguments)(*stage.argument_weights)
        stage.keep_weights = (ctypes.c_int * slots)(*(int(weight) for weight in stage.slot_weights))
        stage.keep_none = (ctypes.c_int * slots)()
        stage.call = AiMXModelCall(
            ctypes.cast(stage.contexts, ctypes.POINTER(ctypes.c_void_p)),
            ctypes.cast(stage.fpgas, ctypes.POINTER(ctypes.c_int)),
            ctypes.cast(stage.pointers, ctypes.POINTER(ctypes.c_void_p)),
            ctypes.cast(stage.weight_flags, ctypes.POINTER(ctypes.c_int)),
            stage.instance_count, stage.p_b, arguments, 0, 0,
        )
        self.stages.append(stage)

    def invoke(self, stage, matrices, output, *, upload):
        arguments = len(stage.scalar_arguments)
        for instance in range(stage.instance_count):
            tensors = (*matrices[instance], output[instance])
            for argument, operand in stage.pointer_operands.items():
                stage.pointers[instance * arguments + argument] = tensors[operand].data_ptr()
        # 함수 인자의 포인터는 instance 0이다. stub은 job 안에서 instance별 표를 쓴다.
        call_arguments = [
            ctypes.c_void_p(stage.pointers[index]) if value is None else value
            for index, value in enumerate(stage.scalar_arguments)
        ]
        stage.call.upload = int(upload)
        stage.function(ctypes.byref(stage.call), *call_arguments)
        # stub이 호출마다 0으로 두고 누적한 native 시간. FPGA thread마다 따로 재서 더하므로
        # 여러 FPGA가 동시에 일하면 합이 wall time보다 클 수 있다(device 시간의 합).
        timing = (ctypes.c_double * len(stage.timing_names)).in_dll(stage.library, "pim_timing_ms")
        stage.last_native_ms = dict(zip(stage.timing_names, timing))
        stage.last_pim_exec = ctypes.c_uint64.in_dll(stage.library, "pim_invoke_count").value

    def enter_layer(self, plan):
        if plan.index in self.loaded:
            return
        for stage in plan.stages:
            if any(stage.weight_mask):
                self.invoke(stage, stage.preparation_matrices, stage.preparation_output,
                            upload=True)
                self.totals["weight_uploads"] += 1
                self.totals["weight_bytes"] += self._bytes(stage, True) * len(stage.fpgas)
        self.loaded.add(plan.index)

    def exit_layer(self, plan):
        if self.policy == MODEL:
            return
        for stage in plan.stages:
            keep = stage.keep_weights if self.policy == WEIGHTS else stage.keep_none
            for context in stage.contexts:
                self.library.aimx_model_release(context, keep)
            self.totals["state_releases"] += 1
        if self.policy == LAYER:
            self.loaded.discard(plan.index)

    def run(self, stage, matrices, output):
        self.invoke(stage, matrices, output, upload=False)
        self.totals["invocations"] += 1

    def weight_table(self, plans):
        table = {}
        for plan in plans:
            for stage in plan.stages:
                for operand, weight in enumerate(stage.weights):
                    if weight is None:
                        continue
                    table[(plan.index, stage.name, operand)] = {
                        "tensor": weight,
                        "bytes_per_unit": sum(
                            size for size, slot_operand, slot_weight in zip(
                                stage.slot_bytes, stage.slot_operands, stage.slot_weights)
                            if slot_weight and slot_operand == operand),
                        "fpgas": tuple(stage.fpgas),
                    }
        return table

    def stats(self):
        totals = dict(self.totals)
        totals["live_bytes"] = sum(
            self.library.aimx_model_live_bytes(context)
            for stage in self.stages for context in stage.contexts)
        return totals

    def close(self):
        for stage in self.stages:
            for context in stage.contexts:
                self.library.aimx_model_destroy(context)
        self.stages.clear()
