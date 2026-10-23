"""Session-owned AiMX contexts and stage execution for compiled model layers.

Each stage runs one model stub (--aimx-to-model-stub). Instances are batched over
FPGAs: job = batch slot * devices + device, and every job owns one context on
one FPGA. Run calls stage activations; weights are staged only by pack calls,
which write a group's weights into one host image per FPGA laid out like the
group's weight range. Entering the first layer of a group copies its images with
one transfer per (region, FPGA) and drops them; the group's layers then run
without further uploads. While a group runs, a background thread packs the next
group: pack calls write host memory only, since PIM computation and DMA to the
same FPGA cannot overlap.

Buffer addresses come from the memory table, laid out as the weights of every
slot, the state of every slot (descriptor chain, PMM), then the activation
scratch. The table is cut at slot boundaries into regions of at most one vendor
allocation, one pim_malloc per region and FPGA. Every region also has a host
shadow of the same size, and a buffer assembles its host copy at its table
offset there: the scratch copy is shared like the device scratch, and descriptor
records are returned once uploaded (unless their addresses are patched later).
Pack calls assemble weights straight into the group image, never the shadow.
Descriptor chains, one per slot since they hold physical addresses, are built
and uploaded at bind, before any request.
"""

import bisect
import ctypes
import time

from .compiler import _run_translate
from .residency import INPUT, OUTPUT, STATE, WEIGHT, OperandRequirement, StageRequirement
from .runtime import GroupLoader, current_invocation, group_order

# operand 시작 주소 정렬. data buffer는 16 KB 정렬이 필요하고(8 KB면 결과가 틀린다, batch 8 실측),
# PMM은 256 KB 정렬이 필요하다(AiMXDialect kPMMAlign). 모든 operand를 이 배수로 보고하므로
# table의 offset은 모두 두 조건을 함께 만족한다.
OPERAND_ALIGNMENT = 256 * 1024
# vendor multi_mmap_drv(int size)가 2 GiB에서 overflow하므로 region(pim_malloc) 하나의 상한이다.
MAX_REGION_BYTES = 1023 * 2 * 1024 * 1024


class AiMXModelRegion(ctypes.Structure):
    # aimx_runtime.h의 aimx_model_region과 같은 배치
    _fields_ = [
        ("fpga", ctypes.c_int),
        ("va", ctypes.c_uint64),
        ("pa", ctypes.c_uint64),
        ("bytes", ctypes.c_int64),
        ("host", ctypes.c_void_p),
    ]


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
        ("pack_hosts", ctypes.POINTER(ctypes.c_void_p)),
        ("pack_vas", ctypes.POINTER(ctypes.c_uint64)),
    ]


class AiMXResidentBuffer(ctypes.Structure):
    # aimx_runtime.h의 aimx_resident_buffer와 같은 배치
    _fields_ = [
        ("va", ctypes.c_uint64),
        ("pa", ctypes.c_uint64),
        ("next", ctypes.c_uint64),
        ("host", ctypes.c_void_p),
        ("bytes", ctypes.c_int64),
        ("alignment", ctypes.c_int64),
        ("idx", ctypes.c_uint32),
        ("kind", ctypes.c_int),
    ]


class AiMXResidentContext(ctypes.Structure):
    _fields_ = [
        ("buffers", ctypes.POINTER(AiMXResidentBuffer)),
        ("count", ctypes.c_size_t),
        ("ready", ctypes.c_bool),
        ("owner", ctypes.c_void_p),
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
        slot_output = tuple((ctypes.c_int64 * slots).in_dll(library, "pim_model_slot_output"))
        slot_kinds = tuple((ctypes.c_int64 * slots).in_dll(library, "pim_model_slot_kind"))

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
        # memory table 종류: weight, state(descriptor chain 1, PMM 2: 실행 사이에 남아야 한다),
        # output(unstage로 비우는 buffer), 나머지는 실행마다 다시 채우는 input(activation, 상수 등)
        stage.slot_table_kinds = tuple(
            WEIGHT if weight else STATE if kind in (1, 2) else OUTPUT if output >= 0 else INPUT
            for weight, kind, output in zip(stage.slot_weights, slot_kinds, slot_output))


class AiMXModelRuntime:
    def __init__(self, adapter):
        self.library = adapter.library
        self.base_fpga = adapter.spec.runtime_config.base_fpga
        self.units = adapter.spec.common_profile.num_units
        # context를 가진 stage. slot을 공유하는 다른 layer의 stage는 여기 없다.
        self.stages = []
        self.memory = None
        # table을 자른 region들의 (시작 offset, bytes). place가 만들고 bind가 FPGA마다 할당한다.
        self.chunks = []
        # (chunk index, FPGA) -> aimx_model_region. close가 context 다음에 해제한다.
        self.regions = {}
        # group index -> 그 group의 layer plan들 (실행 순서). place가 만들고 group 업로드가 쓴다.
        self.plans_of_group = {}
        # group 업로드 순서와 background pack (runtime.GroupLoader). place가 만든다.
        self.loader = None
        # chain_build_ms는 bind에서 slot마다 descriptor chain을 만들어 올린 시간이다 (요청 밖).
        self.totals = {"weight_bytes": 0, "invocations": 0, "run_wall_ms": 0.0, "chain_build_ms": 0.0}
        # "run:mm" 처럼 stage 이름마다 layer를 합친 호출 수, wall time, native 칸별 시간.
        self.breakdown = {}

    def compile_invocation(self, stage, physical_inputs):
        return ModelCompileInvocation(self, stage, physical_inputs)

    @staticmethod
    def _bytes(stage, kind):
        return sum(size for size, table_kind in zip(stage.slot_bytes, stage.slot_table_kinds)
                   if table_kind == kind)

    def requirements(self, plans):
        # weight와 state(descriptor chain, PMM)는 layer의 slot에, activation(input/output)은 모든
        # stage가 처음부터 함께 쓰는 scratch에 놓인다. bytes를 OPERAND_ALIGNMENT 배수로 올려
        # 모든 offset을 정렬한다.
        reading = [stage.name for plan in plans for stage in plan.stages if stage.kv is not None]
        if reading:
            raise NotImplementedError(
                f"AiMX does not read a device key/value cache yet; stages {reading} read one")
        return [
            StageRequirement(plan.index, stage.name, tuple(
                OperandRequirement(index, kind,
                                   (size + OPERAND_ALIGNMENT - 1) // OPERAND_ALIGNMENT * OPERAND_ALIGNMENT)
                for index, (size, kind) in enumerate(zip(stage.slot_bytes, stage.slot_table_kinds))))
            for plan in plans for stage in plan.stages
        ]

    def place(self, plans, memory):
        # phase overlay, key/value cache 영역, 그리고 phase가 바뀔 때 descriptor chain을 다시
        # 만드는 일은 아직 UPMEM만 한다. AiMX는 phase 하나를 offset 0부터 배치한다.
        if len(memory.phases) != 1 or memory.kv_bytes:
            raise NotImplementedError(
                f"AiMX places one phase and no device key/value cache; got phases "
                f"{list(memory.phases)} and {memory.kv_bytes} key/value bytes")
        self.memory = memory
        (self.phase, self.layout), = memory.phases.items()
        for plan in plans:
            self.plans_of_group.setdefault((plan.phase, self.layout.group_of_layer[plan.index]),
                                           []).append(plan)
            plan.weight_bytes_per_unit = sum(self._bytes(stage, WEIGHT) for stage in plan.stages)
            plan.backend_plan = {
                "phase": plan.phase,
                "slot": self.layout.slot_of_layer[plan.index],
                "group": self.layout.group_of_layer[plan.index],
                "weight_bytes": plan.weight_bytes_per_unit,
                "state_bytes": sum(self._bytes(stage, STATE) for stage in plan.stages),
                "scratch_bytes": self.layout.scratch_bytes,
                "bytes_per_unit": memory.bytes_per_unit,
            }
        self.loader = GroupLoader(group_order(memory), self._pack_group, self._push_group,
                                  self._release_group)
        # slot의 weight, slot의 state, scratch 순서로 이어진 조각을 region 상한까지 묶는다.
        # 조각 하나는 나누지 않으므로 한 slot의 weight는 언제나 한 region 안에 있다.
        start = 0
        for size in (*self.layout.slot_weight_bytes, *self.layout.slot_state_bytes,
                     self.layout.scratch_bytes):
            if size > MAX_REGION_BYTES:
                raise ValueError(f"a memory-table piece of {size} bytes exceeds one AiMX region "
                                 f"({MAX_REGION_BYTES} bytes)")
            if self.chunks and self.chunks[-1][1] + size <= MAX_REGION_BYTES:
                self.chunks[-1] = (self.chunks[-1][0], self.chunks[-1][1] + size)
            elif size:
                self.chunks.append((start, size))
            start += size

    def _address(self, entry, fpga, field):
        """va, pa or host (shadow) address of a table entry on one FPGA: the region holding its offset."""
        index = bisect.bisect_right([chunk_start for chunk_start, _ in self.chunks], entry.offset) - 1
        region = ctypes.cast(self.regions[(index, fpga)], ctypes.POINTER(AiMXModelRegion)).contents
        return getattr(region, field) + entry.offset - self.chunks[index][0]

    def bind(self, plans):
        """Runtime: the regions on every FPGA, the contexts at the table addresses (the first
        layer of a slot creates them), then every slot's descriptor chains built and uploaded.
        Weights move only when a group is entered."""
        for fpga in range(self.base_fpga, self.base_fpga + self.units):
            for index, (_, size) in enumerate(self.chunks):
                self.regions[(index, fpga)] = self.library.aimx_model_region_create(
                    fpga, size, OPERAND_ALIGNMENT)
        first = {}
        for plan in plans:
            owner = first.setdefault(self.layout.slot_of_layer[plan.index], plan)
            for stage, slot_stage in zip(plan.stages, owner.stages):
                self._create_stage(plan, stage, slot_stage)
        # chain에는 buffer의 물리 주소가 들어가므로 slot마다 하나다. 요청 전에 만들어 올린다
        # (아직 PIM 연산이 없으므로 업로드가 연산과 겹치지 않는다).
        start = time.perf_counter_ns()
        for stage in self.stages:
            self._call(stage, None, None, upload=3)
        self.totals["chain_build_ms"] += (time.perf_counter_ns() - start) / 1e6

    def _create_stage(self, plan, stage, slot_stage):
        jobs = stage.p_b * stage.devices
        if jobs > self.units:
            raise ValueError(f"{stage.name}: {jobs} jobs exceed {self.units} FPGAs")
        arguments = len(stage.scalar_arguments)
        entries = [self.memory.entries[(plan.phase, plan.index, stage.name, index)]
                   for index in range(len(stage.slot_bytes))]
        stage.offsets = tuple(entry.offset for entry in entries)
        if slot_stage is stage:
            stage.fpgas = (ctypes.c_int * jobs)(*(self.base_fpga + job for job in range(jobs)))
            stage.contexts = (ctypes.c_void_p * jobs)(*(
                self.library.aimx_model_create(
                    fpga, len(entries),
                    (ctypes.c_uint64 * len(entries))(*(self._address(entry, fpga, "va") for entry in entries)),
                    (ctypes.c_int64 * len(entries))(*(entry.bytes for entry in entries)),
                    (ctypes.c_void_p * len(entries))(*(self._address(entry, fpga, "host") for entry in entries)))
                for fpga in stage.fpgas))
            self.stages.append(stage)
        else:
            # 같은 slot의 context(device buffer, descriptor chain)를 쓰므로 배치가 같아야 한다.
            if (stage.slot_bytes, stage.slot_table_kinds, stage.offsets, jobs) != (
                    slot_stage.slot_bytes, slot_stage.slot_table_kinds, slot_stage.offsets,
                    slot_stage.p_b * slot_stage.devices):
                raise ValueError(f"{stage.name}: slot layout differs from {slot_stage.name} "
                                 "of the shared memory slot")
            stage.fpgas = slot_stage.fpgas
            stage.contexts = slot_stage.contexts
        stage.pointers = (ctypes.c_void_p * (stage.instance_count * arguments))()
        stage.weight_flags = (ctypes.c_int * arguments)(*stage.argument_weights)
        stage.call = AiMXModelCall(
            ctypes.cast(stage.contexts, ctypes.POINTER(ctypes.c_void_p)),
            ctypes.cast(stage.fpgas, ctypes.POINTER(ctypes.c_int)),
            ctypes.cast(stage.pointers, ctypes.POINTER(ctypes.c_void_p)),
            ctypes.cast(stage.weight_flags, ctypes.POINTER(ctypes.c_int)),
            stage.instance_count, stage.p_b, arguments, 0, 0, None, None,
        )
        self.breakdown.setdefault(f"run:{stage.name}", {
            "calls": 0, "jobs": jobs, "wall_ms": 0.0, **dict.fromkeys(stage.timing_names, 0.0)})

    def _call(self, stage, matrices, output, upload):
        """One stub call; upload 0 runs the stage, 2 packs its weights into stage.pack_hosts,
        3 builds and uploads its descriptor chains. A chain call reads no tensor, so it
        passes no matrices and the stage keeps its previous pointers."""
        arguments = len(stage.scalar_arguments)
        if matrices is not None:
            for instance in range(stage.instance_count):
                tensors = (*matrices[instance], output[instance])
                for argument, operand in stage.pointer_operands.items():
                    stage.pointers[instance * arguments + argument] = tensors[operand].data_ptr()
        # 함수 인자의 포인터는 instance 0이다. stub은 job 안에서 instance별 표를 쓴다.
        call_arguments = [
            ctypes.c_void_p(stage.pointers[index]) if value is None else value
            for index, value in enumerate(stage.scalar_arguments)
        ]
        stage.call.upload = upload
        stage.function(ctypes.byref(stage.call), *call_arguments)

    def _pack_group(self, key):
        """Host images of one group's weight range, one per FPGA; the range starts at offset 0.

        Pack calls write host memory only, so this may run while other stages compute.
        Returns ({FPGA: image}, bytes of the range).
        """
        layers = self.plans_of_group[key]
        weight_bytes = sum(self.layout.slot_weight_bytes[:len(layers)])
        weight_stages = [(plan, stage) for plan in layers for stage in plan.stages if any(stage.weight_mask)]
        # stage가 image 안 자기 buffer 자리에 바로 조립한다. 아무도 쓰지 않는 정렬 간격은 0으로 읽힌다.
        images = {fpga: self.library.aimx_host_image_create(weight_bytes)
                  for fpga in sorted({fpga for _, stage in weight_stages for fpga in stage.fpgas})}
        for plan, stage in weight_stages:
            entry = next(self.memory.entries[(plan.phase, plan.index, stage.name, index)]
                         for index, kind in enumerate(stage.slot_table_kinds) if kind == WEIGHT)
            # stub은 weight buffer(VA v)를 host + (v - pack_va)에 쓴다. pack_va를 그 FPGA에서
            # offset 0이 있을 VA로 두면 image 안의 위치가 table offset과 같아진다.
            stage.pack_hosts = (ctypes.c_void_p * len(stage.fpgas))(*(images[fpga] for fpga in stage.fpgas))
            stage.pack_vas = (ctypes.c_uint64 * len(stage.fpgas))(
                *(self._address(entry, fpga, "va") - entry.offset for fpga in stage.fpgas))
            stage.call.pack_hosts = ctypes.cast(stage.pack_hosts, ctypes.POINTER(ctypes.c_void_p))
            stage.call.pack_vas = ctypes.cast(stage.pack_vas, ctypes.POINTER(ctypes.c_uint64))
            self._call(stage, stage.preparation_matrices, stage.preparation_output, upload=2)
        # 이 thread의 host 조립 buffer를 돌려준다. 다른 thread는 계속 stage를 실행하고 있을 수 있다.
        self.library.aimx_host_cache_clear_thread()
        return images, weight_bytes

    def _push_group(self, packed):
        """One transfer per region the group's weight range covers, on every FPGA."""
        images, weight_bytes = packed
        items = [(self.regions[(index, fpga)], image + chunk_start,
                  min(chunk_start + size, weight_bytes) - chunk_start)
                 for fpga, image in images.items()
                 for index, (chunk_start, size) in enumerate(self.chunks) if chunk_start < weight_bytes]
        if not items:
            return
        regions, hosts, sizes = zip(*items)
        self.library.aimx_model_push(
            len(items), (ctypes.c_void_p * len(items))(*regions),
            (ctypes.c_void_p * len(items))(*hosts), (ctypes.c_int64 * len(items))(*sizes))
        self.totals["weight_bytes"] += sum(sizes)

    def _release_group(self, packed):
        images, weight_bytes = packed
        for image in images.values():
            self.library.aimx_host_image_destroy(image, weight_bytes)

    def enter_layer(self, plan):
        # group이 바뀔 때만 그 group의 weight를 한 번 올린다 (pack은 대개 앞 group 실행 중에 끝나 있다).
        self.loader.enter((plan.phase, self.layout.group_of_layer[plan.index]))

    def exit_layer(self, plan):
        """Slots stay allocated; the next group overwrites only the weights."""

    def run(self, stage, matrices, output):
        start = time.perf_counter_ns()
        self._call(stage, matrices, output, upload=0)
        wall_ms = (time.perf_counter_ns() - start) / 1e6
        # stub이 호출마다 0으로 두고 누적한 native 시간. FPGA thread마다 따로 재서 더하므로
        # 여러 FPGA가 동시에 일하면 합이 wall time보다 클 수 있다(device 시간의 합).
        # background pack이 같은 stub을 동시에 부르면 이 칸들은 섞인다 (wall time은 정확하다).
        timing = (ctypes.c_double * len(stage.timing_names)).in_dll(stage.library, "pim_timing_ms")
        stage.last_native_ms = dict(zip(stage.timing_names, timing))
        stage.last_pim_exec = ctypes.c_uint64.in_dll(stage.library, "pim_invoke_count").value
        record = self.breakdown[f"run:{stage.name}"]
        record["calls"] += 1
        record["wall_ms"] += wall_ms
        # native는 FPGA thread 합이므로 job 수로 나눠 thread 평균으로 둔다. wall과 같은 척도다.
        for name, milliseconds in stage.last_native_ms.items():
            record[name] += milliseconds / record["jobs"]
        self.totals["run_wall_ms"] += wall_ms
        self.totals["invocations"] += 1

    def weight_table(self, plans):
        table = {}
        for plan in plans:
            for stage in plan.stages:
                for operand, weight in enumerate(stage.weights):
                    if weight is None:
                        continue
                    table[(plan.phase, plan.index, stage.name, operand)] = {
                        "tensor": weight,
                        "bytes_per_unit": sum(
                            size for size, slot_operand, slot_weight in zip(
                                stage.slot_bytes, stage.slot_operands, stage.slot_weights)
                            if slot_weight and slot_operand == operand),
                        "fpgas": tuple(stage.fpgas),
                    }
        return table

    def memory_table(self, plans):
        """One row per table entry on every FPGA of its stage.

        pa is the planned address (region + table offset); device_pa is the address the
        stub allocated, 0 until the stage first runs. They must be equal once allocated.
        Input/output rows of different stages share the scratch addresses.
        """
        first = {}
        layers = {}
        for plan in plans:
            slot = self.layout.slot_of_layer[plan.index]
            first.setdefault(slot, plan)
            layers.setdefault(slot, []).append(plan.index)
        rows = []
        for slot, plan in first.items():
            for stage in plan.stages:
                for fpga, context in zip(stage.fpgas, stage.contexts):
                    resident = ctypes.cast(context, ctypes.POINTER(AiMXResidentContext)).contents
                    for index in range(len(stage.slot_bytes)):
                        entry = self.memory.entries[(plan.phase, plan.index, stage.name, index)]
                        rows.append({"slot": slot, "layers": layers[slot], "stage": stage.name,
                                     "unit": fpga, "buffer": index, "kind": entry.kind,
                                     "offset": entry.offset, "pa": self._address(entry, fpga, "pa"),
                                     "bytes": entry.bytes, "device_pa": resident.buffers[index].pa})
        return rows

    def stats(self):
        totals = dict(self.totals)
        # place 전에 준비가 실패해도 정리 경로가 통계를 읽는다. 그때는 group 업로드가 없다.
        if self.loader is not None:
            totals.update(self.loader.totals)
        totals["breakdown"] = {key: dict(record) for key, record in self.breakdown.items()}
        totals["live_bytes"] = sum(
            self.library.aimx_model_live_bytes(context)
            for stage in self.stages for context in stage.contexts)
        return totals

    def close(self):
        # background pack이 context를 읽고 있을 수 있으므로 먼저 끝낸다.
        if self.loader is not None:
            self.loader.close()
        for stage in self.stages:
            for context in stage.contexts:
                self.library.aimx_model_destroy(context)
        self.stages.clear()
        # context가 region 안의 buffer를 가리키므로 context를 먼저 해제한다.
        for region in self.regions.values():
            self.library.aimx_model_region_destroy(region)
        self.regions.clear()
        # stage 호출이 남긴 host 조립 buffer를 돌려준다. 측정 후보마다 새 DSO가 올라와도 쌓이지 않는다.
        self.library.aimx_host_cache_clear()
