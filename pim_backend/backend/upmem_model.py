"""Session-owned memory and native execution for compiled model layers."""

import ctypes
import hashlib
from itertools import product
import math
from pathlib import Path
import subprocess
import tempfile
import time

from triton.runtime.cache import get_cache_manager

from .compiler import _run_translate
from .residency import INPUT as INPUT_KIND, OUTPUT as OUTPUT_KIND, WEIGHT as WEIGHT_KIND
from .residency import OperandRequirement, StageRequirement, token_rows
from .upmem_artifact import (
    INPUT, KV, MRAM_BASE, OUTPUT, WEIGHT, TransferStats, _artifact_bindings, _build_program, _staging_layout,
)
from .runtime import GroupLoader, current_invocation, group_order


# binding role -> memory table 종류
ROLE_KINDS = {WEIGHT: WEIGHT_KIND, INPUT: INPUT_KIND, OUTPUT: OUTPUT_KIND}

# upmem_model.h UPMEM_MODEL_PACK: weight를 pool에 배치만 하고 보내지 않는다.
PACK = 2
# upmem_model.h UPMEM_MODEL_KV_PACK: 고른 job의 key/value cache operand만 pool에 배치한다.
KV_PACK = 3


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


def _positions_on_rows(stage):
    """Whether the cache positions run along the rows of the GEMV matrix of stage.

    A cache stage is a GEMV (execution.gemm_problem): its matrix is the left operand, or the
    transpose of the right one. An operand that is the transpose of the cache window swaps
    rows and columns once more.
    """
    operand, _, transposed = stage.kv
    return (operand == 0) != transposed


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
            None if stage.kv is None else stage.kv[0],
        )
        stage.bundle = bundle
        if stage.kv is not None:
            symbol, = [binding.symbol for binding in stage.bindings if binding.role == KV]
            # (tile, unit buffer shape): 생성 host 코드가 cache operand를 unit마다 배치하는 단위
            stage.kv_staging = _staging_layout(bundle["target_ir"], symbol)
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
        # (phase, group) -> 그 group의 layer plan들 (실행 순서). place가 만들고 group 업로드가 쓴다.
        self.plans_of_group = {}
        # group 업로드 순서와 background pack (runtime.GroupLoader). place가 만든다.
        self.loader = None
        # 해제한 group 표들의 전송 통계
        self.group_stats = {name: 0 for name, _ in TransferStats._fields_}
        # program_load_ms는 stage 실행 전 IRAM program 적재 시간, invoke_ms는 host pool 준비부터
        # 입력 전송·실행·출력 회수까지 native 호출 시간이다.
        # kv_push_ms는 대기 중인 key/value cache 위치를 host image에 반영하고 cache 영역 전체를 한 번
        # 보낸 시간이고, kv_layout_check_ms는 host 배치식을 생성 pack과 한 번 대조한 시간이다.
        self.totals = {"program_loads": 0, "device_compiles": 0, "weight_pushes": 0,
                       "program_load_ms": 0.0, "invoke_ms": 0.0, "kv_push_ms": 0.0, "kv_layout_check_ms": 0.0}
        # stage -> [(host cache, start, stop)]: cache를 읽는 다음 실행 전에 보낼 위치
        self.kv_pending = {}
        # [DPU, kv_bytes] int8: DPU마다 MRAM [kv_base, capacity)에 있어야 할 bytes. 처음 보낼 때 만든다.
        self.kv_image = None
        # (stage 이름, (M, N, K)) -> {"calls", "program_load_ms", "invoke_ms", "kv_push_ms"}.
        # 모든 layer의 같은 stage가 합쳐진다.
        self.stage_times = {}
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
            "pools_acquire": ([pointer], None),
            "pools_release": ([pointer], None),
            "group_create": ([uint, uint], pointer),
            "group_add": ([pointer, pointer], None),
            "group_push": ([pointer], None),
            "group_stats": ([pointer, ctypes.POINTER(TransferStats)], None),
            "group_close": ([pointer], None),
            "kv_select": ([pointer, uints, uint], None),
            "kv_gather": ([pointer, uint, pointer], None),
            "kv_push_region": ([pointer, uint, pointer, uint], None),
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
        # stage가 같은 영역을 처음부터 쓴다. cache operand는 그 layer의 KEY/VALUE 영역을 읽는다.
        return [
            StageRequirement(plan.index, stage.name, tuple(
                OperandRequirement(index, stage.kv[1] if binding.role == KV else ROLE_KINDS[binding.role],
                                   binding.size)
                for index, binding in enumerate(stage.bindings)))
            for plan in plans for stage in plan.stages
        ]

    def place(self, plans, memory):
        """Link every stage program at the MRAM offsets of the memory table.

        The slots of a phase are contiguous, so one layer group fills the weight regions
        with one transfer. Layers in the same slot get the same weight addresses, so a
        group change overwrites only the weights; the phases share one area, so the first
        group of the next phase overwrites the previous phase.
        """
        self.memory = memory
        # 상주가 없는 배치(shared_slot)는 layer마다 그 자리에서 pack하고 올린다.
        self.loader = GroupLoader(group_order(memory), self._pack_group, self._push_group,
                                  self._release_group, prefetch=not memory.shared_slot)
        for plan in plans:
            layout = memory.phases[plan.phase]
            self.plans_of_group.setdefault((plan.phase, layout.group_of_layer[plan.index]), []).append(plan)
            for stage in plan.stages:
                stage.addresses = {
                    binding.symbol:
                        MRAM_BASE + memory.entries[(plan.phase, plan.index, stage.name, index)].offset
                    for index, binding in enumerate(stage.bindings)}
                stage.program = _build_program(stage.bundle, stage.addresses, stats=self.totals)
            plan.weight_bytes_per_unit = sum(binding.size for stage in plan.stages
                                             for binding in stage.bindings if binding.role == WEIGHT)
            plan.backend_plan = {
                "phase": plan.phase,
                "slot": layout.slot_of_layer[plan.index],
                "group": layout.group_of_layer[plan.index],
                "weight_bytes": plan.weight_bytes_per_unit,
                "bytes_per_unit": memory.bytes_per_unit,
            }

    def memory_table(self, plans):
        """The memory table rows with MRAM addresses; every DPU has this layout."""
        return [{"kind": entry.kind, "phase": entry.phase, "layer": entry.layer, "stage": entry.stage,
                 "operand": entry.operand, "slot": entry.slot, "group": entry.group,
                 "pa": MRAM_BASE + entry.offset, "bytes": entry.bytes}
                for entry in self.memory.entries.values()]

    def bind(self, plans, units):
        """Runtime: create the native handle of every placed stage on the open DPU set.
        Weights move only when a group is entered."""
        for plan in plans:
            for stage in plan.stages:
                self._create_stage(stage, units)

    def _pack_group(self, key):
        """Lay the group's weights out in the stage pools and build its block table.

        PACK calls only fill host pools (no DPU access), so this may run while DPUs compute.
        The table covers the group's weights, which are contiguous in MRAM.
        """
        plans = self.plans_of_group[key]
        stages = [stage for plan in plans for stage in plan.stages if any(stage.weight_mask)]
        if not stages:
            return None
        sizes = [binding.size for stage in stages for binding in stage.bindings
                 if binding.role == WEIGHT]
        table = self.library.upmem_model_group_create(len(sizes), max(sizes))
        for stage in stages:
            self.library.upmem_model_pools_acquire(stage.handle)
            self.invoke(stage, stage.preparation_matrices, stage.preparation_output,
                        upload_weights=PACK)
            self.library.upmem_model_group_add(table, stage.handle)
        return table, stages, plans[0].stages[0]

    def _push_group(self, packed):
        if packed is None:
            return
        table, _, first = packed
        self.load_program(first)
        self.library.upmem_model_group_push(table)
        self.totals["weight_pushes"] += 1

    def _release_group(self, packed):
        if packed is None:
            return
        table, stages, _ = packed
        stats = TransferStats()
        self.library.upmem_model_group_stats(table, ctypes.byref(stats))
        for name in self.group_stats:
            self.group_stats[name] += getattr(stats, name)
        self.library.upmem_model_group_close(table)
        for stage in stages:
            self.library.upmem_model_pools_release(stage.handle)

    def _create_stage(self, stage, units):
        import torch

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
        # cache operand의 unit 안 배치와 DPU마다 맡은 job (image 행 -> job image 행).
        # 생성 pack과의 대조는 처음 보낼 때 한다.
        stage.kv_layout = None if stage.kv is None else self._kv_layout(stage)
        stage.kv_jobs = None if stage.kv is None else torch.tensor(list(stage.physical_mapping), dtype=torch.long)
        stage.kv_checked = False

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

    def write_kv(self, stage, cache, start, stop):
        """Queue positions [start, stop) of cache for the next run of any stage reading the cache.

        Right before that run every queued position goes into the host image of the cache and
        the whole cache range goes out in one transfer. cache is [kv_heads, positions, head_dim]
        and must keep those positions until then; instance i reads the columns
        stage.kv_windows[i] of its token-row view (residency.token_rows), or their transpose if
        stage.kv says so.
        """
        operand, _, transposed = stage.kv
        m, n, k = stage.shape
        rows, cols = (m, k) if operand == 0 else (k, n)
        # operand가 window의 전치면 위치가 열 쪽이다.
        positions, width = (cols, rows) if transposed else (rows, cols)
        windows = {stop - window_start for window_start, stop in stage.kv_windows}
        if (windows != {width}
                or cache.shape[1] != positions
                or max(stop for _, stop in stage.kv_windows) > cache.shape[0] * cache.shape[2]):
            raise ValueError(f"{stage.name}: cache {tuple(cache.shape)} with windows {stage.kv_windows} "
                             f"does not match operand {operand} of (M, N, K) = {stage.shape}")
        if not 0 <= start < stop <= cache.shape[1]:
            raise ValueError(f"positions [{start}, {stop}) are outside the cache of {cache.shape[1]}")
        self.kv_pending.setdefault(stage, []).append((cache, start, stop))

    def _kv_layout(self, stage):
        """Where every element of one unit's cache slot comes from.

        Returns (rows, cols): for each byte of the slot, its row and column in the unit's
        execution tile (UPMEMToHost tileIndex). With one M/N/K round, the unit at coordinates
        (c_m, c_k) holds rows c_m * R + rows and columns c_k * C + cols of the stage matrix.
        """
        import torch

        (tile_rows, tile_cols), (rows, cols) = stage.kv_staging
        config = stage.config
        extent = config.execution_extent
        if (rows, cols) != (extent[0], extent[2]) or rows % tile_rows or cols % tile_cols or config.p_n != 1:
            raise ValueError(f"{stage.name}: cache slot {rows}x{cols} tile {tile_rows}x{tile_cols} does not "
                             f"follow execution extent {extent} with p_n = 1")
        binding, = [binding for binding in stage.bindings if binding.role == KV]
        if binding.size != rows * cols:
            raise ValueError(f"{stage.name}: an INT8 cache slot of {rows}x{cols} needs {rows * cols} bytes, "
                             f"the binding has {binding.size}")
        row = torch.arange(rows).reshape(rows, 1).expand(rows, cols)
        col = torch.arange(cols).reshape(1, cols).expand(rows, cols)
        index = (((row // tile_rows) * (cols // tile_cols) + col // tile_cols) * (tile_rows * tile_cols)
                 + (row % tile_rows) * tile_cols + col % tile_cols).flatten()
        slot_rows = torch.empty(rows * cols, dtype=torch.long)
        slot_cols = torch.empty(rows * cols, dtype=torch.long)
        slot_rows[index] = row.flatten()
        slot_cols[index] = col.flatten()
        return slot_rows, slot_cols

    def _kv_images(self, stage, cache, slot_bytes):
        """The slot_bytes of every job's cache slot from the host cache, [jobs, len(slot_bytes)] int8."""
        import torch

        slot_rows, slot_cols = stage.kv_layout
        config = stage.config
        rows, cols = stage.kv_staging[1]
        # (p_m, 길이), (p_k, 길이): stage 행렬 안의 행과 열
        matrix_rows = (torch.arange(config.p_m) * rows).reshape(-1, 1) + slot_rows[slot_bytes]
        matrix_cols = (torch.arange(config.p_k) * cols).reshape(-1, 1) + slot_cols[slot_bytes]
        positions, features = ((matrix_rows[:, None], matrix_cols[None]) if _positions_on_rows(stage)
                               else (matrix_cols[None], matrix_rows[:, None]))
        positions, features = torch.broadcast_tensors(positions, features)
        # 모든 window의 폭이 같다 (write_kv가 확인한다).
        width = stage.kv_windows[0][1] - stage.kv_windows[0][0]
        # unit이 cache 끝을 넘어 맡는 자리는 생성 pack처럼 0이다.
        inside = (positions < cache.shape[1]) & (features < width)
        starts = torch.tensor([start for start, _ in stage.kv_windows]).reshape(-1, 1, 1, 1)
        # token-row view의 열 c는 KV head c // head_dim의 c % head_dim번째 값이다.
        columns = starts + features.clamp(max=width - 1)
        head_dim = cache.shape[2]
        values = cache[columns // head_dim, positions.clamp(max=cache.shape[1] - 1), columns % head_dim]
        return torch.where(inside, values, torch.zeros((), dtype=cache.dtype)).reshape(-1, len(slot_bytes)).contiguous()

    def _kv_bytes(self, stage, start, stop):
        """Bytes of a slot that hold positions [start, stop) in every unit; the whole slot if they span units."""
        import torch

        on_rows = _positions_on_rows(stage)
        slot_rows, slot_cols = stage.kv_layout
        extent = stage.kv_staging[1][0 if on_rows else 1]
        if stop - start >= extent or start // extent != (stop - 1) // extent:
            return torch.arange(slot_rows.numel())
        # 모든 unit이 같은 배치이므로 unit 안 위치 번호가 같은 자리를 함께 쓴다. 그 위치를 맡지 않은
        # unit의 그 자리는 host cache의 현재 값을 다시 쓴다.
        local = slot_rows if on_rows else slot_cols
        return torch.nonzero((local >= start % extent) & (local <= (stop - 1) % extent)).flatten()

    def _check_kv_layout(self, stage, cache):
        """Compare the host layout with the generated pack of every job; a mismatch raises."""
        import torch

        start = time.perf_counter_ns()
        jobs = stage.p_b * stage.config.p_m * stage.config.p_n * stage.config.p_k
        binding = next(index for index, binding in enumerate(stage.bindings) if binding.role == KV)
        size = stage.bindings[binding].size
        self.library.upmem_model_kv_select(stage.handle, (ctypes.c_uint32 * jobs)(*range(jobs)), jobs)
        windows = [token_rows(cache, slice(None), *window) for window in stage.kv_windows]
        # stage는 cache operand를 연속 배치로 compile했다 (_LayerGemm.prepare).
        operands = [(window.t() if stage.kv[2] else window).contiguous() for window in windows]
        self.library.upmem_model_pools_acquire(stage.handle)
        # 다른 operand는 KV_PACK에서 읽지 않는다. 같은 window를 넘긴다.
        self.invoke(stage, tuple((operand, operand) for operand in operands), stage.preparation_output,
                    upload_weights=KV_PACK)
        packed = torch.empty(jobs * size, dtype=torch.int8)
        self.library.upmem_model_kv_gather(stage.handle, binding, packed.data_ptr())
        self.library.upmem_model_pools_release(stage.handle)
        if not torch.equal(packed, self._kv_images(stage, cache, torch.arange(size)).flatten()):
            raise ValueError(f"{stage.name}: the host cache layout differs from the generated pack")
        self.totals["kv_layout_check_ms"] += (time.perf_counter_ns() - start) / 1e6

    def _push_kv(self, stage):
        """Write every queued position into the host image, then send the cache range once.

        The range [kv_base, capacity) is contiguous and at the same offset on every DPU, so
        one transfer covers every layer. The caller queues positions in bursts (the prefill
        positions, then every few decode steps), so this runs only for those steps.
        stage is the one about to run; any stage reaches every DPU.
        """
        import torch

        memory = self.memory
        if self.kv_image is None:
            self.kv_image = torch.zeros((len(stage.physical_mapping), memory.kv_bytes), dtype=torch.int8)
        for pending, writes in self.kv_pending.items():
            binding, = [binding for binding in pending.bindings if binding.role == KV]
            # 이 stage의 cache slot이 image 안에서 시작하는 byte
            base = pending.addresses[binding.symbol] - MRAM_BASE - memory.kv_base
            for cache, start, stop in writes:
                if not pending.kv_checked:
                    self._check_kv_layout(pending, cache)
                    pending.kv_checked = True
                slot_bytes = self._kv_bytes(pending, start, stop)
                self.kv_image[:, base + slot_bytes] = self._kv_images(pending, cache, slot_bytes)[pending.kv_jobs]
        self.kv_pending.clear()
        self.library.upmem_model_kv_push_region(stage.handle, MRAM_BASE + memory.kv_base,
                                                self.kv_image.data_ptr(), memory.kv_bytes)

    def enter_layer(self, plan):
        # group이 바뀔 때만 그 group 전체 weight를 한 번에 보낸다 (pack은 대개 앞 group 실행 중에 끝나 있다).
        self.loader.enter((plan.phase, self.memory.phases[plan.phase].group_of_layer[plan.index]))

    def exit_layer(self, plan):
        """다음 group이 같은 weight 영역을 덮어쓴다. 이번 요청에서 다시 돌지 않는 layer의 host pool을 놓는다."""
        for stage in plan.stages:
            self.library.upmem_model_pools_release(stage.handle)

    def weight_table(self, plans):
        table = {}
        for plan in plans:
            for stage in plan.stages:
                for binding in stage.bindings:
                    if binding.role == WEIGHT:
                        table[(plan.phase, plan.index, stage.name, binding.operand)] = {
                            "tensor": stage.weights[binding.operand],
                            "address": stage.addresses[binding.symbol],
                            "bytes_per_unit": binding.size,
                            "mapping": stage.physical_mapping,
                        }
        return table

    def run(self, stage, matrices, output):
        times = self.stage_times.setdefault((stage.name, tuple(stage.shape)),
                                            {"calls": 0, "program_load_ms": 0.0, "invoke_ms": 0.0,
                                             "kv_push_ms": 0.0})
        if stage.kv is not None and self.kv_pending:
            # cache를 읽는 stage가 처음 돌기 전에, 모든 layer의 대기 위치를 cache 영역 전송 한 번으로 보낸다.
            start = time.perf_counter_ns()
            self._push_kv(stage)
            push_ms = (time.perf_counter_ns() - start) / 1e6
            self.totals["kv_push_ms"] += push_ms
            times["kv_push_ms"] += push_ms
        start = time.perf_counter_ns()
        self.library.upmem_model_pools_acquire(stage.handle)
        acquired = time.perf_counter_ns()
        self.load_program(stage)
        loaded = time.perf_counter_ns()
        self.invoke(stage, matrices, output)
        end = time.perf_counter_ns()
        load_ms = (loaded - acquired) / 1e6
        invoke_ms = (end - loaded + acquired - start) / 1e6
        self.totals["program_load_ms"] += load_ms
        self.totals["invoke_ms"] += invoke_ms
        times["calls"] += 1
        times["program_load_ms"] += load_ms
        times["invoke_ms"] += invoke_ms

    def stats(self):
        totals = {name: 0 for name, _ in TransferStats._fields_}
        for stage in self.stages:
            stats = TransferStats()
            self.library.upmem_model_stats(stage.handle, ctypes.byref(stats))
            for name in totals:
                totals[name] += getattr(stats, name)
        for name in totals:
            totals[name] += self.group_stats[name]
        totals.update(self.totals)
        # place 전에 준비가 실패해도 정리 경로가 통계를 읽는다. 그때는 group 업로드가 없다.
        if self.loader is not None:
            totals.update(self.loader.totals)
        return totals

    def close(self):
        # background pack이 stage pool을 쓰고 있을 수 있으므로 먼저 끝낸다.
        if self.loader is not None:
            self.loader.close()
        for stage in self.stages:
            self.library.upmem_model_close(stage.handle)
        self.stages.clear()
