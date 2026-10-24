"""Numbers one PIM target supplies to the target-neutral autotuner.

Seed generation, constraints and features call only ``TargetModel``. Every
target-specific rule lives in one subclass below.
"""

from dataclasses import replace
from math import prod

from .targets import AiMXRuntimeConfig, PIMGemmProfileSpec, PIMGemvProfileSpec, UPMEMRuntimeConfig
from .types import TRITON_BLOCK, PIMConfig, ceil_div


AIMX_PMM_BYTES = 0x30000
AIMX_RUNTIME_POOL_BYTES = 2 * 1024 * 1024
AIMX_MAX_SINGLE_ALLOCATION_BYTES = 1023 * AIMX_RUNTIME_POOL_BYTES
# AiMX DRAM은 8 KiB를 256 bank에 32 B씩 interleave한다.
AIMX_INTERLEAVE_BYTES = 8192
AIMX_BANK_SLICE_BYTES = 32


def _round_up(value, unit):
    return ceil_div(value, unit) * unit


def group_extent(config):
    return (
        config.block_m * config.group_m,
        config.block_n * config.group_n,
        config.block_k * config.group_k,
    )


def access_footprint_bytes(problem, extents):
    """Logical bytes of all unique ``pim.place`` buffers over ``extents``."""
    return sum(
        access.elements(extents) * access.element_bytes
        for access in problem.accesses
    )


def aimx_runtime_allocations(problem, target, config):
    """Precompile lower bound for GEMM/GEMV residency and its MAC chain.

    Scratch, constants, copies and fused operations come from the compiled
    Common IR. The target's allocation report and capacity check are final.
    """
    runtime = target.runtime_config
    assert isinstance(runtime, AiMXRuntimeConfig)
    if problem.compute_kind == "matvec":
        # yᵀ = xᵀ·Wᵀ (batch 1): x는 A 한 행, Wᵀ는 B [K × M], y는 C 한 행이다.
        # 출력 lane 4096개가 M 방향이고 vecA 한 번이 K 256개다.
        useful_m, _, useful_k = group_extent(config)
        physical_m = _round_up(useful_m, 4096)
        physical_k = _round_up(useful_k, 256)
        steps = ceil_div(useful_k, 256)
        descriptor_groups = config.tile_m * (
            config.tile_k * 2 * (physical_k // 256)
            + (physical_m // 4096) * (2 * config.tile_k * steps + 1)
        )
        return {
            "PMM": AIMX_PMM_BYTES,
            # x의 K-group 행은 b=1의 A pitch 조건으로 4096 배수다.
            "A": config.tile_k * _round_up(useful_k, 4096) * 2,
            "Arel": config.tile_k * physical_k * 2 * 16,
            "B": config.tile_m * config.tile_k * physical_k * physical_m * 2,
            "C": config.tile_m * physical_m * 2,
            "descriptor chain": descriptor_groups * 512,
        }
    batch = runtime.batch
    useful_m, useful_n, useful_k = group_extent(config)
    physical_m = _round_up(useful_m, batch)
    # B/C Group panel rows are stored 4096 elements wide (aimx_matmul_row_alignment),
    # while MAC/drain covers only useful N in 4096/batch microtiles.
    physical_n = _round_up(useful_n, 4096)
    computed_n = _round_up(useful_n, 4096 // batch)
    physical_k = _round_up(useful_k, 256)
    # A의 K 저장 pitch (lowering의 matmul-k-alignment, 0이면 4096/batch).
    padded_k = _round_up(useful_k, runtime.matmul_k_alignment or 4096 // batch)

    rows = physical_m // batch
    cols = computed_n * batch // 4096
    steps = ceil_div(useful_k, 256 // batch)
    tile_steps = _round_up(config.tile_k * steps, batch)
    descriptor_groups = config.tile_m * config.tile_n * (
        config.tile_k * 2 * (physical_k // 256) * rows
        + rows * cols * (2 * tile_steps + 1)
    )
    # AiMX staging converts host values to BF16; host dtype is not DRAM dtype.
    input_bytes = output_bytes = 2
    return {
        "PMM": AIMX_PMM_BYTES,
        "A": config.tile_m * config.tile_k * physical_m * padded_k * input_bytes,
        "Arel": config.tile_k * physical_m * physical_k * input_bytes * 16,
        "B": config.tile_k * config.tile_n * physical_k * physical_n * input_bytes,
        "C": config.tile_m * config.tile_n * physical_m * physical_n * output_bytes,
        "descriptor chain": descriptor_groups * 512,
    }


class TargetModel:
    """Target-neutral defaults: logical footprint, no padding, any mapping."""

    # True면 activation input/output buffer를 모든 stage가 같은 영역으로 공유한다.
    # False면 stage마다 자기 buffer를 가져 state로 상주한다.
    shared_activation_scratch = False

    @property
    def launch_timing_name(self):
        raise NotImplementedError("target must name its synchronous device invocation timing")

    @property
    def gemm_profile_spec(self):
        raise NotImplementedError("target must provide its GEMM profiling shape and dtypes")

    @property
    def gemv_profile_spec(self):
        raise NotImplementedError("target must provide its GEMV profiling shape and dtypes")

    def __init__(self, target):
        self.target = target
        self.profile = target.common_profile

    @property
    def num_units(self):
        return self.profile.num_units

    @property
    def local_capacity(self):
        return self.profile.local_capacity_limit_bytes

    @property
    def bank_capacity(self):
        return self.profile.bank_capacity_limit_bytes

    @property
    def unit_capacity(self):
        return self.profile.banks_per_unit * self.profile.bank_capacity_limit_bytes

    def stage_target(self, problem):
        """Target spec one GEMM of ``problem`` is tuned and compiled with."""
        return self.target

    def block(self, problem):
        """Triton Block (m, n, k) of ``problem``."""
        return TRITON_BLOCK

    def problem_reason(self, problem):
        return None

    def group_reason(self, problem, extent):
        """Why one device cannot compute a Block Group of ``extent``."""
        return None

    def mapping_reason(self, problem, partitions):
        """Why ``[p_m,p_n,p_k]`` cannot be lowered."""
        return None

    def allocation_reason(self, problem, config):
        """Target allocation limits other than the unit capacity."""
        return None

    def resident_operand_reason(self, problem, config):
        """Why the B operand, kept resident (the key/value cache), would be stored with more
        padding than its own extent needs."""
        return None

    def resident_operand_config(self, problem):
        """A config whose resident B has no extra padding, one device and one round; None if
        the target has no such rule."""
        return None

    def access_allocations(self, problem, config):
        """Allocated bytes of each ``problem.accesses`` in one Execution Tile."""
        return tuple(
            access.elements(config.execution_extent) * access.element_bytes
            for access in problem.accesses
        )

    def overhead_allocations(self, problem, config):
        """Resident bytes of one Execution Tile that belong to no access."""
        return {}

    def scheduled_compute(self, problem, config):
        """Compute points one Execution Tile occupies, padding included."""
        return prod(config.execution_extent)

    def physical_pressure(self, plans):
        """Unit memory pressure of the allocation plans reported by lowering."""
        return max(sum(plan) for plan in plans) / self.unit_capacity


class UPMEMModel(TargetModel):
    launch_timing_name = "kernel"
    # 모든 stage가 MRAM의 같은 input/output 영역을 쓴다 (upmem_model.place).
    shared_activation_scratch = True

    def _profile_rows(self):
        runtime = self.target.runtime_config
        assert isinstance(runtime, UPMEMRuntimeConfig)
        group_rows = runtime.nr_tasklets or self.profile.lanes_per_core
        if group_rows < 1:
            raise ValueError("UPMEM profiling requires a positive active tasklet count")
        block_rows = 1
        # Triton Block은 2의 거듭제곱으로 두고 남는 tasklet 배수는 Group에 둔다.
        while group_rows % 2 == 0:
            block_rows *= 2
            group_rows //= 2
        return block_rows, group_rows

    @property
    def gemm_profile_spec(self):
        runtime = self.target.runtime_config
        assert isinstance(runtime, UPMEMRuntimeConfig)
        tasklets = runtime.nr_tasklets or self.profile.lanes_per_core
        # SingleUtil의 열·누산 크기(32, 16)를 출발점으로, 활성 tasklet마다 출력 한 행을 준다.
        return PIMGemmProfileSpec(
            TRITON_BLOCK, "int8", "int32",
            (tasklets // TRITON_BLOCK[0], 32 // TRITON_BLOCK[1], 16 // TRITON_BLOCK[2]))

    @property
    def gemv_profile_spec(self):
        block_rows, group_rows = self._profile_rows()
        # 기존 GEMV 커널의 누산 크기 64를 출발점으로 비교한다.
        return PIMGemvProfileSpec(
            (block_rows, 1, 64), "int32", "int32", (group_rows, 1, 1))

    def group_reason(self, problem, extent):
        runtime = self.target.runtime_config
        assert isinstance(runtime, UPMEMRuntimeConfig)
        lanes = runtime.nr_tasklets or self.profile.lanes_per_core
        # 모든 tasklet이 Block Group 하나의 출력 m 행을 똑같이 나눠 계산한다.
        if extent[0] % lanes:
            return (
                f"Block Group m extent({extent[0]})가 tasklet 수({lanes})로 "
                "나누어떨어지지 않는다")
        return None


class AiMXModel(TargetModel):
    launch_timing_name = "invoke"

    def _compute_unit(self, compute_kind):
        """Block Group extent one device computes without storage padding.

        GEMM: B/C panels store n 4096 wide, Arel/B store k in 256 and A in
        matmul_k_alignment. GEMV: Wᵀ stores m 4096 wide and k in 256 (batch 1).
        """
        runtime = self.target.runtime_config
        assert isinstance(runtime, AiMXRuntimeConfig)
        if compute_kind == "matvec":
            return (4096, 1, 256)
        if runtime.batch not in (1, 2, 4, 8):
            raise ValueError("AiMX batch must be 1, 2, 4, or 8")
        return (runtime.batch, 4096,
                max(256, runtime.matmul_k_alignment or 4096 // runtime.batch))

    def stage_target(self, problem):
        """GEMM batch = M 이상인 가장 작은 2의 거듭제곱, spec의 batch가 상한이다.

        한 B 행을 batch 행이 함께 읽으므로 M ≥ batch이면 batch가 클수록 B 재읽기가 준다.
        M < 8이면 batch를 M에 맞춰야 output block 행이 0으로 채워지지 않는다. GEMV는 batch 1 고정이다.
        """
        runtime = self.target.runtime_config
        if problem.compute_kind != "matmul":
            return self.target
        batch = min(runtime.batch, 1 << (problem.M - 1).bit_length())
        return replace(self.target, runtime_config=replace(runtime, batch=batch))

    def _matmul_block(self):
        # Block m이 batch보다 크면 M이 작은 GEMM도 Block m 행을 채워 output block 행이 0으로 남는다.
        return (min(TRITON_BLOCK[0], self.target.runtime_config.batch),) + TRITON_BLOCK[1:]

    def block(self, problem):
        if problem.compute_kind != "matmul":
            return TRITON_BLOCK
        return self._matmul_block()

    @property
    def gemm_profile_spec(self):
        unit = self._compute_unit("matmul")
        block = self._matmul_block()
        return PIMGemmProfileSpec(
            block, "float16", "float32",
            tuple(size // axis for size, axis in zip(unit, block)))

    @property
    def gemv_profile_spec(self):
        unit = self._compute_unit("matvec")
        block = (TRITON_BLOCK[0], 1, TRITON_BLOCK[2])
        return PIMGemvProfileSpec(
            block, "float16", "float32",
            tuple(size // axis for size, axis in zip(unit, block)))

    def group_reason(self, problem, extent):
        unit = self._compute_unit(problem.compute_kind)
        # Block Group은 연산 단위까지만 키운다. 그보다 큰 범위는 Execution Tile이 bank에 상주시킨다.
        if any(size > bound for size, bound in zip(extent, unit)):
            return (
                f"AiMX Block Group extent {extent}가 연산 단위 {unit}를 넘는다; "
                "Execution Tile을 키운다")
        return None

    def problem_reason(self, problem):
        if problem.compute_kind not in ("matmul", "matvec"):
            return "현재 AiMX lowering은 matmul과 matvec만 지원한다"
        if any(access.dtype not in ("bfloat16", "float16", "float32")
               for access in problem.accesses):
            return "AiMX host buffer는 bfloat16/float16/float32여야 한다"
        return None

    def resident_operand_reason(self, problem, config):
        if problem.compute_kind != "matmul":
            return None
        _, group_n, group_k = group_extent(config)
        # B Group panel은 행(K)을 256, 열(N)을 4096 원소로 올려 저장한다. Group 경계와 빈 Group마다
        # padding이 다시 생기면 상주 영역과 그 쓰기가 그만큼 커진다.
        rows = config.tile_k * _round_up(group_k, 256)
        cols = config.tile_n * _round_up(group_n, 4096)
        if (rows, cols) != (_round_up(problem.K, 256), _round_up(problem.N, 4096)):
            return (f"AiMX stores the resident B as {rows} x {cols}; it needs "
                    f"{_round_up(problem.K, 256)} x {_round_up(problem.N, 4096)}")
        return None

    def resident_operand_config(self, problem):
        # Group: m는 batch 행, n은 N(4096 이하면 전체, 넘으면 4096), k는 256. Tile이 문제 전체를 덮는다.
        block_m, block_n, block_k = self.block(problem)
        group_m = self.target.runtime_config.batch
        group_n = min(4096, _round_up(problem.N, block_n))
        return PIMConfig(block_m, block_n, block_k,
                         group_m // block_m, group_n // block_n, 256 // block_k,
                         ceil_div(problem.M, group_m), ceil_div(problem.N, group_n), ceil_div(problem.K, 256),
                         1, 1, 1)

    def allocation_reason(self, problem, config):
        allocations = aimx_runtime_allocations(problem, self.target, config)
        for name, size in allocations.items():
            if size > AIMX_MAX_SINGLE_ALLOCATION_BYTES:
                return (
                    f"AiMX {name} allocation overflow: {size} > "
                    f"{AIMX_MAX_SINGLE_ALLOCATION_BYTES}")
        per_bank = self._bank_bytes(allocations.values())
        if per_bank > self.bank_capacity:
            return (
                f"AiMX bank footprint overflow: {per_bank} > "
                f"{self.bank_capacity}")
        return None

    def access_allocations(self, problem, config):
        allocations = aimx_runtime_allocations(problem, self.target, config)
        names = {("m", "k"): "A", ("k", "n"): "B", ("m", "n"): "C"}
        if problem.compute_kind == "matvec":
            # matvec은 행렬 W(m,k)가 B, 벡터 x(k)가 A, 출력 y(m)가 C다.
            names = {("m", "k"): "B", ("k",): "A", ("m",): "C"}
        return tuple(allocations[names[access.axes]] for access in problem.accesses)

    def overhead_allocations(self, problem, config):
        allocations = aimx_runtime_allocations(problem, self.target, config)
        return {
            name: allocations[name]
            for name in ("PMM", "Arel", "descriptor chain")
        }

    def scheduled_compute(self, problem, config):
        gm, gn, gk = group_extent(config)
        if problem.compute_kind == "matvec":
            # 출력 microtile은 M 4096개, MAC step은 K 256개 단위다 (batch 1).
            return (config.tile_m * ceil_div(gm, 4096) * 4096
                    * config.tile_k * ceil_div(gk, 256) * 256)
        batch = self.target.runtime_config.batch
        mac_k = ceil_div(gk, 256 // batch) * (256 // batch)
        return config.tile_m * config.tile_n * (
            ceil_div(gm, batch) * batch
            * ceil_div(gn, 4096 // batch) * (4096 // batch)
            * ceil_div(config.tile_k * mac_k, 256) * 256
        )

    def physical_pressure(self, plans):
        return max(
            self._bank_bytes(plan) for plan in plans
        ) / self.bank_capacity

    @staticmethod
    def _bank_bytes(sizes):
        return sum(
            ceil_div(size, AIMX_INTERLEAVE_BYTES) * AIMX_BANK_SLICE_BYTES
            for size in sizes
        )


def target_model(target):
    models = {
        "upmem": UPMEMModel,
        "aimx": AiMXModel,
    }
    return models[target.common_profile.target](target)
