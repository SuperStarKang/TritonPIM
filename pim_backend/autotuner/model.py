"""Numbers one PIM target supplies to the target-neutral autotuner.

Seed generation, constraints and features call only ``TargetModel``. Every
target-specific rule lives in one subclass below.
"""

from math import prod

from .targets import AiMXRuntimeConfig, PIMGemmProfileSpec, UPMEMRuntimeConfig
from .types import ceil_div


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
    """Precompile lower bound for GEMM residency and its MAC chain.

    Scratch, constants, copies and fused operations come from the compiled
    Common IR. The target's allocation report and capacity check are final.
    """
    runtime = target.runtime_config
    assert isinstance(runtime, AiMXRuntimeConfig)
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

    @property
    def gemm_profile_spec(self):
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
        # SingleUtil의 열·누산 크기를 출발점으로, 활성 tasklet마다 출력 한 행을 준다.
        return PIMGemmProfileSpec(
            (block_rows, 32, 16), "int8", "int32", (group_rows, 1, 1))

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

    @property
    def gemm_profile_spec(self):
        runtime = self.target.runtime_config
        assert isinstance(runtime, AiMXRuntimeConfig)
        if runtime.batch not in (1, 2, 4, 8):
            raise ValueError("AiMX batch must be 1, 2, 4, or 8")
        return PIMGemmProfileSpec(
            (runtime.batch, 4096 // runtime.batch, 256 // runtime.batch),
            "float16", "float32", (1, 1, 1))

    def problem_reason(self, problem):
        if problem.compute_kind != "matmul":
            return "현재 AiMX lowering은 matmul만 지원한다"
        if any(access.dtype not in ("bfloat16", "float16", "float32")
               for access in problem.accesses):
            return "AiMX host buffer는 bfloat16/float16/float32여야 한다"
        return None

    def mapping_reason(self, problem, partitions):
        if partitions[2] != 1:
            return "AiMX cross-device combine은 미구현이므로 p_k=1이어야 한다"
        return None

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
        return tuple(allocations[names[access.axes]] for access in problem.accesses)

    def overhead_allocations(self, problem, config):
        allocations = aimx_runtime_allocations(problem, self.target, config)
        return {
            name: allocations[name]
            for name in ("PMM", "Arel", "descriptor chain")
        }

    def scheduled_compute(self, problem, config):
        batch = self.target.runtime_config.batch
        gm, gn, gk = group_extent(config)
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
