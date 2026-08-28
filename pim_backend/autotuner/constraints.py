"""Static constraints for values represented by PIM Common IR."""

from .targets import AiMXRuntimeConfig
from .types import ceil_div


AIMX_PMM_BYTES = 0x30000
AIMX_RUNTIME_POOL_BYTES = 2 * 1024 * 1024
AIMX_MAX_SINGLE_ALLOCATION_BYTES = 1023 * AIMX_RUNTIME_POOL_BYTES


def _round_up(value, unit):
    return ceil_div(value, unit) * unit


def gemm_footprint_bytes(problem, size_m, size_n, size_k):
    return (
        size_m * size_k * problem.input_bytes
        + size_k * size_n * problem.input_bytes
        + size_m * size_n * problem.output_bytes
    )


def group_extent(config):
    return (
        config.block_m * config.group_m,
        config.block_n * config.group_n,
        config.block_k * config.group_k,
    )


def local_memory_footprint(problem, config):
    return gemm_footprint_bytes(problem, *group_extent(config))


def unit_memory_footprint(problem, config):
    return gemm_footprint_bytes(problem, *config.execution_extent)


def aimx_runtime_allocations(problem, target, config):
    """Allocation sizes produced by AiMX lowering for one Execution Tile."""
    runtime = target.runtime_config
    assert isinstance(runtime, AiMXRuntimeConfig)
    batch = runtime.batch
    useful_m, useful_n, useful_k = group_extent(config)
    physical_m = _round_up(useful_m, batch)
    physical_n = _round_up(useful_n, 4096 // batch)
    physical_k = _round_up(useful_k, 256 // batch)

    t_m = config.tile_m * physical_m
    t_n = config.tile_n * physical_n
    t_k = _round_up(config.tile_k * physical_k, 256)
    padded_k = _round_up(t_k, 4096)
    padded_n = _round_up(t_n, 4096)

    rows = t_m // batch
    cols = padded_n * batch // 4096
    steps = t_k * batch // 256
    descriptor_groups = (
        2 * (t_k // 256) * rows
        + rows * cols * (2 * steps + 1)
    )
    input_bytes = problem.input_bytes
    output_bytes = problem.output_bytes
    return {
        "PMM": AIMX_PMM_BYTES,
        "A": t_m * padded_k * input_bytes,
        "Arel": t_m * t_k * input_bytes * 16,
        "B": t_k * padded_n * input_bytes,
        "C": t_m * padded_n * output_bytes,
        "descriptor chain": descriptor_groups * 512,
    }


class ConstraintEngine:
    def __init__(self, problem, target):
        self.problem = problem
        self.target = target

    def check(self, config):
        problem = self.problem
        profile = self.target.common_profile
        values = config.key()
        if min(values) < 1:
            return "Block, Group, Tile, partition 값은 모두 1 이상이어야 한다"

        local_bytes = local_memory_footprint(problem, config)
        if (profile.local_capacity_limit_bytes > 0
                and local_bytes > profile.local_capacity_limit_bytes):
            return (
                f"Block Group local memory overflow: {local_bytes} > "
                f"{profile.local_capacity_limit_bytes}")

        unit_limit = (
            profile.banks_per_unit * profile.bank_capacity_limit_bytes)
        if profile.target == "aimx":
            allocations = aimx_runtime_allocations(problem, self.target, config)
            for name, size in allocations.items():
                if size > AIMX_MAX_SINGLE_ALLOCATION_BYTES:
                    return (
                        f"AiMX {name} allocation overflow: {size} > "
                        f"{AIMX_MAX_SINGLE_ALLOCATION_BYTES}")
            unit_bytes = sum(allocations.values())
            per_bank = sum(
                ceil_div(size, 8192) * 32
                for size in allocations.values()
            )
            if per_bank > profile.bank_capacity_limit_bytes:
                return (
                    f"AiMX bank footprint overflow: {per_bank} > "
                    f"{profile.bank_capacity_limit_bytes}")
        else:
            unit_bytes = unit_memory_footprint(problem, config)
        if unit_bytes > unit_limit:
            return f"PIM unit memory overflow: {unit_bytes} > {unit_limit}"

        p_m, p_n, p_k = config.p_m, config.p_n, config.p_k
        split_k_granularity = p_k * config.block_k
        if problem.K % split_k_granularity:
            return (
                f"K({problem.K}) must be divisible by "
                f"p_k*block_k({split_k_granularity})")
        active_units = p_m * p_n * p_k
        if active_units > profile.num_units:
            return (
                f"p_m*p_n*p_k({active_units})가 num_units"
                f"({profile.num_units})를 넘는다")

        t_m, t_n, t_k = config.execution_extent
        e_m = ceil_div(problem.M, t_m)
        e_n = ceil_div(problem.N, t_n)
        e_k = ceil_div(problem.K, t_k)
        if p_m > e_m:
            return f"p_m({p_m})이 Execution Tile grid M({e_m})보다 크다"
        if p_n > e_n:
            return f"p_n({p_n})이 Execution Tile grid N({e_n})보다 크다"
        if p_k > e_k:
            return f"p_k({p_k})가 Execution Tile grid K({e_k})보다 크다"

        k_plane = ceil_div(problem.K, p_k)
        if t_k > k_plane:
            return (
                f"Execution Tile K extent({t_k})가 K plane"
                f"({k_plane})보다 크다")
        return None

    def valid(self, config):
        return self.check(config) is None
