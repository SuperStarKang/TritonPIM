"""Static constraints for values represented by PIM Common IR."""

from .targets import AiMXRuntimeConfig
from .types import AXES, ceil_div


AIMX_PMM_BYTES = 0x30000
AIMX_RUNTIME_POOL_BYTES = 2 * 1024 * 1024
AIMX_MAX_SINGLE_ALLOCATION_BYTES = 1023 * AIMX_RUNTIME_POOL_BYTES


def _round_up(value, unit):
    return ceil_div(value, unit) * unit


def access_footprint_bytes(problem, extents):
    """Allocation footprint of all unique ``pim.place`` buffers."""
    return sum(
        access.elements(extents) * access.element_bytes
        for access in problem.accesses
    )


def gemm_footprint_bytes(problem, size_m, size_n, size_k):
    """Compatibility name for callers that still construct a GEMM problem."""
    return access_footprint_bytes(problem, (size_m, size_n, size_k))


def group_extent(config):
    return (
        config.block_m * config.group_m,
        config.block_n * config.group_n,
        config.block_k * config.group_k,
    )


def local_memory_footprint(problem, config):
    return access_footprint_bytes(problem, group_extent(config))


def unit_memory_footprint(problem, config):
    return access_footprint_bytes(problem, config.execution_extent)


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
        if min(problem.axis_extents) < 1:
            return "m/n/k axis extent는 모두 1 이상이어야 한다"
        if not problem.output_accesses:
            return "PIM problem에는 output access가 하나 이상 필요하다"
        for access in problem.accesses:
            if access.direction not in ("in", "out", "inout"):
                return f"지원하지 않는 PIM access direction: {access.direction}"
            if any(axis not in AXES for axis in access.axes):
                return f"지원하지 않는 PIM access axis: {access.axes}"
        if any(axis != "k" for axis in problem.reduction_axes):
            return "현재 PIM hierarchy의 reduction axis는 k여야 한다"
        if problem.K > 1 and not problem.has_reduction:
            return "reduction이 없는 연산은 k extent를 1로 두어야 한다"

        blocks = (config.block_m, config.block_n, config.block_k)
        groups = (config.group_m, config.group_n, config.group_k)
        tiles = (config.tile_m, config.tile_n, config.tile_k)
        partitions = (config.p_m, config.p_n, config.p_k)
        for axis, extent in enumerate(problem.axis_extents):
            if extent == 1 and (
                    blocks[axis] != 1 or groups[axis] != 1
                    or tiles[axis] != 1 or partitions[axis] != 1):
                return (
                    f"extent가 1인 {AXES[axis]} axis의 "
                    "Block/Group/Tile/partition은 모두 1이어야 한다")

        local_bytes = local_memory_footprint(problem, config)
        if (profile.local_capacity_limit_bytes > 0
                and local_bytes > profile.local_capacity_limit_bytes):
            return (
                f"Block Group local memory overflow: {local_bytes} > "
                f"{profile.local_capacity_limit_bytes}")

        unit_limit = (
            profile.banks_per_unit * profile.bank_capacity_limit_bytes)
        if (profile.target == "aimx"
                and problem.uses_aimx_matmul_primitive):
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
        active_units = p_m * p_n * p_k
        if active_units > profile.num_units:
            return (
                f"p_m*p_n*p_k({active_units})가 num_units"
                f"({profile.num_units})를 넘는다")

        return None

    def valid(self, config):
        return self.check(config) is None
