"""Static constraints for values represented by PIM Common IR."""

from .model import access_footprint_bytes, group_extent, target_model
from .types import AXES, ceil_div


class ConstraintEngine:
    def __init__(self, problem, target, constraint=None):
        self.problem = problem
        self.target = target
        self.model = target_model(target)
        # constraint(problem, config) -> 거부 사유 문자열 또는 None. 정적 검사를 모두
        # 통과한 config에만 부른다 (예: residency policy가 정한 weight stage 조건).
        self.constraint = constraint

    def check(self, config):
        problem = self.problem
        model = self.model
        values = config.key()
        if min(values) < 1:
            return "Block, Group, Tile, partition 값은 모두 1 이상이어야 한다"
        if min(problem.axis_extents) < 1:
            return "m/n/k axis extent는 모두 1 이상이어야 한다"
        reason = model.problem_reason(problem)
        if reason is not None:
            return reason
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

        # 각 계층은 아래 계층 단위 수를 넘지 않는다: 완전히 빈 Block을 Group에 두지 않는다
        for axis in range(3):
            blocks_in_axis = ceil_div(problem.axis_extents[axis], blocks[axis])
            if groups[axis] > blocks_in_axis:
                return (
                    f"{AXES[axis]} group({groups[axis]})이 Block 수"
                    f"({blocks_in_axis})보다 크다")

        block_group_extent = group_extent(config)
        total_groups = tuple(
            ceil_div(problem.axis_extents[axis], block_group_extent[axis])
            for axis in range(3))
        for axis in range(3):
            if tiles[axis] > total_groups[axis]:
                return (
                    f"{AXES[axis]} tile({tiles[axis]})이 total groups"
                    f"({total_groups[axis]})보다 크다")

        execution_tiles = tuple(
            ceil_div(total_groups[axis], tiles[axis])
            for axis in range(3))
        for axis in range(3):
            if partitions[axis] > execution_tiles[axis]:
                return (
                    f"{AXES[axis]} partition({partitions[axis]})이 "
                    f"execution tiles({execution_tiles[axis]})보다 크다")

        # Block Group: device 하나의 연산 단위
        reason = model.group_reason(problem, block_group_extent)
        if reason is not None:
            return reason
        local_bytes = access_footprint_bytes(problem, block_group_extent)
        if model.local_capacity > 0 and local_bytes > model.local_capacity:
            return (
                f"Block Group local memory overflow: {local_bytes} > "
                f"{model.local_capacity}")

        # Execution Tile: device 하나의 bank 용량
        reason = model.allocation_reason(problem, config)
        if reason is not None:
            return reason
        unit_bytes = (
            sum(model.access_allocations(problem, config))
            + sum(model.overhead_allocations(problem, config).values()))
        if unit_bytes > model.unit_capacity:
            return (
                f"PIM unit memory overflow: {unit_bytes} > "
                f"{model.unit_capacity}")

        # Device Partition: 여러 device
        active_units = config.p_m * config.p_n * config.p_k
        if active_units > model.num_units:
            return (
                f"p_m*p_n*p_k({active_units})가 num_units"
                f"({model.num_units})를 넘는다")
        reason = model.mapping_reason(problem, partitions)
        if reason is not None or self.constraint is None:
            return reason
        return self.constraint(problem, config)

    def valid(self, config):
        return self.check(config) is None
