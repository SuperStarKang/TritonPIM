"""Search space over PIM Common IR hierarchy and mapping values."""

from dataclasses import dataclass
from math import prod

from .constraints import ConstraintEngine, group_extent
from .types import PIMConfig, ceil_div, pim_hierarchy


BLOCK_VALUES = (16, 32, 64, 128)
GROUP_VALUES = (1, 2, 4, 8, 16, 32)
TILE_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
INITIAL_ROUND_PATTERNS = (
    (1, 1, 1),
    (2, 1, 1), (1, 2, 1), (1, 1, 2),
    (4, 1, 1), (1, 4, 1), (1, 1, 4),
)


def _powers_of_two(limit):
    values = []
    value = 1
    while value <= limit:
        values.append(value)
        value *= 2
    if values[-1] != limit:
        values.append(limit)
    return tuple(values)


@dataclass(frozen=True)
class PIMConfigSpace:
    block_values: tuple = BLOCK_VALUES
    group_values: tuple = GROUP_VALUES
    tile_values: tuple = TILE_VALUES
    partition_values: tuple = (1,)
    initial_round_patterns: tuple = INITIAL_ROUND_PATTERNS
    min_initial_schedule_utilization: float = 0.5
    max_initial_padding_ratio: float = 0.5
    max_initial_scheduling_rounds: int = 4
    max_initial_reduction_contributions: int = 64


def axis_values(problem, spec, kind, axis):
    """Values for one hierarchy level, fixing unused logical axes to one."""
    if problem.axis_extents[axis] == 1:
        return (1,)
    values = {
        "block": spec.block_values,
        "group": spec.group_values,
        "tile": spec.tile_values,
        "partition": spec.partition_values,
    }
    return values[kind]


def build_config_space(problem, target, **overrides):
    fields = {
        "partition_values": _powers_of_two(
            target.common_profile.num_units),
    }
    fields.update(overrides)
    return PIMConfigSpace(**fields)


def make_config(blocks, groups, tiles, partitions):
    return PIMConfig(
        block_m=blocks[0], block_n=blocks[1], block_k=blocks[2],
        group_m=groups[0], group_n=groups[1], group_k=groups[2],
        tile_m=tiles[0], tile_n=tiles[1], tile_k=tiles[2],
        p_m=partitions[0], p_n=partitions[1], p_k=partitions[2],
    )


def _partition_seeds(problem, target, blocks, groups, spec):
    """A bounded set of direct [p_m,p_n,p_k] seeds.

    The first seed uses as many available units as the problem grid permits.
    Axis-oriented and later structured/LFBO candidates remain so measured
    performance, rather than this ordering, decides the final mapping.
    """
    profile = target.common_profile
    g_m, g_n, g_k = group_extent(
        make_config(blocks, groups, (1, 1, 1), (1, 1, 1)))
    grid = (
        ceil_div(problem.M, g_m),
        ceil_div(problem.N, g_n),
        ceil_div(problem.K, g_k),
    )
    candidates = []
    for p_m in axis_values(problem, spec, "partition", 0):
        if p_m > grid[0]:
            continue
        for p_n in axis_values(problem, spec, "partition", 1):
            if p_n > grid[1]:
                continue
            for p_k in axis_values(problem, spec, "partition", 2):
                active = p_m * p_n * p_k
                if active > profile.num_units or p_k > grid[2]:
                    continue
                candidates.append((p_m, p_n, p_k))

    if not candidates:
        raise ValueError("no valid [p_m,p_n,p_k] seed")

    # 같은 active-unit 수라면 p_k가 작은 것을 먼저 둔다. p_k가 커질수록 같은
    # reduction_k round에서 회수하고 결합할 output contribution 수가 늘어난다.
    # p_k가 큰 후보도 아래 K-oriented seed와 LFBO 후보에 남는다.
    full = min(
        candidates,
        key=lambda p: (-(p[0] * p[1] * p[2]), p[2]),
    )
    # (1,1,1)을 비교용으로 강제하지 않는다. 실제 parallel grid가 하나뿐이면
    # full 자체가 (1,1,1)이 되고, 그렇지 않은 큰 문제에서 single-unit 실행은
    # 초기 측정 전체를 막을 수 있다.
    seeds = [full]

    # Reduction은 각 p_k 단계를 K-only로 재지 않고, 그 p_k를 고정한 채
    # p_m/p_n을 채워 총 active unit을 최대화한다.
    # 그러면 비슷한 전체 parallelism에서 K 분할 효과를 비교할 수 있다.
    if problem.has_reduction:
        for p_k in axis_values(problem, spec, "partition", 2):
            with_p_k = [p for p in candidates if p[2] == p_k]
            if with_p_k:
                seeds.append(max(
                    with_p_k,
                    key=lambda p: p[0] * p[1] * p[2],
                ))

    full_active = prod(full)
    for axis in range(3):
        if problem.axis_extents[axis] == 1:
            continue
        along_axis = [
            p for p in candidates
            if all(p[i] == 1 for i in range(3) if i != axis)
            and prod(p) == full_active
        ]
        if along_axis:
            seeds.append(max(along_axis, key=lambda p: p[axis]))

    balanced = [p for p in candidates if p[0] == p[1]]
    if problem.M > 1 and problem.N > 1 and balanced:
        seeds.append(min(
            balanced,
            key=lambda p: (-(p[0] * p[1] * p[2]), p[2]),
        ))

    unique = []
    for seed in seeds:
        if seed not in unique:
            unique.append(seed)
    return unique


def _derive_tiles(problem, target, blocks, groups, partitions, spec):
    """Choose a capacity-safe initial Execution Tile in Block Group counts.

    A small problem stops at the number of groups assigned to one partition.
    A large problem stops when the target capacity rejects the next group.
    """
    base = tuple(blocks[i] * groups[i] for i in range(3))
    total_groups = tuple(
        ceil_div(problem.axis_extents[i], base[i]) for i in range(3))
    groups_per_partition = tuple(
        ceil_div(total_groups[i], partitions[i]) for i in range(3))
    engine = ConstraintEngine(problem, target)
    exact = make_config(
        blocks, groups, groups_per_partition, partitions)
    if engine.check(exact) is None:
        return groups_per_partition

    tiles = [1, 1, 1]
    progressed = True
    while progressed:
        progressed = False
        for axis in range(3):
            current_index = spec.tile_values.index(tiles[axis])
            if current_index + 1 == len(spec.tile_values):
                continue
            next_value = spec.tile_values[current_index + 1]
            if next_value > groups_per_partition[axis]:
                continue
            candidate_tiles = list(tiles)
            candidate_tiles[axis] = next_value
            candidate = make_config(
                blocks, groups, tuple(candidate_tiles), partitions)
            if engine.check(candidate) is None:
                tiles[axis] = next_value
                progressed = True
    return tuple(tiles)


def _safe_initial_schedule(problem, config, spec):
    hierarchy = pim_hierarchy(problem, config)
    active_units = config.p_m * config.p_n * config.p_k
    scheduled_tiles = hierarchy.scheduling_rounds * active_units
    utilization = hierarchy.useful_execution_tiles / scheduled_tiles
    if utilization < spec.min_initial_schedule_utilization:
        return False
    if hierarchy.scheduling_rounds > spec.max_initial_scheduling_rounds:
        return False

    scheduled_compute = scheduled_tiles * prod(config.execution_extent)
    useful_compute = prod(problem.axis_extents)
    padding_ratio = (
        scheduled_compute - useful_compute) / scheduled_compute
    if padding_ratio > spec.max_initial_padding_ratio:
        return False

    if problem.has_reduction:
        k_contributions = hierarchy.rounds[2] * config.p_k
        if k_contributions > spec.max_initial_reduction_contributions:
            return False
    return True


def heuristic_seeds(problem, target, spec):
    """Initial configs expressed only with Common IR values."""
    blocks = tuple(
        axis_values(problem, spec, "block", axis)[0]
        for axis in range(3)
    )
    groups = (1, 1, 1)
    configs = []
    engine = ConstraintEngine(problem, target)
    for partitions in _partition_seeds(
            problem, target, blocks, groups, spec):
        tiles = _derive_tiles(
            problem, target, blocks, groups, partitions, spec)
        config = make_config(blocks, groups, tiles, partitions)
        if (engine.valid(config)
                and _safe_initial_schedule(problem, config, spec)):
            configs.append(config)
    return configs


def structured_valid_configs(problem, target, spec, count, rng, engine):
    """Build initial exploration configs with coordinated P and Tile values.

    Blocks and groups still explore the global space. Partitions preserve the
    useful parallelism of the selected block-group grid, and tiles are derived
    from a small scheduling-round target instead of being sampled independently.
    """
    configs = []
    seen = set()
    attempts = 0
    max_attempts = count * 400
    while len(configs) < count and attempts < max_attempts:
        attempts += 1
        blocks = tuple(rng.choice(axis_values(
            problem, spec, "block", axis)) for axis in range(3))
        groups = tuple(rng.choice(axis_values(
            problem, spec, "group", axis)) for axis in range(3))
        partitions = rng.choice(
            _partition_seeds(problem, target, blocks, groups, spec))
        target_rounds = rng.choice(spec.initial_round_patterns)

        base = tuple(blocks[axis] * groups[axis] for axis in range(3))
        total_groups = tuple(
            ceil_div(problem.axis_extents[axis], base[axis])
            for axis in range(3))
        tiles = tuple(max(
            1,
            total_groups[axis]
            // (partitions[axis] * target_rounds[axis]),
        ) for axis in range(3))
        config = make_config(blocks, groups, tiles, partitions)
        if config.key() in seen or not engine.valid(config):
            continue

        if not _safe_initial_schedule(problem, config, spec):
            continue

        seen.add(config.key())
        configs.append(config)
    return configs
