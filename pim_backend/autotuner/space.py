"""Search space and target-neutral seeds over the PIM Common IR hierarchy.

Seeds follow Triton Block -> Block Group -> Execution Tile -> Device
Partition. Each level reads only ``TargetModel`` numbers: the Block Group
saturates one device's compute, the Execution Tile fills one device's banks,
and the partition spreads Execution Tiles over devices.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import log2, prod

from .constraints import ConstraintEngine
from .features import extract_features
from .model import target_model
from .types import PIMConfig, pim_hierarchy


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
    # m/n/k 중 device로 나눌 수 있는 축. False인 축의 partition은 1뿐이다.
    partition_axes: tuple = (True, True, True)
    # seed와 tile ladder가 instance 하나로 채우려는 unit 수 (build_config_space가 정한다)
    instance_units: int = 0
    block_group_top_k: int = 3
    max_ladder_steps: int = 4
    initial_round_patterns: tuple = INITIAL_ROUND_PATTERNS
    min_initial_schedule_utilization: float = 0.5
    max_initial_padding_ratio: float = 0.5
    max_initial_scheduling_rounds: int = 4
    max_initial_reduction_contributions: int = 64


def axis_values(problem, spec, kind, axis):
    """Values for one hierarchy level, fixing unused logical axes to one."""
    if problem.axis_extents[axis] == 1:
        return (1,)
    if kind == "partition" and not spec.partition_axes[axis]:
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
        "instance_units": target.common_profile.num_units,
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


def _blocks(config):
    return (config.block_m, config.block_n, config.block_k)


def _groups(config):
    return (config.group_m, config.group_n, config.group_k)


def _tiles(config):
    return (config.tile_m, config.tile_n, config.tile_k)


def _partitions(config):
    return (config.p_m, config.p_n, config.p_k)


def legal_partitions(problem, target, spec):
    """Every ``[p_m,p_n,p_k]`` within ``spec.instance_units`` the target can lower."""
    model = target_model(target)
    return [
        partitions
        for partitions in product(*(
            axis_values(problem, spec, "partition", axis)
            for axis in range(3)))
        if prod(partitions) <= spec.instance_units
        and model.mapping_reason(problem, partitions) is None
    ]


def reachable_units(grid, partitions):
    """Most active units a grid of Execution Tiles can occupy."""
    return max(
        prod(p) for p in partitions
        if all(p[axis] <= grid[axis] for axis in range(3)))


def most_active(grid, partitions):
    """Partitions that fit ``grid`` with the most active units."""
    most = reachable_units(grid, partitions)
    return [
        p for p in partitions
        if prod(p) == most
        and all(p[axis] <= grid[axis] for axis in range(3))]


def _execution_tiles(problem, blocks, groups, tiles):
    return pim_hierarchy(
        problem, make_config(blocks, groups, tiles, (1, 1, 1))
    ).execution_tiles


def rank_block_groups(problem, target, spec):
    """(blocks, groups) ordered by how fully one device computes a Block Group.

    Keys, in order: compute efficiency over the whole problem, units the
    Block Group grid can still occupy, Block Group compute per allocated
    access byte, and the larger Block. One entry per Block Group extent.
    """
    model = target_model(target)
    engine = ConstraintEngine(problem, target)
    partitions = legal_partitions(problem, target, spec)
    useful_compute = prod(problem.axis_extents)
    # 같은 grid는 여러 (block, group)에서 반복되므로 한 번만 계산한다
    units_by_grid = {}
    best = {}
    for blocks in product(*(
            axis_values(problem, spec, "block", axis) for axis in range(3))):
        for groups in product(*(
                axis_values(problem, spec, "group", axis)
                for axis in range(3))):
            config = make_config(blocks, groups, (1, 1, 1), (1, 1, 1))
            if not engine.valid(config):
                continue
            hierarchy = pim_hierarchy(problem, config)
            grid = hierarchy.total_groups
            if grid not in units_by_grid:
                units_by_grid[grid] = reachable_units(grid, partitions)
            efficiency = Fraction(
                useful_compute,
                prod(grid) * model.scheduled_compute(problem, config))
            compute_per_byte = Fraction(
                prod(hierarchy.block_group_extent),
                sum(model.access_allocations(problem, config)))
            key = (efficiency, units_by_grid[grid], compute_per_byte,
                   prod(blocks))
            extent = hierarchy.block_group_extent
            if extent not in best or key > best[extent][0]:
                best[extent] = (key, blocks, groups)
    ranked = sorted(best.values(), key=lambda entry: entry[0], reverse=True)
    return [(blocks, groups) for _, blocks, groups in ranked]


def tile_ladder(problem, target, blocks, groups, spec):
    """Execution Tiles from the largest one device holds to the smallest
    whose grid lets every reachable unit work in one round.

    Returns [] when a single Block Group does not fit one device.
    """
    model = target_model(target)
    engine = ConstraintEngine(problem, target)
    partitions = legal_partitions(problem, target, spec)
    total_groups = _execution_tiles(problem, blocks, groups, (1, 1, 1))
    steps = [
        [value for value in spec.tile_values if value < total] + [total]
        for total in total_groups
    ]

    def compute_per_byte(config):
        resident = (
            sum(model.access_allocations(problem, config))
            + sum(model.overhead_allocations(problem, config).values()))
        return Fraction(prod(config.execution_extent), resident)

    def moved(tiles, offset, axes=range(3)):
        configs = []
        for axis in axes:
            index = steps[axis].index(tiles[axis]) + offset
            if not 0 <= index < len(steps[axis]):
                continue
            new = list(tiles)
            new[axis] = steps[axis][index]
            config = make_config(blocks, groups, tuple(new), (1, 1, 1))
            if engine.valid(config):
                configs.append(config)
        return configs

    tiles = (1, 1, 1)
    if not engine.valid(make_config(blocks, groups, tiles, (1, 1, 1))):
        return []
    # bank 용량이 허용하는 동안 compute/byte가 가장 좋은 축을 키운다
    while True:
        options = moved(tiles, 1)
        if not options:
            break
        tiles = _tiles(max(options, key=compute_per_byte))

    ladder = [tiles]
    target_units = reachable_units(total_groups, partitions)
    # 나눌 수 없는 축을 줄이면 unit은 늘지 않고 round만 는다
    split_axes = [axis for axis in range(3)
                  if axis_values(problem, spec, "partition", axis) != (1,)]
    # 모든 unit을 한 round에 채울 수 있을 때까지 compute/byte 손실이 가장 작은 축을 줄인다
    while reachable_units(
            _execution_tiles(problem, blocks, groups, tiles),
            partitions) < target_units:
        tiles = _tiles(max(moved(tiles, -1, split_axes), key=compute_per_byte))
        ladder.append(tiles)
    return ladder


def best_partitions(problem, target, blocks, groups, tiles, partitions):
    """At most two partitions with the most active units over the grid.

    Ties prefer fewer rounds, then fewer reduction bytes, then fewer staged
    bytes. ``partitions`` is ``legal_partitions(...)``.
    """
    grid = _execution_tiles(problem, blocks, groups, tiles)

    def cost(p):
        features = extract_features(
            problem, target, make_config(blocks, groups, tiles, p))
        return (features.scheduling_rounds, features.reduction_bytes,
                features.stage_bytes)

    return sorted(most_active(grid, partitions), key=cost)[:2]


def _sample(ladder, count):
    """At most ``count`` ladder points, both ends included."""
    if len(ladder) <= count:
        return ladder
    return [
        ladder[round(index * (len(ladder) - 1) / (count - 1))]
        for index in range(count)
    ]


def heuristic_seeds(problem, target, spec):
    """Frontier seeds: top Block Groups x Execution Tile ladder x partitions.

    Per Block Group the ladder runs from the most parallel point to the most
    work per device, so the first seed is the default without search.
    """
    partitions = legal_partitions(problem, target, spec)
    seeds = []
    for blocks, groups in rank_block_groups(
            problem, target, spec)[:spec.block_group_top_k]:
        ladder = _sample(
            tile_ladder(problem, target, blocks, groups, spec),
            spec.max_ladder_steps)
        for tiles in reversed(ladder):
            for p in best_partitions(
                    problem, target, blocks, groups, tiles, partitions):
                seeds.append(make_config(blocks, groups, tiles, p))
    return seeds


def _closest_step(problem, ladder, blocks, groups, config):
    """Ladder index whose Execution Tile count is closest to ``config``'s."""
    count = pim_hierarchy(problem, config).useful_execution_tiles
    return min(
        range(len(ladder)),
        key=lambda index: abs(log2(prod(_execution_tiles(
            problem, blocks, groups, ladder[index])) / count)))


def regroup(problem, target, config, blocks, groups, spec, partitions):
    """``config`` moved to a new Block Group at the same frontier position.

    The Execution Tile is the new ladder point with the closest Execution
    Tile count; the partition is kept when it still fits.
    """
    ladder = tile_ladder(problem, target, blocks, groups, spec)
    if not ladder:
        return None
    tiles = ladder[_closest_step(problem, ladder, blocks, groups, config)]
    grid = _execution_tiles(problem, blocks, groups, tiles)
    current = _partitions(config)
    if all(current[axis] <= grid[axis] for axis in range(3)):
        return make_config(blocks, groups, tiles, current)
    return make_config(blocks, groups, tiles, best_partitions(
        problem, target, blocks, groups, tiles, partitions)[0])


def ladder_neighbors(problem, target, config, spec, partitions):
    """Adjacent frontier points of ``config`` on its own Block Group ladder."""
    blocks = _blocks(config)
    groups = _groups(config)
    ladder = tile_ladder(problem, target, blocks, groups, spec)
    if not ladder:
        return []
    index = _closest_step(problem, ladder, blocks, groups, config)
    neighbors = []
    for step in (index - 1, index + 1):
        if not 0 <= step < len(ladder):
            continue
        for p in best_partitions(
                problem, target, blocks, groups, ladder[step], partitions):
            neighbors.append(make_config(blocks, groups, ladder[step], p))
    return neighbors


def _safe_initial_schedule(problem, target, config, spec):
    hierarchy = pim_hierarchy(problem, config)
    active_units = config.p_m * config.p_n * config.p_k
    scheduled_tiles = hierarchy.scheduling_rounds * active_units
    utilization = hierarchy.useful_execution_tiles / scheduled_tiles
    if utilization < spec.min_initial_schedule_utilization:
        return False
    if hierarchy.scheduling_rounds > spec.max_initial_scheduling_rounds:
        return False

    scheduled_compute = scheduled_tiles * target_model(
        target).scheduled_compute(problem, config)
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


def structured_valid_configs(problem, target, spec, count, rng, engine):
    """Build initial exploration configs with coordinated P and Tile values.

    Blocks and groups still explore the global space. Partitions take the
    most active units of the Block Group grid, and tiles are derived from a
    small scheduling-round target instead of being sampled independently.
    """
    partitions_all = legal_partitions(problem, target, spec)
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
        total_groups = _execution_tiles(problem, blocks, groups, (1, 1, 1))
        partitions = rng.choice(most_active(total_groups, partitions_all))
        target_rounds = rng.choice(spec.initial_round_patterns)

        tiles = tuple(max(
            1,
            total_groups[axis]
            // (partitions[axis] * target_rounds[axis]),
        ) for axis in range(3))
        config = make_config(blocks, groups, tiles, partitions)
        if config.key() in seen or not engine.valid(config):
            continue

        if not _safe_initial_schedule(problem, target, config, spec):
            continue

        seen.add(config.key())
        configs.append(config)
    return configs
