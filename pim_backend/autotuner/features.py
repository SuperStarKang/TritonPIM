"""Features derived from the three PIM Common IR hierarchy levels."""

from dataclasses import dataclass
import math

from .constraints import local_memory_footprint, unit_memory_footprint
from .types import AXES, pim_hierarchy


@dataclass(frozen=True)
class PIMFeatures:
    block_group_compute_per_byte: float
    execution_tile_compute_per_byte: float
    active_units: int
    useful_execution_tiles: int
    scheduled_execution_tiles: int
    schedule_utilization: float
    scheduling_rounds: int
    stage_bytes: int
    h2d_bytes: int
    d2h_bytes: int
    reduction_bytes: int
    padding_ratio: float
    local_memory_pressure: float
    unit_memory_pressure: float


def extract_features(problem, target, config):
    profile = target.common_profile
    hierarchy = pim_hierarchy(problem, config)
    partitions = (config.p_m, config.p_n, config.p_k)
    active_units = math.prod(partitions)
    scheduling_rounds = hierarchy.scheduling_rounds
    useful_tiles = hierarchy.useful_execution_tiles
    scheduled_tiles = scheduling_rounds * active_units

    block_group_bytes = local_memory_footprint(problem, config)
    execution_tile_bytes = unit_memory_footprint(problem, config)
    block_group_compute = math.prod(hierarchy.block_group_extent)
    execution_tile_compute = math.prod(config.execution_extent)

    # Host staging은 pim.place.shared_over 축의 같은 buffer를 공유한다.
    # H2D는 그 buffer를 공유하더라도 모든 active device의 MRAM으로 전송한다.
    stage_bytes = 0
    input_tile_bytes = 0
    for access in problem.input_accesses:
        tile_bytes = (
            access.elements(config.execution_extent) * access.element_bytes)
        distinct_per_round = math.prod(
            partitions[axis]
            for axis, name in enumerate(AXES)
            if name in access.axes)
        stage_bytes += scheduling_rounds * distinct_per_round * tile_bytes
        input_tile_bytes += tile_bytes
    h2d_bytes = scheduling_rounds * active_units * input_tile_bytes

    output_tile_bytes = sum(
        access.elements(config.execution_extent) * access.element_bytes
        for access in problem.output_accesses)
    d2h_bytes = scheduling_rounds * active_units * output_tile_bytes

    final_output_bytes = sum(
        access.elements(problem.axis_extents) * access.element_bytes
        for access in problem.output_accesses)
    k_contributions = (
        hierarchy.rounds[2] * config.p_k if problem.has_reduction else 1)
    reduction_bytes = final_output_bytes * (k_contributions - 1)

    scheduled_compute = scheduled_tiles * execution_tile_compute
    useful_compute = math.prod(problem.axis_extents)
    padding_ratio = (scheduled_compute - useful_compute) / scheduled_compute

    local_limit = profile.local_capacity_limit_bytes
    unit_limit = profile.banks_per_unit * profile.bank_capacity_limit_bytes
    return PIMFeatures(
        block_group_compute_per_byte=(
            block_group_compute / block_group_bytes),
        execution_tile_compute_per_byte=(
            execution_tile_compute / execution_tile_bytes),
        active_units=active_units,
        useful_execution_tiles=useful_tiles,
        scheduled_execution_tiles=scheduled_tiles,
        schedule_utilization=useful_tiles / scheduled_tiles,
        scheduling_rounds=scheduling_rounds,
        stage_bytes=stage_bytes,
        h2d_bytes=h2d_bytes,
        d2h_bytes=d2h_bytes,
        reduction_bytes=reduction_bytes,
        padding_ratio=padding_ratio,
        local_memory_pressure=(
            block_group_bytes / local_limit if local_limit else 0.0),
        unit_memory_pressure=execution_tile_bytes / unit_limit,
    )


def feature_names():
    return [
        "log2_block_m", "log2_block_n", "log2_block_k",
        "log2_group_m", "log2_group_n", "log2_group_k",
        "log2_tile_m", "log2_tile_n", "log2_tile_k",
        "p_m", "p_n", "p_k",
        "block_group_compute_per_byte",
        "execution_tile_compute_per_byte",
        "active_units",
        "useful_execution_tiles", "scheduled_execution_tiles",
        "schedule_utilization", "scheduling_rounds",
        "stage_bytes", "h2d_bytes", "d2h_bytes", "reduction_bytes",
        "padding_ratio", "local_memory_pressure", "unit_memory_pressure",
    ]


def config_feature_vector(config, features):
    return [
        math.log2(config.block_m), math.log2(config.block_n),
        math.log2(config.block_k),
        math.log2(config.group_m), math.log2(config.group_n),
        math.log2(config.group_k),
        math.log2(config.tile_m), math.log2(config.tile_n),
        math.log2(config.tile_k),
        float(config.p_m), float(config.p_n), float(config.p_k),
        features.block_group_compute_per_byte,
        features.execution_tile_compute_per_byte,
        float(features.active_units),
        float(features.useful_execution_tiles),
        float(features.scheduled_execution_tiles),
        features.schedule_utilization,
        float(features.scheduling_rounds),
        float(features.stage_bytes), float(features.h2d_bytes),
        float(features.d2h_bytes), float(features.reduction_bytes),
        features.padding_ratio, features.local_memory_pressure,
        features.unit_memory_pressure,
    ]
