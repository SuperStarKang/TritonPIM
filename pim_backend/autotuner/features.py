"""Derived search features calculated from PIM Common IR values."""

from dataclasses import dataclass
import math

from .constraints import local_memory_footprint, unit_memory_footprint
from .types import ceil_div


@dataclass(frozen=True)
class PIMFeatures:
    active_units: int
    effective_utilization: float
    execution_tile_count: int
    scheduling_rounds: int
    h2d_bytes: int
    d2h_bytes: int
    a_replication_bytes: int
    b_replication_bytes: int
    reduction_bytes: int
    padding_ratio: float
    load_imbalance: float
    local_memory_pressure: float
    unit_memory_pressure: float


def extract_features(problem, target, config):
    profile = target.common_profile
    t_m, t_n, t_k = config.execution_extent
    e_m = ceil_div(problem.M, t_m)
    e_n = ceil_div(problem.N, t_n)
    p_m, p_n, p_k = config.p_m, config.p_n, config.p_k
    active_units = p_m * p_n * p_k
    rounds_m = ceil_div(e_m, p_m)
    rounds_n = ceil_div(e_n, p_n)
    scheduling_rounds = rounds_m * rounds_n
    assigned = e_m * e_n * p_k
    effective_utilization = assigned / (
        scheduling_rounds * profile.num_units)

    mean_assignments = assigned / active_units
    max_assignments = rounds_m * rounds_n
    load_imbalance = max_assignments / mean_assignments

    k_plane = ceil_div(problem.K, p_k)
    k_windows = ceil_div(k_plane, t_k)
    padded_k = k_windows * t_k
    executed = e_m * t_m * e_n * t_n * p_k * padded_k
    useful = problem.M * problem.N * problem.K
    padding_ratio = (executed - useful) / executed

    input_bytes = problem.input_bytes
    output_bytes = problem.output_bytes
    items = e_m * e_n * p_k
    raw_a_bytes = items * t_m * padded_k * input_bytes
    raw_b_bytes = items * padded_k * t_n * input_bytes

    # replicated_over=["n"]인 A는 같은 scheduling round의 p_n device들이,
    # replicated_over=["m"]인 B는 같은 round의 p_m device들이 공유한다.
    # round가 바뀌면 stage/to_mram이 다시 실행되므로 서로 다른 round까지 한 번의
    # 전송으로 세면 안 된다.
    a_bytes = e_m * rounds_n * p_k * t_m * padded_k * input_bytes
    b_bytes = rounds_m * e_n * p_k * padded_k * t_n * input_bytes
    output_bytes_per_tile = t_m * t_n * output_bytes
    d2h_bytes = items * output_bytes_per_tile
    final_output_bytes = e_m * e_n * output_bytes_per_tile
    reduction_bytes = max(0, d2h_bytes - final_output_bytes)

    local_limit = profile.local_capacity_limit_bytes
    unit_limit = profile.banks_per_unit * profile.bank_capacity_limit_bytes
    return PIMFeatures(
        active_units=active_units,
        effective_utilization=effective_utilization,
        execution_tile_count=e_m * e_n * p_k * k_windows,
        scheduling_rounds=scheduling_rounds,
        h2d_bytes=a_bytes + b_bytes,
        d2h_bytes=d2h_bytes,
        a_replication_bytes=max(0, raw_a_bytes - a_bytes),
        b_replication_bytes=max(0, raw_b_bytes - b_bytes),
        reduction_bytes=reduction_bytes,
        padding_ratio=padding_ratio,
        load_imbalance=load_imbalance,
        local_memory_pressure=(
            local_memory_footprint(problem, config) / local_limit
            if local_limit else 0.0),
        unit_memory_pressure=(
            unit_memory_footprint(problem, config) / unit_limit),
    )


def feature_names():
    return [
        "log2_block_m", "log2_block_n", "log2_block_k",
        "log2_group_m", "log2_group_n", "log2_group_k",
        "log2_tile_m", "log2_tile_n", "log2_tile_k",
        "p_m", "p_n", "p_k",
        "active_units", "effective_utilization",
        "execution_tile_count", "scheduling_rounds",
        "h2d_bytes", "d2h_bytes",
        "a_replication_bytes", "b_replication_bytes",
        "reduction_bytes", "padding_ratio", "load_imbalance",
        "local_memory_pressure", "unit_memory_pressure",
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
        float(features.active_units), features.effective_utilization,
        float(features.execution_tile_count),
        float(features.scheduling_rounds),
        float(features.h2d_bytes), float(features.d2h_bytes),
        float(features.a_replication_bytes),
        float(features.b_replication_bytes),
        float(features.reduction_bytes), features.padding_ratio,
        features.load_imbalance, features.local_memory_pressure,
        features.unit_memory_pressure,
    ]
