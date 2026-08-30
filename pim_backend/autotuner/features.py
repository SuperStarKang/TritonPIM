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
    input_reuse_bytes: int
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

    # 입력 transfer는 pim.place의 access dependency에서 계산한다. access가
    # 의존하는 축은 모든 Execution Tile을 방문하고, replicated_over 축은 같은
    # device scheduling round 안에서 재사용한다. 이 식은 GEMM A(m,k), B(k,n)
    # 뿐 아니라 GEMV, reduction, convolution과 fused placement에도 동일하다.
    raw_instances = e_m * e_n * p_k * k_windows
    raw_input_bytes = 0
    h2d_bytes = 0
    for access in problem.input_accesses:
        tile_bytes = access.elements((t_m, t_n, t_k)) \
            * access.element_bytes
        raw_input_bytes += raw_instances * tile_bytes
        staged_m = e_m if "m" in access.axes else rounds_m
        staged_n = e_n if "n" in access.axes else rounds_n
        staged_k = p_k * k_windows if "k" in access.axes else 1
        h2d_bytes += staged_m * staged_n * staged_k * tile_bytes

    partials = p_k if problem.has_reduction else 1
    d2h_bytes = 0
    final_output_bytes = 0
    for access in problem.output_accesses:
        tile_bytes = access.elements((t_m, t_n, t_k)) \
            * access.element_bytes
        d2h_bytes += e_m * e_n * partials * tile_bytes
        final_output_bytes += e_m * e_n * tile_bytes
    reduction_bytes = max(0, d2h_bytes - final_output_bytes)

    local_limit = profile.local_capacity_limit_bytes
    unit_limit = profile.banks_per_unit * profile.bank_capacity_limit_bytes
    return PIMFeatures(
        active_units=active_units,
        effective_utilization=effective_utilization,
        execution_tile_count=e_m * e_n * p_k * k_windows,
        scheduling_rounds=scheduling_rounds,
        h2d_bytes=h2d_bytes,
        d2h_bytes=d2h_bytes,
        input_reuse_bytes=max(0, raw_input_bytes - h2d_bytes),
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
        "input_reuse_bytes",
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
        float(features.input_reuse_bytes),
        float(features.reduction_bytes), features.padding_ratio,
        features.load_imbalance, features.local_memory_pressure,
        features.unit_memory_pressure,
    ]
