# Stage 5 — Plan Feature Extraction (design/AutoTuner_Design.md §20~§27)
#
# AutoTuner는 BM/BN/PK 같은 knob 값만 Random Forest에 넣지 않는다.
# Planner가 생성한 실제 execution plan으로부터 PIM-specific feature를
# 계산한다 (§20). utilization은 objective가 아니라 성능을 설명하는
# feature다 (§31).

import math

from .constraints import device_memory_footprint, local_memory_footprint
from .space import REDUCTION_POLICIES
from .types import PIMPlanFeatures


def extract_plan_features(problem, hardware, config, plan):
    e_m, e_n, e_k = plan.execution_grid
    p_m, p_n, p_k = plan.active_device_grid
    t_m, t_n, t_k = plan.execution_tile_shape
    active_units = p_m * p_n * p_k

    # Feature 1 — effective device utilization (§21):
    #   U_eff = sum(A_r) / (R * N_units)
    total_assigned = sum(len(items) for items in plan.device_assignments)
    effective_utilization = total_assigned / (
        plan.num_rounds * hardware.num_units)

    # Feature 2 — load imbalance (§22): I = max(W_d) / mean(W_d).
    # W_d는 전체 schedule에서 device d가 받은 (padding 포함) MAC 수다.
    per_item_mac = t_m * t_n * plan.k_windows_per_plane * t_k
    counts = [0] * active_units
    for items in plan.device_assignments:
        for device, _, _, _ in items:
            counts[device] += 1
    workloads = [c * per_item_mac for c in counts]
    mean_workload = sum(workloads) / active_units
    load_imbalance = max(workloads) / mean_workload

    # Feature 5 — padding (§25): R_padding = padded work / executed work.
    executed = total_assigned * per_item_mac
    useful = problem.M * problem.N * problem.K
    padding_ratio = (executed - useful) / executed

    # memory pressure (§27).
    local_memory_pressure = (
        local_memory_footprint(problem, config) / hardware.local_memory_bytes)
    device_memory_pressure = (
        device_memory_footprint(problem, config) / hardware.device_memory_bytes)

    transfer = plan.transfer_plan
    reduction = plan.reduction_plan
    descriptor = plan.descriptor_plan
    return PIMPlanFeatures(
        effective_utilization=effective_utilization,
        load_imbalance=load_imbalance,
        h2d_bytes=transfer.h2d_bytes,
        d2h_bytes=transfer.d2h_bytes,
        a_replication_bytes=transfer.a_replication_bytes,
        b_replication_bytes=transfer.b_replication_bytes,
        reduction_bytes=reduction.reduction_bytes,
        reduction_rounds=reduction.reduction_rounds,
        padding_ratio=padding_ratio,
        scheduling_rounds=plan.num_rounds,
        local_memory_pressure=local_memory_pressure,
        device_memory_pressure=device_memory_pressure,
        active_units=active_units,
        execution_tile_count=e_m * e_n * e_k,
        a_resident_bytes=transfer.a_resident_bytes,
        b_resident_bytes=transfer.b_resident_bytes,
        descriptor_build_count=descriptor.descriptor_build_count,
        descriptor_transfer_bytes=descriptor.descriptor_transfer_bytes,
        descriptor_reuse_factor=descriptor.descriptor_reuse_factor,
        command_build_count=descriptor.command_build_count,
        kernel_launch_count=descriptor.kernel_launch_count,
        device_alloc_count=descriptor.device_alloc_count,
    )


def feature_names(spec):
    """§27 feature vector의 이름 목록. vector와 같은 순서다."""
    names = [
        "log2_block_m", "log2_block_n", "log2_block_k",
        "log2_group_m", "log2_group_n", "log2_group_k",
        "log2_tile_m", "log2_tile_n", "log2_tile_k",
        "p_m", "p_n", "p_k",
    ]
    # 순서 없는 categorical 값은 숫자 크기에 의미가 생기지 않도록
    # one-hot encoding을 쓴다 (§27).
    names += [f"distribution={p}" for p in spec.distribution_policies]
    names += [f"reduction={p}" for p in REDUCTION_POLICIES]
    # Phase 2 knob (이진이라 열 하나씩이면 충분하다)
    names += ["a_residency", "b_residency",
              "replication=full", "streaming=all_rounds"]
    # Phase 3 knob
    names += ["device_set_reuse", "descriptor_reuse",
              "command_reuse", "persistent_execution"]
    names += [
        "active_units",
        "effective_utilization",
        "execution_tile_count",
        "scheduling_rounds",
        "h2d_bytes",
        "d2h_bytes",
        "a_replication_bytes",
        "b_replication_bytes",
        "reduction_bytes",
        "reduction_rounds",
        "padding_ratio",
        "load_imbalance",
        "local_memory_pressure",
        "device_memory_pressure",
        "a_resident_bytes",
        "b_resident_bytes",
        "descriptor_build_count",
        "descriptor_transfer_bytes",
        "descriptor_reuse_factor",
        "command_build_count",
        "kernel_launch_count",
        "device_alloc_count",
    ]
    return names


def config_plan_vector(config, features, spec):
    """LFBO에 전달할 feature vector: X = [Config, PlanFeature] (§27, §33)."""
    vector = [
        math.log2(config.block_m), math.log2(config.block_n),
        math.log2(config.block_k),
        math.log2(config.group_m), math.log2(config.group_n),
        math.log2(config.group_k),
        math.log2(config.tile_m), math.log2(config.tile_n),
        math.log2(config.tile_k),
        float(config.p_m), float(config.p_n), float(config.p_k),
    ]
    vector += [1.0 if config.distribution_policy == p else 0.0
               for p in spec.distribution_policies]
    vector += [1.0 if config.reduction_policy == p else 0.0
               for p in REDUCTION_POLICIES]
    vector += [
        1.0 if config.a_residency == "resident" else 0.0,
        1.0 if config.b_residency == "resident" else 0.0,
        1.0 if config.replication_policy == "full" else 0.0,
        1.0 if config.streaming_policy == "all_rounds" else 0.0,
    ]
    vector += [
        1.0 if config.device_set_reuse else 0.0,
        1.0 if config.descriptor_reuse else 0.0,
        1.0 if config.command_reuse else 0.0,
        1.0 if config.persistent_execution else 0.0,
    ]
    vector += [
        float(features.active_units),
        features.effective_utilization,
        float(features.execution_tile_count),
        float(features.scheduling_rounds),
        float(features.h2d_bytes),
        float(features.d2h_bytes),
        float(features.a_replication_bytes),
        float(features.b_replication_bytes),
        float(features.reduction_bytes),
        float(features.reduction_rounds),
        features.padding_ratio,
        features.load_imbalance,
        features.local_memory_pressure,
        features.device_memory_pressure,
        float(features.a_resident_bytes),
        float(features.b_resident_bytes),
        float(features.descriptor_build_count),
        float(features.descriptor_transfer_bytes),
        features.descriptor_reuse_factor,
        float(features.command_build_count),
        float(features.kernel_launch_count),
        float(features.device_alloc_count),
    ]
    return vector
