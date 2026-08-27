# Stage 4 — PIM Planner (design/AutoTuner_Design.md §14~§19)
#
# Constraint를 통과한 Config를 구체적인 실행 계획으로 변환한다.
# Planner는 AutoTuner가 아니다: "이 Config가 주어졌을 때 실제로 어떻게
# 실행해야 하는가?"만 결정한다 (§14).

from .types import (DescriptorPlan, PIMPlan, ReductionPlan, TransferPlan,
                    ceil_div)


def resolve_distribution(problem, hardware, tile_m, tile_n, p_k, policy):
    """distribution_policy로부터 P_M, P_N을 계산한다 (§6.1).

    AutoTuner는 숫자가 아니라 배치 전략 자체를 탐색하고, 실제 P_M/P_N은
    tile grid와 device 수를 보고 여기서 결정된다. P_K는 KSplitFragment가
    이미 정했으므로 (§6) 남은 budget = num_units // P_K 를 M/N 축에 나눈다.
    """
    e_m = ceil_div(problem.M, tile_m)
    e_n = ceil_div(problem.N, tile_n)
    budget = hardware.num_units // p_k
    if budget == 0:
        # P_K가 device 수를 넘는 경우. Constraint 4(§12)가 제거한다.
        return 0, 0
    if policy == "M-major":
        p_m = min(e_m, budget)
        p_n = min(e_n, budget // p_m)
    elif policy == "N-major":
        p_n = min(e_n, budget)
        p_m = min(e_m, budget // p_n)
    elif policy == "MN-balanced":
        p_m = min(e_m, max(1, int(budget ** 0.5)))
        p_n = min(e_n, budget // p_m)
        p_m = min(e_m, budget // p_n)
    else:
        raise ValueError(f"알 수 없는 distribution_policy: {policy}")
    return p_m, p_n


class PIMPlanner:
    def __init__(self, problem, hardware):
        self.problem = problem
        self.hardware = hardware

    def build(self, config):
        problem = self.problem
        M, N, K = problem.M, problem.N, problem.K
        t_m = config.capacity_limit_m
        t_n = config.capacity_limit_n
        t_k = config.capacity_limit_k

        # §15 — execution tile grid
        e_m = ceil_div(M, t_m)
        e_n = ceil_div(N, t_n)
        e_k = ceil_div(K, t_k)

        # §16 — active device grid
        p_m, p_n, p_k = config.p_m, config.p_n, config.p_k

        # K축: P_K개의 spatial plane이 K를 나눠 갖고, plane 안에서는
        # T_K temporal window를 순차 누산한다 (§16, §18).
        k_plane = ceil_div(K, p_k)
        k_windows = ceil_div(k_plane, t_k)

        # §17 — device scheduling: 같은 device set을 여러 round 재사용한다.
        rounds_m = ceil_div(e_m, p_m)
        rounds_n = ceil_div(e_n, p_n)
        num_rounds = rounds_m * rounds_n
        assignments = []
        # Phase 3: descriptor는 (device, round) 작업마다 하나다. 재사용이 켜져
        # 있으면 같은 active 모양의 round는 descriptor를 다시 만들지 않는다.
        descriptor_builds = 0
        built_shapes = set()
        for r_m in range(rounds_m):
            active_m = min(p_m, e_m - r_m * p_m)
            for r_n in range(rounds_n):
                active_n = min(p_n, e_n - r_n * p_n)
                round_items = []
                for i in range(active_m):
                    for j in range(active_n):
                        for plane in range(p_k):
                            device = (i * p_n + j) * p_k + plane
                            round_items.append(
                                (device, r_m * p_m + i, r_n * p_n + j, plane))
                assignments.append(round_items)
                if config.descriptor_reuse:
                    if (active_m, active_n) not in built_shapes:
                        built_shapes.add((active_m, active_n))
                        descriptor_builds += len(round_items)
                else:
                    descriptor_builds += len(round_items)

        # §23 — transfer placement: 실제 전송량과 operand replication.
        # 한 (output tile, plane) 작업은 A의 T_M×(plane K 구간), B의
        # (plane K 구간)×T_N을 받는다. 전송은 padding된 크기로 계산한다 (§25).
        # Phase 2: resident operand는 steady-state 전송(a_bytes/b_bytes)에서
        # 빠지고 1회 placement(a_resident_bytes/b_resident_bytes)로 잡힌다.
        in_b = problem.input_bytes
        out_b = problem.output_bytes
        padded_k = k_windows * t_k
        items = e_m * e_n * p_k

        if config.a_residency == "resident":
            a_bytes = 0
            if config.replication_policy == "full":
                a_resident_bytes = p_m * p_n * p_k * M * K * in_b
            else:
                a_resident_bytes = (p_n * p_k * (rounds_m * p_m)
                                    * t_m * padded_k * in_b)
        else:
            a_bytes = items * t_m * padded_k * in_b
            a_resident_bytes = 0

        if config.b_residency == "resident":
            b_bytes = 0
            if config.replication_policy == "full":
                b_resident_bytes = p_m * p_n * p_k * K * N * in_b
            else:
                b_resident_bytes = (p_m * p_k * (rounds_n * p_n)
                                    * t_n * padded_k * in_b)
        else:
            b_bytes = items * padded_k * t_n * in_b
            b_resident_bytes = 0

        a_replication = (a_bytes + a_resident_bytes) - M * K * in_b
        b_replication = (b_bytes + b_resident_bytes) - K * N * in_b

        # §18 — reduction plan: partial output ownership / location /
        # ordering / rounds / data movement.
        tile_out = t_m * t_n * out_b
        policy = config.reduction_policy
        if policy == "none":
            partial_outputs = 0
            reduction_bytes = 0
            reduction_rounds = 0
            d2h_bytes = e_m * e_n * tile_out
        elif policy == "host":
            # 모든 partial을 host로 보내 host가 합친다.
            partial_outputs = items
            reduction_bytes = items * tile_out
            reduction_rounds = 1
            d2h_bytes = items * tile_out
        elif policy == "device":
            # plane 0 device가 나머지 partial을 순차 누산한다.
            partial_outputs = items
            reduction_bytes = e_m * e_n * (p_k - 1) * tile_out
            reduction_rounds = p_k - 1
            d2h_bytes = e_m * e_n * tile_out
        elif policy == "tree":
            # 이진 tree로 합친다. 이동 총량은 device와 같고 round 수가 log2다.
            partial_outputs = items
            reduction_bytes = e_m * e_n * (p_k - 1) * tile_out
            reduction_rounds = (p_k - 1).bit_length()
            d2h_bytes = e_m * e_n * tile_out
        else:
            raise ValueError(f"알 수 없는 reduction_policy: {policy}")

        # Phase 3 — descriptor/command/launch 계획 (§53).
        descriptor_transfer_bytes = (descriptor_builds
                                     * self.hardware.descriptor_bytes)
        descriptor = DescriptorPlan(
            descriptor_build_count=descriptor_builds,
            descriptor_transfer_bytes=descriptor_transfer_bytes,
            descriptor_reuse_factor=items / descriptor_builds,
            command_build_count=1 if config.command_reuse else num_rounds,
            kernel_launch_count=1 if config.persistent_execution
            else num_rounds,
            device_alloc_count=1 if config.device_set_reuse else num_rounds,
        )

        transfer = TransferPlan(
            a_bytes=a_bytes,
            b_bytes=b_bytes,
            a_replication_bytes=a_replication,
            b_replication_bytes=b_replication,
            a_resident_bytes=a_resident_bytes,
            b_resident_bytes=b_resident_bytes,
            # §23: B_H2D = Σ(B_A + B_B + B_metadata)
            h2d_bytes=a_bytes + b_bytes + descriptor_transfer_bytes,
            d2h_bytes=d2h_bytes,
        )
        reduction = ReductionPlan(
            policy=policy,
            owner_plane=0,
            ordering="plane_ascending",
            partial_outputs=partial_outputs,
            reduction_bytes=reduction_bytes,
            reduction_rounds=reduction_rounds,
        )
        return PIMPlan(
            execution_tile_shape=(t_m, t_n, t_k),
            execution_grid=(e_m, e_n, e_k),
            active_device_grid=(p_m, p_n, p_k),
            num_rounds=num_rounds,
            k_windows_per_plane=k_windows,
            device_assignments=assignments,
            transfer_plan=transfer,
            reduction_plan=reduction,
            descriptor_plan=descriptor,
        )
