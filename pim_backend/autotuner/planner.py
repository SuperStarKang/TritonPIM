# Stage 4 — PIM Planner (design/AutoTuner_Design.md §14~§19)
#
# Constraint를 통과한 Config를 구체적인 실행 계획으로 변환한다.
# Planner는 AutoTuner가 아니다: "이 Config가 주어졌을 때 실제로 어떻게
# 실행해야 하는가?"만 결정한다 (§14).

from .types import PIMPlan, ReductionPlan, TransferPlan, ceil_div


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
        t_m, t_n, t_k = config.tile_m, config.tile_n, config.tile_k

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

        # §23 — transfer placement: 실제 전송량과 operand replication.
        # 한 (output tile, plane) 작업은 A의 T_M×(plane K 구간), B의
        # (plane K 구간)×T_N을 받는다. 전송은 padding된 크기로 계산한다 (§25).
        in_b = problem.input_bytes
        out_b = problem.output_bytes
        padded_k = k_windows * t_k
        items = e_m * e_n * p_k
        a_bytes = items * t_m * padded_k * in_b
        b_bytes = items * padded_k * t_n * in_b
        a_replication = a_bytes - M * K * in_b
        b_replication = b_bytes - K * N * in_b

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

        transfer = TransferPlan(
            a_bytes=a_bytes,
            b_bytes=b_bytes,
            a_replication_bytes=a_replication,
            b_replication_bytes=b_replication,
            h2d_bytes=a_bytes + b_bytes,
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
        )
