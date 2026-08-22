# Stage 3 — Constraint Engine (design/AutoTuner_Design.md §8~§13)
#
# 실행해 볼 가치가 없는 configuration을 compile 이전에 제거한다.
# 수학적으로 확실히 invalid인 configuration은 surrogate model이 배우게
# 할 필요 없이 여기서 제거한다 (§10). Static invalid는 LFBO training에도
# 넣지 않는다 (§35).

from .types import ceil_div


def gemm_footprint_bytes(problem, size_m, size_n, size_k):
    """(m,n,k) 작업 묶음이 상주하는 데 필요한 바이트: S_A + S_B + S_C."""
    in_b = problem.input_bytes
    out_b = problem.output_bytes
    return (size_m * size_k * in_b
            + size_k * size_n * in_b
            + size_m * size_n * out_b)


def local_memory_footprint(problem, config):
    """block group 하나를 실행하는 데 필요한 local memory (§10)."""
    return gemm_footprint_bytes(problem,
                                config.block_m * config.group_m,
                                config.block_n * config.group_n,
                                config.block_k * config.group_k)


def device_memory_footprint(problem, config):
    """PIM device 하나에 상주해야 하는 memory (§11, Phase 2 확장).

    S_tile + S_resident ≤ C_device 의 좌변. operand별로:
      resident + slice     자기 몫 전체 (자기 round들의 조각 × K plane)
      resident + full      operand 전체 사본
      stream + per_round   temporal window 하나 (T_M×T_K 또는 T_K×T_N)
      stream + all_rounds  자기 몫 전체를 호출마다 적재 (footprint는 slice와 동일)
    """
    t_m, t_n, t_k = config.tile_m, config.tile_n, config.tile_k
    in_b = problem.input_bytes
    out_b = problem.output_bytes
    e_m = ceil_div(problem.M, t_m)
    e_n = ceil_div(problem.N, t_n)
    rounds_m = ceil_div(e_m, config.p_m)
    rounds_n = ceil_div(e_n, config.p_n)
    k_plane = ceil_div(problem.K, config.p_k)
    k_plane_padded = ceil_div(k_plane, t_k) * t_k

    def operand_bytes(residency, window_bytes, share_bytes, full_bytes):
        if residency == "resident":
            if config.replication_policy == "full":
                return full_bytes
            return share_bytes
        if config.streaming_policy == "all_rounds":
            return share_bytes
        return window_bytes

    a_bytes = operand_bytes(
        config.a_residency,
        t_m * t_k * in_b,
        t_m * rounds_m * k_plane_padded * in_b,
        problem.M * problem.K * in_b)
    b_bytes = operand_bytes(
        config.b_residency,
        t_k * t_n * in_b,
        t_n * rounds_n * k_plane_padded * in_b,
        problem.K * problem.N * in_b)
    c_bytes = t_m * t_n * out_b
    return a_bytes + b_bytes + c_bytes


class ConstraintEngine:
    def __init__(self, problem, hardware):
        self.problem = problem
        self.hardware = hardware

    def check(self, config):
        """invalid이면 이유 문자열을, valid이면 None을 돌려준다."""
        problem = self.problem
        hardware = self.hardware

        # Constraint 1 — block/group 구조 (§9).
        # tile = block * group * execution 은 derived configuration(§5)이라
        # 항상 성립하므로 값의 양수 여부와 reduction legality(§13)를 본다.
        values = (config.block_m, config.block_n, config.block_k,
                  config.group_m, config.group_n, config.group_k,
                  config.execution_m, config.execution_n, config.execution_k,
                  config.p_m, config.p_n, config.p_k)
        if min(values) < 1:
            return "모든 크기와 partition 수는 1 이상이어야 한다"
        if (config.p_k == 1) != (config.reduction_policy == "none"):
            return (f"reduction legality 위반: P_K={config.p_k}, "
                    f"reduction={config.reduction_policy}")
        # Phase 2 legality: replication_policy는 resident operand에만 의미가
        # 있다 (§6의 무의미 조합 제거와 같은 논리).
        if (config.a_residency == "stream"
                and config.b_residency == "stream"
                and config.replication_policy != "slice"):
            return ("replication legality 위반: resident operand 없이 "
                    f"replication={config.replication_policy}")

        # Constraint 2 — local memory (§10): S_A+S_B+S_C+S_runtime ≤ C_local.
        s_local = (local_memory_footprint(problem, config)
                   + hardware.local_memory_reserved_bytes)
        if s_local > hardware.local_memory_bytes:
            return (f"local memory overflow: {s_local} > "
                    f"{hardware.local_memory_bytes}")

        # Constraint 3 — PIM device memory (§11).
        s_tile = device_memory_footprint(problem, config)
        if s_tile > hardware.device_memory_bytes:
            return (f"device memory overflow: {s_tile} > "
                    f"{hardware.device_memory_bytes}")

        # Constraint 4 — active device grid (§12).
        active = config.p_m * config.p_n * config.p_k
        if active > hardware.num_units:
            return (f"P_M*P_N*P_K({active})가 num_units"
                    f"({hardware.num_units})를 넘는다")
        e_m = ceil_div(problem.M, config.tile_m)
        e_n = ceil_div(problem.N, config.tile_n)
        e_k = ceil_div(problem.K, config.tile_k)
        if config.p_m > e_m:
            return f"P_M({config.p_m})이 tile grid M({e_m})보다 크다"
        if config.p_n > e_n:
            return f"P_N({config.p_n})이 tile grid N({e_n})보다 크다"
        if config.p_k > e_k:
            return f"P_K({config.p_k})가 tile grid K({e_k})보다 크다"
        # temporal window는 spatial K plane 하나 안에 들어가야 한다 (§16).
        k_plane = ceil_div(problem.K, config.p_k)
        if config.tile_k > k_plane:
            return (f"tile_k({config.tile_k})가 spatial K plane"
                    f"({k_plane})보다 크다")

        # Constraint 5 — target-specific (§13).
        in_b = problem.input_bytes
        out_b = problem.output_bytes
        align = hardware.alignment_bytes
        if (config.tile_k * in_b) % align != 0:
            return f"tile_k 전송 단위가 alignment({align})에 맞지 않는다"
        if (config.tile_n * in_b) % align != 0:
            return f"tile_n 전송 단위가 alignment({align})에 맞지 않는다"
        if (config.tile_n * out_b) % align != 0:
            return f"출력 tile_n 전송 단위가 alignment({align})에 맞지 않는다"
        # blocked split-K 커널의 전제: 각 K 몫이 BLOCK_K 격자에 맞아야 한다
        # (pim_backend/examples/split_k.py의 host assert와 같은 조건).
        if hardware.target_constraints.get("k_split_block_multiple"):
            if problem.K % (config.p_k * config.block_k) != 0:
                return (f"K({problem.K})가 P_K*BLOCK_K"
                        f"({config.p_k * config.block_k})의 배수가 아니다")

        return None

    def valid(self, config):
        return self.check(config) is None
