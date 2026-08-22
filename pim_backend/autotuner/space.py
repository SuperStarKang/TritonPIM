# Stage 1 / 1-A / 1-B / 2 — PIM Config Space와 initial candidate 생성
# (design/AutoTuner_Design.md §4~§7)
#
# tuner는 BM/BN/BK, GM/GN/GK, EM/EN/EK만 탐색하고 TM/TN/TK는 자동으로
# 계산된다 (§5). 강하게 연관된 parameter는 semantic fragment로 묶는다 (§6):
#   KSplitFragment       (P_K, reduction_policy) 쌍
#   DistributionFragment 배치 전략 → Planner가 P_M/P_N 계산 (§6.1)

from dataclasses import dataclass

from .constraints import gemm_footprint_bytes
from .planner import resolve_distribution
from .types import PIMConfig, ceil_div

# Phase 1 knob 값 목록 (§53). power-of-two라 log2 feature 표현과 맞는다 (§27).
# 값 목록은 탐색 격자일 뿐이고, shape·memory에 맞지 않는 조합은 Constraint
# Engine(§8)이 제거한다.
BLOCK_VALUES = (16, 32, 64, 128)
GROUP_VALUES = (1, 2, 4, 8, 16, 32)
EXECUTION_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

# §6 KSplitFragment: PK와 reduction strategy를 무관하게 탐색하면 invalid
# configuration이 과도하게 생기므로 하나의 categorical knob으로 묶는다.
# 문서의 예시는 PK≤8이지만, UPMEM 실측 최적점(split_k.py, SPLIT_K=16)이
# 탐색 범위 안에 있어야 하므로 16까지 연다.
KSPLIT_STRATEGIES = (
    (1, "none"),
    (2, "device"),
    (4, "device"),
    (4, "host"),
    (8, "device"),
    (8, "tree"),
    (16, "device"),
    (16, "tree"),
)

# §6.1 DistributionFragment. K축 분할은 KSplitFragment가 가지므로
# 배치 전략은 남은 budget을 M/N 축에 나누는 방법을 고른다.
DISTRIBUTION_POLICIES = ("M-major", "N-major", "MN-balanced")

# categorical encoding 순서 (§27).
REDUCTION_POLICIES = ("none", "device", "host", "tree")

# Phase 2 (§53) — residency / replication / streaming.
# resident: operand를 device memory에 상주시켜 steady-state H2D에서 제거.
# replication slice: 각 device가 자기 조각만 상주 / full: 전체 사본 상주.
# streaming per_round: round마다 window 하나 적재 / all_rounds: 자기 몫 전부 적재.
RESIDENCY_POLICIES = ("stream", "resident")
REPLICATION_POLICIES = ("slice", "full")
STREAMING_POLICIES = ("per_round", "all_rounds")


@dataclass(frozen=True)
class PIMConfigSpace:
    block_values: tuple = BLOCK_VALUES
    group_values: tuple = GROUP_VALUES
    execution_values: tuple = EXECUTION_VALUES
    ksplit_strategies: tuple = KSPLIT_STRATEGIES
    distribution_policies: tuple = DISTRIBUTION_POLICIES
    residency_policies: tuple = RESIDENCY_POLICIES
    replication_policies: tuple = REPLICATION_POLICIES
    streaming_policies: tuple = STREAMING_POLICIES


def build_config_space(problem, hardware):
    """§52 build_config_space. Phase 1은 고정 knob 목록을 쓴다 (§53).

    shape/hardware에 따라 불가능한 값은 여기서 자르지 않고 Constraint
    Engine(§8)이 제거한다.
    """
    return PIMConfigSpace()


def make_config(problem, hardware, blocks, groups, executions,
                distribution_policy, ksplit_strategy,
                a_residency="stream", b_residency="stream",
                replication_policy="slice", streaming_policy="per_round"):
    """knob 값들로부터 완전한 PIMConfig를 만든다.

    P_K/reduction은 KSplitFragment에서 (§6), P_M/P_N은 distribution
    policy와 tile grid에서 결정된다 (§6.1). residency/replication/streaming은
    Phase 2 knob이다 (§53).
    """
    p_k, reduction_policy = ksplit_strategy
    tile_m = blocks[0] * groups[0] * executions[0]
    tile_n = blocks[1] * groups[1] * executions[1]
    p_m, p_n = resolve_distribution(
        problem, hardware, tile_m, tile_n, p_k, distribution_policy)
    return PIMConfig(
        block_m=blocks[0], block_n=blocks[1], block_k=blocks[2],
        group_m=groups[0], group_n=groups[1], group_k=groups[2],
        execution_m=executions[0], execution_n=executions[1],
        execution_k=executions[2],
        distribution_policy=distribution_policy,
        p_m=p_m, p_n=p_n, p_k=p_k,
        reduction_policy=reduction_policy,
        a_residency=a_residency, b_residency=b_residency,
        replication_policy=replication_policy,
        streaming_policy=streaming_policy,
    )


def _derive_groups(problem, hardware, blocks, spec):
    """seed 규칙 1 — local memory 예산을 채울 때까지 group을 m→n→k 번갈아 2배.

    MLIR 공통 pass의 block group 자동 산출과 같은 철학이다. 예산은 profile의
    local_memory_bytes에서만 오므로 target-중립이다. 축 배분(block↔group 포함)은
    이후 search가 fragment mutation으로 조정한다.
    """
    dims = (problem.M, problem.N, problem.K)
    groups = [1, 1, 1]
    axis = 0
    stalled = 0
    while stalled < 3:
        trial = groups[axis] * 2
        size = [blocks[i] * groups[i] for i in range(3)]
        size[axis] = blocks[axis] * trial
        if (trial in spec.group_values
                and size[axis] <= dims[axis]
                and gemm_footprint_bytes(problem, *size)
                <= hardware.local_memory_bytes):
            groups[axis] = trial
            stalled = 0
        else:
            stalled += 1
        axis = (axis + 1) % 3
    return tuple(groups)


def _derive_executions(problem, hardware, blocks, groups, p_k, spec,
                       util_scale=1):
    """seed 규칙 2 — execution tile을 device의 최대 활용 단위로 묶는다.

    tile은 utilization_tile(한 device의 실행 자원을 가득 채우는 단위)까지만
    키운다. device memory 용량(§11)은 GEMM이 PIM DRAM보다 클 때만 상한으로
    작동한다: UPMEM처럼 memory가 작으면 용량 상한이 먼저 걸려 채우기가 되고,
    AiMX처럼 실행 단위·memory가 크면 활용 단위에서 멈춘다.
    """
    base = [blocks[i] * groups[i] for i in range(3)]
    util = (hardware.utilization_tile_m,
            hardware.utilization_tile_n,
            hardware.utilization_tile_k)
    # K축의 문제 크기는 spatial plane 하나의 구간이다 (§16).
    dims = (problem.M, problem.N, ceil_div(problem.K, p_k))
    executions = [1, 1, 1]
    for axis in range(3):
        target = min(util[axis] * util_scale, dims[axis])
        # 2배 단위 성장이 문제·K plane 크기를 넘지 않게 한다. 넘으면
        # Constraint 4(tile_k ≤ K plane)가 seed를 버리게 된다.
        while (executions[axis] * 2 in spec.execution_values
               and base[axis] * executions[axis] < target
               and base[axis] * executions[axis] * 2 <= dims[axis]):
            executions[axis] *= 2
    # 용량 상한 (§11): 넘으면 k→n→m 순으로 반감한다.
    axis = 2
    while (gemm_footprint_bytes(
            problem,
            base[0] * executions[0],
            base[1] * executions[1],
            base[2] * executions[2]) > hardware.device_memory_bytes
           and max(executions) > 1):
        if executions[axis] > 1:
            executions[axis] //= 2
        axis = (axis - 1) % 3
    return tuple(executions)


def heuristic_seeds(problem, hardware, spec):
    """PIM 의미를 갖는 initial seed (§7 Seed A~F).

    seed 값은 상수가 아니라 profile 예산에서 유도한다:
      block/group  ← local memory 예산 채우기 (_derive_groups)
      execution    ← device 최대 활용 단위로 묶기 (_derive_executions)
    seed는 전략 축(PK, distribution, tile 방향)만 서로 다르게 열고,
    어느 전략이 실제로 빠른지는 실측이 결정한다 (§31, §49).
    """
    blocks = (spec.block_values[0],) * 3
    groups = _derive_groups(problem, hardware, blocks, spec)

    def seed(policy, strategy, util_scale=1):
        executions = _derive_executions(
            problem, hardware, blocks, groups, strategy[0], spec, util_scale)
        return make_config(problem, hardware, blocks, groups, executions,
                           policy, strategy)

    seeds = [
        # Seed A — reduction 최소화: PK=1, N-major
        seed("N-major", (1, "none")),
        # Seed B — parallelism 최대화: PM/PN을 최대한 사용, PK=1
        seed("MN-balanced", (1, "none")),
        # Seed C — mild split-K: PK=2
        seed("N-major", (2, "device")),
        # Seed D — aggressive split-K: PK=4 / 8 / 16
        seed("N-major", (4, "device")),
        seed("N-major", (8, "tree")),
        seed("N-major", (16, "tree")),
        # Seed E — large execution tile: reuse ↑, round 수 ↓
        seed("MN-balanced", (1, "none"), util_scale=2),
        # Seed F — small execution tile: parallelism ↑, memory footprint ↓
        seed("MN-balanced", (1, "none"), util_scale=0.5),
    ]
    # Seed G (Phase 2, §53) — weight-stationary: B를 상주시켜 steady-state
    # 전송을 최소화한다. runtime이 residency를 지원하지 않아 space를
    # 제한한 경우에는 만들지 않는다.
    if "resident" in spec.residency_policies:
        executions = _derive_executions(
            problem, hardware, blocks, groups, 1, spec)
        seeds.append(make_config(
            problem, hardware, blocks, groups, executions,
            "N-major", (1, "none"), b_residency="resident"))
    return seeds


def random_valid_configs(problem, hardware, spec, count, rng, engine):
    """나머지 initial population을 채우는 random valid configuration (§7)."""
    configs = []
    attempts = 0
    max_attempts = count * 200
    while len(configs) < count and attempts < max_attempts:
        attempts += 1
        blocks = tuple(rng.choice(spec.block_values) for _ in range(3))
        groups = tuple(rng.choice(spec.group_values) for _ in range(3))
        executions = tuple(rng.choice(spec.execution_values) for _ in range(3))
        policy = rng.choice(spec.distribution_policies)
        strategy = rng.choice(spec.ksplit_strategies)
        config = make_config(
            problem, hardware, blocks, groups, executions, policy, strategy,
            a_residency=rng.choice(spec.residency_policies),
            b_residency=rng.choice(spec.residency_policies),
            replication_policy=rng.choice(spec.replication_policies),
            streaming_policy=rng.choice(spec.streaming_policies))
        if engine.valid(config):
            configs.append(config)
    return configs
