# 한 _int_mm을 PIM에서 실행하기 위한 problem / hardware / config / plan 묶음.
#
# 네 객체는 전부 autotuner/types.py의 것을 그대로 쓴다. 이 파일이 하는 일은
# "어느 config를 고를까"와 "고른 것을 template metadata로 어떻게 옮길까" 둘뿐이다.
#
#   PIMProblem          실행해야 하는 GEMM          ← lowering이 mm_args에서 만든다
#   PIMHardwareProfile  사용 가능한 PIM 자원         ← targets.py 등록표
#   PIMConfig           실행 계획 knob 전부          ← seed 또는 autotune cache
#   PIMPlan             config를 hardware에 배치한 결과 ← PIMPlanner
#
# [값이 한 군데에만 있게 하는 것이 목적이다]
# 예전에는 block/group/tile과 num_units/local_lanes와 축별 분할 수를 torch 쪽에서
# 따로 들고 있었다. 그러면 (a) tile = block*group*execution 관계를 손으로 검사해야
# 하고, (b) tile_m 하나에서 (block, group, execution)을 복원할 수 없어 config를
# ConstraintEngine이나 PIMPlanner에 다시 넣을 수 없고, (c) shape에 의존하는 조건
# (p_n <= ceil(N/tile_n))을 config 단독으로는 검사할 수 없다.
# (c)는 실제로 손해를 봤다 — attention의 P·V가 N=128인데 p_n=8을 받아 device
# 절반이 놀았고, 그것을 생성 코드를 읽고서야 알았다. ConstraintEngine의
# Constraint 4가 바로 그 조건이다.

from dataclasses import dataclass, replace

from ._autotuner import (
    ceil_div,
    ConstraintEngine,
    PIMConfig,
    PIMConfigCache,
    PIMHardwareProfile,
    PIMPlan,
    PIMPlanner,
    PIMProblem,
    build_config_space,
    heuristic_seeds,
    pim_launch_options,
    strategy_for_pk,
    target_profile,
)
from .config import PIMTorchConfig, UnsupportedPIMIntMM, validate_pim_options

# split-K template이 실행에 반영하지 못하는 knob은 값 하나로 제한한다.
# examples/autotune_gemm.py의 SPACE_OVERRIDES와 같은 이유다 — 커널이 residency나
# persistent execution을 구현하지 않으므로 그 축을 열어 두면 무의미한 config가
# 나온다.
SPACE_OVERRIDES = dict(
    residency_policies=("stream",),
    replication_policies=("slice",),
    streaming_policies=("per_round",),
    persistence_strategies=((True, False, False, False),),
)


@dataclass(frozen=True)
class PIMExecutionPlan:
    """template 하나를 만드는 데 필요한 것 전부."""

    problem: PIMProblem
    hardware: PIMHardwareProfile
    config: PIMConfig
    plan: PIMPlan
    torch_config: PIMTorchConfig

    def template_meta(self) -> dict:
        """template의 tl.constexpr / jinja 변수.

        SPLIT_K는 config.p_k다. 커널이 K를 가르는 수와 device K 분할 수는 같은
        값이다 — examples/autotune_gemm.py도 SPLIT_K=config.p_k로 넘긴다."""
        return {
            "BLOCK_M": self.config.block_m,
            "BLOCK_N": self.config.block_n,
            "BLOCK_K": self.config.block_k,
            "SPLIT_K": self.config.p_k,
            "EVEN_TILES": even_tiles(self.problem, self.config),
        }

    def backend_options(self) -> dict:
        """triton_meta에 실어 PIMBackend.parse_options까지 보낼 option dict.

        본체는 autotuner의 pim_launch_options다 — 손으로 띄우는 경로와 인덕터
        경로가 같은 plan에서 같은 문자열을 만들어야 두 경로가 갈라지지 않는다.
        그 helper가 다루지 않는 두 knob만 여기서 얹는다."""
        options = dict(pim_launch_options(self.config, self.plan, self.hardware))
        options.update(
            lane_mode=self.torch_config.lane_mode,
            host_threads=self.torch_config.host_threads,
        )
        return validate_pim_options(options)

    def describe(self) -> str:
        c, p = self.config, self.plan
        p_m, p_n, p_k = p.active_device_grid
        return (f"block=({c.block_m},{c.block_n},{c.block_k}) "
                f"group=({c.group_m},{c.group_n},{c.group_k}) "
                f"execution=({c.execution_m},{c.execution_n},{c.execution_k}) "
                f"tile=({c.tile_m},{c.tile_n},{c.tile_k}) "
                f"devices=({p_m},{p_n},{p_k})={p_m * p_n * p_k}/"
                f"{self.hardware.num_units} "
                f"rounds={p.num_rounds} "
                f"distribution={c.distribution_policy} "
                f"reduction={c.reduction_policy}")


def even_tiles(problem: PIMProblem, config: PIMConfig) -> bool:
    """M/N이 block에 딱 맞는가 = mask 없는 본문을 낼 수 있는가.

    K는 Constraint 5(k_split_block_multiple)와 아래 check_template_support가
    K % (p_k*block_k) == 0을 보장하므로 K축 mask도 항상 참이다."""
    return (problem.M % config.block_m == 0) and (problem.N % config.block_n == 0)


def check_template_support(problem: PIMProblem, config: PIMConfig,
                           torch_config: PIMTorchConfig) -> None:
    """split-K template 고유 조건. ConstraintEngine이 보지 않는 것만 본다.

    - EVEN_TILES: mask extent가 상수면 PIMPlaceDataPass의 isExtentCandidate가
      extent 복원에 실패한다. ConstraintEngine에는 없는 조건이다 — 커널이
      mask를 쓰는지 여부는 template이 정하기 때문이다.
    - K % (p_k*block_k): Constraint 5도 같은 것을 보지만 그쪽은
      target_constraints["k_split_block_multiple"]가 있는 target에서만 본다.
      이 template은 target과 무관하게 blocked 분할을 전제하므로 여기서도 본다.
    """
    divisor = config.p_k * config.block_k
    if problem.K % divisor:
        raise UnsupportedPIMIntMM(
            f"K={problem.K} must be divisible by "
            f"P_K*BLOCK_K={config.p_k}*{config.block_k}={divisor} — "
            f"kernel이 K를 P_K개의 연속 구간으로 가른다")
    if not even_tiles(problem, config) and not torch_config.allow_masked_tiles:
        raise UnsupportedPIMIntMM(
            f"M={problem.M} must be divisible by BLOCK_M={config.block_m} and "
            f"N={problem.N} by BLOCK_N={config.block_n}.\n"
            f"나누어떨어지지 않으면 template이 mask를 만들어야 하는데, mask의 "
            f"extent가 상수면 PIMPlaceDataPass가 extent를 복원하지 못한다. "
            f"제약이 풀렸는지 확인하려면 allow_masked_tiles=True로 둔다.")


def _seed_config(problem, hardware, torch_config) -> PIMConfig:
    """탐색 없이 쓰는 결정론적 config (단계 A).

    default_config()를 그대로 쓰지 않는 이유: 그쪽은 첫 유효 seed를 고르는데
    Seed A가 P_K=1(split-K 없음)이다. 이 template의 검증 대상은 split-K
    경로이므로 원하는 P_K의 seed를 고른다. seed를 만드는 규칙(block/group은
    local memory 예산에서, execution은 device 활용 단위에서)은 autotuner
    것을 그대로 쓴다."""
    spec = build_config_space(problem, hardware, **SPACE_OVERRIDES)
    if strategy_for_pk(spec, torch_config.split_k) is None:
        raise UnsupportedPIMIntMM(
            f"split_k={torch_config.split_k} is not in the tuning space. "
            f"available: {[pk for pk, _ in spec.ksplit_strategies]}")
    seeds = [c for c in heuristic_seeds(problem, hardware, spec)
             if c.p_k == torch_config.split_k]
    if not seeds:
        raise UnsupportedPIMIntMM(
            f"no heuristic seed with P_K={torch_config.split_k} for {problem}")
    return _fit_k_axis(problem, seeds[0])


def _fit_k_axis(problem: PIMProblem, config: PIMConfig) -> PIMConfig:
    """seed의 K축 execution tile을 이 shape의 K 몫 안으로 줄인다.

    heuristic_seeds의 execution 유도는 "device 활용 단위"에서 나오므로 K가 작은
    shape에서는 tile_k가 K 몫(§16의 spatial K plane)을 넘긴다. 예: K=128, P_K=4면
    plane이 32인데 seed는 tile_k=256을 낸다. Constraint 4가 이것을 거부한다.

    autotuner를 고치지 않고 torch 쪽에서 맞춘다. 줄이는 방향이라 local memory
    footprint는 작아지기만 하므로 Constraint 2를 새로 어길 수 없다. 그래도
    최종 판정은 ConstraintEngine이 한다."""
    k_plane = ceil_div(problem.K, config.p_k)
    while config.block_k * config.group_k * config.execution_k > k_plane:
        if config.execution_k > 1:
            config = replace(config, execution_k=config.execution_k // 2)
        elif config.group_k > 1:
            config = replace(config, group_k=config.group_k // 2)
        else:
            # block_k 하나가 이미 plane보다 크다. block은 template의 constexpr라
            # 여기서 바꾸지 않는다 — ConstraintEngine이 이유를 붙여 거부한다.
            break
    return config


def _cached_config(problem, hardware, torch_config) -> PIMConfig:
    """offline autotune cache의 best config (단계 B).

    lowering은 fake tensor를 다루므로 여기서 실측 autotuning을 돌리지 않는다 —
    잴 입력이 없다. cache miss는 fail-fast다."""
    cache = PIMConfigCache(torch_config.autotune_cache_path)
    found = cache.lookup_result(problem, hardware)
    if found is None:
        raise UnsupportedPIMIntMM(
            f"autotune cache miss for (M,N,K)=({problem.M},{problem.N},"
            f"{problem.K}) at {torch_config.autotune_cache_path}. "
            f"Tune this shape offline first (examples/autotune_gemm.py), "
            f"or set autotune_cache_path=None to use a seed config.")
    config, _result = found
    return config


def resolve_execution_plan(torch_config: PIMTorchConfig,
                           problem: PIMProblem) -> PIMExecutionPlan:
    """이 GEMM을 어떻게 실행할지 확정한다."""
    hardware = target_profile(torch_config.target)

    if torch_config.autotune_cache_path is not None:
        config = _cached_config(problem, hardware, torch_config)
    else:
        config = _seed_config(problem, hardware, torch_config)

    if torch_config.pin_row_partitions:
        # P_M=1은 직접 경로(examples/split_k.py)에서 확인된 형태다. N-major는
        # P_N을 먼저 최대로 잡고 남은 budget을 P_M에 주므로 num_units가 크면
        # P_M>1이 나오는데, 그 형태는 아직 검증하지 않았다. 출력 access가
        # rank-3(plane,row,col)로 승격되므로 구조상 문제는 없어 보인다.
        config = replace(config, p_m=1)

    reason = ConstraintEngine(problem, hardware).check(config)
    if reason is not None:
        raise UnsupportedPIMIntMM(
            f"config rejected for (M,N,K)=({problem.M},{problem.N},{problem.K}): "
            f"{reason}")
    check_template_support(problem, config, torch_config)

    plan = PIMPlanner(problem, hardware).build(config)
    return PIMExecutionPlan(problem=problem, hardware=hardware, config=config,
                            plan=plan, torch_config=torch_config)
