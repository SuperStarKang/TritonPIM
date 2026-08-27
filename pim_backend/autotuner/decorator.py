# @pim_autotune — problem(M,N,K)만으로 동작하는 사용면.
#
# 사용자는 config/plan을 받아 커널을 launch하는 함수를 쓰고, 데코레이터가
# 나머지를 담당한다:
#   - problem은 텐서 shape·dtype에서 자동 구성
#   - hardware는 target 등록표(targets.py)에서
#   - cache 정확 일치     → best config/plan 주입해 실행 (측정·검증 없음)
#   - cache miss          → 데코레이트된 함수를 benchmark로 tune()을 돌려
#                           best를 cache에 저장하고 best로 실행
#                           (search=False / PIM_AUTOTUNE=0 이면 탐색 대신
#                            seed 유도 config 하나로 실행)
#   - retune=True         → cache에 있어도 재탐색. 이미 잰 config는 저장값을
#     (PIM_RETUNE=1)        재사용하고 실측 예산은 새 후보에만 쓴다
#
# wall-clock 측정(§30)과 정답 검증(탐색 중에만)은 데코레이터가 수행하고,
# config→커널 인자 매핑은 함수 본문이 소유한다. backend 공통 launch 옵션은
# pim_launch_options helper가 조립한다.

import functools
import os
import statistics
import time

import torch

from .cache import PIMConfigCache
from .constraints import ConstraintEngine
from .planner import PIMPlanner
from .search import tune
from .space import build_config_space, heuristic_seeds
from .targets import target_profile
from .types import PIMBenchmarkResult, PIMProblem

DEFAULT_CACHE_PATH = os.path.expanduser("~/.pim_autotune_cache.json")

TORCH_DTYPE_NAMES = {
    torch.int8: "int8",
    torch.int32: "int32",
    torch.float16: "float16",
    torch.float32: "float32",
}
# GEMM 누산 dtype — 지원 조합만 나열한다.
OUTPUT_DTYPES = {"int8": "int32", "float16": "float32", "float32": "float32"}


def problem_from_tensors(a, b):
    """A[M,K] @ B[K,N] 텐서에서 PIMProblem을 구성한다."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"GEMM 형상 불일치: A[{M},{K}] @ B[{K2},{N}]"
    assert a.dtype == b.dtype, "A와 B의 dtype이 같아야 합니다"
    input_dtype = TORCH_DTYPE_NAMES[a.dtype]
    return PIMProblem(M=M, N=N, K=K, input_dtype=input_dtype,
                      output_dtype=OUTPUT_DTYPES[input_dtype])


def default_config(problem, hardware, spec):
    """cache miss이고 탐색을 안 할 때 쓰는 결정론적 config — 첫 유효 seed."""
    engine = ConstraintEngine(problem, hardware)
    for config in heuristic_seeds(problem, hardware, spec):
        if engine.valid(config):
            return config
    raise ValueError("유효한 seed configuration이 없습니다")


def pim_launch_options(config, plan, hardware):
    """backend가 정의하는 launch 옵션을 조립한다 (커널 공통부).

    constexpr(BLOCK_*, SPLIT_K)·grid·출력 버퍼는 커널 고유라 함수 본문이
    직접 쓴다.
    """
    p_m, p_n, p_k = plan.active_device_grid
    options = {
        "pim_target": hardware.target,
        "num_units": hardware.num_units,
        "local_lanes": hardware.lanes_per_core,
        "device_axis_partitions": (
            f"m:{p_m},n:{p_n},"
            f"k:{p_k}"),
        "group_m": config.group_m, "group_n": config.group_n,
        "group_k": config.group_k,
        # tile_* is the number of block groups.  The Common pass still accepts
        # logical element caps, so carry the derived values separately.
        "tile_m": config.tile_m, "tile_n": config.tile_n,
        "tile_k": config.tile_k,
        "capacity_limit_m": config.capacity_limit_m,
        "capacity_limit_n": config.capacity_limit_n,
        "capacity_limit_k": config.capacity_limit_k,
    }
    if config.bank_p_m and config.bank_p_n and config.bank_p_k:
        options["bank_partitions"] = (
            f"{config.bank_p_m},{config.bank_p_n},{config.bank_p_k}")
    if config.core_p_m and config.core_p_n and config.core_p_k:
        options["core_partitions"] = (
            f"{config.core_p_m},{config.core_p_n},{config.core_p_k}")
    if config.intra_device_lanes:
        options["intra_device_lanes"] = config.intra_device_lanes
    if config.lane_axis:
        options["lane_axis"] = config.lane_axis
    if hardware.target == "aimx" and config.bank_p_m:
        options["batch"] = config.bank_p_m
    return options


def _sample_indices(size, limit=32):
    """표본 검증용 결정론적 index — 균등 간격으로 최대 limit개."""
    step = -(-size // limit)
    return torch.arange(0, size, step)


def _env_flag(name, default):
    """env가 정책을 덮어쓴다. 값이 없으면 데코레이터 인자를 그대로 쓴다."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value == "1"


def pim_autotune(target=None, hardware=None, cache_path=DEFAULT_CACHE_PATH,
                 budget=None, space_overrides=None, ref_fn=None, reps=1,
                 seed=0, search=True, retune=False, verbose=True,
                 rtol=0.0, atol=0.0):
    """PIM GEMM launch 함수를 autotune한다.

    데코레이트된 함수의 계약: fn(a, b, *, config, plan) -> 출력 텐서 (M, N).

    정책은 두 값이 정한다. 데코레이터 인자가 기본이고, env가 그 위를
    덮어쓴다 — 코드를 고치지 않고 같은 스크립트를 다른 모드로 돌리기 위한
    통로다.

        search  (PIM_AUTOTUNE=0|1)  cache miss일 때 탐색할까
                                    True(기본): 탐색해서 best를 저장
                                    False: seed 유도 config 하나로 실행
        retune  (PIM_RETUNE=0|1)    cache에 있어도 다시 탐색할까
                                    False(기본): 저장된 best를 그대로 사용
                                    True: 재탐색 — 이미 잰 config는 저장값을
                                          재사용하고 새 후보만 실측한다

    - target: targets.py 등록표의 이름 (hardware를 직접 주면 무시)
    - space_overrides: runtime이 지원 못 하는 knob 제한
      (예: residency_policies=("stream",))
    - ref_fn: 탐색 중 정답 검증 기준. None이면 표본 검증 —
      출력의 균등 표본 위치만 작은 참조 GEMM으로 직접 계산해 비교한다.
    - rtol, atol: correctness의 상대/절대 허용 오차. 둘 다 0이면 기존의
      bit-exact 검사와 같다. BF16 target은 명시적인 허용 오차를 주어야 한다.
    - reps: config당 측정 반복(median, §32). warmup 1회는 항상 수행해
      compile 시간을 측정에서 뺀다.
    """
    profile = hardware if hardware is not None else target_profile(target)
    overrides = space_overrides if space_overrides is not None else {}

    def decorate(fn):
        # 허용 오차가 달라지면 과거 correctness 결과를 재사용하면 안 된다.
        # cache tuning key에 검증 정책을 포함해 자동으로 분리한다.
        verify_kind = "custom" if ref_fn is not None else "sampled"
        cache = PIMConfigCache(
            cache_path,
            template_version=(f"tile-counts-v1;verify={verify_kind};"
                              f"rtol={rtol};atol={atol}"))

        @functools.wraps(fn)
        def wrapper(a, b):
            problem = problem_from_tensors(a, b)
            planner = PIMPlanner(problem, profile)
            do_search = _env_flag("PIM_AUTOTUNE", search)
            do_retune = _env_flag("PIM_RETUNE", retune)

            if not do_retune:
                hit = cache.lookup_result(problem, profile)
                if hit is not None:
                    config = hit[0]
                    return fn(a, b, config=config, plan=planner.build(config))

            spec = build_config_space(problem, profile, **overrides)
            if not (do_search or do_retune):
                config = default_config(problem, profile, spec)
                return fn(a, b, config=config, plan=planner.build(config))

            # ── 탐색 모드 ── 검증 기준 준비 (탐색 중에만 검증한다).
            out_dtype = getattr(torch, problem.output_dtype)
            if ref_fn is not None:
                reference = ref_fn(a, b)

                def verify(out):
                    return bool(torch.allclose(out, reference,
                                               rtol=rtol, atol=atol))
            else:
                rows = _sample_indices(problem.M)
                cols = _sample_indices(problem.N)
                ref_block = (a.index_select(0, rows).to(out_dtype)
                             @ b.index_select(1, cols).to(out_dtype))

                def verify(out):
                    got = out.index_select(0, rows).index_select(1, cols)
                    return bool(torch.allclose(got, ref_block,
                                               rtol=rtol, atol=atol))

            def benchmark_fn(problem, hardware, config, plan):
                # 첫 호출은 compile을 포함하므로 측정에서 뺀다.
                out = fn(a, b, config=config, plan=plan)
                samples = []
                for _ in range(reps):
                    start = time.perf_counter_ns()
                    out = fn(a, b, config=config, plan=plan)
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                return PIMBenchmarkResult(
                    total_ms=statistics.median(samples),
                    correctness=verify(out))

            result = tune(problem, profile, benchmark_fn, space=spec,
                          cache=cache, budget=budget, seed=seed,
                          retune=do_retune, verbose=verbose)
            best = result.best
            return fn(a, b, config=best.config, plan=best.plan)

        return wrapper

    return decorate
