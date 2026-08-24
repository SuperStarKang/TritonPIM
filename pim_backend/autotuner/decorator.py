# @pim_autotune — problem(M,N,K)만으로 동작하는 사용면.
#
# 사용자는 config/plan을 받아 커널을 launch하는 함수를 쓰고, 데코레이터가
# 나머지를 담당한다:
#   - problem은 텐서 shape·dtype에서 자동 구성
#   - hardware는 target 등록표(targets.py)에서
#   - cache 정확 일치       → best config/plan 주입해 실행 (측정·검증 없음)
#   - miss, PIM_AUTOTUNE≠1 → heuristic seed의 첫 유효 config 하나로 실행
#   - miss, PIM_AUTOTUNE=1 → 데코레이트된 함수를 benchmark로 tune()을 돌려
#                             best를 cache에 저장하고 best로 실행
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
    return {
        "pim_target": hardware.target,
        "num_units": hardware.num_units,
        "local_lanes": hardware.execution_units_per_device,
        "device_axis_partitions": (
            f"output_row_tile:{p_m},output_column_tile:{p_n},"
            f"explicit_k_partition:{p_k}"),
        "group_m": config.group_m, "group_n": config.group_n,
        "group_k": config.group_k,
        "tile_m": config.tile_m, "tile_n": config.tile_n,
        "tile_k": config.tile_k,
    }


def _sample_indices(size, limit=32):
    """표본 검증용 결정론적 index — 균등 간격으로 최대 limit개."""
    step = -(-size // limit)
    return torch.arange(0, size, step)


def pim_autotune(target=None, hardware=None, cache_path=DEFAULT_CACHE_PATH,
                 budget=None, space_overrides=None, ref_fn=None, reps=1,
                 seed=0, verbose=True):
    """PIM GEMM launch 함수를 autotune한다.

    데코레이트된 함수의 계약: fn(a, b, *, config, plan) -> 출력 텐서 (M, N).

    - target: targets.py 등록표의 이름 (hardware를 직접 주면 무시)
    - space_overrides: runtime이 지원 못 하는 knob 제한
      (예: residency_policies=("stream",))
    - ref_fn: 탐색 중 정답 검증 기준. None이면 표본 검증 —
      출력의 균등 표본 위치만 작은 참조 GEMM으로 직접 계산해 비교한다.
    - reps: config당 측정 반복(median, §32). warmup 1회는 항상 수행해
      compile 시간을 측정에서 뺀다.
    """
    profile = hardware if hardware is not None else target_profile(target)
    overrides = space_overrides if space_overrides is not None else {}

    def decorate(fn):
        cache = PIMConfigCache(cache_path)

        @functools.wraps(fn)
        def wrapper(a, b):
            problem = problem_from_tensors(a, b)
            planner = PIMPlanner(problem, profile)

            hit = cache.lookup_result(problem, profile)
            if hit is not None:
                config = hit[0]
                return fn(a, b, config=config, plan=planner.build(config))

            spec = build_config_space(problem, profile, **overrides)
            if os.environ.get("PIM_AUTOTUNE") != "1":
                config = default_config(problem, profile, spec)
                return fn(a, b, config=config, plan=planner.build(config))

            # ── 탐색 모드 ── 검증 기준 준비 (탐색 중에만 검증한다).
            out_dtype = getattr(torch, problem.output_dtype)
            if ref_fn is not None:
                reference = ref_fn(a, b)

                def verify(out):
                    return bool(torch.equal(out, reference))
            else:
                rows = _sample_indices(problem.M)
                cols = _sample_indices(problem.N)
                ref_block = (a.index_select(0, rows).to(out_dtype)
                             @ b.index_select(1, cols).to(out_dtype))

                def verify(out):
                    got = out.index_select(0, rows).index_select(1, cols)
                    return bool(torch.equal(got, ref_block))

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
                          verbose=verbose)
            best = result.best
            return fn(a, b, config=best.config, plan=best.plan)

        return wrapper

    return decorate
