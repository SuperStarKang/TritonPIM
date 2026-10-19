# @pim_autotune 데코레이터 사용 예시 — UPMEM semantic-K GEMM.
#
# 실행:
#   conda activate pim_triton
#   python autotune_gemm.py --m 1024 --n 8192 --k 28672 --max-benchmarks 30
#     cache에 이 shape가 있으면 그 best로 실행, 없으면 탐색해서 저장한다.
#   PIM_RETUNE=1 python autotune_gemm.py ...   cache가 있어도 재탐색
#   PIM_AUTOTUNE=0 python autotune_gemm.py ... 탐색 금지 (miss면 seed 유도값)
#
# 하드웨어 수치는 compiler와 autotuner가 공유하는 PIMTargetSpec에서 온다.
# config → 커널 인자 매핑은 아래 함수 본문이 소유하고,
# backend 공통 launch 옵션은 pim_launch_options helper가 조립한다.

import argparse
from functools import partial
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # autotuner

from triton.backends.pim.driver import PIMDriver  # noqa: E402

from autotuner import PIMPerformanceProfile, SearchBudget  # noqa: E402
from autotuner.artifacts import RESULTS_ROOT, prepare_run_directory, project_path  # noqa: E402
from autotuner.decorator import pim_autotune  # noqa: E402
from autotuner.gemm import (GEMM_KERNEL_VERSION, ensure_gemm_profile,  # noqa: E402
                            profile_gemm, run_gemm)
from autotuner.model import target_model  # noqa: E402
from autotuner.targets import target_spec  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="PIM AutoTuner 데코레이터 예시 — UPMEM int8 GEMM")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=5,
                        help="configuration당 측정 반복 수 (median, §32)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--max-benchmarks", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--cache", type=str)
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT / "upmem_gemm"))
    parser.add_argument("--retune", action="store_true", help="기존 tuning cache가 있어도 재탐색")
    parser.add_argument("--performance", help="single-unit saturation JSON")
    profiling = parser.add_mutually_exclusive_group()
    profiling.add_argument("--profile-only", action="store_true",
                           help="target 용량 안에서 단일-device profiling 후 종료")
    profiling.add_argument("--calibrate-performance", nargs=3, type=int, metavar=("M", "N", "K"),
                           help="측정 크기 상한을 지정해 profiling 후 종료")
    parser.add_argument("--num-units", type=int, default=2048,
                        help="UPMEM profile의 실제 DPU 수")
    parser.add_argument("--lanes-per-core", type=int, default=16,
                        help="DPU core의 tasklet capability")
    parser.add_argument("--nr-tasklets", type=int, default=0,
                        help="DPU당 활성 tasklet 수. 0이면 lanes-per-core")
    parser.add_argument("--local-capacity", type=int, default=0,
                        help="Block Group 타일 버퍼 WRAM 한도(bytes). 0이면 target 기본값")
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    output_dir = prepare_run_directory(args.output_dir)
    cache_path = project_path(args.cache) if args.cache is not None else output_dir / "autotune_cache.json"

    triton.runtime.driver.set_active(PIMDriver())

    target = target_spec(
        "upmem",
        num_units=args.num_units,
        lanes_per_core=args.lanes_per_core,
        nr_tasklets=args.nr_tasklets,
        local_capacity_limit_bytes=args.local_capacity,
    )
    profile = target.common_profile
    runtime = target.runtime_config
    active_tasklets = runtime.nr_tasklets or profile.lanes_per_core
    print(
        f"UPMEM profile: num_units={profile.num_units}, "
        f"lanes_per_core={profile.lanes_per_core}, "
        f"nr_tasklets={active_tasklets}, "
        f"local_capacity={profile.local_capacity_limit_bytes}")

    if args.profile_only or args.calibrate_performance is not None:
        performance = profile_gemm(
            target, output_dir=output_dir, repeat=args.reps,
            sweep_extents=args.calibrate_performance)
        if args.performance is not None:
            performance.save(args.performance)
        return

    budget = SearchBudget(
        max_generations=args.max_generations,
        max_benchmarks=args.max_benchmarks,
        max_autotune_seconds=args.max_seconds,
    )

    performance = None
    if args.performance is not None:
        performance = PIMPerformanceProfile.load(args.performance, kernel_version=GEMM_KERNEL_VERSION)

    gemm_spec = target_model(target).gemm_profile_spec

    @pim_autotune(spec=target, cache_path=str(cache_path), budget=budget, performance=performance,
                  profile_fn=partial(ensure_gemm_profile, output_dir=output_dir, repeat=args.reps),
                  reps=args.reps, seed=args.seed, retune=args.retune,
                  template_version=GEMM_KERNEL_VERSION)
    def gemm(a, b, *, config):
        return run_gemm(a, b, target, config, output_dtype=gemm_spec.output_dtype)

    torch.manual_seed(0)
    a = torch.randn((args.m, args.k)).to(getattr(torch, gemm_spec.input_dtype))
    b = torch.randn((args.k, args.n)).to(getattr(torch, gemm_spec.input_dtype))

    start_ns = time.perf_counter_ns()
    out = gemm(a, b)
    wall_ms = (time.perf_counter_ns() - start_ns) / 1e6

    print(f"\nwall  : {wall_ms:.2f} ms (profiling/autotuning 포함)")
    # ref = torch.matmul(a.to(torch.int32), b.to(torch.int32))
    # print(f"오차  : {(ref - out).abs().max().item()}")


if __name__ == "__main__":
    main()
