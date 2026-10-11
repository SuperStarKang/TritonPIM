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
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent))         # GEMM 커널
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # autotuner

from triton.backends.pim.driver import PIMDriver  # noqa: E402

from gemm import gemm_kernel  # noqa: E402
from autotuner import SearchBudget  # noqa: E402
from autotuner.decorator import (DEFAULT_CACHE_PATH, pim_autotune,  # noqa: E402
                                 pim_launch_options)
from autotuner.targets import target_spec  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="PIM AutoTuner 데코레이터 예시 — UPMEM int8 GEMM")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=3,
                        help="configuration당 측정 반복 수 (median, §32)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--max-benchmarks", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--cache", type=str, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--num-units", type=int, default=2048,
                        help="UPMEM profile의 실제 DPU 수")
    parser.add_argument("--lanes-per-core", type=int, default=16,
                        help="DPU core의 tasklet capability")
    parser.add_argument("--nr-tasklets", type=int, default=0,
                        help="DPU당 활성 tasklet 수. 0이면 lanes-per-core")
    args = parser.parse_args()

    triton.runtime.driver.set_active(PIMDriver())

    target = target_spec(
        "upmem",
        num_units=args.num_units,
        lanes_per_core=args.lanes_per_core,
        nr_tasklets=args.nr_tasklets,
    )
    profile = target.common_profile
    runtime = target.runtime_config
    active_tasklets = runtime.nr_tasklets or profile.lanes_per_core
    print(
        f"UPMEM profile: num_units={profile.num_units}, "
        f"lanes_per_core={profile.lanes_per_core}, "
        f"nr_tasklets={active_tasklets}")

    budget = SearchBudget(
        max_generations=args.max_generations,
        max_benchmarks=args.max_benchmarks,
        max_autotune_seconds=args.max_seconds,
    )

    @pim_autotune(spec=target, cache_path=args.cache, budget=budget,
                  reps=args.reps, seed=args.seed,
                  template_version="semantic-k-gemm-v1")
    def gemm(a, b, *, config):
        M, K = a.shape
        _, N = b.shape
        # Source kernel은 하나의 semantic K reduction과 최종 C[M,N]만 표현한다.
        # p_k/tile_k는 아래 launch option으로 Common scheduling에 전달되며,
        # Triton grid, constexpr, output ABI에는 나타나지 않는다.
        c = torch.zeros((M, N), dtype=torch.int32)
        grid = (triton.cdiv(M, config.block_m),
                triton.cdiv(N, config.block_n))
        gemm_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=config.block_m, BLOCK_N=config.block_n,
            BLOCK_K=config.block_k,
            #upmem_reuse_staging=True,
            **pim_launch_options(config, target),
        )
        return c

    torch.manual_seed(0)
    a = torch.randn((args.m, args.k)).to(torch.int8)
    b = torch.randn((args.k, args.n)).to(torch.int8)

    start_ns = time.perf_counter_ns()
    out = gemm(a, b)
    wall_ms = (time.perf_counter_ns() - start_ns) / 1e6

    print(f"\nwall  : {wall_ms:.2f} ms (Tuning Time)")
    # ref = torch.matmul(a.to(torch.int32), b.to(torch.int32))
    # print(f"오차  : {(ref - out).abs().max().item()}")


if __name__ == "__main__":
    main()
