# @pim_autotune 데코레이터 사용 예시 — AiMX float16 GEMM.
#
# 실행:
#   python autotune_gemm.py --m 1024 --n 8192 --k 28672 --max-benchmarks 30
#   정확성 검사는 기본 OFF이며 --check-correctness로 켠다.
#   retune=True로 매번 탐색하고 선택된 config를 cache에 저장한다.
#
# 같은 target spec을 autotuner와 kernel launch에 전달한다.
# config → 커널 인자 매핑은 아래 함수 본문이 소유하고,
# backend 공통 launch 옵션은 pim_launch_options helper가 조립한다.

import argparse
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent))         # gemm 커널
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # autotuner

from triton.backends.pim.driver import PIMDriver  # noqa: E402

from gemm import gemm_kernel  # noqa: E402
from autotuner import SearchBudget  # noqa: E402
from autotuner.decorator import (DEFAULT_CACHE_PATH, pim_autotune,  # noqa: E402
                                 pim_launch_options)
from autotuner.targets import target_spec  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="PIM AutoTuner 데코레이터 예시 — AiMX float16 GEMM")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=28672)
    parser.add_argument("--reps", type=int, default=1,
                        help="configuration당 측정 반복 수 (median, §32)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--max-benchmarks", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--cache", type=str, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--check-correctness",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="정확성 검사 ON/OFF (기본 OFF)")
    parser.add_argument("--rtol", type=float, default=5e-2,
                        help="BF16 correctness 상대 허용 오차")
    parser.add_argument("--atol", type=float, default=5e-2,
                        help="BF16 correctness 절대 허용 오차")
    args = parser.parse_args()

    triton.runtime.driver.set_active(PIMDriver())
    # hardware = target_spec("aimx")
    hardware = target_spec("aimx", num_units=1, base_fpga=0)

    budget = SearchBudget(
        max_generations=args.max_generations,
        max_benchmarks=args.max_benchmarks,
        max_autotune_seconds=args.max_seconds,
    )

    @pim_autotune(spec=hardware, cache_path=args.cache, budget=budget,
                  reps=args.reps, seed=args.seed,
                  rtol=args.rtol, atol=args.atol,
                  check_correctness=args.check_correctness,
                  retune=True)
    def gemm(a, b, *, config):
        M, K = a.shape
        _, N = b.shape
        c = torch.zeros((M, N), dtype=torch.float32)
        grid = (triton.cdiv(M, config.block_m),
                triton.cdiv(N, config.block_n))
        gemm_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=config.block_m, BLOCK_N=config.block_n,
            BLOCK_K=config.block_k,
            **pim_launch_options(config, hardware),
        )
        return c

    torch.manual_seed(0)
    a = torch.randn((args.m, args.k)).to(torch.float16)
    b = torch.randn((args.k, args.n)).to(torch.float16)

    start_ns = time.perf_counter_ns()
    out = gemm(a, b)
    wall_ms = (time.perf_counter_ns() - start_ns) / 1e6

    print(f"\nwall  : {wall_ms:.2f} ms (탐색이 돌았다면 그 시간을 포함한다)")
    if args.check_correctness:
        ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        print(f"오차  : {(ref - out).abs().max().item()}")


if __name__ == "__main__":
    main()
