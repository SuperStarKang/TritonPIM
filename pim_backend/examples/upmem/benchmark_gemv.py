# GEMV(y = A·x, int32)를 UPMEM에서 반복 실행하고 stage별 평균 시간을 출력한다.
#
# 실행:
#   ./cmd.sh benchmark_gemv.py --m 4096 --k 4096 --block-m 32 --block-k 64 --p-m 64 --p-k 32
#
# 마지막 두 줄은 run_benchmark_gemv.sh가 CSV로 모으는 형식이다:
#   HEADER,m,k,block_m,block_k,p_m,p_k,tasklets,wall,<stage...>
#   RESULT,...

import argparse
import time

import torch
import triton

import triton.backends.pim.driver as pim_driver
from triton.backends.pim.driver import PIMDriver

from gemv import gemv_kernel


def launch(a, x, args):
    M, K = a.shape
    y = torch.empty((M,), dtype=torch.int32)
    gemv_kernel[(triton.cdiv(M, args.block_m),)](
        a,
        x, y,
        M, K,
        a.stride(0), a.stride(1),
        x.stride(0), y.stride(0),
        BLOCK_M=args.block_m, BLOCK_K=args.block_k,
        pim_target="upmem",
        num_units=args.p_m * args.p_k,
        nr_tasklets=args.tasklets,
        group_m=1, group_n=1, group_k=1,
        tile_m=1, tile_k=1,
        p_m=args.p_m, p_n=1, p_k=args.p_k,
    )
    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--p-m", type=int, required=True, help="M축 DPU 분할 수")
    parser.add_argument("--p-k", type=int, default=1, help="K축 DPU 분할 수")
    parser.add_argument("--tasklets", type=int, default=0, help="0이면 target profile 값")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    triton.runtime.driver.set_active(PIMDriver())
    torch.manual_seed(0)
    a = torch.randint(0, 50, (args.m, args.k), dtype=torch.int32)
    x = torch.randint(0, 50, (args.k,), dtype=torch.int32)

    # 첫 호출에는 device.elf 컴파일과 DPU 예약이 섞이므로 측정에서 뺀다
    for _ in range(args.warmup):
        launch(a, x, args)

    walls = []
    stages = []
    for _ in range(args.repeat):
        start = time.perf_counter_ns()
        y = launch(a, x, args)
        walls.append((time.perf_counter_ns() - start) / 1e6)
        stages.append(list(pim_driver.timing))

    max_diff = (y - torch.mv(a, x)).abs().max().item()
    assert max_diff == 0, f"결과 불일치: max_diff={max_diff}"

    wall_ms = sum(walls) / len(walls)
    stage_ms = [sum(run[i] for run in stages) / len(stages)
                for i in range(len(pim_driver.TIMING_STAGES))]

    print(f"M={args.m} K={args.k} BLOCK_M={args.block_m} BLOCK_K={args.block_k} "
          f"p_m={args.p_m} p_k={args.p_k} tasklets={args.tasklets} repeat={args.repeat}")
    for name, ms in zip(pim_driver.TIMING_STAGES, stage_ms):
        print(f"{name:12s}: {ms:9.3f} ms")
    print(f"{'stage_sum':12s}: {sum(stage_ms):9.3f} ms")
    print(f"{'wall':12s}: {wall_ms:9.3f} ms")

    print(",".join(["HEADER", "m", "k", "block_m", "block_k", "p_m", "p_k", "tasklets", "wall",
                    *pim_driver.TIMING_STAGES]))
    print(",".join(["RESULT",
                    *map(str, [args.m, args.k, args.block_m, args.block_k, args.p_m, args.p_k, args.tasklets]),
                    f"{wall_ms:.4f}",
                    *(f"{ms:.4f}" for ms in stage_ms)]))


if __name__ == "__main__":
    main()
