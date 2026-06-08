# Multi-tile matmul: grid>1 + K-loop을 가진 표준 Triton matmul.
#
# 현재 상태: build-pim-offload가 이 커널을 faithful-body PIM Common IR로 변환하고(본문을
# pim.compute region에 보존 + program_id 전역 오프셋을 tm/tn 분배층으로 이동), 2D SPLIT
# (split_m/n) + multi-tasklet(local_lanes)로 실제 DPU에서 정답까지 통과한다.
# (전체 M/N/K는 런타임 인자로 흐르고, BLOCK_M/N/K는 컴파일타임 상수.)
#
# 단계별 IR 확인:
#   TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/home/dlrkdals/TritonPIM/dump TRITON_ALWAYS_COMPILE=1 \
#     python pim_backend/examples/test_pim_matmul_tiled.py
#   → dump/<hash>/matmul_kernel_tiled.{ttsharedir,pim_common,vendor_ir}

import time

import torch
import triton
import triton.language as tl

from triton.backends.pim.driver import PIMDriver

triton.runtime.driver.set_active(PIMDriver())


@triton.jit
def matmul_kernel_tiled(A, B, C, M, N, K,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # grid (cdiv(M,BLOCK_M), cdiv(N,BLOCK_N)) 중 이 program이 맡는 출력 타일
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # K축을 BLOCK_K씩 streaming하며 누적 (이게 K-loop = 추후 chunk_plan의 원천)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])   # A(M,K) row-major
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])   # B(K,N) row-major
        acc += tl.dot(a, b)

    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)


def run(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, split_m=1, split_n=1, local_lanes=1):
    A = torch.rand((M, K), dtype=torch.float32)
    B = torch.rand((K, N), dtype=torch.float32)
    C = torch.empty((M, N), dtype=torch.float32)

    # 2D SPLIT: split_m×split_n개 DPU에 출력 분배(inter-DPU, 각 DPU = M/split_m × N/split_n).
    # local_lanes: 한 DPU 안에서 BLOCK 타일을 나눠 협력 계산(intra-DPU). 모두 PIM backend 옵션.
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    # warmup: 첫 호출의 컴파일·HW 백엔드 초기화를 타이밍에서 제외
    matmul_kernel_tiled[grid](A, B, C, M, N, K,
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                              vendor="upmem", split_m=split_m, split_n=split_n, local_lanes=local_lanes)
    t0 = time.perf_counter()
    matmul_kernel_tiled[grid](A, B, C, M, N, K,
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                              vendor="upmem", split_m=split_m, split_n=split_n, local_lanes=local_lanes)
    ms = (time.perf_counter() - t0) * 1e3  # warmup 이후라 compile/init 제외 (alloc/load/transfer/compute 포함)

    torch.testing.assert_close(C, A @ B, rtol=1e-3, atol=1e-3)
    print(f"OK: tiled matmul ({M}x{K})x({K}x{N}) -> ({M}x{N}), grid={grid}, "
          f"split_m={split_m}, split_n={split_n}, local_lanes={local_lanes}  |  {ms:.3f} ms")


if __name__ == "__main__":
    # 2D SPLIT: M·N을 split_m×split_n개 DPU에 분배. 각 DPU = M/split_m × N/split_n.
    #   유효: M%split_m==0, N%split_n==0, 그리고 Mu/Nu가 BLOCK_M/N의 배수.
    #   (M=N=128, BLOCK=64 → split_m,split_n ≤ 2.)
    # run(M=128, N=128, K=256, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, split_m=1, split_n=1, local_lanes=8)  # 1 DPU
    run(M=128, N=128, K=256, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, split_m=2, split_n=1, local_lanes=8)  # 2 DPU (M분할)
    run(M=128, N=128, K=256, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, split_m=2, split_n=2, local_lanes=8)  # 4 DPU (2D)
