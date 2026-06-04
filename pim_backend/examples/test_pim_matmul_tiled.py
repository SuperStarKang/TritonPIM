# Multi-tile matmul: grid>1 + K-loop을 가진 표준 Triton matmul.
#
# 목적(O1.8): 단일 타일 가정을 제거하기 위해, 실제 multi-tile 커널의 TTSharedIR에서
# program_id·grid·K-loop이 어떤 형태로 나타나는지 확인한다. 이 IR을 보고 전체 M/N/K와
# grid→tile 매핑을 복원하는 로직(O1.8)을 설계한다.
#
# 주의: 현재 PIM 변환(convert-ttshared-to-pim)은 single-tile만 가정하므로 이 커널에 대해서는
# 잘못 변환되거나 이후 단계에서 실패할 수 있다. 지금은 ttsharedir(=PIM 변환 이전 단계)을
# 확인하는 것이 목적이므로, dump로 ttsharedir만 보면 된다.
#
# ttsharedir 확인:
#   TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/home/dlrkdals/TritonPIM/dump TRITON_ALWAYS_COMPILE=1 \
#     python pim_backend/examples/test_pim_matmul_tiled.py
#   → dump/<hash>/matmul_kernel_tiled.ttsharedir

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


def run(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, split_n=1, local_lanes=1):
    A = torch.rand((M, K), dtype=torch.float32)
    B = torch.rand((K, N), dtype=torch.float32)
    C = torch.empty((M, N), dtype=torch.float32)

    # split_n>1: N축을 split_n개 DPU에 분배(inter-DPU). local_lanes>1: 한 DPU 안에서
    # 출력 행을 local_lanes개 tasklet이 나눠 병렬(intra-DPU). 둘 다 PIM backend 옵션.
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel_tiled[grid](A, B, C, M, N, K,
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                              vendor="upmem", split_n=split_n, local_lanes=local_lanes)

    torch.testing.assert_close(C, A @ B, rtol=1e-3, atol=1e-3)
    print(f"OK: tiled matmul ({M}x{K})x({K}x{N}) -> ({M}x{N}), "
          f"grid={grid}, split_n={split_n}, local_lanes={local_lanes}")


if __name__ == "__main__":
    # ①a: device가 자기 부분행렬을 BLOCK_M×BLOCK_N 타일로 순회, BLOCK_K로 WRAM 누적.
    # 단일 tasklet(local_lanes=1) — cooperative 다중 tasklet은 다음 단계(①b).
    run(M=128, N=128, K=256, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, split_n=1, local_lanes=1)  # 1 DPU
    run(M=128, N=128, K=256, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, split_n=2, local_lanes=1)  # 2 DPU (n-split)
