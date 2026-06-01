# MVP matmul: A(M,K) x B(K,N) -> C(M,N), no-split 단일 tile (DESIGN 22장 형태).
#
# 현재 lowering은 CPU 경로를 재사용하므로 이 테스트는 지금도 정답까지 통과해야 한다
# (PIM 전용 lowering은 M3~M6에서 교체). 동시에 이 커널의 TTSharedIR을 dump해서
# M3 변환 pass(convert-ttshared-to-pim)의 입력 형태를 확인하는 용도로도 쓴다.
#
# TTSharedIR 확인:
#   TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton_pim python test_pim_matmul.py
#   → /tmp/triton_pim/<hash>/matmul_kernel.ttsharedir 확인

import torch
import triton
import triton.language as tl

from triton.backends.pim.driver import PIMDriver

triton.runtime.driver.set_active(PIMDriver())


@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A는 (M,K) row-major, B는 (K,N) row-major.
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])

    acc = tl.dot(a, b)

    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)


def run(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    A = torch.rand((M, K), dtype=torch.float32)
    B = torch.rand((K, N), dtype=torch.float32)
    C = torch.empty((M, N), dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](A, B, C, M, N, K,
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                        vendor="upmem")

    torch.testing.assert_close(C, A @ B)
    print(f"OK: matmul ({M}x{K})x({K}x{N}) -> ({M}x{N})")


if __name__ == "__main__":
    # 단일 tile (grid=(1,1)): BLOCK이 전체 M,N,K를 덮는다.
    run(M=32, N=64, K=256, BLOCK_M=32, BLOCK_N=64, BLOCK_K=256)
