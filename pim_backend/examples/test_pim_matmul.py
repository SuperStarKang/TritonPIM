# MVP matmul: A(M,K) x B(K,N) -> C(M,N), no-split 단일 tile.
#
# PIM 파이프라인 end-to-end 검증: 이 커널을 부르면 TTIR→TTSharedIR→PIM IR→UPMEM IR→
# host.cpp/device.c 생성→컴파일→DPU 실행까지 자동으로 돈다(PIMBackend.add_stages + PIMLauncher).
#
# 중간 IR 확인:
#   TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/path/to/dump python test_pim_matmul.py
#   → /path/to/dump/<hash>/ 에 단계별 IR

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

    # DPU는 순차 fp32 누적, torch는 블록/SIMD 누적이라 누적 순서 차이로 작은 오차가 정상적으로 난다.
    # 정확성 자체(동일 순서면 max_diff=0)는 runtime_test로 이미 검증됨 → 여기선 순서 차이만큼만 허용.
    torch.testing.assert_close(C, A @ B, rtol=1e-3, atol=1e-3)
    print(f"OK: matmul ({M}x{K})x({K}x{N}) -> ({M}x{N})")


if __name__ == "__main__":
    # 단일 tile (grid=(1,1)): BLOCK이 전체 M,N,K를 덮는다.
    run(M=32, N=64, K=256, BLOCK_M=32, BLOCK_N=64, BLOCK_K=256)
    # run(M=256, N=512, K=512, BLOCK_M=256, BLOCK_N=512, BLOCK_K=512) # Medium size
