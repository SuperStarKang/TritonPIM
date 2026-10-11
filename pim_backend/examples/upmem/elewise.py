import torch, triton
import triton.language as tl
from triton.backends.pim.driver import PIMDriver

@triton.jit
def ew_kernel(a_ptr, b_ptr, c_ptr,
            M, N,
            stride_am, stride_an,
            stride_bm, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 2D grid → logical 축 m, n 으로 그대로 복원된다
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = b_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn

    # pad 는 0 만 허용된다
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    # arith 스칼라 연산만. 세 텐서 shape 이 모두 같아야 한다
    c = a * b + b

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask)


def ew(a, b):
    M, N = a.shape
    c = torch.empty_like(a)
    BLOCK_M, BLOCK_N = 32, 64          # ← 아래 예산식으로 직접 고르십시오
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    ew_kernel[grid](
        a, b, c, M, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        pim_target="upmem",
    )
    return c


if __name__ == "__main__":
    triton.runtime.driver.set_active(PIMDriver())
    a = torch.randn((1024, 1024)); b = torch.randn((1024, 1024))
    out = ew(a, b)
    print((out - (a * b + b)).abs().max())