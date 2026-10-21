import torch
import triton
import triton.language as tl

from triton.backends.pim.driver import PIMDriver


@triton.jit
def gemv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    stride_x,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute y[m] = sum_k A[m, k] * x[k].

    각 Triton program은 BLOCK_M개의 출력 행을 맡고, 내부 loop는 K를
    BLOCK_K씩 순회한다. 이 형태를 TTShared IR에서

      elementwise multiply + reduction

    로 확인한 뒤 linalg.matvec로 canonicalize하는 것이 다음 구현 단계다.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M,), dtype=tl.int32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k = k_block * BLOCK_K + offs_k
        a = tl.load(
            a_ptr
            + offs_m[:, None] * stride_am
            + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        x = tl.load(
            x_ptr + k * stride_x,
            mask=k < K,
            other=0.0,
        )
        # 곱하기 전에 INT32로 넓힌다. INT8끼리 곱하면 곱이 INT8에서 넘친다.
        acc += tl.sum(a.to(tl.int32) * x.to(tl.int32)[None, :], axis=1)

    tl.store(y_ptr + offs_m * stride_y, acc, mask=offs_m < M)


def gemv(a, x):
    assert a.ndim == 2
    assert x.ndim == 1
    M, K = a.shape
    assert x.shape[0] == K

    y = torch.empty((M,), device=a.device, dtype=torch.int32)
    BLOCK_M = 32
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M),)

    # TTShared의 broadcast -> multiply -> transpose -> reduce(sum) 연쇄는
    # PIM Common에서 linalg.matvec로 정규화되고, UPMEM은 M 행을
    # tasklet에 나누며 x vector를 공유하는 device loop로 lowering한다.
    gemv_kernel[grid](
        a,
        x, y,
        M, K,
        a.stride(0), a.stride(1),
        x.stride(0), y.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        pim_target="upmem",
        num_units=4,
        group_m=1, group_n=1, group_k=1,
        tile_m=1, tile_k=1,
        p_m=4, p_n=1, p_k=1,
    )
    return y


if __name__ == "__main__":
    triton.runtime.driver.set_active(PIMDriver())
    torch.manual_seed(0)

    M, K = 128, 256
    a = torch.randn((M, K)).to(torch.int8)
    x = torch.randn((K,)).to(torch.int8)

    out = gemv(a, x)
    # torch.mv는 INT8 입력에 INT8 결과를 낸다. INT32로 넓혀 기준값을 만든다.
    ref = torch.mv(a.to(torch.int32), x.to(torch.int32))
    print((out - ref).abs().max())
