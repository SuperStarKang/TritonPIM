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
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_x,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute y[m] = sum_k A[m, k] * x[k].

    각 Triton program은 BLOCK_M개의 출력 행을 맡고, 내부 loop는 K를
    BLOCK_K씩 순회한다. Common lowering은 이 multiply/reduce 형태를
    linalg.matvec와 pim.combine으로 정규화하고 AiMX는 N=1 MAC으로 실행한다.

    현재 실행 가능한 AiMX GEMV template은 M % BLOCK_M == 0 및
    K % BLOCK_K == 0인 full-block 형태다. rank-1 masked x/y의 dynamic valid를
    Common lowering이 아직 정규화하지 못하므로 tail은 host에서 거부한다.
    """
    pid_m = tl.program_id(0)
    local_m = tl.arange(0, BLOCK_M)
    m_start = pid_m * BLOCK_M
    offs_m = m_start + local_m
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        a_ptr
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    x_ptrs = x_ptr + offs_k * stride_x

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        x = tl.load(x_ptrs)
        acc += tl.sum(a * x[None, :], axis=1)
        a_ptrs += BLOCK_K * stride_ak
        x_ptrs += BLOCK_K * stride_x

    tl.store(y_ptr + offs_m * stride_y, acc)


def gemv(a, x, *, num_units=1, base_fpga=0):
    assert a.ndim == 2
    assert x.ndim == 1
    M, K = a.shape
    assert x.shape[0] == K

    y = torch.empty((M,), device=a.device, dtype=torch.float32)
    BLOCK_M = 32
    BLOCK_K = 64
    assert M % BLOCK_M == 0, (
        "현재 AiMX GEMV template은 M tail을 지원하지 않습니다: "
        f"M={M}, BLOCK_M={BLOCK_M}"
    )
    assert K % BLOCK_K == 0, (
        "현재 AiMX GEMV template은 K tail을 지원하지 않습니다: "
        f"K={K}, BLOCK_K={BLOCK_K}"
    )
    grid = (triton.cdiv(M, BLOCK_M),)

    # group_*와 tile_*은 원소 수가 아니라 각각 Block 수와 Group 수다.
    # 모든 K-Group을 한 Tile에 넣고, AiMX 기본 배치에서 출력 영역별 vACC에 누적한다.
    gemv_kernel[grid](
        a,
        x, y,
        M, K,
        a.stride(0), a.stride(1),
        x.stride(0), y.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        pim_target="aimx",

        num_units=num_units, base_fpga=base_fpga,
        group_m=1, group_n=1, group_k=1,
        tile_m=1, tile_n=1, tile_k=K // BLOCK_K,
        p_m=num_units, p_n=1, p_k=1,
    )
    return y


if __name__ == "__main__":
    triton.runtime.driver.set_active(PIMDriver())

    M, K = 128, 256
    k_start = 96
    scale = 0.5

    a = torch.ones((M, K), dtype=torch.float32)
    x = torch.zeros((K,), dtype=torch.float32)
    x[k_start:k_start + 32] = scale

    out = gemv(a, x)

    print(f"K[{k_start}:{k_start + 32}] scale={scale}")
    print("expected:", 32.0 * scale)
    print("out, 8 rows per line:")
    print(out.reshape(M // 8, 8))
    print("max_abs_error:", (out - 32.0 * scale).abs().max().item())
