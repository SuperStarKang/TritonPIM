import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver
from triton.backends.pim.driver import PIMDriver

import triton.backends.pim.driver as pim_driver

@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # --- 나는 C의 몇 번째 조각을 담당하는가 ---
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
 
    # --- 내가 만질 행 번호 / 열 번호 목록 ---
    # 예: pid_m=3, BLOCK_M=64  ->  offs_m = [192, 193, ..., 255]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
 
    # --- 주소 표 만들기 ---
    # [:, None] = 세로 방향, [None, :] = 가로 방향
    # 두 방향을 더하면 2차원 주소 표가 자동으로 만들어짐
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
 
    # --- 누적 변수는 반드시 fp32 ---
    # 입력이 fp16이어도 수백 번 더하면 정밀도가 무너지기 때문
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
 
    # --- K 루프: K를 BLOCK_K씩 잘라서 나눠 곱하고 더한다 ---
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
 
        # mask: 진짜 존재하는 칸만 읽기
        # other: 없는 칸은 0으로 채우기 (0을 곱해 더하면 결과에 영향 없음)
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
 
        # (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N) -> (BLOCK_M x BLOCK_N)
        # 이 한 줄이 텐서 코어 명령으로 내려간다
        acc += tl.dot(a, b)
 
        # 다음 K 조각으로 주소 표 전체를 이동
        a_ptrs += BLOCK_K * stride_ak   # A는 가로로
        b_ptrs += BLOCK_K * stride_bk   # B는 세로로
 
    # --- 결과 쓰기 (읽기와 완전히 대칭) ---
    c = acc.to(tl.int32)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.zeros((M, N), dtype=torch.int32)

    configs = [
        # 1024, 28672, 8192
        (
            (16, 16, 16),       # BLOCK_M, BLOCK_N, BLOCK_K
            (1, 1, 1),          # group_m, group_n, group_k
            (64, 7, 64),        # tile_m, tile_n, tile_k
            (2048, 1, 256, 8),  # num_units, p_m, p_n, p_k
        ),
        (
            (64, 32, 64),
            (1, 1, 1),
            (4, 7, 32),
            (2048, 4, 128, 4),
        ),
        (
            (16,128,16),
            (4,1,2),
            (2,1,128),
            (2048, 8, 128, 2),
        ),
        # 4096, 4096, 4096
        (
            (16,16,16),
            (1,1,2),
            (8,8,64),
            (2048, 32, 32, 2),
        ),
        # 1024, 1024, 1024
        (
            (16, 16, 32),
            (1, 1, 1),
            (2, 1, 32),
            (2048, 32, 64, 1)
        ),
        (
            (1, 1, 1),
            (16, 16, 32),
            (2, 1, 32),
            (2048, 32, 64, 1)
        ),
        # 512, 768, 3072
        (
            (16, 16, 16),
            (1, 1, 2),
            (1, 6, 24),
            (2048, 32, 32, 1)
        ),
    ]

    # 사용할 config 선택
    config = configs[2]

    block, group, tile, partition = config

    BLOCK_M, BLOCK_N, BLOCK_K = block
    group_m, group_n, group_k = group
    tile_m, tile_n, tile_k = tile
    num_units, p_m, p_n, p_k = partition

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    gemm_kernel[grid](
        a, b, c,
        M, N, K,

        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),

        # target
        pim_target="upmem",

        # 계층 1: Triton Block
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,

        # 계층 2: Block Group
        group_m=group_m, group_n=group_n, group_k=group_k,

        # 계층 3: Execution Tile
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,

        # Device planning
        num_units=num_units,
        p_m=p_m, p_n=p_n, p_k=p_k,

        #upmem_reuse_staging=True,
    )

    return c


def test():
    torch.manual_seed(0)
    # M, N, K = 1024, 8192, 28672
    M, N, K = 1024, 28672, 8192
    #M, N, K = 4096, 4096, 4096
    # M, N, K = 1024, 1024, 1024

    a = torch.randn((M, K)).to(torch.int8)
    b = torch.randn((K, N)).to(torch.int8)

    ref = torch.matmul(a.to(torch.int32), b.to(torch.int32))
    gemm(a,b) # warmup

    timings = []
    out = None
    for _ in range(3):
        out = gemm(a, b)
        timings.append(list(pim_driver.timing))

    avg_timing = [
        sum(run[i] for run in timings) / len(timings)
        for i in range(len(timings[0]))
    ]

    max_diff = torch.max(torch.abs(ref - out)).item()
    print(f"M={M}, N={N}, K={K}")
    print(f"max_diff = {max_diff:.6f}")
    assert max_diff < 1e-3, f"결과 불일치: max_diff={max_diff}"
    print("PASS")

    for name, ms in zip(pim_driver.TIMING_STAGES, avg_timing):
        print(f"{name:8s}: {ms:9.2f} ms")
    print(f"{'total':8s}: {sum(avg_timing):9.2f} ms")


if __name__ == "__main__":
    # triton.runtime.driver.set_active(CPUDriver())
    triton.runtime.driver.set_active(PIMDriver())
    test()
