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
#   → dump/<hash>/matmul_kernel_f32.{ttsharedir,pim_common,vendor_ir}

import time

import torch
import triton
import triton.language as tl

from triton.backends.pim.driver import PIMDriver

triton.runtime.driver.set_active(PIMDriver())


@triton.jit
def matmul_kernel_f32(A, B, C, M, N, K,
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


@triton.jit
def matmul_kernel_i8(A, B, C, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # int8 입력 + int32 누적. UPMEM 정수 ALU가 fp32 소프트 에뮬보다 빠른지 실험용.
    # 구조는 float 커널과 동일 — dtype만 다르고, codegen이 IR matmul에서 타입을 자동 인식한다.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])   # int8
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])   # int8
        acc += tl.dot(a, b, out_dtype=tl.int32)                  # int8×int8 → int32

    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)


def run_f32(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, split_m=1, split_n=1, split_k=1, local_lanes=1):
    A = torch.rand((M, K), dtype=torch.float32)
    B = torch.rand((K, N), dtype=torch.float32)
    C = torch.empty((M, N), dtype=torch.float32)

    # SPLIT: split_m×split_n×split_k개 DPU 사용. split_m/n=출력 2D 분배, split_k=K분담(→host reduce).
    # local_lanes: 한 DPU 안에서 BLOCK 타일을 나눠 협력 계산(intra-DPU). 모두 PIM backend 옵션.
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    # warmup: 첫 호출의 컴파일·HW 백엔드 초기화를 타이밍에서 제외
    matmul_kernel_f32[grid](A, B, C, M, N, K,
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                              vendor="upmem", split_m=split_m, split_n=split_n, split_k=split_k, local_lanes=local_lanes)
    t0 = time.perf_counter()
    matmul_kernel_f32[grid](A, B, C, M, N, K,
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                              vendor="upmem", split_m=split_m, split_n=split_n, split_k=split_k, local_lanes=local_lanes)
    ms = (time.perf_counter() - t0) * 1e3  # warmup 이후라 compile/init 제외 (alloc/load/transfer/compute 포함)

    torch.testing.assert_close(C, A @ B, rtol=1e-3, atol=1e-3)
    print(f"OK: tiled matmul ({M}x{K})x({K}x{N}) -> ({M}x{N}), grid={grid}, "
          f"split_m={split_m}, split_n={split_n}, split_k={split_k}, local_lanes={local_lanes}, "
          f"DPUs={split_m*split_n*split_k}  |  {ms:.3f} ms")


def run_i8(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, split_m=1, split_n=1, split_k=1, local_lanes=1):
    # int8 입력 / int32 출력. dtype은 텐서에서 결정 → codegen이 자동 인식(typedef Tin=int8_t, Tacc=int32_t).
    # 주의: int8은 MRAM 8-byte 정렬 때문에 BLOCK_K가 8의 배수 + K/N 짝수여야 한다(아니면 정렬 에러).
    A = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    B = torch.randint(-8, 8, (K, N), dtype=torch.int8)
    C = torch.empty((M, N), dtype=torch.int32)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # warmup
    for _ in range(3):  # int8은 초기 호출이 더 느려서 여러 번 warmup
        matmul_kernel_i8[grid](A, B, C, M, N, K,
                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                               vendor="upmem", split_m=split_m, split_n=split_n, split_k=split_k, local_lanes=local_lanes)
    
    # timed run
    t0 = time.perf_counter()
    matmul_kernel_i8[grid](A, B, C, M, N, K,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                           vendor="upmem", split_m=split_m, split_n=split_n, split_k=split_k, local_lanes=local_lanes)
    ms = (time.perf_counter() - t0) * 1e3

    # 정수 matmul은 라운딩이 없어 정확히 일치해야 한다(불일치면 dtype/정렬 버그 신호).
    torch.testing.assert_close(C, A.to(torch.int32) @ B.to(torch.int32))
    print(f"OK[i8]: tiled matmul ({M}x{K})x({K}x{N}) -> ({M}x{N}), grid={grid}, "
          f"split_m={split_m}, split_n={split_n}, split_k={split_k}, local_lanes={local_lanes}, "
          f"DPUs={split_m*split_n*split_k}  |  {ms:.3f} ms")


# if __name__ == "__main__":
    # SPLIT: 총 DPU = split_m × split_n × split_k.
    #   유효: M%split_m==0, N%split_n==0, K%split_k==0, 그리고 Mu/Nu/Ku가 BLOCK_M/N/K의 배수.
    # run_f32(M=128, N=128, K=256, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, split_m=2, split_n=1, local_lanes=8)  # 2 DPU (M분할)
    # run_f32(M=128, N=128, K=256, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, split_m=2, split_n=2, local_lanes=8)  # 4 DPU (2D)
    # # split_k: 한 출력 타일을 여러 DPU가 K분담 → host tree-reduce. Ku=K/split_k가 BLOCK_K의 배수여야 함.
    # run_f32(M=128, N=128, K=256, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, split_m=1, split_n=1, split_k=2, local_lanes=8)  # 2 DPU (K분담)
    # run_f32(M=128, N=128, K=256, BLOCK_M=8, BLOCK_N=8, BLOCK_K=8, split_m=1, split_n=8, split_k=16, local_lanes=16)
    # # big gemm (1024, 2880, 4096)
    # run_f32(M=1024, N=4096, K=2880, BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, split_m=32, split_n=64, split_k=1, local_lanes=16)

    # === int8 실험 (int8 입력 + int32 누적) ===
    # BLOCK_K=32(8의 배수), K/N 짝수라 int8 MRAM 정렬 OK. f32와 같은 분배에서 시간 비교.
    # run_i8(M=128, N=128, K=256, BLOCK_M=8, BLOCK_N=8, BLOCK_K=8, split_m=1, split_n=8, local_lanes=16)
    # run_i8(M=1024, N=4096, K=1024, BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, split_m=1, split_n=64, split_k=32, local_lanes=16)
    # run_i8(M=1024, N=4096, K=1024, BLOCK_M=32, BLOCK_N=32, BLOCK_K=8, split_m=16, split_n=32, split_k=4, local_lanes=16)
    
    # block k test
    # run_i8(M=1024, N=4096, K=1024, BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, split_m=16, split_n=32, split_k=4, local_lanes=16)
    # run_i8(M=1024, N=4096, K=1024, BLOCK_M=32, BLOCK_N=32, BLOCK_K=128, split_m=16, split_n=32, split_k=4, local_lanes=16)
    # run_i8(M=1024, N=4096, K=1024, BLOCK_M=32, BLOCK_N=32, BLOCK_K=256, split_m=16, split_n=32, split_k=4, local_lanes=16)
    # run_i8(M=1024, N=4096, K=1024, BLOCK_M=32, BLOCK_N=32, BLOCK_K=512, split_m=16, split_n=32, split_k=2, local_lanes=16)

    # split k test
    # run_i8(M=1024, N=4096, K=1024, BLOCK_M=32, BLOCK_N=32, BLOCK_K=8, split_m=16, split_n=32, split_k=1, local_lanes=16)
    # run_i8(M=1024, N=4096, K=1024, BLOCK_M=32, BLOCK_N=32, BLOCK_K=8, split_m=16, split_n=32, split_k=2, local_lanes=16)
    # run_i8(M=1024, N=4096, K=1024, BLOCK_M=32, BLOCK_N=32, BLOCK_K=8, split_m=16, split_n=32, split_k=4, local_lanes=16)
    # run_i8(M=1024, N=4096, K=1024, BLOCK_M=32, BLOCK_N=32, BLOCK_K=256, split_m=16, split_n=128, split_k=1, local_lanes=16)

    # llama-3.1-70b
    # run_i8(M=1024, N=8192,  K=8192,  BLOCK_M=32, BLOCK_N=32, BLOCK_K=256, split_m=8,  split_n=64,  split_k=4, local_lanes=16)   # q_proj, o_proj
    # run_i8(M=1024, N=1024,  K=8192,  BLOCK_M=32, BLOCK_N=32, BLOCK_K=256, split_m=16, split_n=32,  split_k=4, local_lanes=16)   # k/v_proj
    # run_i8(M=1024, N=28672, K=8192,  BLOCK_M=32, BLOCK_N=32, BLOCK_K=256, split_m=4,  split_n=128, split_k=4, local_lanes=16)   # ffn1(gate_proj, up_proj)
    # run_i8(M=1024, N=8192,  K=28672, BLOCK_M=32, BLOCK_N=32, BLOCK_K=256, split_m=16, split_n=128, split_k=1, local_lanes=16)   # ffn2(down_proj)

if __name__ == "__main__":
    M, N, K = 1024, 4096, 1024   # K, N 짝수 (int8 정렬 OK)

    # ── Sweep A: BLOCK_K (WRAM 청크 재사용) ── 가장 직접적인 타일링 경향
    #   split_k=1로 고정해 reduce/D2H 혼입 제거 → kernel이 순수 재사용 효과만 반영.
    #   Mu=64, Nu=128 (÷BLOCK 32 OK), Ku=1024 (모든 BLOCK_K로 나눠떨어짐)
    for bk in [8, 16, 32, 64, 128, 256, 512]:   # 512=36KB OK, 1024는 WRAM 초과라 제외
        run_i8(M, N, K, BLOCK_M=32, BLOCK_N=32, BLOCK_K=bk,
               split_m=16, split_n=32, split_k=1, local_lanes=16)   # 512 DPU 고정

    # ── Sweep B: BLOCK_M=BLOCK_N (acc 타일 재사용) ──
    #   local_lanes=8로 고정(BLOCK_M=8도 8을 나눔). BLOCK_K=256 고정.
    for bm in [8, 16, 32, 64]:                   # Mu=64, Nu=128 모두 나눔. 64×64×256=49KB OK
        run_i8(M, N, K, BLOCK_M=bm, BLOCK_N=bm, BLOCK_K=256,
               split_m=16, split_n=32, split_k=1, local_lanes=8)

    # ── Sweep C: split_k (K분담 → reduce/D2H 팽창 경향) ──
    #   kernel은 줄지만 D2H·pack이 커지는 tradeoff(IMPLEMENT ⑧)를 본다.
    #   유닛수 바뀌므로 매번 acquire ~305ms 발생 → total*/kernel만 읽을 것.
    for sk in [1, 2, 4]:                          # Ku=1024/512/256, 모두 ÷256 OK. DPU 512/1024/2048
        run_i8(M, N, K, BLOCK_M=32, BLOCK_N=32, BLOCK_K=256,
               split_m=16, split_n=32, split_k=sk, local_lanes=16)

    # ── Sweep D: local_lanes (intra-DPU tasklet 병렬) ──
    for ll in [1, 2, 4, 8, 16, 24]:               # 모두 BLOCK_M=32를 나눔
        run_i8(M, N, K, BLOCK_M=32, BLOCK_N=32, BLOCK_K=256,
               split_m=16, split_n=32, split_k=1, local_lanes=ll)