// UPMEM IR → host C++ / device C 소스 생성 확인.
// RUN: triton-shared-translate --upmem-to-host %s | FileCheck %s --check-prefix=HOST
// RUN: triton-shared-translate --upmem-to-device %s | FileCheck %s --check-prefix=DEVICE

func.func @matmul_kernel(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>) {
  %0 = upmem.alloc_units {num_units = 1 : i64} : !upmem.dpu_set
  upmem.load_program %0, @matmul_kernel_device : !upmem.dpu_set
  upmem.copy_to_global %0, %A {dst = "mram_a"} : !upmem.dpu_set, memref<*xf32>
  upmem.copy_to_global %0, %B {dst = "mram_b"} : !upmem.dpu_set, memref<*xf32>
  upmem.launch %0 : !upmem.dpu_set
  upmem.copy_from_global %0, %C {src = "mram_c"} : !upmem.dpu_set, memref<*xf32>
  return
}
upmem.device_func "matmul_kernel_device" {
  // m/n/k = BLOCK (DPU 내부 타일). BLOCK_K는 WRAM에 맞게 작게.
  upmem.local_gemm {m = 32 : i64, n = 64 : i64, k = 64 : i64, accumulation = "fp32"}
}

// HOST: extern "C" void matmul_kernel_host(float* A, float* B, float* C, uint32_t M, uint32_t N, uint32_t K)
// HOST: const uint32_t NUM_UNITS = 1;
// HOST: dpu_alloc(NUM_UNITS, NULL, &set)
// HOST: dpu_load(set, DPU_BINARY, NULL)
// HOST: dpu_broadcast_to(set, "A_mram"
// HOST: dpu_push_xfer(set, DPU_XFER_TO_DPU, "Bt_mram"
// HOST: dpu_launch(set, DPU_SYNCHRONOUS)
// HOST: dpu_push_xfer(set, DPU_XFER_FROM_DPU, "C_mram"

// DEVICE: #define BLOCK_M 32
// DEVICE: #define BLOCK_N 64
// DEVICE: #define BLOCK_K 64
// DEVICE: __host uint32_t M;
// DEVICE: __dma_aligned float aW[BLOCK_M * BLOCK_K]
// DEVICE: BARRIER_INIT(my_barrier, NR_TASKLETS)
// DEVICE: int main()
// DEVICE: uint32_t t = me();
// DEVICE: for (uint32_t tm = 0; tm < M; tm += BLOCK_M)
// DEVICE: for (uint32_t r = t; r < BLOCK_M; r += NR_TASKLETS)
// DEVICE: barrier_wait(&my_barrier)
// DEVICE: acc[r * BLOCK_N + c] += s
// DEVICE: mram_write(&acc[r * BLOCK_N], &C_mram[(tm + r) * N + tn], BLOCK_N * sizeof(float))
