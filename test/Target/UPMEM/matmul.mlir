// UPMEM IR → host C++ / device C 소스 생성 확인.
// RUN: triton-shared-translate --upmem-to-host %s | FileCheck %s --check-prefix=HOST
// RUN: triton-shared-translate --upmem-to-device %s | FileCheck %s --check-prefix=DEVICE

func.func @matmul_kernel(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>) {
  %0 = upmem.alloc_units {num_units = 1 : i64, split_m = 1 : i64, split_n = 1 : i64, split_k = 1 : i64} : !upmem.dpu_set
  upmem.load_program %0, @matmul_kernel_device : !upmem.dpu_set
  upmem.copy_to_global %0, %A {dst = "mram_a"} : !upmem.dpu_set, memref<*xf32>
  upmem.copy_to_global %0, %B {dst = "mram_b"} : !upmem.dpu_set, memref<*xf32>
  upmem.launch %0 : !upmem.dpu_set
  upmem.copy_from_global %0, %C {src = "mram_c"} : !upmem.dpu_set, memref<*xf32>
  return
}
// device_func 본문 = faithful-body gemm nest. translate가 scf.for step에서 BLOCK을 읽는다
// (tm step=BLOCK_M=32, tn step=BLOCK_N=64, kc step=BLOCK_K=64). block arg = 유닛-로컬 슬라이스.
upmem.device_func "matmul_kernel_device" {
^unit(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>):
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %mu = memref.dim %a, %c0 : memref<?x?xf32>
  %k = memref.dim %a, %c1 : memref<?x?xf32>
  %nu = memref.dim %c, %c1 : memref<?x?xf32>
  scf.for %tm = %c0 to %mu step %c32 {
    scf.for %tn = %c0 to %nu step %c64 {
      %acc = memref.alloc() : memref<32x64xf32>
      linalg.fill ins(%cst : f32) outs(%acc : memref<32x64xf32>)
      scf.for %kc = %c0 to %k step %c64 {
        %at = memref.subview %a[%tm, %kc] [32, 64] [1, 1] : memref<?x?xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
        %bt = memref.subview %b[%kc, %tn] [64, 64] [1, 1] : memref<?x?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
        linalg.matmul ins(%at, %bt : memref<32x64xf32, strided<[?, 1], offset: ?>>, memref<64x64xf32, strided<[?, 1], offset: ?>>) outs(%acc : memref<32x64xf32>)
      }
      %cv = memref.subview %c[%tm, %tn] [32, 64] [1, 1] : memref<?x?xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
      memref.copy %acc, %cv : memref<32x64xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
    }
  }
}

// HOST: extern "C" void matmul_kernel_host(float* A, float* B, float* C, uint32_t M, uint32_t N, uint32_t K)
// HOST: const uint32_t SPLIT_M = 1, SPLIT_N = 1, SPLIT_K = 1;
// HOST: dpu_alloc(NUM_UNITS, NULL, &set)
// HOST: dpu_load(set, DPU_BINARY, NULL)
// HOST: dpu_push_xfer(set, DPU_XFER_TO_DPU, "A_mram"
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
