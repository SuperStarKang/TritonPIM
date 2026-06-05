// PIM Common IR(gemm) → UPMEM IR(host 시퀀스 + device_func) 변환 확인.
// RUN: triton-shared-opt --legalize-pim-to-upmem %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_kernel
func.func @matmul_kernel(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>) {
  %0 = pim.memory_region %A {space = "host", layout = "row_major", role = "input_activation"} : (memref<*xf32>) -> !pim.region
  %1 = pim.memory_region %B {space = "device_global", layout = "row_major", role = "weight"} : (memref<*xf32>) -> !pim.region
  %2 = pim.memory_region %C {space = "host", layout = "row_major", role = "output"} : (memref<*xf32>) -> !pim.region
  %3 = pim.tile_plan {block_m = 32 : i64, block_n = 64 : i64, block_k = 256 : i64, split_m = 1 : i64, split_n = 1 : i64, split_k = 1 : i64} : !pim.tile_plan
  pim.transfer %0 {src = "host", dst = "device_global"} : !pim.region
  pim.transfer %1 {src = "host", dst = "device_global"} : !pim.region
  %4 = pim.compute(%0, %1, %2) using %3 {op = "gemm"} : (!pim.region, !pim.region, !pim.region, !pim.tile_plan) -> !pim.region
  pim.transfer %2 {src = "device_global", dst = "host"} : !pim.region
  pim.launch @matmul_kernel {vendor = "upmem", device_id = 0 : i64}
  return
}

// host 시퀀스가 함수 본문으로 들어간다.
// CHECK: upmem.alloc_units {num_units = 1 : i64, split_m = 1 : i64, split_n = 1 : i64} : !upmem.dpu_set
// CHECK: upmem.load_program %{{.*}}, @matmul_kernel_device : !upmem.dpu_set
// CHECK: upmem.copy_to_global %{{.*}}, %arg0 {dst = "mram_a"}
// CHECK: upmem.copy_to_global %{{.*}}, %arg1 {dst = "mram_b"}
// CHECK: upmem.launch %{{.*}} : !upmem.dpu_set
// CHECK: upmem.copy_from_global %{{.*}}, %arg2 {src = "mram_c"}
// CHECK-NOT: pim.compute

// device_func가 module 레벨에 생성된다.
// CHECK: upmem.device_func "matmul_kernel_device"
// CHECK: upmem.local_gemm {accumulation = "fp32", k = 256 : i64, m = 32 : i64, n = 64 : i64}
