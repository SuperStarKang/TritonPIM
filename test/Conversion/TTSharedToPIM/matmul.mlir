// triton.dot_origin이 붙은 linalg.matmul을 가진 함수가 PIM Common IR로 교체되는지 확인.
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_kernel
func.func @matmul_kernel(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
                         %M: i32, %N: i32, %K: i32) {
  %a = tensor.empty() : tensor<32x256xf32>
  %b = tensor.empty() : tensor<256x64xf32>
  %z = tensor.empty() : tensor<32x64xf32>
  // BLOCK_M=32, BLOCK_K=256 (lhs), BLOCK_N=64 (result)
  %r = linalg.matmul {triton.dot_origin}
       ins(%a, %b : tensor<32x256xf32>, tensor<256x64xf32>)
       outs(%z : tensor<32x64xf32>) -> tensor<32x64xf32>
  return
}

// 변환 후 함수 본문은 pim 시퀀스로 교체된다.
// CHECK: pim.memory_region %arg0 {layout = "row_major", role = "input_activation", space = "host"}
// CHECK: pim.memory_region %arg1 {layout = "row_major", role = "weight", space = "device_global"}
// CHECK: pim.memory_region %arg2 {layout = "row_major", role = "output", space = "host"}
// CHECK: pim.tile_plan {block_k = 256 : i64, block_m = 32 : i64, block_n = 64 : i64, split_k = 1 : i64, split_m = 1 : i64, split_n = 1 : i64}
// CHECK: pim.transfer
// CHECK: pim.compute(%{{.*}}, %{{.*}}, %{{.*}}) using %{{.*}} {op = "gemm"}
// CHECK: pim.launch @matmul_kernel {device_id = 0 : i64, vendor = "upmem"}
// CHECK-NOT: linalg.matmul
