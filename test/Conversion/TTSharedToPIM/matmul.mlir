// linalg.matmul을 가진 함수가 faithful-body PIM Common IR로 변환되는지 확인.
// (triton-to-linalg는 별도 marker attribute 없이 linalg.matmul을 emit한다.)
// RUN: triton-shared-opt --build-pim-offload %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_kernel
func.func @matmul_kernel(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
                         %M: i32, %N: i32, %K: i32) {
  %a = tensor.empty() : tensor<32x256xf32>
  %b = tensor.empty() : tensor<256x64xf32>
  %z = tensor.empty() : tensor<32x64xf32>
  // BLOCK_M=32, BLOCK_K=256 (lhs), BLOCK_N=64 (result)
  %r = linalg.matmul
       ins(%a, %b : tensor<32x256xf32>, tensor<256x64xf32>)
       outs(%z : tensor<32x64xf32>) -> tensor<32x64xf32>
  return
}

// 변환 후: 분배 구조(memory_region/tile_plan/transfer/launch)는 합성되고,
// 계산 본문은 pim.compute region에 유닛-로컬 nest로 보존된다.
// CHECK: pim.memory_region %arg0 {layout = "row_major", role = "input_activation", space = "host"}
// CHECK: pim.memory_region %arg1 {layout = "row_major", role = "weight", space = "host"}
// CHECK: pim.memory_region %arg2 {layout = "row_major", role = "output", space = "host"}
// CHECK: pim.tile_plan {block_k = 256 : i64, block_m = 32 : i64, block_n = 64 : i64, split_k = 1 : i64, split_m = 1 : i64, split_n = 1 : i64}
// CHECK: pim.execution_group {local_lanes = 1 : i64, mapping_policy = "none", num_units = 1 : i64, stationary_data = "B"}
// CHECK: pim.transfer
// CHECK: pim.compute ins(%{{.*}}, %{{.*}}) outs(%{{.*}}) using %{{.*}}, %{{.*}} {primitive = "gemm"}
// CHECK-SAME: : (!pim.region, !pim.region) (!pim.region) (!pim.exec_group, !pim.tile_plan)
// faithful-body: tm/tn(복원된 grid) + K-loop + 보존된 linalg.matmul(memref form). 본문 op은 pim.class로 태깅.
// CHECK: ^{{.*}}(%[[A:.*]]: memref<?x?xf32>, %[[B:.*]]: memref<?x?xf32>, %[[C:.*]]: memref<?x?xf32>):
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.fill
// CHECK: scf.for
// CHECK: memref.subview
// CHECK: linalg.matmul {pim.class = "gemm"} ins({{.*}} : memref<32x256xf32{{.*}}>, memref<256x64xf32{{.*}}>) outs({{.*}} : memref<32x64xf32>)
// CHECK: memref.copy
// CHECK: pim.launch @matmul_kernel {device_id = 0 : i64, vendor = "upmem"}
