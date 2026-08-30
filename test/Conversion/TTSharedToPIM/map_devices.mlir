// Common IR은 비용식이나 자동 인수분해 없이 [p_m,p_n,p_k]를 그대로 materialize한다.
// RUN: triton-shared-opt "--pim-map-devices=p-m=2 p-n=1 p-k=1" %s | FileCheck %s

// CHECK: #pim.device_mapping<partitions = [2, 1, 1]>
// CHECK: pim.execution_tile = #pim.execution_tile<groups = [1, 256, 1]>
// CHECK: ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK: arith.maxsi

module attributes {pim.target_profile = #pim.target_profile<
    target = "aimx", num_units = 2,
    banks_per_unit = 256,
    cores_per_bank = 1,
    lanes_per_core = 16,
    bank_capacity_limit_bytes = 268435456, local_capacity_limit_bytes = 0>} {

// Group 하나의 원소 수는 #pim.block × #pim.block_group에서 파생한다 — 이
// 커널은 16이다. pim.access는 metadata와 실제 access의 정합성을 verifier가
// 확인할 때만 사용한다.
func.func @two_axes(%C: memref<*xf32>, %tx: index, %ty: index)
    attributes {
      pim.block = #pim.block<shape = [16, 16, 1]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
      pim.execution_tile = #pim.execution_tile<groups = [1, 256, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %ldc = arith.constant 4096 : index
  scf.for %x = %c0 to %tx step %c1 {
    %m = arith.muli %x, %c16 : index
    scf.for %y = %c0 to %ty step %c1 {
      %n = arith.muli %y, %c16 : index
      %d = pim.access %C offset %n sizes [16, 16] strides[%ldc, %c1] origin[%m, %n]
           {id = 0 : i64} : memref<*xf32> -> !pim.access
    } {pim.axis = #pim.axis<"n">}
  } {pim.axis = #pim.axis<"m">}
  return
}

}
