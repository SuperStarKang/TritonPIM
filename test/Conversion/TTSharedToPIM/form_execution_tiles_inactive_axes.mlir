// 비활성 N과 reduction 축에 명시적인 항등값 1을 전달해도 고정 [m,n,k]
// Execution Tile에 그대로 1로 materialize한다.
// RUN: triton-shared-opt --pim-form-block-groups "--pim-form-execution-tiles=tile-m=32 tile-n=1 tile-k=1" %s | FileCheck %s

// CHECK-LABEL: func.func @one_axis
// CHECK-SAME: pim.block = #pim.block<shape = [16, 1, 1]>
// CHECK-SAME: pim.block_group = #pim.block_group<blocks = [1, 1, 1]>
// CHECK-SAME: pim.execution_tile = #pim.execution_tile<groups = [32, 1, 1]>

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 2048,
    banks_per_unit = 1,
    cores_per_bank = 1,
    lanes_per_core = 16,
    bank_capacity_limit_bytes = 62914560,
    local_capacity_limit_bytes = 49152>} {

func.func @one_axis(%A: memref<*xf32>, %tm: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %tm step %c1 {
    %m = arith.muli %i, %c16 : index
    %a = pim.access %A offset %m sizes [16] strides[%c1]
         origin[%m] {id = 0 : i64} : memref<*xf32> -> !pim.access
  } {pim.axis = #pim.axis<"m">}
  return
}
}
