// Reduction 없는 elementwise/map 커널은 존재하는 공간축마다 block group 하나를
// 보수적인 자동 execution tile로 선택한다.
// RUN: triton-shared-opt --pim-form-execution-tiles %s | FileCheck %s
// RUN: triton-shared-opt "--pim-form-execution-tiles=tile-m=2 tile-n=3" %s | FileCheck %s --check-prefix=EXPLICIT

// CHECK-LABEL: func.func @elementwise
// CHECK-SAME: pim.execution_tile = #pim.execution_tile<groups = [1, 1, 0]>
// EXPLICIT-LABEL: func.func @elementwise
// EXPLICIT-SAME: pim.execution_tile = #pim.execution_tile<groups = [2, 3, 0]>

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 8,
    banks_per_unit = 1,
    cores_per_bank = 1,
    lanes_per_core = 4,
    bank_capacity_limit_bytes = 62914560,
    local_capacity_limit_bytes = 49152>} {

func.func @elementwise(%A: memref<*xf32>, %B: memref<*xf32>,
                       %tm: index, %tn: index) attributes {
    pim.block = #pim.block<shape = [8, 32, 0]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  scf.for %i = %c0 to %tm step %c1 {
    %m = arith.muli %i, %c8 : index
    scf.for %j = %c0 to %tn step %c1 {
      %n = arith.muli %j, %c32 : index
      %a = pim.access %A offset %n sizes [8, 32] strides[%c32, %c1]
           origin[%m, %n] {id = 0 : i64} : memref<*xf32> -> !pim.access
      %b = pim.access %B offset %n sizes [8, 32] strides[%c32, %c1]
           origin[%m, %n] {id = 1 : i64} : memref<*xf32> -> !pim.access
    } {pim.axis = #pim.axis<"n">}
  } {pim.axis = #pim.axis<"m">}
  return
}
}
