// Common IR이 temporal K를 target에 맡기지 않고 scheduling round, device-local
// reduction window, 정확한 resident placement, partial combine으로 모두 명시한다.
// RUN: triton-shared-opt "--pim-map-devices=p-m=1 p-n=1 p-k=1" --pim-place-data --pim-verify -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @gemm
// CHECK: scf.for
// CHECK: pim.device #pim.device_mapping<partitions = [1, 1, 1]>
// CHECK-SAME: pim.execution_tile = #pim.execution_tile<groups = [2, 2, 2]>
// A 한 window는 M=8*2, K=4*2 원소만 상주시킨다. 전체 K=4*8은 allocation에 없다.
// CHECK: pim.place %arg0
// CHECK-SAME: allocation_shape[{{.*}}, {{.*}}]
// CHECK: scf.for {{.*}} iter_args
// CHECK: } {pim.temporal_reduction}
// CHECK: pim.store {{.*}} {pim.combine = "add"}
// CHECK: } {pim.scheduling_axis = "reduction_k"}

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1048576,
    local_capacity_limit_bytes = 0>} {

func.func @gemm(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>)
    attributes {
      pim.block = #pim.block<shape = [8, 8, 4]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
      pim.execution_tile = #pim.execution_tile<groups = [2, 2, 2]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %zero = arith.constant 0 : i32

  scf.for %mTile = %c0 to %c4 step %c1 {
    %m = arith.muli %mTile, %c8 : index
    scf.for %nTile = %c0 to %c4 step %c1 {
      %n = arith.muli %nTile, %c8 : index
      %empty = tensor.empty() : tensor<8x8xi32>
      %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<8x8xi32>)
          -> tensor<8x8xi32>
      %result = scf.for %kTile = %c0 to %c8 step %c1
          iter_args(%acc = %init) -> tensor<8x8xi32> {
        %k = arith.muli %kTile, %c4 : index
        %aAccess = pim.access %A offset %m sizes [8, 4]
            strides[%c32, %c1] origin[%m, %k] {id = 0 : i64}
            : memref<*xi8> -> !pim.access
        %a = pim.load %aAccess : !pim.access -> tensor<8x4xi8>
        %bAccess = pim.access %B offset %k sizes [4, 8]
            strides[%c32, %c1] origin[%k, %n] {id = 1 : i64}
            : memref<*xi8> -> !pim.access
        %b = pim.load %bAccess : !pim.access -> tensor<4x8xi8>
        %next = linalg.matmul
            ins(%a, %b : tensor<8x4xi8>, tensor<4x8xi8>)
            outs(%acc : tensor<8x8xi32>) -> tensor<8x8xi32>
        scf.yield %next : tensor<8x8xi32>
      }
      %cAccess = pim.access %C offset %n sizes [8, 8]
          strides[%c32, %c1] origin[%m, %n] {id = 2 : i64}
          : memref<*xi32> -> !pim.access
      pim.store %result, %cAccess : tensor<8x8xi32>, !pim.access
    } {pim.axis = #pim.axis<"n">}
  } {pim.axis = #pim.axis<"m">}
  return
}
}
