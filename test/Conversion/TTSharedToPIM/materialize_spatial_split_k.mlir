// Spatial split-K plane, p_k devices, plane 내부 temporal window와 partial
// combiner를 Common IR 하나에 모두 남긴다. Target은 이 구조를 재해석하지 않고
// combine의 실행 위치만 선택한다.
// RUN: triton-shared-opt "--pim-map-devices=p-m=1 p-n=1 p-k=2" --pim-place-data --pim-verify -canonicalize %s | FileCheck %s

// Spatial plane 전체를 p_k=2 device에 배치하고 남는 plane은 같은 device set의
// scheduling round에서 반복한다.
// CHECK-LABEL: func.func @spatial_split_k
// CHECK: pim.device <partitions = [1, 1, 2]>
// CHECK-SAME: pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>
// spatial K plane은 한 device에 하나만 resident하고, plane 내부 reduction K는
// execution_tile.groups[k]=2 Group만 resident한다.
// CHECK: pim.place %arg0
// CHECK-SAME: allocation_shape[{{.*}}, {{.*}}]
// CHECK: pim.place %arg1
// CHECK-SAME: allocation_shape[{{.*}}, {{.*}}]
// 최종 output은 split-K plane을 외부 차원으로 노출하지 않는다.
// CHECK: pim.place %arg2
// CHECK-SAME: allocation_shape[{{.*}}, {{.*}}]
// CHECK: scf.for {{.*}} iter_args
// CHECK: } {pim.reduction = "add", pim.temporal_reduction}
// 같은 add가 temporal window, spatial plane, p_k device partial 모두의 combiner다.
// CHECK: pim.combine {{.*}} into {{.*}} kind "add"
// CHECK: } {pim.axis = #pim.axis<"k">}
// CHECK: } {pim.scheduling_axis = "reduction_k"}
// CHECK: } {pim.scheduling_axis = "k"}

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 2,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1048576,
    local_capacity_limit_bytes = 0>} {

func.func @spatial_split_k(
    %A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>,
    %mTiles: index, %nTiles: index, %planes: index, %kGroups: index,
    %kPerPlane: index, %lda: index, %ldb: index, %ldc: index)
    attributes {
      pim.block = #pim.block<shape = [8, 8, 4]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
      pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0 : i32

  scf.for %mi = %c0 to %mTiles step %c1 {
    %m = arith.muli %mi, %c8 : index
    scf.for %ni = %c0 to %nTiles step %c1 {
      %n = arith.muli %ni, %c8 : index
      scf.for %plane = %c0 to %planes step %c1 {
        %kBase = arith.muli %plane, %kPerPlane : index
        %empty = tensor.empty() : tensor<8x8xi32>
        %init = linalg.fill ins(%zero : i32)
            outs(%empty : tensor<8x8xi32>) -> tensor<8x8xi32>
        %result = scf.for %kg = %c0 to %kGroups step %c1
            iter_args(%acc = %init) -> tensor<8x8xi32> {
          %kOffset = arith.muli %kg, %c4 : index
          %k = arith.addi %kBase, %kOffset : index

          %aRow = arith.muli %m, %lda : index
          %aOffset = arith.addi %aRow, %k : index
          %aAccess = pim.access %A offset %aOffset sizes [8, 4]
              strides[%lda, %c1] origin[%m, %k] {id = 0 : i64}
              : memref<*xi8> -> !pim.access
          %a = pim.load %aAccess : !pim.access -> tensor<8x4xi8>

          %bRow = arith.muli %k, %ldb : index
          %bOffset = arith.addi %bRow, %n : index
          %bAccess = pim.access %B offset %bOffset sizes [4, 8]
              strides[%ldb, %c1] origin[%k, %n] {id = 1 : i64}
              : memref<*xi8> -> !pim.access
          %b = pim.load %bAccess : !pim.access -> tensor<4x8xi8>

          %next = linalg.matmul
              ins(%a, %b : tensor<8x4xi8>, tensor<4x8xi8>)
              outs(%acc : tensor<8x8xi32>) -> tensor<8x8xi32>
          scf.yield %next : tensor<8x8xi32>
        } {pim.reduction = "add"}

        %cRow = arith.muli %m, %ldc : index
        %cMN = arith.addi %cRow, %n : index
        %cAccess = pim.access %C offset %cMN sizes [8, 8]
            strides[%ldc, %c1] origin[%m, %n] {id = 2 : i64}
            : memref<*xi32> -> !pim.access
        pim.store %result, %cAccess : tensor<8x8xi32>, !pim.access
      } {pim.axis = #pim.axis<"k">}
    } {pim.axis = #pim.axis<"n">}
  } {pim.axis = #pim.axis<"m">}
  return
}
}
