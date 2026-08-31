// M/N 축이 없는 linalg.reduce도 zero-init, temporal K window, resident
// placement, partial-to-final combine이 같은 pim.device Execution Tile에 들어간다.
// RUN: triton-shared-opt --pim-form-compute "--pim-map-devices=p-m=1 p-n=1 p-k=1" --pim-place-data --pim-verify %s | FileCheck %s

// CHECK-LABEL: func.func @k_only_sum
// CHECK: scf.for
// CHECK: pim.device <partitions = [1, 1, 1]>
// CHECK-SAME: pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>
// zero-init이 host round 바깥이 아니라 pim.device 안, reduction 직전에 있다.
// Placement는 같은 device 안에서 init 뒤, compute loop 앞에 materialize된다.
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1xi32>
// CHECK: %[[INIT:.*]] = linalg.fill
// Input K slot은 Block K=4 × Tile K=2인 8개 원소로 고정된다.
// CHECK: pim.place %arg0
// CHECK-SAME: allocation_shape[%{{.*}}, %{{.*}}]
// Scalar output slot은 한 원소다.
// CHECK: pim.place %arg1
// CHECK-SAME: allocation_shape[%{{.*}}]
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[INIT]]) -> (tensor<1xi32>)
// CHECK: linalg.reduce
// CHECK: } {pim.reduction = "add", pim.temporal_reduction}
// partial과 최종 logical output access가 명시적으로 연결된다.
// CHECK: pim.combine {{.*}} into {{.*}} kind "add"
// CHECK: } {pim.scheduling_axis = "reduction_k"}

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @k_only_sum(%A: memref<*xi32>, %C: memref<*xi32>,
                      %totalGroups: index)
    attributes {
      pim.block = #pim.block<shape = [1, 1, 4]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
      pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<1xi32>
  %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<1xi32>)
      -> tensor<1xi32>
  %result = scf.for %kGroup = %c0 to %totalGroups step %c1
      iter_args(%acc = %init) -> tensor<1xi32> {
    %k = arith.muli %kGroup, %c4 : index
    %aAccess = pim.access %A offset %k sizes [1, 4] strides[%c4, %c1]
        origin[%c0, %k]
        {id = 0 : i64} : memref<*xi32> -> !pim.access
    %tile = pim.load %aAccess : !pim.access -> tensor<1x4xi32>
    %next = linalg.reduce ins(%tile : tensor<1x4xi32>)
        outs(%acc : tensor<1xi32>) dimensions = [1]
        (%value: i32, %old: i32) {
      %sum = arith.addi %value, %old : i32
      linalg.yield %sum : i32
    }
    scf.yield %next : tensor<1xi32>
  }
  %cAccess = pim.access %C offset %c0 sizes [1] strides[%c1] origin[%c0]
      {id = 1 : i64} : memref<*xi32> -> !pim.access
  pim.store %result, %cAccess : tensor<1xi32>, !pim.access
  return
}
}
