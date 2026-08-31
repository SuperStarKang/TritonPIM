// 하나의 normalized reduction result를 둘 이상의 외부 output으로 복제하면 어느
// destination이 operation의 결과인지 Common이 하나로 확정할 수 없다.
// RUN: triton-shared-opt "--pim-map-devices=p-m=1 p-n=1 p-k=1" -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @two_final_outputs(%A: memref<*xi32>, %C: memref<*xi32>,
                             %D: memref<*xi32>, %groups: index)
    attributes {
      pim.block = #pim.block<shape = [1, 1, 1]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
      pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<1xi32>
  // expected-error @below {{a K-only reduction result must have exactly one direct final output pim.store}}
  %result = scf.for %k = %c0 to %groups step %c1
      iter_args(%acc = %empty) -> tensor<1xi32> {
    %input = pim.access %A offset %k sizes [1] strides[%c1] origin[%k]
        {id = 0 : i64} : memref<*xi32> -> !pim.access
    scf.yield %acc : tensor<1xi32>
  } {pim.reduction = "add"}
  %cAccess = pim.access %C offset %c0 sizes [1] strides[%c1] origin[%c0]
      {id = 1 : i64} : memref<*xi32> -> !pim.access
  pim.store %result, %cAccess : tensor<1xi32>, !pim.access
  %dAccess = pim.access %D offset %c0 sizes [1] strides[%c1] origin[%c0]
      {id = 2 : i64} : memref<*xi32> -> !pim.access
  pim.store %result, %dAccess : tensor<1xi32>, !pim.access
  return
}
}
