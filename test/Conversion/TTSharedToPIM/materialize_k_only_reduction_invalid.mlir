// Temporal partial에 epilogue를 먼저 적용하면 partial끼리 combine한 뒤 epilogue를
// 적용하는 원래 의미와 달라질 수 있으므로 K-only 경로도 direct store만 허용한다.
// RUN: triton-shared-opt --pim-form-compute "--pim-map-devices=p-m=1 p-n=1 p-k=1" -verify-diagnostics %s

#identity = affine_map<(d0) -> (d0)>

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @epilogue_before_combine(%A: memref<*xi32>, %C: memref<*xi32>,
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
  %epilogueEmpty = tensor.empty() : tensor<1xi32>
  %epilogue = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]}
      ins(%result : tensor<1xi32>)
      outs(%epilogueEmpty : tensor<1xi32>) {
  ^bb0(%value: i32, %out: i32):
    linalg.yield %value : i32
  } -> tensor<1xi32>
  %cAccess = pim.access %C offset %c0 sizes [1] strides[%c1] origin[%c0]
      {id = 1 : i64} : memref<*xi32> -> !pim.access
  // expected-error @below {{an epilogue consumes the K-only reduction result before partial combine}}
  pim.store %epilogue, %cAccess : tensor<1xi32>, !pim.access
  return
}
}
