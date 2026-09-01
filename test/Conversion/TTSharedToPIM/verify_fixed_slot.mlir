// start와 size는 runtime 값일 수 있지만 allocation_shape는 반드시 hierarchy에서
// 파생된 정적 Execution Tile slot이어야 한다.
// RUN: triton-shared-opt --pim-verify -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @dynamic_allocation_shape(%A: memref<*xi8>, %capacity: index)
    attributes {
      pim.block = #pim.block<shape = [1, 1, 1]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  ^bb0(%device_m: index, %device_n: index, %device_k: index):
    // expected-error @below {{allocation_shape[0] must be a positive compile-time constant}}
    pim.place %A start[%c0] size[%c4] allocation_shape[%capacity]
        strides[%c1] {direction = "in", access = 0 : i64} : memref<*xi8>
    %access = pim.access %A offset %c0 sizes [4] strides[%c1] origin[%c0]
        {id = 0 : i64} : memref<*xi8> -> !pim.access
  }
  return
}
}
