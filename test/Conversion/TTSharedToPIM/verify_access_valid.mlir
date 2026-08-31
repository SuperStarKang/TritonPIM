// dynamic valid는 항상 같은 pim.access의 static sizes로 clamp되어야 한다.
// Block size mask를 Group size 변경 뒤 그대로 남기는 모순을 verifier가 거부한다.
// RUN: triton-shared-opt --pim-verify -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024,
    local_capacity_limit_bytes = 0>} {

func.func @stale_block_valid(%A: memref<*xi8>, %extent: index) attributes {
    pim.block = #pim.block<shape = [16, 1, 1]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  ^bb0(%device_m: index, %device_n: index, %device_k: index):
    pim.place %A start[%c0] size[%c16] allocation_shape[%c16]
        strides[%c1] {direction = "in", access = 0 : i64} : memref<*xi8>
    scf.for %mTile = %c0 to %c1 step %c1 {
      %origin = arith.muli %mTile, %c16 : index
      %remaining = arith.subi %extent, %origin : index
      %nonnegative = arith.maxsi %remaining, %c0 : index
      %staleBlockValid = arith.minsi %nonnegative, %c8 : index
      // expected-error @below {{dynamic valid extent on buffer axis 0 is not clamped to pim.access size 16}}
      %access = pim.access %A offset %origin sizes [16] strides[%c1]
          origin[%origin] valid[%staleBlockValid] {id = 0 : i64}
          : memref<*xi8> -> !pim.access
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
