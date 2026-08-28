// RUN: triton-shared-opt --pim-verify -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1,
    cores_per_bank = 1,
    lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024,
    local_capacity_limit_bytes = 0>} {

// expected-error @below {{#pim.block × #pim.block_group gives 2 elements on m but the local compute tile uses 1}}
func.func @inconsistent_group_shape(%arg0: memref<*xi8>) attributes {
    pim.block = #pim.block<shape = [2, 0, 0]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 0, 0]>} {
  ^bb0(%device_m: index, %device_n_unused: index, %device_k_unused: index):
    scf.for %m = %c0 to %c1 step %c1 {
      %0 = pim.access %arg0 offset %m sizes [1] strides[%c1] origin[%m]
          {id = 0 : i64} : memref<*xi8> -> !pim.access
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
