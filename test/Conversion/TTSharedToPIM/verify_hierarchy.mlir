// RUN: triton-shared-opt --pim-verify -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", number_of_pim_devices = 1,
    number_of_banks_per_pim_device = 1,
    number_of_pim_cores_per_bank = 1,
    number_of_parallel_lanes_per_pim_core = 1,
    parallel_execution_kind = "multithread",
    usable_bank_capacity_bytes = 1024,
    usable_local_capacity_bytes = 0>} {

// expected-error @below {{#pim.block × #pim.block_group gives 2 elements on m but the local compute tile uses 1}}
func.func @inconsistent_group_shape(%arg0: memref<*xi8>) attributes {
    pim.block = #pim.block<shape = [2, 0, 0]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
    pim.intra_device_plan = #pim.intra_device_plan<
        bank_partitions = [1, 1, 1], core_partitions = [1, 1, 1],
        lanes = 1, lane_axis = "m">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pim.device #pim.device_mapping<axes = ["m"], partitions = [1]>
      extents[%c1] attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 0, 0]>} {
  ^bb0(%device_m: index):
    scf.for %m = %c0 to %c1 step %c1 {
      %0 = pim.access %arg0 offset %m sizes [1] strides[%c1] origin[%m]
          {id = 0 : i64} : memref<*xi8> -> !pim.access
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
