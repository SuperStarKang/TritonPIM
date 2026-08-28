// RUN: triton-shared-opt --pim-verify -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", number_of_pim_devices = 1,
    number_of_banks_per_pim_device = 4,
    number_of_pim_cores_per_bank = 2,
    number_of_parallel_lanes_per_pim_core = 8,
    parallel_execution_kind = "simd",
    usable_bank_capacity_bytes = 1024,
    usable_local_capacity_bytes = 0>} {

// expected-error @below {{intra-device plan needs 8 banks but a PIM Device has 4}}
func.func @too_many_banks(%arg0: memref<*xi8>) attributes {
    pim.block = #pim.block<shape = [1, 0, 0]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
    pim.intra_device_plan = #pim.intra_device_plan<
        bank_partitions = [2, 4, 1], core_partitions = [1, 1, 1],
        lanes = 8, lane_axis = "n">} {
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
