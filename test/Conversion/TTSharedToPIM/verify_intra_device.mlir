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
    pim.intra_device_plan = #pim.intra_device_plan<
        bank_partitions = [2, 4, 1], core_partitions = [1, 1, 1],
        lanes = 8, lane_axis = "n">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = pim.access %arg0 offset %c0 sizes [1] strides[%c1] origin[%c0]
      {id = 0 : i64} : memref<*xi8> -> !pim.access
  return
}
}
