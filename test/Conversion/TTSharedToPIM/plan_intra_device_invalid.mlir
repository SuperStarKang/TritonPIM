// Invalid intra-device axis names are rejected before target lowering.
// RUN: triton-shared-opt --pim-plan-intra-device=lane-axis=diagonal -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", number_of_pim_devices = 8,
    number_of_banks_per_pim_device = 1,
    number_of_pim_cores_per_bank = 1,
    number_of_parallel_lanes_per_pim_core = 16,
    parallel_execution_kind = "multithread",
    usable_bank_capacity_bytes = 3072, usable_local_capacity_bytes = 4096>} {

// expected-error @+1 {{lane-axis must be m, n, or k}}
func.func @matmul(%A: memref<*xi8>, %lda: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %da = pim.access %A offset %c0 sizes [8, 32] strides[%lda, %c1] origin[%c0, %c0]
        {id = 0 : i64} : memref<*xi8> -> !pim.access
  %ta = pim.load %da : !pim.access -> tensor<8x32xi8>
  return
}
}
