// device-axis-partitions 명시 분할: 자동 인수분해를 대체하고 explicit_k_partition 도
// 나눌 수 있다. 곱이 device 수를 넘으면 진단이다.
// RUN: triton-shared-opt "--pim-map-devices=device-axis-partitions=output_row_tile:2,output_column_tile:2,explicit_k_partition:2" %s | FileCheck %s
// RUN: triton-shared-opt "--pim-map-devices=device-axis-partitions=output_row_tile:64" -verify-diagnostics %s

// CHECK: pim.device
// CHECK: axes = ["output_row_tile", "output_column_tile", "explicit_k_partition"]
// CHECK: 2, 2, 2

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", number_of_pim_devices = 8,
    number_of_parallel_lanes_per_pim_core = 1,
    parallel_execution_kind = "multithread",
    usable_bank_capacity_bytes = 3072, usable_local_capacity_bytes = 49152>} {

// expected-error @below {{the explicit partitions need 64 PIM Devices but the machine has 8}}
func.func @three_axes(%tr: index, %tc: index, %tp: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %x = %c0 to %tr step %c1 {
    scf.for %y = %c0 to %tc step %c1 {
      scf.for %p = %c0 to %tp step %c1 {
      } {pim.axis = "explicit_k_partition"}
    } {pim.axis = "output_column_tile"}
  } {pim.axis = "output_row_tile"}
  return
}
}
