// device-axis-partitions 명시 분할: 자동 인수분해를 대체하고 k 도
// 나눌 수 있다. 곱이 device 수를 넘으면 진단이다.
// RUN: triton-shared-opt "--pim-map-devices=device-axis-partitions=m:2,n:2,k:2" %s | FileCheck %s
// RUN: triton-shared-opt "--pim-map-devices=device-axis-partitions=m:64" -verify-diagnostics %s

// CHECK: pim.device
// CHECK: axes = ["m", "n", "k"]
// CHECK: 2, 2, 2

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", number_of_pim_devices = 8,
    number_of_banks_per_pim_device = 1,
    number_of_pim_cores_per_bank = 1,
    number_of_parallel_lanes_per_pim_core = 1,
    parallel_execution_kind = "multithread",
    usable_bank_capacity_bytes = 3072, usable_local_capacity_bytes = 49152>} {

// expected-error @below {{the explicit partitions need 64 PIM Devices but the machine has 8}}
func.func @three_axes(%tr: index, %tc: index, %tp: index)
    attributes {pim.execution_tile = #pim.execution_tile<m = 0, n = 0, k = 0>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %x = %c0 to %tr step %c1 {
    scf.for %y = %c0 to %tc step %c1 {
      scf.for %p = %c0 to %tp step %c1 {
      } {pim.axis = #pim.axis<"k">}
    } {pim.axis = #pim.axis<"n">}
  } {pim.axis = #pim.axis<"m">}
  return
}
}
