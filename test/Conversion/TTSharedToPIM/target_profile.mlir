// RUN: triton-shared-opt "--pim-set-target-profile=target=aimx number-of-pim-devices=2 number-of-banks-per-pim-device=256 number-of-pim-cores-per-bank=1 number-of-parallel-lanes-per-pim-core=16 parallel-execution-kind=simd usable-bank-capacity-bytes=268435456 usable-local-capacity-bytes=0" %s | FileCheck %s

// CHECK: #pim.target_profile<target = "aimx"
// CHECK-SAME: number_of_pim_devices = 2
// CHECK-SAME: number_of_banks_per_pim_device = 256
// CHECK-SAME: number_of_pim_cores_per_bank = 1
// CHECK-SAME: number_of_parallel_lanes_per_pim_core = 16
// CHECK-SAME: usable_bank_capacity_bytes = 268435456
// CHECK-SAME: usable_local_capacity_bytes = 0
// CHECK-NOT: execution_unit
// CHECK-NOT: lane_mode

module {
}
