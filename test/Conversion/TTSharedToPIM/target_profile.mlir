// RUN: triton-shared-opt "--pim-set-target-profile=target=aimx num-units=2 banks-per-unit=256 cores-per-bank=1 lanes-per-core=16 bank-capacity-limit-bytes=268435456 local-capacity-limit-bytes=0" %s | FileCheck %s

// CHECK: #pim.target_profile<target = "aimx"
// CHECK-SAME: num_units = 2
// CHECK-SAME: banks_per_unit = 256
// CHECK-SAME: cores_per_bank = 1
// CHECK-SAME: lanes_per_core = 16
// CHECK-SAME: bank_capacity_limit_bytes = 268435456
// CHECK-SAME: local_capacity_limit_bytes = 0
// CHECK-NOT: execution_unit
// CHECK-NOT: lane_mode

module {
}
