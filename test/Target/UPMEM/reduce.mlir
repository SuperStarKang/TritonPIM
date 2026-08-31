// linalg.reduce의 dimensions와 scalar add region이 그대로 DPU C loop가 된다.
// RUN: triton-shared-translate --upmem-to-device %s | FileCheck %s

// CHECK: int main()
// CHECK: for (int32_t _r0 = 0; _r0 < 2; ++_r0) {
// CHECK-NEXT: for (int32_t _r1 = 0; _r1 < 4; ++_r1) {
// CHECK: int32_t {{.*}} = w0[(_r0 * 4 + _r1)] + w1[_r0];
// CHECK: w1[_r0] = {{.*}};

upmem.device_func @reduce attributes {upmem.lanes = 1 : i64} {
  %zero = arith.constant 0 : i32
  %input = memref.alloc() : memref<2x4xi32, #upmem<space wram>>
  %output = memref.alloc() : memref<2xi32, #upmem<space wram>>
  linalg.fill ins(%zero : i32)
      outs(%output : memref<2xi32, #upmem<space wram>>)
  linalg.reduce ins(%input : memref<2x4xi32, #upmem<space wram>>)
      outs(%output : memref<2xi32, #upmem<space wram>>)
      dimensions = [1]
      (%value: i32, %old: i32) {
    %sum = arith.addi %value, %old : i32
    linalg.yield %sum : i32
  }
}
