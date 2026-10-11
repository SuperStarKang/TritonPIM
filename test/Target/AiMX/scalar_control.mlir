// Integer induction variables and integer/index loop state stay control values.
// RUN: triton-shared-translate --aimx-to-descriptors %s | FileCheck %s
// CHECK-LABEL: static uint32_t aimx_prog_scalar_control(
// CHECK-SAME: uint64_t [[A:v[0-9]+]], uint64_t [[B:v[0-9]+]]) {
// CHECK: int64_t [[ZERO:v[0-9]+]] = 0;
// CHECK: int64_t [[STRIDE:v[0-9]+]] = 8192;
// CHECK: int32_t [[ZERO32:v[0-9]+]] = 0;
// CHECK: int32_t [[ONE32:v[0-9]+]] = 1;
// CHECK: int32_t [[TWO32:v[0-9]+]] = 2;
// CHECK: int32_t [[LOW:v[0-9]+]] = 2147483646;
// CHECK: int32_t [[HIGH:v[0-9]+]] = -2147483646;
// CHECK: int64_t [[LEFT:v[0-9]+]] = [[ZERO]];
// CHECK-NEXT: int64_t [[RIGHT:v[0-9]+]] = [[STRIDE]];
// CHECK-NEXT: for (int32_t [[IV:v[0-9]+]] = [[ZERO32]]; [[IV]] < [[TWO32]]; [[IV]] += [[ONE32]]) {
// CHECK: PIM_RD_INSTR_MC(&idx, next, ([[A]] + (uint64_t)[[LEFT]]), ([[B]] + (uint64_t)[[RIGHT]]), 8192u, 0x0u,
// CHECK-NEXT: int64_t [[Y0:_yield[0-9]+]] = [[RIGHT]];
// CHECK-NEXT: int64_t [[Y1:_yield[0-9]+]] = [[LEFT]];
// CHECK-NEXT: [[LEFT]] = [[Y0]];
// CHECK-NEXT: [[RIGHT]] = [[Y1]];
// CHECK-NEXT: }
// Zero-trip results are initialized before testing the loop condition.
// CHECK-NEXT: int64_t [[SAVEDLEFT:v[0-9]+]] = [[LEFT]];
// CHECK-NEXT: int64_t [[SAVEDRIGHT:v[0-9]+]] = [[RIGHT]];
// CHECK-NEXT: for (int32_t [[EMPTY:v[0-9]+]] = [[ONE32]]; [[EMPTY]] < [[ZERO32]]; [[EMPTY]] += [[ONE32]]) {
// CHECK: PIM_RD_INSTR_MC(&idx, next, ([[A]] + (uint64_t)[[SAVEDLEFT]]), ([[B]] + (uint64_t)[[SAVEDRIGHT]]), 8192u, 0x0u,
// Crossing the sign bit requires unsigned comparison and unsigned addition.
// CHECK: for (int32_t [[UNSIGNED:v[0-9]+]] = [[LOW]]; (uint32_t)[[UNSIGNED]] < (uint32_t)[[HIGH]]; [[UNSIGNED]] = (int32_t)((uint32_t)[[UNSIGNED]] + (uint32_t)[[ONE32]])) {
// CHECK: PIM_RD_INSTR_MC
// CHECK: return idx;

module {
  aimx.descriptor_program @scalar_control batch 1 {
  ^bb0(%a: !aimx.buffer, %b: !aimx.buffer):
    %zero = arith.constant 0 : index
    %stride = arith.constant 8192 : index
    %zero32 = arith.constant 0 : i32
    %one32 = arith.constant 1 : i32
    %two32 = arith.constant 2 : i32
    %low = arith.constant 2147483646 : i32
    %high = arith.constant -2147483646 : i32
    %left, %right = scf.for %iv = %zero32 to %two32 step %one32
        iter_args(%x = %zero, %y = %stride) -> (index, index) : i32 {
      aimx.desc src(%a, %x) dst(%b, %y) len 8192 opcode copy
      scf.yield %y, %x : index, index
    }
    %savedLeft, %savedRight = scf.for %empty = %one32 to %zero32 step %one32
        iter_args(%x = %left, %y = %right) -> (index, index) : i32 {
      scf.yield %zero, %zero : index, index
    }
    aimx.desc src(%a, %savedLeft) dst(%b, %savedRight) len 8192 opcode copy
    scf.for unsigned %iv = %low to %high step %one32 : i32 {
      aimx.desc src(%a, %zero) dst(%b, %stride) len 8192 opcode copy
    }
  }
}
