// 주소를 loop iter_arg로 나르는 형태를 offset(i) = initial + i*advance로 다시 쓴다.
// RUN: triton-shared-opt --pim-normalize-address %s | FileCheck %s

// CHECK-LABEL: func.func @advance
func.func @advance(%A: memref<*xf32>, %n: i32, %adv: index, %init: index) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  scf.for %k = %c0_i32 to %n step %c1_i32 iter_args(%off = %init) -> (index) : i32 {
    %rc = memref.reinterpret_cast %A to offset: [%off], sizes: [8], strides: [%c1]
        : memref<*xf32> to memref<8xf32, strided<[?], offset: ?>>
    %v = memref.load %rc[%c0] : memref<8xf32, strided<[?], offset: ?>>
    %next = arith.addi %off, %adv : index
    scf.yield %next : index
  }
  return
}

// 반복 번호는 (유도변수 - 하한) / step 이고, 주소는 그 번호로 다시 만든다.
// CHECK:      scf.for %[[K:.*]] = %{{.*}} to %arg1
// CHECK:        %[[S:.*]] = arith.subi %[[K]], %{{.*}}
// CHECK:        %[[I:.*]] = arith.divsi %[[S]], %{{.*}}
// CHECK:        %[[IC:.*]] = arith.index_cast %[[I]]
// CHECK:        %[[M:.*]] = arith.muli %[[IC]], %arg2
// CHECK:        %[[OFF:.*]] = arith.addi %arg3, %[[M]]
// CHECK:        memref.reinterpret_cast %arg0 to offset: [%[[OFF]]]
