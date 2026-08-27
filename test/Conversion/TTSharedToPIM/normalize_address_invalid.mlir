// 닫힌 형태로 못 만드는 주소 재귀는 진단을 낸다. 조용히 두면 배치가 통째로 사라진다.
// RUN: triton-shared-opt --pim-normalize-address -verify-diagnostics %s

func.func @not_closed(%A: memref<*xf32>, %n: i32, %adv: index) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  // expected-error @+3 {{cannot be put in closed form}}
  // expected-note @+2 {{PIM placement needs offset(i) = initial + i*advance}}
  // expected-note @+1 {{index with an affine expression of the loop variable}}
  scf.for %k = %c0_i32 to %n step %c1_i32 iter_args(%off = %c0) -> (index) : i32 {
    %rc = memref.reinterpret_cast %A to offset: [%off], sizes: [8], strides: [%c1]
        : memref<*xf32> to memref<8xf32, strided<[?], offset: ?>>
    %v = memref.load %rc[%c0] : memref<8xf32, strided<[?], offset: ?>>
    %next = arith.muli %off, %adv : index
    scf.yield %next : index
  }
  return
}
