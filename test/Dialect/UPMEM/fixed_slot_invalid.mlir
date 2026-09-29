// UPMEM stage의 결과도 Common allocation_shape를 옮긴 정적 fixed slot이어야 한다.
// RUN: not triton-shared-opt %s 2>&1 | FileCheck %s

// CHECK: 'upmem.stage' op fixed-slot result must have a static shape

module {
func.func @dynamic_stage(%A: memref<*xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %pool = memref.alloc() : memref<1x8xi8>
  %stage = upmem.stage %A origin[%c0] sizes[%c8] strides[%c1]
      pad_to[%c8] tile [8] into %pool
      : memref<*xi8>, memref<1x8xi8> -> memref<?xi8>
  return
}
}
