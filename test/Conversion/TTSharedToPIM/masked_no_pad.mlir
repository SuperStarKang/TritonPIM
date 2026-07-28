// 채움값(other) 없는 마스크 로드: reinterpret_cast + alloc + subview×2 + copy → pim.access{valid}(pad 없음) + load.
// linalg.fill 이 없어도 valid 만으로 올라가는지 확인 (fill 있는 경우는 masked.mlir 이 커버).
// vec_add/reduce/layernorm 의 가중치·입력 로드가 이 형태다.
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s --check-prefix=CLEAN

// CHECK-LABEL: func.func @masked_no_pad
func.func @masked_no_pad(%A: memref<*xf32>, %e: index) -> tensor<1024xf32> {
  %c0 = arith.constant 0 : index
  %rc = memref.reinterpret_cast %A to offset: [%c0], sizes: [1024], strides: [1]
      : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
  %buf = memref.alloc() : memref<1024xf32>
  %vsub = memref.subview %rc[0] [%e] [1]
      : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  %bsub = memref.subview %buf[0] [%e] [1]
      : memref<1024xf32> to memref<?xf32, strided<[1]>>
  memref.copy %vsub, %bsub
      : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
  %t = bufferization.to_tensor %buf restrict writable : memref<1024xf32> to tensor<1024xf32>
  return %t : tensor<1024xf32>
}

// 채움값이 없으므로 valid 만 있고 pad 는 없다.
// CHECK:      pim.access %arg0 offset %{{.*}} sizes [1024] strides[%{{.*}}] origin[%{{.*}}] valid[%arg1] : memref<*xf32>
// CHECK:      pim.load

// 버퍼 관용구는 전부 사라진다.
// CLEAN-LABEL: func.func @masked_no_pad
// CLEAN-NOT:   memref.alloc
// CLEAN-NOT:   memref.subview
// CLEAN-NOT:   memref.copy
// CLEAN-NOT:   bufferization.
// CLEAN:       return
