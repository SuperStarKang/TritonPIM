// 스칼라 저장: affine.store %scalar, %reinterpret_cast[0] (크기 1) → pim.access + pim.store(tensor<1xf32>).
// reduce/layernorm 의 mean/rstd 같은 한 칸짜리 결과가 이 형태다.
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s --check-prefix=CLEAN

// CHECK-LABEL: func.func @scalar_store
func.func @scalar_store(%C: memref<*xf32>, %i: index, %v: f32) {
  %rc = memref.reinterpret_cast %C to offset: [%i], sizes: [1], strides: [1]
      : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
  affine.store %v, %rc[0] : memref<1xf32, strided<[1], offset: ?>>
  return
}

// 스칼라를 tensor<1xf32>로 감싸 pim.store로 올린다. origin은 offset을 축으로 되돌린 것이라
// 크기 1 저장에서는 offset과 같은 값이다.
// CHECK:      %[[D:.*]] = pim.access %arg0 offset %arg1 sizes [1] strides[%{{.*}}] origin[%arg1] : memref<*xf32> -> !pim.access
// CHECK:      %[[T:.*]] = tensor.from_elements %arg2 : tensor<1xf32>
// CHECK:      pim.store %[[T]], %[[D]] : tensor<1xf32>, !pim.access

// affine.store 와 reinterpret_cast 는 사라진다.
// CLEAN-LABEL: func.func @scalar_store
// CLEAN-NOT:   affine.store
// CLEAN-NOT:   memref.reinterpret_cast
// CLEAN:       return
