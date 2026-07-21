// wrap-around 로드: 두 reinterpret_cast + 두 subview-copy가 한 alloc으로 들어간다(§8 표 "추후 지원").
// 하나의 pim.access로 표현 불가하므로 변환하지 않고 관용구를 그대로 둔다.
// (과거엔 ⑤ masked walk가 두 copy를 각각 매치해 공유 alloc/fill/to_tensor를 이중 erase → double free 크래시.)
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s

// CHECK-LABEL: func.func @wrap_load
func.func @wrap_load(%A: memref<*xf32>, %o0: index, %o1: index, %s: index,
                     %v0: index, %v1: index) -> tensor<32x16xf32> {
  %pad = arith.constant 0.0 : f32
  %rc0 = memref.reinterpret_cast %A to offset: [%o0], sizes: [%v0, 16], strides: [%s, 1]
      : memref<*xf32> to memref<?x16xf32, strided<[?, 1], offset: ?>>
  %rc1 = memref.reinterpret_cast %A to offset: [%o1], sizes: [%v1, 16], strides: [%s, 1]
      : memref<*xf32> to memref<?x16xf32, strided<[?, 1], offset: ?>>
  %alloc = memref.alloc() : memref<32x16xf32>
  linalg.fill ins(%pad : f32) outs(%alloc : memref<32x16xf32>)
  %sv0 = memref.subview %rc0[0, 0] [%v0, 16] [1, 1]
      : memref<?x16xf32, strided<[?, 1], offset: ?>> to memref<?x16xf32, strided<[?, 1], offset: ?>>
  %sv1 = memref.subview %rc1[0, 0] [%v1, 16] [1, 1]
      : memref<?x16xf32, strided<[?, 1], offset: ?>> to memref<?x16xf32, strided<[?, 1], offset: ?>>
  %d0 = memref.subview %alloc[0, 0] [%v0, 16] [1, 1]
      : memref<32x16xf32> to memref<?x16xf32, strided<[16, 1]>>
  %d1 = memref.subview %alloc[%v0, 0] [%v1, 16] [1, 1]
      : memref<32x16xf32> to memref<?x16xf32, strided<[16, 1], offset: ?>>
  memref.copy %sv0, %d0
      : memref<?x16xf32, strided<[?, 1], offset: ?>> to memref<?x16xf32, strided<[16, 1]>>
  memref.copy %sv1, %d1
      : memref<?x16xf32, strided<[?, 1], offset: ?>> to memref<?x16xf32, strided<[16, 1], offset: ?>>
  %t = bufferization.to_tensor %alloc restrict writable : memref<32x16xf32> to tensor<32x16xf32>
  return %t : tensor<32x16xf32>
}

// 변환하지 않으므로 관용구가 그대로 남고 pim.access/load는 생기지 않는다(크래시 없이).
// CHECK:     memref.copy
// CHECK:     memref.copy
// CHECK-NOT: pim.access
// CHECK-NOT: pim.load
