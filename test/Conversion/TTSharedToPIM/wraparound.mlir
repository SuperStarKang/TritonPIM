// wrap-around 로드(두 reinterpret_cast + 두 subview-copy → 한 alloc)는 단일 pim.access로 표현 불가(§8).
// PIM에선 되돌려 감기가 다른 뱅크 데이터를 읽어 비효율적이므로, convert 단계에서 명확히 오류를 내
// 사용자가 마스크+채움 방식으로 커널을 고치게 한다. (두 copy가 각각 오류를 낸다.)
// RUN: triton-shared-opt --convert-ttshared-to-pim -verify-diagnostics %s

func.func @wrap_load(%A: memref<*xf32>, %o0: index, %o1: index, %s: index,
                     %v0: index, %v1: index) -> tensor<32x16xf32> {
  %pad = arith.constant 0.0 : f32
  // expected-error @+1 {{this memory access pattern is not supported on PIM}}
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
  // expected-error @+1 {{wrap-around load}}
  memref.copy %sv0, %d0
      : memref<?x16xf32, strided<[?, 1], offset: ?>> to memref<?x16xf32, strided<[16, 1]>>
  // expected-error @+1 {{wrap-around load}}
  memref.copy %sv1, %d1
      : memref<?x16xf32, strided<[?, 1], offset: ?>> to memref<?x16xf32, strided<[16, 1], offset: ?>>
  %t = bufferization.to_tensor %alloc restrict writable : memref<32x16xf32> to tensor<32x16xf32>
  return %t : tensor<32x16xf32>
}
