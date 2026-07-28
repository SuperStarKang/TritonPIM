// ⑤ masked 로드/저장이 pim.access{valid,pad}+load / pim.access{valid}+store로 올라가고
// alloc/fill/subview/scf.if/copy/extract_slice/materialize 관용구가 사라지는지 확인.
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s --check-prefix=CLEAN

// CHECK-LABEL: func.func @masked
func.func @masked(%A: memref<*xf32>, %C: memref<*xf32>, %e: index, %pad: f32) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index

  // 로드: 경계 안쪽만 copy, 나머지는 pad로 채움.
  %rc = memref.reinterpret_cast %A to offset: [%c0], sizes: [128], strides: [1]
      : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
  %buf = memref.alloc() : memref<128xf32>
  %cond = arith.cmpi slt, %e, %c128 : index
  scf.if %cond {
    linalg.fill ins(%pad : f32) outs(%buf : memref<128xf32>)
  }
  %vsub = memref.subview %rc[0] [%e] [1]
      : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  %bsub = memref.subview %buf[0] [%e] [1]
      : memref<128xf32> to memref<?xf32, strided<[1]>>
  memref.copy %vsub, %bsub
      : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
  %t = bufferization.to_tensor %buf restrict writable : memref<128xf32> to tensor<128xf32>

  // 저장: 경계 안쪽만 씀(pad 없음).
  %rc2 = memref.reinterpret_cast %C to offset: [%c0], sizes: [128], strides: [1]
      : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
  %es = tensor.extract_slice %t[0] [%e] [1] : tensor<128xf32> to tensor<?xf32>
  %vsub2 = memref.subview %rc2[0] [%e] [1]
      : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %es in writable %vsub2
      : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
  return
}

// 로드는 valid+pad clause를 달고 올라간다.
// CHECK:      pim.access %arg0 offset %{{.*}} sizes [128] strides[%{{.*}}] origin[%{{.*}}] valid[%arg2] pad %arg3
// CHECK:      pim.load
// 저장은 valid clause만(채움값 없음).
// CHECK:      pim.access %arg1 offset %{{.*}} sizes [128] strides[%{{.*}}] origin[%{{.*}}] valid[%arg2]
// CHECK:      pim.store

// 버퍼 관용구는 전부 사라진다.
// CLEAN-LABEL: func.func @masked
// CLEAN-NOT:   memref.alloc
// CLEAN-NOT:   memref.subview
// CLEAN-NOT:   linalg.fill
// CLEAN-NOT:   scf.if
// CLEAN-NOT:   bufferization.
// CLEAN:       return
