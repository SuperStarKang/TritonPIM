// ⑥ row-gather(1축만 인덱스, 각 행은 연속) 로드가 pim.access{gather_dim,index}+load로 올라가고
// scf.for + tensor.extract + 행별 reinterpret_cast + subview + copy 관용구가 사라지는지 확인.
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s --check-prefix=CLEAN

// CHECK-LABEL: func.func @row_gather
func.func @row_gather(%A: memref<*xf32>, %C: memref<*xf32>, %idx: tensor<8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %alloc = memref.alloc() : memref<8x8xf32>
  scf.for %i = %c0 to %c8 step %c1 {
    %e = tensor.extract %idx[%i] : tensor<8xi32>
    %ic = arith.index_cast %e : i32 to index
    %rc = memref.reinterpret_cast %A to offset: [%ic], sizes: [1, 8], strides: [1, 1]
        : memref<*xf32> to memref<1x8xf32, strided<[1, 1], offset: ?>>
    %sub = memref.subview %alloc[%i, 0] [1, 8] [1, 1]
        : memref<8x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
    memref.copy %rc, %sub
        : memref<1x8xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[8, 1], offset: ?>>
  }
  %t = bufferization.to_tensor %alloc restrict writable : memref<8x8xf32> to tensor<8x8xf32>
  // 모은 타일을 그대로 저장(①).
  %rc2 = memref.reinterpret_cast %C to offset: [%c0], sizes: [8, 8], strides: [8, 1]
      : memref<*xf32> to memref<8x8xf32, strided<[8, 1], offset: ?>>
  bufferization.materialize_in_destination %t in writable %rc2
      : (tensor<8x8xf32>, memref<8x8xf32, strided<[8, 1], offset: ?>>) -> ()
  return
}

// gather 축(0)과 인덱스 텐서를 clause로 달고 올라간다.
// CHECK:      pim.access %arg0 offset %{{.*}} sizes [8, 8] strides[%{{.*}}, %{{.*}}] gather 0 index %arg2
// CHECK:      pim.load
// CHECK:      pim.access %arg1
// CHECK:      pim.store

// 루프와 행별 관용구는 사라진다.
// CLEAN-LABEL: func.func @row_gather
// CLEAN-NOT:   scf.for
// CLEAN-NOT:   memref.alloc
// CLEAN-NOT:   memref.subview
// CLEAN-NOT:   tensor.extract
// CLEAN-NOT:   bufferization.
// CLEAN:       return
