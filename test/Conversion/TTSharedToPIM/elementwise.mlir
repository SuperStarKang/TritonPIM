// 벡터 덧셈 커널(①strided + store)의 메모리 접근이 pim.access/load/store로 올라가고
// staging 버퍼(alloc/copy/to_tensor/materialize)가 사라지는지 확인. 계산(generic)은 유지되고
// pim.op 태그만 붙는다.
// RUN: triton-shared-opt --convert-ttshared-to-pim --pim-classify-compute %s | FileCheck %s
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s --check-prefix=CLEAN

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @add_kernel
func.func @add_kernel(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>, %n: i32) {
  %c0 = arith.constant 0 : index
  %ra = memref.reinterpret_cast %A to offset: [%c0], sizes: [32], strides: [1]
      : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
  %aa = memref.alloc() : memref<32xf32>
  memref.copy %ra, %aa : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32>
  %at = bufferization.to_tensor %aa restrict writable : memref<32xf32> to tensor<32xf32>
  %rb = memref.reinterpret_cast %B to offset: [%c0], sizes: [32], strides: [1]
      : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
  %ab = memref.alloc() : memref<32xf32>
  memref.copy %rb, %ab : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32>
  %bt = bufferization.to_tensor %ab restrict writable : memref<32xf32> to tensor<32xf32>
  %z = tensor.empty() : tensor<32xf32>
  %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%at, %bt : tensor<32xf32>, tensor<32xf32>) outs(%z : tensor<32xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<32xf32>
  %rc = memref.reinterpret_cast %C to offset: [%c0], sizes: [32], strides: [1]
      : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %sum in writable %rc
      : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
  return
}

// 주소/전송으로 올라간다.
// CHECK:      pim.access %arg0 offset %{{.*}} sizes [32] strides
// CHECK:      pim.load
// CHECK:      pim.access %arg1
// CHECK:      pim.load
// CHECK:      linalg.generic
// CHECK-SAME: pim.op = "elementwise"
// CHECK:      pim.access %arg2
// CHECK:      pim.store

// 버퍼 관용구는 마지막 store 이전(=본문 전체)에서 사라진다.
// CLEAN-LABEL: func.func @add_kernel
// CLEAN-NOT:   memref.
// CLEAN-NOT:   bufferization.
// CLEAN:       pim.store
