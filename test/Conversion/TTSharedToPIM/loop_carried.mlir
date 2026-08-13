// ② offset이 iter_arg로 전진하는 루프에서도 같은 3규칙으로 올라가고 scf.for는 보존되는지 확인.
// 같은 reinterpret_cast를 load와 store가 함께 쓰므로 하나의 pim.access를 공유한다.
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s --check-prefix=CLEAN

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @advance
func.func @advance(%A: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c256 = arith.constant 256 : index
  %one = arith.constant 1.0 : f32
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%off = %c0) -> (index) {
    %rc = memref.reinterpret_cast %A to offset: [%off], sizes: [256], strides: [1]
        : memref<*xf32> to memref<256xf32, strided<[1], offset: ?>>
    %buf = memref.alloc() : memref<256xf32>
    memref.copy %rc, %buf : memref<256xf32, strided<[1], offset: ?>> to memref<256xf32>
    %t = bufferization.to_tensor %buf restrict writable : memref<256xf32> to tensor<256xf32>
    %z = tensor.empty() : tensor<256xf32>
    %add = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
        ins(%t : tensor<256xf32>) outs(%z : tensor<256xf32>) {
    ^bb0(%x: f32, %o: f32):
      %s = arith.addf %x, %one : f32
      linalg.yield %s : f32
    } -> tensor<256xf32>
    bufferization.materialize_in_destination %add in writable %rc
        : (tensor<256xf32>, memref<256xf32, strided<[1], offset: ?>>) -> ()
    %next = arith.addi %off, %c256 : index
    scf.yield %next : index
  }
  return
}

// 루프는 보존되고, offset(iter_arg) 위에서 load/store가 올라간다.
// CHECK:      scf.for %{{.*}} iter_args(%[[OFF:.*]] = %{{.*}})
// CHECK:      pim.access %arg0 offset %[[OFF]] sizes [256] strides
// CHECK:      pim.load
// CHECK:      pim.store

// 버퍼 관용구는 사라진다.
// CLEAN-LABEL: func.func @advance
// CLEAN-NOT:   memref.
// CLEAN-NOT:   bufferization.
// CLEAN:       return
