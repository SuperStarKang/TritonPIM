// 비-gemm 커널(elementwise add)이 재생성 없이 generic preserve 경로로 변환되는지 확인.
// build-pim-offload가 로드/저장 경계만 memory_region/transfer로 들어올리고, 계산 본문
// (reinterpret_cast/copy/to_tensor/linalg.generic/materialize)은 pim.compute region에 그대로
// 보존하며, linalg.generic에 pim.class="elementwise"를 태깅한다(DESIGN 4.1/10.6/11.1).
// RUN: triton-shared-opt --build-pim-offload %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @add_kernel
func.func @add_kernel(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>, %n: i32) {
  %c0 = arith.constant 0 : index
  // load A tile
  %ra = memref.reinterpret_cast %A to offset: [%c0], sizes: [32], strides: [1]
      : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
  %aa = memref.alloc() : memref<32xf32>
  memref.copy %ra, %aa : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32>
  %at = bufferization.to_tensor %aa restrict writable : memref<32xf32> to tensor<32xf32>
  // load B tile
  %rb = memref.reinterpret_cast %B to offset: [%c0], sizes: [32], strides: [1]
      : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
  %ab = memref.alloc() : memref<32xf32>
  memref.copy %rb, %ab : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32>
  %bt = bufferization.to_tensor %ab restrict writable : memref<32xf32> to tensor<32xf32>
  // elementwise add
  %z = tensor.empty() : tensor<32xf32>
  %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%at, %bt : tensor<32xf32>, tensor<32xf32>) outs(%z : tensor<32xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<32xf32>
  // store C tile
  %rc = memref.reinterpret_cast %C to offset: [%c0], sizes: [32], strides: [1]
      : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %sum in writable %rc
      : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
  return
}

// 경계 인식: A/B는 input, C는 store idiom 대상이라 output.
// CHECK: pim.memory_region %arg0 {{.*}}role = "input_activation"
// CHECK: pim.memory_region %arg1 {{.*}}role = "input_activation"
// CHECK: pim.memory_region %arg2 {{.*}}role = "output"
// CHECK: pim.transfer %{{.*}} {dst = "device_global", src = "host"}
// 본문은 재생성 없이 그대로 보존되고(reinterpret_cast/generic/materialize), generic은 태깅된다.
// CHECK: pim.compute ins(%{{.*}}, %{{.*}}) outs(%{{.*}}) using
// CHECK: memref.reinterpret_cast
// CHECK: linalg.generic {{.*}}pim.class = "elementwise"
// CHECK: arith.addf
// CHECK: bufferization.materialize_in_destination
// CHECK: pim.transfer %{{.*}} {dst = "host", src = "device_global"}
// CHECK: pim.launch @add_kernel
