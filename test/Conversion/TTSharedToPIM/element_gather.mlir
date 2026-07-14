// ⑦ 원소별(완전 random) gather/scatter가 pim.gather / pim.scatter로 올라가고
// memref.cast + to_tensor + linalg.generic(tensor.extract / memref.store) 관용구가 사라지는지 확인.
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s --check-prefix=CLEAN

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @element
func.func @element(%A: memref<*xf32>, %C: memref<*xf32>, %gidx: tensor<64xi32>, %sidx: tensor<64xi32>) {
  // gather: 각 인덱스 위치에서 원소를 읽는다.
  %cast = memref.cast %A : memref<*xf32> to memref<?xf32>
  %src = bufferization.to_tensor %cast restrict : memref<?xf32> to tensor<?xf32>
  %empty = tensor.empty() : tensor<64xf32>
  %g = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%gidx : tensor<64xi32>) outs(%empty : tensor<64xf32>) {
  ^bb0(%off: i32, %o: f32):
    %ic = arith.index_cast %off : i32 to index
    %ex = tensor.extract %src[%ic] : tensor<?xf32>
    linalg.yield %ex : f32
  } -> tensor<64xf32>

  // scatter: 각 인덱스 위치에 원소를 쓴다.
  %cast2 = memref.cast %C : memref<*xf32> to memref<?xf32>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%sidx, %g : tensor<64xi32>, tensor<64xf32>) {
  ^bb0(%off: i32, %v: f32):
    %ic = arith.index_cast %off : i32 to index
    memref.store %v, %cast2[%ic] : memref<?xf32>
    linalg.yield
  }
  return
}

// gather/scatter 전용 op으로 올라간다.
// CHECK:      %[[G:.*]] = pim.gather %arg0 index %arg2 : memref<*xf32>, tensor<64xi32> -> tensor<64xf32>
// CHECK:      pim.scatter %[[G]], %arg1 index %arg3 : tensor<64xf32>, memref<*xf32>, tensor<64xi32>

// generic/cast/to_tensor 관용구는 사라진다.
// CLEAN-LABEL: func.func @element
// CLEAN-NOT:   linalg.generic
// CLEAN-NOT:   memref.cast
// CLEAN-NOT:   bufferization.
// CLEAN:       return
