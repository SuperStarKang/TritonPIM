// mask 있는 ⑦ gather/scatter. body가 scf.if{extract/store}(else{default})로 마스크가 들어있다.
// pim.gather는 mask+default clause, pim.scatter는 mask clause를 달고 변환된다.
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s --check-prefix=CLEAN

#map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @masked_element
func.func @masked_element(%A: memref<*xf32>, %C: memref<*xf32>, %gidx: tensor<64xi32>,
                          %sidx: tensor<64xi32>, %mask: tensor<64xi1>, %default: f32) {
  // masked gather: masked-out 원소는 %default로 채운다.
  %cast = memref.cast %A : memref<*xf32> to memref<?xf32>
  %src = bufferization.to_tensor %cast restrict : memref<?xf32> to tensor<?xf32>
  %empty = tensor.empty() : tensor<64xf32>
  %g = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%gidx, %mask : tensor<64xi32>, tensor<64xi1>) outs(%empty : tensor<64xf32>) {
  ^bb0(%off: i32, %m: i1, %o: f32):
    %r = scf.if %m -> (f32) {
      %ic = arith.index_cast %off : i32 to index
      %ex = tensor.extract %src[%ic] : tensor<?xf32>
      scf.yield %ex : f32
    } else {
      scf.yield %default : f32
    }
    linalg.yield %r : f32
  } -> tensor<64xf32>

  // masked scatter: masked-out 원소는 쓰지 않는다.
  %cast2 = memref.cast %C : memref<*xf32> to memref<?xf32>
  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%sidx, %g, %mask : tensor<64xi32>, tensor<64xf32>, tensor<64xi1>) {
  ^bb0(%off: i32, %v: f32, %m: i1):
    scf.if %m {
      %ic = arith.index_cast %off : i32 to index
      memref.store %v, %cast2[%ic] : memref<?xf32>
    }
    linalg.yield
  }
  return
}

// gather는 mask+default, scatter는 mask를 달고 pim op으로 올라간다.
// CHECK:      %[[G:.*]] = pim.gather %arg0 index %arg2 mask %arg4 : tensor<64xi1> other %arg5 : f32 : memref<*xf32>, tensor<64xi32> -> tensor<64xf32>
// CHECK:      pim.scatter %[[G]], %arg1 index %arg3 mask %arg4 : tensor<64xi1> : tensor<64xf32>, memref<*xf32>, tensor<64xi32>

// generic/scf.if/cast 관용구는 사라진다.
// CLEAN-LABEL: func.func @masked_element
// CLEAN-NOT:   linalg.generic
// CLEAN-NOT:   scf.if
// CLEAN-NOT:   memref.cast
// CLEAN:       return
