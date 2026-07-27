// --pim-classify-compute가 각 연산에 target-무관 종류를 pim.op로 다는지 확인.
// 입력은 raw linalg(태그 없음), 출력은 종류 태그가 붙은 것.
// RUN: triton-shared-opt --pim-classify-compute %s | FileCheck %s

#id1  = affine_map<(d0) -> (d0)>
#id2  = affine_map<(d0, d1) -> (d0, d1)>
#proj = affine_map<(d0, d1) -> (d0)>       // d1 drop = broadcast
#red  = affine_map<(d0, d1) -> (d0)>       // 출력이 d1을 접음(reduction)

// CHECK-LABEL: func.func @classify
func.func @classify(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %v: tensor<4xf32>,
                    %x: tensor<32xf32>, %y: tensor<32xf32>, %s: f32) {
  // matmul → gemm
  %z0 = tensor.empty() : tensor<4x16xf32>
  // CHECK: linalg.matmul
  // CHECK-SAME: pim.op = "gemm"
  %mm = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%z0 : tensor<4x16xf32>) -> tensor<4x16xf32>

  // fill → fill
  %z1 = tensor.empty() : tensor<32xf32>
  // CHECK: linalg.fill
  // CHECK-SAME: pim.op = "fill"
  %fl = linalg.fill ins(%s : f32) outs(%z1 : tensor<32xf32>) -> tensor<32xf32>

  // 전부 parallel + identity map → elementwise
  %z2 = tensor.empty() : tensor<32xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: pim.op = "elementwise"
  %ew = linalg.generic {indexing_maps = [#id1, #id1, #id1], iterator_types = ["parallel"]}
      ins(%x, %y : tensor<32xf32>, tensor<32xf32>) outs(%z2 : tensor<32xf32>) {
  ^bb0(%p: f32, %q: f32, %o: f32):
    %r = arith.addf %p, %q : f32
    linalg.yield %r : f32
  } -> tensor<32xf32>

  // 입력 맵이 차원 drop → broadcast
  %z3 = tensor.empty() : tensor<4x8xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: pim.op = "broadcast"
  %bc = linalg.generic {indexing_maps = [#proj, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%v : tensor<4xf32>) outs(%z3 : tensor<4x8xf32>) {
  ^bb0(%p: f32, %o: f32):
    linalg.yield %p : f32
  } -> tensor<4x8xf32>

  // body에 linalg.index → index_generation
  %z4 = tensor.empty() : tensor<32xi32>
  // CHECK: linalg.generic
  // CHECK-SAME: pim.op = "index_generation"
  %gn = linalg.generic {indexing_maps = [#id1], iterator_types = ["parallel"]}
      outs(%z4 : tensor<32xi32>) {
  ^bb0(%o: i32):
    %i = linalg.index 0 : index
    %ic = arith.index_cast %i : index to i32
    linalg.yield %ic : i32
  } -> tensor<32xi32>

  // reduction iterator → reduce (broadcast보다 먼저 판별)
  %z5 = tensor.empty() : tensor<4xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: pim.op = "reduce"
  %rd = linalg.generic {indexing_maps = [#id2, #red], iterator_types = ["parallel", "reduction"]}
      ins(%a : tensor<4x8xf32>) outs(%z5 : tensor<4xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %s2 = arith.addf %in, %acc : f32
    linalg.yield %s2 : f32
  } -> tensor<4xf32>

  return
}
