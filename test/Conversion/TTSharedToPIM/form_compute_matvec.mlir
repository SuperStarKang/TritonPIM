// Triton GEMV의 broadcast -> mul -> transpose -> reduce 연쇄는 linalg.matvec로
// 정규화하고, frontend가 이미 만든 linalg.matvec는 그대로 유지한다.
// RUN: triton-shared-opt --pim-form-compute -canonicalize %s | FileCheck %s

#broadcast = affine_map<(d0, d1) -> (0, d1)>
#identity2 = affine_map<(d0, d1) -> (d0, d1)>
#identity1 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @from_triton_pattern
// CHECK: %[[RESULT:.*]] = linalg.matvec ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<16xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NOT: linalg.generic
// CHECK-NOT: linalg.transpose
// CHECK-NOT: linalg.reduce
// CHECK: return %[[RESULT]]
func.func @from_triton_pattern(%A: tensor<8x16xf32>, %x: tensor<16xf32>)
    -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %initEmpty = tensor.empty() : tensor<8xf32>
  %init = linalg.fill ins(%cst : f32) outs(%initEmpty : tensor<8xf32>)
      -> tensor<8xf32>
  %result = scf.for %k = %c0 to %c1 step %c1
      iter_args(%acc = %init) -> tensor<8xf32> {
    %broadcastEmpty = tensor.empty() : tensor<8x16xf32>
    %productEmpty = tensor.empty() : tensor<8x16xf32>
    %transposeEmpty = tensor.empty() : tensor<16x8xf32>
    %reduceEmpty = tensor.empty() : tensor<8xf32>
    %partialInit = linalg.fill ins(%cst : f32)
        outs(%reduceEmpty : tensor<8xf32>) -> tensor<8xf32>
    %expanded = tensor.expand_shape %x [[0, 1]] output_shape [1, 16]
        : tensor<16xf32> into tensor<1x16xf32>
    %broadcasted = linalg.generic {
        indexing_maps = [#broadcast, #identity2],
        iterator_types = ["parallel", "parallel"]}
        ins(%expanded : tensor<1x16xf32>)
        outs(%broadcastEmpty : tensor<8x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x16xf32>
    %product = linalg.generic {
        indexing_maps = [#identity2, #identity2, #identity2],
        iterator_types = ["parallel", "parallel"]}
        ins(%A, %broadcasted : tensor<8x16xf32>, tensor<8x16xf32>)
        outs(%productEmpty : tensor<8x16xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %mul = arith.mulf %a, %b : f32
      linalg.yield %mul : f32
    } -> tensor<8x16xf32>
    %transposed = linalg.transpose
        ins(%product : tensor<8x16xf32>)
        outs(%transposeEmpty : tensor<16x8xf32>) permutation = [1, 0]
    %reduced = linalg.reduce ins(%transposed : tensor<16x8xf32>)
        outs(%partialInit : tensor<8xf32>) dimensions = [0]
        (%in: f32, %old: f32) {
      %sum = arith.addf %in, %old : f32
      linalg.yield %sum : f32
    }
    %accumulated = linalg.generic {
        indexing_maps = [#identity1, #identity1, #identity1],
        iterator_types = ["parallel"]}
        ins(%acc, %reduced : tensor<8xf32>, tensor<8xf32>)
        outs(%acc : tensor<8xf32>) {
    ^bb0(%old: f32, %partial: f32, %out: f32):
      %sum = arith.addf %old, %partial : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    scf.yield %accumulated : tensor<8xf32>
  }
  return %result : tensor<8xf32>
}

// CHECK-LABEL: func.func @already_matvec
// CHECK: %[[RESULT:.*]] = linalg.matvec ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<16xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK: return %[[RESULT]]
func.func @already_matvec(%A: tensor<8x16xf32>, %x: tensor<16xf32>)
    -> tensor<8xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<8xf32>)
      -> tensor<8xf32>
  %result = linalg.matvec
      ins(%A, %x : tensor<8x16xf32>, tensor<16xf32>)
      outs(%init : tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}
