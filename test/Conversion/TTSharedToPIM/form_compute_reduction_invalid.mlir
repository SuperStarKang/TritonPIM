// Common split reduction은 독립 contribution 하나를 zero accumulator에 더하는
// 단일-result 형태만 받는다.
// RUN: triton-shared-opt --split-input-file --pim-form-compute -verify-diagnostics %s

#identity = affine_map<(d0) -> (d0)>

func.func @accumulator_dependent_partial(%input: tensor<4xf32>, %count: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4xf32>)
      -> tensor<4xf32>
  %result = scf.for %k = %c0 to %count step %c1
      iter_args(%acc = %init) -> tensor<4xf32> {
    // expected-error @below {{the add contribution must not depend on the previous accumulator before it is split into Execution Tiles}}
    %next = linalg.generic {
        indexing_maps = [#identity, #identity],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%acc : tensor<4xf32>) {
    ^bb0(%value: f32, %old: f32):
      %dependent = arith.mulf %old, %value : f32
      %sum = arith.addf %old, %dependent : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>
    scf.yield %next : tensor<4xf32>
  }
  return
}

// -----

#identity = affine_map<(d0) -> (d0)>

func.func @multiple_reduction_results(%input: tensor<4xf32>, %count: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %empty0 = tensor.empty() : tensor<4xf32>
  %empty1 = tensor.empty() : tensor<4xf32>
  %init0 = linalg.fill ins(%zero : f32) outs(%empty0 : tensor<4xf32>)
      -> tensor<4xf32>
  %init1 = linalg.fill ins(%zero : f32) outs(%empty1 : tensor<4xf32>)
      -> tensor<4xf32>
  // expected-error @below {{a normalized PIM reduction loop must carry exactly one accumulator result}}
  %result:2 = scf.for %k = %c0 to %count step %c1
      iter_args(%acc = %init0, %side = %init1)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %next = linalg.generic {
        indexing_maps = [#identity, #identity],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xf32>) outs(%acc : tensor<4xf32>) {
    ^bb0(%value: f32, %old: f32):
      %sum = arith.addf %old, %value : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>
    scf.yield %next, %side : tensor<4xf32>, tensor<4xf32>
  }
  return
}
