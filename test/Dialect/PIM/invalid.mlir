// pim.access / pim.device / pim.place / pim.combine verifier가 말이 안 되는 조합을
// 거부하는지 확인.
// RUN: triton-shared-opt --split-input-file -verify-diagnostics %s

// expected-error @+1 {{axis must be m, n, or k}}
module attributes {test.axis = #pim.axis<"x">} {
}

// -----

// 축별 stride 개수는 타일 rank와 같아야 한다.
func.func @bad_strides(%A: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{strides count must equal sizes rank}}
  %d = pim.access %A offset %c0 sizes [4, 4] strides [%c1] {id = 0 : i64}
      : memref<*xf32> -> !pim.access
  return
}

// -----

// valid를 쓰면 축별로 하나씩이어야 한다.
func.func @bad_valid(%A: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{valid count must equal sizes rank}}
  %d = pim.access %A offset %c0 sizes [4, 4] strides [%c1, %c1] valid [%c1] {id = 0 : i64}
      : memref<*xf32> -> !pim.access
  return
}

// -----

// 채움값은 mask와 함께만 의미가 있다.
func.func @pad_without_valid(%A: memref<*xf32>, %f: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{pad requires valid}}
  %d = pim.access %A offset %c0 sizes [4] strides [%c1] pad %f : f32 {id = 0 : i64}
      : memref<*xf32> -> !pim.access
  return
}

// -----

// block argument는 항상 [m,n,k] 좌표 세 개다.
func.func @device_arity(%e: index) {
  // expected-error @+1 {{body must take the [m, n, k] unit coordinates}}
  pim.device #pim.device_mapping<partitions = [2, 4, 1]> {
  ^bb0(%px: index):
  }
  return
}

// -----

// 네 리스트가 같은 rank를 기술해야 한다.
func.func @place_rank(%A: memref<*xi8>, %e: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> {
  ^bb0(%px: index, %device_n_unused: index, %device_k_unused: index):
    // expected-error @+1 {{start, size, allocation_shape, and strides must have the same rank}}
    pim.place %A start[%c0, %c0] size[%c1] allocation_shape[%c1] strides[%c1]
        {direction = "in", access = 0 : i64} : memref<*xi8>
  }
  return
}

// -----

// Common IR이 아직 정규화하지 않는 combiner는 target으로 넘기지 않는다.
func.func @unsupported_combine(%C: memref<*xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<1xi32>
  %access = pim.access %C offset %c0 sizes [1] strides[%c1] origin[%c0]
      {id = 0 : i64} : memref<*xi32> -> !pim.access
  // expected-error @+1 {{currently supports only the normalized add combiner}}
  pim.combine %empty into %access kind "max" : tensor<1xi32>, !pim.access
  return
}

// -----

// split-K partial과 final output tile은 같은 M/N 논리 shape를 가져야 한다.
func.func @combine_rank(%C: memref<*xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<2x2xi32>
  %access = pim.access %C offset %c0 sizes [4] strides[%c1] origin[%c0]
      {id = 0 : i64} : memref<*xi32> -> !pim.access
  // expected-error @+1 {{partial rank must equal the final destination access rank}}
  pim.combine %empty into %access kind "add" : tensor<2x2xi32>, !pim.access
  return
}

// -----

// 같은 rank라도 partial과 final destination의 tile shape가 다르면 안 된다.
func.func @combine_shape(%C: memref<*xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<2xi32>
  %access = pim.access %C offset %c0 sizes [4] strides[%c1] origin[%c0]
      {id = 0 : i64} : memref<*xi32> -> !pim.access
  // expected-error @+1 {{partial shape must equal the final destination access shape}}
  pim.combine %empty into %access kind "add" : tensor<2xi32>, !pim.access
  return
}

// -----

// destination은 Target이 추측하는 handle이 아니라 명시적인 최종 pim.access다.
func.func @combine_destination(%destination: !pim.access) {
  %empty = tensor.empty() : tensor<1xi32>
  // expected-error @+1 {{destination must be defined by pim.access}}
  pim.combine %empty into %destination kind "add" : tensor<1xi32>, !pim.access
  return
}
