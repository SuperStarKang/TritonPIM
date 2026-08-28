// pim.access / pim.device / pim.place verifier가 말이 안 되는 조합을 거부하는지 확인.
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
