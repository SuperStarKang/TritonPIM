// pim.access / pim.gather verifier가 말이 안 되는 clause 조합을 거부하는지 확인.
// RUN: triton-shared-opt --split-input-file -verify-diagnostics %s

// R1: strides 개수 != sizes rank
func.func @bad_strides(%A: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{strides count must equal sizes rank}}
  %d = pim.access %A offset %c0 sizes [4, 4] strides [%c1] : memref<*xf32> -> !pim.access
  return
}

// -----

// R2: valid 개수 != sizes rank
func.func @bad_valid(%A: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{valid count must equal sizes rank}}
  %d = pim.access %A offset %c0 sizes [4, 4] strides [%c1, %c1] valid [%c1]
      : memref<*xf32> -> !pim.access
  return
}

// -----

// R3: pad는 valid 없이 못 쓴다
func.func @pad_without_valid(%A: memref<*xf32>, %f: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{pad requires valid}}
  %d = pim.access %A offset %c0 sizes [4] strides [%c1] pad %f : f32
      : memref<*xf32> -> !pim.access
  return
}

// -----

// gather: mask만 있고 other가 없으면 안 된다
func.func @gather_mask_without_other(%A: memref<*xf32>, %idx: tensor<8xi32>, %m: tensor<8xi1>) {
  // expected-error @+1 {{mask and other must be set together}}
  %v = pim.gather %A index %idx mask %m : tensor<8xi1>
      : memref<*xf32>, tensor<8xi32> -> tensor<8xf32>
  return
}
