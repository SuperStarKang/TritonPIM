// 계층으로 분류되지 않은 runtime loop count를 allocation_shape로 사용하는 fallback은
// fixed resident slot 계약을 깨므로 pim-place-data가 거부한다.
// RUN: triton-shared-opt --pim-place-data -verify-diagnostics %s

module {
func.func @unclassified_resident_loop(%A: memref<*xi8>, %trips: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  ^bb0(%device_m: index, %device_n: index, %device_k: index):
    scf.for %i = %c0 to %trips step %c1 {
      // expected-error @below {{is driven by a loop outside the Block-Group-Execution Tile hierarchy, so a fixed resident slot cannot be derived}}
      %access = pim.access %A offset %i sizes [1] strides[%c1] origin[%i]
          {id = 0 : i64} : memref<*xi8> -> !pim.access
    }
  }
  return
}
}
