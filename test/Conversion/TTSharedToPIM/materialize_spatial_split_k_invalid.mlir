// Spatial K output은 주소 모양만 보고 reduction으로 추측하지 않는다. linalg 의미에서
// 확정된 pim.reduction이 없으면 target이 partial combine을 재해석해야 하므로 Common
// lowering 단계에서 진단한다.
// RUN: triton-shared-opt "--pim-map-devices=p-m=1 p-n=1 p-k=1" -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024,
    local_capacity_limit_bytes = 0>} {

// expected-error @below {{a spatial K output needs one semantic pim.reduction before device mapping}}
func.func @spatial_k_without_reduction(%Partial: memref<*xi32>,
                                       %planes: index) attributes {
    pim.block = #pim.block<shape = [1, 1, 1]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
    pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %plane = %c0 to %planes step %c1 {
    %value = tensor.empty() : tensor<1xi32>
    %access = pim.access %Partial offset %plane sizes [1] strides[%c1]
        origin[%plane] {id = 0 : i64} : memref<*xi32> -> !pim.access
    pim.store %value, %access : tensor<1xi32>, !pim.access
  } {pim.axis = #pim.axis<"k">}
  return
}
}
