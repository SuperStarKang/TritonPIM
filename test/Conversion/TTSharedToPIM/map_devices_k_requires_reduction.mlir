// p_k는 source plane 수가 아니라 semantic reduction의 spatial partition 수다.
// 따라서 reduction이 없는 연산에는 p_k > 1을 적용할 수 없다.
// RUN: triton-shared-opt "--pim-map-devices=p-m=1 p-n=1 p-k=2" -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 2,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024,
    local_capacity_limit_bytes = 0>} {

// expected-error @below {{p_k > 1 requires one semantic pim.reduction loop}}
func.func @elementwise(%tiles: index) attributes {
    pim.block = #pim.block<shape = [1, 1, 1]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
    pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %m = %c0 to %tiles step %c1 {
  } {pim.axis = #pim.axis<"m">}
  return
}
}
