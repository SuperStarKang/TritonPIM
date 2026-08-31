// [p_m,p_n,p_k] 명시 분할에서 p_k는 semantic reduction을 직접 나눈다.
// 곱이 unit 수를 넘으면 진단이다.
// 이 테스트에는 pim.access가
// 없다. Device scheduling은 세 hierarchy attribute만으로 완결되어야 하며 실제
// access shape와의 정합성 검사는 뒤의 pim-verify가 담당한다.
// RUN: triton-shared-opt "--pim-map-devices=p-m=2 p-n=2 p-k=2" %s | FileCheck %s
// RUN: triton-shared-opt "--pim-map-devices=p-m=64" -verify-diagnostics %s

// CHECK: pim.device <partitions = [2, 2, 2]>

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 8, banks_per_unit = 1,
    cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 3072,
    local_capacity_limit_bytes = 49152>} {

// expected-error @below {{needs 64 PIM units but the profile has 8}}
func.func @three_axes(%tr: index, %tc: index, %tp: index)
    attributes {
      pim.block = #pim.block<shape = [1, 1, 1]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
      pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %x = %c0 to %tr step %c1 {
    scf.for %y = %c0 to %tc step %c1 {
      scf.for %p = %c0 to %tp step %c1 {
      } {pim.reduction = "add"}
    } {pim.axis = #pim.axis<"n">}
  } {pim.axis = #pim.axis<"m">}
  return
}
}
