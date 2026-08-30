// cyclic 분할(k_part 가 tile 간격으로 시작하고 reduction 이 SPLIT_K·tile 씩 건너뜀)은
// 나뉜 device 의 K 영역이 띄엄띄엄이라 배치가 성립하지 않는다 — 진단으로 막는다.
// reduction 항의 계수 256 ≠ tile 16 이 판정 근거다.
// RUN: triton-shared-opt "--pim-map-devices=p-k=2" --pim-place-data -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 8,
    banks_per_unit = 1,
    cores_per_bank = 1,
    lanes_per_core = 1,
    bank_capacity_limit_bytes = 3072, local_capacity_limit_bytes = 49152>} {

func.func @split_k_cyclic(%A: memref<*xi8>, %tr: index, %tp: index, %tk: index,
                          %lda: index)
    attributes {
      pim.block = #pim.block<shape = [8, 1, 16]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
      pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  scf.for %x = %c0 to %tr step %c1 {
    %m = arith.muli %x, %c8 : index
    scf.for %p = %c0 to %tp step %c1 {
      %kbase = arith.muli %p, %c16 : index
      scf.for %i = %c0 to %tk step %c1 {
        %koff = arith.muli %i, %c256 : index
        %k = arith.addi %kbase, %koff : index
        // expected-error @below {{not a blocked split}}
        %da = pim.access %A offset %k sizes [8, 16] strides[%lda, %c1]
              origin[%m, %k] {id = 0 : i64} : memref<*xi8> -> !pim.access
      }
    } {pim.axis = #pim.axis<"k">}
  } {pim.axis = #pim.axis<"m">}
  return
}
}
