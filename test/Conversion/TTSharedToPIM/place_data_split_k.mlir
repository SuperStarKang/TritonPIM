// k 를 device 축으로 나눈 blocked 분할: 나뉜 k_part 축이 reduction loop 와 함께
// 버퍼의 K 축을 몰아도, reduction 이 정확히 tile(16)씩 전진하므로 device 의 K 영역은
// 연속 구간 하나다 — 배치가 성립한다.
// RUN: triton-shared-opt "--pim-map-devices=device-axis-partitions=m:2,n:2,k:2" --pim-place-data %s | FileCheck %s

// CHECK: pim.place %arg0 {{.*}}axis_sources = ["m", "k"]
// CHECK: pim.place %arg0 {{.*}}axis_sources = ["m", "k"]

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", number_of_pim_devices = 8,
    number_of_banks_per_pim_device = 1,
    number_of_pim_cores_per_bank = 1,
    number_of_parallel_lanes_per_pim_core = 1,
    parallel_execution_kind = "multithread",
    usable_bank_capacity_bytes = 3072, usable_local_capacity_bytes = 49152>} {

func.func @split_k(%A: memref<*xi8>, %C: memref<*xi8>,
                   %tr: index, %tc: index, %tp: index,
                   %tk: index, %kper: index, %lda: index)
    attributes {pim.execution_tile = #pim.execution_tile<m = 8, n = 1, k = 0>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  scf.for %x = %c0 to %tr step %c1 {
    %m = arith.muli %x, %c8 : index
    scf.for %y = %c0 to %tc step %c1 {
      %n = arith.muli %y, %c1 : index
      %dc = pim.access %C offset %n sizes [1] strides[%c1]
            origin[%n] {id = 2 : i64} : memref<*xi8> -> !pim.access
      scf.for %p = %c0 to %tp step %c1 {
        // blocked: k_part 의 시작 = p · (연속 구간 크기), reduction 은 16 씩 전진.
        %kbase = arith.muli %p, %kper : index
        scf.for %i = %c0 to %tk step %c1 {
          %koff = arith.muli %i, %c16 : index
          %k = arith.addi %kbase, %koff : index
          %da = pim.access %A offset %k sizes [8, 16] strides[%lda, %c1]
                origin[%m, %k] {id = 0 : i64} : memref<*xi8> -> !pim.access
          // group-k 가 끼면 계수가 사슬(muli·muli)이 된다 — 곱을 벗겨 2·16 = 32 == tile
          // 로 판정해야 한다.
          %kg = arith.muli %i, %c2 : index
          %koff2 = arith.muli %kg, %c16 : index
          %k2 = arith.addi %kbase, %koff2 : index
          %db = pim.access %A offset %k2 sizes [8, 32] strides[%lda, %c1]
                origin[%m, %k2] {id = 1 : i64} : memref<*xi8> -> !pim.access
        }
      } {pim.axis = #pim.axis<"k">}
    } {pim.axis = #pim.axis<"n">}
  } {pim.axis = #pim.axis<"m">}
  return
}
}
