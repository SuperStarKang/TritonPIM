// UPMEM은 PIM Common IR의 정적 allocation_shape를 정확한 MRAM slot으로 옮긴다.
// runtime trip count나 bank 예산 균등 분할 fallback은 허용하지 않는다.
// RUN: not triton-shared-opt --convert-pim-to-upmem %s 2>&1 | FileCheck %s

// CHECK: convert-pim-to-upmem: allocation_shape must be the static fixed slot decided by PIM Common IR

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 49152>} {

func.func @dynamic_slot(%A: memref<*xi8>, %capacity: index) attributes {
    pim.block = #pim.block<shape = [8, 1, 1]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  ^bb0(%pm: index, %pn: index, %pk: index):
    pim.place %A start[%c0] size[%c8] allocation_shape[%capacity]
        strides[%c1]
        {direction = "in", access = 0 : i64,
         replicated_over = ["n", "k"]} : memref<*xi8>

    scf.for %m = %c0 to %c1 step %c1 {
      %globalM = arith.addi %c0, %m {pim.device_origin} : index
      %origin = arith.muli %globalM, %c8 : index
      %access = pim.access %A offset %origin sizes [8] strides[%c1]
          origin[%origin] {id = 0 : i64}
          : memref<*xi8> -> !pim.access
      %tile = pim.load %access : !pim.access -> tensor<8xi8>
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
