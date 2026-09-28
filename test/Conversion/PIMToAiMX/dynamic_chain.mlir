// The Common N-group upper bound may vary within its declared two-group bound.
// Keep physical allocations, but reconstruct descriptors for this dynamic loop.
// RUN: triton-shared-opt --mlir-print-op-generic --convert-pim-to-aimx %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --implicit-check-not="if (!_resident->ready)"
// RUN: triton-shared-opt --mlir-print-op-generic --convert-pim-to-aimx %s | triton-shared-translate --aimx-to-descriptors | FileCheck %s --check-prefix=DESC --implicit-check-not=aimx_patch_
// CHECK: aimx_resident_acquire(AIMX_KERNEL_KEY ":site0", _fpga,
// CHECK: .idx = 0;
// CHECK: for (int64_t {{.*}} < a3;
// CHECK: aimx_prog_dynamic_chain_device0_elementwise_
// CHECK: descriptor upload failed
// CHECK: pim_exec
// DESC: static uint32_t aimx_prog_dynamic_chain_device0_elementwise_

#identity = affine_map<(m,n) -> (m,n)>
module attributes {pim.target_profile = #pim.target_profile<target = "aimx",
    num_units = 2, banks_per_unit = 256, cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 268435456, local_capacity_limit_bytes = 0>} {
func.func @dynamic_chain(%A: memref<*xbf16>, %B: memref<*xbf16>, %C: memref<*xbf16>, %limit: index)
    attributes {pim.block = #pim.block<shape = [2, 8, 1]>,
                pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  scf.for %rn = %c0 to %c1 step %c1 {
  scf.for %rm = %c0 to %c1 step %c1 {
    pim.device #pim.device_mapping<partitions = [2, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 2, 1]>} {
    ^bb0(%dm: index, %dn: index, %dk: index):
      %start = arith.muli %dm, %c2 : index
      pim.place %A start[%start, %c0] size[%c2, %c16] allocation_shape[%c2, %c16]
          strides[%c16, %c1] {direction = "in", access = 0 : i64} : memref<*xbf16>
      pim.place %B start[%start, %c0] size[%c2, %c16] allocation_shape[%c2, %c16]
          strides[%c16, %c1] {direction = "in", access = 1 : i64} : memref<*xbf16>
      pim.place %C start[%start, %c0] size[%c2, %c16] allocation_shape[%c2, %c16]
          strides[%c16, %c1] {direction = "out", access = 2 : i64} : memref<*xbf16>
      scf.for %gm = %c0 to %c1 step %c1 {
        %mo = arith.muli %gm, %c2 : index
        %m = arith.addi %start, %mo : index
        scf.for %gn = %c0 to %limit step %c1 {
          %n = arith.muli %gn, %c8 : index
          %row = arith.muli %m, %c16 : index
          %offset = arith.addi %row, %n : index
          %aa = pim.access %A offset %offset sizes [2, 8] strides[%c16, %c1]
              origin[%m, %n] {id = 0 : i64} : memref<*xbf16> -> !pim.access
          %ba = pim.access %B offset %offset sizes [2, 8] strides[%c16, %c1]
              origin[%m, %n] {id = 1 : i64} : memref<*xbf16> -> !pim.access
          %ca = pim.access %C offset %offset sizes [2, 8] strides[%c16, %c1]
              origin[%m, %n] {id = 2 : i64} : memref<*xbf16> -> !pim.access
          %a = pim.load %aa : !pim.access -> tensor<2x8xbf16>
          %b = pim.load %ba : !pim.access -> tensor<2x8xbf16>
          %empty = tensor.empty() : tensor<2x8xbf16>
          %result = linalg.generic {
              indexing_maps = [#identity, #identity, #identity],
              iterator_types = ["parallel", "parallel"]}
              ins(%a, %b : tensor<2x8xbf16>, tensor<2x8xbf16>)
              outs(%empty : tensor<2x8xbf16>) {
          ^bb0(%lhs: bf16, %rhs: bf16, %unused: bf16):
            %value = arith.addf %lhs, %rhs : bf16
            linalg.yield %value : bf16
          } -> tensor<2x8xbf16>
          pim.store %result, %ca : tensor<2x8xbf16>, !pim.access
        } {pim.axis = #pim.axis<"n">}
      } {pim.axis = #pim.axis<"m">}
    }
  } {pim.scheduling_axis = "m"}
  } {pim.scheduling_axis = "n"}
  return
}
}
