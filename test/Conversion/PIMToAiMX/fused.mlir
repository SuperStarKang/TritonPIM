// Two linalg operations, with a multi-operation scalar epilogue.
// RUN: triton-shared-opt --convert-pim-to-aimx %s | FileCheck %s --check-prefix=IR --implicit-check-not=linalg. --implicit-check-not=aimx.compute_template
// RUN: triton-shared-opt --mlir-print-op-generic --convert-pim-to-aimx --cse --canonicalize %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=STUB
// RUN: triton-shared-opt --convert-pim-to-aimx %s | FileCheck %s --check-prefix=REUSE
// IR-DAG: opcode descr_dst_mask|a_mask|add
// IR-DAG: opcode a_mask|mul|pim_mode
// IR-DAG: opcode descr_dst_mask|a_mask|sub
// IR-LABEL: func.func @fused
// IR: scf.execute_region no_inline
// IR: aimx.build_descriptors @fused_device0_elementwise_
// IR: aimx.build_descriptors @fused_device0_elementwise_
// IR: aimx.build_descriptors @fused_device0_elementwise_
// IR: aimx.build_descriptors @fused_device0_copy_
// IR: aimx.invoke
// IR-NOT: aimx.invoke
// REUSE-LABEL: func.func @fused
// REUSE: scf.execute_region no_inline
// First generic result remains separate from the second generic's arguments.
// REUSE: aimx.build_descriptors @fused_device0_elementwise_
// Multiply output becomes the subtract's second operand and output, preserving B-A order.
// REUSE: aimx.build_descriptors @fused_device0_elementwise_{{[0-9]+}}({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, %[[TEMP:[a-zA-Z0-9_]+]],
// REUSE: aimx.build_descriptors @fused_device0_elementwise_{{[0-9]+}}({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, %[[TEMP]], {{[^,]+}}, {{[^,]+}}, %[[TEMP]],
// STUB: static const uint16_t aimx_data_fused_device0_constant_
// STUB: if (!_resident->ready) {
// STUB: aimx_prog_fused_device0_elementwise_
// STUB: aimx_prog_fused_device0_elementwise_
// STUB: aimx_prog_fused_device0_elementwise_
// STUB-NOT: descriptor upload failed
// STUB: aimx_prog_fused_device0_copy_
// STUB: descriptor upload failed
// STUB: _resident->ready = true
// STUB: pim_exec

#identity = affine_map<(m,n) -> (m,n)>
module attributes {pim.target_profile = #pim.target_profile<target = "aimx",
    num_units = 2, banks_per_unit = 256, cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 268435456, local_capacity_limit_bytes = 0>} {
func.func @fused(%A: memref<*xbf16>, %B: memref<*xbf16>, %C: memref<*xbf16>)
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
        scf.for %gn = %c0 to %c2 step %c1 {
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
          %epilogue = linalg.generic {
              indexing_maps = [#identity, #identity, #identity],
              iterator_types = ["parallel", "parallel"]}
              ins(%result, %b : tensor<2x8xbf16>, tensor<2x8xbf16>)
              outs(%empty : tensor<2x8xbf16>) {
          ^bb0(%input: bf16, %bias: bf16, %unused2: bf16):
            %two = arith.constant 2.0 : bf16
            %scaled = arith.mulf %input, %two : bf16
            %shifted = arith.subf %bias, %scaled : bf16
            linalg.yield %shifted : bf16
          } -> tensor<2x8xbf16>
          pim.store %epilogue, %ca : tensor<2x8xbf16>, !pim.access
        } {pim.axis = #pim.axis<"n">}
      } {pim.axis = #pim.axis<"m">}
    }
  } {pim.scheduling_axis = "m"}
  } {pim.scheduling_axis = "n"}
  return
}
}
