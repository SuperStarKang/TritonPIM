// The masked row axis has one resident group; the unmasked column axis has two.
// The exact valid value clips staging and drain, never changes descriptors.
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx %s | FileCheck %s --check-prefix=IR
// RUN: triton-shared-opt --mlir-print-op-generic --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=STUB --implicit-check-not=aimx_patch_
// A constant zero mask still stages zero padding and drains no output rows.
// RUN: sed 's/valid\[%limit, %c8\]/valid[%c0, %c8]/g' %s | triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize | FileCheck %s --check-prefix=ZERO
// An empty mask on the two-group axis is also representable without holes.
// RUN: sed 's/valid\[%limit, %c8\]/valid[%limit, %c0]/g' %s | triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize | FileCheck %s --check-prefix=EMPTY-COLS
// RUN: sed 's/0.0 : bf16/1.0 : bf16/' %s | not triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx 2>&1 | FileCheck %s --check-prefix=PAD
// RUN: sed -e 's/pim.place %C /pim.place %A /' -e 's/pim.access %C /pim.access %A /' %s | not triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx 2>&1 | FileCheck %s --check-prefix=ALIAS
// Read-only views of the same host buffer with different masks need separate PAs.
// RUN: sed -e 's/pim.place %B /pim.place %A /' -e 's/pim.access %B /pim.access %A /' -e '/id = 1 : i64/s/valid\[%limit, %c8\]/valid[%c2, %c8]/' %s | triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx | FileCheck %s --check-prefix=INPUTS
// Mask changes along another multi-group axis cannot be evaluated just once.
// RUN: sed '/scf.for %gn =/a\          %varying = arith.addi %limit, %gn : index' %s | sed 's/valid\[%limit, %c8\]/valid[%varying, %c8]/g' | not triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx 2>&1 | FileCheck %s --check-prefix=VARYING
// IR-LABEL: func.func @masked_pointwise
// IR: %[[ROWS0:.*]] = arith.minsi {{.*}}%arg3{{.*}} : index
// IR: %[[ROWS1:.*]] = arith.minsi {{.*}}%arg3{{.*}} : index
// IR: %[[ROWS2:.*]] = arith.minsi {{.*}}%arg3{{.*}} : index
// IR: scf.execute_region no_inline
// IR: aimx.stage %arg0 {{.*}} sizes[%[[ROWS0]],
// IR: aimx.stage %arg1 {{.*}} sizes[%[[ROWS1]],
// IR: aimx.invoke
// IR: aimx.unstage {{.*}} sizes[%[[ROWS2]],
// STUB: if (!_resident->ready) {
// STUB: aimx_prog_masked_pointwise_device0_elementwise_
// STUB: descriptor upload failed
// STUB: _resident->ready = true;
// STUB: {  // aimx.stage
// STUB: pim_exec
// ZERO-LABEL: func.func @masked_pointwise
// ZERO: %[[ZERO:c0[^ ]*]] = arith.constant 0 : index
// ZERO: aimx.stage {{.*}} sizes[%[[ZERO]],
// ZERO: aimx.unstage {{.*}} sizes[%[[ZERO]],
// EMPTY-COLS-LABEL: func.func @masked_pointwise
// EMPTY-COLS: %[[ZERO:c0[^ ]*]] = arith.constant 0 : index
// EMPTY-COLS: aimx.stage {{.*}} sizes[{{.*}}, %[[ZERO]]]
// EMPTY-COLS: aimx.unstage {{.*}} sizes[{{.*}}, %[[ZERO]]]
// PAD: AiMX masked input staging supports only positive-zero or unspecified padding
// ALIAS: AiMX masked read/write aliases need a masked descriptor merge
// INPUTS: %[[A:.*]] = aimx.alloc {{.*}} kind data
// INPUTS: %[[B:.*]] = aimx.alloc {{.*}} kind data
// INPUTS: aimx.stage %arg0 {{.*}} into %[[A]]
// INPUTS: aimx.stage %arg0 {{.*}} into %[[B]]
// VARYING: AiMX varying masks across multiple groups need per-group masked transfers

#identity = affine_map<(m,n) -> (m,n)>
module attributes {pim.target_profile = #pim.target_profile<target = "aimx",
    num_units = 2, banks_per_unit = 256, cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 268435456, local_capacity_limit_bytes = 0>} {
func.func @masked_pointwise(%A: memref<*xbf16>, %B: memref<*xbf16>, %C: memref<*xbf16>, %limit: index)
    attributes {pim.block = #pim.block<shape = [2, 8, 1]>,
                pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %pad = arith.constant 0.0 : bf16
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
              origin[%m, %n] valid[%limit, %c8] pad %pad : bf16 {id = 0 : i64} : memref<*xbf16> -> !pim.access
          %ba = pim.access %B offset %offset sizes [2, 8] strides[%c16, %c1]
              origin[%m, %n] valid[%limit, %c8] pad %pad : bf16 {id = 1 : i64} : memref<*xbf16> -> !pim.access
          %ca = pim.access %C offset %offset sizes [2, 8] strides[%c16, %c1]
              origin[%m, %n] valid[%limit, %c8] pad %pad : bf16 {id = 2 : i64} : memref<*xbf16> -> !pim.access
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
