// One K scheduling round: MAC followed by a scalar multiply/add epilogue.
// Target-only fixture: it does not ask Common normalization to invent fusion.
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx %s | FileCheck %s --check-prefix=IR --implicit-check-not=linalg. --implicit-check-not=aimx.compute_template
// RUN: triton-shared-opt --mlir-print-op-generic --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=STUB
// IR-LABEL: func.func @matmul_epilogue
// IR: scf.execute_region no_inline
// IR: aimx.build_descriptors @matmul_epilogue_device0_matmul_relayout_
// IR: aimx.build_descriptors @matmul_epilogue_device0_matmul_accumulate_
// IR: aimx.build_descriptors @matmul_epilogue_device0_matmul_drain_
// Only the two explicit epilogue operations remain.
// IR: aimx.build_descriptors @matmul_epilogue_device0_elementwise_
// IR: aimx.build_descriptors @matmul_epilogue_device0_elementwise_
// IR: aimx.invoke
// IR-NOT: aimx.invoke
// STUB: if (!_resident->ready) {
// STUB: aimx_prog_matmul_epilogue_device0_matmul_relayout_
// STUB: aimx_prog_matmul_epilogue_device0_matmul_accumulate_
// STUB: aimx_prog_matmul_epilogue_device0_matmul_drain_
// STUB: aimx_prog_matmul_epilogue_device0_elementwise_
// STUB: aimx_prog_matmul_epilogue_device0_elementwise_
// STUB: descriptor upload failed
// STUB: pim_exec

module attributes {pim.target_profile = #pim.target_profile<
    target = "aimx", num_units = 2, banks_per_unit = 256,
    cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 268435456, local_capacity_limit_bytes = 0>} {
func.func @matmul_epilogue(%A: memref<*xbf16>, %B: memref<*xbf16>, %C: memref<*xf32>)
    attributes {
      pim.block = #pim.block<shape = [8, 16, 32]>,
      pim.block_group = #pim.block_group<blocks = [1, 2, 3]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c96 = arith.constant 96 : index
  %c192 = arith.constant 192 : index
  %zero = arith.constant 0.0 : f32
  scf.for %roundN = %c0 to %c1 step %c1 {
  scf.for %roundM = %c0 to %c1 step %c1 {
  scf.for %roundK = %c0 to %c1 step %c1 {
    pim.device #pim.device_mapping<partitions = [2, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 2, 1]>} {
    ^bb0(%dm: index, %dn: index, %dk: index):
      %mStart = arith.muli %dm, %c8 : index
      %kStart = arith.muli %roundK, %c96 : index
      pim.place %A start[%mStart, %kStart] size[%c8, %c96]
          allocation_shape[%c8, %c96] strides[%c192, %c1]
          {direction = "in", access = 0 : i64} : memref<*xbf16>
      pim.place %B start[%kStart, %c0] size[%c96, %c64]
          allocation_shape[%c96, %c64] strides[%c64, %c1]
          {direction = "in", access = 1 : i64, shared_over = ["m"]}
          : memref<*xbf16>
      pim.place %C start[%mStart, %c0] size[%c8, %c64]
          allocation_shape[%c8, %c64] strides[%c64, %c1]
          {direction = "out", access = 2 : i64} : memref<*xf32>
      scf.for %gm = %c0 to %c1 step %c1 {
        %mOffset = arith.muli %gm, %c8 : index
        %m = arith.addi %mStart, %mOffset : index
        scf.for %gn = %c0 to %c2 step %c1 {
          %n = arith.muli %gn, %c32 : index
          %empty = tensor.empty() : tensor<8x32xf32>
          %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x32xf32>)
              -> tensor<8x32xf32>
          %partial = scf.for %gk = %c0 to %c1 step %c1
              iter_args(%acc = %init) -> tensor<8x32xf32> {
            %kOffset = arith.muli %gk, %c96 : index
            %k = arith.addi %kStart, %kOffset : index
            %aRow = arith.muli %m, %c192 : index
            %aOffset = arith.addi %aRow, %k : index
            %aAccess = pim.access %A offset %aOffset sizes [8, 96]
                strides[%c192, %c1] origin[%m, %k] {id = 0 : i64}
                : memref<*xbf16> -> !pim.access
            %a = pim.load %aAccess : !pim.access -> tensor<8x96xbf16>
            %bRow = arith.muli %k, %c64 : index
            %bOffset = arith.addi %bRow, %n : index
            %bAccess = pim.access %B offset %bOffset sizes [96, 32]
                strides[%c64, %c1] origin[%k, %n] {id = 1 : i64}
                : memref<*xbf16> -> !pim.access
            %b = pim.load %bAccess : !pim.access -> tensor<96x32xbf16>
            %next = linalg.matmul
                ins(%a, %b : tensor<8x96xbf16>, tensor<96x32xbf16>)
                outs(%acc : tensor<8x32xf32>) -> tensor<8x32xf32>
            scf.yield %next : tensor<8x32xf32>
          } {pim.reduction = "add", pim.temporal_reduction}
          %cRow = arith.muli %m, %c64 : index
          %cOffset = arith.addi %cRow, %n : index
          %cAccess = pim.access %C offset %cOffset sizes [8, 32]
              strides[%c64, %c1] origin[%m, %n] {id = 2 : i64}
              : memref<*xf32> -> !pim.access
          %epilogue = linalg.generic {
              indexing_maps = [affine_map<(m,n) -> (m,n)>, affine_map<(m,n) -> (m,n)>],
              iterator_types = ["parallel", "parallel"]}
              ins(%partial : tensor<8x32xf32>)
              outs(%empty : tensor<8x32xf32>) {
          ^bb0(%value: f32, %unused: f32):
            %two = arith.constant 2.0 : f32
            %one = arith.constant 1.0 : f32
            %scaled = arith.mulf %value, %two : f32
            %biased = arith.addf %scaled, %one : f32
            linalg.yield %biased : f32
          } -> tensor<8x32xf32>
          pim.store %epilogue, %cAccess : tensor<8x32xf32>, !pim.access
        } {pim.axis = #pim.axis<"n">}
      } {pim.axis = #pim.axis<"m">}
    }
  } {pim.scheduling_axis = "reduction_k"}
  } {pim.scheduling_axis = "m"}
  } {pim.scheduling_axis = "n"}
  return
}
}
