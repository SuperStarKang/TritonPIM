// Rank-one x/y stay rank-one in Common; AiMX uses a padded column view.
// RUN: triton-shared-opt --pim-verify --pim-form-host-plan --convert-pim-to-aimx %s | FileCheck %s --check-prefix=IR --implicit-check-not=aimx.compute_template --implicit-check-not=linalg.
// RUN: triton-shared-opt --mlir-print-op-generic --pim-form-host-plan --convert-pim-to-aimx %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=STUB
// IR: aimx.descriptor_program @mv_device0_matvec_{{[0-9]+}} batch 8
// IR: opcode a_mask|matmul|pim_mode
// IR: opcode b_mask|matmul|pim_mode|stride
// IR: opcode c_mask|matmul|stride
// IR-LABEL: func.func @mv
// IR: aimx.foreach_device {{.*}} counts [2, 1, 1]
// IR: scf.execute_region no_inline
// IR: aimx.build_descriptors @mv_device0_matvec_
// IR: aimx.stage {{.*}}group_layout = array<i64: 96, 1, 256, 512, 1, 1>
// IR: aimx.invoke
// IR: aimx.unstage {{.*}}group_layout = array<i64: 8, 1, 8, 512, 1, 1>
// STUB: aimx_runtime_init(AIMX_BASE_FPGA, 2)
// STUB: aimx_resident_acquire(AIMX_KERNEL_KEY ":site0", _fpga,
// STUB: if (!_resident->ready) {
// STUB: aimx_prog_mv_device0_matvec_
// STUB: descriptor upload failed
// STUB: pim_exec
// STUB: aimx_resident_release(_resident)

module attributes {pim.target_profile = #pim.target_profile<
    target = "aimx", num_units = 2, banks_per_unit = 256,
    cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 268435456, local_capacity_limit_bytes = 0>} {
func.func @mv(%A: memref<*xbf16>, %X: memref<*xbf16>, %Y: memref<*xf32>)
    attributes {pim.block = #pim.block<shape = [8, 1, 32]>,
                pim.block_group = #pim.block_group<blocks = [1, 1, 3]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c96 = arith.constant 96 : index
  %c192 = arith.constant 192 : index
  %zero = arith.constant 0.0 : f32
  scf.for %roundM = %c0 to %c1 step %c1 {
  scf.for %roundK = %c0 to %c2 step %c1 {
    pim.device #pim.device_mapping<partitions = [2, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
    ^bb0(%dm: index, %dn: index, %dk: index):
      %mStart = arith.muli %dm, %c8 : index
      %kStart = arith.muli %roundK, %c96 : index
      pim.place %A start[%mStart, %kStart] size[%c8, %c96]
          allocation_shape[%c8, %c96] strides[%c192, %c1]
          {direction = "in", access = 0 : i64} : memref<*xbf16>
      pim.place %X start[%kStart] size[%c96] allocation_shape[%c96]
          strides[%c1] {direction = "in", access = 1 : i64, shared_over = ["m"]}
          : memref<*xbf16>
      pim.place %Y start[%mStart] size[%c8] allocation_shape[%c8]
          strides[%c1] {direction = "out", access = 2 : i64} : memref<*xf32>
      scf.for %gm = %c0 to %c1 step %c1 {
        %mOffset = arith.muli %gm, %c8 : index
        %m = arith.addi %mStart, %mOffset : index
        %empty = tensor.empty() : tensor<8xf32>
        %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
        %partial = scf.for %gk = %c0 to %c1 step %c1
            iter_args(%acc = %init) -> tensor<8xf32> {
          %kOffset = arith.muli %gk, %c96 : index
          %k = arith.addi %kStart, %kOffset : index
          %row = arith.muli %m, %c192 : index
          %off = arith.addi %row, %k : index
          %aa = pim.access %A offset %off sizes [8, 96]
              strides[%c192, %c1] origin[%m, %k] {id = 0 : i64}
              : memref<*xbf16> -> !pim.access
          %xa = pim.access %X offset %k sizes [96]
              strides[%c1] origin[%k] {id = 1 : i64} : memref<*xbf16> -> !pim.access
          %a = pim.load %aa : !pim.access -> tensor<8x96xbf16>
          %x = pim.load %xa : !pim.access -> tensor<96xbf16>
          %next = linalg.matvec ins(%a, %x : tensor<8x96xbf16>, tensor<96xbf16>)
              outs(%acc : tensor<8xf32>) -> tensor<8xf32>
          scf.yield %next : tensor<8xf32>
        } {pim.reduction = "add", pim.temporal_reduction}
        %ya = pim.access %Y offset %m sizes [8] strides[%c1]
            origin[%m] {id = 2 : i64} : memref<*xf32> -> !pim.access
        pim.combine %partial into %ya kind "add" : tensor<8xf32>, !pim.access
      } {pim.axis = #pim.axis<"m">}
    }
  } {pim.scheduling_axis = "reduction_k"}
  } {pim.scheduling_axis = "m"}
  return
}
}
