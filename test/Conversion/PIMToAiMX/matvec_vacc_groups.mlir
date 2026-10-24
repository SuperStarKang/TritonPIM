// Match gemv.py: four M Tiles, each with four K groups of 32x64.
// AiMX runs y = W·x as yᵀ = xᵀ·Wᵀ with batch 1: x is one A row, W is staged
// transposed as B [K x M], y is one C row, and output lanes follow M.
// RUN: triton-shared-opt --pim-verify --pim-form-host-plan --convert-pim-to-aimx %s | FileCheck %s --check-prefix=IR --implicit-check-not=elementwise_
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize --mlir-print-op-generic %s | triton-shared-translate --aimx-to-descriptors | FileCheck %s --check-prefix=DESC
// A 256-element MAC step covers each 64-element K group; no final zero MACs.
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize %s | FileCheck %s --check-prefix=FULL
// The transposed stages and unstage reach the runtime stub.
// RUN: triton-shared-opt --mlir-print-op-generic --pim-form-host-plan --convert-pim-to-aimx %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=STUB
// An empty reduction must not drain stale vACC.
// RUN: sed 's/to %k4 step %k1/to %k0 step %k1/' %s | triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize | FileCheck %s --check-prefix=EMPTY
// Multiple contributions to the same rank-one output must retain ADDs.
// RUN: sed '/^        pim.combine /p' %s | triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx | FileCheck %s --check-prefix=COMBINE
// IR-DAG: aimx.descriptor_program @mv_groups_device0_matmul_relayout_{{[0-9]+}} batch 1
// IR-DAG: aimx.descriptor_program @mv_groups_device0_matmul_accumulate_{{[0-9]+}} batch 1
// IR-DAG: aimx.descriptor_program @mv_groups_device0_matmul_drain_{{[0-9]+}} batch 1
// IR-LABEL: func.func @mv_groups
// Placements: W is 4 K panels of 256 x 4096, x is 4 panels of 1 x 4096.
// Arel holds the x row of all four groups: 4 * 1 * 256 * 2 * 16 bytes.
// IR-DAG: aimx.alloc {{.*}} bytes 8388608 align 8192 kind data
// IR-DAG: aimx.alloc {{.*}} bytes 32768 align 8192 kind data
// IR-DAG: aimx.alloc {{.*}} bytes 32768 align 8192 kind data
// IR: scf.execute_region no_inline
// IR: aimx.build_descriptors @mv_groups_device0_matmul_relayout_
// IR: aimx.matmul_phase = "relayout"
// Output microtile loops enclose the K-group loop.
// IR: scf.for %[[ROW:.*]] =
// IR: scf.for %[[COL:.*]] =
// IR: scf.for %[[K:.*]] = {{.*}} : i32 {
// IR: aimx.build_descriptors @mv_groups_device0_matmul_accumulate_
// IR-NOT: matmul_drain_
// IR: aimx.matmul_phase = "accumulate"
// IR: aimx.build_descriptors @mv_groups_device0_matmul_drain_
// IR-NOT: aimx.build_descriptors {{.*}}matmul_
// IR-DAG: aimx.stage {{.*}}transpose into {{.*}}group_layout = array<i64: 1, 64, 1, 4096, 1, 4>
// IR-DAG: aimx.stage {{.*}}transpose into {{.*}}group_layout = array<i64: 64, 32, 256, 4096, 4, 1>
// IR: aimx.invoke
// IR-NOT: aimx.invoke
// IR: aimx.unstage {{.*}}transpose {{.*}}group_layout = array<i64: 1, 32, 1, 4096, 1, 1>
// Batch 1: A reads 8192 bytes of Arel, B reads one full 4096-lane row of Wᵀ.
// DESC-LABEL: static uint32_t aimx_prog_mv_groups_device0_matmul_accumulate_
// DESC: 8192u, 0x1012u, fpga_id, code);
// DESC: 8192u, 0x41014u, fpga_id, code);
// DESC-NOT: 0x40018u
// DESC: return idx;
// FULL-LABEL: func.func @mv_groups
// FULL: aimx.build_descriptors @mv_groups_device0_matmul_accumulate_
// FULL-NOT: aimx.build_descriptors @mv_groups_device0_matmul_padding_
// FULL: aimx.matmul_phase = "accumulate"
// FULL: aimx.build_descriptors @mv_groups_device0_matmul_drain_
// STUB: // aimx.stage
// STUB: aimx_copy_to_pim
// STUB: aimx.unstage contiguous groups
// EMPTY-LABEL: func.func @mv_groups
// EMPTY-NOT: aimx.build_descriptors @mv_groups_device0_matmul_
// EMPTY: aimx.invoke
// COMBINE-LABEL: func.func @mv_groups
// COMBINE: aimx.build_descriptors @mv_groups_device0_matmul_drain_
// COMBINE-COUNT-2: aimx.build_descriptors @mv_groups_device0_elementwise_

module attributes {pim.target_profile = #pim.target_profile<target = "aimx",
    num_units = 1, banks_per_unit = 256, cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 268435456, local_capacity_limit_bytes = 0>} {
func.func @mv_groups(%A: memref<*xf32>, %X: memref<*xf32>, %Y: memref<*xf32>)
    attributes {pim.block = #pim.block<shape = [32, 1, 64]>,
                pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %k0 = arith.constant 0 : i32
  %k1 = arith.constant 1 : i32
  %k4 = arith.constant 4 : i32
  %zero = arith.constant 0.0 : f32
  scf.for %roundM = %c0 to %c4 step %c1 {
    pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 4]>} {
    ^bb0(%dm: index, %dn: index, %dk: index):
      %mStart = arith.muli %roundM, %c32 : index
      pim.place %A start[%mStart, %c0] size[%c32, %c256]
          allocation_shape[%c32, %c256] strides[%c256, %c1]
          {direction = "in", access = 0 : i64} : memref<*xf32>
      pim.place %X start[%c0] size[%c256] allocation_shape[%c256]
          strides[%c1] {direction = "in", access = 1 : i64, shared_over = ["m"]}
          : memref<*xf32>
      pim.place %Y start[%mStart] size[%c32] allocation_shape[%c32]
          strides[%c1] {direction = "out", access = 2 : i64} : memref<*xf32>
      scf.for %gm = %c0 to %c1 step %c1 {
        %mOffset = arith.muli %gm, %c32 : index
        %m = arith.addi %mStart, %mOffset : index
        %empty = tensor.empty() : tensor<32xf32>
        %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
        %partial = scf.for %gk = %k0 to %k4 step %k1
            iter_args(%acc = %init) -> tensor<32xf32> : i32 {
          %slot = arith.index_cast %gk : i32 to index
          %k = arith.muli %slot, %c64 : index
          %row = arith.muli %m, %c256 : index
          %off = arith.addi %row, %k : index
          %aa = pim.access %A offset %off sizes [32, 64]
              strides[%c256, %c1] origin[%m, %k] {id = 0 : i64}
              : memref<*xf32> -> !pim.access
          %xa = pim.access %X offset %k sizes [64]
              strides[%c1] origin[%k] {id = 1 : i64} : memref<*xf32> -> !pim.access
          %a = pim.load %aa : !pim.access -> tensor<32x64xf32>
          %x = pim.load %xa : !pim.access -> tensor<64xf32>
          %next = linalg.matvec ins(%a, %x : tensor<32x64xf32>, tensor<64xf32>)
              outs(%acc : tensor<32xf32>) -> tensor<32xf32>
          scf.yield %next : tensor<32xf32>
        } {pim.reduction = "add", pim.temporal_reduction}
        %ya = pim.access %Y offset %m sizes [32] strides[%c1]
            origin[%m] {id = 2 : i64} : memref<*xf32> -> !pim.access
        pim.combine %partial into %ya kind "add" : tensor<32xf32>, !pim.access
      } {pim.axis = #pim.axis<"m">}
    }
  } {pim.scheduling_axis = "m"}
  return
}
}
