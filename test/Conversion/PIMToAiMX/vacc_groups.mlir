// Each 8x512 output microtile accumulates both K groups before its only drain.
// The signed i32 K loop has a nonzero lower bound and a non-unit step.
// RUN: triton-shared-opt --pim-verify --pim-form-host-plan --convert-pim-to-aimx %s | FileCheck %s --check-prefix=IR --implicit-check-not=elementwise_
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize --mlir-print-op-generic %s -o %t.aimx
// RUN: triton-shared-translate --aimx-to-descriptors %t.aimx | FileCheck %s --check-prefix=DESC
// RUN: triton-shared-translate --aimx-to-runtime-stub %t.aimx | FileCheck %s --check-prefix=STUB
// A partially filled K tile has one group, with final zero MACs up to K=256.
// RUN: sed 's/to %k7 step %k2/to %k5 step %k2/' %s | triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize | FileCheck %s --check-prefix=TAIL
// An empty reduction must copy zero; draining stale vACC would be incorrect.
// RUN: sed 's/to %k7 step %k2/to %k3 step %k2/' %s | triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize | FileCheck %s --check-prefix=EMPTY
// Observed intermediates and nonzero initial accumulators must fail explicitly.
// RUN: sed '/scf.yield %next/i\          "test.observe"(%acc) : (tensor<8x512xf32>) -> ()' %s | not triton-shared-opt --allow-unregistered-dialect --pim-form-host-plan --convert-pim-to-aimx 2>&1 | FileCheck %s --check-prefix=OBSERVED
// RUN: sed 's/0.0 : f32/1.0 : f32/' %s | not triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx 2>&1 | FileCheck %s --check-prefix=NONZERO
// Two contributions to one output must still use real ADDs.
// RUN: sed '/^        pim.combine /p' %s | triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx | FileCheck %s --check-prefix=COMBINE
// GEMM storage pitch and computed N can be varied without changing Common groups.
// RUN: triton-shared-opt --pim-form-host-plan "--convert-pim-to-aimx=matmul-row-alignment=4096" %s | FileCheck %s --check-prefix=WIDE --implicit-check-not=elementwise_
// RUN: triton-shared-opt --pim-form-host-plan "--convert-pim-to-aimx=matmul-row-alignment=4096 matmul-useful-n-only=true" %s | FileCheck %s --check-prefix=USEFUL --implicit-check-not=elementwise_
// Two useful N microtiles must survive; this option is not GEMV's first-N-only rule.
// RUN: sed 's/1024/2048/g; s/512/1024/g' %s | triton-shared-opt --pim-form-host-plan "--convert-pim-to-aimx=matmul-row-alignment=4096 matmul-useful-n-only=true" | FileCheck %s --check-prefix=USEFUL1024 --implicit-check-not=elementwise_
// RUN: not triton-shared-opt --pim-form-host-plan "--convert-pim-to-aimx=matmul-row-alignment=513" %s 2>&1 | FileCheck %s --check-prefix=BAD-ALIGN
// IR-LABEL: func.func @vacc_groups
// Arel holds both K groups at once: 2 * 8 * 256 * 2 * 16 bytes.
// IR: aimx.alloc {{.*}} bytes 131072 align 8192 kind data
// IR: scf.execute_region no_inline
// IR: aimx.build_descriptors @vacc_groups_device0_matmul_relayout_
// IR: aimx.matmul_phase = "relayout"
// Output microtile loops enclose the original K-group loop.
// IR: scf.for %[[ROW:.*]] =
// IR: scf.for %[[COL:.*]] =
// IR: scf.for %[[K:.*]] = {{.*}} : i32 {
// IR: aimx.build_descriptors @vacc_groups_device0_matmul_accumulate_
// IR-NOT: matmul_drain_
// IR: aimx.matmul_phase = "accumulate"
// IR: aimx.build_descriptors @vacc_groups_device0_matmul_drain_
// IR-NOT: aimx.build_descriptors {{.*}}matmul_
// IR: aimx.stage {{.*}}group_layout = array<i64: 128, 512, 256, 512, 2, 2>
// IR: aimx.invoke
// IR-NOT: aimx.invoke
// IR: aimx.unstage
// Accumulate programs contain A/B MAC descriptors and no C drain.
// DESC-LABEL: static uint32_t aimx_prog_vacc_groups_device0_matmul_accumulate_
// DESC-DAG: = 128;
// DESC-DAG: = 32;
// DESC: 8192u, 0x19012u, fpga_id, code);
// DESC: 1024u, 0x59014u, fpga_id, code);
// DESC-NOT: 0x58018u
// DESC: return idx;
// STUB: aimx_prog_vacc_groups_device0_matmul_relayout_
// STUB: aimx_prog_vacc_groups_device0_matmul_accumulate_
// STUB: aimx_prog_vacc_groups_device0_matmul_drain_
// STUB: pim_exec
// STUB-NOT: pim_exec
// TAIL-LABEL: func.func @vacc_groups
// TAIL: aimx.build_descriptors @vacc_groups_device0_matmul_accumulate_
// TAIL: aimx.build_descriptors @vacc_groups_device0_matmul_padding_
// TAIL: aimx.build_descriptors @vacc_groups_device0_matmul_drain_
// EMPTY-LABEL: func.func @vacc_groups
// EMPTY-NOT: aimx.build_descriptors @vacc_groups_device0_matmul_
// EMPTY: aimx.invoke
// OBSERVED: AiMX vACC reduction requires an unobserved MAC accumulator
// NONZERO: AiMX vACC reduction requires a positive-zero accumulator
// COMBINE-LABEL: func.func @vacc_groups
// COMBINE: aimx.build_descriptors @vacc_groups_device0_matmul_drain_
// COMBINE-COUNT-2: aimx.build_descriptors @vacc_groups_device0_elementwise_
// WIDE-LABEL: func.func @vacc_groups
// WIDE: aimx.matmul_phase = "relayout"
// WIDE: scf.for
// WIDE: %[[COLUMNS:.*]] = arith.constant 4096 : index
// WIDE: scf.for {{.*}} to %[[COLUMNS]] step {{.*}} {
// WIDE: aimx.build_descriptors @vacc_groups_device0_matmul_accumulate_
// WIDE-NOT: matmul_drain_
// WIDE: aimx.matmul_phase = "accumulate"
// WIDE: aimx.build_descriptors @vacc_groups_device0_matmul_drain_
// WIDE: aimx.stage {{.*}}group_layout = array<i64: 128, 512, 256, 4096, 2, 2>
// WIDE: aimx.unstage {{.*}}group_layout = array<i64: 8, 512, 8, 4096, 2, 2>
// USEFUL-LABEL: func.func @vacc_groups
// USEFUL: aimx.matmul_phase = "relayout"
// USEFUL: scf.for
// USEFUL-NOT: arith.constant 4096 : index
// USEFUL: arith.constant 512 : index
// USEFUL-NOT: arith.constant 4096 : index
// USEFUL: scf.for
// USEFUL: aimx.build_descriptors @vacc_groups_device0_matmul_accumulate_
// USEFUL-NOT: matmul_drain_
// USEFUL: aimx.matmul_phase = "accumulate"
// USEFUL: aimx.build_descriptors @vacc_groups_device0_matmul_drain_
// USEFUL: aimx.stage {{.*}}group_layout = array<i64: 128, 512, 256, 4096, 2, 2>
// USEFUL: aimx.unstage {{.*}}group_layout = array<i64: 8, 512, 8, 4096, 2, 2>
// USEFUL1024-LABEL: func.func @vacc_groups
// USEFUL1024: aimx.matmul_phase = "relayout"
// USEFUL1024: scf.for
// USEFUL1024: %[[COLUMNS:.*]] = arith.constant 1024 : index
// USEFUL1024: scf.for {{.*}} to %[[COLUMNS]] step {{.*}} {
// USEFUL1024: aimx.build_descriptors @vacc_groups_device0_matmul_accumulate_
// USEFUL1024-NOT: matmul_drain_
// USEFUL1024: aimx.matmul_phase = "accumulate"
// USEFUL1024: aimx.build_descriptors @vacc_groups_device0_matmul_drain_
// USEFUL1024: aimx.stage {{.*}}group_layout = array<i64: 128, 1024, 256, 4096, 2, 2>
// USEFUL1024: aimx.unstage {{.*}}group_layout = array<i64: 8, 1024, 8, 4096, 2, 2>
// BAD-ALIGN: AiMX matmul row alignment must be zero or a positive multiple of 4096/batch

module attributes {pim.target_profile = #pim.target_profile<target = "aimx",
    num_units = 1, banks_per_unit = 256, cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 268435456, local_capacity_limit_bytes = 0>} {
func.func @vacc_groups(%A: memref<*xbf16>, %B: memref<*xbf16>, %C: memref<*xf32>)
    attributes {pim.block = #pim.block<shape = [8, 512, 128]>,
                pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  %k2 = arith.constant 2 : i32
  %k3 = arith.constant 3 : i32
  %k5 = arith.constant 5 : i32
  %k7 = arith.constant 7 : i32
  %zero = arith.constant 0.0 : f32
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [2, 2, 2]>} {
  ^bb0(%dm: index, %dn: index, %dk: index):
    pim.place %A start[%c0, %c0] size[%c16, %c256]
        allocation_shape[%c16, %c256] strides[%c256, %c1]
        {direction = "in", access = 0 : i64} : memref<*xbf16>
    pim.place %B start[%c0, %c0] size[%c256, %c1024]
        allocation_shape[%c256, %c1024] strides[%c1024, %c1]
        {direction = "in", access = 1 : i64} : memref<*xbf16>
    pim.place %C start[%c0, %c0] size[%c16, %c1024]
        allocation_shape[%c16, %c1024] strides[%c1024, %c1]
        {direction = "out", access = 2 : i64} : memref<*xf32>
    scf.for %gm = %c0 to %c2 step %c1 {
      %m = arith.muli %gm, %c8 : index
      scf.for %gn = %c0 to %c2 step %c1 {
        %n = arith.muli %gn, %c512 : index
        %empty = tensor.empty() : tensor<8x512xf32>
        %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x512xf32>) -> tensor<8x512xf32>
        %partial = scf.for %gk = %k3 to %k7 step %k2
            iter_args(%acc = %init) -> tensor<8x512xf32> : i32 {
          %distance = arith.subi %gk, %k3 : i32
          %slot_i32 = arith.divsi %distance, %k2 : i32
          %slot = arith.index_cast %slot_i32 : i32 to index
          %k = arith.muli %slot, %c128 : index
          %aRow = arith.muli %m, %c256 : index
          %aOffset = arith.addi %aRow, %k : index
          %aa = pim.access %A offset %aOffset sizes [8, 128]
              strides[%c256, %c1] origin[%m, %k] {id = 0 : i64} : memref<*xbf16> -> !pim.access
          %a = pim.load %aa : !pim.access -> tensor<8x128xbf16>
          %bRow = arith.muli %k, %c1024 : index
          %bOffset = arith.addi %bRow, %n : index
          %ba = pim.access %B offset %bOffset sizes [128, 512]
              strides[%c1024, %c1] origin[%k, %n] {id = 1 : i64} : memref<*xbf16> -> !pim.access
          %b = pim.load %ba : !pim.access -> tensor<128x512xbf16>
          %next = linalg.matmul ins(%a, %b : tensor<8x128xbf16>, tensor<128x512xbf16>)
              outs(%acc : tensor<8x512xf32>) -> tensor<8x512xf32>
          scf.yield %next : tensor<8x512xf32>
        } {pim.reduction = "add", pim.temporal_reduction}
        %cRow = arith.muli %m, %c1024 : index
        %cOffset = arith.addi %cRow, %n : index
        %ca = pim.access %C offset %cOffset sizes [8, 512]
            strides[%c1024, %c1] origin[%m, %n] {id = 2 : i64} : memref<*xf32> -> !pim.access
        pim.combine %partial into %ca kind "add" : tensor<8x512xf32>, !pim.access
      } {pim.axis = #pim.axis<"n">}
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
