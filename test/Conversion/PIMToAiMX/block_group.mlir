// Physical batch changes only hardware descriptors, not Common scheduling.
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx=batch=2 %S/convert.mlir | FileCheck %s --check-prefix=PAD
// RUN: sed 's/partitions = \[2, 1, 1\]/partitions = [1, 1, 2]/' %S/convert.mlir | not triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx 2>&1 | FileCheck %s --check-prefix=SPLITK
// RUN: sed '/pim.combine %partial/i\          "test.side_effect"() : () -> ()' %S/convert.mlir | not triton-shared-opt --allow-unregistered-dialect --pim-form-host-plan --convert-pim-to-aimx 2>&1 | FileCheck %s --check-prefix=EFFECT
// RUN: sed 's/bank_capacity_limit_bytes = 268435456/bank_capacity_limit_bytes = 1/' %S/convert.mlir | not triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx 2>&1 | FileCheck %s --check-prefix=CAPACITY
// Source masks not represented by the whole placement cannot be discarded.
// RUN: sed 's/origin\[%m, %n\]/origin[%m, %n] valid[%c2, %c1]/g' %S/elementwise.mlir | not triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx 2>&1 | FileCheck %s --check-prefix=MASK
// PAD: aimx.descriptor_program @matmul_device0_matmul_{{[0-9]+}} batch 2
// PAD: len 16384 opcode bc|tp
// PAD: len 4096 opcode b_mask|matmul|pim_mode|stride
// PAD: aimx.foreach_device {{.*}} counts [2, 1, 1]
// PAD: scf.execute_region no_inline
// PAD: aimx.stage {{.*}} group_rows [96, 256] group_cols [32, 4096]
// SPLITK: AiMX cross-device pim.combine is not implemented
// EFFECT: AiMX has no lowering for this Common operation
// CAPACITY: AiMX physical buffers and descriptors exceed per-bank capacity
// MASK: AiMX masks spanning multiple resident groups need a shared rectangular bound
