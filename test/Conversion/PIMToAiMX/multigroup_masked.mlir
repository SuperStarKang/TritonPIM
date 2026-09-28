// Rectangular access masks project to the whole execution-tile placement.
// This is a copy, not a GEMM pattern. Both axes contain two groups, and the
// placement starts/ends are runtime values (including device/scheduling tails).
// RUN: triton-shared-opt --convert-pim-to-aimx --cse --canonicalize --mlir-print-op-generic %s -o %t.aimx
// RUN: triton-shared-opt %t.aimx | FileCheck %s --check-prefix=IR
// RUN: triton-shared-translate --aimx-to-runtime-stub %t.aimx | FileCheck %s --check-prefix=STUB
// A partial/empty execution tile keeps fixed allocation and clips only I/O.
// RUN: sed '/^    %remaining_m =/,$ {s/%M/%c1/g; s/%N/%c8/g; s/%start_m/%c0/g; s/%start_n/%c0/g;}' %s | triton-shared-opt --convert-pim-to-aimx --cse --canonicalize | FileCheck %s --check-prefix=TAIL
// RUN: sed '/^    %remaining_m =/,$ {s/%M/%c0/g; s/%N/%c8/g; s/%start_m/%c0/g; s/%start_n/%c0/g;}' %s | triton-shared-opt --convert-pim-to-aimx --cse --canonicalize | FileCheck %s --check-prefix=EMPTY
// A repeated partial mask cannot be replaced by the bounding rectangle.
// RUN: sed 's/valid\[%valid_m, %valid_n\]/valid[%c1, %valid_n]/g' %s | not triton-shared-opt --convert-pim-to-aimx 2>&1 | FileCheck %s --check-prefix=HOLES
// The bound must clip this access coordinate, not an unrelated group axis.
// RUN: sed 's/arith.subi %N, %n/arith.subi %N, %m/' %s | not triton-shared-opt --convert-pim-to-aimx 2>&1 | FileCheck %s --check-prefix=HOLES
// IR-LABEL: func.func @masked_groups
// IR: %[[MREST:.*]] = arith.subi %arg2, %arg4 : index
// IR: %[[MNONNEG:.*]] = arith.maxsi %[[MREST]], {{.*}} : index
// IR: %[[ROWS:.*]] = arith.minsi {{.*}}%[[MNONNEG]]{{.*}} : index
// IR: %[[NREST:.*]] = arith.subi %arg3, %arg5 : index
// IR: %[[NNONNEG:.*]] = arith.maxsi %[[NREST]], {{.*}} : index
// IR: %[[COLS:.*]] = arith.minsi {{.*}}%[[NNONNEG]]{{.*}} : index
// IR: scf.execute_region no_inline
// IR: scf.for
// IR: scf.for
// IR: aimx.build_descriptors
// IR: aimx.stage %arg0 origin[%arg4, %arg5] sizes[%[[ROWS]], %[[COLS]]]
// IR-SAME: group_cols [8, 4096]
// IR: aimx.invoke
// IR: aimx.unstage {{.*}} sizes[%[[ROWS]], %[[COLS]]]
// IR-NOT: aimx.invoke
// IR: return
// STUB: extern "C" void masked_groups
// STUB: {  // aimx.stage
// STUB: datacopy_cpu2pim
// STUB: pim_exec
// STUB: datacopy_pim2cpu
// TAIL-LABEL: func.func @masked_groups
// TAIL-DAG: %[[ONE:.*]] = arith.constant 1 : index
// TAIL-DAG: %[[EIGHT:.*]] = arith.constant 8 : index
// TAIL: aimx.stage %arg0 {{.*}} sizes[%[[ONE]], %[[EIGHT]]]
// TAIL: aimx.invoke
// TAIL: aimx.unstage {{.*}} sizes[%[[ONE]], %[[EIGHT]]]
// EMPTY-LABEL: func.func @masked_groups
// EMPTY: %[[ZERO:.*]] = arith.constant 0 : index
// EMPTY: aimx.stage %arg0 {{.*}} sizes[%[[ZERO]],
// EMPTY: aimx.unstage {{.*}} sizes[%[[ZERO]],
// HOLES: AiMX masks spanning multiple resident groups need a shared rectangular bound

module attributes {pim.target_profile = #pim.target_profile<target = "aimx",
    num_units = 1, banks_per_unit = 256, cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 268435456, local_capacity_limit_bytes = 0>} {
func.func @masked_groups(%A: memref<*xbf16>, %C: memref<*xbf16>,
                          %M: index, %N: index,
                          %start_m: index, %start_n: index)
    attributes {pim.block = #pim.block<shape = [2, 8, 1]>,
                pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [2, 2, 1]>} {
  ^bb0(%dm: index, %dn: index, %dk: index):
    %remaining_m = arith.subi %M, %start_m : index
    %nonnegative_m = arith.maxsi %remaining_m, %c0 : index
    %rows = arith.minsi %nonnegative_m, %c4 : index
    %remaining_n = arith.subi %N, %start_n : index
    %nonnegative_n = arith.maxsi %remaining_n, %c0 : index
    %cols = arith.minsi %nonnegative_n, %c16 : index
    pim.place %A start[%start_m, %start_n] size[%rows, %cols]
        allocation_shape[%c4, %c16] strides[%N, %c1]
        {direction = "in", access = 0 : i64} : memref<*xbf16>
    pim.place %C start[%start_m, %start_n] size[%rows, %cols]
        allocation_shape[%c4, %c16] strides[%N, %c1]
        {direction = "out", access = 1 : i64} : memref<*xbf16>
    scf.for %gm = %c0 to %c2 step %c1 {
      %mo = arith.muli %gm, %c2 : index
      %m = arith.addi %start_m, %mo : index
      // Intersection form, with redundant clamp wrappers and reversed operands.
      %end_m = arith.addi %m, %c2 : index
      %clipped_m = arith.minsi %M, %end_m : index
      %begin_m = arith.maxsi %m, %clipped_m : index
      %length_m = arith.subi %begin_m, %m : index
      %positive_m = arith.maxsi %length_m, %c0 : index
      %valid_m = arith.minsi %c2, %positive_m : index
      scf.for %gn = %c0 to %c2 step %c1 {
        %no = arith.muli %gn, %c8 : index
        %n = arith.addi %start_n, %no : index
        // Remaining-distance form, including a redundant min(size).
        %rest_n = arith.subi %N, %n : index
        %clipped_n = arith.minsi %c8, %rest_n : index
        %positive_n = arith.maxsi %c0, %clipped_n : index
        %valid_n = arith.minsi %positive_n, %c8 : index
        %row = arith.muli %m, %N : index
        %offset = arith.addi %row, %n : index
        %aa = pim.access %A offset %offset sizes [2, 8] strides[%N, %c1]
            origin[%m, %n] valid[%valid_m, %valid_n] pad %pad : bf16
            {id = 0 : i64} : memref<*xbf16> -> !pim.access
        %ca = pim.access %C offset %offset sizes [2, 8] strides[%N, %c1]
            origin[%m, %n] valid[%valid_m, %valid_n]
            {id = 1 : i64} : memref<*xbf16> -> !pim.access
        %a = pim.load %aa : !pim.access -> tensor<2x8xbf16>
        pim.store %a, %ca : tensor<2x8xbf16>, !pim.access
      } {pim.axis = #pim.axis<"n">}
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
