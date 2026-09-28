// Exercise actual Common scheduling/placement, not hand-written constant bounds.
// rowGroups=4 and colGroups=10 give five full tiles per M device. Other sizes
// keep the original min(tileGroups, remainingGroups) loops, including zero trips.
// Like the backend, leave constant allocation products and loop metadata in
// Common IR: target-local scalar folding must suffice without canonicalizing it.
// RUN: triton-shared-opt --mlir-print-op-generic --pim-map-devices='p-m=2 p-n=1 p-k=1' --pim-place-data --cse --convert-pim-to-aimx --cse --canonicalize %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=STUB --implicit-check-not=aimx_patch_
// RUN: triton-shared-opt --mlir-print-op-generic --pim-map-devices='p-m=2 p-n=1 p-k=1' --pim-place-data --cse --canonicalize --convert-pim-to-aimx --cse --canonicalize %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=STUB --implicit-check-not=aimx_patch_
// RUN: triton-shared-opt --mlir-print-op-generic --pim-map-devices='p-m=2 p-n=1 p-k=1' --pim-place-data --cse --canonicalize --convert-pim-to-aimx --cse --canonicalize %s | triton-shared-translate --aimx-to-descriptors | FileCheck %s --check-prefix=DESC --implicit-check-not=aimx_patch_
// No persistent policy is forced onto the transient path.
// RUN: triton-shared-opt --mlir-print-op-generic --pim-map-devices='p-m=2 p-n=1 p-k=1' --pim-place-data --cse --canonicalize --convert-pim-to-aimx=persistent=false --cse --canonicalize %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=TRANSIENT --implicit-check-not=_resident
// STUB: extern "C" void scheduled_add
// STUB: num_threads(2)
// STUB: aimx_resident_acquire
// Both M and N must be full; using just the total record count is insufficient.
// STUB: if (!(([[BOUND0:v[0-9]+]] == 2) && ([[BOUND1:v[0-9]+]] == 2))) _resident->ready = false;
// STUB-NEXT: if (!_resident->ready) {
// STUB: .idx = 0;
// STUB: aimx_prog_scheduled_add_device0_elementwise_
// STUB: descriptor upload failed
// Tail construction does not mark its overwritten storage as a cached full tile.
// STUB-NEXT: _resident->ready = ([[BOUND0]] == 2) && ([[BOUND1]] == 2);
// STUB-NEXT: }
// Staging and execution stay outside the once-per-full-chain construction guard.
// STUB: {  // aimx.stage
// STUB: pim_exec
// DESC: static uint32_t aimx_prog_scheduled_add_device0_elementwise_
// TRANSIENT: aimx_prog_scheduled_add_device0_elementwise_
// TRANSIENT: descriptor upload failed
// TRANSIENT: pim_exec

#identity = affine_map<(m,n) -> (m,n)>
module attributes {pim.target_profile = #pim.target_profile<target = "aimx",
    num_units = 2, banks_per_unit = 256, cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 268435456, local_capacity_limit_bytes = 0>} {
func.func @scheduled_add(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
                         %rowGroups: index, %colGroups: index)
    attributes {pim.block = #pim.block<shape = [2, 8, 1]>,
                pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
                pim.execution_tile = #pim.execution_tile<groups = [2, 2, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %stride = arith.muli %colGroups, %c8 : index
  scf.for %gm = %c0 to %rowGroups step %c1 {
    %m = arith.muli %gm, %c2 : index
    scf.for %gn = %c0 to %colGroups step %c1 {
      %n = arith.muli %gn, %c8 : index
      %row = arith.muli %m, %stride : index
      %offset = arith.addi %row, %n : index
      %aa = pim.access %A offset %offset sizes [2, 8] strides[%stride, %c1]
          origin[%m, %n] {id = 0 : i64} : memref<*xf32> -> !pim.access
      %ba = pim.access %B offset %offset sizes [2, 8] strides[%stride, %c1]
          origin[%m, %n] {id = 1 : i64} : memref<*xf32> -> !pim.access
      %ca = pim.access %C offset %offset sizes [2, 8] strides[%stride, %c1]
          origin[%m, %n] {id = 2 : i64} : memref<*xf32> -> !pim.access
      %a = pim.load %aa : !pim.access -> tensor<2x8xf32>
      %b = pim.load %ba : !pim.access -> tensor<2x8xf32>
      %empty = tensor.empty() : tensor<2x8xf32>
      %result = linalg.generic {
          indexing_maps = [#identity, #identity, #identity],
          iterator_types = ["parallel", "parallel"]}
          ins(%a, %b : tensor<2x8xf32>, tensor<2x8xf32>)
          outs(%empty : tensor<2x8xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %value = arith.addf %lhs, %rhs : f32
        linalg.yield %value : f32
      } -> tensor<2x8xf32>
      pim.store %result, %ca : tensor<2x8xf32>, !pim.access
    } {pim.axis = #pim.axis<"n">}
  } {pim.axis = #pim.axis<"m">}
  return
}
}
