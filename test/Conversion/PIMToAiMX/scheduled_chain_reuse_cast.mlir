// Triton computes block coordinates in i32, so the placement start and the
// access origin both pass through index_cast around the scheduling term. Their
// difference must still cancel the round/device coordinate, leaving a
// resident-local descriptor offset: the chain is built once and reused unchanged.
// RUN: triton-shared-opt --mlir-print-op-generic --pim-map-devices='p-m=2 p-n=1 p-k=1' --pim-place-data --cse --canonicalize --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=STUB --implicit-check-not=aimx_patch_
// RUN: triton-shared-opt --mlir-print-op-generic --pim-map-devices='p-m=2 p-n=1 p-k=1' --pim-place-data --cse --canonicalize --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize %s | triton-shared-translate --aimx-to-descriptors | FileCheck %s --check-prefix=DESC --implicit-check-not=aimx_patch_
// STUB: extern "C" void cast_add
// STUB: aimx_resident_acquire
// STUB: if (!_resident->ready) {
// STUB: aimx_prog_cast_add_device0_elementwise_
// STUB: descriptor upload failed
// STUB: _resident->ready =
// STUB: pim_exec
// DESC: static uint32_t aimx_prog_cast_add_device0_elementwise_
// The model-session stub takes session contexts and per-instance tensors instead of a kernel key.
// RUN: triton-shared-opt --mlir-print-op-generic --pim-map-devices='p-m=2 p-n=1 p-k=1' --pim-place-data --cse --canonicalize --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize %s | triton-shared-translate --aimx-to-model-stub | FileCheck %s --check-prefix=MODEL --implicit-check-not=aimx_resident_acquire
// MODEL: extern "C" void cast_add(aimx_model_call *_model,
// MODEL: for (_model->batch_base = 0; _model->batch_base < _model->instances;
// MODEL: num_threads(_model->batch_size * 2)
// MODEL: if (_instance >= _model->instances) continue;
// MODEL: *a0 = (float*)_model->pointers[(size_t)_instance * _model->arguments + 0];
// MODEL: int _fpga = _model->fpgas[_job];
// MODEL: aimx_resident_context *_resident = _model->contexts[_job];
// MODEL: if (_model->weights[0] == _model->upload) {
// MODEL: if (!_resident->ready) {
// MODEL: if (!_model->upload) {
// MODEL: pim_exec
// MODEL: pim_model_slot_count =
// MODEL: pim_model_device_count = 2;

#identity = affine_map<(m,n) -> (m,n)>
module attributes {pim.target_profile = #pim.target_profile<target = "aimx",
    num_units = 2, banks_per_unit = 256, cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 268435456, local_capacity_limit_bytes = 0>} {
func.func @cast_add(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
                    %rowGroups: index, %colGroups: index)
    attributes {pim.block = #pim.block<shape = [2, 8, 1]>,
                pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
                pim.execution_tile = #pim.execution_tile<groups = [2, 2, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c2_i32 = arith.constant 2 : i32
  %stride = arith.muli %colGroups, %c8 : index
  scf.for %gm = %c0 to %rowGroups step %c1 {
    %gm_i32 = arith.index_cast %gm : index to i32
    %m_i32 = arith.muli %gm_i32, %c2_i32 : i32
    %m = arith.index_cast %m_i32 : i32 to index
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
