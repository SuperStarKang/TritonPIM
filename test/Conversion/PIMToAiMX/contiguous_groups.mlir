// Equal tensor types must not unify the unrelated A, B and C physical layouts.
// Eight logical GEMMs (2 M x 2 N x 2 K groups) form one invocation.
// RUN: triton-shared-opt --pim-verify --pim-form-host-plan --convert-pim-to-aimx %s | FileCheck %s --check-prefix=IR
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx --mlir-print-op-generic %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=STUB
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx --mlir-print-op-generic %s | triton-shared-translate --aimx-to-descriptors | FileCheck %s --check-prefix=DESC
// IR-LABEL: func.func @contiguous_groups
// Complete allocation plan: PMM, chain, placements, and all SSA workspaces.
// IR: aimx.allocations = array<i64: 196608, 51200, 262144, 1048576, 32768, 8192, 8192, 131072>
// IR-SAME: aimx.bank_bytes = 6816 : i64
// IR-SAME: aimx.descriptor_bytes = 51200 : i64
// IR: aimx.alloc {{.*}} bytes 262144 align 8192 kind data
// IR: aimx.alloc {{.*}} bytes 1048576 align 8192 kind data
// IR: aimx.alloc {{.*}} bytes 32768 align 8192 kind data
// IR: scf.execute_region no_inline
// IR: scf.for
// IR: scf.for
// IR: scf.for
// IR: aimx.require {{.*}}, "AiMX access must start at a contiguous group boundary"
// IR: aimx.build_descriptors @contiguous_groups_device0_matmul_relayout_
// IR: aimx.build_descriptors @contiguous_groups_device0_matmul_accumulate_
// IR: aimx.build_descriptors @contiguous_groups_device0_matmul_drain_
// IR: aimx.stage %arg0 {{.*}}group_layout = array<i64: 8, 8, 8, 4096, 2, 2>
// IR: aimx.stage %arg1 {{.*}}group_layout = array<i64: 8, 8, 256, 512, 2, 2>
// IR: aimx.invoke
// IR: aimx.unstage {{.*}}group_layout = array<i64: 8, 8, 8, 512, 2, 2>
// IR-NOT: aimx.invoke
// IR: return
// STUB: extern "C" void contiguous_groups
// STUB: aimx_require({{.*}}, "AiMX access must start at a contiguous group boundary");
// STUB: contiguous group layout
// STUB: * 32768ll;
// STUB: contiguous group layout
// STUB: * 131072ll;
// STUB: pim_exec
// STUB: // aimx.unstage contiguous groups
// STUB: datacopy_pim2cpu({{.*}}, 32768, _fpga)
// STUB: / 8ll) * 2ll + {{.*}} / 8ll) * 4096ll + ({{.*}} % 8ll) * 512ll;
// An 8x512 compact C group is one 8192-byte copy, spanning all eight rows.
// DESC-LABEL: static uint32_t aimx_prog_contiguous_groups_device0_copy_{{[0-9]+}}(
// DESC: int64_t {{v[0-9]+}} = 4096;
// DESC: for (int64_t {{.*}} < {{v[0-9]+}}; {{.*}} += {{v[0-9]+}}) {
// DESC-NOT: for (
// DESC: 8192u, 0x0u, fpga_id, code);  // copy

module attributes {pim.target_profile = #pim.target_profile<target = "aimx",
    num_units = 1, banks_per_unit = 256, cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 268435456, local_capacity_limit_bytes = 0>} {
func.func @contiguous_groups(%A: memref<*xbf16>, %B: memref<*xbf16>, %C: memref<*xbf16>)
    attributes {pim.block = #pim.block<shape = [8, 8, 8]>,
                pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.0 : bf16
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [2, 2, 2]>} {
  ^bb0(%dm: index, %dn: index, %dk: index):
    pim.place %A start[%c0, %c0] size[%c16, %c16]
        allocation_shape[%c16, %c16] strides[%c16, %c1]
        {direction = "in", access = 0 : i64} : memref<*xbf16>
    pim.place %B start[%c0, %c0] size[%c16, %c16]
        allocation_shape[%c16, %c16] strides[%c16, %c1]
        {direction = "in", access = 1 : i64} : memref<*xbf16>
    pim.place %C start[%c0, %c0] size[%c16, %c16]
        allocation_shape[%c16, %c16] strides[%c16, %c1]
        {direction = "out", access = 2 : i64} : memref<*xbf16>
    scf.for %gm = %c0 to %c2 step %c1 {
      %m = arith.muli %gm, %c8 : index
      scf.for %gn = %c0 to %c2 step %c1 {
        %n = arith.muli %gn, %c8 : index
        %empty = tensor.empty() : tensor<8x8xbf16>
        %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<8x8xbf16>) -> tensor<8x8xbf16>
        %partial = scf.for %gk = %c0 to %c2 step %c1
            iter_args(%acc = %init) -> tensor<8x8xbf16> {
          %k = arith.muli %gk, %c8 : index
          %aRow = arith.muli %m, %c16 : index
          %aOffset = arith.addi %aRow, %k : index
          %aa = pim.access %A offset %aOffset sizes [8, 8]
              strides[%c16, %c1] origin[%m, %k] {id = 0 : i64} : memref<*xbf16> -> !pim.access
          %a = pim.load %aa : !pim.access -> tensor<8x8xbf16>
          %bRow = arith.muli %k, %c16 : index
          %bOffset = arith.addi %bRow, %n : index
          %ba = pim.access %B offset %bOffset sizes [8, 8]
              strides[%c16, %c1] origin[%k, %n] {id = 1 : i64} : memref<*xbf16> -> !pim.access
          %b = pim.load %ba : !pim.access -> tensor<8x8xbf16>
          %next = linalg.matmul ins(%a, %b : tensor<8x8xbf16>, tensor<8x8xbf16>)
              outs(%acc : tensor<8x8xbf16>) -> tensor<8x8xbf16>
          scf.yield %next : tensor<8x8xbf16>
        } {pim.reduction = "add", pim.temporal_reduction}
        %cRow = arith.muli %m, %c16 : index
        %cOffset = arith.addi %cRow, %n : index
        %ca = pim.access %C offset %cOffset sizes [8, 8]
            strides[%c16, %c1] origin[%m, %n] {id = 2 : i64} : memref<*xbf16> -> !pim.access
        pim.store %partial, %ca : tensor<8x8xbf16>, !pim.access
      } {pim.axis = #pim.axis<"n">}
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
