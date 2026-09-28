// Code generation only: fixed record traversal with changing address/pitch.
// RUN: triton-shared-translate --aimx-to-runtime-stub %s | FileCheck %s --check-prefix=PATCH
// RUN: triton-shared-translate --aimx-to-descriptors %s | FileCheck %s --check-prefix=PATCH-DESC
// A scalar-only loop is not descriptor topology, even with a changing bound.
// RUN: sed '/^[[:space:]]*scf.for %i/i\    %selected = scf.for %probe = %zero to %count step %one iter_args(%state = %offset) -> index { scf.yield %state : index }' %s | sed 's/%base, %offset/%base, %selected/' | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=PATCH
// A loop bound inside descriptor_program is topology, not just an address.
// RUN: sed 's/to %two /to %count /' %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=REBUILD --implicit-check-not="if (_resident->ready)" --implicit-check-not="if (!_resident->ready)"
// RUN: sed 's/to %two /to %count /' %s | triton-shared-translate --aimx-to-descriptors | FileCheck %s --check-prefix=NO-PATCH --implicit-check-not=aimx_patch_
// A changing parameter that is unused by descriptors must not disable reuse.
// RUN: sed 's/%offset, %pitch, %limit/%zero, %fixedPitch, %limit/' %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=FIXED --implicit-check-not=aimx_patch_
// RUN: sed 's/%offset, %pitch, %limit/%zero, %fixedPitch, %limit/' %s | triton-shared-translate --aimx-to-descriptors | FileCheck %s --check-prefix=NO-PATCH --implicit-check-not=aimx_patch_
// The transient path has no retained records to patch.
// RUN: sed 's/ attributes {aimx.persistent}//' %s | triton-shared-translate --aimx-to-descriptors | FileCheck %s --check-prefix=NO-PATCH --implicit-check-not=aimx_patch_
// RUN: FileCheck %s --check-prefix=FIELDS --input-file=%S/../../../pim_backend/backend/aimx_runtime.h
// PATCH: .idx = 0;
// PATCH-NEXT: if (_resident->ready) {
// PATCH-NEXT: {{.*}}.idx = aimx_patch_relocate(
// PATCH-NOT: .next =
// PATCH: } else {
// PATCH-NEXT: {{.*}}.next =
// PATCH-NEXT: {{.*}}.idx = aimx_prog_relocate(
// PATCH: descriptor upload failed
// PATCH-NEXT: _resident->ready = true;
// PATCH: pim_exec
// PATCH-DESC-LABEL: static uint32_t aimx_patch_relocate(
// PATCH-DESC-SAME: pim_isa_32_t *code, uint32_t idx, uint64_t [[SRC:v[0-9]+]], uint64_t [[DST:v[0-9]+]], int64_t {{v[0-9]+}}, int64_t [[PITCH:v[0-9]+]], int64_t {{v[0-9]+}}) {
// PATCH-DESC-NOT: PIM_RD_INSTR_MC
// PATCH-DESC: aimx_patch_descriptor_addresses(code + idx, ([[SRC]] + (uint64_t){{v[0-9]+}}) | ((uint64_t)[[PITCH]] << 33), ([[DST]] + (uint64_t){{v[0-9]+}}) | ((uint64_t)[[PITCH]] << 33));
// PATCH-DESC-NEXT: idx += NUM_CHANNELS;
// PATCH-DESC-NOT: PIM_RD_INSTR_MC
// PATCH-DESC: return idx;
// REBUILD: .idx = 0;
// REBUILD-NEXT: {{.*}}.next =
// REBUILD-NEXT: {{.*}}.idx = aimx_prog_relocate(
// REBUILD: descriptor upload failed
// REBUILD: pim_exec
// FIXED: if (!_resident->ready) {
// FIXED: aimx_prog_relocate(
// FIXED: descriptor upload failed
// FIXED-NEXT: _resident->ready = true;
// FIXED: pim_exec
// NO-PATCH: static uint32_t aimx_prog_relocate(
// FIELDS-LABEL: static inline void aimx_patch_descriptor_addresses(
// FIELDS: for (unsigned channel = 0; channel < NUM_CHANNELS; ++channel) {
// FIELDS-NEXT: code[channel].src_l = (uint32_t)src;
// FIELDS-NEXT: code[channel].src_h = (uint32_t)(src >> 32);
// FIELDS-NEXT: code[channel].dst_l = (uint32_t)dst;
// FIELDS-NEXT: code[channel].dst_h = (uint32_t)(dst >> 32);
// FIELDS-NEXT: }
// FIELDS-NEXT: }

module {
  aimx.descriptor_program @relocate batch 1 {
  ^bb0(%src: !aimx.buffer, %dst: !aimx.buffer, %offset: index, %pitch: index, %count: index):
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %stride = arith.constant 8192 : index
    scf.for %i = %zero to %two step %one {
      %base = arith.muli %i, %stride : index
      %address = arith.addi %base, %offset : index
      aimx.desc src(%src, %address) pitch %pitch dst(%dst, %base) pitch %pitch
          len 512 opcode c_mask|matmul|stride
    }
  }
  func.func @relocation_host(%offset: index, %pitch: index, %limit: index) {
    %zero = arith.constant 0 : index
    %fixedPitch = arith.constant 8192 : index
    %set = aimx.acquire 1 : !aimx.device_set
    aimx.foreach_device %set : !aimx.device_set axes ["m", "n", "k"] counts [1, 1, 1] attributes {aimx.persistent} {
    ^bb0(%device: !aimx.device, %dm: index, %dn: index, %dk: index):
      %pmm = aimx.alloc %device bytes 196608 align 262144 kind pmm : !aimx.buffer
      %chain = aimx.alloc %device bytes 2048 align 512 kind descriptor : !aimx.buffer
      %data = aimx.alloc %device bytes 32768 align 512 kind data : !aimx.buffer
      scf.execute_region no_inline {
        aimx.build_descriptors @relocate(%pmm, %data, %offset, %pitch, %limit) into %chain
            : !aimx.buffer, !aimx.buffer, index, index, index
        scf.yield
      }
      aimx.invoke %device, %chain
    }
    return
  }
}
