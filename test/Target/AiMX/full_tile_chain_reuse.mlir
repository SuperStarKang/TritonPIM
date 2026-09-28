// A full descriptor-program loop has the same topology on every visit, even
// though its trip bound is expressed as a dynamic clamp by the scheduler.
// RUN: triton-shared-translate --aimx-to-runtime-stub %s | FileCheck %s --check-prefix=FULL --implicit-check-not=aimx_patch_
// RUN: triton-shared-translate --aimx-to-descriptors %s | FileCheck %s --check-prefix=BUILD --implicit-check-not=aimx_patch_
// Changing actual addresses still needs patching and upload on full tiles.
// RUN: sed 's/%pmm, %data, %zero, %trips/%pmm, %data, %offset, %trips/' %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=PATCH
// RUN: sed 's/%pmm, %data, %zero, %trips/%pmm, %data, %offset, %trips/' %s | triton-shared-translate --aimx-to-descriptors | FileCheck %s --check-prefix=PATCH-DESC
// A value computed inside construction cannot appear in an enclosing guard.
// RUN: sed '/^[[:space:]]*%trips = arith.minsi/d; /scf.execute_region no_inline {/a\        %trips = arith.minsi %two, %remaining : index' %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=REBUILD --implicit-check-not="if (!_resident->ready)" --implicit-check-not="if (_resident->ready)" --implicit-check-not=aimx_patch_
// FULL: if (!(([[BOUND:v[0-9]+]] == 2))) _resident->ready = false;
// FULL-NEXT: if (!_resident->ready) {
// FULL: .idx = 0;
// FULL: aimx_prog_tile_copy(
// FULL: descriptor upload failed
// FULL-NEXT: _resident->ready = ([[BOUND]] == 2);
// FULL-NEXT: }
// FULL: pim_exec
// BUILD: static uint32_t aimx_prog_tile_copy(
// PATCH: if (!(([[BOUND:v[0-9]+]] == 2))) _resident->ready = false;
// PATCH: .idx = 0;
// PATCH-NEXT: if (_resident->ready) {
// PATCH: aimx_patch_tile_copy(
// PATCH: } else {
// PATCH: aimx_prog_tile_copy(
// PATCH: descriptor upload failed
// PATCH-NEXT: _resident->ready = ([[BOUND]] == 2);
// PATCH: pim_exec
// PATCH-DESC: static uint32_t aimx_patch_tile_copy(
// PATCH-DESC: aimx_patch_descriptor_addresses(
// REBUILD: .idx = 0;
// REBUILD: aimx_prog_tile_copy(
// REBUILD: descriptor upload failed
// REBUILD: pim_exec

module {
  aimx.descriptor_program @tile_copy batch 1 {
  ^bb0(%src: !aimx.buffer, %dst: !aimx.buffer, %offset: index, %count: index):
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %stride = arith.constant 512 : index
    scf.for %i = %zero to %count step %one {
      %base = arith.muli %i, %stride : index
      %address = arith.addi %base, %offset : index
      aimx.desc src(%src, %address) dst(%dst, %base) len 512 opcode copy
    }
  }
  func.func @tile_copy_host(%remaining: index, %offset: index) {
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %set = aimx.acquire 1 : !aimx.device_set
    aimx.foreach_device %set : !aimx.device_set axes ["m", "n", "k"] counts [1, 1, 1] attributes {aimx.persistent} {
    ^bb0(%device: !aimx.device, %dm: index, %dn: index, %dk: index):
      %pmm = aimx.alloc %device bytes 196608 align 262144 kind pmm : !aimx.buffer
      %chain = aimx.alloc %device bytes 1024 align 512 kind descriptor : !aimx.buffer
      %data = aimx.alloc %device bytes 32768 align 512 kind data : !aimx.buffer
      %trips = arith.minsi %two, %remaining : index
      scf.execute_region no_inline {
        aimx.build_descriptors @tile_copy(%pmm, %data, %zero, %trips) into %chain
            : !aimx.buffer, !aimx.buffer, index, index
        scf.yield
      }
      aimx.invoke %device, %chain
    }
    return
  }
}
