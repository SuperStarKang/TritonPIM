// Two groups on each axis, runtime tails and nonzero strided host origins.
// RUN: triton-shared-translate --aimx-to-runtime-stub %s | FileCheck %s
// RUN: sed 's/8, 512, 2, 2/7, 512, 2, 2/g' %s | not triton-shared-opt 2>&1 | FileCheck %s --check-prefix=ALIGN
// RUN: sed 's/8, 512, 2, 2/8, 512, 0, 2/g' %s | not triton-shared-opt 2>&1 | FileCheck %s --check-prefix=COUNT
// RUN: sed 's/pitch %pitch into/pitch %pitch group_rows [3, 8] into/' %s | not triton-shared-opt 2>&1 | FileCheck %s --check-prefix=MIXED
// CHECK: extern "C" void packed_io
// CHECK: <= 6ll && {{.*}} <= 10ll && {{.*}} == 16ll && {{.*}} == 1024ll
// CHECK: for (int64_t [[GR:_gr[0-9]+]] = 0; [[GR]] < 2ll; ++[[GR]]) {
// CHECK: if ({{.*}} > 3ll)
// CHECK: for (int64_t [[GC:_gc[0-9]+]] = 0; [[GC]] < 2ll; ++[[GC]]) {
// CHECK: if ({{.*}} > 5ll)
// CHECK: + ([[GR]] * 2ll + [[GC]]) * 4096ll;
// CHECK: + ({{.*}} + ([[GR]] * 3ll + {{.*}})) *
// CHECK: aimx_to_bf16({{.*}}[([[GC]] * 5ll + {{.*}}) * {{.*}}])
// CHECK: memset({{.*}}512ll -
// CHECK: memset({{.*}}8ll -
// CHECK: input upload failed
// CHECK: // aimx.unstage contiguous groups
// CHECK: datacopy_pim2cpu({{.*}}, 32768, _fpga)
// CHECK: / 3ll) * 2ll + {{.*}} / 5ll) * 4096ll + ({{.*}} % 3ll) * 512ll;
// CHECK: + aimx_from_bf16(
// CHECK: = aimx_from_bf16(
// ALIGN: each contiguous group must contain a multiple of 4096 elements
// COUNT: group_layout needs representable
// MIXED: group_layout cannot be combined with periodic group padding

module {
  func.func @packed_io(%src: memref<*xf32>, %dst: memref<*xf32>,
                      %rows: index, %cols: index, %row: index, %col: index,
                      %rowStride: index, %colStride: index, %accumulate: index) {
    %padRows = arith.constant 16 : index
    %pitch = arith.constant 1024 : index
    %set = aimx.acquire 1 : !aimx.device_set
    aimx.foreach_device %set : !aimx.device_set axes ["m", "n", "k"] counts [1, 1, 1] {
    ^bb0(%device: !aimx.device, %dm: index, %dn: index, %dk: index):
      %pmm = aimx.alloc %device bytes 196608 align 262144 kind pmm : !aimx.buffer
      %data = aimx.alloc %device bytes 32768 align 8192 kind data : !aimx.buffer
      aimx.stage %src origin[%row, %col] sizes[%rows, %cols]
          strides[%rowStride, %colStride] pad_to[%padRows, %pitch] pitch %pitch into %data
          {element_type = bf16, group_layout = array<i64: 3, 5, 8, 512, 2, 2>}
          : memref<*xf32> -> !aimx.buffer
      aimx.unstage %data into %dst origin[%row, %col] sizes[%rows, %cols]
          strides[%rowStride, %colStride] pitch %pitch accumulate %accumulate
          {element_type = bf16, group_layout = array<i64: 3, 5, 8, 512, 2, 2>}
          : !aimx.buffer, memref<*xf32>
    }
    return
  }
}
