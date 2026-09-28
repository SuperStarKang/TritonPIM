// Empty/all shared_over sets select separate slots/one common slot respectively.
// Each round clears the slot even when its valid input shrinks.
// RUN: triton-shared-translate --upmem-to-host %s | FileCheck %s --implicit-check-not=malloc --implicit-check-not='free('

// CHECK: int8_t* [[SEPARATE:v[0-9]+]] = (int8_t*)malloc(32 * sizeof(int8_t));
// CHECK: int8_t* [[SHARED:v[0-9]+]] = (int8_t*)malloc(8 * sizeof(int8_t));
// CHECK: for (int64_t [[ROUND:v[0-9]+]] =
// CHECK: int64_t [[VALID:v[0-9]+]] = {{.*}} - [[ROUND]];
// CHECK: for (int64_t _u = 0; _u < 4; ++_u)
// CHECK: int64_t [[M:v[0-9]+]] = (_u / 2) % 2;
// CHECK: int64_t [[N:v[0-9]+]] = (_u / 1) % 2;
// CHECK: [[S:v[0-9]+]]_key = [[S]]_key * 2 + [[M]];
// CHECK-NEXT: [[S]]_key = [[S]]_key * 2 + [[N]];
// CHECK-NEXT: int8_t* [[S]] = [[SEPARATE]] + [[S]]_key * 8;
// CHECK-NEXT: _pim_stop(0);
// CHECK-NEXT: _pim_start(8);
// CHECK-NEXT: memset([[S]], 0, 8);
// CHECK: _p0[_u] = [[S]];
// CHECK: int64_t [[B:v[0-9]+]]_key = 0;
// CHECK-NEXT: int8_t* [[B]] = [[SHARED]] + [[B]]_key * 8;
// CHECK-NEXT: if ([[M]] == 0 && [[N]] == 0) {
// CHECK: memset([[B]], 0, 8);
// CHECK: if (_t0 * 8 + _e0 < [[VALID]])
// CHECK: _p1[_u] = [[B]];
// CHECK: dpu_push_xfer(_set, DPU_XFER_TO_DPU, "separate", 0, 8,
// CHECK: dpu_push_xfer(_set, DPU_XFER_TO_DPU, "shared", 0, 8,
// CHECK: free([[SHARED]]);
// CHECK: free([[SEPARATE]]);

upmem.mram @separate : memref<8xi8, #upmem<space mram>>
upmem.mram @shared : memref<8xi8, #upmem<space mram>>

func.func @pack(%A: memref<*xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %set = upmem.acquire 4, @kernel : !upmem.set
  %assignment = upmem.bind_devices %set partitions [2, 2] device_mapping [1, 2]
      : !upmem.set -> !upmem.assignment
  %separate = memref.alloc() : memref<4x8xi8>
  %shared = memref.alloc() : memref<1x8xi8>
  scf.for %round = %c0 to %c2 step %c1 {
    %valid = arith.subi %c8, %round : index
    upmem.foreach_unit %set assignment %assignment : !upmem.set axes ["m", "n"] counts [2, 2] {
    ^bb0(%dpu: !upmem.unit, %pm: index, %pn: index):
      %row = arith.muli %pm, %c2 : index
      %slot = arith.addi %row, %pn : index
      %origin = arith.muli %slot, %c8 : index
      %s = upmem.stage %A origin[%origin] sizes[%valid] strides[%c1]
          pad_to[%c8] tile [8] into %separate
          : memref<*xi8>, memref<4x8xi8> -> memref<8xi8>
      upmem.to_mram %dpu, %s, @separate : memref<8xi8>
      %b = upmem.stage %A origin[%c0] sizes[%valid] strides[%c1]
          pad_to[%c8] tile [8] into %shared {shared_over = ["m", "n"]}
          : memref<*xi8>, memref<1x8xi8> -> memref<8xi8>
      upmem.to_mram %dpu, %b, @shared : memref<8xi8>
    }
  }
  memref.dealloc %shared : memref<1x8xi8>
  memref.dealloc %separate : memref<4x8xi8>
  return
}
