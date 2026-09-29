// UPMEM IR이 host C++ / DPU C 소스로 나오는지 확인한다.
// 예시는 device 4개(x축 2 × y축 2)로 나눈 아주 작은 GEMM이다.
// RUN: triton-shared-translate --upmem-to-host   %s | FileCheck %s --check-prefix=HOST
// RUN: triton-shared-translate --upmem-to-device %s | FileCheck %s --check-prefix=DEVICE

upmem.mram @A_mram : memref<4096xi8, #upmem<space mram>>
upmem.mram @C_mram : memref<1024xi32, #upmem<space mram>>
upmem.host_var @tiles_x : index
upmem.host_var @pitch_a : index

// ── host ────────────────────────────────────────────────────────────────────
// 런타임 헤더는 UPMEM target의 것이고, driver가 -I로 잡아 주는 이름과 같아야 한다.
// HOST: #include "upmem_runtime.h"
// HOST: uint32_t pim_timing_version = 3;
// HOST: double pim_timing_ms[13];
// HOST: extern "C" void kernel(int8_t* a0, int32_t* a1, int64_t a2, int64_t a3)
// HOST: memset(pim_timing_ms, 0, sizeof pim_timing_ms);
// 확보와 적재는 런타임이 한 번만 한다.
// HOST: _pim_start(7);
// HOST: _set = upmem_acquire(4, DPU_BINARY);
// HOST-NEXT: _pim_stop(7);
// Pools are allocated once; stage/gather only select and fill their slots.
// HOST: int8_t* [[A:v[0-9]+]] = (int8_t*)malloc(512 * sizeof(int8_t));
// HOST: int32_t* [[C:v[0-9]+]] = (int32_t*)malloc(1024 * sizeof(int32_t));
// HOST: int8_t* _p0[4];
// HOST: for (int64_t _u = 0; _u < 4; ++_u)
// HOST: int64_t [[M:v[0-9]+]] = (_u / 2) % 2;
// HOST: int64_t [[N:v[0-9]+]] = (_u / 1) % 2;
// HOST: [[SA:v[0-9]+]]_key = [[SA]]_key * 2 + [[M]];
// HOST-NEXT: int8_t* [[SA]] = [[A]] + [[SA]]_key * 256;
// HOST-NEXT: if ([[N]] == 0) {
// HOST-NEXT: _pim_stop(0);
// HOST-NEXT: _pim_start(8);
// HOST-NEXT: memset([[SA]], 0, 256);
// HOST-NEXT: _pim_stop(8);
// HOST-NEXT: _pim_start(11);
// HOST: for (int64_t _t0 = 0; _t0 < (({{.*}} + 7) / 8); ++_t0)
// HOST: if (_t0 * 8 + _e0 < {{.*}} && _t1 * 16 + _e1 < {{.*}})
// HOST: [[SA]][{{.*}} * 128 + (_e0 * 16 + _e1)] =
// HOST-NEXT: _pim_stop(11);
// HOST-NEXT: _pim_start(0);
// HOST: _p0[_u] = [[SA]];
// HOST: _pim_stop(0);
// HOST-NEXT: _pim_start(1);
// HOST: dpu_push_xfer(_set, DPU_XFER_TO_DPU, "A_mram", 0, 256,
// Only tiles_x is written: preserve pitch_a.
// HOST: dpu_push_xfer(_set, DPU_XFER_TO_DPU, "_pim_metadata", 0, 4,
// HOST-NEXT: _pim_stop(1);
// HOST-NEXT: _pim_start(2);
// HOST-NEXT: DPU_ASSERT(dpu_launch(_set, DPU_SYNCHRONOUS));
// HOST-NEXT: _pim_stop(2);
// HOST-NEXT: _pim_start(3);
// HOST: dpu_prepare_xfer(_dpu, [[C]] + (uint64_t){{v[0-9]+}}[_idx] * 256)
// HOST: dpu_push_xfer(_set, DPU_XFER_FROM_DPU, "C_mram", 0, 1024,
// HOST-NEXT: _pim_stop(3);
// HOST-NEXT: _pim_start(0);
// HOST: for (int64_t _g = 0; _g < 4; ++_g)
// HOST: int32_t* {{.*}} = [[C]] + _g * 256;
// HOST: _pim_start(12);
// HOST: _pim_stop(12);
// HOST: free([[C]]);
// HOST: free([[A]]);
func.func @kernel(%A: memref<*xi8>, %C: memref<*xi32>, %lda: index, %ldc: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %set = upmem.acquire 4, @kernel_device : !upmem.set
  %assignment = upmem.bind_devices %set partitions [2, 2] device_mapping [1, 2]
      : !upmem.set -> !upmem.assignment
  %a_pool = memref.alloc() : memref<2x16x16xi8>
  %parts = memref.alloc() : memref<4x16x16xi32>
  upmem.foreach_unit %set assignment %assignment : !upmem.set axes ["x", "y"] counts [2, 2] {
  ^bb0(%dpu: !upmem.unit, %pm: index, %pn: index):
    %r = arith.muli %pm, %c16 : index
    %sa = upmem.stage %A origin[%r, %c0] sizes[%c16, %c16] strides[%lda, %c1]
          pad_to[%c16, %c16] tile [8, 16] into %a_pool {shared_over = ["y"]}
          : memref<*xi8>, memref<2x16x16xi8> -> memref<16x16xi8>
    upmem.to_mram %dpu, %sa, @A_mram : memref<16x16xi8>
    upmem.to_host_var %dpu, %c8, @tiles_x : index
  }
  upmem.launch %set : !upmem.set
  upmem.gather %set assignment %assignment, @C_mram capacity[%c16, %c16] into %parts
           : !upmem.set, memref<4x16x16xi32>
  upmem.foreach_group %parts : memref<4x16x16xi32> axes ["x", "y"] counts [2, 2] {
  ^bb0(%part: memref<16x16xi32>, %pm: index, %pn: index):
    %cr = arith.muli %pm, %c16 : index
    upmem.unstage %part into %C origin[%cr, %c0] sizes[%c16, %c16] strides[%ldc, %c1]
        capacity[%c16, %c16] tile [8, 16]
        : memref<16x16xi32>, memref<*xi32>
  }
  memref.dealloc %parts : memref<4x16x16xi32>
  memref.dealloc %a_pool : memref<2x16x16xi8>
  return
}

// ── device ──────────────────────────────────────────────────────────────────
// DEVICE: __mram_noinit int8_t A_mram[4096];
// DEVICE: __mram_noinit int32_t C_mram[1024];
// DEVICE: __host struct _pim_metadata_t _pim_metadata;
// WRAM 버퍼는 전역이다.
// DEVICE: __dma_aligned int8_t w0[128];
// DEVICE: __dma_aligned int32_t w1[256];
// DEVICE: int main()
// DEVICE: int32_t {{.*}} = _pim_metadata.f0;
// 행마다 DMA 하나. 크기는 버퍼 타입에서 나온다(8x16 i8 의 한 행 = 16 바이트).
// DEVICE: mram_read((__mram_ptr void*)(A_mram + {{.*}}), (void*){{.*}}, 16);
// DEVICE: mram_write((void*){{.*}}, (__mram_ptr void*)(C_mram + {{.*}}), 64);
upmem.device_func @kernel_device {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %aw = memref.alloc() : memref<8x16xi8, #upmem<space wram>>
  %acc = memref.alloc() : memref<8x32xi32, #upmem<space wram>>
  %n = upmem.get_host_var @tiles_x : index
  %p = upmem.get_host_var @pitch_a : index
  scf.for %tx = %c0 to %n step %c1 {
    %off = arith.muli %tx, %p : index
    scf.for %r = %c0 to %c8 step %c1 {
      %row = memref.subview %aw[%r, 0] [1, 16] [1, 1]
             : memref<8x16xi8, #upmem<space wram>>
            to memref<16xi8, strided<[1], offset: ?>, #upmem<space wram>>
      upmem.mram_read @A_mram[%off], %row
             : memref<16xi8, strided<[1], offset: ?>, #upmem<space wram>>
    }
    scf.for %r2 = %c0 to %c8 step %c1 {
      %crow = memref.subview %acc[%r2, 0] [1, 16] [1, 1]
             : memref<8x32xi32, #upmem<space wram>>
            to memref<16xi32, strided<[1], offset: ?>, #upmem<space wram>>
      upmem.mram_write %crow, @C_mram[%off]
             : memref<16xi32, strided<[1], offset: ?>, #upmem<space wram>>
    }
  }
}
