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
// HOST-NEXT: _pim_start(0);
// staging: dedup 축을 뺀 조합마다 한 번씩만 만들고, 남는 자리는 0이다.
// HOST: _pim_stop(0);
// HOST-NEXT: _pim_start(5);
// HOST-NEXT: int8_t** _s0 = (int8_t**)malloc(2 * sizeof(int8_t*));
// HOST-NEXT: _pim_stop(5);
// HOST-NEXT: _pim_start(8);
// HOST-NEXT: memset(_s0, 0, 2 * sizeof(int8_t*));
// HOST-NEXT: _pim_stop(8);
// HOST-NEXT: _pim_start(0);
// 유닛 좌표는 선형 번호에서 자리값으로 푼다.
// HOST: for (int64_t _u = 0; _u < 4; ++_u)
// HOST: _pim_stop(0);
// HOST-NEXT: _pim_start(5);
// HOST-NEXT: int8_t* _b = (int8_t*)malloc(
// HOST-NEXT: _pim_stop(5);
// HOST-NEXT: _pim_start(8);
// HOST-NEXT: memset(_b, 0,
// HOST-NEXT: _pim_stop(8);
// HOST-NEXT: _pim_start(11);
// 부분 타일도 돌되 원본 범위를 넘는 원소는 건너뛴다.
// HOST: for (int64_t _t0 = 0; _t0 < (({{.*}} + 7) / 8); ++_t0)
// HOST: if (_t0 * 8 + _e0 < {{.*}} && _t1 * 16 + _e1 < {{.*}})
// HOST: _b[{{.*}} * 128 + (_e0 * 16 + _e1)] =
// HOST-NEXT: _pim_stop(11);
// HOST-NEXT: _pim_start(0);
// MRAM 심볼과 host staging은 같은 fixed-slot 크기이며 전송 직전에도 상한을 확인한다.
// HOST: if ({{.*}}_bytes > 4096) {
// H2D는 PGEMM과 같이 행렬+메타데이터의 prepare/push만 잰다.
// HOST: _pim_stop(0);
// HOST-NEXT: _pim_start(1);
// HOST: dpu_prepare_xfer
// HOST: dpu_push_xfer(_set, DPU_XFER_TO_DPU, "A_mram"
// Only tiles_x is written in this hand-authored IR: preserve pitch_a.
// HOST: dpu_push_xfer(_set, DPU_XFER_TO_DPU, "_pim_metadata", 0, 4,
// HOST-NEXT: _pim_stop(1);
// HOST-NEXT: _pim_start(6);
// HOST: for (int64_t _i = 0; _i < 2; ++_i) free(_s0[_i]);
// HOST-NEXT: free(_s0);
// HOST-NEXT: free(_p0);
// HOST-NEXT: _pim_stop(6);
// 해제 비용이 kernel 타이머로 넘어가면 안 된다.
// HOST-NEXT: _pim_start(2);
// HOST-NEXT: DPU_ASSERT(dpu_launch(_set, DPU_SYNCHRONOUS));
// HOST-NEXT: _pim_stop(2);
// 수신 버퍼 할당은 D2H 시작 전에 끝낸다.
// HOST-NEXT: _pim_start(5);
// HOST: malloc(4 * sizeof(int32_t*));
// HOST: for (int64_t _u = 0; _u < 4; ++_u) {{.*}}malloc(
// HOST-NEXT: _pim_stop(5);
// HOST-NEXT: _pim_start(3);
// HOST: dpu_prepare_xfer
// HOST: dpu_push_xfer(_set, DPU_XFER_FROM_DPU, "C_mram"
// HOST-NEXT: _pim_stop(3);
// HOST-NEXT: _pim_start(0);
// device마다 자기 버퍼를 원본으로 되돌린다.
// HOST: for (int64_t _g = 0; _g < 4; ++_g)
// HOST: _pim_stop(0);
// HOST-NEXT: _pim_start(12);
// HOST: _pim_stop(12);
// HOST-NEXT: _pim_start(0);
// HOST: _pim_stop(0);
// HOST-NEXT: _pim_start(6);
// HOST-NEXT: for (int64_t _u = 0; _u < 4; ++_u) free(
// HOST-NEXT: free(
// HOST-NEXT: _pim_stop(6);
func.func @kernel(%A: memref<*xi8>, %C: memref<*xi32>, %lda: index, %ldc: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %set = upmem.acquire 4, @kernel_device : !upmem.set
  upmem.foreach_unit %set : !upmem.set axes ["x", "y"] counts [2, 2] {
  ^bb0(%dpu: !upmem.unit, %pm: index, %pn: index):
    %r = arith.muli %pm, %c16 : index
    %c = arith.muli %pn, %c16 : index
    %sa = upmem.stage %A origin[%r, %c] sizes[%c16, %c16] strides[%lda, %c1]
          pad_to[%c16, %c16] tile [8, 16] {dedup_over = ["y"]}
          : memref<*xi8> -> memref<16x16xi8>
    upmem.to_mram %dpu, %sa, @A_mram : memref<16x16xi8>
    upmem.to_host_var %dpu, %c8, @tiles_x : index
  }
  upmem.launch %set : !upmem.set
  %parts = upmem.gather %set, @C_mram capacity[%c16, %c16]
           : !upmem.set -> memref<4x16x16xi32>
  upmem.foreach_group %parts : memref<4x16x16xi32> axes ["x", "y"] counts [2, 2] {
  ^bb0(%part: memref<16x16xi32>, %pm: index, %pn: index):
    %cr = arith.muli %pm, %c16 : index
    upmem.unstage %part into %C origin[%cr, %c0] sizes[%c16, %c16] strides[%ldc, %c1]
        capacity[%c16, %c16] tile [8, 16]
        : memref<16x16xi32>, memref<*xi32>
  }
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
