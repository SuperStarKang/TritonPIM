// UPMEM IR이 host C++ / DPU C 소스로 나오는지 확인한다.
// 예시는 유닛 4개(x축 2 × k축 2)로 나눈 아주 작은 GEMM이다.
// RUN: triton-shared-translate --upmem-to-host   %s | FileCheck %s --check-prefix=HOST
// RUN: triton-shared-translate --upmem-to-device %s | FileCheck %s --check-prefix=DEVICE

upmem.mram @A_mram : memref<4096xi8, #upmem.space<mram>>
upmem.mram @C_mram : memref<1024xi32, #upmem.space<mram>>
upmem.host_var @tiles_x : index
upmem.host_var @pitch_a : index

// ── host ────────────────────────────────────────────────────────────────────
// HOST: extern "C" void kernel(int8_t* a0, int32_t* a1, int64_t a2, int64_t a3)
// HOST: dpu_alloc(4
// HOST: dpu_load(_set, DPU_BINARY, NULL)
// 유닛 좌표는 선형 번호에서 자리값으로 푼다.
// HOST: for (int64_t _u = 0; _u < 4; ++_u)
// staging: dedup 축을 뺀 조합마다 한 번씩만 만들고, 남는 자리는 0이다.
// HOST: calloc(2, sizeof(int8_t*))
// HOST: calloc(
// HOST: dpu_prepare_xfer
// HOST: dpu_push_xfer(_set, DPU_XFER_TO_DPU, "A_mram"
// HOST: dpu_push_xfer(_set, DPU_XFER_TO_DPU, "tiles_x"
// HOST: dpu_launch(_set, DPU_SYNCHRONOUS)
// HOST: dpu_push_xfer(_set, DPU_XFER_FROM_DPU, "C_mram"
// k축 2개를 더해 합친 뒤 되돌린다.
// HOST: for (int64_t _g = 0; _g < 2; ++_g)
// HOST: for (int64_t _j = 1; _j < 2; ++_j)
// HOST: {{.*}} +=
func.func @kernel(%A: memref<*xi8>, %C: memref<*xi32>, %lda: index, %ldc: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %set = upmem.alloc 4 : !upmem.set
  upmem.load_program %set, @kernel_device : !upmem.set
  upmem.foreach_unit %set : !upmem.set axes ["x", "k"] counts [2, 2] {
  ^bb0(%dpu: !upmem.unit, %pm: index, %kj: index):
    %r = arith.muli %pm, %c16 : index
    %c = arith.muli %kj, %c16 : index
    %sa = upmem.stage %A origin[%r, %c] sizes[%c16, %c16] strides[%lda, %c1]
          pad_to[%c16, %c16] tile array<i64: 8, 16> {dedup_over = ["k"]}
          : memref<*xi8> -> memref<?x?xi8>
    upmem.to_mram %dpu, %sa, @A_mram : memref<?x?xi8>
    upmem.to_host_var %dpu, %c8, @tiles_x : index
  }
  upmem.launch %set : !upmem.set
  %parts = upmem.gather %set, @C_mram capacity[%c16, %c16]
           : !upmem.set -> memref<4x?x?xi32>
  upmem.foreach_group %parts : memref<4x?x?xi32> axes ["x"] counts [2]
      {reduce_axis = "k", reduce_count = 2 : i64, combine = "add"} {
  ^bb0(%part: memref<?x?xi32>, %pm: index):
    %cr = arith.muli %pm, %c16 : index
    upmem.unstage %part into %C origin[%cr, %c0] sizes[%c16, %c16] strides[%ldc, %c1]
        capacity[%c16, %c16] tile array<i64: 8, 16>
        : memref<?x?xi32>, memref<*xi32>
  }
  return
}

// ── device ──────────────────────────────────────────────────────────────────
// DEVICE: __mram_noinit int8_t A_mram[4096];
// DEVICE: __mram_noinit int32_t C_mram[1024];
// DEVICE: __host int32_t tiles_x;
// WRAM 버퍼는 전역이다.
// DEVICE: __dma_aligned int8_t w0[128];
// DEVICE: __dma_aligned int32_t w1[256];
// DEVICE: int main()
// DEVICE: int32_t {{.*}} = tiles_x;
// 행마다 DMA 하나. 크기는 버퍼 타입에서 나온다(8x16 i8 의 한 행 = 16 바이트).
// DEVICE: mram_read((__mram_ptr void*)(A_mram + {{.*}}), (void*){{.*}}, 16);
// DEVICE: mram_write((void*){{.*}}, (__mram_ptr void*)(C_mram + {{.*}}), 64);
upmem.device_func @kernel_device {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %aw = memref.alloc() : memref<8x16xi8, #upmem.space<wram>>
  %acc = memref.alloc() : memref<8x32xi32, #upmem.space<wram>>
  %n = upmem.get_host_var @tiles_x : index
  %p = upmem.get_host_var @pitch_a : index
  scf.for %tx = %c0 to %n step %c1 {
    %off = arith.muli %tx, %p : index
    scf.for %r = %c0 to %c8 step %c1 {
      %row = memref.subview %aw[%r, 0] [1, 16] [1, 1]
             : memref<8x16xi8, #upmem.space<wram>>
            to memref<16xi8, strided<[1], offset: ?>, #upmem.space<wram>>
      upmem.mram_read @A_mram[%off], %row
             : memref<16xi8, strided<[1], offset: ?>, #upmem.space<wram>>
    }
    scf.for %r2 = %c0 to %c8 step %c1 {
      %crow = memref.subview %acc[%r2, 0] [1, 16] [1, 1]
             : memref<8x32xi32, #upmem.space<wram>>
            to memref<16xi32, strided<[1], offset: ?>, #upmem.space<wram>>
      upmem.mram_write %crow, @C_mram[%off]
             : memref<16xi32, strided<[1], offset: ?>, #upmem.space<wram>>
    }
  }
}
