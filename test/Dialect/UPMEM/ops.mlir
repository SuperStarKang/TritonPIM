// UPMEM dialect가 파싱·출력되는지 확인한다(round-trip).
// 예시는 i8 GEMM을 (4, 32, 16) 유닛으로 나눈 모습이다.
// M=1024, N=8192, K=8192, BLOCK=(8,32,16)이면 유닛 하나가 받는 것은
//   A [256, 512] = 128 KB,  B [512, 256] = 128 KB,  C [256, 256] = 256 KB.
// RUN: triton-shared-opt %s | triton-shared-opt | FileCheck %s

// CHECK: upmem.mram @A_mram : memref<256x512xi8, #upmem.space<mram>>
upmem.mram @A_mram : memref<256x512xi8, #upmem.space<mram>>
upmem.mram @B_mram : memref<512x256xi8, #upmem.space<mram>>
upmem.mram @C_mram : memref<256x256xi32, #upmem.space<mram>>

// device가 알아야 하는 수. host가 launch 전에 채운다.
// tiles_m/n/k는 이 유닛이 실제로 도는 횟수다 — 마지막 유닛이 덜 맡는 것이 이 값 하나로
// 표현되므로, device는 언제나 [0, tiles)를 돌면 된다.
// CHECK: upmem.host_var @tiles_m : i32
upmem.host_var @tiles_m : i32
upmem.host_var @tiles_n : i32
upmem.host_var @tiles_k : i32

// ── host ────────────────────────────────────────────────────────────────────
// CHECK-LABEL: func.func @matmul_kernel_i8_host
func.func @matmul_kernel_i8_host(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>,
                                 %M: index, %N: index, %K: index) {
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index

  // CHECK: %[[SET:.*]] = upmem.acquire 2048, @matmul_kernel_i8_device : !upmem.set
  %set = upmem.acquire 2048, @matmul_kernel_i8_device : !upmem.set
  // 입력은 유닛마다 영역이 다르므로 region 안에서 좌표로 계산한다.
  // CHECK: upmem.foreach_unit %[[SET]] : !upmem.set axes ["x", "y", "k"] counts [4, 32, 16]
  upmem.foreach_unit %set : !upmem.set axes ["x", "y", "k"] counts [4, 32, 16] {
  ^bb0(%dpu: !upmem.unit, %pm: index, %pn: index, %kj: index):
    // A: 행 [pm*256, +256), 열 [kj*512, +512). y축 유닛끼리 같은 것을 받는다.
    %ar = arith.muli %pm, %c256 : index
    %ac = arith.muli %kj, %c512 : index
    // CHECK: %[[SA:.*]] = upmem.stage %{{.*}} origin[%{{.*}}, %{{.*}}] sizes[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}] pad_to[%{{.*}}, %{{.*}}] tile array<i64: 64, 64>
    // CHECK-SAME: dedup_over = ["y"]
    %sa = upmem.stage %A origin[%ar, %ac] sizes[%c256, %c512] strides[%K, %c1]
          pad_to[%c256, %c512] tile array<i64: 64, 64> {dedup_over = ["y"]}
          : memref<*xi8> -> memref<256x512xi8>
    // CHECK: upmem.to_mram %{{.*}}, %[[SA]], @A_mram
    upmem.to_mram %dpu, %sa, @A_mram : memref<256x512xi8>

    // B: 행 [kj*512, +512), 열 [pn*256, +256). x축 유닛끼리 같은 것을 받는다.
    %br = arith.muli %kj, %c512 : index
    %bc = arith.muli %pn, %c256 : index
    %sb = upmem.stage %B origin[%br, %bc] sizes[%c512, %c256] strides[%N, %c1]
          pad_to[%c512, %c256] tile array<i64: 64, 64> {dedup_over = ["x"]}
          : memref<*xi8> -> memref<512x256xi8>
    upmem.to_mram %dpu, %sb, @B_mram : memref<512x256xi8>
  }

  // CHECK: upmem.launch %[[SET]]
  upmem.launch %set : !upmem.set

  // 출력은 집합 전체에서 한 번에 회수한다(앞 축이 유닛 축).
  // CHECK: %[[P:.*]] = upmem.gather %[[SET]], @C_mram : !upmem.set -> memref<2048x256x256xi32>
  %parts = upmem.gather %set, @C_mram : !upmem.set -> memref<2048x256x256xi32>

  // k축 16개가 한 출력 타일의 부분합이다. 합친 뒤 원본 C로 되돌린다.
  // CHECK: upmem.foreach_group %[[P]] : memref<2048x256x256xi32> axes ["x", "y"] counts [4, 32]
  // CHECK-SAME: combine = "add"
  // CHECK-SAME: reduce_axis = "k"
  // CHECK-SAME: reduce_count = 16
  upmem.foreach_group %parts : memref<2048x256x256xi32> axes ["x", "y"] counts [4, 32]
      {reduce_axis = "k", reduce_count = 16 : i64, combine = "add"} {
  ^bb0(%part: memref<256x256xi32>, %pm: index, %pn: index):
    %cr = arith.muli %pm, %c256 : index
    %cc = arith.muli %pn, %c256 : index
    // CHECK: upmem.unstage %{{.*}} into %{{.*}} origin[%{{.*}}, %{{.*}}] sizes[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}] capacity[%{{.*}}, %{{.*}}] tile array<i64: 64, 64>
    upmem.unstage %part into %C origin[%cr, %cc] sizes[%c256, %c256] strides[%N, %c1]
        capacity[%c256, %c256] tile array<i64: 64, 64>
        : memref<256x256xi32>, memref<*xi32>
  }
  return
}

// ── device ──────────────────────────────────────────────────────────────────
// 유닛 하나가 맡은 타일들을 순회한다. 좌표가 유닛 로컬이라 언제나 0부터 시작한다.
// CHECK-LABEL: upmem.device_func @matmul_kernel_i8_device
upmem.device_func @matmul_kernel_i8_device {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32

  // CHECK: %[[T:.*]] = upmem.tasklet_id : index
  %t = upmem.tasklet_id : index
  // CHECK: %[[NT:.*]] = upmem.num_tasklets : index
  %nt = upmem.num_tasklets : index

  // WRAM 타일 버퍼. 합계 128 + 512 + 1024 = 1664 B.
  // CHECK: memref.alloc() : memref<8x16xi8, #upmem.space<wram>>
  %aw = memref.alloc() : memref<8x16xi8, #upmem.space<wram>>
  %bw = memref.alloc() : memref<16x32xi8, #upmem.space<wram>>
  %acc = memref.alloc() : memref<8x32xi32, #upmem.space<wram>>

  scf.for %tm = %c0 to %c32 step %c1 {          // 이 유닛의 m 타일 32개
    scf.for %tn = %c0 to %c8 step %c1 {         // 이 유닛의 n 타일 8개
      linalg.fill ins(%c0_i32 : i32) outs(%acc : memref<8x32xi32, #upmem.space<wram>>)
      scf.for %k = %c0 to %c32 step %c1 {       // 이 유닛의 K 반복 32번
        // MRAM → WRAM. 128 B와 512 B로 둘 다 8의 배수이고 2048 이하다.
        // CHECK: upmem.mram_read @A_mram[%{{.*}}], %{{.*}} : memref<8x16xi8, #upmem.space<wram>>
        %ao = arith.muli %tm, %c8 : index
        upmem.mram_read @A_mram[%ao], %aw : memref<8x16xi8, #upmem.space<wram>>
        %bo = arith.muli %k, %c16 : index
        upmem.mram_read @B_mram[%bo], %bw : memref<16x32xi8, #upmem.space<wram>>
        linalg.matmul ins(%aw, %bw : memref<8x16xi8, #upmem.space<wram>>,
                                     memref<16x32xi8, #upmem.space<wram>>)
                      outs(%acc : memref<8x32xi32, #upmem.space<wram>>)
      }
      // CHECK: upmem.mram_write %{{.*}}, @C_mram[%{{.*}}] : memref<8x32xi32, #upmem.space<wram>>
      %co = arith.muli %tm, %c32 : index
      upmem.mram_write %acc, @C_mram[%co] : memref<8x32xi32, #upmem.space<wram>>
    }
  }
}
