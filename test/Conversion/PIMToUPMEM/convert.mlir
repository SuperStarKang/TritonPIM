// PIM Common IR이 UPMEM host/device 구조로 바뀌는지 확인한다.
// bank 예산을 3072 바이트로 주어 MRAM 크기를 눈으로 확인한다 → 버퍼 3개에 1024씩.
// RUN: triton-shared-opt --convert-pim-to-upmem %s | FileCheck %s
// RUN: triton-shared-opt --convert-pim-to-upmem %s | FileCheck %s --check-prefix=PIM-CLEAN
// RUN: triton-shared-opt --convert-pim-to-upmem %s | triton-shared-translate --upmem-to-host > /dev/null
// RUN: triton-shared-opt --convert-pim-to-upmem %s | triton-shared-translate --upmem-to-device > /dev/null

// PIM-CLEAN-LABEL: module
// PIM-CLEAN-NOT: #pim.target_profile
// PIM-CLEAN-NOT: #pim.intra_device_plan
// PIM-CLEAN-NOT: #pim.block
// PIM-CLEAN-NOT: #pim.block_group
// PIM-CLEAN-NOT: #pim.execution_tile
// PIM-CLEAN-NOT: pim.axis
// PIM-CLEAN-NOT: pim.device_origin

// 버퍼마다 MRAM 심볼 하나. i8은 1024개, i32는 256개다.
// CHECK-DAG: upmem.mram @matmul_mram_0 : memref<1024xi8, #upmem<space mram>>
// CHECK-DAG: upmem.mram @matmul_mram_1 : memref<1024xi8, #upmem<space mram>>
// CHECK-DAG: upmem.mram @matmul_mram_2 : memref<256xi32, #upmem<space mram>>

// 버퍼마다 MRAM 안의 축 간격 하나(rank 2라 축 1 뿐).
// CHECK-DAG: upmem.host_var @matmul_pitch_0_1 : index
// CHECK-DAG: upmem.host_var @matmul_pitch_1_1 : index
// CHECK-DAG: upmem.host_var @matmul_pitch_2_1 : index

// 축마다 "이 device가 실제로 도는 횟수" 하나.
// CHECK-DAG: upmem.host_var @matmul_tiles_m : index
// CHECK-DAG: upmem.host_var @matmul_tiles_n : index

// ── device ──────────────────────────────────────────────────────────────────
// 축 loop 상한이 host가 넣어 준 값에서 오고, device 시작이 0이 되어 주소가 device 로컬이 된다.
// tile은 WRAM 버퍼가 되고, 전송은 타일마다 DMA 하나가 된다.
// CHECK-LABEL: upmem.device_func @matmul_device
// CHECK-SAME:    upmem.lanes = 1
// CHECK-DAG:     memref.alloc() : memref<8x32xi8, #upmem<space wram>>
// CHECK-DAG:     memref.alloc() : memref<32x32xi8, #upmem<space wram>>
// CHECK-DAG:     memref.alloc() : memref<8x32xi32, #upmem<space wram>>
// CHECK:         %[[TX:.*]] = upmem.get_host_var @matmul_tiles_m : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %[[TX]]
// CHECK:           arith.addi %c0{{.*}}, %{{.*}} : index
// CHECK:           scf.for
// CHECK:             %[[SV:.*]] = memref.subview
// CHECK:             upmem.mram_read @matmul_mram_0[%{{.*}}], %[[SV]]
// 계산은 WRAM memref 위에서 돌고 tensor iter_arg 는 없어진다.
// CHECK:         linalg.matmul ins(%{{.*}}, %{{.*}} : memref<8x32xi8, #upmem<space wram>>, memref<32x32xi8, #upmem<space wram>>) outs(%{{.*}} : memref<8x32xi32, #upmem<space wram>>)
// CHECK:         upmem.mram_write %{{.*}}, @matmul_mram_2
// CHECK-NOT:     iter_args
// device 몸체에 pim op이 남지 않는다.
// CHECK-NOT:     pim.access
// CHECK-NOT:     pim.load
// CHECK-NOT:     pim.store

// ── host ────────────────────────────────────────────────────────────────────
// CHECK-LABEL: func.func @matmul
// 확보와 적재는 한 op이다. 런타임이 같은 device 수·같은 프로그램이면 쥐고 있던 것을 준다.
// CHECK:         %[[SET:.*]] = upmem.acquire 8, @matmul_device : !upmem.set
// CHECK:         upmem.foreach_unit %[[SET]] : !upmem.set axes ["m", "n"] counts [2, 4]
// A는 column 축 device끼리 같은 것을 받는다. pad_to는 size가 아니라 capacity다 —
// device마다 같은 바이트 수를 보내야 하므로.
// CHECK:           %[[SA:.*]] = upmem.stage %{{.*}} origin[%{{.*}}, %{{.*}}] sizes[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}] pad_to[%{{.*}}, %{{.*}}] tile [8, 32]
// CHECK-SAME:        dedup_over = ["n"]
// CHECK:           upmem.to_mram %{{.*}}, %[[SA]], @matmul_mram_0
// CHECK:           upmem.stage
// CHECK-SAME:        dedup_over = ["m"]
// CHECK:           upmem.to_mram %{{.*}}, %{{.*}}, @matmul_mram_1
// CHECK:         upmem.launch %[[SET]]
// 출력은 집합 전체에서 한 번에 회수한 뒤 device마다 되돌린다.
// CHECK:         %[[P:.*]] = upmem.gather %[[SET]], @matmul_mram_2
// CHECK:         upmem.foreach_group %[[P]] : {{.*}} axes ["m", "n"] counts [2, 4]
// 되돌릴 때 읽는 범위는 실제 크기(size)지만 축 사이 간격은 배정량(capacity)이다.
// CHECK:           upmem.unstage %{{.*}} into %{{.*}} origin[%{{.*}}, %{{.*}}] sizes[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}] capacity[%{{.*}}, %{{.*}}] tile [8, 32]
// pim 쪽은 남지 않는다.
// CHECK-NOT:     pim.device
// CHECK-NOT:     pim.place

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", number_of_pim_devices = 8,
    number_of_banks_per_pim_device = 1,
    number_of_pim_cores_per_bank = 1,
    number_of_parallel_lanes_per_pim_core = 1,
    parallel_execution_kind = "multithread",
    usable_bank_capacity_bytes = 3072, usable_local_capacity_bytes = 49152>} {

func.func @matmul(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>,
                  %K: i32, %N: i32, %ex: index, %ey: index)
    attributes {pim.intra_device_plan = #pim.intra_device_plan<bank_partitions = [1, 1, 1], core_partitions = [1, 1, 1], lanes = 1, lane_axis = "m">,
        pim.block = #pim.block<shape = [8, 32, 32]>,
        pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32
  %lda = arith.index_cast %K : i32 to index
  %ldb = arith.index_cast %N : i32 to index
  %kk = arith.index_cast %K : i32 to index

  pim.device #pim.device_mapping<axes = ["m", "n"],
                                 partitions = [2, 4]>
      extents [%ex, %ey]
      attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  ^bb0(%px: index, %py: index):
    // device 범위. 실제 pass가 내는 것과 같은 모양으로 손으로 썼다.
    %capx = arith.ceildivsi %ex, %c2 : index
    %startx = arith.muli %px, %capx : index
    %tripsx = arith.minsi %capx, %capx : index
    %capy = arith.ceildivsi %ey, %c4 : index
    %starty = arith.muli %py, %capy : index
    %tripsy = arith.minsi %capy, %capy : index
    %tripsk = arith.ceildivsi %kk, %c32 : index

    %ar = arith.muli %startx, %c8 : index
    %arn = arith.muli %tripsx, %c8 : index
    %arc = arith.muli %capx, %c8 : index
    %akn = arith.muli %tripsk, %c32 : index
    pim.place %A start[%ar, %c0] size[%arn, %akn] capacity[%arc, %akn]
        strides[%lda, %c1]
        {direction = "in", access = 0 : i64,
         mapping = #pim.place_mapping<axis_sources = ["m", ""]>,
         replicated_over = ["n"]} : memref<*xi8>

    %bc = arith.muli %starty, %c32 : index
    %bcn = arith.muli %tripsy, %c32 : index
    %bcc = arith.muli %capy, %c32 : index
    pim.place %B start[%c0, %bc] size[%akn, %bcn] capacity[%akn, %bcc]
        strides[%ldb, %c1]
        {direction = "in", access = 1 : i64,
         mapping = #pim.place_mapping<axis_sources = ["", "n"]>,
         replicated_over = ["m"]} : memref<*xi8>

    pim.place %C start[%ar, %bc] size[%arn, %bcn] capacity[%arc, %bcc]
        strides[%ldb, %c1]
        {direction = "out", access = 2 : i64,
         mapping = #pim.place_mapping<axis_sources = ["m",
                                                      "n"]>} : memref<*xi32>

    // device가 실행하는 부분. 축 loop는 0부터 실제 횟수까지 돌고, 전역 좌표는 몸체
    // 첫머리에서 복원한다.
    scf.for %tx = %c0 to %tripsx step %c1 {
      %gx = arith.addi %startx, %tx {pim.device_origin} : index
      %m = arith.muli %gx, %c8 : index
      scf.for %ty = %c0 to %tripsy step %c1 {
        %gy = arith.addi %starty, %ty {pim.device_origin} : index
        %n = arith.muli %gy, %c32 : index
        %e = tensor.empty() : tensor<8x32xi32>
        %init = linalg.fill ins(%c0_i32 : i32) outs(%e : tensor<8x32xi32>) -> tensor<8x32xi32>
        %r = scf.for %tk = %c0 to %tripsk step %c1
             iter_args(%acc = %init) -> (tensor<8x32xi32>) {
          %k = arith.muli %tk, %c32 : index
          %da = pim.access %A offset %m sizes [8, 32] strides[%lda, %c1] origin[%m, %k]
                {id = 0 : i64} : memref<*xi8> -> !pim.access
          %ta = pim.load %da : !pim.access -> tensor<8x32xi8>
          %db = pim.access %B offset %k sizes [32, 32] strides[%ldb, %c1] origin[%k, %n]
                {id = 1 : i64} : memref<*xi8> -> !pim.access
          %tb = pim.load %db : !pim.access -> tensor<32x32xi8>
          %mm = linalg.matmul ins(%ta, %tb : tensor<8x32xi8>, tensor<32x32xi8>)
                outs(%acc : tensor<8x32xi32>) -> tensor<8x32xi32>
          scf.yield %mm : tensor<8x32xi32>
        }
        %dc = pim.access %C offset %n sizes [8, 32] strides[%ldb, %c1] origin[%m, %n]
              {id = 2 : i64} : memref<*xi32> -> !pim.access
        pim.store %r, %dc : tensor<8x32xi32>, !pim.access
      } {pim.axis = #pim.axis<"n">}
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
