// PIM IR의 유닛 형태가 UPMEM host/device 구조로 바뀌는지 확인한다.
// 입력은 --pim-partition="splits=2,4,8"까지 돌린 모습이다(축 loop가 0부터 도는 유닛 로컬,
// pim.unit_data에 sizes와 capacity가 따로).
// MRAM 크기를 눈으로 확인하려고 유닛 메모리를 3072 바이트로 준다 → 버퍼 3개에 1024씩.
// RUN: triton-shared-opt --convert-pim-to-upmem="unit-memory-bytes=3072" %s | FileCheck %s

// 버퍼마다 MRAM 심볼 하나. i8은 1024개, i32는 256개다.
// CHECK: upmem.mram @matmul_mram_0 : memref<1024xi8, #upmem.space<mram>>
// CHECK: upmem.mram @matmul_mram_1 : memref<1024xi8, #upmem.space<mram>>
// CHECK: upmem.mram @matmul_mram_2 : memref<256xi32, #upmem.space<mram>>

// 버퍼마다 MRAM 안의 축 간격 하나(rank 2라 축 1 뿐).
// CHECK: upmem.host_var @matmul_pitch_0_1 : index
// CHECK: upmem.host_var @matmul_pitch_1_1 : index
// CHECK: upmem.host_var @matmul_pitch_2_1 : index

// 축마다 "이 유닛이 실제로 도는 횟수" 하나.
// CHECK: upmem.host_var @matmul_tiles_x : index
// CHECK: upmem.host_var @matmul_tiles_y : index
// CHECK: upmem.host_var @matmul_tiles_k : index

// ── device ──────────────────────────────────────────────────────────────────
// 축 loop 상한이 host가 넣어 준 값에서 오고, 유닛 시작이 0이 되어 주소가 유닛 로컬이 된다.
// tile은 WRAM 버퍼가 되고, 전송은 타일마다 DMA 하나가 된다.
// CHECK-LABEL: upmem.device_func @matmul_device
// CHECK:         memref.alloc() : memref<8x32xi8, #upmem.space<wram>>
// CHECK:         memref.alloc() : memref<32x32xi8, #upmem.space<wram>>
// CHECK:         memref.alloc() : memref<8x32xi32, #upmem.space<wram>>
// CHECK:         %[[TX:.*]] = upmem.get_host_var @matmul_tiles_x : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %[[TX]]
// CHECK:           arith.addi %c0{{.*}} {pim.unit_origin}
// MRAM 안의 타일 시작을 구해 그 타일을 옮긴다(2048 byte를 넘으면 행 묶음으로 쪼갠다).
// CHECK:           scf.for
// CHECK:             %[[SV:.*]] = memref.subview
// CHECK:             upmem.mram_read @matmul_mram_0[%{{.*}}], %[[SV]]
// CHECK:           upmem.mram_write %{{.*}}, @matmul_mram_2
// 계산은 WRAM memref 위에서 돌고 tensor iter_arg 는 없어진다.
// CHECK:         linalg.matmul ins(%{{.*}}, %{{.*}} : memref<8x32xi8, #upmem.space<wram>>, memref<32x32xi8, #upmem.space<wram>>) outs(%{{.*}} : memref<8x32xi32, #upmem.space<wram>>)
// CHECK-NOT:     iter_args
// device 몸체에 pim op이 남지 않는다.
// CHECK-NOT:     pim.access
// CHECK-NOT:     pim.load
// CHECK-NOT:     pim.store

// ── host ────────────────────────────────────────────────────────────────────
// CHECK-LABEL: func.func @matmul
// 확보와 적재는 한 op이다. 런타임이 같은 유닛 수·같은 프로그램이면 쥐고 있던 것을 준다.
// CHECK:         %[[SET:.*]] = upmem.acquire 64, @matmul_device : !upmem.set
// CHECK:         upmem.foreach_unit %[[SET]] : !upmem.set axes ["x", "y", "k"] counts [2, 4, 8]
// A는 y축 유닛끼리 같은 것을 받는다. pad_to는 sizes가 아니라 capacity다 —
// 유닛마다 같은 바이트 수를 보내야 하므로.
// CHECK:           %[[SA:.*]] = upmem.stage %{{.*}} origin[%{{.*}}, %{{.*}}] sizes[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}] pad_to[%{{.*}}, %{{.*}}] tile array<i64: 8, 32>
// CHECK-SAME:        dedup_over = ["y"]
// CHECK:           upmem.to_mram %{{.*}}, %[[SA]], @matmul_mram_0
// CHECK:           upmem.stage
// CHECK-SAME:        dedup_over = ["x"]
// CHECK:           upmem.to_mram %{{.*}}, %{{.*}}, @matmul_mram_1
// 반복 횟수는 유닛마다 다르므로 broadcast가 아니라 유닛별 전송이다.
// CHECK:           upmem.to_host_var %{{.*}}, %{{.*}}, @matmul_tiles_x : index
// CHECK:         upmem.launch %[[SET]]
// 출력은 집합 전체에서 한 번에 회수하고, k축 8개를 묶어 더한 뒤 되돌린다.
// CHECK:         %[[P:.*]] = upmem.gather %[[SET]], @matmul_mram_2
// CHECK:         upmem.foreach_group %[[P]] : {{.*}} axes ["x", "y"] counts [2, 4]
// CHECK-SAME:      combine = "add"
// CHECK-SAME:      reduce_axis = "k"
// CHECK-SAME:      reduce_count = 8
// 되돌릴 때 읽는 범위는 실제 크기(sizes)지만 축 사이 간격은 배정량(capacity)이다.
// CHECK:           upmem.unstage %{{.*}} into %{{.*}} origin[%{{.*}}, %{{.*}}] sizes[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}] capacity[%{{.*}}, %{{.*}}] tile array<i64: 8, 32>
// pim 쪽은 남지 않는다.
// CHECK-NOT:     pim.units
// CHECK-NOT:     pim.unit_data

func.func @matmul(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>,
                  %K: i32, %N: i32, %npx: index, %npy: index)
    attributes {pim.execution_model = "spmd", pim.grid_dims = ["x", "y"],
                pim.split_axes = [{axis = "x", source = "grid"},
                                  {axis = "y", source = "grid"},
                                  {axis = "k", source = "reduction", combine = "add"}]} {
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

  pim.units axes ["x", "y", "k"] counts [2, 4, 8] {
  ^bb0(%pm: index, %pn: index, %kj: index):
    // 유닛 범위. 실제 pass가 내는 것과 같은 모양으로 손으로 썼다.
    %perx = arith.ceildivsi %npx, %c2 : index
    %lbx = arith.muli %pm, %perx : index
    %tripsx = arith.minsi %perx, %perx : index
    %pery = arith.ceildivsi %npy, %c4 : index
    %lby = arith.muli %pn, %pery : index
    %tripsy = arith.minsi %pery, %pery : index
    %perk = arith.ceildivsi %kk, %c8 : index
    %lbk = arith.muli %kj, %perk : index
    %tripsk = arith.minsi %perk, %perk : index

    // 영역: origin은 유닛의 첫 인덱스에서, sizes는 실제 횟수에서, capacity는 배정량에서.
    %ar = arith.muli %lbx, %c8 : index
    %ak = arith.muli %lbk, %c32 : index
    %arn = arith.muli %tripsx, %c8 : index
    %akn = arith.muli %tripsk, %c32 : index
    %arc = arith.muli %perx, %c8 : index
    %akc = arith.muli %perk, %c32 : index
    pim.unit_data %A origin[%ar, %ak] sizes[%arn, %akn] capacity[%arc, %akc]
        strides[%lda, %c1]
        {direction = "in", access = 0 : i64, replicated_over = ["y"]} : memref<*xi8>

    %bc = arith.muli %lby, %c32 : index
    %bcn = arith.muli %tripsy, %c32 : index
    %bcc = arith.muli %pery, %c32 : index
    pim.unit_data %B origin[%ak, %bc] sizes[%akn, %bcn] capacity[%akc, %bcc]
        strides[%ldb, %c1]
        {direction = "in", access = 1 : i64, replicated_over = ["x"]} : memref<*xi8>

    pim.unit_data %C origin[%ar, %bc] sizes[%arn, %bcn] capacity[%arc, %bcc]
        strides[%ldb, %c1]
        {direction = "out", access = 2 : i64,
         combine = "add", combine_over = ["k"]} : memref<*xi32>

    // 유닛이 실행하는 부분. 축 loop는 0부터 실제 횟수까지 돌고, 전역 인덱스는
    // 몸체 첫머리에서 복원한다. 누산은 --pim-normalize-accumulator를 지난 모습이다.
    scf.for %tx = %c0 to %tripsx step %c1 {
      %gx = arith.addi %lbx, %tx {pim.unit_origin} : index
      %m = arith.muli %gx, %c8 : index
      scf.for %ty = %c0 to %tripsy step %c1 {
        %gy = arith.addi %lby, %ty {pim.unit_origin} : index
        %n = arith.muli %gy, %c32 : index
        %e = tensor.empty() : tensor<8x32xi32>
        %init = linalg.fill ins(%c0_i32 : i32) outs(%e : tensor<8x32xi32>) -> tensor<8x32xi32>
        %r = scf.for %tk = %c0 to %tripsk step %c1
             iter_args(%acc = %init) -> (tensor<8x32xi32>) {
          %gk = arith.addi %lbk, %tk {pim.unit_origin} : index
          %k = arith.muli %gk, %c32 : index
          %da = pim.access %A offset %m sizes [8, 32] strides[%lda, %c1] origin[%m, %k]
                {pim.access_id = 0 : i64, pim.role = "input"} : memref<*xi8> -> !pim.access
          %ta = pim.load %da : !pim.access -> tensor<8x32xi8>
          %db = pim.access %B offset %k sizes [32, 32] strides[%ldb, %c1] origin[%k, %n]
                {pim.access_id = 1 : i64, pim.role = "input"} : memref<*xi8> -> !pim.access
          %tb = pim.load %db : !pim.access -> tensor<32x32xi8>
          %mm = linalg.matmul ins(%ta, %tb : tensor<8x32xi8>, tensor<32x32xi8>)
                outs(%acc : tensor<8x32xi32>) -> tensor<8x32xi32>
          scf.yield %mm : tensor<8x32xi32>
        } {pim.reduction_combine = "add"}
        %dc = pim.access %C offset %n sizes [8, 32] strides[%ldb, %c1] origin[%m, %n]
              {pim.access_id = 2 : i64, pim.role = "output"} : memref<*xi32> -> !pim.access
        pim.store %r, %dc : tensor<8x32xi32>, !pim.access
      } {pim.grid_dim = "y"}
    } {pim.grid_dim = "x"}
  }
  return
}
