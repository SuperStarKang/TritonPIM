// PIM Common IR이 UPMEM host/device 구조로 바뀌는지 확인한다.
// 각 MRAM은 Common allocation_shape의 정확한 fixed-slot 크기다.
// RUN: triton-shared-opt --pim-form-host-plan %s | triton-shared-opt | FileCheck %s --check-prefix=PLAN
// RUN: not triton-shared-opt --convert-pim-to-upmem %s 2>&1 | FileCheck %s --check-prefix=UNPLANNED
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-upmem %s | FileCheck %s
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-upmem %s | FileCheck %s --check-prefix=PIM-CLEAN
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-upmem %s | triton-shared-translate --upmem-to-host | FileCheck %s --check-prefix=HOST
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-upmem %s | triton-shared-translate --upmem-to-device | FileCheck %s --check-prefix=DEVICE

// UNPLANNED: requires pim-form-host-plan before UPMEM lowering
// PLAN-LABEL: func.func @matmul
// PLAN: pim.partition <partitions = [2, 4, 1]>
// PLAN: pim.place {{.*}}access = 0
// PLAN-SAME: shared_over = ["n"]
// PLAN: pim.place {{.*}}access = 1
// PLAN-SAME: shared_over = ["m"]
// PLAN: pim.place {{.*}}access = 2
// PLAN: pim.broadcast 0
// PLAN-NEXT: pim.broadcast 1
// PLAN-NEXT: pim.launch {
// PLAN: linalg.matmul
// PLAN: pim.combine {{.*}} kind "add"
// PLAN: pim.gather 2

// DEVICE: __dma_aligned int8_t {{.*}}[512];
// DEVICE: for (int32_t _n = 0; _n < 16; ++_n)
// DEVICE: _s += {{.*}}[_m * 32 + _k] * {{.*}}[_k * 16 + _n];

// Both inputs retain row-major inner tiles: A=[8,32], B=[32,16].
// HOST: uint32_t [[MAP:v[0-9]+]][8];
// HOST: upmem_bind_devices(_set, 3, [[MAP]]_partitions, [[MAP]]_mapping, [[MAP]]);
// HOST: _pim_start(9);
// HOST: {{v[0-9]+}}[{{.*}} * 256 + (_e0 * 32 + _e1)] =
// HOST: _pim_stop(9);
// HOST: _pim_start(10);
// HOST: {{v[0-9]+}}[{{.*}} * 512 + (_e0 * 16 + _e1)] =
// HOST: _pim_stop(10);

// Each round has exactly three H2D pushes: A, B, and all metadata together.
// HOST: _pim_start(1);
// HOST-NEXT: _idx = 0;
// HOST-NEXT: DPU_FOREACH{{.*}}dpu_prepare_xfer(_dpu, _p0{{\[}}[[MAP]][_idx]]));
// HOST-NEXT: DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "matmul_mram_0",
// HOST-NEXT: _idx = 0;
// HOST-NEXT: DPU_FOREACH{{.*}}dpu_prepare_xfer(_dpu, _p1{{\[}}[[MAP]][_idx]]));
// HOST-NEXT: DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "matmul_mram_1",
// HOST-NEXT: _idx = 0;
// HOST-NEXT: DPU_FOREACH{{.*}}dpu_prepare_xfer(_dpu, (uint8_t*)&_args0{{\[}}[[MAP]][_idx]] + 0));
// HOST-NEXT: DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "_pim_metadata", 0,
// HOST-NEXT: _pim_stop(1);
// HOST: dpu_prepare_xfer(_dpu, {{v[0-9]+}} + (uint64_t)[[MAP]][_idx] * 128)

// PIM-CLEAN-LABEL: module
// PIM-CLEAN-NOT: #pim.target_profile
// PIM-CLEAN-NOT: #pim.block
// PIM-CLEAN-NOT: #pim.block_group
// PIM-CLEAN-NOT: #pim.execution_tile
// PIM-CLEAN-NOT: pim.axis
// PIM-CLEAN-NOT: pim.device_origin
// PIM-CLEAN-NOT: pim.temporal_reduction
// PIM-CLEAN-NOT: pim.scheduling_axis
// PIM-CLEAN-NOT: pim.combine
// PIM-CLEAN-NOT: pim.partition
// PIM-CLEAN-NOT: pim.broadcast
// PIM-CLEAN-NOT: pim.scatter
// PIM-CLEAN-NOT: pim.launch
// PIM-CLEAN-NOT: pim.gather

// 버퍼마다 MRAM 심볼 하나. A=8x32, logical B=32x16, C=8x16이다.
// CHECK-DAG: upmem.mram @matmul_mram_0 : memref<256xi8, #upmem<space mram>>
// CHECK-DAG: upmem.mram @matmul_mram_1 : memref<512xi8, #upmem<space mram>>
// CHECK-DAG: upmem.mram @matmul_mram_2 : memref<128xi32, #upmem<space mram>>

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
// CHECK-DAG:     memref.alloc() : memref<32x16xi8, #upmem<space wram>>
// CHECK-DAG:     memref.alloc() : memref<8x16xi32, #upmem<space wram>>
// CHECK:         %[[TX:.*]] = upmem.get_host_var @matmul_tiles_m : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %[[TX]]
// CHECK:           arith.addi %c0{{.*}}, %{{.*}} : index
// CHECK:           scf.for
// CHECK:             %[[SV:.*]] = memref.subview
// CHECK:             upmem.mram_read @matmul_mram_0[%{{.*}}], %[[SV]]
// 계산은 WRAM memref 위에서 돌고 tensor iter_arg 는 없어진다.
// CHECK:         linalg.matmul ins(%{{.*}}, %{{.*}} : memref<8x32xi8, #upmem<space wram>>, memref<32x16xi8, #upmem<space wram>>) outs(%{{.*}} : memref<8x16xi32, #upmem<space wram>>)
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
// CHECK:         %[[ASSIGN:.*]] = upmem.bind_devices %[[SET]] partitions [2, 4, 1] device_mapping [2, 4, 1]
// CHECK:         %[[AP:.*]] = memref.alloc() : memref<2x8x32xi8>
// CHECK:         %[[BP:.*]] = memref.alloc() : memref<4x32x16xi8>
// CHECK:         %[[P:.*]] = memref.alloc() : memref<8x8x16xi32>
// CHECK:         upmem.foreach_unit %[[SET]] assignment %[[ASSIGN]] : !upmem.set axes ["m", "n", "k"] counts [2, 4, 1]
// A는 column 축 device끼리 같은 것을 받는다. pad_to는 size가 아니라 capacity다 —
// device마다 같은 바이트 수를 보내야 하므로.
// CHECK:           %[[SA:.*]] = upmem.stage %{{.*}} origin[%{{.*}}, %{{.*}}] sizes[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}] pad_to[%{{.*}}, %{{.*}}] tile [8, 32]
// CHECK-SAME:        operand_role = "lhs"
// CHECK-SAME:        shared_over = ["n"]
// CHECK-SAME:        -> memref<8x32xi8>
// CHECK:           upmem.to_mram %{{.*}}, %[[SA]], @matmul_mram_0
// CHECK:           upmem.stage
// CHECK-SAME:        operand_role = "rhs"
// CHECK-SAME:        shared_over = ["m"]
// CHECK:           upmem.to_mram %{{.*}}, %{{.*}}, @matmul_mram_1
// CHECK:         upmem.launch %[[SET]]
// 출력은 집합 전체에서 한 번에 회수한 뒤 device마다 되돌린다.
// CHECK:         upmem.gather %[[SET]] assignment %[[ASSIGN]], @matmul_mram_2 {{.*}} into %[[P]] : !upmem.set, memref<8x8x16xi32>
// CHECK:         upmem.foreach_group %[[P]] : {{.*}} axes ["m", "n", "k"] counts [2, 4, 1]
// 되돌릴 때 읽는 범위는 실제 크기(size)지만 축 사이 간격은 배정량(capacity)이다.
// Common pim.combine과 reduction_k round가 명시한 partial 결합을 그대로 전달한다.
// CHECK:           upmem.unstage %{{.*}} into %{{.*}} origin[%{{.*}}, %{{.*}}] sizes[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}] capacity[%{{.*}}, %{{.*}}] accumulate %{{.*}} tile [8, 16]
// CHECK:         memref.dealloc %[[AP]]
// CHECK:         memref.dealloc %[[BP]]
// CHECK:         memref.dealloc %[[P]]
// pim 쪽은 남지 않는다.
// CHECK-NOT:     pim.device
// CHECK-NOT:     pim.place

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 8,
    banks_per_unit = 1,
    cores_per_bank = 1,
    lanes_per_core = 1,
    bank_capacity_limit_bytes = 3072, local_capacity_limit_bytes = 49152>} {

func.func @matmul(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>,
                  %K: i32, %N: i32, %ex: index, %ey: index)
    attributes {
        pim.block = #pim.block<shape = [8, 16, 32]>,
        pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32
  %lda = arith.index_cast %K : i32 to index
  %ldb = arith.index_cast %N : i32 to index
  %kk = arith.index_cast %K : i32 to index
  %reductionRounds = arith.ceildivsi %kk, %c32 : index

  scf.for %reductionRound = %c0 to %reductionRounds step %c1 {
  pim.device #pim.device_mapping<partitions = [2, 4, 1]>
      attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  ^bb0(%px: index, %py: index, %device_k_unused: index):
    // device 범위. 실제 pass가 내는 것과 같은 모양으로 손으로 썼다.
    %capx = arith.ceildivsi %ex, %c2 : index
    %startx = arith.muli %px, %capx : index
    %tripsx = arith.minsi %capx, %capx : index
    %capy = arith.ceildivsi %ey, %c4 : index
    %starty = arith.muli %py, %capy : index
    %tripsy = arith.minsi %capy, %capy : index
    %remainingk0 = arith.subi %reductionRounds, %reductionRound : index
    %remainingk = arith.maxsi %remainingk0, %c0 : index
    %tripsk = arith.minsi %remainingk, %c1 : index
    %startk = arith.muli %reductionRound, %c32 : index

    %ar = arith.muli %startx, %c8 : index
    %arn = arith.muli %tripsx, %c8 : index
    %akn = arith.muli %tripsk, %c32 : index
    pim.place %A start[%ar, %startk] size[%arn, %akn] allocation_shape[%c8, %c32]
        strides[%lda, %c1]
        {direction = "in", access = 0 : i64,
         shared_over = ["n"]} : memref<*xi8>

    %bc = arith.muli %starty, %c16 : index
    %bcn = arith.muli %tripsy, %c16 : index
    pim.place %B start[%startk, %bc] size[%akn, %bcn] allocation_shape[%c32, %c16]
        strides[%ldb, %c1]
        {direction = "in", access = 1 : i64,
         shared_over = ["m"]} : memref<*xi8>

    pim.place %C start[%ar, %bc] size[%arn, %bcn] allocation_shape[%c8, %c16]
        strides[%ldb, %c1]
        {direction = "out", access = 2 : i64} : memref<*xi32>

    // device가 실행하는 부분. 축 loop는 0부터 실제 횟수까지 돌고, 전역 좌표는 몸체
    // 첫머리에서 복원한다.
    scf.for %tx = %c0 to %tripsx step %c1 {
      %gx = arith.addi %startx, %tx {pim.device_origin} : index
      %m = arith.muli %gx, %c8 : index
      scf.for %ty = %c0 to %tripsy step %c1 {
        %gy = arith.addi %starty, %ty {pim.device_origin} : index
        %n = arith.muli %gy, %c16 : index
        %e = tensor.empty() : tensor<8x16xi32>
        %init = linalg.fill ins(%c0_i32 : i32) outs(%e : tensor<8x16xi32>) -> tensor<8x16xi32>
        %r = scf.for %tk = %c0 to %tripsk step %c1
             iter_args(%acc = %init) -> (tensor<8x16xi32>) {
          %globalTk = arith.addi %reductionRound, %tk {pim.device_origin} : index
          %k = arith.muli %globalTk, %c32 : index
          %da = pim.access %A offset %m sizes [8, 32] strides[%lda, %c1] origin[%m, %k]
                {id = 0 : i64} : memref<*xi8> -> !pim.access
          %ta = pim.load %da : !pim.access -> tensor<8x32xi8>
          %db = pim.access %B offset %k sizes [32, 16] strides[%ldb, %c1] origin[%k, %n]
                {id = 1 : i64} : memref<*xi8> -> !pim.access
          %tb = pim.load %db : !pim.access -> tensor<32x16xi8>
          %mm = linalg.matmul ins(%ta, %tb : tensor<8x32xi8>, tensor<32x16xi8>)
                outs(%acc : tensor<8x16xi32>) -> tensor<8x16xi32>
          scf.yield %mm : tensor<8x16xi32>
        } {pim.reduction = "add", pim.temporal_reduction}
        %dc = pim.access %C offset %n sizes [8, 16] strides[%ldb, %c1] origin[%m, %n]
              {id = 2 : i64} : memref<*xi32> -> !pim.access
        pim.combine %r into %dc kind "add" : tensor<8x16xi32>, !pim.access
      } {pim.axis = #pim.axis<"n">}
    } {pim.axis = #pim.axis<"m">}
  }
  } {pim.scheduling_axis = "reduction_k"}
  return
}
}
