// 협력 lane 모드(lane_axis = m): tasklet 들이 tile 하나를 출력 행으로
// 나눠 함께 계산한다. 버퍼는 한 벌, 적재는 chunk 분담, 경계는 barrier 다.
// 입력은 convert.mlir 과 같은 GEMM 이고 lanes = 4 만 다르다. 8행 tile 을 4 lane 이
// 2행씩 맡는다.
// RUN: triton-shared-opt --convert-pim-to-upmem %s | FileCheck %s

// CHECK-LABEL: upmem.device_func @matmul_device
// CHECK-SAME:    upmem.lanes = 4

// 자기 행의 시작 = tasklet 번호 × 2. 버퍼는 lane 차원 없이 한 벌씩이다.
// CHECK:         %[[TID:.*]] = upmem.tasklet_id : index
// CHECK-DAG:     memref.alloc() : memref<8x32xi8, #upmem<space wram>>
// CHECK-DAG:     memref.alloc() : memref<32x32xi8, #upmem<space wram>>
// CHECK-DAG:     memref.alloc() : memref<8x32xi32, #upmem<space wram>>

// 출력 행 slice. A 와 누산기만 갈리고 B 는 전 행이 필요해 그대로다.
// CHECK-DAG:     memref.subview %{{.*}}[%{{.*}}, 0] [2, 32] [1, 1] : memref<8x32xi8
// CHECK-DAG:     memref.subview %{{.*}}[%{{.*}}, 0] [2, 32] [1, 1] : memref<8x32xi32

// fill 은 자기 행 slice 에만 — 행이 사유라 barrier 가 없다.
// CHECK:         linalg.fill ins(%{{.*}} : i32) outs(%{{.*}} : memref<2x32xi32

// 적재는 chunk 를 tasklet 번호부터 lane 수만큼 건너뛰며 분담한다.
// CHECK:         scf.for %{{.*}} = %[[TID]] to %{{.*}} step %{{.*}}
// CHECK:           upmem.mram_read @matmul_mram_0

// 공유 B 의 적재가 끝난 뒤에만 계산하고(앞 barrier), 다음 적재가 덮기 전에 전원이
// 소비를 마친다(뒤 barrier). 계산은 행 slice 위에서 돈다.
// CHECK:         upmem.barrier
// CHECK:         linalg.matmul ins(%{{.*}}, %{{.*}} : memref<2x32xi8, {{.*}}>, memref<32x32xi8, #upmem<space wram>>) outs(%{{.*}} : memref<2x32xi32, {{.*}}>)
// CHECK:         upmem.barrier

// 저장은 chunk 분담이라 다른 tasklet 의 행을 읽는다 — 앞뒤로 barrier.
// CHECK:         upmem.barrier
// CHECK:         upmem.mram_write
// CHECK:         upmem.barrier

// whole_tile 의 tile 분배(loop 평탄화)는 없다 — 전원이 같은 tile 순서를 돈다.
// CHECK-NOT:     upmem.num_tasklets
// lane 복제 버퍼(4x…)도 없다.
// CHECK-NOT:     memref.alloc() : memref<4x

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 8,
    banks_per_unit = 1,
    cores_per_bank = 1,
    lanes_per_core = 4,
    bank_capacity_limit_bytes = 3072, local_capacity_limit_bytes = 49152>} {

func.func @matmul(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>,
                  %K: i32, %N: i32, %ex: index, %ey: index)
    attributes {
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
  %reductionRounds = arith.ceildivsi %kk, %c32 : index

  scf.for %reductionRound = %c0 to %reductionRounds step %c1 {
  pim.device #pim.device_mapping<partitions = [2, 4, 1]>
      attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  ^bb0(%px: index, %py: index, %device_k_unused: index):
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
    %arc = arith.muli %capx, %c8 : index
    %akn = arith.muli %tripsk, %c32 : index
    pim.place %A start[%ar, %startk] size[%arn, %akn] allocation_shape[%arc, %c32]
        strides[%lda, %c1]
        {direction = "in", access = 0 : i64,
         replicated_over = ["n"]} : memref<*xi8>

    %bc = arith.muli %starty, %c32 : index
    %bcn = arith.muli %tripsy, %c32 : index
    %bcc = arith.muli %capy, %c32 : index
    pim.place %B start[%startk, %bc] size[%akn, %bcn] allocation_shape[%c32, %bcc]
        strides[%ldb, %c1]
        {direction = "in", access = 1 : i64,
         replicated_over = ["m"]} : memref<*xi8>

    pim.place %C start[%ar, %bc] size[%arn, %bcn] allocation_shape[%arc, %bcc]
        strides[%ldb, %c1]
        {direction = "out", access = 2 : i64} : memref<*xi32>

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
          %globalTk = arith.addi %reductionRound, %tk {pim.device_origin} : index
          %k = arith.muli %globalTk, %c32 : index
          %da = pim.access %A offset %m sizes [8, 32] strides[%lda, %c1] origin[%m, %k]
                {id = 0 : i64} : memref<*xi8> -> !pim.access
          %ta = pim.load %da : !pim.access -> tensor<8x32xi8>
          %db = pim.access %B offset %k sizes [32, 32] strides[%ldb, %c1] origin[%k, %n]
                {id = 1 : i64} : memref<*xi8> -> !pim.access
          %tb = pim.load %db : !pim.access -> tensor<32x32xi8>
          %mm = linalg.matmul ins(%ta, %tb : tensor<8x32xi8>, tensor<32x32xi8>)
                outs(%acc : tensor<8x32xi32>) -> tensor<8x32xi32>
          scf.yield %mm : tensor<8x32xi32>
        } {pim.temporal_reduction}
        %dc = pim.access %C offset %n sizes [8, 32] strides[%ldb, %c1] origin[%m, %n]
              {id = 2 : i64} : memref<*xi32> -> !pim.access
        pim.store %r, %dc {pim.combine = "add"} : tensor<8x32xi32>, !pim.access
      } {pim.axis = #pim.axis<"n">}
    } {pim.axis = #pim.axis<"m">}
  }
  } {pim.scheduling_axis = "reduction_k"}
  return
}
}
