// 공통 planner가 만든 scheduling round를 UPMEM lowering이 그대로 소비한다.
// acquire는 round 밖에서 한 번, foreach/stage/launch/gather는 round마다 한 번이다.
// RUN: triton-shared-opt --convert-pim-to-upmem %s | FileCheck %s

// Device set은 round loop 밖에서 한 번 확보한다.
// CHECK: arith.ceildivsi
// CHECK: %[[SET:.*]] = upmem.acquire 8, @matmul_device
// CHECK: scf.for %{{.*}} = %{{.*}} to
// CHECK:   upmem.foreach_unit %[[SET]]
// Common spatial/temporal scheduling 안의 pim.place가 각 round의 입력 전송을 정한다.
// target이 별도 temporal foreach를 만들지 않고 launch 전에 그대로 준비하는지만 본다.
// CHECK-DAG: upmem.stage
// CHECK-DAG: upmem.to_mram %{{.*}}, %{{.*}}, @matmul_mram_0
// CHECK-DAG: upmem.stage
// CHECK-DAG: upmem.to_mram %{{.*}}, %{{.*}}, @matmul_mram_1
// CHECK-DAG: upmem.to_host_var %{{.*}}, %{{.*}}, @matmul_tiles_n
// CHECK-COUNT-1: upmem.launch
// CHECK:   upmem.gather %{{.*}}, @matmul_mram_2
// CHECK:   upmem.foreach_group
// CHECK:     upmem.unstage

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 8,
    banks_per_unit = 1,
    cores_per_bank = 1,
    lanes_per_core = 1,
    bank_capacity_limit_bytes = 8192, local_capacity_limit_bytes = 49152>} {

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
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %lda = arith.index_cast %K : i32 to index
  %ldb = arith.index_cast %N : i32 to index
  %kk = arith.index_cast %K : i32 to index
  %reductionRounds = arith.ceildivsi %kk, %c32 : index

  %ety = arith.ceildivsi %ey, %c2 : index
  %rounds = arith.ceildivsi %ety, %c4 : index
  scf.for %round = %c0 to %rounds step %c1 {
  scf.for %reductionRound = %c0 to %reductionRounds step %c1 {
  pim.device #pim.device_mapping<partitions = [2, 4, 1]>
      attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 2, 1]>} {
  ^bb0(%px: index, %py: index, %device_k_unused: index):
    %capx = arith.ceildivsi %ex, %c2 : index
    %startx = arith.muli %px, %capx : index
    %tripsx = arith.minsi %capx, %capx : index
    %roundBase = arith.muli %round, %c4 : index
    %tileY = arith.addi %roundBase, %py : index
    %starty = arith.muli %tileY, %c2 : index
    %remainingY0 = arith.subi %ey, %starty : index
    %remainingY = arith.maxsi %remainingY0, %c0 : index
    %tripsy = arith.minsi %c2, %remainingY : index
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
         replicated_over = ["n"]} : memref<*xi8>

    %bc = arith.muli %starty, %c32 : index
    %bcn = arith.muli %tripsy, %c32 : index
    pim.place %B start[%startk, %bc] size[%akn, %bcn] allocation_shape[%c32, %c64]
        strides[%ldb, %c1]
        {direction = "in", access = 1 : i64,
         replicated_over = ["m"]} : memref<*xi8>

    pim.place %C start[%ar, %bc] size[%arn, %bcn] allocation_shape[%c8, %c64]
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
        } {pim.reduction = "add", pim.temporal_reduction}
        %dc = pim.access %C offset %n sizes [8, 32] strides[%ldb, %c1] origin[%m, %n]
              {id = 2 : i64} : memref<*xi32> -> !pim.access
        pim.combine %r into %dc kind "add" : tensor<8x32xi32>, !pim.access
      } {pim.axis = #pim.axis<"n">}
    } {pim.axis = #pim.axis<"m">}
  }
  } {pim.scheduling_axis = "reduction_k"}
  } {pim.scheduling_axis = "n"}
  return
}
}
