// 공통 planner가 만든 scheduling round를 UPMEM lowering이 그대로 소비한다.
// acquire는 round 밖에서 한 번, foreach/stage/launch/gather는 round마다 한 번이다.
// RUN: triton-shared-opt --convert-pim-to-upmem %s | FileCheck %s

// Device set은 round loop 밖에서 한 번 확보한다.
// CHECK: arith.ceildivsi
// CHECK: %[[SET:.*]] = upmem.acquire 8, @matmul_device
// CHECK: scf.for %{{.*}} = %{{.*}} to
// CHECK:   upmem.foreach_unit %[[SET]]
// CHECK:     upmem.stage
// CHECK:     upmem.to_mram %{{.*}}, %{{.*}}, @matmul_mram_0
// CHECK:     upmem.stage
// CHECK:     upmem.to_mram %{{.*}}, %{{.*}}, @matmul_mram_1
// CHECK:     upmem.to_host_var %{{.*}}, %{{.*}}, @matmul_tiles_n
// CHECK:   upmem.launch
// CHECK:   upmem.gather %{{.*}}, @matmul_mram_2
// CHECK:   upmem.foreach_group
// CHECK:     upmem.unstage

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", number_of_pim_devices = 8,
    number_of_banks_per_pim_device = 1,
    number_of_pim_cores_per_bank = 1,
    number_of_parallel_lanes_per_pim_core = 1,
    parallel_execution_kind = "multithread",
    usable_bank_capacity_bytes = 3072, usable_local_capacity_bytes = 49152>} {

func.func @matmul(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>,
                  %K: i32, %N: i32, %ex: index, %ey: index)
    attributes {
      pim.block = #pim.block<shape = [8, 32, 32]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
      pim.intra_device_plan = #pim.intra_device_plan<bank_partitions = [1, 1, 1], core_partitions = [1, 1, 1], lanes = 1, lane_axis = "m">} {
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

  %ety = arith.ceildivsi %ey, %c2 : index
  %rounds = arith.ceildivsi %ety, %c4 : index
  scf.for %round = %c0 to %rounds step %c1 {
  pim.device #pim.device_mapping<axes = ["m", "n"],
                                 partitions = [2, 4]>
      extents [%ex, %ey]
      attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 2, 1]>} {
  ^bb0(%px: index, %py: index):
    %capx = arith.ceildivsi %ex, %c2 : index
    %startx = arith.muli %px, %capx : index
    %tripsx = arith.minsi %capx, %capx : index
    %roundBase = arith.muli %round, %c4 : index
    %tileY = arith.addi %roundBase, %py : index
    %starty = arith.muli %tileY, %c2 : index
    %remainingY0 = arith.subi %ey, %starty : index
    %remainingY = arith.maxsi %remainingY0, %c0 : index
    %tripsy = arith.minsi %c2, %remainingY : index
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
    %bcc = arith.muli %c2, %c32 : index
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
  } {pim.scheduling_axis = "n"}
  return
}
}
