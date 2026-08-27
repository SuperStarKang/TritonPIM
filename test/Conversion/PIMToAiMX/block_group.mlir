// Triton block이 하드웨어 눈금을 나누지 못하는 경우다. 예전에는 진단으로 거부했고,
// 지금은 block group이 어긋남을 padding으로 흡수한다 — Triton block 정수 개(group)를
// 하드웨어 단위의 정수 배(execution)에 채우고 남는 자리를 stage가 0으로 채운다.
// RUN: triton-shared-opt --convert-pim-to-aimx=batch=2 %s | FileCheck %s
// RUN: triton-shared-opt --convert-pim-to-aimx=batch=2 --aimx-lower-matmul %s | FileCheck %s --check-prefix=PROGRAM
// RUN: not triton-shared-opt --convert-pim-to-aimx=batch=8 %s 2>&1 | FileCheck %s --check-prefix=BATCH-ERR

// BATCH-ERR: intra-device plan must use bank_partitions=[batch, 256/batch, 1]

// batch=2는 AiMX target option에서 온다. 공통 #pim.block_group의 useful shape을
// target lowering이 [2, 2048, 128] 눈금으로 padding한다:
//   n: Triton block 21개 = 2016 유효 + 32 padding = 물리 2048 (출력 block 1개)
//   k: Triton block 5개  =  120 유효 +  8 padding = 물리 128  (reduction step 1개)
//   m: BLOCK_M=32는 단위 2의 배수 — block 1개 = 32 유효 = 물리 32, padding 0
// 입력 PIM IR의 유효 k 상한 111616은 block-group useful 눈금 240으로 내림한 뒤,
// 물리 눈금 256으로 바뀐다: k_max = 119040, padded_k = 122880.
// CHECK-LABEL: func.func @matmul
// CHECK-SAME:    aimx.tile_group = #aimx.tile_group<block = [32, 96, 24], group = [1, 21, 5], execution = [32, 2048, 128]>
// CHECK-SAME:    aimx.tiling = #aimx.tiling<batch = 2, t_m = 32, t_n = 4096, t_k = 119040, padded_k = 122880, padded_n = 4096, transposed = false>

// buffer 크기는 물리 크기에서 나온다. A(32*122880*2), Arel(32*119040*32),
// B(119040*4096*2), C(32*4096*2), descriptor chain도 같은 상한을 쓴다.
// CHECK: aimx.alloc %{{.*}} bytes 196608 align 262144 kind pmm
// CHECK: aimx.alloc %{{.*}} bytes 7864320 align 512 kind data
// CHECK: aimx.alloc %{{.*}} bytes 121896960 align 512 kind data
// CHECK: aimx.alloc %{{.*}} bytes 975175680 align 512 kind data
// CHECK: aimx.alloc %{{.*}} bytes 262144 align 512 kind data
// CHECK: aimx.alloc %{{.*}} bytes 38109184 align 512 kind descriptor

// padding이 있는 축만 stage/unstage가 block group 배치([useful, phys])를 싣는다.
// B는 행(k)·열(n) 둘 다, A는 열(k)만, C는 열(n)만이다. host loop의 전진 폭은 유효
// 크기이고 pad_to·pitch는 물리 크기다.
// CHECK:   aimx.stage {{.*}} group_rows [120, 128] group_cols [2016, 2048] into
// CHECK:   aimx.stage {{.*}} group_cols [120, 128] into
// CHECK:   aimx.invoke
// CHECK:   aimx.unstage {{.*}} group_cols [2016, 2048]

// chain은 물리 눈금만 본다 — padding은 stage가 채운 0이 데이터로 흐를 뿐 descriptor
// program은 달라지지 않는다. batch=2라 B operand와 C drain이 8192/2 = 4096이다.
// PROGRAM-LABEL: aimx.descriptor_program @matmul_relayout batch 2
// PROGRAM:         aimx.desc src(%{{.*}}, %{{.*}}) pitch %{{.*}} len 512 opcode pim_mode|stride
// PROGRAM:         aimx.desc dst(%{{.*}}, %{{.*}}) len 16384 opcode bc|tp
// PROGRAM-LABEL: aimx.descriptor_program @matmul_compute batch 2
// PROGRAM:         aimx.desc src(%{{.*}}, %{{.*}}) len 8192 opcode a_mask|matmul|pim_mode
// PROGRAM:         aimx.desc src(%{{.*}}, %{{.*}}) pitch %{{.*}} len 4096 opcode b_mask|matmul|pim_mode|stride
// PROGRAM:         aimx.desc src(%{{.*}}, %{{.*}}) pitch %{{.*}} dst(%{{.*}}, %{{.*}}) pitch %{{.*}} len 4096 opcode c_mask|matmul|stride

module attributes {pim.target_profile = #pim.target_profile<
    target = "aimx", number_of_pim_devices = 1,
    number_of_banks_per_pim_device = 256,
    number_of_pim_cores_per_bank = 1,
    number_of_parallel_lanes_per_pim_core = 16,
    parallel_execution_kind = "simd",
    usable_bank_capacity_bytes = 268435456, usable_local_capacity_bytes = 0>} {

func.func @matmul(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
                  %K: i32, %N: i32, %ex: index, %ey: index)
    attributes {pim.intra_device_plan = #pim.intra_device_plan<bank_partitions = [2, 128, 1], core_partitions = [1, 1, 1], lanes = 16, lane_axis = "n">,
        pim.block_group = #pim.block_group<
            tiles = [1, 21, 5],
            useful = [32, 2016, 120]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c24 = arith.constant 24 : index
  %c32 = arith.constant 32 : index
  %c96 = arith.constant 96 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %lda = arith.index_cast %K : i32 to index
  %ldb = arith.index_cast %N : i32 to index
  %kk = arith.index_cast %K : i32 to index

  pim.device #pim.device_mapping<axes = ["m", "n"],
                                 partitions = [1, 1]>
      extents [%ex, %ey]
      attributes {
        pim.execution_tile = #pim.execution_tile<m = 32, n = 4096, k = 111616>,
        pim.inter_device_plan = #pim.inter_device_plan<
            available_devices = 1,
            axes = ["m", "n"],
            active_grid = [1, 1],
            execution_tile = [32, 4096, 111616],
            scheduling_order = ["n", "m"]>,
        pim.reduction_plan = #pim.reduction_plan<
            spatial_partials = "none", temporal_accumulation = "host",
            final_accumulation = "none">} {
  ^bb0(%px: index, %py: index):
    %capx = arith.ceildivsi %ex, %c1 : index
    %startx = arith.muli %px, %capx : index
    %tripsx = arith.minsi %capx, %capx : index
    %capy = arith.ceildivsi %ey, %c1 : index
    %starty = arith.muli %py, %capy : index
    %tripsy = arith.minsi %capy, %capy : index
    %tripsk = arith.ceildivsi %kk, %c24 : index

    %ar = arith.muli %startx, %c32 : index
    %arn = arith.muli %tripsx, %c32 : index
    %arc = arith.muli %capx, %c32 : index
    %akn = arith.muli %tripsk, %c24 : index
    pim.place %A start[%ar, %c0] size[%arn, %akn] capacity[%arc, %akn]
        strides[%lda, %c1]
        {direction = "in", access = 0 : i64,
         mapping = #pim.place_mapping<axis_sources = ["m", ""]>,
         replicated_over = ["n"]} : memref<*xf32>

    %bc = arith.muli %starty, %c96 : index
    %bcn = arith.muli %tripsy, %c96 : index
    %bcc = arith.muli %capy, %c96 : index
    pim.place %B start[%c0, %bc] size[%akn, %bcn] capacity[%akn, %bcc]
        strides[%ldb, %c1]
        {direction = "in", access = 1 : i64,
         mapping = #pim.place_mapping<axis_sources = ["", "n"]>,
         replicated_over = ["m"]} : memref<*xf32>

    pim.place %C start[%ar, %bc] size[%arn, %bcn] capacity[%arc, %bcc]
        strides[%ldb, %c1]
        {direction = "out", access = 2 : i64,
         mapping = #pim.place_mapping<axis_sources = ["m",
                                                      "n"]>} : memref<*xf32>

    scf.for %tx = %c0 to %tripsx step %c1 {
      %gx = arith.addi %startx, %tx {pim.device_origin} : index
      %m = arith.muli %gx, %c32 : index
      scf.for %ty = %c0 to %tripsy step %c1 {
        %gy = arith.addi %starty, %ty {pim.device_origin} : index
        %n = arith.muli %gy, %c96 : index
        %e = tensor.empty() : tensor<32x96xf32>
        %init = linalg.fill ins(%f0 : f32) outs(%e : tensor<32x96xf32>) -> tensor<32x96xf32>
        %r = scf.for %tk = %c0 to %tripsk step %c1
             iter_args(%acc = %init) -> (tensor<32x96xf32>) {
          %k = arith.muli %tk, %c24 : index
          %da = pim.access %A offset %m sizes [32, 24] strides[%lda, %c1] origin[%m, %k]
                {id = 0 : i64} : memref<*xf32> -> !pim.access
          %ta = pim.load %da : !pim.access -> tensor<32x24xf32>
          %db = pim.access %B offset %k sizes [24, 96] strides[%ldb, %c1] origin[%k, %n]
                {id = 1 : i64} : memref<*xf32> -> !pim.access
          %tb = pim.load %db : !pim.access -> tensor<24x96xf32>
          %mm = linalg.matmul ins(%ta, %tb : tensor<32x24xf32>, tensor<24x96xf32>)
                outs(%acc : tensor<32x96xf32>) -> tensor<32x96xf32>
          scf.yield %mm : tensor<32x96xf32>
        }
        %dc = pim.access %C offset %n sizes [32, 96] strides[%ldb, %c1] origin[%m, %n]
              {id = 2 : i64} : memref<*xf32> -> !pim.access
        pim.store %r, %dc : tensor<32x96xf32>, !pim.access
      } {pim.axis = #pim.axis<"n">}
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
