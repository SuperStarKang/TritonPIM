// Triton block이 하드웨어 눈금을 나누지 못하는 경우다. 예전에는 진단으로 거부했고,
// 지금은 block group이 어긋남을 padding으로 흡수한다 — Triton block 정수 개(group)를
// 하드웨어 단위의 정수 배(execution)에 채우고 남는 자리를 stage가 0으로 채운다.
// RUN: triton-shared-opt --convert-pim-to-aimx %s | FileCheck %s
// RUN: triton-shared-opt --convert-pim-to-aimx --aimx-lower-matmul %s | FileCheck %s --check-prefix=PROGRAM

// batch는 padding 비율(padN/physN + padK/physK)로 고른다. BLOCK_N=96, BLOCK_K=24는
// 어떤 batch에서도 padding이 남고, 비율은 b=1과 b=2가 같으므로(0.078125) 큰 쪽 2가
// 뽑힌다. 그때 눈금은 출력 block 폭 2048, reduction step 128이다:
//   n: Triton block 21개 = 2016 유효 + 32 padding = 물리 2048 (출력 block 1개)
//   k: Triton block 5개  =  120 유효 +  8 padding = 물리 128  (reduction step 1개)
//   m: lcm(BLOCK_M=32, batch=2) = 32 — m은 언제나 padding 0이다
// CHECK-LABEL: func.func @matmul
// CHECK-SAME:    aimx.tile_group = #aimx.tile_group<block = [32, 96, 24], group = [1, 21, 5], execution = [32, 2048, 128]>
// CHECK-SAME:    aimx.tiling = #aimx.tiling<batch = 2, m_t = 32, k_t = 32768, n_t = 4096, padded_k = 32768, padded_n = 4096, transposed = false>

// buffer 크기는 물리 크기에서 나온다. A(32*32768*2), Arel(32*32768*32),
// B(32768*4096*2), C(32*4096*2), chain(20512 group * 512).
// CHECK: aimx.alloc %{{.*}} bytes 196608 align 262144 kind pmm
// CHECK: aimx.alloc %{{.*}} bytes 2097152 align 512 kind data
// CHECK: aimx.alloc %{{.*}} bytes 33554432 align 512 kind data
// CHECK: aimx.alloc %{{.*}} bytes 268435456 align 512 kind data
// CHECK: aimx.alloc %{{.*}} bytes 262144 align 512 kind data
// CHECK: aimx.alloc %{{.*}} bytes 10502144 align 512 kind descriptor

// padding이 있는 축만 stage/unstage가 block group 배치([useful, phys])를 싣는다.
// B는 행(k)·열(n) 둘 다, A는 열(k)만, C는 열(n)만이다. host loop의 전진 폭은 유효
// 크기이고 pad_to·pitch는 물리 크기다.
// CHECK:   aimx.stage {{.*}} group_rows array<i64: 120, 128> group_cols array<i64: 2016, 2048> into
// CHECK:   aimx.stage {{.*}} group_cols array<i64: 120, 128> into
// CHECK:   aimx.invoke
// CHECK:   aimx.unstage {{.*}} group_cols array<i64: 2016, 2048> :

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
    target = "aimx", number_of_pim_devices = 1, pim_core_shape = [1],
    number_of_parallel_lanes_per_pim_core = 4096,
    parallel_execution_kind = "simd",
    requires_uniform_instruction_sequence = true,
    usable_bank_capacity_bytes = 1073741824, usable_local_capacity_bytes = 0>} {

func.func @matmul(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
                  %K: i32, %N: i32, %ex: index, %ey: index)
    attributes {pim.core_mapping = #pim.core_mapping<
        core_axis_roles = ["none"], tile_footprint = [1],
        lanes = 4096, lane_axis_role = "whole_tile">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c24 = arith.constant 24 : index
  %c32 = arith.constant 32 : index
  %c96 = arith.constant 96 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %lda = arith.index_cast %K : i32 to index
  %ldb = arith.index_cast %N : i32 to index
  %kk = arith.index_cast %K : i32 to index

  pim.device #pim.device_mapping<axes = ["output_row_tile", "output_column_tile"],
                                 partitions = [1, 1]>
      extents [%ex, %ey] {
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
         mapping = #pim.place_mapping<axis_sources = ["output_row_tile", ""]>,
         replicated_over = ["output_column_tile"]} : memref<*xf32>

    %bc = arith.muli %starty, %c96 : index
    %bcn = arith.muli %tripsy, %c96 : index
    %bcc = arith.muli %capy, %c96 : index
    pim.place %B start[%c0, %bc] size[%akn, %bcn] capacity[%akn, %bcc]
        strides[%ldb, %c1]
        {direction = "in", access = 1 : i64,
         mapping = #pim.place_mapping<axis_sources = ["", "output_column_tile"]>,
         replicated_over = ["output_row_tile"]} : memref<*xf32>

    pim.place %C start[%ar, %bc] size[%arn, %bcn] capacity[%arc, %bcc]
        strides[%ldb, %c1]
        {direction = "out", access = 2 : i64,
         mapping = #pim.place_mapping<axis_sources = ["output_row_tile",
                                                      "output_column_tile"]>} : memref<*xf32>

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
      } {pim.axis = "output_column_tile"}
    } {pim.axis = "output_row_tile"}
  }
  return
}
}
