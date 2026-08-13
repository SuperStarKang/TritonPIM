// round(시간 창): 공통 IR 의 #pim.execution_tile 출력 상한(n=64원소 = tile 2개)이
// 창이 된다. 창 축(output_column_tile)에 매인 B·C 는 round 마다 재전송·재수거되고,
// 매이지 않은 A 는 첫 foreach 에서 한 번만 전송된다. launch 는 round 마다 1회이고
// device 바이너리는 변하지 않는다(loop 상한이 host 변수).
// RUN: triton-shared-opt --pim-form-execution-tiles=tile-n=64 --convert-pim-to-upmem %s | FileCheck %s

// A(mram_0)만 첫 foreach 에서 전송된다. 창 축의 반복 횟수는 여기서 보내지 않고,
// B(mram_1)의 전송도 여기 없다.
// CHECK: upmem.foreach_unit
// CHECK:   upmem.to_mram %{{.*}}, %{{.*}}, @matmul_mram_0
// CHECK:   upmem.to_host_var %{{.*}}, %{{.*}}, @matmul_tiles_output_row_tile
// CHECK-NOT: upmem.to_mram %{{.*}}, %{{.*}}, @matmul_mram_1

// round 수 = ceildiv(창 축 배정량, 창 2). 창 foreach 가 이 round 의 창 축 반복 횟수와
// B 창을 보낸 뒤, launch → C 창 회수 → 되돌리기가 loop 안에서 반복된다.
// CHECK: arith.ceildivsi
// CHECK: scf.for %{{.*}} = %{{.*}} to
// CHECK:   upmem.foreach_unit
// CHECK:     upmem.to_host_var %{{.*}}, %{{.*}}, @matmul_tiles_output_column_tile
// CHECK:     upmem.stage
// CHECK:     upmem.to_mram %{{.*}}, %{{.*}}, @matmul_mram_1
// CHECK:   upmem.launch
// CHECK:   upmem.gather %{{.*}}, @matmul_mram_2
// CHECK:   upmem.foreach_group
// CHECK:     upmem.unstage

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", number_of_pim_devices = 8, pim_core_shape = array<i64: 1>,
    number_of_parallel_lanes_per_pim_core = 1,
    parallel_execution_kind = "multithread",
    requires_uniform_instruction_sequence = false,
    usable_bank_capacity_bytes = 3072, usable_local_capacity_bytes = 49152>} {

func.func @matmul(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>,
                  %K: i32, %N: i32, %ex: index, %ey: index)
    attributes {pim.group_execution = #pim.group_execution<
        lanes = 1, lane_axis_role = "whole_tile">} {
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

  pim.device #pim.device_mapping<axes = ["output_row_tile", "output_column_tile"],
                                 partitions = array<i64: 2, 4>>
      extents [%ex, %ey] {
  ^bb0(%px: index, %py: index):
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
         mapping = #pim.place_mapping<axis_sources = ["output_row_tile", ""]>,
         replicated_over = ["output_column_tile"]} : memref<*xi8>

    %bc = arith.muli %starty, %c32 : index
    %bcn = arith.muli %tripsy, %c32 : index
    %bcc = arith.muli %capy, %c32 : index
    pim.place %B start[%c0, %bc] size[%akn, %bcn] capacity[%akn, %bcc]
        strides[%ldb, %c1]
        {direction = "in", access = 1 : i64,
         mapping = #pim.place_mapping<axis_sources = ["", "output_column_tile"]>,
         replicated_over = ["output_row_tile"]} : memref<*xi8>

    pim.place %C start[%ar, %bc] size[%arn, %bcn] capacity[%arc, %bcc]
        strides[%ldb, %c1]
        {direction = "out", access = 2 : i64,
         mapping = #pim.place_mapping<axis_sources = ["output_row_tile",
                                                      "output_column_tile"]>} : memref<*xi32>

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
      } {pim.axis = "output_column_tile"}
    } {pim.axis = "output_row_tile"}
  }
  return
}
}
