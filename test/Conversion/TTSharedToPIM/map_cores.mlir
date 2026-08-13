// lane-mode 가 lane 수와 lane_axis_role 을 어떻게 정하는지 확인한다.
// RUN: triton-shared-opt --pim-map-cores %s | FileCheck %s --check-prefix=WHOLE
// RUN: triton-shared-opt --pim-map-cores=lane-mode=output_row %s | FileCheck %s --check-prefix=ROW

// whole_tile(기본): lane 마다 tile 버퍼를 복제하므로 예산이 lane 수를 자른다.
//   tile 바이트 = A 256 + B 1024 + 누산기 1024 = 2304 → 4096 / 2304 = 1
// WHOLE: lanes = 1, lane_axis_role = "whole_tile"

// output_row: 버퍼가 한 벌이라 lane 수가 예산과 무관하다. 능력값 16 을 그대로 쓴다.
// ROW: lanes = 16, lane_axis_role = "output_row"

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", number_of_pim_devices = 8, pim_core_shape = array<i64: 1>,
    number_of_parallel_lanes_per_pim_core = 16,
    parallel_execution_kind = "multithread",
    requires_uniform_instruction_sequence = false,
    usable_bank_capacity_bytes = 3072, usable_local_capacity_bytes = 4096>} {

func.func @matmul(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>,
                  %lda: index, %ldb: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %da = pim.access %A offset %c0 sizes [8, 32] strides[%lda, %c1] origin[%c0, %c0]
        {id = 0 : i64} : memref<*xi8> -> !pim.access
  %ta = pim.load %da : !pim.access -> tensor<8x32xi8>
  %db = pim.access %B offset %c0 sizes [32, 32] strides[%ldb, %c1] origin[%c0, %c0]
        {id = 1 : i64} : memref<*xi8> -> !pim.access
  %tb = pim.load %db : !pim.access -> tensor<32x32xi8>
  %e = tensor.empty() : tensor<8x32xi32>
  %init = linalg.fill ins(%c0_i32 : i32) outs(%e : tensor<8x32xi32>) -> tensor<8x32xi32>
  %mm = linalg.matmul ins(%ta, %tb : tensor<8x32xi8>, tensor<32x32xi8>)
        outs(%init : tensor<8x32xi32>) -> tensor<8x32xi32>
  %dc = pim.access %C offset %c0 sizes [8, 32] strides[%ldb, %c1] origin[%c0, %c0]
        {id = 2 : i64} : memref<*xi32> -> !pim.access
  pim.store %mm, %dc : tensor<8x32xi32>, !pim.access
  return
}
}
