// One block group is decomposed through the bank -> core -> lane hierarchy.
// RUN: triton-shared-opt --pim-plan-intra-device %s | FileCheck %s --check-prefix=AUTO
// RUN: triton-shared-opt "--pim-plan-intra-device=bank-partitions=1,2,1 core-partitions=1,1,1 lanes=4 lane-axis=n" %s | FileCheck %s --check-prefix=EXPLICIT

// AUTO: #pim.intra_device_plan<bank_partitions = [1, 1, 1], core_partitions = [1, 1, 1], lanes = 16, lane_axis = "m">
// EXPLICIT: #pim.intra_device_plan<bank_partitions = [1, 2, 1], core_partitions = [1, 1, 1], lanes = 4, lane_axis = "n">

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", number_of_pim_devices = 8,
    number_of_banks_per_pim_device = 1,
    number_of_pim_cores_per_bank = 1,
    number_of_parallel_lanes_per_pim_core = 16,
    parallel_execution_kind = "multithread",
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
