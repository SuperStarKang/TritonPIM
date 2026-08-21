// PIM dialect op의 parse/print roundtrip 확인.
// RUN: triton-shared-opt %s | triton-shared-opt | FileCheck %s

// CHECK-LABEL: func.func @access_load_store
func.func @access_load_store(%A: memref<*xf32>, %C: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: pim.access %{{.*}} offset %{{.*}} sizes [32] strides[%{{.*}}] {{.*}} : memref<*xf32> -> !pim.access
  %da = pim.access %A offset %c0 sizes [32] strides [%c1] {id = 0 : i64}
      : memref<*xf32> -> !pim.access
  // CHECK: pim.load %{{.*}} : !pim.access -> tensor<32xf32>
  %v = pim.load %da : !pim.access -> tensor<32xf32>

  %dc = pim.access %C offset %c0 sizes [32] strides [%c1] {id = 1 : i64}
      : memref<*xf32> -> !pim.access
  // CHECK: pim.store %{{.*}}, %{{.*}} : tensor<32xf32>, !pim.access
  pim.store %v, %dc : tensor<32xf32>, !pim.access
  return
}

// CHECK-LABEL: func.func @access_clauses
func.func @access_clauses(%A: memref<*xf32>, %e: index, %pad: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // mask 로드: valid + pad clause.
  // CHECK: pim.access %{{.*}} offset %{{.*}} sizes [128] strides[%{{.*}}] valid[%{{.*}}] pad %{{.*}} : f32
  %dm = pim.access %A offset %c0 sizes [128] strides [%c1] valid [%e] pad %pad : f32
      {id = 0 : i64} : memref<*xf32> -> !pim.access

  // 축별 시작 위치를 아는 access: origin clause.
  // CHECK: pim.access %{{.*}} offset %{{.*}} sizes [128] strides[%{{.*}}] origin[%{{.*}}]
  %do = pim.access %A offset %c0 sizes [128] strides [%c1] origin [%c0] {id = 1 : i64}
      : memref<*xf32> -> !pim.access
  return
}

// device region과 배치. block argument는 이 device의 partition 좌표다.
// CHECK-LABEL: func.func @device_place
func.func @device_place(%A: memref<*xi8>, %C: memref<*xi32>, %ex: index, %ey: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index

  // CHECK: pim.device #pim.device_mapping<axes = ["output_row_tile", "output_column_tile"], partitions = array<i64: 2, 4>> extents [%{{.*}}, %{{.*}}]
  pim.device #pim.device_mapping<axes = ["output_row_tile", "output_column_tile"],
                                 partitions = array<i64: 2, 4>>
      extents [%ex, %ey] {
  ^bb0(%px: index, %py: index):
    // CHECK: pim.place %{{.*}} start[%{{.*}}, %{{.*}}] size[%{{.*}}, %{{.*}}] capacity[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}]
    // CHECK-SAME: replicated_over = ["output_column_tile"]
    pim.place %A start[%c0, %c0] size[%c32, %c32] capacity[%c32, %c32]
        strides[%c32, %c1]
        {direction = "in", access = 0 : i64,
         mapping = #pim.place_mapping<axis_sources = ["output_row_tile", ""]>,
         replicated_over = ["output_column_tile"]} : memref<*xi8>
    // CHECK: pim.place %{{.*}} start[%{{.*}}, %{{.*}}] size[%{{.*}}, %{{.*}}]
    pim.place %C start[%c0, %c0] size[%c32, %c32] capacity[%c32, %c32]
        strides[%c32, %c1]
        {direction = "out", access = 1 : i64,
         mapping = #pim.place_mapping<axis_sources = ["output_row_tile",
                                                      "output_column_tile"]>} : memref<*xi32>
    scf.for %i = %c0 to %c32 step %c1 {
    } {pim.axis = "output_row_tile"}
  }
  return
}

// 기계 profile은 module에, core mapping은 func에 붙는다.
// CHECK: pim.target_profile = #pim.target_profile<target = "upmem"
module attributes {
    pim.target_profile = #pim.target_profile<
        target = "upmem", number_of_pim_devices = 2048,
        number_of_parallel_lanes_per_pim_core = 16,
        parallel_execution_kind = "multithread",
        usable_bank_capacity_bytes = 33554432,
        usable_local_capacity_bytes = 49152>} {
  // CHECK: pim.group_execution = #pim.group_execution<lanes = 16, lane_axis_role = "whole_tile">
  func.func @profiled() attributes {
      pim.group_execution = #pim.group_execution<lanes = 16,
                                                 lane_axis_role = "whole_tile">} {
    return
  }
}
