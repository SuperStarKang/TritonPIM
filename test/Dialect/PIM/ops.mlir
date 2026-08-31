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

// 3계층은 각 층의 기본 단위만 저장한다. Group/Tile 원소 shape는 곱으로 파생된다.
// CHECK-LABEL: func.func @hierarchy
// CHECK-SAME: pim.block = #pim.block<shape = [32, 32, 8]>
// CHECK-SAME: pim.block_group = #pim.block_group<blocks = [1, 4, 1]>
// CHECK-SAME: pim.execution_tile = #pim.execution_tile<groups = [4, 2, 64]>
func.func @hierarchy() attributes {
    pim.block = #pim.block<shape = [32, 32, 8]>,
    pim.block_group = #pim.block_group<blocks = [1, 4, 1]>,
    pim.execution_tile = #pim.execution_tile<groups = [4, 2, 64]>} {
  return
}

// device region과 배치. block argument는 이 device의 partition 좌표다.
// CHECK-LABEL: func.func @device_place
func.func @device_place(%A: memref<*xi8>, %C: memref<*xi32>, %ex: index, %ey: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index

 // CHECK: pim.device <partitions = [2, 4, 1]>
  pim.device #pim.device_mapping<partitions = [2, 4, 1]> {
  ^bb0(%px: index, %py: index, %device_k_unused: index):
    // CHECK: pim.place %{{.*}} start[%{{.*}}, %{{.*}}] size[%{{.*}}, %{{.*}}] allocation_shape[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}] {{.*}}replicated_over = ["n"]
    pim.place %A start[%c0, %c0] size[%c32, %c32] allocation_shape[%c32, %c32]
        strides[%c32, %c1]
        {direction = "in", access = 0 : i64,
         replicated_over = ["n"]} : memref<*xi8>
    // CHECK: pim.place %{{.*}} start[%{{.*}}, %{{.*}}] size[%{{.*}}, %{{.*}}]
    pim.place %C start[%c0, %c0] size[%c32, %c32] allocation_shape[%c32, %c32]
        strides[%c32, %c1]
        {direction = "out", access = 1 : i64} : memref<*xi32>
    scf.for %i = %c0 to %c32 step %c1 {
    } {pim.axis = #pim.axis<"m">}
  }
  return
}

// 공통 기계 profile은 module에 붙는다.
// CHECK: pim.target_profile = #pim.target_profile<target = "upmem"
module attributes {
    pim.target_profile = #pim.target_profile<
        target = "upmem", num_units = 2048,
        banks_per_unit = 1,
        cores_per_bank = 1,
        lanes_per_core = 16,
        bank_capacity_limit_bytes = 33554432,
        local_capacity_limit_bytes = 49152>} {
  func.func @profiled() {
    return
  }
}
