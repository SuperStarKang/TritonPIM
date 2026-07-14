// PIM dialect 메모리 op의 parse/print roundtrip 확인.
// RUN: triton-shared-opt %s | triton-shared-opt | FileCheck %s

// CHECK-LABEL: func.func @access_load_store
func.func @access_load_store(%A: memref<*xf32>, %C: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: pim.access %{{.*}} offset %{{.*}} sizes [32] strides[%{{.*}}] : memref<*xf32> -> !pim.access
  %da = pim.access %A offset %c0 sizes [32] strides [%c1]
      : memref<*xf32> -> !pim.access
  // CHECK: pim.load %{{.*}} : !pim.access -> tensor<32xf32>
  %v = pim.load %da : !pim.access -> tensor<32xf32>

  // CHECK: pim.access %{{.*}} offset %{{.*}} sizes [32] strides[%{{.*}}]
  %dc = pim.access %C offset %c0 sizes [32] strides [%c1]
      : memref<*xf32> -> !pim.access
  // CHECK: pim.store %{{.*}}, %{{.*}} : tensor<32xf32>, !pim.access
  pim.store %v, %dc : tensor<32xf32>, !pim.access
  return
}

// CHECK-LABEL: func.func @access_clauses
func.func @access_clauses(%A: memref<*xf32>, %e: index, %pad: f32, %idx: tensor<8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // ⑤ mask 로드: valid + pad clause.
  // CHECK: pim.access %{{.*}} offset %{{.*}} sizes [128] strides[%{{.*}}] valid[%{{.*}}] pad %{{.*}} : f32
  %dm = pim.access %A offset %c0 sizes [128] strides [%c1] valid [%e] pad %pad : f32
      : memref<*xf32> -> !pim.access

  // ⑤ mask 저장: valid만(pad 없음) — 독립 그룹이라 valid가 단독으로 찍혀야 한다.
  // CHECK: pim.access %{{.*}} offset %{{.*}} sizes [128] strides[%{{.*}}] valid[%{{.*}}] : memref<*xf32>
  %ds = pim.access %A offset %c0 sizes [128] strides [%c1] valid [%e]
      : memref<*xf32> -> !pim.access

  // ⑥ row-gather: gather_dim + index clause.
  // CHECK: pim.access %{{.*}} offset %{{.*}} sizes [8, 8] strides[%{{.*}}, %{{.*}}] gather 0 index %{{.*}} : tensor<8xi32>
  %dg = pim.access %A offset %c0 sizes [8, 8] strides [%c1, %c1] gather 0 index %idx : tensor<8xi32>
      : memref<*xf32> -> !pim.access
  return
}

// CHECK-LABEL: func.func @copy
func.func @copy(%A: memref<*xf32>, %C: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ds = pim.access %A offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  %dd = pim.access %C offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  // CHECK: pim.copy %{{.*}}, %{{.*}} : !pim.access, !pim.access
  pim.copy %ds, %dd : !pim.access, !pim.access
  return
}

// CHECK-LABEL: func.func @gather_scatter
func.func @gather_scatter(%A: memref<*xf32>, %C: memref<*xf32>, %idx: tensor<64xi32>) {
  // CHECK: pim.gather %{{.*}} index %{{.*}} : memref<*xf32>, tensor<64xi32> -> tensor<64xf32>
  %v = pim.gather %A index %idx : memref<*xf32>, tensor<64xi32> -> tensor<64xf32>
  // CHECK: pim.scatter %{{.*}}, %{{.*}} index %{{.*}} : tensor<64xf32>, memref<*xf32>, tensor<64xi32>
  pim.scatter %v, %C index %idx : tensor<64xf32>, memref<*xf32>, tensor<64xi32>
  return
}

// CHECK-LABEL: func.func @gather_scatter_masked
func.func @gather_scatter_masked(%A: memref<*xf32>, %C: memref<*xf32>, %idx: tensor<64xi32>,
                                 %mask: tensor<64xi1>, %def: f32) {
  // CHECK: pim.gather %{{.*}} index %{{.*}} mask %{{.*}} : tensor<64xi1> other %{{.*}} : f32 : memref<*xf32>, tensor<64xi32> -> tensor<64xf32>
  %v = pim.gather %A index %idx mask %mask : tensor<64xi1> other %def : f32
      : memref<*xf32>, tensor<64xi32> -> tensor<64xf32>
  // CHECK: pim.scatter %{{.*}}, %{{.*}} index %{{.*}} mask %{{.*}} : tensor<64xi1> : tensor<64xf32>, memref<*xf32>, tensor<64xi32>
  pim.scatter %v, %C index %idx mask %mask : tensor<64xi1>
      : tensor<64xf32>, memref<*xf32>, tensor<64xi32>
  return
}
