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
