// Block을 Group으로 확대하면 access/tensor shape뿐 아니라 valid도 새 Group 시작과
// 크기로 다시 clamp되어야 한다. 입력 pad와 출력 mask는 같은 원본 extent를 쓴다.
// RUN: triton-shared-opt "--pim-form-block-groups=group-m=2" -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @masked_elementwise
// CHECK-SAME: pim.block = #pim.block<shape = [8, 1, 1]>
// CHECK-SAME: pim.block_group = #pim.block_group<blocks = [2, 1, 1]>
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: arith.ceildivsi
// CHECK: %[[REMAINING:.*]] = arith.subi %arg2, %{{.*}} : index
// CHECK: %[[NONNEGATIVE:.*]] = arith.maxsi %[[REMAINING]], %{{.*}} : index
// CHECK: %[[VALID:.*]] = arith.minsi %[[NONNEGATIVE]], %[[C16]] : index
// CHECK: pim.access %arg0 {{.*}} sizes [16] {{.*}} valid[%[[VALID]]] pad %{{.*}} : f32
// CHECK: pim.load %{{.*}} : !pim.access -> tensor<16xf32>
// CHECK: pim.access %arg1 {{.*}} sizes [16] {{.*}} valid[%{{.*}}]
// CHECK: pim.store %{{.*}}, %{{.*}} : tensor<16xf32>, !pim.access

func.func @masked_elementwise(%A: memref<*xf32>, %C: memref<*xf32>,
                              %elements: index, %tiles: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.0 : f32
  scf.for %tile = %c0 to %tiles step %c1 {
    %origin = arith.muli %tile, %c8 : index
    %end = arith.addi %origin, %c8 : index
    %clipped = arith.minsi %end, %elements : index
    %notBeforeOrigin = arith.maxsi %clipped, %origin : index
    %span = arith.subi %notBeforeOrigin, %origin : index
    %valid = arith.minsi %span, %c8 : index
    %inAccess = pim.access %A offset %origin sizes [8] strides[%c1]
        origin[%origin] valid[%valid] pad %zero : f32 {id = 0 : i64}
        : memref<*xf32> -> !pim.access
    %value = pim.load %inAccess : !pim.access -> tensor<8xf32>
    %outAccess = pim.access %C offset %origin sizes [8] strides[%c1]
        origin[%origin] valid[%valid] {id = 1 : i64}
        : memref<*xf32> -> !pim.access
    pim.store %value, %outAccess : tensor<8xf32>, !pim.access
  } {pim.axis = #pim.axis<"m">}
  return
}
