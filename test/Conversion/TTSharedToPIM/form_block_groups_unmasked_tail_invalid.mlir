// valid가 없는 access는 Group에 phantom Block이 생길 가능성이 없어야 한다.
// RUN: triton-shared-opt --split-input-file "--pim-form-block-groups=group-m=2" -verify-diagnostics %s

// 정적 Block 수 3은 group-m=2로 나누어지지 않는다.
func.func @static_incomplete_group(%A: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  scf.for %tile = %c0 to %c3 step %c1 {
    %origin = arith.muli %tile, %c8 : index
    // expected-error @below {{an unmasked buffer axis may have an incomplete Block Group; its Block count must be statically divisible by the Group size 2}}
    %access = pim.access %A offset %origin sizes [8] strides[%c1]
        origin[%origin] {id = 0 : i64} : memref<*xf32> -> !pim.access
    %value = pim.load %access : !pim.access -> tensor<8xf32>
  } {pim.axis = #pim.axis<"m">}
  return
}

// -----

// runtime Block 수는 divisibility를 증명할 수 없으므로 dynamic valid가 필요하다.
func.func @runtime_group_count(%A: memref<*xf32>, %tiles: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %tile = %c0 to %tiles step %c1 {
    %origin = arith.muli %tile, %c8 : index
    // expected-error @below {{an unmasked buffer axis may have an incomplete Block Group; its Block count must be statically divisible by the Group size 2}}
    %access = pim.access %A offset %origin sizes [8] strides[%c1]
        origin[%origin] {id = 0 : i64} : memref<*xf32> -> !pim.access
    %value = pim.load %access : !pim.access -> tensor<8xf32>
  } {pim.axis = #pim.axis<"m">}
  return
}

// -----

// valid=Block 크기라는 상수도 마지막 Group의 실제 남은 원소 수를 복원하지 못한다.
func.func @constant_full_valid(%A: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  scf.for %tile = %c0 to %c3 step %c1 {
    %origin = arith.muli %tile, %c8 : index
    // expected-error @below {{a constant full valid extent cannot describe an incomplete Block Group tail on buffer axis 0}}
    %access = pim.access %A offset %origin sizes [8] strides[%c1]
        origin[%origin] valid[%c8] {id = 0 : i64}
        : memref<*xf32> -> !pim.access
    %value = pim.load %access : !pim.access -> tensor<8xf32>
  } {pim.axis = #pim.axis<"m">}
  return
}
