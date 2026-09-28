// Constant masks must not expand into an unmasked resident placement.
// RUN: triton-shared-opt --pim-place-data --canonicalize %s | FileCheck %s --check-prefix=ZERO
// RUN: sed 's/valid\[%c0\]/valid[%c8]/' %s | triton-shared-opt --pim-place-data --canonicalize | FileCheck %s --check-prefix=FULL
// RUN: sed 's/valid\[%c0\]/valid[%c3]/' %s | not triton-shared-opt --pim-place-data 2>&1 | FileCheck %s --check-prefix=HOLES
// RUN: sed -e 's/valid\[%c0\]/valid[%c3]/' -e 's/groups = \[2, 1, 1\]/groups = [1, 1, 1]/' -e 's/to %c2 step/to %c1 step/' %s | triton-shared-opt --pim-place-data --canonicalize | FileCheck %s --check-prefix=PARTIAL

// ZERO-LABEL: func.func @constant_mask
// ZERO-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// ZERO-DAG: %[[CAP:.*]] = arith.constant 16 : index
// ZERO: pim.place {{.*}} size[%[[ZERO]]] allocation_shape[%[[CAP]]]
// FULL-LABEL: func.func @constant_mask
// FULL: %[[CAP:.*]] = arith.constant 16 : index
// FULL: pim.place {{.*}} size[%[[CAP]]] allocation_shape[%[[CAP]]]
// PARTIAL-LABEL: func.func @constant_mask
// PARTIAL-DAG: %[[VALID:.*]] = arith.constant 3 : index
// PARTIAL-DAG: %[[CAP:.*]] = arith.constant 8 : index
// PARTIAL: pim.place {{.*}} size[%[[VALID]]] allocation_shape[%[[CAP]]]
// HOLES: repeated constant partial masks need a non-contiguous placement

module {
func.func @constant_mask(%A: memref<*xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 0 : i8
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [2, 1, 1]>} {
  ^bb0(%dm: index, %dn: index, %dk: index):
    scf.for %group = %c0 to %c2 step %c1 {
      %origin = arith.muli %group, %c8 : index
      %access = pim.access %A offset %origin sizes [8] strides[%c1]
          origin[%origin] valid[%c0] pad %pad : i8 {id = 0 : i64}
          : memref<*xi8> -> !pim.access
      %tile = pim.load %access : !pim.access -> tensor<8xi8>
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
