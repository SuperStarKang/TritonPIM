// RUN: triton-shared-opt --pim-reuse-execution-tile-inputs %s | FileCheck %s

func.func private @unknown_effect()

// Dynamic zero-trip loops retain their execution condition. The guard is
// uniform device control, not a new Group or an additional input buffer.
// CHECK-LABEL: func.func @guard_empty
// CHECK: %[[NONEMPTY:.*]] = arith.cmpi slt, %{{.*}}, %arg1
// CHECK: %[[COUNT:.*]] = arith.select %[[NONEMPTY]]
// CHECK: scf.for {{.*}} to %[[COUNT]]
// CHECK: %[[ACCESS:.*]] = pim.access
// CHECK: %[[A:.*]] = pim.load %[[ACCESS]]
// CHECK: scf.for {{.*}} to %arg1
// CHECK: linalg.generic {{.*}}ins(%[[A]]
func.func @guard_empty(%A: memref<*xf32>, %trips: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  pim.device #pim.device_mapping<partitions = [1, 2, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [1, 4, 1]>} {
  ^bb0(%dm: index, %dn: index, %dk: index):
    pim.place %A start[%c0] size[%c8] allocation_shape[%c8] strides[%c1]
        {direction = "in", access = 0 : i64} : memref<*xf32>
    scf.for %n = %c0 to %trips step %c1 {
      %aa = pim.access %A offset %c0 sizes [8] strides[%c1] origin[%c0]
          {id = 0 : i64} : memref<*xf32> -> !pim.access
      %a = pim.load %aa : !pim.access -> tensor<8xf32>
      %empty = tensor.empty() : tensor<8xf32>
      %r = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]}
          ins(%a : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
        ^bb0(%x: f32, %old: f32):
          linalg.yield %x : f32
      } -> tensor<8xf32>
    } {pim.axis = #pim.axis<"n">}
  }
  return
}

// An invariant address is insufficient when the valid region varies by Group.
// CHECK-LABEL: func.func @varying_mask
// CHECK: scf.for
// CHECK: pim.access {{.*}}valid[%{{.*}}]
// CHECK: pim.load
func.func @varying_mask(%A: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.0 : f32
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [1, 4, 1]>} {
  ^bb0(%dm: index, %dn: index, %dk: index):
    pim.place %A start[%c0] size[%c8] allocation_shape[%c8] strides[%c1]
        {direction = "in", access = 0 : i64} : memref<*xf32>
    scf.for %n = %c0 to %c4 step %c1 {
      %valid = arith.subi %c8, %n : index
      %aa = pim.access %A offset %c0 sizes [8] strides[%c1] origin[%c0]
          valid[%valid] pad %zero : f32
          {id = 0 : i64} : memref<*xf32> -> !pim.access
      %a = pim.load %aa : !pim.access -> tensor<8xf32>
      %empty = tensor.empty() : tensor<8xf32>
      %r = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]}
          ins(%a : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
        ^bb0(%x: f32, %old: f32):
          linalg.yield %x : f32
      } -> tensor<8xf32>
    } {pim.axis = #pim.axis<"n">}
  }
  return
}

// Cross a statically singleton K loop, substituting only its induction value.
// Its loop and semantic marker remain in place.
// CHECK-LABEL: func.func @singleton_k
// CHECK: %[[AA:.*]] = pim.access
// CHECK: %[[A:.*]] = pim.load %[[AA]]
// CHECK: scf.for
// CHECK: scf.for
// CHECK: linalg.generic {{.*}}ins(%[[A]]
// CHECK: pim.temporal_reduction
func.func @singleton_k(%A: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [1, 4, 1]>} {
  ^bb0(%dm: index, %dn: index, %dk: index):
    pim.place %A start[%c0] size[%c8] allocation_shape[%c8] strides[%c1]
        {direction = "in", access = 0 : i64} : memref<*xf32>
    scf.for %n = %c0 to %c4 step %c1 {
      scf.for %k = %c0 to %c1 step %c1 {
        %origin = arith.muli %k, %c8 : index
        %aa = pim.access %A offset %origin sizes [8] strides[%c1] origin[%origin]
            {id = 0 : i64} : memref<*xf32> -> !pim.access
        %a = pim.load %aa : !pim.access -> tensor<8xf32>
        %empty = tensor.empty() : tensor<8xf32>
        %r = linalg.generic {
            indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
            iterator_types = ["parallel"]}
            ins(%a : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
          ^bb0(%x: f32, %old: f32):
            linalg.yield %x : f32
        } -> tensor<8xf32>
      } {pim.temporal_reduction}
    } {pim.axis = #pim.axis<"n">}
  }
  return
}

// Unknown/multi-iteration K loops are not crossed, even when A itself is
// invariant. A zero-trip K loop must not acquire a new input read either.
// CHECK-LABEL: func.func @unknown_k
// CHECK: scf.for
// CHECK: scf.for {{.*}} to %arg1
// CHECK: pim.load
func.func @unknown_k(%A: memref<*xf32>, %kTrips: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  pim.device #pim.device_mapping<partitions = [1, 1, 2]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [1, 4, 4]>} {
  ^bb0(%dm: index, %dn: index, %dk: index):
    pim.place %A start[%c0] size[%c8] allocation_shape[%c8] strides[%c1]
        {direction = "in", access = 0 : i64} : memref<*xf32>
    scf.for %n = %c0 to %c4 step %c1 {
      scf.for %k = %c0 to %kTrips step %c1 {
        %aa = pim.access %A offset %c0 sizes [8] strides[%c1] origin[%c0]
            {id = 0 : i64} : memref<*xf32> -> !pim.access
        %a = pim.load %aa : !pim.access -> tensor<8xf32>
        %empty = tensor.empty() : tensor<8xf32>
        %r = linalg.generic {
            indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
            iterator_types = ["parallel"]}
            ins(%a : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
          ^bb0(%x: f32, %old: f32):
            linalg.yield %x : f32
        } -> tensor<8xf32>
      } {pim.temporal_reduction}
    } {pim.axis = #pim.axis<"n">}
  }
  return
}

// A tensor used as a DPS initializer can be overwritten by local bufferization.
// CHECK-LABEL: func.func @mutable_local_tile
// CHECK: scf.for
// CHECK: %[[A:.*]] = pim.load
// CHECK: linalg.fill {{.*}}outs(%[[A]]
func.func @mutable_local_tile(%A: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.0 : f32
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [1, 4, 1]>} {
  ^bb0(%dm: index, %dn: index, %dk: index):
    pim.place %A start[%c0] size[%c8] allocation_shape[%c8] strides[%c1]
        {direction = "in", access = 0 : i64} : memref<*xf32>
    scf.for %n = %c0 to %c4 step %c1 {
      %aa = pim.access %A offset %c0 sizes [8] strides[%c1] origin[%c0]
          {id = 0 : i64} : memref<*xf32> -> !pim.access
      %a = pim.load %aa : !pim.access -> tensor<8xf32>
      %r = linalg.fill ins(%zero : f32) outs(%a : tensor<8xf32>) -> tensor<8xf32>
    } {pim.axis = #pim.axis<"n">}
  }
  return
}

// Unknown effects block reuse even when the address and tensor consumers pass.
// CHECK-LABEL: func.func @unknown_write
// CHECK: scf.for
// CHECK: func.call @unknown_effect
// CHECK: pim.load
func.func @unknown_write(%A: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [1, 4, 1]>} {
  ^bb0(%dm: index, %dn: index, %dk: index):
    pim.place %A start[%c0] size[%c8] allocation_shape[%c8] strides[%c1]
        {direction = "in", access = 0 : i64} : memref<*xf32>
    scf.for %n = %c0 to %c4 step %c1 {
      func.call @unknown_effect() : () -> ()
      %aa = pim.access %A offset %c0 sizes [8] strides[%c1] origin[%c0]
          {id = 0 : i64} : memref<*xf32> -> !pim.access
      %a = pim.load %aa : !pim.access -> tensor<8xf32>
      %empty = tensor.empty() : tensor<8xf32>
      %r = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]}
          ins(%a : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
        ^bb0(%x: f32, %old: f32):
          linalg.yield %x : f32
      } -> tensor<8xf32>
    } {pim.axis = #pim.axis<"n">}
  }
  return
}
