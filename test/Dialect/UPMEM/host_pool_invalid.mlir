// RUN: triton-shared-opt -split-input-file -verify-diagnostics %s

func.func @empty_output_pool(%set: !upmem.set, %assignment: !upmem.assignment,
                             %pool: memref<0x8xi8>) {
  %c8 = arith.constant 8 : index
  // expected-error @below {{fixed-slot shape must have positive extents}}
  upmem.gather %set assignment %assignment, @C capacity[%c8] into %pool
      : !upmem.set, memref<0x8xi8>
  return
}

// -----

func.func @too_few_input_slots(%A: memref<*xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %set = upmem.acquire 4, @kernel : !upmem.set
  %assignment = upmem.bind_devices %set partitions [2, 2] device_mapping [1, 2]
      : !upmem.set -> !upmem.assignment
  %pool = memref.alloc() : memref<1x8xi8>
  upmem.foreach_unit %set assignment %assignment : !upmem.set axes ["m", "n"] counts [2, 2] {
  ^bb0(%dpu: !upmem.unit, %pm: index, %pn: index):
    // expected-error @below {{pool leading extent must equal the number of distinct inputs}}
    %tile = upmem.stage %A origin[%c0] sizes[%c8] strides[%c1]
        pad_to[%c8] tile [8] into %pool {shared_over = ["n"]}
        : memref<*xi8>, memref<1x8xi8> -> memref<8xi8>
  }
  memref.dealloc %pool : memref<1x8xi8>
  return
}

// -----

func.func @wrong_output_slots() {
  %c8 = arith.constant 8 : index
  %set = upmem.acquire 4, @kernel : !upmem.set
  %assignment = upmem.bind_devices %set partitions [2, 2] device_mapping [1, 2]
      : !upmem.set -> !upmem.assignment
  %pool = memref.alloc() : memref<2x8xi8>
  // expected-error @below {{pool leading extent must equal the acquired DPU count}}
  upmem.gather %set assignment %assignment, @C capacity[%c8] into %pool
      : !upmem.set, memref<2x8xi8>
  memref.dealloc %pool : memref<2x8xi8>
  return
}

// -----

func.func @unknown_shared_axis(%A: memref<*xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %set = upmem.acquire 4, @kernel : !upmem.set
  %assignment = upmem.bind_devices %set partitions [2, 2] device_mapping [1, 2]
      : !upmem.set -> !upmem.assignment
  %pool = memref.alloc() : memref<4x8xi8>
  upmem.foreach_unit %set assignment %assignment : !upmem.set axes ["m", "n"] counts [2, 2] {
  ^bb0(%dpu: !upmem.unit, %pm: index, %pn: index):
    // expected-error @below {{shared_over must name a foreach_unit axis}}
    %tile = upmem.stage %A origin[%c0] sizes[%c8] strides[%c1]
        pad_to[%c8] tile [8] into %pool {shared_over = ["k"]}
        : memref<*xi8>, memref<4x8xi8> -> memref<8xi8>
  }
  memref.dealloc %pool : memref<4x8xi8>
  return
}

// -----

func.func @wrong_input_element_type(%A: memref<*xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %set = upmem.acquire 1, @kernel : !upmem.set
  %assignment = upmem.bind_devices %set partitions [1] device_mapping [1]
      : !upmem.set -> !upmem.assignment
  %pool = memref.alloc() : memref<1x8xi32>
  upmem.foreach_unit %set assignment %assignment : !upmem.set axes ["m"] counts [1] {
  ^bb0(%dpu: !upmem.unit, %pm: index):
    // expected-error @below {{source, pool, and result element types must match}}
    %tile = upmem.stage %A origin[%c0] sizes[%c8] strides[%c1]
        pad_to[%c8] tile [8] into %pool
        : memref<*xi8>, memref<1x8xi32> -> memref<8xi8>
  }
  memref.dealloc %pool : memref<1x8xi32>
  return
}
