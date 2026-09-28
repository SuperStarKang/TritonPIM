// RUN: triton-shared-opt -split-input-file -verify-diagnostics %s

func.func @missing_input(%A: memref<*xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pim.partition #pim.device_mapping<partitions = [1, 1, 1]> {
  ^bb0(%pm: index, %pn: index, %pk: index):
    // expected-error @below {{input placement has no transfer}}
    pim.place %A start[%c0] size[%c1] allocation_shape[%c1]
        strides[%c1] {access = 0 : i64, direction = "in"} : memref<*xi8>
    pim.launch {
      %unused = arith.constant 0 : index
    }
  }
  return
}

// -----

func.func @duplicate_input(%A: memref<*xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pim.partition #pim.device_mapping<partitions = [1, 1, 1]> {
  ^bb0(%pm: index, %pn: index, %pk: index):
    pim.place %A start[%c0] size[%c1] allocation_shape[%c1]
        strides[%c1] {access = 0 : i64, direction = "in"} : memref<*xi8>
    pim.scatter 0
    // expected-error @below {{input placement is transferred more than once}}
    pim.scatter 0
    pim.launch {
      %unused = arith.constant 0 : index
    }
  }
  return
}

// -----

func.func @broadcast_requires_sharing(%A: memref<*xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pim.partition #pim.device_mapping<partitions = [1, 1, 1]> {
  ^bb0(%pm: index, %pn: index, %pk: index):
    pim.place %A start[%c0] size[%c1] allocation_shape[%c1]
        strides[%c1] {access = 0 : i64, direction = "in"} : memref<*xi8>
    // expected-error @below {{broadcast requires nonempty shared_over}}
    pim.broadcast 0
    pim.launch {
      %unused = arith.constant 0 : index
    }
  }
  return
}

// -----

func.func @early_gather(%C: memref<*xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pim.partition #pim.device_mapping<partitions = [1, 1, 1]> {
  ^bb0(%pm: index, %pn: index, %pk: index):
    pim.place %C start[%c0] size[%c1] allocation_shape[%c1]
        strides[%c1] {access = 0 : i64, direction = "out"} : memref<*xi8>
    // expected-error @below {{gather must follow pim.launch}}
    pim.gather 0
    pim.launch {
      %unused = arith.constant 0 : index
    }
  }
  return
}

// -----

func.func @missing_gather(%C: memref<*xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pim.partition #pim.device_mapping<partitions = [1, 1, 1]> {
  ^bb0(%pm: index, %pn: index, %pk: index):
    // expected-error @below {{output placement has no gather}}
    pim.place %C start[%c0] size[%c1] allocation_shape[%c1]
        strides[%c1] {access = 0 : i64, direction = "out"} : memref<*xi8>
    pim.launch {
      %unused = arith.constant 0 : index
    }
  }
  return
}

// -----

func.func @duplicate_launch() {
  pim.partition #pim.device_mapping<partitions = [1, 1, 1]> {
  ^bb0(%pm: index, %pn: index, %pk: index):
    pim.launch {
      %unused = arith.constant 0 : index
    }
    // expected-error @below {{partition must contain exactly one pim.launch}}
    pim.launch {
      %unused = arith.constant 0 : index
    }
  }
  return
}
