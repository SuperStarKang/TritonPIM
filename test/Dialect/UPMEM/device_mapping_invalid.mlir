// RUN: triton-shared-opt -split-input-file -verify-diagnostics %s

func.func @non_divisor() {
  %set = upmem.acquire 12, @kernel : !upmem.set
  // expected-error @below {{device_mapping entries must be positive divisors of partitions}}
  %map = upmem.bind_devices %set partitions [3, 4, 1] device_mapping [2, 4, 1]
      : !upmem.set -> !upmem.assignment
  return
}

// -----

func.func @too_wide() {
  %set = upmem.acquire 64, @kernel : !upmem.set
  // expected-error @below {{device_mapping must fit in an 8-DPU member group}}
  %map = upmem.bind_devices %set partitions [4, 4, 4] device_mapping [4, 4, 1]
      : !upmem.set -> !upmem.assignment
  return
}

// -----

func.func @wrong_count() {
  %set = upmem.acquire 8, @kernel : !upmem.set
  // expected-error @below {{partitions must match the acquired DPU count}}
  %map = upmem.bind_devices %set partitions [4, 4, 4] device_mapping [2, 4, 1]
      : !upmem.set -> !upmem.assignment
  return
}

// -----

func.func @wrong_set() {
  %first = upmem.acquire 8, @kernel : !upmem.set
  %second = upmem.acquire 8, @kernel : !upmem.set
  %map = upmem.bind_devices %first partitions [2, 4, 1] device_mapping [2, 4, 1]
      : !upmem.set -> !upmem.assignment
  // expected-error @below {{assignment must bind this set with the same partition counts}}
  upmem.foreach_unit %second assignment %map : !upmem.set
      axes ["m", "n", "k"] counts [2, 4, 1] {
  ^bb0(%dpu: !upmem.unit, %m: index, %n: index, %k: index):
  }
  return
}

// -----

func.func @wrong_gather_set(%pool: memref<8x8xi8>) {
  %c8 = arith.constant 8 : index
  %first = upmem.acquire 8, @kernel : !upmem.set
  %second = upmem.acquire 8, @kernel : !upmem.set
  %map = upmem.bind_devices %first partitions [2, 4, 1] device_mapping [2, 4, 1]
      : !upmem.set -> !upmem.assignment
  // expected-error @below {{assignment must bind this set}}
  upmem.gather %second assignment %map, @C capacity[%c8] into %pool
      : !upmem.set, memref<8x8xi8>
  return
}
