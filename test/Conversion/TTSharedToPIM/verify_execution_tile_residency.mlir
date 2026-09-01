// 최종 verifier는 scheduling 산술의 정확한 형태 대신 marker 대응과 target이
// 소비하는 고정 allocation 계약만 확인한다.
// RUN: triton-shared-opt --split-input-file --pim-verify -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024,
    local_capacity_limit_bytes = 0>} {

func.func @wrong_temporal_window(%A: memref<*xi8>) attributes {
    pim.block = #pim.block<shape = [1, 1, 4]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %round = %c0 to %c1 step %c1 {
    pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>} {
    ^bb0(%device_m: index, %device_n: index, %device_k: index):
      pim.place %A start[%c0] size[%c8] allocation_shape[%c8]
          strides[%c1] {direction = "in", access = 0 : i64} : memref<*xi8>
      scf.for %k = %c0 to %c4 step %c1 {
        %access = pim.access %A offset %k sizes [4] strides[%c1] origin[%k]
            {id = 0 : i64} : memref<*xi8> -> !pim.access
      } {pim.reduction = "add", pim.temporal_reduction}
    }
  } {pim.scheduling_axis = "reduction_k"}
  return
}
}

// -----

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024,
    local_capacity_limit_bytes = 0>} {

func.func @temporal_without_semantics(%A: memref<*xi8>) attributes {
    pim.block = #pim.block<shape = [1, 1, 1]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %round = %c0 to %c1 step %c1 {
    pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
    ^bb0(%device_m: index, %device_n: index, %device_k: index):
      pim.place %A start[%c0] size[%c1] allocation_shape[%c1]
          strides[%c1] {direction = "in", access = 0 : i64} : memref<*xi8>
      // expected-error @below {{a temporal K loop needs a semantic reduction}}
      scf.for %k = %c0 to %c1 step %c1 {
        %access = pim.access %A offset %k sizes [1] strides[%c1] origin[%k]
            {id = 0 : i64} : memref<*xi8> -> !pim.access
      } {pim.temporal_reduction}
    }
  } {pim.scheduling_axis = "reduction_k"}
  return
}
}

// -----

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024,
    local_capacity_limit_bytes = 0>} {

func.func @wrong_resident_allocation(%A: memref<*xi8>) attributes {
    pim.block = #pim.block<shape = [4, 1, 1]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [2, 1, 1]>} {
  ^bb0(%device_m: index, %device_n: index, %device_k: index):
    // expected-error @below {{allocation_shape[0] = 4 but the Execution Tile requires 8}}
    pim.place %A start[%c0] size[%c4] allocation_shape[%c4]
        strides[%c1] {direction = "in", access = 0 : i64} : memref<*xi8>
    scf.for %m = %c0 to %c2 step %c1 {
      %access = pim.access %A offset %m sizes [4] strides[%c1] origin[%m]
          {id = 0 : i64} : memref<*xi8> -> !pim.access
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
