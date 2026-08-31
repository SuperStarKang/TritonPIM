// Temporal K의 host round, device-local window, global origin, resident start가
// 모두 같은 Execution Tile 시작을 사용하는지 검증한다.
// RUN: triton-shared-opt --split-input-file --pim-verify -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

// 정상 계약:
//   rounds = ceildiv(totalGroups, 2)
//   first = round * 2
//   window = min(2, max(totalGroups - first, 0))
func.func @valid_temporal_schedule(%totalGroups: index, %A: memref<*xi8>)
    attributes {
      pim.block = #pim.block<shape = [1, 1, 4]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rounds = arith.ceildivsi %totalGroups, %c2 : index
  scf.for %round = %c0 to %rounds step %c1 {
    %first = arith.muli %round, %c2 : index
    %left = arith.subi %totalGroups, %first : index
    %nonnegative = arith.maxsi %left, %c0 : index
    %window = arith.minsi %c2, %nonnegative : index
    %firstElement = arith.muli %first, %c4 : index
    %windowElements = arith.muli %window, %c4 : index
    pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>} {
    ^bb0(%device_m: index, %device_n: index, %device_k: index):
      pim.place %A start[%firstElement] size[%windowElements]
          allocation_shape[%c8] strides[%c1]
          {direction = "in", access = 0 : i64} : memref<*xi8>
      scf.for %local = %c0 to %window step %c1 {
        %global = arith.addi %first, %local {pim.device_origin} : index
        %globalElement = arith.muli %global, %c4 : index
        %access = pim.access %A offset %globalElement sizes [4] strides[%c1]
            origin[%globalElement] {id = 0 : i64}
            : memref<*xi8> -> !pim.access
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
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @wrong_round_count(%totalGroups: index, %A: memref<*xi8>)
    attributes {
      pim.block = #pim.block<shape = [1, 1, 4]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  // tileK=2인데 전체 Group 수를 1로 나눠 round를 너무 많이 만든다.
  %rounds = arith.ceildivsi %totalGroups, %c1 : index
  // expected-error @below {{the temporal K scheduling round upper bound must be ceildiv(total reduction groups, execution_tile.groups[k])}}
  scf.for %round = %c0 to %rounds step %c1 {
    %first = arith.muli %round, %c2 : index
    %left = arith.subi %totalGroups, %first : index
    %nonnegative = arith.maxsi %left, %c0 : index
    %window = arith.minsi %c2, %nonnegative : index
    %firstElement = arith.muli %first, %c4 : index
    %windowElements = arith.muli %window, %c4 : index
    pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>} {
    ^bb0(%device_m: index, %device_n: index, %device_k: index):
      pim.place %A start[%firstElement] size[%windowElements]
          allocation_shape[%c8] strides[%c1]
          {direction = "in", access = 0 : i64} : memref<*xi8>
      scf.for %local = %c0 to %window step %c1 {
        %global = arith.addi %first, %local {pim.device_origin} : index
        %globalElement = arith.muli %global, %c4 : index
        %access = pim.access %A offset %globalElement sizes [4] strides[%c1]
            origin[%globalElement] {id = 0 : i64}
            : memref<*xi8> -> !pim.access
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
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @duplicate_round(%totalGroups: index, %A: memref<*xi8>)
    attributes {
      pim.block = #pim.block<shape = [1, 1, 4]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rounds = arith.ceildivsi %totalGroups, %c2 : index
  scf.for %outer = %c0 to %c1 step %c1 {
    scf.for %round = %c0 to %rounds step %c1 {
      %first = arith.muli %round, %c2 : index
      %left = arith.subi %totalGroups, %first : index
      %nonnegative = arith.maxsi %left, %c0 : index
      %window = arith.minsi %c2, %nonnegative : index
      %firstElement = arith.muli %first, %c4 : index
      %windowElements = arith.muli %window, %c4 : index
      // expected-error @below {{one temporal K reduction requires exactly one reduction_k scheduling round}}
      pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
          pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>} {
      ^bb0(%device_m: index, %device_n: index, %device_k: index):
        pim.place %A start[%firstElement] size[%windowElements]
            allocation_shape[%c8] strides[%c1]
            {direction = "in", access = 0 : i64} : memref<*xi8>
        scf.for %local = %c0 to %window step %c1 {
          %global = arith.addi %first, %local {pim.device_origin} : index
          %globalElement = arith.muli %global, %c4 : index
          %access = pim.access %A offset %globalElement sizes [4] strides[%c1]
              origin[%globalElement] {id = 0 : i64}
              : memref<*xi8> -> !pim.access
        } {pim.reduction = "add", pim.temporal_reduction}
      }
    } {pim.scheduling_axis = "reduction_k"}
  } {pim.scheduling_axis = "reduction_k"}
  return
}
}

// -----

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @origin_does_not_advance(%totalGroups: index, %A: memref<*xi8>)
    attributes {
      pim.block = #pim.block<shape = [1, 1, 4]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rounds = arith.ceildivsi %totalGroups, %c2 : index
  scf.for %round = %c0 to %rounds step %c1 {
    %first = arith.muli %round, %c2 : index
    %left = arith.subi %totalGroups, %first : index
    %nonnegative = arith.maxsi %left, %c0 : index
    %window = arith.minsi %c2, %nonnegative : index
    %firstElement = arith.muli %first, %c4 : index
    %windowElements = arith.muli %window, %c4 : index
    pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>} {
    ^bb0(%device_m: index, %device_n: index, %device_k: index):
      pim.place %A start[%firstElement] size[%windowElements]
          allocation_shape[%c8] strides[%c1]
          {direction = "in", access = 0 : i64} : memref<*xi8>
      scf.for %local = %c0 to %window step %c1 {
        // expected-error @below {{the temporal K global origin does not use the current Execution Tile window start}}
        %global = arith.addi %c0, %local {pim.device_origin} : index
        %globalElement = arith.muli %global, %c4 : index
        %access = pim.access %A offset %globalElement sizes [4] strides[%c1]
            origin[%globalElement] {id = 0 : i64}
            : memref<*xi8> -> !pim.access
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
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @placement_does_not_advance(%totalGroups: index, %A: memref<*xi8>)
    attributes {
      pim.block = #pim.block<shape = [1, 1, 4]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rounds = arith.ceildivsi %totalGroups, %c2 : index
  scf.for %round = %c0 to %rounds step %c1 {
    %first = arith.muli %round, %c2 : index
    %left = arith.subi %totalGroups, %first : index
    %nonnegative = arith.maxsi %left, %c0 : index
    %window = arith.minsi %c2, %nonnegative : index
    %windowElements = arith.muli %window, %c4 : index
    pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>} {
    ^bb0(%device_m: index, %device_n: index, %device_k: index):
      // expected-error @below {{resident buffer axis 0 is driven by temporal K but start does not use the current Execution Tile window start}}
      pim.place %A start[%c0] size[%windowElements]
          allocation_shape[%c8] strides[%c1]
          {direction = "in", access = 0 : i64} : memref<*xi8>
      scf.for %local = %c0 to %window step %c1 {
        %global = arith.addi %first, %local {pim.device_origin} : index
        %globalElement = arith.muli %global, %c4 : index
        %access = pim.access %A offset %globalElement sizes [4] strides[%c1]
            origin[%globalElement] {id = 0 : i64}
            : memref<*xi8> -> !pim.access
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
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @placement_ignores_tail(%totalGroups: index, %A: memref<*xi8>)
    attributes {
      pim.block = #pim.block<shape = [1, 1, 4]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rounds = arith.ceildivsi %totalGroups, %c2 : index
  scf.for %round = %c0 to %rounds step %c1 {
    %first = arith.muli %round, %c2 : index
    %left = arith.subi %totalGroups, %first : index
    %nonnegative = arith.maxsi %left, %c0 : index
    %window = arith.minsi %c2, %nonnegative : index
    %firstElement = arith.muli %first, %c4 : index
    pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>} {
    ^bb0(%device_m: index, %device_n: index, %device_k: index):
      // expected-error @below {{resident buffer axis 0 is driven by temporal K but size does not use the current local window length}}
      pim.place %A start[%firstElement] size[%c8]
          allocation_shape[%c8] strides[%c1]
          {direction = "in", access = 0 : i64} : memref<*xi8>
      scf.for %local = %c0 to %window step %c1 {
        %global = arith.addi %first, %local {pim.device_origin} : index
        %globalElement = arith.muli %global, %c4 : index
        %access = pim.access %A offset %globalElement sizes [4] strides[%c1]
            origin[%globalElement] {id = 0 : i64}
            : memref<*xi8> -> !pim.access
      } {pim.reduction = "add", pim.temporal_reduction}
    }
  } {pim.scheduling_axis = "reduction_k"}
  return
}
}
