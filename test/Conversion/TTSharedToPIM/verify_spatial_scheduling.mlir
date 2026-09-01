// M/N scheduling의 산술 형태는 생성 pass가 책임진다. 최종 verifier는 marker의
// 일대일 대응과 source spatial-K 금지만 확인한다.
// RUN: triton-shared-opt --split-input-file --pim-verify -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 2,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @m_start_ignores_device(%total: index) attributes {
    pim.block = #pim.block<shape = [1, 1, 1]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %tiles = arith.ceildivsi %total, %c2 : index
  %rounds = arith.ceildivsi %tiles, %c2 : index
  scf.for %round = %c0 to %rounds step %c1 {
    // 잘못된 식: device_m 좌표가 빠졌다.
    %roundBase = arith.muli %round, %c2 : index
    %first = arith.muli %roundBase, %c2 : index
    %left = arith.subi %total, %first : index
    %nonnegative = arith.maxsi %left, %c0 : index
    %window = arith.minsi %c2, %nonnegative : index
    pim.device #pim.device_mapping<partitions = [2, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [2, 1, 1]>} {
    ^bb0(%device_m: index, %device_n: index, %device_k: index):
      scf.for %local = %c0 to %window step %c1 {
        %start = arith.addi %c0, %first : index
        %scaled = arith.muli %local, %c1 : index
        %global = arith.addi %start, %scaled {pim.device_origin} : index
      } {pim.axis = #pim.axis<"m">}
    }
  } {pim.scheduling_axis = "m"}
  return
}
}

// -----

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 2,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @n_round_uses_wrong_partition(%total: index) attributes {
    pim.block = #pim.block<shape = [1, 1, 1]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %tiles = arith.ceildivsi %total, %c2 : index
  // 잘못된 식: p_n=2인데 1로 나눈다.
  %rounds = arith.ceildivsi %tiles, %c1 : index
  scf.for %round = %c0 to %rounds step %c1 {
    %roundBase = arith.muli %round, %c2 : index
    pim.device #pim.device_mapping<partitions = [1, 2, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 2, 1]>} {
    ^bb0(%device_m: index, %device_n: index, %device_k: index):
      %tile = arith.addi %roundBase, %device_n : index
      %first = arith.muli %tile, %c2 : index
      %left = arith.subi %total, %first : index
      %nonnegative = arith.maxsi %left, %c0 : index
      %window = arith.minsi %c2, %nonnegative : index
      scf.for %local = %c0 to %window step %c1 {
        %start = arith.addi %c0, %first : index
        %scaled = arith.muli %local, %c1 : index
        %global = arith.addi %start, %scaled {pim.device_origin} : index
      } {pim.axis = #pim.axis<"n">}
    }
  } {pim.scheduling_axis = "n"}
  return
}
}

// -----

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 2,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @source_spatial_k_is_rejected(%total: index) attributes {
    pim.block = #pim.block<shape = [1, 1, 1]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // 아래는 제거된 source spatial-K 표현이다.
  %tiles = arith.ceildivsi %total, %c1 : index
  %rounds = arith.ceildivsi %tiles, %c2 : index
  scf.for %round = %c0 to %rounds step %c1 {
    %roundBase = arith.muli %round, %c2 : index
    pim.device #pim.device_mapping<partitions = [1, 1, 2]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
    ^bb0(%device_m: index, %device_n: index, %device_k: index):
      %first = arith.addi %roundBase, %device_k : index
      %left = arith.subi %total, %first : index
      %nonnegative = arith.maxsi %left, %c0 : index
      // scheduling 식과 무관하게 source pim.axis="k" 자체가 불법이다.
      %window = arith.minsi %c2, %nonnegative : index
      // expected-error @below {{K must be a semantic reduction, not a spatial axis}}
      scf.for %local = %c0 to %window step %c1 {
        %start = arith.addi %c0, %first : index
        %scaled = arith.muli %local, %c1 : index
        %global = arith.addi %start, %scaled {pim.device_origin} : index
      } {pim.axis = #pim.axis<"k">}
    }
  } {pim.scheduling_axis = "k"}
  return
}
}

// -----

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @m_without_round() attributes {
    pim.block = #pim.block<shape = [1, 1, 1]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  ^bb0(%device_m: index, %device_n: index, %device_k: index):
    // expected-error @below {{spatial M axis needs one scheduling round}}
    scf.for %local = %c0 to %c1 step %c1 {
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
