// pim.combine은 독립적인 write policy가 아니라 linalg에서 확정된 semantic
// reduction의 partial-to-final 관계다.
// RUN: triton-shared-opt --split-input-file --pim-verify -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

func.func @combine_without_reduction(%C: memref<*xi32>) attributes {
    pim.block = #pim.block<shape = [1, 1, 1]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
      pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
  ^bb0(%device_m: index, %device_n: index, %device_k: index):
    pim.place %C start[%c0] size[%c1] allocation_shape[%c1] strides[%c1]
        {direction = "out", access = 0 : i64} : memref<*xi32>
    %access = pim.access %C offset %c0 sizes [1] strides[%c1] origin[%c0]
        {id = 0 : i64} : memref<*xi32> -> !pim.access
    %partial = tensor.empty() : tensor<1xi32>
    // expected-error @below {{pim.combine requires one semantic pim.reduction}}
    pim.combine %partial into %access kind "add" : tensor<1xi32>, !pim.access
  }
  return
}
}

// -----

module attributes {pim.target_profile = #pim.target_profile<
    target = "test", num_units = 2,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 0>} {

// spatial K scheduling 자체는 정확하지만 final destination이 plane/device K에
// 의존한다. 이는 [p_k,...] partial buffer를 외부 output으로 노출하므로 거부한다.
func.func @k_dependent_final_output(%A: memref<*xi32>, %C: memref<*xi32>)
    attributes {
    pim.block = #pim.block<shape = [1, 1, 1]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %executionTiles = arith.ceildivsi %c2, %c1 : index
  %rounds = arith.ceildivsi %executionTiles, %c2 : index
  %reductionRounds = arith.ceildivsi %c1, %c1 : index
  scf.for %round = %c0 to %rounds step %c1 {
    %roundBase = arith.muli %round, %c2 : index
    scf.for %reductionRound = %c0 to %reductionRounds step %c1 {
      pim.device #pim.device_mapping<partitions = [1, 1, 2]> attributes {
          pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
      ^bb0(%device_m: index, %device_n: index, %device_k: index):
        %tile = arith.addi %roundBase, %device_k : index
        %first = arith.muli %tile, %c1 : index
        %left = arith.subi %c2, %first : index
        %nonnegative = arith.maxsi %left, %c0 : index
        %trips = arith.minsi %c1, %nonnegative : index
        %start = arith.addi %c0, %first : index
        %reductionFirst = arith.muli %reductionRound, %c1 : index
        %reductionLeft = arith.subi %c1, %reductionFirst : index
        %reductionNonnegative = arith.maxsi %reductionLeft, %c0 : index
        %reductionTrips = arith.minsi %c1, %reductionNonnegative : index
        pim.place %A start[%reductionFirst] size[%reductionTrips]
            allocation_shape[%c1] strides[%c1]
            {direction = "in", access = 1 : i64} : memref<*xi32>
        pim.place %C start[%start] size[%trips] allocation_shape[%c1]
            strides[%c1] {direction = "out", access = 0 : i64}
            : memref<*xi32>
        scf.for %local = %c0 to %trips step %c1 {
          %global = arith.addi %start, %local {pim.device_origin} : index
          %empty = tensor.empty() : tensor<1xi32>
          %partial = scf.for %localK = %c0 to %reductionTrips step %c1
              iter_args(%acc = %empty) -> tensor<1xi32> {
            %globalK = arith.addi %reductionFirst, %localK
                {pim.device_origin} : index
            %input = pim.access %A offset %globalK sizes [1] strides[%c1]
                origin[%globalK] {id = 1 : i64}
                : memref<*xi32> -> !pim.access
            scf.yield %acc : tensor<1xi32>
          } {pim.reduction = "add", pim.temporal_reduction}
          %access = pim.access %C offset %global sizes [1] strides[%c1]
              origin[%global] {id = 0 : i64} : memref<*xi32> -> !pim.access
          // expected-error @below {{final output access must not depend on temporal, spatial, or device K scheduling}}
          pim.combine %partial into %access kind "add"
              : tensor<1xi32>, !pim.access
        } {pim.axis = #pim.axis<"k">}
      }
    } {pim.scheduling_axis = "reduction_k"}
  } {pim.scheduling_axis = "k"}
  return
}
}
