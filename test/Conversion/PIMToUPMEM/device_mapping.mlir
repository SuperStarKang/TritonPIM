// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-upmem %s | FileCheck %s
// Physical grouping follows shared_over and fixed-slot bytes, not input names.
// These are valid one-round GEMMs with P=[4,128,4] and split-K outputs.
module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 2048, banks_per_unit = 1,
    cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 62914560, local_capacity_limit_bytes = 49152>} {

// CHECK-LABEL: func.func @large_lhs
// CHECK: %[[SET_large_lhs:.*]] = upmem.acquire 2048
// CHECK: %[[MAP_large_lhs:.*]] = upmem.bind_devices %[[SET_large_lhs]] partitions [4, 128, 4] device_mapping [1, 8, 1]
// CHECK: upmem.foreach_unit %[[SET_large_lhs]] assignment %[[MAP_large_lhs]]
// CHECK: upmem.to_mram
// CHECK: upmem.to_host_var
// CHECK: upmem.gather %[[SET_large_lhs]] assignment %[[MAP_large_lhs]]
// CHECK: upmem.foreach_group
// CHECK: accumulate
func.func @large_lhs(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>)
    attributes {pim.block = #pim.block<shape = [64, 8, 8]>,
                pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %bm = arith.constant 64 : index
  %bn = arith.constant 8 : index
  %N = arith.constant 1024 : index
  %zero = arith.constant 0 : i32
  scf.for %round = %c0 to %c1 step %c1 {
    pim.device #pim.device_mapping<partitions = [4, 128, 4]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
    ^bb0(%pm: index, %pn: index, %pk: index):
      %mStart = arith.muli %pm, %bm : index
      %nStart = arith.muli %pn, %bn : index
      %kStart = arith.muli %pk, %c8 : index
      pim.place %A start[%mStart, %kStart] size[%bm, %c8]
          allocation_shape[%bm, %c8] strides[%c32, %c1]
          {direction = "in", access = 0 : i64, shared_over = ["n"]} : memref<*xi8>
      pim.place %B start[%kStart, %nStart] size[%c8, %bn]
          allocation_shape[%c8, %bn] strides[%N, %c1]
          {direction = "in", access = 1 : i64, shared_over = ["m"]} : memref<*xi8>
      pim.place %C start[%mStart, %nStart] size[%bm, %bn]
          allocation_shape[%bm, %bn] strides[%N, %c1]
          {direction = "out", access = 2 : i64} : memref<*xi32>
      scf.for %mt = %c0 to %c1 step %c1 {
        %gm = arith.addi %pm, %mt {pim.device_origin} : index
        %m = arith.muli %gm, %bm : index
        scf.for %nt = %c0 to %c1 step %c1 {
          %gn = arith.addi %pn, %nt {pim.device_origin} : index
          %n = arith.muli %gn, %bn : index
          %empty = tensor.empty() : tensor<64x8xi32>
          %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<64x8xi32>)
              -> tensor<64x8xi32>
          %sum = scf.for %rk = %c0 to %c1 step %c1
              iter_args(%acc = %init) -> tensor<64x8xi32> {
            %gk = arith.addi %pk, %rk {pim.device_origin} : index
            %k = arith.muli %gk, %c8 : index
            %ar = arith.muli %m, %c32 : index
            %ao = arith.addi %ar, %k : index
            %aa = pim.access %A offset %ao sizes [64, 8]
                strides[%c32, %c1] origin[%m, %k] {id = 0 : i64}
                : memref<*xi8> -> !pim.access
            %a = pim.load %aa : !pim.access -> tensor<64x8xi8>
            %br = arith.muli %k, %N : index
            %bo = arith.addi %br, %n : index
            %ba = pim.access %B offset %bo sizes [8, 8]
                strides[%N, %c1] origin[%k, %n] {id = 1 : i64}
                : memref<*xi8> -> !pim.access
            %b = pim.load %ba : !pim.access -> tensor<8x8xi8>
            %next = linalg.matmul ins(%a, %b : tensor<64x8xi8>, tensor<8x8xi8>)
                outs(%acc : tensor<64x8xi32>) -> tensor<64x8xi32>
            scf.yield %next : tensor<64x8xi32>
          } {pim.reduction = "add", pim.temporal_reduction}
          %cr = arith.muli %m, %N : index
          %co = arith.addi %cr, %n : index
          %ca = pim.access %C offset %co sizes [64, 8]
              strides[%N, %c1] origin[%m, %n] {id = 2 : i64}
              : memref<*xi32> -> !pim.access
          pim.combine %sum into %ca kind "add" : tensor<64x8xi32>, !pim.access
        } {pim.axis = #pim.axis<"n">}
      } {pim.axis = #pim.axis<"m">}
    }
  } {pim.scheduling_axis = "reduction_k"}
  return
}

// CHECK-LABEL: func.func @large_rhs
// CHECK: %[[SET_large_rhs:.*]] = upmem.acquire 2048
// CHECK: %[[MAP_large_rhs:.*]] = upmem.bind_devices %[[SET_large_rhs]] partitions [4, 128, 4] device_mapping [4, 2, 1]
// CHECK: upmem.foreach_unit %[[SET_large_rhs]] assignment %[[MAP_large_rhs]]
// CHECK: upmem.to_mram
// CHECK: upmem.to_host_var
// CHECK: upmem.gather %[[SET_large_rhs]] assignment %[[MAP_large_rhs]]
// CHECK: upmem.foreach_group
// CHECK: accumulate
func.func @large_rhs(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>)
    attributes {pim.block = #pim.block<shape = [8, 64, 8]>,
                pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %bm = arith.constant 8 : index
  %bn = arith.constant 64 : index
  %N = arith.constant 8192 : index
  %zero = arith.constant 0 : i32
  scf.for %round = %c0 to %c1 step %c1 {
    pim.device #pim.device_mapping<partitions = [4, 128, 4]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
    ^bb0(%pm: index, %pn: index, %pk: index):
      %mStart = arith.muli %pm, %bm : index
      %nStart = arith.muli %pn, %bn : index
      %kStart = arith.muli %pk, %c8 : index
      pim.place %A start[%mStart, %kStart] size[%bm, %c8]
          allocation_shape[%bm, %c8] strides[%c32, %c1]
          {direction = "in", access = 0 : i64, shared_over = ["n"]} : memref<*xi8>
      pim.place %B start[%kStart, %nStart] size[%c8, %bn]
          allocation_shape[%c8, %bn] strides[%N, %c1]
          {direction = "in", access = 1 : i64, shared_over = ["m"]} : memref<*xi8>
      pim.place %C start[%mStart, %nStart] size[%bm, %bn]
          allocation_shape[%bm, %bn] strides[%N, %c1]
          {direction = "out", access = 2 : i64} : memref<*xi32>
      scf.for %mt = %c0 to %c1 step %c1 {
        %gm = arith.addi %pm, %mt {pim.device_origin} : index
        %m = arith.muli %gm, %bm : index
        scf.for %nt = %c0 to %c1 step %c1 {
          %gn = arith.addi %pn, %nt {pim.device_origin} : index
          %n = arith.muli %gn, %bn : index
          %empty = tensor.empty() : tensor<8x64xi32>
          %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<8x64xi32>)
              -> tensor<8x64xi32>
          %sum = scf.for %rk = %c0 to %c1 step %c1
              iter_args(%acc = %init) -> tensor<8x64xi32> {
            %gk = arith.addi %pk, %rk {pim.device_origin} : index
            %k = arith.muli %gk, %c8 : index
            %ar = arith.muli %m, %c32 : index
            %ao = arith.addi %ar, %k : index
            %aa = pim.access %A offset %ao sizes [8, 8]
                strides[%c32, %c1] origin[%m, %k] {id = 0 : i64}
                : memref<*xi8> -> !pim.access
            %a = pim.load %aa : !pim.access -> tensor<8x8xi8>
            %br = arith.muli %k, %N : index
            %bo = arith.addi %br, %n : index
            %ba = pim.access %B offset %bo sizes [8, 64]
                strides[%N, %c1] origin[%k, %n] {id = 1 : i64}
                : memref<*xi8> -> !pim.access
            %b = pim.load %ba : !pim.access -> tensor<8x64xi8>
            %next = linalg.matmul ins(%a, %b : tensor<8x8xi8>, tensor<8x64xi8>)
                outs(%acc : tensor<8x64xi32>) -> tensor<8x64xi32>
            scf.yield %next : tensor<8x64xi32>
          } {pim.reduction = "add", pim.temporal_reduction}
          %cr = arith.muli %m, %N : index
          %co = arith.addi %cr, %n : index
          %ca = pim.access %C offset %co sizes [8, 64]
              strides[%N, %c1] origin[%m, %n] {id = 2 : i64}
              : memref<*xi32> -> !pim.access
          pim.combine %sum into %ca kind "add" : tensor<8x64xi32>, !pim.access
        } {pim.axis = #pim.axis<"n">}
      } {pim.axis = #pim.axis<"m">}
    }
  } {pim.scheduling_axis = "reduction_k"}
  return
}

// CHECK-LABEL: func.func @balanced_inputs
// CHECK: %[[SET_balanced_inputs:.*]] = upmem.acquire 2048
// CHECK: %[[MAP_balanced_inputs:.*]] = upmem.bind_devices %[[SET_balanced_inputs]] partitions [4, 128, 4] device_mapping [2, 4, 1]
// CHECK: upmem.foreach_unit %[[SET_balanced_inputs]] assignment %[[MAP_balanced_inputs]]
// CHECK: upmem.to_mram
// CHECK: upmem.to_host_var
// CHECK: upmem.gather %[[SET_balanced_inputs]] assignment %[[MAP_balanced_inputs]]
// CHECK: upmem.foreach_group
// CHECK: accumulate
func.func @balanced_inputs(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>)
    attributes {pim.block = #pim.block<shape = [64, 56, 8]>,
                pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %bm = arith.constant 64 : index
  %bn = arith.constant 56 : index
  %N = arith.constant 7168 : index
  %zero = arith.constant 0 : i32
  scf.for %round = %c0 to %c1 step %c1 {
    pim.device #pim.device_mapping<partitions = [4, 128, 4]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
    ^bb0(%pm: index, %pn: index, %pk: index):
      %mStart = arith.muli %pm, %bm : index
      %nStart = arith.muli %pn, %bn : index
      %kStart = arith.muli %pk, %c8 : index
      pim.place %A start[%mStart, %kStart] size[%bm, %c8]
          allocation_shape[%bm, %c8] strides[%c32, %c1]
          {direction = "in", access = 0 : i64, shared_over = ["n"]} : memref<*xi8>
      pim.place %B start[%kStart, %nStart] size[%c8, %bn]
          allocation_shape[%c8, %bn] strides[%N, %c1]
          {direction = "in", access = 1 : i64, shared_over = ["m"]} : memref<*xi8>
      pim.place %C start[%mStart, %nStart] size[%bm, %bn]
          allocation_shape[%bm, %bn] strides[%N, %c1]
          {direction = "out", access = 2 : i64} : memref<*xi32>
      scf.for %mt = %c0 to %c1 step %c1 {
        %gm = arith.addi %pm, %mt {pim.device_origin} : index
        %m = arith.muli %gm, %bm : index
        scf.for %nt = %c0 to %c1 step %c1 {
          %gn = arith.addi %pn, %nt {pim.device_origin} : index
          %n = arith.muli %gn, %bn : index
          %empty = tensor.empty() : tensor<64x56xi32>
          %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<64x56xi32>)
              -> tensor<64x56xi32>
          %sum = scf.for %rk = %c0 to %c1 step %c1
              iter_args(%acc = %init) -> tensor<64x56xi32> {
            %gk = arith.addi %pk, %rk {pim.device_origin} : index
            %k = arith.muli %gk, %c8 : index
            %ar = arith.muli %m, %c32 : index
            %ao = arith.addi %ar, %k : index
            %aa = pim.access %A offset %ao sizes [64, 8]
                strides[%c32, %c1] origin[%m, %k] {id = 0 : i64}
                : memref<*xi8> -> !pim.access
            %a = pim.load %aa : !pim.access -> tensor<64x8xi8>
            %br = arith.muli %k, %N : index
            %bo = arith.addi %br, %n : index
            %ba = pim.access %B offset %bo sizes [8, 56]
                strides[%N, %c1] origin[%k, %n] {id = 1 : i64}
                : memref<*xi8> -> !pim.access
            %b = pim.load %ba : !pim.access -> tensor<8x56xi8>
            %next = linalg.matmul ins(%a, %b : tensor<64x8xi8>, tensor<8x56xi8>)
                outs(%acc : tensor<64x56xi32>) -> tensor<64x56xi32>
            scf.yield %next : tensor<64x56xi32>
          } {pim.reduction = "add", pim.temporal_reduction}
          %cr = arith.muli %m, %N : index
          %co = arith.addi %cr, %n : index
          %ca = pim.access %C offset %co sizes [64, 56]
              strides[%N, %c1] origin[%m, %n] {id = 2 : i64}
              : memref<*xi32> -> !pim.access
          pim.combine %sum into %ca kind "add" : tensor<64x56xi32>, !pim.access
        } {pim.axis = #pim.axis<"n">}
      } {pim.axis = #pim.axis<"m">}
    }
  } {pim.scheduling_axis = "reduction_k"}
  return
}
}
