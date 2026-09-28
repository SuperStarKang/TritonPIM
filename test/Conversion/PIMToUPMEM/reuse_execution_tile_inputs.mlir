// Common hoists A, keeps B per Group, and UPMEM carries the hoisted input into
// device code even when it is outside the original outermost compute loop.
// RUN: triton-shared-opt --pim-reuse-execution-tile-inputs %s | FileCheck %s --check-prefix=COMMON
// RUN: triton-shared-opt --pim-reuse-execution-tile-inputs --pim-form-host-plan %s | FileCheck %s --check-prefix=PLAN
// RUN: triton-shared-opt --pim-reuse-execution-tile-inputs --pim-form-host-plan --convert-pim-to-upmem %s | FileCheck %s --check-prefix=UPMEM
// RUN: triton-shared-opt --pim-reuse-execution-tile-inputs --pim-form-host-plan --convert-pim-to-upmem %s | triton-shared-translate --upmem-to-device | FileCheck %s --check-prefix=C
// RUN: triton-shared-opt --pim-reuse-execution-tile-inputs --pim-form-host-plan --convert-pim-to-upmem %s | triton-shared-translate --upmem-to-host > /dev/null

// PLAN-LABEL: func.func @reuse_n
// PLAN: pim.partition
// PLAN: pim.scatter 0
// PLAN-NEXT: pim.scatter 1
// PLAN-NEXT: pim.launch {
// PLAN: %[[AA:.*]] = pim.access {{.*}}id = 0
// PLAN: %[[A:.*]] = pim.load %[[AA]]
// PLAN: scf.for
// PLAN: linalg.matmul ins(%[[A]],
// PLAN: pim.gather 2

// COMMON-LABEL: func.func @reuse_n
// COMMON: pim.execution_tile = #pim.execution_tile<groups = [1, 4, 1]>
// COMMON: %[[AA:.*]] = pim.access {{.*}}id = 0
// COMMON: %[[A:.*]] = pim.load %[[AA]]
// COMMON: scf.for
// COMMON-NOT: pim.load %[[AA]]
// COMMON: %[[BB:.*]] = pim.access {{.*}}id = 1
// COMMON: %[[B:.*]] = pim.load %[[BB]]
// COMMON: linalg.matmul ins(%[[A]], %[[B]]
// COMMON: pim.store

// UPMEM-LABEL: upmem.device_func @reuse_n_device
// UPMEM: %[[N:.*]] = upmem.get_host_var @reuse_n_tiles_n
// UPMEM: upmem.mram_read @reuse_n_mram_0
// UPMEM: scf.for {{.*}} to %[[N]]
// UPMEM-NOT: upmem.mram_read @reuse_n_mram_0
// UPMEM: upmem.mram_read @reuse_n_mram_1
// UPMEM: upmem.barrier
// UPMEM: linalg.matmul
// UPMEM: upmem.barrier
// UPMEM: upmem.mram_write
// UPMEM: upmem.barrier

// C: int main()
// C: mram_read(
// C: mram_read(
// C: barrier_wait(&pim_barrier)
// C: mram_write(

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 1, banks_per_unit = 1,
    cores_per_bank = 1, lanes_per_core = 2,
    bank_capacity_limit_bytes = 8192, local_capacity_limit_bytes = 49152>} {
  func.func @reuse_n(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>)
      attributes {pim.block = #pim.block<shape = [8, 8, 8]>,
                  pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %zero = arith.constant 0 : i32
    pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 4, 1]>} {
    ^bb0(%dm: index, %dn: index, %dk: index):
      %start = arith.muli %dn, %c4 : index
      %startN = arith.muli %start, %c8 : index
      pim.place %A start[%c0, %c0] size[%c8, %c8]
          allocation_shape[%c8, %c8] strides[%c8, %c1]
          {direction = "in", access = 0 : i64} : memref<*xi8>
      pim.place %B start[%c0, %startN] size[%c8, %c32]
          allocation_shape[%c8, %c32] strides[%c32, %c1]
          {direction = "in", access = 1 : i64} : memref<*xi8>
      pim.place %C start[%c0, %startN] size[%c8, %c32]
          allocation_shape[%c8, %c32] strides[%c32, %c1]
          {direction = "out", access = 2 : i64} : memref<*xi32>
      scf.for %ng = %c0 to %c4 step %c1 {
        %globalN = arith.addi %start, %ng {pim.device_origin} : index
        %n = arith.muli %globalN, %c8 : index
        %aa = pim.access %A offset %c0 sizes [8, 8]
            strides[%c8, %c1] origin[%c0, %c0]
            {id = 0 : i64} : memref<*xi8> -> !pim.access
        %a = pim.load %aa : !pim.access -> tensor<8x8xi8>
        %bb = pim.access %B offset %n sizes [8, 8]
            strides[%c32, %c1] origin[%c0, %n]
            {id = 1 : i64} : memref<*xi8> -> !pim.access
        %b = pim.load %bb : !pim.access -> tensor<8x8xi8>
        %empty = tensor.empty() : tensor<8x8xi32>
        %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<8x8xi32>) -> tensor<8x8xi32>
        %c = linalg.matmul ins(%a, %b : tensor<8x8xi8>, tensor<8x8xi8>)
            outs(%init : tensor<8x8xi32>) -> tensor<8x8xi32>
        %cc = pim.access %C offset %n sizes [8, 8]
            strides[%c32, %c1] origin[%c0, %n]
            {id = 2 : i64} : memref<*xi32> -> !pim.access
        pim.store %c, %cc : tensor<8x8xi32>, !pim.access
      } {pim.axis = #pim.axis<"n">}
    }
    return
  }
}
