// M/N축이 없는 Common K-only reduction의 init, temporal loop, store가 한 DPU
// program으로 이동하고 linalg.reduce의 명시적 의미가 보존되는지 확인한다.
// 출력은 UPMEM DMA 최소 단위인 8 byte가 되도록 두 i32 원소인 예시를 쓴다.
// RUN: triton-shared-opt --pim-form-host-plan "--convert-pim-to-upmem=nr-tasklets=1" %s | FileCheck %s
// RUN: triton-shared-opt --pim-form-host-plan "--convert-pim-to-upmem=nr-tasklets=1" %s | triton-shared-translate --upmem-to-device | FileCheck %s --check-prefix=DEVICE

// CHECK-LABEL: upmem.device_func @k_only_sum_device
// CHECK-SAME: upmem.lanes = 1
// init도 temporal loop와 같은 device_func 안에 있다.
// CHECK: %[[OUT:.*]] = memref.alloc() : memref<2xi32, #upmem<space wram>>
// CHECK: linalg.fill
// CHECK: scf.for
// CHECK: upmem.mram_read
// CHECK: linalg.reduce
// CHECK-SAME: ins(%{{.*}} : memref<2x4xi32, #upmem<space wram>>)
// CHECK-SAME: outs(%[[OUT]] : memref<2xi32, #upmem<space wram>>)
// store도 temporal loop의 결과와 함께 device_func 안에 있다.
// CHECK: upmem.mram_write
// CHECK-NOT: iter_args
// CHECK-NOT: pim.

// DEVICE: for (int32_t _r0 = 0; _r0 < 2; ++_r0) {
// DEVICE: for (int32_t _r1 = 0; _r1 < 4; ++_r1) {
// DEVICE: int32_t {{.*}} = w{{[0-9]+}}[(_r0 * 4 + _r1)] + w{{[0-9]+}}[_r0];
// DEVICE: w{{[0-9]+}}[_r0] = {{.*}};

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 49152>} {

func.func @k_only_sum(%A: memref<*xi32>, %C: memref<*xi32>)
    attributes {
      pim.block = #pim.block<shape = [1, 1, 4]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
      pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0 : i32

  scf.for %round = %c0 to %c1 step %c1 {
    pim.device #pim.device_mapping<partitions = [1, 1, 1]>
        attributes {
          pim.execution_tile = #pim.execution_tile<groups = [1, 1, 2]>} {
    ^bb0(%device_m: index, %device_n: index, %device_k: index):
      pim.place %A start[%c0, %c0] size[%c2, %c8]
          allocation_shape[%c2, %c8] strides[%c8, %c1]
          {direction = "in", access = 0 : i64} : memref<*xi32>
      pim.place %C start[%c0] size[%c2] allocation_shape[%c2]
          strides[%c1]
          {direction = "out", access = 1 : i64} : memref<*xi32>

      %empty = tensor.empty() : tensor<2xi32>
      %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<2xi32>)
          -> tensor<2xi32>
      %result = scf.for %tk = %c0 to %c2 step %c1
          iter_args(%acc = %init) -> tensor<2xi32> {
        %localK = arith.muli %tk, %c4 : index
        %globalK = arith.addi %c0, %localK {pim.device_origin} : index
        %aAccess = pim.access %A offset %globalK sizes [2, 4]
            strides[%c8, %c1] origin[%c0, %globalK]
            {id = 0 : i64} : memref<*xi32> -> !pim.access
        %tile = pim.load %aAccess : !pim.access -> tensor<2x4xi32>
        %next = linalg.reduce ins(%tile : tensor<2x4xi32>)
            outs(%acc : tensor<2xi32>) dimensions = [1]
            (%value: i32, %old: i32) {
          %sum = arith.addi %value, %old : i32
          linalg.yield %sum : i32
        }
        scf.yield %next : tensor<2xi32>
      } {pim.reduction = "add", pim.temporal_reduction}
      %cAccess = pim.access %C offset %c0 sizes [2] strides[%c1]
          origin[%c0] {id = 1 : i64} : memref<*xi32> -> !pim.access
      pim.combine %result into %cAccess kind "add"
          : tensor<2xi32>, !pim.access
    }
  } {pim.scheduling_axis = "reduction_k"}
  return
}
}
