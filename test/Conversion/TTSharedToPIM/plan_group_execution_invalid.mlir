// 모르는 lane-mode 는 기본값으로 때우지 않고 진단이다.
// RUN: triton-shared-opt --pim-plan-group-execution=lane-mode=diagonal -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", number_of_pim_devices = 8,
    number_of_parallel_lanes_per_pim_core = 16,
    parallel_execution_kind = "multithread",
    usable_bank_capacity_bytes = 3072, usable_local_capacity_bytes = 4096>} {

// expected-error @+1 {{unknown lane-mode "diagonal"}}
func.func @matmul(%A: memref<*xi8>, %lda: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %da = pim.access %A offset %c0 sizes [8, 32] strides[%lda, %c1] origin[%c0, %c0]
        {id = 0 : i64} : memref<*xi8> -> !pim.access
  %ta = pim.load %da : !pim.access -> tensor<8x32xi8>
  return
}
}
