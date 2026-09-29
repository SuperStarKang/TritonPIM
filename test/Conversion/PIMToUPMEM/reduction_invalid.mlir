// K-only reduction에는 아직 tasklet-local accumulator와 DPU 내부 combine이 없으므로
// 여러 tasklet이 하나의 WRAM output을 동시에 갱신하게 두지 않는다.
// RUN: triton-shared-opt --pim-form-host-plan "--convert-pim-to-upmem=nr-tasklets=2" -verify-diagnostics %s

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 1,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 16,
    bank_capacity_limit_bytes = 1024, local_capacity_limit_bytes = 49152>} {

func.func @k_only_two_tasklets()
    attributes {
      pim.block = #pim.block<shape = [1, 1, 4]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  // expected-error @below {{convert-pim-to-upmem: K-only linalg.reduce requires nr-tasklets=1 until tasklet-local reduction combine is represented}}
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> {
  ^bb0(%device_m: index, %device_n: index, %device_k: index):
    %input = tensor.empty() : tensor<2x4xi32>
    %output = tensor.empty() : tensor<2xi32>
    %init = linalg.fill ins(%zero : i32) outs(%output : tensor<2xi32>)
        -> tensor<2xi32>
    %reduced = linalg.reduce ins(%input : tensor<2x4xi32>)
        outs(%init : tensor<2xi32>) dimensions = [1]
        (%value: i32, %old: i32) {
      %sum = arith.addi %value, %old : i32
      linalg.yield %sum : i32
    }
    scf.for %i = %c0 to %c1 step %c1 {
      scf.yield
    }
  }
  return
}
}
