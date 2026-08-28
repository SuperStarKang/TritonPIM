// linalg.matvec의 A/y는 m 행을 tasklet에 나누고 x는 전체 K vector를
// 공유한다. 공유 DMA 전·후 barrier와 DPU C의 MxK 누산 loop까지 확인한다.
// RUN: triton-shared-opt --convert-pim-to-upmem %s | FileCheck %s --check-prefix=IR
// RUN: triton-shared-opt --convert-pim-to-upmem %s | triton-shared-translate --upmem-to-device | FileCheck %s --check-prefix=DEVICE

// IR-LABEL: upmem.device_func @matvec_device
// IR-SAME: upmem.lanes = 4
// IR: %[[Y:.*]] = memref.subview {{.*}} [2]
// IR: %[[A:.*]] = memref.subview {{.*}} [2, 16]
// x는 m 행 slice를 만들지 않고 전체 memref<16xf32>로 남는다.
// IR: upmem.barrier
// IR: linalg.matvec ins(%[[A]], %{{.*}} : {{.*}}, memref<16xf32, #upmem<space wram>>) outs(%[[Y]]
// IR: upmem.barrier
// IR-NOT: pim.

// DEVICE-LABEL: int main()
// tasklet당 8 / 4 = 2개의 출력 행을 계산한다.
// DEVICE: for (int32_t _m = 0; _m < 2; ++_m) {
// DEVICE: float _s = {{.*}}[_m];
// DEVICE: #pragma unroll
// DEVICE: for (int32_t _k = 0; _k < 16; ++_k)
// DEVICE: _s += (float){{.*}}[_m * 16 + _k] * (float){{.*}}[_k];
// DEVICE: {{.*}}[_m] = _s;

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 1,
    banks_per_unit = 1,
    cores_per_bank = 1,
    lanes_per_core = 4,
    bank_capacity_limit_bytes = 12288,
    local_capacity_limit_bytes = 49152>} {

func.func @matvec(%A: memref<*xf32>, %x: memref<*xf32>, %y: memref<*xf32>)
    attributes {
      pim.block = #pim.block<shape = [8, 0, 16]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f32

  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 0, 1]>} {
  ^bb0(%pm: index, %device_n_unused: index, %device_k_unused: index):
    pim.place %A start[%c0, %c0] size[%c8, %c16] allocation_shape[%c8, %c16]
        strides[%c16, %c1] {direction = "in", access = 0 : i64}
        : memref<*xf32>
    pim.place %x start[%c0] size[%c16] allocation_shape[%c16] strides[%c1]
        {direction = "in", access = 1 : i64,
         replicated_over = ["m"]} : memref<*xf32>
    pim.place %y start[%c0] size[%c8] allocation_shape[%c8] strides[%c1]
        {direction = "out", access = 2 : i64}
        : memref<*xf32>

    scf.for %mTile = %c0 to %c1 step %c1 {
      %m = arith.muli %mTile, %c8 : index
      %empty = tensor.empty() : tensor<8xf32>
      %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<8xf32>)
          -> tensor<8xf32>
      %result = scf.for %kTile = %c0 to %c1 step %c1
          iter_args(%acc = %init) -> tensor<8xf32> {
        %k = arith.muli %kTile, %c16 : index
        %aAccess = pim.access %A offset %m sizes [8, 16]
            strides[%c16, %c1] origin[%m, %k] {id = 0 : i64}
            : memref<*xf32> -> !pim.access
        %aTile = pim.load %aAccess : !pim.access -> tensor<8x16xf32>
        %xAccess = pim.access %x offset %k sizes [16] strides[%c1]
            origin[%k] {id = 1 : i64} : memref<*xf32> -> !pim.access
        %xTile = pim.load %xAccess : !pim.access -> tensor<16xf32>
        %partial = linalg.matvec
            ins(%aTile, %xTile : tensor<8x16xf32>, tensor<16xf32>)
            outs(%acc : tensor<8xf32>) -> tensor<8xf32>
        scf.yield %partial : tensor<8xf32>
      }
      %yAccess = pim.access %y offset %m sizes [8] strides[%c1]
          origin[%m] {id = 2 : i64} : memref<*xf32> -> !pim.access
      pim.store %result, %yAccess : tensor<8xf32>, !pim.access
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
