// Identity-map linalg.generic의 add+sub+mul fusion이 공통 3계층 plan을 유지한 채 UPMEM
// WRAM 계산과 DPU C loop까지 내려가는지 확인한다. lanes=4이므로 block group의 8행을
// tasklet마다 2행씩 맡는다.
// RUN: triton-shared-opt --convert-pim-to-upmem %s | FileCheck %s --check-prefix=IR
// RUN: triton-shared-opt --convert-pim-to-upmem %s | triton-shared-translate --upmem-to-device | FileCheck %s --check-prefix=DEVICE

#map = affine_map<(d0, d1) -> (d0, d1)>

// IR-LABEL: upmem.device_func @elementwise_device
// IR-SAME: upmem.lanes = 4
// IR: upmem.tasklet_id
// 입력 둘과 출력이 모두 같은 m 행 slice를 사용한다.
// IR-COUNT-3: memref.subview {{.*}} [2, 32]
// 분담 DMA가 모두 끝난 뒤 pointwise compute를 시작한다.
// IR: upmem.barrier
// IR: linalg.generic
// IR: arith.addf
// IR: arith.subf
// IR: arith.mulf
// IR-NOT: pim.

// DEVICE-LABEL: int main()
// 2 x 32 원소를 하나의 fused loop에서 계산한다.
// DEVICE: for (int32_t _i = 0; _i < 64; ++_i) {
// DEVICE: float {{.*}} = {{.*}}[_i] + {{.*}}[_i];
// DEVICE: float {{.*}} = {{.*}} - {{.*}}[_i];
// DEVICE: float {{.*}} = {{.*}} * {{.*}}[_i];
// DEVICE: {{.*}}[_i] = {{.*}};

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 1,
    banks_per_unit = 1,
    cores_per_bank = 1,
    lanes_per_core = 4,
    bank_capacity_limit_bytes = 12288,
    local_capacity_limit_bytes = 49152>} {

func.func @elementwise(%A: memref<*xf32>, %B: memref<*xf32>,
                       %C: memref<*xf32>) attributes {
    pim.block = #pim.block<shape = [8, 32, 0]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index

  pim.device #pim.device_mapping<partitions = [1, 1, 1]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 0]>} {
  ^bb0(%pm: index, %pn: index, %device_k_unused: index):
    pim.place %A start[%c0, %c0] size[%c8, %c32] allocation_shape[%c8, %c32]
        strides[%c32, %c1] {direction = "in", access = 0 : i64}
        : memref<*xf32>
    pim.place %B start[%c0, %c0] size[%c8, %c32] allocation_shape[%c8, %c32]
        strides[%c32, %c1] {direction = "in", access = 1 : i64}
        : memref<*xf32>
    pim.place %C start[%c0, %c0] size[%c8, %c32] allocation_shape[%c8, %c32]
        strides[%c32, %c1] {direction = "out", access = 2 : i64}
        : memref<*xf32>

    scf.for %i = %c0 to %c1 step %c1 {
      %m = arith.muli %i, %c8 : index
      scf.for %j = %c0 to %c1 step %c1 {
        %n = arith.muli %j, %c32 : index
        %da = pim.access %A offset %n sizes [8, 32] strides[%c32, %c1]
              origin[%m, %n] {id = 0 : i64}
              : memref<*xf32> -> !pim.access
        %ta = pim.load %da : !pim.access -> tensor<8x32xf32>
        %db = pim.access %B offset %n sizes [8, 32] strides[%c32, %c1]
              origin[%m, %n] {id = 1 : i64}
              : memref<*xf32> -> !pim.access
        %tb = pim.load %db : !pim.access -> tensor<8x32xf32>
        %empty = tensor.empty() : tensor<8x32xf32>
        %result = linalg.generic {
            indexing_maps = [#map, #map, #map],
            iterator_types = ["parallel", "parallel"]}
            ins(%ta, %tb : tensor<8x32xf32>, tensor<8x32xf32>)
            outs(%empty : tensor<8x32xf32>) {
        ^bb0(%a: f32, %b: f32, %old: f32):
          %sum = arith.addf %a, %b : f32
          %difference = arith.subf %sum, %b : f32
          %product = arith.mulf %difference, %b : f32
          linalg.yield %product : f32
        } -> tensor<8x32xf32>
        %dc = pim.access %C offset %n sizes [8, 32] strides[%c32, %c1]
              origin[%m, %n] {id = 2 : i64}
              : memref<*xf32> -> !pim.access
        pim.store %result, %dc : tensor<8x32xf32>, !pim.access
      } {pim.axis = #pim.axis<"n">}
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
