// Triton Shared IR의 메모리 관용구부터 시작해 PIM Common의 세 계층을 계획하고,
// identity pointwise add+mul을 UPMEM DPU C까지 내리는 전체 경로를 확인한다.
// RUN: triton-shared-opt --pim-normalize-address -cse -loop-invariant-code-motion -canonicalize --pim-form-compute --convert-ttshared-to-pim --pim-build-logical-grid --pim-recover-access-origin "--pim-set-target-profile=target=upmem num-units=1 banks-per-unit=1 cores-per-bank=1 lanes-per-core=4 bank-capacity-limit-bytes=62914560 local-capacity-limit-bytes=49152" "--pim-form-block-groups=group-m=1 group-n=1 group-k=1" --pim-form-execution-tiles --pim-map-devices --pim-place-data --pim-verify -cse -canonicalize --pim-form-host-plan "--convert-pim-to-upmem=nr-tasklets=4" %s | triton-shared-translate --upmem-to-device | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: // pim.lanes = 4
// CHECK-LABEL: int main()
// block 8행을 4 tasklet이 나눠 각자 2 x 32 원소를 한 loop에서 계산한다.
// CHECK: for (int32_t _i = 0; _i < 64; ++_i) {
// CHECK: float {{.*}} = {{.*}}[_i] + {{.*}}[_i];
// CHECK: float {{.*}} = {{.*}} * {{.*}}[_i];
// CHECK: {{.*}}[_i] = {{.*}};

func.func @elementwise_pipeline(
    %A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
    %num_m: i32, %num_n: i32, %num_unused: i32,
    %pid_m: i32, %pid_n: i32, %pid_unused: i32) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index

  %pm = arith.index_cast %pid_m : i32 to index
  %pn = arith.index_cast %pid_n : i32 to index
  %row = arith.muli %pm, %c8 : index
  %col = arith.muli %pn, %c32 : index
  %row_offset = arith.muli %row, %c128 : index
  %offset = arith.addi %row_offset, %col : index

  %ra = memref.reinterpret_cast %A to offset: [%offset], sizes: [8, 32],
      strides: [%c128, 1]
      : memref<*xf32> to memref<8x32xf32, strided<[?, 1], offset: ?>>
  %wa = memref.alloc() : memref<8x32xf32>
  memref.copy %ra, %wa
      : memref<8x32xf32, strided<[?, 1], offset: ?>> to memref<8x32xf32>
  %ta = bufferization.to_tensor %wa restrict writable
      : memref<8x32xf32> to tensor<8x32xf32>

  %rb = memref.reinterpret_cast %B to offset: [%offset], sizes: [8, 32],
      strides: [%c128, 1]
      : memref<*xf32> to memref<8x32xf32, strided<[?, 1], offset: ?>>
  %wb = memref.alloc() : memref<8x32xf32>
  memref.copy %rb, %wb
      : memref<8x32xf32, strided<[?, 1], offset: ?>> to memref<8x32xf32>
  %tb = bufferization.to_tensor %wb restrict writable
      : memref<8x32xf32> to tensor<8x32xf32>

  %empty = tensor.empty() : tensor<8x32xf32>
  %result = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%ta, %tb : tensor<8x32xf32>, tensor<8x32xf32>)
      outs(%empty : tensor<8x32xf32>) {
  ^bb0(%a: f32, %b: f32, %old: f32):
    %sum = arith.addf %a, %b : f32
    %product = arith.mulf %sum, %b : f32
    linalg.yield %product : f32
  } -> tensor<8x32xf32>

  %rc = memref.reinterpret_cast %C to offset: [%offset], sizes: [8, 32],
      strides: [%c128, 1]
      : memref<*xf32> to memref<8x32xf32, strided<[?, 1], offset: ?>>
  bufferization.materialize_in_destination %result in writable %rc
      : (tensor<8x32xf32>, memref<8x32xf32, strided<[?, 1], offset: ?>>) -> ()
  return
}
