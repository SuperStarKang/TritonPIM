// 분할 비용 식에 들어갈 항을 계산해 속성으로 단다. 항을 곱해 시간으로 바꾸는 것은
// Target IR이 한다.
// grid loop는 상수 범위로 직접 쓴다. --pim-build-grid가 만드는 상한은 num_programs
// 인자라 runtime 값이고, 그 경우는 아래 @runtime_grid에서 따로 확인한다.
// RUN: triton-shared-opt --pim-analyze-reuse --pim-analyze-transfer %s | FileCheck %s

// C[1024,1024] = A[1024,512] x B[512,1024], f32, BLOCK M/N/K = 32/64/16.
// grid는 m=32, n=16이고 K loop는 32번 돈다.
//
//   access | sizes    | tile_bytes | transfer_count | invariant
//   A      | [32, 16] |       2048 |             32 | {n}
//   B      | [16, 64] |       4096 |             32 | {m}
//   C      | [32, 64] |       8192 |              1 | {}
//
// 동시 상주: A tile 2048 + B tile 4096 (매 반복 새로 읽으므로 streaming)
//          + 누산기 8192 (K loop를 관통하므로 persistent).
// 연산량: matmul 반복 공간 32x64x16 = 32768, 원소당 곱셈+덧셈 2개, K loop 32번
//        → 32768 x 2 x 32 = 2097152. (linalg.fill은 산술이 없어 0이다.)
//
// CHECK-LABEL: func.func @matmul
// CHECK-SAME:  pim.co_resident = [{access = 0 : i64, bytes = 2048 : i64, lifetime = "streaming"}, {access = 1 : i64, bytes = 4096 : i64, lifetime = "streaming"}, {bytes = 8192 : i64, lifetime = "persistent"}]
// CHECK-SAME:  pim.compute_ops = 2097152 : i64
// CHECK-SAME:  pim.grid_extents = array<i64: 32, 16>
//
// CHECK:       pim.access %arg0 {{.*}} pim.tile_bytes = 2048 : i64, pim.transfer_count = 32 : i64
// CHECK:       pim.access %arg1 {{.*}} pim.tile_bytes = 4096 : i64, pim.transfer_count = 32 : i64
// CHECK:       pim.access %arg2 {{.*}} pim.tile_bytes = 8192 : i64, pim.transfer_count = 1 : i64
func.func @matmul(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.0 : f32
  scf.for %gm = %c0 to %c32 step %c1 {
    scf.for %gn = %c0 to %c16 step %c1 {
      %empty = tensor.empty() : tensor<32x64xf32>
      %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
      %r = scf.for %k = %c0 to %c32 step %c1 iter_args(%acc = %init) -> tensor<32x64xf32> {
        // A: m과 k에 의존, n에 불변.
        %aoff = arith.addi %gm, %k : index
        %dA = pim.access %A offset %aoff sizes [32, 16] strides [%c1, %c1] : memref<*xf32> -> !pim.access
        %a = pim.load %dA : !pim.access -> tensor<32x16xf32>
        // B: n과 k에 의존, m에 불변.
        %boff = arith.addi %gn, %k : index
        %dB = pim.access %B offset %boff sizes [16, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
        %b = pim.load %dB : !pim.access -> tensor<16x64xf32>
        %next = linalg.matmul ins(%a, %b : tensor<32x16xf32>, tensor<16x64xf32>)
            outs(%acc : tensor<32x64xf32>) -> tensor<32x64xf32>
        scf.yield %next : tensor<32x64xf32>
      }
      // C: 두 축 모두 의존. K loop 밖이라 반복 1번.
      %coff = arith.addi %gm, %gn : index
      %dC = pim.access %C offset %coff sizes [32, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
      pim.store %r, %dC : tensor<32x64xf32>, !pim.access
    } {pim.grid_dim = "y"}
  } {pim.grid_dim = "x"}
  return
}

// 출력 C가 y축에 불변이면 y만 다른 instance들이 같은 C에 쓴다. 그 축을 쪼개면 부분 결과가
// 생겨 유닛 밖 결합이 필요하다. 여기서는 전송 항만 확인하고, 결합 필요 여부는
// analyze_reuse.mlir이 확인한다.
// 연산량: 반복 공간 32x64x16 = 32768, 원소당 2개, K loop 8번 → 524288.
// CHECK-LABEL: func.func @splitk
// CHECK-SAME:  pim.compute_ops = 524288 : i64
// CHECK-SAME:  pim.grid_extents = array<i64: 8, 4>
// CHECK:       pim.access %arg0 {{.*}} pim.tile_bytes = 2048 : i64, pim.transfer_count = 8 : i64
// CHECK:       pim.access %arg1 {{.*}} pim.tile_bytes = 4096 : i64, pim.transfer_count = 8 : i64
// CHECK:       pim.access %arg2 {{.*}} pim.tile_bytes = 8192 : i64, pim.transfer_count = 1 : i64
func.func @splitk(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.0 : f32
  scf.for %gm = %c0 to %c8 step %c1 {
    scf.for %gk = %c0 to %c4 step %c1 {
      %empty = tensor.empty() : tensor<32x64xf32>
      %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
      %r = scf.for %k = %c0 to %c8 step %c1 iter_args(%acc = %init) -> tensor<32x64xf32> {
        %aoff = arith.addi %gm, %gk : index
        %dA = pim.access %A offset %aoff sizes [32, 16] strides [%c1, %c1] : memref<*xf32> -> !pim.access
        %a = pim.load %dA : !pim.access -> tensor<32x16xf32>
        %dB = pim.access %B offset %gk sizes [16, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
        %b = pim.load %dB : !pim.access -> tensor<16x64xf32>
        %next = linalg.matmul ins(%a, %b : tensor<32x16xf32>, tensor<16x64xf32>)
            outs(%acc : tensor<32x64xf32>) -> tensor<32x64xf32>
        scf.yield %next : tensor<32x64xf32>
      }
      %dC = pim.access %C offset %gm sizes [32, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
      pim.store %r, %dC : tensor<32x64xf32>, !pim.access
    } {pim.grid_dim = "y"}
  } {pim.grid_dim = "x"}
  return
}

// grid 크기가 runtime 값이면 pim.grid_extents를 붙이지 않는다. 근거 없는 수를 IR에 넣지
// 않는다. 나머지 항은 grid 크기와 무관하므로 그대로 단다.
// 아래는 func 속성 전체를 그대로 맞춰, grid_extents가 없음을 확인한다.
// CHECK-LABEL: func.func @runtime_grid
// CHECK-SAME:  attributes {pim.co_resident = [{access = 0 : i64, bytes = 1024 : i64, lifetime = "streaming"}], pim.compute_ops = 0 : i64, pim.execution_model = "spmd", pim.grid_dims = ["x"], pim.split_axes = [{axis = "x", source = "grid"}]}
// CHECK:       pim.access %arg0 {{.*}} pim.tile_bytes = 1024 : i64, pim.transfer_count = 1 : i64
func.func @runtime_grid(%A: memref<*xf32>, %C: memref<*xf32>, %np: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %g = %c0 to %np step %c1 {
    %dA = pim.access %A offset %g sizes [256] strides [%c1] : memref<*xf32> -> !pim.access
    %v = pim.load %dA : !pim.access -> tensor<256xf32>
    %dC = pim.access %C offset %g sizes [256] strides [%c1] : memref<*xf32> -> !pim.access
    pim.store %v, %dC : tensor<256xf32>, !pim.access
  } {pim.grid_dim = "x"}
  return
}

// 8비트 미만 타입은 올림 나눗셈으로 센다. i1 타일 100개면 13 바이트다.
// CHECK-LABEL: func.func @sub_byte
// CHECK-SAME:  pim.co_resident = [{access = 0 : i64, bytes = 13 : i64, lifetime = "streaming"}]
// CHECK:       pim.access %arg0 {{.*}} pim.tile_bytes = 13 : i64, pim.transfer_count = 1 : i64
func.func @sub_byte(%M: memref<*xi1>, %C: memref<*xi1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %g = %c0 to %c4 step %c1 {
    %dM = pim.access %M offset %g sizes [100] strides [%c1] : memref<*xi1> -> !pim.access
    %m = pim.load %dM : !pim.access -> tensor<100xi1>
    %dC = pim.access %C offset %g sizes [100] strides [%c1] : memref<*xi1> -> !pim.access
    pim.store %m, %dC : tensor<100xi1>, !pim.access
  } {pim.grid_dim = "x"}
  return
}

// pim.gather는 !pim.access를 거치지 않으므로 전송량 항이 붙지 않는다. 원소별 임의 접근이라
// 전송량이 index 텐서의 값에 달려 컴파일 시점에 셀 수 없기 때문이다.
// 반면 상주량은 결과 tensor 타입에서 나오므로 목록에 든다. gather 결과에는 access 인덱스가
// 없다(누산기와 같다). index tile 8xi32 = 32 바이트, gather 결과 8xf32 = 32 바이트.
// CHECK-LABEL: func.func @element_gather
// CHECK-SAME:  pim.co_resident = [{access = 0 : i64, bytes = 32 : i64, lifetime = "streaming"}, {bytes = 32 : i64, lifetime = "streaming"}]
// CHECK:       pim.access %arg1 {{.*}} pim.tile_bytes = 32 : i64, pim.transfer_count = 1 : i64
// CHECK:       pim.gather
// CHECK-NOT:   pim.tile_bytes
// CHECK:       pim.access %arg2 {{.*}} pim.tile_bytes = 32 : i64, pim.transfer_count = 1 : i64
func.func @element_gather(%A: memref<*xf32>, %I: memref<*xi32>, %C: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %g = %c0 to %c4 step %c1 {
    %dI = pim.access %I offset %g sizes [8] strides [%c1] : memref<*xi32> -> !pim.access
    %idx = pim.load %dI : !pim.access -> tensor<8xi32>
    %v = pim.gather %A index %idx : memref<*xf32>, tensor<8xi32> -> tensor<8xf32>
    %dC = pim.access %C offset %g sizes [8] strides [%c1] : memref<*xf32> -> !pim.access
    pim.store %v, %dC : tensor<8xf32>, !pim.access
  } {pim.grid_dim = "x"}
  return
}
