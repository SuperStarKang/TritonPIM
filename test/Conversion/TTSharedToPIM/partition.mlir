// 분할을 고르기 위한 수치를 계산해 속성으로 단다. 고르는 것은 Target IR이 한다.
// 전송량 식이 pim.invariant_program_dims를 입력으로 쓰므로 analyze를 먼저 돌린다.
// grid loop는 상수 범위로 직접 쓴다. --pim-build-grid가 만드는 상한은 num_programs
// 인자라 runtime 값이고, 그 경우는 아래 @runtime_grid에서 따로 확인한다.
// RUN: triton-shared-opt --pim-analyze-logical-work --pim-partition %s | FileCheck %s

// C[1024,1024] = A[1024,512] x B[512,1024], f32, BLOCK M/N/K = 32/64/16.
// grid는 m=32, n=16이고 K loop는 32번 돈다.
//
//   access | sizes    | tile |  반복 | access_bytes | invariant
//   A      | [32, 16] | 2 KB |    32 |        64 KB | {n}
//   B      | [16, 64] | 4 KB |    32 |       128 KB | {m}
//   C      | [32, 64] | 8 KB |     1 |         8 KB | {}
//
// working set = A tile 2 KB + B tile 4 KB + 누산기 8 KB = 14 KB.
//
// CHECK-LABEL: func.func @matmul
// CHECK-SAME:  pim.grid_extents = array<i64: 32, 16>
// CHECK-SAME:  pim.grid_permutable = true
// 후보는 각 축의 약수를 모두 조합한 것이라 32의 약수 6개 x 16의 약수 5개 = 30개다.
// s가 전부 1이면 유닛이 하나라는 뜻이고, 그때 값이 전송량 하한(실제로 필요한 데이터양)이다.
// CHECK-SAME:  bytes = 8388608 : i64, splits = array<i64: 1, 1>
// n축만 쪼갠다: 유닛 16개, A를 16번 중복 전송.
// CHECK-SAME:  bytes = 39845888 : i64, splits = array<i64: 1, 16>
// m축만 쪼갠다: 유닛 32개, B를 32번 중복 전송.
// CHECK-SAME:  bytes = 73400320 : i64, splits = array<i64: 32, 1>
// 둘 다 쪼갠다: 유닛 512개, 필요량의 12배를 전송.
// CHECK-SAME:  bytes = 104857600 : i64, splits = array<i64: 32, 16>
// CHECK-SAME:  pim.working_set_bytes = 14336 : i64
//
// CHECK:       pim.access %arg0 {{.*}} pim.access_bytes = 65536 : i64
// CHECK:       pim.access %arg1 {{.*}} pim.access_bytes = 131072 : i64
// CHECK:       pim.access %arg2 {{.*}} pim.access_bytes = 8192 : i64
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

// 출력 C가 y축에 불변이면 y만 다른 instance들이 같은 C에 쓴다. 순서를 바꾸면 결과가
// 달라지므로 grid_permutable이 거짓이다.
// CHECK-LABEL: func.func @splitk
// CHECK-SAME:  pim.grid_extents = array<i64: 8, 4>
// CHECK-SAME:  pim.grid_permutable = false
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

// grid 크기가 runtime 값이면 표를 만들지 않는다. 근거 없는 수를 IR에 넣지 않는다.
// access_bytes와 working_set_bytes는 grid와 무관하므로 그대로 단다.
// 아래는 func 속성 전체를 그대로 맞춰, grid_extents와 partition_cost가 없음을 확인한다.
// CHECK-LABEL: func.func @runtime_grid
// CHECK-SAME:  attributes {pim.execution_model = "spmd", pim.grid_permutable = true, pim.program_dims = ["x"], pim.working_set_bytes = 1024 : i64}
// CHECK:       pim.access %arg0 {{.*}} pim.access_bytes = 1024 : i64
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
// CHECK-SAME:  pim.working_set_bytes = 13 : i64
// CHECK:       pim.access %arg0 {{.*}} pim.access_bytes = 13 : i64
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
