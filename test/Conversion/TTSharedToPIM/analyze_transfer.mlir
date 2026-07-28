// 전송 한 번의 크기(pim.tile_bytes)와 유닛 하나가 동시에 들고 있어야 하는 양
// (pim.co_resident)을 확인한다.
// RUN: triton-shared-opt --pim-analyze-reuse --pim-analyze-transfer %s | FileCheck %s

// C[1024,1024] = A[1024,512] x B[512,1024], f32, BLOCK M/N/K = 32/64/16.
// 동시 상주: A tile 2048 + B tile 4096 (매 반복 새로 읽으므로 streaming)
//          + 누산기 8192 (K loop를 관통하므로 persistent).
// CHECK-LABEL: func.func @matmul
// CHECK-SAME:  pim.co_resident = [{access = 0 : i64, bytes = 2048 : i64, lifetime = "streaming"}, {access = 1 : i64, bytes = 4096 : i64, lifetime = "streaming"}, {bytes = 8192 : i64, lifetime = "persistent"}]
// CHECK:       pim.access %arg0 {{.*}} pim.tile_bytes = 2048 : i64
// CHECK:       pim.access %arg1 {{.*}} pim.tile_bytes = 4096 : i64
// CHECK:       pim.access %arg2 {{.*}} pim.tile_bytes = 8192 : i64
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
        %aoff = arith.addi %gm, %k : index
        %dA = pim.access %A offset %aoff sizes [32, 16] strides [%c1, %c1] : memref<*xf32> -> !pim.access
        %a = pim.load %dA : !pim.access -> tensor<32x16xf32>
        %boff = arith.addi %gn, %k : index
        %dB = pim.access %B offset %boff sizes [16, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
        %b = pim.load %dB : !pim.access -> tensor<16x64xf32>
        %next = linalg.matmul ins(%a, %b : tensor<32x16xf32>, tensor<16x64xf32>)
            outs(%acc : tensor<32x64xf32>) -> tensor<32x64xf32>
        scf.yield %next : tensor<32x64xf32>
      }
      %coff = arith.addi %gm, %gn : index
      %dC = pim.access %C offset %coff sizes [32, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
      pim.store %r, %dC : tensor<32x64xf32>, !pim.access
    } {pim.grid_dim = "y"}
  } {pim.grid_dim = "x"}
  return
}

// K loop 밖에서 읽어 안에서 쓰는 tile도 그 loop 내내 자리를 차지한다. 목록에서 빠지면
// Target IR이 용량을 실제보다 작게 보고 들어가지 않는 분할을 고른다.
// --pim-remove-redundant-transfers가 loop 밖으로 뺀 load가 이 형태다.
// CHECK-LABEL: func.func @hoisted
// CHECK-SAME:  pim.co_resident = [{access = 1 : i64, bytes = 4096 : i64, lifetime = "streaming"}, {bytes = 8192 : i64, lifetime = "persistent"}, {access = 0 : i64, bytes = 2048 : i64, lifetime = "persistent"}]
func.func @hoisted(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.0 : f32
  scf.for %gm = %c0 to %c32 step %c1 {
    %empty = tensor.empty() : tensor<32x64xf32>
    %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
    // A는 K에 무관해 K loop 밖에 있다. 그래도 loop가 도는 내내 살아 있어야 한다.
    %dA = pim.access %A offset %gm sizes [32, 16] strides [%c1, %c1] : memref<*xf32> -> !pim.access
    %a = pim.load %dA : !pim.access -> tensor<32x16xf32>
    %r = scf.for %k = %c0 to %c32 step %c1 iter_args(%acc = %init) -> tensor<32x64xf32> {
      %dB = pim.access %B offset %k sizes [16, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
      %b = pim.load %dB : !pim.access -> tensor<16x64xf32>
      %next = linalg.matmul ins(%a, %b : tensor<32x16xf32>, tensor<16x64xf32>)
          outs(%acc : tensor<32x64xf32>) -> tensor<32x64xf32>
      scf.yield %next : tensor<32x64xf32>
    }
    %dC = pim.access %C offset %gm sizes [32, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
    pim.store %r, %dC : tensor<32x64xf32>, !pim.access
  } {pim.grid_dim = "x"}
  return
}

// 8비트 미만 타입은 올림 나눗셈으로 센다. i1 타일 100개면 13 바이트다.
// CHECK-LABEL: func.func @sub_byte
// CHECK-SAME:  pim.co_resident = [{access = 0 : i64, bytes = 13 : i64, lifetime = "streaming"}]
// CHECK:       pim.access %arg0 {{.*}} pim.tile_bytes = 13 : i64
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

// pim.gather는 !pim.access를 거치지 않으므로 tile 바이트가 붙지 않는다. 반면 상주량은
// 결과 tensor 타입에서 나오므로 목록에 든다(access 필드는 없다 — 누산기와 같다).
// index tile 8xi32 = 32 바이트, gather 결과 8xf32 = 32 바이트.
// CHECK-LABEL: func.func @element_gather
// CHECK-SAME:  pim.co_resident = [{access = 0 : i64, bytes = 32 : i64, lifetime = "streaming"}, {bytes = 32 : i64, lifetime = "streaming"}]
// CHECK:       pim.access %arg1 {{.*}} pim.tile_bytes = 32 : i64
// CHECK:       pim.gather
// CHECK-NOT:   pim.tile_bytes
// CHECK:       pim.access %arg2 {{.*}} pim.tile_bytes = 32 : i64
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
