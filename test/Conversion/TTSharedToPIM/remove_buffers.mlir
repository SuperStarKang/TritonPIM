// Stage B(중복 전송 제거) 검증. 입력은 -cse가 끝난 형태(같은 영역은 하나의 pim.access 공유)로 두고,
// --pim-remove-buffers로 재작성한 뒤 -canonicalize로 죽은 load를 정리한다.
// RUN: triton-shared-opt --split-input-file --pim-remove-buffers -canonicalize %s | FileCheck %s

// ── passthrough: store(load(%ds), %dd) → pim.copy ──────────────────────────────
// CHECK-LABEL: @passthrough
// CHECK-NOT:   pim.load
// CHECK:       pim.copy %{{.*}}, %{{.*}} : !pim.access, !pim.access
// CHECK-NOT:   pim.store
func.func @passthrough(%A: memref<*xf32>, %C: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ds = pim.access %A offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  %v = pim.load %ds : !pim.access -> tensor<32xf32>
  %dd = pim.access %C offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  pim.store %v, %dd : tensor<32xf32>, !pim.access
  return
}

// -----

// ── 반복 load: 뒤 load가 앞 load 결과로 대체(→ 하나만 남음) ─────────────────────
// CHECK-LABEL: @dedup
// CHECK:       %[[A:.*]] = pim.load
// CHECK-NOT:   pim.load
// CHECK:       return %[[A]], %[[A]]
func.func @dedup(%A: memref<*xf32>) -> (tensor<32xf32>, tensor<32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d = pim.access %A offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  %a = pim.load %d : !pim.access -> tensor<32xf32>
  %b = pim.load %d : !pim.access -> tensor<32xf32>
  return %a, %b : tensor<32xf32>, tensor<32xf32>
}

// -----

// ── RAW 왕복: store 뒤 같은 영역 load → 저장한 값 전달(load 소멸) ────────────────
// CHECK-LABEL: @forward
// CHECK:       pim.store %arg1
// CHECK-NOT:   pim.load
// CHECK:       return %arg1
func.func @forward(%A: memref<*xf32>, %v: tensor<32xf32>) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d = pim.access %A offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  pim.store %v, %d : tensor<32xf32>, !pim.access
  %w = pim.load %d : !pim.access -> tensor<32xf32>
  return %w : tensor<32xf32>
}

// -----

// ── 장벽: 두 load 사이에 store가 있으면 재사용 안 함(두 load 모두 남음) ──────────
// CHECK-LABEL: @blocked
// CHECK:       pim.load
// CHECK:       pim.store
// CHECK:       pim.load
func.func @blocked(%A: memref<*xf32>, %C: memref<*xf32>, %v: tensor<32xf32>)
    -> (tensor<32xf32>, tensor<32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d  = pim.access %A offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  %dc = pim.access %C offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  %a = pim.load %d : !pim.access -> tensor<32xf32>
  pim.store %v, %dc : tensor<32xf32>, !pim.access
  %b = pim.load %d : !pim.access -> tensor<32xf32>
  return %a, %b : tensor<32xf32>, tensor<32xf32>
}
