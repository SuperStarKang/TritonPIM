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

// -----

// ── loop-invariant load hoisting ──────────────────────────────────────────────
// 입력의 pim.access는 이미 loop 밖에 있다(파이프라인에서 -loop-invariant-code-motion이
// Pure한 access를 먼저 빼 준다). 여기서는 load만 따라 나온다.

// %d는 %i에 의존하지 않고, loop 안의 쓰기는 C(다른 버퍼)뿐 → A의 load를 밖으로 뺀다.
// 4번 읽던 것이 1번이 된다.
// CHECK-LABEL: @hoist
// CHECK:       pim.load
// CHECK:       scf.for
// CHECK-NOT:   pim.load
func.func @hoist(%A: memref<*xf32>, %C: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %d = pim.access %A offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  scf.for %i = %c0 to %c4 step %c1 {
    %v = pim.load %d : !pim.access -> tensor<32xf32>
    %w = arith.addf %v, %v : tensor<32xf32>
    %dc = pim.access %C offset %i sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
    pim.store %w, %dc : tensor<32xf32>, !pim.access
  }
  return
}

// -----

// loop 안에서 같은 버퍼 A에 쓴다 → 읽는 영역이 매 반복 달라질 수 있으므로 빼지 않는다.
// (영역이 아니라 base memref 단위로 본다.)
// CHECK-LABEL: @hoist_blocked
// CHECK:       scf.for
// CHECK:         pim.load
func.func @hoist_blocked(%A: memref<*xf32>, %C: memref<*xf32>, %v: tensor<32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %d  = pim.access %A offset %c0  sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  %da = pim.access %A offset %c32 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  scf.for %i = %c0 to %c4 step %c1 {
    %r = pim.load %d : !pim.access -> tensor<32xf32>
    pim.store %v, %da : tensor<32xf32>, !pim.access
    %dc = pim.access %C offset %i sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
    pim.store %r, %dc : tensor<32xf32>, !pim.access
  }
  return
}

// -----

// grid loop에서는 빼지 않는다. 여기서 빼면 grid 전체가 한 번만 읽는 모양이 되어,
// 유닛마다 사본이 필요하다는 사실이 사라진다. 그 결정은 분할 구조가 정해진 뒤에 한다.
// CHECK-LABEL: @hoist_skips_grid
// CHECK:       scf.for
// CHECK:         pim.load
func.func @hoist_skips_grid(%A: memref<*xf32>, %C: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %d = pim.access %A offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  scf.for %g = %c0 to %c4 step %c1 {
    %v = pim.load %d : !pim.access -> tensor<32xf32>
    %w = arith.addf %v, %v : tensor<32xf32>
    %dc = pim.access %C offset %g sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
    pim.store %w, %dc : tensor<32xf32>, !pim.access
  } {pim.grid_dim = "x"}
  return
}

// -----

// 안쪽 loop에서 빠져나온 load가 바깥 loop의 직속 자식이 되어 한 번 더 올라간다.
// CHECK-LABEL: @hoist_nested
// CHECK:       pim.load
// CHECK:       scf.for
// CHECK:       scf.for
// CHECK-NOT:   pim.load
func.func @hoist_nested(%A: memref<*xf32>, %C: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %d = pim.access %A offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %v = pim.load %d : !pim.access -> tensor<32xf32>
      %w = arith.addf %v, %v : tensor<32xf32>
      %dc = pim.access %C offset %j sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
      pim.store %w, %dc : tensor<32xf32>, !pim.access
    }
  }
  return
}

// -----

// scf.if 안의 load는 조건이 참일 때만 실행된다. 밖으로 빼면 원래 하지 않았을 읽기를
// 하게 되므로 옮기지 않는다.
// CHECK-LABEL: @hoist_skips_if
// CHECK:       scf.for
// CHECK:         scf.if
// CHECK:           pim.load
func.func @hoist_skips_if(%A: memref<*xf32>, %C: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %d = pim.access %A offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  scf.for %i = %c0 to %c4 step %c1 {
    %cond = arith.cmpi slt, %i, %c2 : index
    %v = scf.if %cond -> tensor<32xf32> {
      %l = pim.load %d : !pim.access -> tensor<32xf32>
      scf.yield %l : tensor<32xf32>
    } else {
      %z = tensor.empty() : tensor<32xf32>
      scf.yield %z : tensor<32xf32>
    }
    %dc = pim.access %C offset %i sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
    pim.store %v, %dc : tensor<32xf32>, !pim.access
  }
  return
}
