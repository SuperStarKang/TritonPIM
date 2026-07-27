// grid 축과 각 pim.access의 의존 관계를 분석해 func엔 execution_model/grid_dims/split_axes,
// access엔 role/invariant_grid_dims/reuse_realizable을 단다.
// grid 차원은 pim.grid_dim이 붙은 scf.for로 식별하므로 --pim-build-grid를 먼저 돌린다.
// RUN: triton-shared-opt --pim-build-grid --pim-analyze-reuse %s | FileCheck %s

// GEMM: A는 pid0(m)만, B는 pid1(n)만, C는 둘 다에 의존.
// → grid_dims=["x","y"]. A는 n(=dim1)에 불변, B는 m(=dim0)에 불변, C는 불변 없음.
// CHECK-LABEL: func.func @matmul
// CHECK-SAME:  pim.execution_model = "spmd"
// CHECK-SAME:  pim.grid_dims = ["x", "y"]
func.func @matmul(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
                  %np0: i32, %np1: i32, %np2: i32,
                  %pid0: i32, %pid1: i32, %pid2: i32) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %m = arith.index_cast %pid0 : i32 to index
  %n = arith.index_cast %pid1 : i32 to index
  %aoff = arith.muli %m, %c32 : index
  %boff = arith.muli %n, %c32 : index
  %coff = arith.addi %aoff, %boff : index

  // A: m 방향으로만 움직임 → n(dim1) 불변, 재사용은 n방향 program 사이.
  // CHECK: pim.access %arg0 {{.*}} {pim.invariant_grid_dims = array<i64: 1>, pim.reuse_realizable = true, pim.role = "input"}
  %dA = pim.access %A offset %aoff sizes [32, 256] strides [%c1, %c1] : memref<*xf32> -> !pim.access
  %a = pim.load %dA : !pim.access -> tensor<32x256xf32>

  // B: n 방향으로만 움직임 → m(dim0) 불변.
  // CHECK: pim.access %arg1 {{.*}} {pim.invariant_grid_dims = array<i64: 0>, pim.reuse_realizable = true, pim.role = "input"}
  %dB = pim.access %B offset %boff sizes [256, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
  %b = pim.load %dB : !pim.access -> tensor<256x64xf32>

  // C: 두 축 모두 의존 → 불변 없음, output role.
  // CHECK: pim.access %arg2 {{.*}} {pim.invariant_grid_dims = array<i64>, pim.reuse_realizable = true, pim.role = "output"}
  %z = tensor.empty() : tensor<32x64xf32>
  %dC = pim.access %C offset %coff sizes [32, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
  pim.store %z, %dC : tensor<32x64xf32>, !pim.access
  return
}

// loop-carried offset(iter_arg)은 program ID에 의존하지 않는다 → execution_model만 달고
// grid_dims/invariant_grid_dims/split_axes는 달지 않는다.
// CHECK-LABEL: func.func @advance
// CHECK-SAME:  pim.execution_model = "spmd"
// CHECK-NOT:   pim.grid_dims
// CHECK-NOT:   pim.invariant_grid_dims
func.func @advance(%A: memref<*xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c256 = arith.constant 256 : index
  // iter_arg 가 tensor 누산기가 아니라 index 라서 reduction loop 가 아니다.
  // CHECK-NOT:   pim.reduction_combine
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%off = %c0) -> (index) {
    %d = pim.access %A offset %off sizes [256] strides [%c1] : memref<*xf32> -> !pim.access
    %v = pim.load %d : !pim.access -> tensor<256xf32>
    pim.store %v, %d : tensor<256xf32>, !pim.access
    %next = arith.addi %off, %c256 : index
    scf.yield %next : index
  }
  return
}

// reduction loop + 결합이 필요한 grid 축.
// K 루프에서 matmul 이 누산기 %acc 로 누적 → reduction_combine="add".
// 출력 C 는 x(pid0)에만 의존하고 y(pid1)에 불변 → y 만 다른 instance 들이 같은 C 에 쓰므로
// y 축을 쪼개면 부분 결과가 생긴다 → split_axes 의 y 항목에 combine 이 붙는다.
// K 루프도 쪼갤 수 있는 축이라 source="reduction" 으로 함께 올라간다.
// CHECK-LABEL: func.func @splitk
// CHECK-SAME:  pim.grid_dims = ["x", "y"]
// CHECK-SAME:  pim.split_axes = [{axis = "x", source = "grid"}, {axis = "y", combine = "add", source = "grid"}, {axis = "k", combine = "add", source = "reduction"}]
func.func @splitk(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
                  %np0: i32, %np1: i32, %np2: i32,
                  %pid0: i32, %pid1: i32, %pid2: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.0 : f32
  %x = arith.index_cast %pid0 : i32 to index      // 출력 행 타일
  %y = arith.index_cast %pid1 : i32 to index      // K 분할 인덱스
  %coff = arith.muli %x, %c32 : index             // C offset: x 만
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>

  // CHECK:      scf.for
  // CHECK:      pim.reduction_combine = "add"
  %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %init) -> tensor<32x64xf32> {
    %koff = arith.muli %k, %c32 : index
    %aoff0 = arith.addi %coff, %y : index          // A: x, y 의존
    %aoff = arith.addi %aoff0, %koff : index
    %dA = pim.access %A offset %aoff sizes [32, 16] strides [%c1, %c1] : memref<*xf32> -> !pim.access
    %a = pim.load %dA : !pim.access -> tensor<32x16xf32>
    %boff = arith.addi %y, %koff : index           // B: y 의존
    %dB = pim.access %B offset %boff sizes [16, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
    %b = pim.load %dB : !pim.access -> tensor<16x64xf32>
    %next = linalg.matmul ins(%a, %b : tensor<32x16xf32>, tensor<16x64xf32>)
        outs(%acc : tensor<32x64xf32>) -> tensor<32x64xf32>
    scf.yield %next : tensor<32x64xf32>
  }

  // 출력 C 는 y 에 불변 → invariant=[1], role=output.
  // CHECK:      pim.access %arg2 {{.*}} {pim.invariant_grid_dims = array<i64: 1>, pim.reuse_realizable = true, pim.role = "output"}
  %dC = pim.access %C offset %coff sizes [32, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
  pim.store %r, %dC : tensor<32x64xf32>, !pim.access
  return
}

// ── GEMM 과 GEMV: 같은 규칙이 서로 다른 분배 구조를 낸다 ───────────────────────
// 축을 쪼갤 때 access 가 그 축에 의존하면 나누고, 불변이면 복제한다. 무엇이 복제 대상이
// 되는지가 두 연산에서 갈린다.

// GEMM: C[M,N] = A[M,K] x B[K,N]. grid 는 m(x), n(y) 두 축이고 K loop 가 안에 있다.
// A 는 n 에 불변, B 는 m 에 불변 → 쪼갤 때 둘 다 복제 대상이고 둘 다 크다.
// C 는 두 축 모두 의존 → 결합이 필요 없어 x, y 항목에 combine 이 없다.
// A 와 B 의 주소가 K loop 에 의존하므로 그 재사용은 loop 순서를 바꾸지 않는 한 실현되지 않는다.
// CHECK-LABEL: func.func @gemm
// CHECK-SAME:  pim.grid_dims = ["x", "y"]
// CHECK-SAME:  pim.split_axes = [{axis = "x", source = "grid"}, {axis = "y", source = "grid"}, {axis = "k", combine = "add", source = "reduction"}]
func.func @gemm(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
                %np0: i32, %np1: i32, %np2: i32,
                %pid0: i32, %pid1: i32, %pid2: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f32
  %m = arith.index_cast %pid0 : i32 to index
  %n = arith.index_cast %pid1 : i32 to index
  %moff = arith.muli %m, %c32 : index
  %noff = arith.muli %n, %c64 : index
  %coff = arith.addi %moff, %noff : index
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>

  %r = scf.for %k = %c0 to %c32 step %c1 iter_args(%acc = %init) -> tensor<32x64xf32> {
    %koff = arith.muli %k, %c32 : index
    // A: m 과 k 에 의존, n 에 불변 → n 축을 쪼개면 복제된다.
    // CHECK: pim.access %arg0 {{.*}} {pim.invariant_grid_dims = array<i64: 1>, pim.reuse_realizable = false, pim.role = "input"}
    %aoff = arith.addi %moff, %koff : index
    %dA = pim.access %A offset %aoff sizes [32, 16] strides [%c1, %c1] : memref<*xf32> -> !pim.access
    %a = pim.load %dA : !pim.access -> tensor<32x16xf32>
    // B: n 과 k 에 의존, m 에 불변 → m 축을 쪼개면 복제된다.
    // CHECK: pim.access %arg1 {{.*}} {pim.invariant_grid_dims = array<i64: 0>, pim.reuse_realizable = false, pim.role = "input"}
    %boff = arith.addi %noff, %koff : index
    %dB = pim.access %B offset %boff sizes [16, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
    %b = pim.load %dB : !pim.access -> tensor<16x64xf32>
    %next = linalg.matmul ins(%a, %b : tensor<32x16xf32>, tensor<16x64xf32>)
        outs(%acc : tensor<32x64xf32>) -> tensor<32x64xf32>
    scf.yield %next : tensor<32x64xf32>
  }
  // C: 두 축 모두 의존 → 복제 없이 나뉜다. 주소가 grid 에만 의존하므로 재사용 실현 가능.
  // CHECK: pim.access %arg2 {{.*}} {pim.invariant_grid_dims = array<i64>, pim.reuse_realizable = true, pim.role = "output"}
  %dC = pim.access %C offset %coff sizes [32, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
  pim.store %r, %dC : tensor<32x64xf32>, !pim.access
  return
}

// GEMV: y[M] = A[M,K] x[K]. grid 는 m(x) 한 축이고 K loop 가 안에 있다.
// 복제 대상이 x 하나뿐이고 A 에 비해 아주 작다. GEMM 과 갈리는 지점이 이것이다.
// CHECK-LABEL: func.func @gemv
// CHECK-SAME:  pim.grid_dims = ["x"]
// CHECK-SAME:  pim.split_axes = [{axis = "x", source = "grid"}, {axis = "k", combine = "add", source = "reduction"}]
func.func @gemv(%A: memref<*xf32>, %X: memref<*xf32>, %Y: memref<*xf32>,
                %np0: i32, %np1: i32, %np2: i32,
                %pid0: i32, %pid1: i32, %pid2: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f32
  %m = arith.index_cast %pid0 : i32 to index
  %moff = arith.muli %m, %c64 : index
  %empty = tensor.empty() : tensor<64xf32>
  %init = linalg.fill ins(%cst : f32) outs(%empty : tensor<64xf32>) -> tensor<64xf32>

  %r = scf.for %k = %c0 to %c16 step %c1 iter_args(%acc = %init) -> tensor<64xf32> {
    %koff = arith.muli %k, %c64 : index
    // A: m 과 k 에 의존 → 불변 축 없음. 쪼개면 유닛마다 서로 다른 행 밴드를 받는다.
    // CHECK: pim.access %arg0 {{.*}} {pim.invariant_grid_dims = array<i64>, pim.reuse_realizable = false, pim.role = "input"}
    %aoff = arith.addi %moff, %koff : index
    %dA = pim.access %A offset %aoff sizes [64, 256] strides [%c1, %c1] : memref<*xf32> -> !pim.access
    %a = pim.load %dA : !pim.access -> tensor<64x256xf32>
    // x: k 에만 의존 → m 에 불변. 모든 instance 가 같은 x 를 읽으므로 쪼개면 복제된다.
    // CHECK: pim.access %arg1 {{.*}} {pim.invariant_grid_dims = array<i64: 0>, pim.reuse_realizable = false, pim.role = "input"}
    %dX = pim.access %X offset %koff sizes [256] strides [%c1] : memref<*xf32> -> !pim.access
    %x = pim.load %dX : !pim.access -> tensor<256xf32>
    %next = linalg.matvec ins(%a, %x : tensor<64x256xf32>, tensor<256xf32>)
        outs(%acc : tensor<64xf32>) -> tensor<64xf32>
    scf.yield %next : tensor<64xf32>
  }
  // y: m 에 의존 → 복제 없이 나뉜다.
  // CHECK: pim.access %arg2 {{.*}} {pim.invariant_grid_dims = array<i64>, pim.reuse_realizable = true, pim.role = "output"}
  %dY = pim.access %Y offset %moff sizes [64] strides [%c1] : memref<*xf32> -> !pim.access
  pim.store %r, %dY : tensor<64xf32>, !pim.access
  return
}

// 영역을 정하는 것은 offset 하나가 아니다. strides·valid·index·pad도 함께 봐야 한다.

// row gather의 index tensor는 y에서, offset은 x에서 온다.
// offset만 보면 y에 불변이라고 판정하지만, 읽는 행이 y마다 달라지므로 불변이 아니다.
// 이걸 놓치면 target이 tile을 유닛에 상주시켰다가 잘못된 행을 읽는다.
// CHECK-LABEL: func.func @gather_index_dep
// CHECK-SAME:  pim.grid_dims = ["x", "y"]
// index를 만드는 access 자체는 y에만 의존 → x에 불변.
// CHECK:      pim.access %arg1 {{.*}} {pim.invariant_grid_dims = array<i64: 0>, pim.reuse_realizable = true, pim.role = "input"}
// gather access는 offset(x) + index(y) 둘 다 의존 → 불변 없음.
// CHECK:      pim.access %arg0 {{.*}} {pim.invariant_grid_dims = array<i64>, pim.reuse_realizable = true, pim.role = "input"}
func.func @gather_index_dep(%A: memref<*xf32>, %I: memref<*xi32>,
                            %np0: i32, %np1: i32, %np2: i32,
                            %pid0: i32, %pid1: i32, %pid2: i32) {
  %c1 = arith.constant 1 : index
  %x = arith.index_cast %pid0 : i32 to index
  %y = arith.index_cast %pid1 : i32 to index
  %dI = pim.access %I offset %y sizes [8] strides [%c1] : memref<*xi32> -> !pim.access
  %idx = pim.load %dI : !pim.access -> tensor<8xi32>
  %dA = pim.access %A offset %x sizes [8, 64] strides [%c1, %c1] gather 0 index %idx : tensor<8xi32>
      : memref<*xf32> -> !pim.access
  %a = pim.load %dA : !pim.access -> tensor<8x64xf32>
  return
}

// 경계 유효 범위(valid)만 y에 의존한다. 접근하는 원소 개수가 y마다 달라지므로 불변이 아니다.
// pad(유효 범위 밖 원소의 값)도 같은 이유로 추적한다.
// CHECK-LABEL: func.func @valid_pad_dep
// CHECK-SAME:  pim.grid_dims = ["x", "y"]
// valid가 y에 의존 → 불변 없음.
// CHECK:      pim.access %arg0 {{.*}} valid[%{{.*}}] {pim.invariant_grid_dims = array<i64>, pim.reuse_realizable = true, pim.role = "input"}
// pad가 y에 의존 → 불변 없음.
// CHECK:      pim.access %arg1 {{.*}} pad %{{.*}} {pim.invariant_grid_dims = array<i64>, pim.reuse_realizable = true, pim.role = "input"}
func.func @valid_pad_dep(%A: memref<*xf32>, %B: memref<*xf32>,
                         %np0: i32, %np1: i32, %np2: i32,
                         %pid0: i32, %pid1: i32, %pid2: i32) {
  %c1 = arith.constant 1 : index
  %x = arith.index_cast %pid0 : i32 to index
  %y = arith.index_cast %pid1 : i32 to index
  %dA = pim.access %A offset %x sizes [128] strides [%c1] valid [%y]
      : memref<*xf32> -> !pim.access
  %a = pim.load %dA : !pim.access -> tensor<128xf32>
  %p = arith.sitofp %pid1 : i32 to f32
  %dB = pim.access %B offset %x sizes [128] strides [%c1] valid [%c1] pad %p : f32
      : memref<*xf32> -> !pim.access
  %b = pim.load %dB : !pim.access -> tensor<128xf32>
  return
}
