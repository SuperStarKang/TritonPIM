// grid 차원과 각 pim.access의 의존 관계를 분석해 func엔 execution_model/program_dims,
// access엔 role/invariant_program_dims를 단다.
// grid 차원은 pim.grid_dim이 붙은 scf.for로 식별하므로 --pim-build-grid를 먼저 돌린다.
// RUN: triton-shared-opt --pim-build-grid --pim-analyze-logical-work %s | FileCheck %s

// GEMM: A는 pid0(m)만, B는 pid1(n)만, C는 둘 다에 의존.
// → program_dims=["x","y"]. A는 n(=dim1)에 불변, B는 m(=dim0)에 불변, C는 불변 없음.
// CHECK-LABEL: func.func @matmul
// CHECK-SAME:  pim.execution_model = "spmd"
// CHECK-SAME:  pim.program_dims = ["x", "y"]
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
  // CHECK: pim.access %arg0 {{.*}} {pim.invariant_program_dims = array<i64: 1>, pim.role = "input"}
  %dA = pim.access %A offset %aoff sizes [32, 256] strides [%c1, %c1] : memref<*xf32> -> !pim.access
  %a = pim.load %dA : !pim.access -> tensor<32x256xf32>

  // B: n 방향으로만 움직임 → m(dim0) 불변.
  // CHECK: pim.access %arg1 {{.*}} {pim.invariant_program_dims = array<i64: 0>, pim.role = "input"}
  %dB = pim.access %B offset %boff sizes [256, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
  %b = pim.load %dB : !pim.access -> tensor<256x64xf32>

  // C: 두 축 모두 의존 → 불변 없음, output role.
  // CHECK: pim.access %arg2 {{.*}} {pim.invariant_program_dims = array<i64>, pim.role = "output"}
  %z = tensor.empty() : tensor<32x64xf32>
  %dC = pim.access %C offset %coff sizes [32, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
  pim.store %z, %dC : tensor<32x64xf32>, !pim.access
  return
}

// loop-carried offset(iter_arg)은 program ID에 의존하지 않는다 → execution_model만 달고
// program_dims/invariant_program_dims는 달지 않는다.
// CHECK-LABEL: func.func @advance
// CHECK-SAME:  pim.execution_model = "spmd"
// CHECK-NOT:   pim.program_dims
// CHECK-NOT:   pim.invariant_program_dims
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

// ⑥ reduction loop + ⑦ reduce_across_programs.
// K 루프에서 matmul 이 누산기 %acc 로 누적 → reduction_combine="add".
// 출력 C 는 x(pid0)에만 의존하고 y(pid1)에 불변 → y 만 다른 program 들이 같은 C 에 씀
//   → reduce_across_programs=["y"] (Split-K 형태).
// CHECK-LABEL: func.func @splitk
// CHECK-SAME:  pim.program_dims = ["x", "y"]
// CHECK-SAME:  pim.reduce_across_programs = ["y"]
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
  // CHECK:      pim.access %arg2 {{.*}} {pim.invariant_program_dims = array<i64: 1>, pim.role = "output"}
  %dC = pim.access %C offset %coff sizes [32, 64] strides [%c1, %c1] : memref<*xf32> -> !pim.access
  pim.store %r, %dC : tensor<32x64xf32>, !pim.access
  return
}

// 영역을 정하는 것은 offset 하나가 아니다. strides·valid·index·pad도 함께 봐야 한다.

// row gather의 index tensor는 y에서, offset은 x에서 온다.
// offset만 보면 y에 불변이라고 판정하지만, 읽는 행이 y마다 달라지므로 불변이 아니다.
// 이걸 놓치면 target이 tile을 유닛에 상주시켰다가 잘못된 행을 읽는다.
// CHECK-LABEL: func.func @gather_index_dep
// CHECK-SAME:  pim.program_dims = ["x", "y"]
// index를 만드는 access 자체는 y에만 의존 → x에 불변.
// CHECK:      pim.access %arg1 {{.*}} {pim.invariant_program_dims = array<i64: 0>, pim.role = "input"}
// gather access는 offset(x) + index(y) 둘 다 의존 → 불변 없음.
// CHECK:      pim.access %arg0 {{.*}} {pim.invariant_program_dims = array<i64>, pim.role = "input"}
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
// CHECK-SAME:  pim.program_dims = ["x", "y"]
// valid가 y에 의존 → 불변 없음.
// CHECK:      pim.access %arg0 {{.*}} valid[%{{.*}}] {pim.invariant_program_dims = array<i64>, pim.role = "input"}
// pad가 y에 의존 → 불변 없음.
// CHECK:      pim.access %arg1 {{.*}} pad %{{.*}} {pim.invariant_program_dims = array<i64>, pim.role = "input"}
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
