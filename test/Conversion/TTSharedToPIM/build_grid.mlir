// --pim-build-grid: runtime가 돌던 grid 반복을 IR 안의 scf.for로 옮긴다.
// program_id 인자 → grid loop 유도변수(인자는 삭제), num_programs 인자 → loop 상한(유지).
// 몸체 코드는 건드리지 않는다.
// RUN: triton-shared-opt --pim-build-grid %s | FileCheck %s

// 2차원 grid(PIM 권장 형태): 축마다 loop 하나, 바깥이 x·안쪽이 y.
// 인자 9개 → 6개. program_id 3개는 지우고 num_programs 3개는 상한으로 남는다.
// CHECK-LABEL: func.func @grid_2d
// CHECK-SAME:  (%{{.+}}: memref<*xf32>, %[[NP0:.+]]: i32, %[[NP1:.+]]: i32, %{{.+}}: i32) {
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// 상한 cast는 두 loop보다 앞에 나온다(안쪽 loop 상한이 바깥 loop 안에 갇히지 않게).
// CHECK:         %[[UB0:.+]] = arith.index_cast %[[NP0]] : i32 to index
// CHECK:         %[[UB1:.+]] = arith.index_cast %[[NP1]] : i32 to index
// CHECK:         scf.for %[[GX:.+]] = %[[C0]] to %[[UB0]] step %[[C1]] {
// CHECK:           scf.for %[[GY:.+]] = %[[C0]] to %[[UB1]] step %[[C1]] {
// 유도변수 cast는 몸체 맨 앞 — 몸체 전체를 지배해야 한다.
// CHECK:             %[[PX:.+]] = arith.index_cast %[[GX]] : index to i32
// CHECK:             %[[PY:.+]] = arith.index_cast %[[GY]] : index to i32
// CHECK:             arith.index_cast %[[PX]] : i32 to index
// CHECK:             arith.index_cast %[[PY]] : i32 to index
// CHECK:             pim.access
// CHECK:           } {pim.grid_dim = "y"}
// CHECK:         } {pim.grid_dim = "x"}
func.func @grid_2d(%A: memref<*xf32>,
                   %np0: i32, %np1: i32, %np2: i32,
                   %pid0: i32, %pid1: i32, %pid2: i32) {
  %c1 = arith.constant 1 : index
  %m = arith.index_cast %pid0 : i32 to index
  %n = arith.index_cast %pid1 : i32 to index
  %off = arith.addi %m, %n : index
  %d = pim.access %A offset %off sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  %v = pim.load %d : !pim.access -> tensor<32xf32>
  pim.store %v, %d : tensor<32xf32>, !pim.access
  return
}

// 1차원 grid + grouped ordering(Triton matmul 튜토리얼 형태).
// 이 패스는 몸체를 알아보지 않으므로 divsi/remsi 연쇄가 그대로 남고, 축은 하나만 생긴다.
// 축이 하나뿐이면 분할 후보가 없다 — 그래서 PIM용 커널은 다차원 grid로 쓴다.
// CHECK-LABEL: func.func @grid_1d_grouped
// CHECK-SAME:  (%{{.+}}: memref<*xf32>, %{{.+}}: i32, %{{.+}}: i32, %[[NP0:.+]]: i32, %{{.+}}: i32, %{{.+}}: i32) {
// CHECK:         %[[UB:.+]] = arith.index_cast %[[NP0]] : i32 to index
// CHECK:         scf.for %[[GX:.+]] = %{{.+}} to %[[UB]] step %{{.+}} {
// CHECK:           %[[PID:.+]] = arith.index_cast %[[GX]] : index to i32
// grouped ordering 연산은 손대지 않고 그대로 둔다.
// CHECK:           arith.divsi %[[PID]], %{{.+}} : i32
// CHECK:           arith.remsi %[[PID]], %{{.+}} : i32
// CHECK:         } {pim.grid_dim = "x"}
// CHECK-NOT:     pim.grid_dim = "y"
func.func @grid_1d_grouped(%A: memref<*xf32>, %M: i32, %N: i32,
                           %np0: i32, %np1: i32, %np2: i32,
                           %pid0: i32, %pid1: i32, %pid2: i32) {
  %c1 = arith.constant 1 : index
  %c8_i32 = arith.constant 8 : i32
  %c31_i32 = arith.constant 31 : i32
  %c32_i32 = arith.constant 32 : i32
  %c63_i32 = arith.constant 63 : i32
  %c64_i32 = arith.constant 64 : i32
  %0 = arith.addi %M, %c31_i32 : i32
  %num_pid_m = arith.divsi %0, %c32_i32 : i32
  %1 = arith.addi %N, %c63_i32 : i32
  %num_pid_n = arith.divsi %1, %c64_i32 : i32
  %group = arith.muli %num_pid_n, %c8_i32 : i32
  %gid = arith.divsi %pid0, %group : i32
  %first_m = arith.muli %gid, %c8_i32 : i32
  %2 = arith.subi %num_pid_m, %first_m : i32
  %gsize = arith.minsi %2, %c8_i32 : i32
  %3 = arith.remsi %pid0, %gsize : i32
  %pid_m = arith.addi %first_m, %3 : i32
  %4 = arith.remsi %pid0, %group : i32
  %pid_n = arith.divsi %4, %gsize : i32
  %m = arith.index_cast %pid_m : i32 to index
  %n = arith.index_cast %pid_n : i32 to index
  %off = arith.addi %m, %n : index
  %d = pim.access %A offset %off sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  %v = pim.load %d : !pim.access -> tensor<32xf32>
  pim.store %v, %d : tensor<32xf32>, !pim.access
  return
}

// ABI는 맞지만 program_id를 아무도 쓰지 않으면 축이 없다 → 그대로 둔다.
// 인자도 지우지 않는다(변환 자체를 하지 않으므로).
// CHECK-LABEL: func.func @unused_pid
// CHECK-SAME:  (%{{.+}}: memref<*xf32>, %{{.+}}: i32, %{{.+}}: i32, %{{.+}}: i32, %{{.+}}: i32, %{{.+}}: i32, %{{.+}}: i32) {
// CHECK-NOT:     scf.for
func.func @unused_pid(%A: memref<*xf32>,
                      %np0: i32, %np1: i32, %np2: i32,
                      %pid0: i32, %pid1: i32, %pid2: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d = pim.access %A offset %c0 sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  %v = pim.load %d : !pim.access -> tensor<32xf32>
  pim.store %v, %d : tensor<32xf32>, !pim.access
  return
}

// triton-shared ABI가 아닌 손으로 쓴 함수(인자 6개 미만) → 건드리지 않는다.
// CHECK-LABEL: func.func @no_abi
// CHECK-NOT:     scf.for
func.func @no_abi(%A: memref<*xf32>, %off: index) {
  %c1 = arith.constant 1 : index
  %d = pim.access %A offset %off sizes [32] strides [%c1] : memref<*xf32> -> !pim.access
  %v = pim.load %d : !pim.access -> tensor<32xf32>
  pim.store %v, %d : tensor<32xf32>, !pim.access
  return
}
