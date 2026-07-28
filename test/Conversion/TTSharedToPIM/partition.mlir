// 반복 공간을 유닛 수만큼 나눠 pim.units + pim.unit_data를 만드는지 확인.
// 입력은 --pim-analyze-reuse까지 돌린 모습의 2차원 GEMM이다(실제 커널과 같은 형태:
// K loop 상한이 런타임 인자, 주소는 유도변수에서 매번 계산).
// 파이프라인 순서대로 --pim-normalize-accumulator를 함께 돌린다. 누산기 초기값이 K loop
// 앞으로 내려와야 유닛 region 안에 들어간다.
// RUN: triton-shared-opt --pim-normalize-accumulator --pim-partition="splits=2,4,8" %s | FileCheck %s

// CHECK-LABEL: func.func @matmul
func.func @matmul(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>,
                  %K: i32, %N: i32, %npx: index, %npy: index)
    attributes {pim.execution_model = "spmd", pim.grid_dims = ["x", "y"],
                pim.split_axes = [{axis = "x", source = "grid"},
                                  {axis = "y", source = "grid"},
                                  {axis = "k", source = "reduction", combine = "add"}]} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %lda = arith.index_cast %K : i32 to index
  %ldb = arith.index_cast %N : i32 to index
  %init = tensor.empty() : tensor<8x32xi32>

  scf.for %gx = %c0 to %npx step %c1 {
    %m = arith.muli %gx, %c8 : index
    scf.for %gy = %c0 to %npy step %c1 {
      %n = arith.muli %gy, %c32 : index
      %acc = scf.for %k = %c0_i32 to %K step %c16_i32
             iter_args(%a = %init) -> (tensor<8x32xi32>) : i32 {
        %ki = arith.index_cast %k : i32 to index
        %arow = arith.muli %m, %lda : index
        %aoff = arith.addi %arow, %ki : index
        %da = pim.access %A offset %aoff sizes [8, 16] strides[%lda, %c1] origin[%m, %ki]
              {pim.access_id = 0 : i64, pim.role = "input", pim.invariant_grid_dims = array<i64: 1>}
              : memref<*xi8> -> !pim.access
        %ta = pim.load %da : !pim.access -> tensor<8x16xi8>
        %brow = arith.muli %ki, %ldb : index
        %boff = arith.addi %brow, %n : index
        %db = pim.access %B offset %boff sizes [16, 32] strides[%ldb, %c1] origin[%ki, %n]
              {pim.access_id = 1 : i64, pim.role = "input", pim.invariant_grid_dims = array<i64: 0>}
              : memref<*xi8> -> !pim.access
        %tb = pim.load %db : !pim.access -> tensor<16x32xi8>
        %mm = linalg.matmul ins(%ta, %tb : tensor<8x16xi8>, tensor<16x32xi8>)
              outs(%a : tensor<8x32xi32>) -> tensor<8x32xi32>
        scf.yield %mm : tensor<8x32xi32>
      } {pim.reduction_combine = "add"}
      %crow = arith.muli %m, %ldb : index
      %coff = arith.addi %crow, %n : index
      %dc = pim.access %C offset %coff sizes [8, 32] strides[%ldb, %c1] origin[%m, %n]
            {pim.access_id = 2 : i64, pim.role = "output", pim.invariant_grid_dims = array<i64>}
            : memref<*xi32> -> !pim.access
      pim.store %acc, %dc : tensor<8x32xi32>, !pim.access
    } {pim.grid_dim = "y"}
  } {pim.grid_dim = "x"}
  return
}

// 누산기 초기화는 instance마다 따로 있어야 하므로 유닛 region 밖에 남으면 안 된다.
// CHECK-NOT:  tensor.empty

// 세 축 모두 쪼개므로 좌표가 셋이다.
// CHECK:      pim.units axes ["x", "y", "k"] counts [2, 4, 8]
// CHECK-NEXT: ^{{.*}}(%[[UX:.*]]: index, %[[UY:.*]]: index, %[[UK:.*]]: index)

// A: x축 loop가 시작을, k축 loop가 열 범위를 정한다. y축에 불변이라 y만 다른 유닛은 같은 것을 받는다.
// CHECK:      pim.unit_data %arg0 origin[%{{.*}}, %{{.*}}] sizes[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}]
// CHECK-SAME:   direction = "in"
// CHECK-SAME:   access =
// CHECK-SAME:   replicated_over = ["y"]

// B: x축에 불변.
// CHECK:      pim.unit_data %arg1 origin[%{{.*}}, %{{.*}}] sizes[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}]
// CHECK-SAME:   direction = "in"
// CHECK-SAME:   access =
// CHECK-SAME:   replicated_over = ["x"]

// C: 어느 grid 축에도 불변이 아니라 복제가 없고, k축을 쪼갰으므로 부분 결과를 더해야 한다.
// CHECK:      pim.unit_data %arg2 origin[%{{.*}}, %{{.*}}] sizes[%{{.*}}, %{{.*}}] strides[%{{.*}}, %{{.*}}]
// CHECK-SAME:   combine = "add"
// CHECK-SAME:   combine_over = ["k"]
// CHECK-SAME:   direction = "out"

// 축 loop 셋의 범위가 유닛 몫으로 좁혀진다(올림 나눗셈 + 원래 상한으로 자르기).
// 크기는 배정량이 아니라 좁힌 범위의 실제 횟수에서 나온다(마지막 유닛이 덜 맡는 경우).
// CHECK:      arith.ceildivsi
// CHECK:      arith.minsi
// CHECK:      arith.maxsi

// 쪼갤 축이 하나도 없으면(전부 1) 유닛 형태를 만들지 않는다.
// RUN: triton-shared-opt --pim-partition="splits=1,1,1" %s | FileCheck %s --check-prefix=NOSPLIT
// NOSPLIT-LABEL: func.func @matmul
// NOSPLIT-NOT:   pim.units

// 기계에 있는 것보다 많은 유닛을 요구하면 컴파일 시점에 실패한다.
// RUN: not triton-shared-opt --pim-partition="splits=2,4,8 num-units=16" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TOOMANY
// TOOMANY: splits need 64 units but the machine has 16
