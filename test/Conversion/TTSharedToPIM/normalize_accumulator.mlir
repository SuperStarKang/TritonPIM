// matmul(outs=0으로 채운 tile) + generic(add)이 matmul(outs=누산기)로 접히는지 확인.
// RUN: triton-shared-opt --pim-normalize-accumulator %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @fold
func.func @fold(%A: tensor<8x16xi8>, %B: tensor<16x32xi8>, %n: i32) -> tensor<8x32xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %e = tensor.empty() : tensor<8x32xi32>
  %z = linalg.fill ins(%c0_i32 : i32) outs(%e : tensor<8x32xi32>) -> tensor<8x32xi32>
  %r = scf.for %k = %c0_i32 to %n step %c16_i32
       iter_args(%acc = %z) -> (tensor<8x32xi32>) : i32 {
    %mm = linalg.matmul ins(%A, %B : tensor<8x16xi8>, tensor<16x32xi8>)
          outs(%z : tensor<8x32xi32>) -> tensor<8x32xi32>
    %add = linalg.generic {indexing_maps = [#map, #map, #map],
                           iterator_types = ["parallel", "parallel"]}
           ins(%acc, %mm : tensor<8x32xi32>, tensor<8x32xi32>)
           outs(%acc : tensor<8x32xi32>) {
    ^bb0(%in: i32, %in2: i32, %out: i32):
      %s = arith.addi %in, %in2 : i32
      linalg.yield %s : i32
    } -> tensor<8x32xi32>
    scf.yield %add : tensor<8x32xi32>
  }
  return %r : tensor<8x32xi32>
}

// matmul이 누산기에 직접 누적하고, 덧셈 generic과 임시 tile이 사라진다.
// 초기화는 loop 바로 앞으로 내려온다(그래야 instance마다 새로 0이 된다).
// CHECK:      %[[E:.*]] = tensor.empty
// CHECK-NEXT: %[[ACC:.*]] = linalg.fill ins({{.*}}) outs(%[[E]]
// CHECK-NEXT: scf.for {{.*}} iter_args(%[[IT:.*]] = %[[ACC]])
// CHECK:        %[[MM:.*]] = linalg.matmul ins({{.*}}) outs(%[[IT]] : tensor<8x32xi32>)
// CHECK-NOT:    linalg.generic
// CHECK:        scf.yield %[[MM]]

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>

// outs가 0이 아니면 접을 수 없다 — 0에 더하는 것과 그 값에 더하는 것이 다르다.
// CHECK-LABEL: func.func @nonzero_init
func.func @nonzero_init(%A: tensor<8x16xi8>, %B: tensor<16x32xi8>, %bias: tensor<8x32xi32>,
                        %n: i32) -> tensor<8x32xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %e = tensor.empty() : tensor<8x32xi32>
  %one = linalg.fill ins(%c1_i32 : i32) outs(%e : tensor<8x32xi32>) -> tensor<8x32xi32>
  %r = scf.for %k = %c0_i32 to %n step %c16_i32
       iter_args(%acc = %bias) -> (tensor<8x32xi32>) : i32 {
    %mm = linalg.matmul ins(%A, %B : tensor<8x16xi8>, tensor<16x32xi8>)
          outs(%one : tensor<8x32xi32>) -> tensor<8x32xi32>
    %add = linalg.generic {indexing_maps = [#map1, #map1, #map1],
                           iterator_types = ["parallel", "parallel"]}
           ins(%acc, %mm : tensor<8x32xi32>, tensor<8x32xi32>)
           outs(%acc : tensor<8x32xi32>) {
    ^bb0(%in: i32, %in2: i32, %out: i32):
      %s = arith.addi %in, %in2 : i32
      linalg.yield %s : i32
    } -> tensor<8x32xi32>
    scf.yield %add : tensor<8x32xi32>
  }
  return %r : tensor<8x32xi32>
}

// 그대로 둔다.
// CHECK:      linalg.matmul
// CHECK:      linalg.generic
