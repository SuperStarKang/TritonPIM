// source grid 축을 logical tile 좌표로 되돌린다.
// RUN: triton-shared-opt --split-input-file --pim-build-logical-grid %s | FileCheck %s

// 2차원 grid: 축이 그대로 logical 축이다.
// CHECK-LABEL: func.func @plain
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:           } {pim.axis = "output_column_tile"}
// CHECK:         } {pim.axis = "output_row_tile"}
func.func @plain(%A: memref<*xf32>, %npx: i32, %npy: i32, %npz: i32,
                 %pidx: i32, %pidy: i32, %pidz: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = arith.index_cast %pidx : i32 to index
  %n = arith.index_cast %pidy : i32 to index
  %o = arith.addi %m, %n : index
  %rc = memref.reinterpret_cast %A to offset: [%o], sizes: [8], strides: [%c1]
      : memref<*xf32> to memref<8xf32, strided<[?], offset: ?>>
  %v = memref.load %rc[%c0] : memref<8xf32, strided<[?], offset: ?>>
  return
}

// -----

// pid / d 와 pid % d 쌍은 한 축을 두 logical 축으로 가른다. 남는 source 축이 명시적 K 분할이다.
// 행 tile 수는 IR에 없으므로 ceildiv(grid, d)로 유도한다.
// CHECK-LABEL: func.func @div_rem
// CHECK:         %[[D:.*]] = arith.divsi
// CHECK:         %[[R:.*]] = arith.ceildivsi %arg1, %[[D]]
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}}
// CHECK:           } {pim.axis = "explicit_k_partition"}
// CHECK:         } {pim.axis = "output_column_tile"}
// CHECK:         } {pim.axis = "output_row_tile"}
func.func @div_rem(%A: memref<*xf32>, %npx: i32, %npy: i32, %npz: i32,
                   %pidx: i32, %pidy: i32, %pidz: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c127 = arith.constant 127 : i32
  %c128 = arith.constant 128 : i32
  %0 = arith.addi %npy, %c127 : i32
  %d = arith.divsi %0, %c128 : i32
  %row = arith.divsi %pidx, %d : i32
  %col = arith.remsi %pidx, %d : i32
  %m = arith.index_cast %row : i32 to index
  %n = arith.index_cast %col : i32 to index
  %k = arith.index_cast %pidy : i32 to index
  %o = arith.addi %m, %n : index
  %o2 = arith.addi %o, %k : index
  %rc = memref.reinterpret_cast %A to offset: [%o2], sizes: [8], strides: [%c1]
      : memref<*xf32> to memref<8xf32, strided<[?], offset: ?>>
  %v = memref.load %rc[%c0] : memref<8xf32, strided<[?], offset: ?>>
  return
}

// -----

// GROUP_M 순회 순서. 그룹 좌표 사슬은 통째로 죽고 2차원 nest만 남는다.
// CHECK-LABEL: func.func @group_order
// CHECK-NOT:     arith.remsi
// CHECK:         } {pim.axis = "output_column_tile"}
// CHECK:         } {pim.axis = "output_row_tile"}
func.func @group_order(%A: memref<*xf32>, %M: i32, %N: i32,
                       %npx: i32, %npy: i32, %npz: i32,
                       %pidx: i32, %pidy: i32, %pidz: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : i32
  %c31 = arith.constant 31 : i32
  %c32 = arith.constant 32 : i32
  %0 = arith.addi %M, %c31 : i32
  %rows = arith.divsi %0, %c32 : i32
  %1 = arith.addi %N, %c31 : i32
  %cols = arith.divsi %1, %c32 : i32
  %tpg = arith.muli %cols, %c2 : i32
  %group = arith.divsi %pidx, %tpg : i32
  %first = arith.muli %group, %c2 : i32
  %rest = arith.subi %rows, %first : i32
  %rig = arith.minsi %rest, %c2 : i32
  %within = arith.remsi %pidx, %tpg : i32
  %rrem = arith.remsi %within, %rig : i32
  %row = arith.addi %first, %rrem : i32
  %col = arith.divsi %within, %rig : i32
  %m = arith.index_cast %row : i32 to index
  %n = arith.index_cast %col : i32 to index
  %o = arith.addi %m, %n : index
  %rc = memref.reinterpret_cast %A to offset: [%o], sizes: [8], strides: [%c1]
      : memref<*xf32> to memref<8xf32, strided<[?], offset: ?>>
  %v = memref.load %rc[%c0] : memref<8xf32, strided<[?], offset: ?>>
  return
}
