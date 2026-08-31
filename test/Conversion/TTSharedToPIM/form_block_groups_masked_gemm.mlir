// M/N/K가 각각 확대되는 GEMM access도 새 Group 크기로 valid를 다시 clamp한다.
// A는 [M,K], B는 [K,N]만 커지고 각 valid rank는 새 sizes와 그대로 대응한다.
// RUN: triton-shared-opt "--pim-form-block-groups=group-m=2 group-n=2 group-k=2" %s | FileCheck %s

// CHECK-LABEL: func.func @masked_gemm_accesses
// CHECK-SAME: pim.block = #pim.block<shape = [8, 8, 4]>
// CHECK-SAME: pim.block_group = #pim.block_group<blocks = [2, 2, 2]>
// CHECK: pim.access %arg0 {{.*}} sizes [16, 8] {{.*}} valid[%{{.*}}, %{{.*}}] pad %{{.*}} : i8
// CHECK: pim.load %{{.*}} : !pim.access -> tensor<16x8xi8>
// CHECK: pim.access %arg1 {{.*}} sizes [8, 16] {{.*}} valid[%{{.*}}, %{{.*}}] pad %{{.*}} : i8
// CHECK: pim.load %{{.*}} : !pim.access -> tensor<8x16xi8>

func.func @masked_gemm_accesses(
    %A: memref<*xi8>, %B: memref<*xi8>,
    %M: index, %N: index, %K: index,
    %mTiles: index, %nTiles: index, %kTiles: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0 : i8
  scf.for %mTile = %c0 to %mTiles step %c1 {
    %m = arith.muli %mTile, %c8 : index
    %mRemaining = arith.subi %M, %m : index
    %mNonnegative = arith.maxsi %mRemaining, %c0 : index
    %mValid = arith.minsi %mNonnegative, %c8 : index
    scf.for %nTile = %c0 to %nTiles step %c1 {
      %n = arith.muli %nTile, %c8 : index
      %nRemaining = arith.subi %N, %n : index
      %nNonnegative = arith.maxsi %nRemaining, %c0 : index
      %nValid = arith.minsi %nNonnegative, %c8 : index
      scf.for %kTile = %c0 to %kTiles step %c1 {
        %k = arith.muli %kTile, %c4 : index
        %kRemaining = arith.subi %K, %k : index
        %kNonnegative = arith.maxsi %kRemaining, %c0 : index
        %kValid = arith.minsi %kNonnegative, %c4 : index
        %aRow = arith.muli %m, %K : index
        %aOffset = arith.addi %aRow, %k : index
        %aAccess = pim.access %A offset %aOffset sizes [8, 4]
            strides[%K, %c1] origin[%m, %k] valid[%mValid, %kValid]
            pad %zero : i8 {id = 0 : i64} : memref<*xi8> -> !pim.access
        %aValue = pim.load %aAccess : !pim.access -> tensor<8x4xi8>
        %bRow = arith.muli %k, %N : index
        %bOffset = arith.addi %bRow, %n : index
        %bAccess = pim.access %B offset %bOffset sizes [4, 8]
            strides[%N, %c1] origin[%k, %n] valid[%kValid, %nValid]
            pad %zero : i8 {id = 1 : i64} : memref<*xi8> -> !pim.access
        %bValue = pim.load %bAccess : !pim.access -> tensor<4x8xi8>
      } {pim.reduction = "add"}
    } {pim.axis = #pim.axis<"n">}
  } {pim.axis = #pim.axis<"m">}
  return
}
