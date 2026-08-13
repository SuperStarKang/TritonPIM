// 이웃 tile 을 block group 으로 묶는다: loop 는 group 좌표가 되고, access 의 sizes 와
// tile tensor 가 group 크기로 커진다. 경계는 언제나 tile 경계 위다.
// RUN: triton-shared-opt --pim-form-block-groups %s | FileCheck %s --check-prefix=IDENT
// RUN: triton-shared-opt "--pim-form-block-groups=group-m=2 group-n=2" %s | FileCheck %s

// 기본값 1,1 은 항등이다.
// IDENT: sizes [8, 32]
// IDENT: tensor<8x32xi32>

// 축 loop: 상한 = ceildiv(tiles, 2), 몸체 첫머리가 첫 tile 좌표(iv·2)를 복원한다.
// CHECK:      arith.ceildivsi
// CHECK:      scf.for %[[XG:.*]] = %{{.*}} to
// CHECK:        arith.muli %[[XG]], %{{.*}}
// CHECK:        arith.ceildivsi
// CHECK:        scf.for %[[YG:.*]] = %{{.*}} to
// CHECK:          arith.muli %[[YG]], %{{.*}}

// 누산기 사슬은 저장 access 의 크기(16×64)로, load 사슬은 각자 access 의 크기로 커진다.
// A 는 행 축만(16×32), B 는 열 축만(32×64) 커진다.
// CHECK:          tensor.empty() : tensor<16x64xi32>
// CHECK:          sizes [16, 32]
// CHECK:          pim.load %{{.*}} : !pim.access -> tensor<16x32xi8>
// CHECK:          sizes [32, 64]
// CHECK:          pim.load %{{.*}} : !pim.access -> tensor<32x64xi8>
// CHECK:          linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<16x32xi8>, tensor<32x64xi8>) outs(%{{.*}} : tensor<16x64xi32>)
// CHECK:          sizes [16, 64]
// CHECK:          pim.store %{{.*}} : tensor<16x64xi32>, !pim.access

func.func @matmul(%A: memref<*xi8>, %B: memref<*xi8>, %C: memref<*xi32>,
                  %tr: index, %tc: index, %tk: index, %lda: index, %ldb: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32
  scf.for %x = %c0 to %tr step %c1 {
    %m = arith.muli %x, %c8 : index
    scf.for %y = %c0 to %tc step %c1 {
      %n = arith.muli %y, %c32 : index
      %e = tensor.empty() : tensor<8x32xi32>
      %init = linalg.fill ins(%c0_i32 : i32) outs(%e : tensor<8x32xi32>) -> tensor<8x32xi32>
      %r = scf.for %z = %c0 to %tk step %c1 iter_args(%acc = %init) -> (tensor<8x32xi32>) {
        %k = arith.muli %z, %c32 : index
        %da = pim.access %A offset %m sizes [8, 32] strides[%lda, %c1] origin[%m, %k]
              {id = 0 : i64} : memref<*xi8> -> !pim.access
        %ta = pim.load %da : !pim.access -> tensor<8x32xi8>
        %db = pim.access %B offset %k sizes [32, 32] strides[%ldb, %c1] origin[%k, %n]
              {id = 1 : i64} : memref<*xi8> -> !pim.access
        %tb = pim.load %db : !pim.access -> tensor<32x32xi8>
        %mm = linalg.matmul ins(%ta, %tb : tensor<8x32xi8>, tensor<32x32xi8>)
              outs(%acc : tensor<8x32xi32>) -> tensor<8x32xi32>
        scf.yield %mm : tensor<8x32xi32>
      }
      %dc = pim.access %C offset %n sizes [8, 32] strides[%ldb, %c1] origin[%m, %n]
            {id = 2 : i64} : memref<*xi32> -> !pim.access
      pim.store %r, %dc : tensor<8x32xi32>, !pim.access
    } {pim.axis = "output_column_tile"}
  } {pim.axis = "output_row_tile"}
  return
}
