// 이웃 tile 을 block group 으로 묶는다: loop 는 group 좌표가 되고, access 의 sizes 와
// tile tensor 가 group 크기로 커진다. 경계는 언제나 tile 경계 위다.
// RUN: triton-shared-opt --pim-form-block-groups %s | FileCheck %s --check-prefix=IDENT
// RUN: triton-shared-opt "--pim-form-block-groups=group-m=2 group-n=2" %s | FileCheck %s
// RUN: triton-shared-opt --pim-form-block-groups=group-k=2 %s | FileCheck %s --check-prefix=GK
// RUN: triton-shared-opt "--pim-set-target-profile=target=aimx num-units=1 banks-per-unit=256 cores-per-bank=1 lanes-per-core=16 bank-capacity-limit-bytes=268435456 local-capacity-limit-bytes=0" "--pim-form-block-groups=group-m=1 group-n=1 group-k=1" --pim-form-execution-tiles %s | FileCheck %s --check-prefix=AIMX
// RUN: triton-shared-opt "--pim-form-block-groups=group-m=1 group-n=1 group-k=1" "--pim-form-execution-tiles=tile-m=2 tile-n=3 tile-k=4" %s | FileCheck %s --check-prefix=GEMM-TILE

// AIMX: #pim.target_profile<target = "aimx"
// AIMX-NOT: execution_unit
// AIMX: #pim.block<shape = [8, 32, 32]>
// AIMX: #pim.block_group<blocks = [1, 1, 1]>
// AIMX: #pim.execution_tile<groups = [1, 1, 1]>

// GEMM-TILE: #pim.block<shape = [8, 32, 32]>
// GEMM-TILE: #pim.block_group<blocks = [1, 1, 1]>
// GEMM-TILE: #pim.execution_tile<groups = [2, 3, 4]>

// 기본값(전부 1)은 항등이다.
// IDENT: #pim.block<shape = [8, 32, 32]>
// IDENT: #pim.block_group<blocks = [1, 1, 1]>
// IDENT: sizes [8, 32]
// IDENT: tensor<8x32xi32>

// group-k 는 reduction loop 를 넓힌다: 상한은 그 loop 앞에서 반으로 줄고, A·B 의
// K 변만 커진다 — 누산기(8×32)는 그대로다.
// GK:      arith.ceildivsi
// GK:      scf.for %[[Z:.*]] = %{{.*}} iter_args
// GK:        arith.muli %[[Z]], %{{.*}}
// GK:        sizes [8, 64]
// GK:        pim.load %{{.*}} : !pim.access -> tensor<8x64xi8>
// GK:        sizes [64, 32]
// GK:        pim.load %{{.*}} : !pim.access -> tensor<64x32xi8>
// GK:        outs(%{{.*}} : tensor<8x32xi32>)
// GK:      sizes [8, 32]

// 축 loop: 상한 = ceildiv(tiles, 2), 몸체 첫머리가 첫 tile 좌표(iv·2)를 복원한다.
// 새 상한 둘은 반드시 nest 밖(가장 바깥 축 loop 앞)에 있어야 한다 — pim-map-devices 가
// nest 밖에서 상한을 읽어 extents 를 만들기 때문이다.
// CHECK:      #pim.block<shape = [8, 32, 32]>
// CHECK:      #pim.block_group<blocks = [2, 2, 1]>
// CHECK:      arith.ceildivsi
// CHECK:      arith.ceildivsi
// CHECK:      scf.for %[[XG:.*]] = %{{.*}} to
// CHECK:        arith.muli %[[XG]], %{{.*}}
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
    } {pim.axis = #pim.axis<"n">}
  } {pim.axis = #pim.axis<"m">}
  return
}
