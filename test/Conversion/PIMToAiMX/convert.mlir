// PIM Common IR이 AiMX level 1 IR로 바뀌고, 고정 execution-tile 모양에서 descriptor
// program이 한 번 펼쳐지는지 확인한다. tail의 유효 크기는 stage/unstage만 바꾼다.
// RUN: triton-shared-opt --convert-pim-to-aimx %s | FileCheck %s
// RUN: triton-shared-opt --convert-pim-to-aimx --aimx-lower-matmul %s | FileCheck %s --check-prefix=PROGRAM
// RUN: triton-shared-opt --convert-pim-to-aimx --aimx-lower-matmul %s | FileCheck %s --check-prefix=PIM-CLEAN
// RUN: triton-shared-opt --convert-pim-to-aimx --aimx-lower-matmul %s | triton-shared-translate --aimx-to-runtime-stub > /dev/null
// RUN: triton-shared-opt --convert-pim-to-aimx --aimx-lower-matmul %s | triton-shared-translate --aimx-to-descriptors > /dev/null
// RUN: not triton-shared-opt "--convert-pim-to-aimx=batch=8 bank-capacity-bytes=4192415" %s 2>&1 | FileCheck %s --check-prefix=BANK-OOM

// BANK-OOM: allocation-wise interleaved footprint needs 4192416 byte in each of 256 banks

// PIM-CLEAN-LABEL: module
// PIM-CLEAN-NOT: #pim.target_profile
// PIM-CLEAN-NOT: #pim.intra_device_plan
// PIM-CLEAN-NOT: #pim.block
// PIM-CLEAN-NOT: #pim.block_group
// PIM-CLEAN-NOT: #pim.execution_tile

// 이 attribute는 할당된 buffer와 재사용 descriptor chain의 고정 모양이다.
// 계층은 Triton block → block group → execution tile이고 전부 공통 IR이 정했다 —
// logical Block Group은 Block 하나이고 batch=8은 AiMX target option에서 온다.
// N Group 32는 target 실행 단위 512로 padding된다. 입력 Common IR의 명시적
// Execution Tile [1, 8, 3248]에서 n_max=4096, k_max=103936이 파생된다.
// CHECK-LABEL: func.func @matmul
// CHECK-SAME:    aimx.tile_group = #aimx.tile_group<block = [32, 32, 32], group = [1, 1, 1], execution = [32, 512, 32]>
// CHECK-SAME:    aimx.tiling = #aimx.tiling<batch = 8, t_m = 32, t_n = 4096, t_k = 103936, padded_k = 106496, padded_n = 4096, transposed = false>

// FPGA 하나를 확보하고 그 안에서 buffer를 잡는다.
// CHECK: aimx.acquire 1 : !aimx.device_set
// CHECK: aimx.foreach_device %{{.*}} : !aimx.device_set axes ["m", "n"] counts [1, 1]

// PMM이 가장 먼저다 — pim_private_malloc이 pim_malloc이 이미 내준 주소를 돌려줄 수 있다.
// 그다음이 A(32*106496*2), Arel(32*103936*32), B(103936*4096*2), C(32*4096*2),
// descriptor chain(211152 group * 512). 합쳐서 1,073,258,496 byte로 1 GiB 예산에 찬다 —
// 명시된 execution tile이 target 물리 눈금으로 padding된 결과다.
// CHECK:   aimx.alloc %{{.*}} bytes 196608 align 262144 kind pmm
// CHECK:   aimx.alloc %{{.*}} bytes 6815744 align 512 kind data
// CHECK:   aimx.alloc %{{.*}} bytes 106430464 align 512 kind data
// CHECK:   aimx.alloc %{{.*}} bytes 851443712 align 512 kind data
// CHECK:   aimx.alloc %{{.*}} bytes 262144 align 512 kind data
// CHECK:   aimx.alloc %{{.*}} bytes 108109824 align 512 kind descriptor

// chain은 tile loop 밖에서 한 번만 만든다. 재배치가 먼저, 계산이 다음이다.
// CHECK:   aimx.build_descriptors @matmul_relayout(
// CHECK:   aimx.build_descriptors @matmul_compute(
// CHECK-NOT: aimx.build_descriptors

// loop 순서는 n → k → m 이다. 가장 큰 buffer인 B가 (k, n)에만 의존하므로 그 둘이
// 바깥이고, 훨씬 작은 A는 안쪽에서 m마다 올라간다. k가 상한(k_max) 안이면 k loop은 한
// 번만 돈다 — K가 런타임 값이라 tile 수도 런타임에 정해지고, 그래서 loop 자체는 언제나
// 나온다.
// padding이 없으므로 stage/unstage에 block group attr(group_rows/group_cols)가 없어야
// 한다 — pitch 값 바로 다음이 into/타입이다.
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       aimx.stage {{.*}} pad_to[{{.*}}] pitch %{{.*}} into
// CHECK:       scf.for
// CHECK:         aimx.stage {{.*}} pad_to[{{.*}}] pitch %{{.*}} into
// CHECK:         aimx.invoke
// k tile은 전부 **같은 C 영역**에 쓴다 — n이나 m을 나눌 때와 다른 점이다. 그래서 두 번째
// tile부터는 덮는 것이 아니라 이전 부분합에 더한다.
// CHECK:         aimx.unstage {{.*}} accumulate %{{.*}} :

// 계산은 descriptor chain이 한다. PIM 쪽도 profile도 남지 않는다.
// CHECK-NOT: linalg.matmul
// CHECK-NOT: pim.device
// CHECK-NOT: pim.place
// CHECK-NOT: pim.access
// CHECK-NOT: pim.target_profile
// CHECK-NOT: pim.block_group

// ── descriptor program ──────────────────────────────────────────────────────
// 재배치: A를 256개씩 읽어(행 간격 padded_k) BC|TP로 다시 쓴다. 두 descriptor가 한 쌍이다.
// PROGRAM-LABEL: aimx.descriptor_program @matmul_relayout batch 8
// PROGRAM:         aimx.desc src(%{{.*}}, %{{.*}}) pitch %{{.*}} len 512 opcode pim_mode|stride
// PROGRAM:         aimx.desc dst(%{{.*}}, %{{.*}}) len 65536 opcode bc|tp

// 계산: reduction step 하나가 A operand와 B operand 한 쌍이고, output block이 끝나면
// vACC를 C로 흘려보낸다. C만 src와 dst 양쪽에 pitch가 붙는다.
// A operand는 vecA 한 칸(8192 = 512*16)으로 batch와 무관하고, B operand와 C drain은
// 출력 block 하나분이라 8192/batch = 1024다.
// PROGRAM-LABEL: aimx.descriptor_program @matmul_compute batch 8
// PROGRAM:         aimx.desc src(%{{.*}}, %{{.*}}) len 8192 opcode a_mask|matmul|pim_mode
// PROGRAM:         aimx.desc src(%{{.*}}, %{{.*}}) pitch %{{.*}} len 1024 opcode b_mask|matmul|pim_mode|stride
// PROGRAM:         aimx.desc src(%{{.*}}, %{{.*}}) pitch %{{.*}} dst(%{{.*}}, %{{.*}}) pitch %{{.*}} len 1024 opcode c_mask|matmul|stride

module attributes {pim.target_profile = #pim.target_profile<
    target = "aimx", number_of_pim_devices = 1,
    number_of_banks_per_pim_device = 256,
    number_of_pim_cores_per_bank = 1,
    number_of_parallel_lanes_per_pim_core = 16,
    parallel_execution_kind = "simd",
    usable_bank_capacity_bytes = 268435456, usable_local_capacity_bytes = 0>} {

// pim.block_group은 --pim-form-block-groups가 만든 target-neutral block 묶음이다.
// AiMX lowering이 useful 크기를 실행 단위 [8, 512, 32]에 맞춰 padding한다.
func.func @matmul(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
                  %K: i32, %N: i32, %ex: index, %ey: index)
    attributes {pim.intra_device_plan = #pim.intra_device_plan<bank_partitions = [8, 32, 1], core_partitions = [1, 1, 1], lanes = 16, lane_axis = "n">,
        pim.block = #pim.block<shape = [32, 32, 32]>,
        pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %lda = arith.index_cast %K : i32 to index
  %ldb = arith.index_cast %N : i32 to index
  %kk = arith.index_cast %K : i32 to index

  pim.device #pim.device_mapping<axes = ["m", "n"],
                                 partitions = [1, 1]>
      extents [%ex, %ey]
      attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 8, 3248]>} {
  ^bb0(%px: index, %py: index):
    // PIM Device 가 하나라 device 몫이 곧 전체다. 실제 pass 가 내는 모양 그대로 손으로 썼다.
    %capx = arith.ceildivsi %ex, %c1 : index
    %startx = arith.muli %px, %capx : index
    %tripsx = arith.minsi %capx, %capx : index
    %capy = arith.ceildivsi %ey, %c1 : index
    %starty = arith.muli %py, %capy : index
    %tripsy = arith.minsi %capy, %capy : index
    %tripsk = arith.ceildivsi %kk, %c32 : index

    %ar = arith.muli %startx, %c32 : index
    %arn = arith.muli %tripsx, %c32 : index
    %arc = arith.muli %capx, %c32 : index
    %akn = arith.muli %tripsk, %c32 : index
    pim.place %A start[%ar, %c0] size[%arn, %akn] capacity[%arc, %akn]
        strides[%lda, %c1]
        {direction = "in", access = 0 : i64,
         mapping = #pim.place_mapping<axis_sources = ["m", ""]>,
         replicated_over = ["n"]} : memref<*xf32>

    %bc = arith.muli %starty, %c32 : index
    %bcn = arith.muli %tripsy, %c32 : index
    %bcc = arith.muli %capy, %c32 : index
    pim.place %B start[%c0, %bc] size[%akn, %bcn] capacity[%akn, %bcc]
        strides[%ldb, %c1]
        {direction = "in", access = 1 : i64,
         mapping = #pim.place_mapping<axis_sources = ["", "n"]>,
         replicated_over = ["m"]} : memref<*xf32>

    pim.place %C start[%ar, %bc] size[%arn, %bcn] capacity[%arc, %bcc]
        strides[%ldb, %c1]
        {direction = "out", access = 2 : i64,
         mapping = #pim.place_mapping<axis_sources = ["m",
                                                      "n"]>} : memref<*xf32>

    scf.for %tx = %c0 to %tripsx step %c1 {
      %gx = arith.addi %startx, %tx {pim.device_origin} : index
      %m = arith.muli %gx, %c32 : index
      scf.for %ty = %c0 to %tripsy step %c1 {
        %gy = arith.addi %starty, %ty {pim.device_origin} : index
        %n = arith.muli %gy, %c32 : index
        %e = tensor.empty() : tensor<32x32xf32>
        %init = linalg.fill ins(%f0 : f32) outs(%e : tensor<32x32xf32>) -> tensor<32x32xf32>
        %r = scf.for %tk = %c0 to %tripsk step %c1
             iter_args(%acc = %init) -> (tensor<32x32xf32>) {
          %k = arith.muli %tk, %c32 : index
          %da = pim.access %A offset %m sizes [32, 32] strides[%lda, %c1] origin[%m, %k]
                {id = 0 : i64} : memref<*xf32> -> !pim.access
          %ta = pim.load %da : !pim.access -> tensor<32x32xf32>
          %db = pim.access %B offset %k sizes [32, 32] strides[%ldb, %c1] origin[%k, %n]
                {id = 1 : i64} : memref<*xf32> -> !pim.access
          %tb = pim.load %db : !pim.access -> tensor<32x32xf32>
          %mm = linalg.matmul ins(%ta, %tb : tensor<32x32xf32>, tensor<32x32xf32>)
                outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
          scf.yield %mm : tensor<32x32xf32>
        }
        %dc = pim.access %C offset %n sizes [32, 32] strides[%ldb, %c1] origin[%m, %n]
              {id = 2 : i64} : memref<*xf32> -> !pim.access
        pim.store %r, %dc : tensor<32x32xf32>, !pim.access
      } {pim.axis = #pim.axis<"n">}
    } {pim.axis = #pim.axis<"m">}
  }
  return
}
}
