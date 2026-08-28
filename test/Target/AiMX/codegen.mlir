// AiMX IR이 runtime stub(C++)과 descriptor program(C 헤더)으로 나가는지 확인한다.
// 입력은 PIM Common IR이고 두 pass를 거쳐 온다 — 손으로 쓴 AiMX IR과 실제 pass 출력이
// 어긋나지 않도록.
// RUN: triton-shared-opt --convert-pim-to-aimx --aimx-lower-matmul %s | triton-shared-translate --aimx-to-descriptors | FileCheck %s --check-prefix=DESC
// RUN: triton-shared-opt --convert-pim-to-aimx --aimx-lower-matmul %s | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=STUB

// ── descriptor program ──────────────────────────────────────────────────────
// program 하나가 함수 하나다. buffer는 물리 주소로 받는다.
// DESC: #include "aimx_runtime.h"
// DESC: static uint32_t aimx_prog_matmul_relayout(pim_isa_32_t *code, uint32_t cap, uint32_t idx, uint64_t *next, int fpga_id
// 쓰기 전에 경계를 확인한다. tile 모양이 런타임 값이라 group 수도 런타임에 정해진다.
// DESC: aimx_require(idx + NUM_CHANNELS <= cap
// pitch는 물리 주소(33 bit) 바로 위에 얹는다. 쓰지 않는 쪽은 0x0이다.
// DESC-NEXT: PIM_RD_INSTR_MC(&idx, next, ({{.*}} + (uint64_t){{.*}}) | ((uint64_t){{.*}} << 33), 0x0, 512u, 0x59000u
// DESC: PIM_RD_INSTR_MC(&idx, next, 0x0, ({{.*}} + (uint64_t){{.*}}), 65536u, 0x1a400u
// DESC: static uint32_t aimx_prog_matmul_compute
// DESC: PIM_RD_INSTR_MC(&idx, next, ({{.*}} + (uint64_t){{.*}}), 0x0, 8192u, 0x19012u
// DESC: PIM_RD_INSTR_MC(&idx, next, ({{.*}} + (uint64_t){{.*}}) | ((uint64_t){{.*}} << 33), 0x0, 1024u, 0x59014u
// DESC: PIM_RD_INSTR_MC(&idx, next, ({{.*}} + (uint64_t){{.*}}) | ((uint64_t){{.*}} << 33), ({{.*}} + (uint64_t){{.*}}) | ((uint64_t){{.*}} << 33), 1024u, 0x58018u

// ── runtime stub ────────────────────────────────────────────────────────────
// 진입 함수 이름을 compiler.py가 이 정규식으로 읽어 metadata["name"]에 채운다.
// STUB: #include "aimx_descriptors.h"
// STUB: extern "C" void matmul(float* a0, float* a1, float* a2, int32_t a3, int32_t a4, int64_t a5, int64_t a6) {
// STUB:   int _ndev = init_pim_drv();
// STUB:   aimx_require(_ndev >= 0
// fpga_id 는 런타임이 pl_dma[id] 로 **경계 검사 없이** 인덱싱한다. 여기서 막지 않으면
// 배열 밖 fd 로 ioctl 이 나간다.
// STUB: aimx_require(AIMX_BASE_FPGA + 1 <= _ndev
// PIM Device 마다 host thread 하나다. device 가 하나여도 형태는 같다.
// STUB: #pragma omp parallel for collapse(2) num_threads(1)
// PMM이 먼저이고, 그다음 buffer마다 정렬과 PMM 겹침을 확인한다.
// STUB: pim_private_malloc(196608, _fpga, 7)
// STUB: aimx_require(({{.*}} % 262144ull) == 0
// STUB: pim_malloc(262144, _fpga)
// STUB: overlaps the PMM region
// chain은 host에서 만들어 한 번에 건너간다.
// STUB: aimx_prog_matmul_relayout((pim_isa_32_t*)
// STUB: aimx_prog_matmul_compute((pim_isa_32_t*)
// STUB: datacopy_cpu2pim(
// stage는 남는 자리를 0으로 채운다 — 그 0이 누산에 그대로 더해지므로 정확성 요구사항이다.
// STUB: memset(
// STUB: aimx_to_bf16(
// STUB: pim_exec(&_args, _fpga)
// unstage는 두 벌이 나온다. 대입이냐 누산이냐는 tile 전체에 대해 한 번 정해지고 원소마다가
// 아니다 — 안쪽 loop에 분기를 두면 벡터화가 통째로 사라진다. 누산 쪽이 먼저다.
// STUB: + aimx_from_bf16(
// STUB: = aimx_from_bf16(
// 참조 구현의 함수는 부르지 않는다.
// STUB-NOT: _matmul_silent
// STUB-NOT: matmul_tiled

module attributes {pim.target_profile = #pim.target_profile<
    target = "aimx", number_of_pim_devices = 1,
    number_of_banks_per_pim_device = 256,
    number_of_pim_cores_per_bank = 1,
    number_of_parallel_lanes_per_pim_core = 16,
    parallel_execution_kind = "simd",
    usable_bank_capacity_bytes = 268435456, usable_local_capacity_bytes = 0>} {

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
