#ifndef PIM_TO_AIMX_CONVERSION_PASSES
#define PIM_TO_AIMX_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertPIMToAiMX : Pass<"convert-pim-to-aimx", "mlir::ModuleOp"> {
  let summary = "Convert PIM Common IR into the AiMX host placement model";
  let description = [{
    `pim.device` + `pim.place` 를 AiMX level 1 IR(acquire / alloc / stage /
    build_descriptors / invoke / unstage)로 옮긴다. PIM IR 이 정하지 않은 것을 여기서
    정한다: target 물리 padding, PIM DRAM 안의 pitch layout, 전치 여부.

    진입 즉시 legality gate 를 지난다 — target 이 aimx 이고, lane 이 SIMD 이며,
    Device 하나가 256 bank, bank 하나가 core 하나, core 하나가 16 lane 이어야 한다.
    한 block group은 `#pim.intra_device_plan`에 따라 이 계층 전체에 분해된다.

    **buffer 크기만 compile time 상수여야 하고, tile 모양은 아니다.** `pim_malloc` 은
    실패해도 NULL 을 돌려주지 않고 프로세스를 죽이므로 할당량은 compile time 에
    확정해야 한다. 그러나 `--aimx-lower-matmul` 이 만드는 descriptor program 은
    `m_t`·`k_t`·`padded_k`·`padded_n` 을 **함수 인자로 받는다** — compile time 상수로
    쓰는 것은 `batch` 하나뿐이다. 그래서 둘을 나눈다.

      k_max, n_max : buffer 를 잡는 크기. compile time. 공통 execution tile이 정한다
      k_t,   n_t   : descriptor program 이 실제로 도는 길이. 런타임. pim.place 에서 온다

    선택 (§6.1): 세 계층은 전부 공통 IR 이 정한다. block group 은 `#pim.block_group`
    (Triton block 정수 개를 묶은 target-neutral 단위), execution
    tile 은 `#pim.execution_tile` 이다 (원소 단위, 유효 데이터 기준 — 공통
    pim-form-execution-tiles 가 사용자 값 또는 예산 탐색으로 세 축 모두 구체값을
    확정한다. footprint 식은 AiMXFootprint.h 로 공유된다). batch 는 이 target
    lowering의 option이며 profile에는 들어가지 않는다. 이 pass는 block-group useful
    shape을 [batch, 4096/batch, 256/batch] 눈금으로 padding하고 buffer를 잡는다.

      m_t   = group 수 × 물리 group 높이 — 전송량 식에서 m_t 는 A 의
              ceil(M/m_t)*m_t 항에만 나타나므로 작을수록 좋다
      n_max = group 수 × 물리 group 폭
      k_max = kUnit 수 × kUnit. allocation별 per-bank 예산 재검증을 넘으면 진단
      k_t   = min(256 * ceil(k / 256), k_max)                  (런타임)
      n_t   = min((4096/batch) * ceil(N / (4096/batch)), n_max)  (런타임)

    상한은 유효 원소 기준이고 buffer 는 물리 크기다 — block group 에 padding 이 있으면
    (block 이 실행 단위와 어긋날 때) host loop 은 유효 크기로 걷고 하드웨어는 물리
    크기를 돈다. padding 이 없으면 둘이 같다.

    n 의 올림 단위는 4096 이 아니라 **출력 block 폭 4096/batch** 다. 그것이 하드웨어가
    한 번에 소비하는 최소 단위이고, 그보다 거칠게 올리면 남는 열이 그대로 노는 lane 이
    된다 (N = 2560, batch = 8 이면 n_t 가 4096 이 아니라 2560 이다).

    host loop 은 **n → k → m** 이다. 가장 큰 buffer 인 B 가 `(k, n)` 에만 의존하므로 그
    둘을 바깥에 두고, 훨씬 작은 A 를 안쪽에서 m 마다 올린다. n 과 m 의 상대 순서는 §8
    I1(chain 을 재배치용과 계산용으로 분리)이 구현된 뒤에만 의미가 있다 — 지금은 chain
    하나에 둘 다 붙어 있어 invoke 마다 재배치가 다시 돈다.

    **k 를 자르는 것은 손해이고 마지막 수단이다.** k tile 하나가 통째로 한 chain 에
    들어가면 reduction 이 vACC 안에서 끝나고 bf16 반올림이 출력당 한 번뿐이다. 자르면
    부분합이 tile 마다 PIM 밖으로 나오므로 C 전송과 C drain descriptor 가 k tile 수만큼
    곱해지고 반올림도 그만큼 늘어난다. vACC 는 출력 block 하나 깊이(주소가
    `pmm + VACC_OFFSET` 하나)라 invoke 를 건너 들고 있을 수 없어 이것은 피할 수 있는
    비용이 아니다.

    그래서 k 상한이 없으면 남는 예산 전부가 k_max 가 되고, `k > k_max` 일
    때만 k loop 이 두 번 이상 돈다. 그때 두 번째 tile 부터는
    `aimx.unstage` 의 `accumulate` 가 1 이 되어 host 가 이전 부분합에 더한다 — host 가
    f32 면 그것이 fp32 누산이다. 현실적인 k 는 전부 한 번에 들어가므로 이 경로는
    쓰이지 않는다.

    전치(C^T = B^T*A^T)는 아직 고르지 않는다. 유리한 쪽을 정하는 규칙이 `m` 과 `n` 의
    비교인데 둘 다 런타임 값이라 compile time 에 답이 없다. IR 은 이미 그것을 표현할 수
    있고(`aimx.stage` 의 `transpose`, `#aimx.tiling` 의 `transposed`) 이 pass 가 아직
    쓰지 않을 뿐이다.
  }];
  let constructor = "mlir::triton::createConvertPIMToAiMXPass()";
  let options = [
    Option<"batch", "batch", "int64_t", "8",
           "AiMX matmul batch. 1, 2, 4, 8 중 하나이며 intra-device plan과 일치해야 한다.">,
    Option<"bankCapacityBytes", "bank-capacity-bytes", "int64_t", "-1",
           "bank 당 usable PIM DRAM 예산. 음수면 profile 의 "
           "usable_bank_capacity_bytes 를 쓴다.">,
  ];
  let dependentDialects = ["mlir::aimx::AiMXDialect", "mlir::pim::PIMDialect",
                           "mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::func::FuncDialect", "mlir::memref::MemRefDialect",
                           "mlir::linalg::LinalgDialect", "mlir::tensor::TensorDialect"];
}

def AiMXLowerMatmul : Pass<"aimx-lower-matmul", "mlir::ModuleOp"> {
  let summary = "Expand the AiMX matmul descriptor programs";
  let description = [{
    `--convert-pim-to-aimx` 가 func 에 붙인 `#aimx.tiling` 을 읽어
    `aimx.descriptor_program` 둘을 module 수준에 만든다.

    - `@<kernel>_relayout` : A 를 vecA 가 원하는 순서로 다시 깐다. vecA 는 "행 방향으로
      늘어선 A" 가 아니라 "reduction 방향으로 늘어선 A" 를 원한다.
    - `@<kernel>_compute`  : A operand / B operand / C drain descriptor.

    주소식은 참조 구현(`_matmul_silent`)의 offset 식 그대로이고 전부 byte 단위다.
    이 pass 가 정하는 것은 descriptor 의 순서와 주소식뿐이며, 그것을 정하는 값
    (batch, m_t, k_t, n_t, padded_k, padded_n)은 전부 `#aimx.tiling` 에서 온다.
  }];
  let constructor = "mlir::triton::createAiMXLowerMatmulPass()";
  let dependentDialects = ["mlir::aimx::AiMXDialect", "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect", "mlir::func::FuncDialect"];
}

#endif
