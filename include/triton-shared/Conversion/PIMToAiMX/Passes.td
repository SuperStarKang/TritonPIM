#ifndef PIM_TO_AIMX_CONVERSION_PASSES
#define PIM_TO_AIMX_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertPIMToAiMX : Pass<"convert-pim-to-aimx", "mlir::ModuleOp"> {
  let summary = "Convert PIM Common IR into the AiMX host placement model";
  let description = [{
    `pim-form-host-plan`이 만든 `pim.partition`의 A/B 전송 → launch → C 회수 계획을
    AiMX level 1 IR(acquire / alloc / stage /
    build_descriptors / invoke / unstage)로 옮긴다. PIM IR 이 정하지 않은 것을 여기서
    정한다: target 물리 padding, PIM DRAM 안의 pitch layout, 전치 여부.
    중간 분석 IR인 `pim.device`나 지원하지 않는 전송 계획은 진단한다.

    진입 즉시 legality gate 를 지난다 — target 이 aimx 이고, unit 하나가 256 bank,
    bank 하나가 core 하나, core 하나가 16 lane 이어야 한다. AiMX는 profile의
    lanes_per_core 전체를 사용하며 batch에 맞는 bank 분해는 이 target lowering이 한다.

    **buffer 크기만 compile time 상수여야 하고, tile 모양은 아니다.** `pim_malloc` 은
    실패해도 NULL 을 돌려주지 않고 프로세스를 죽이므로 할당량은 compile time 에
    확정해야 한다. 그러나 `--aimx-lower-matmul` 이 만드는 descriptor program 은
    `t_m`·`t_k`·`padded_k`·`padded_n` 을 **함수 인자로 받는다** — compile time 상수로
    쓰는 것은 `batch` 하나뿐이다. 그래서 둘을 나눈다.

      k_max, n_max : buffer 를 잡는 크기. compile time. 공통 execution tile이 정한다
      t_k,   t_n   : descriptor program 이 실제로 도는 길이. 런타임. pim.place 에서 온다

    선택 (§6.1): 세 계층은 전부 공통 IR 이 정한다. `#pim.block`은 원소 shape,
    `#pim.block_group`은 Block 개수, `#pim.execution_tile`은 Group 개수를 저장한다.
    파생된 Group shape를 [batch, 4096/batch, 256/batch] 눈금으로 padding하고 buffer를
    잡는다. batch는 target lowering의 option이며 profile에는 들어가지 않는다.

      t_m   = group 수 × 물리 group 높이 — 전송량 식에서 t_m 는 A 의
              ceil(M/t_m)*t_m 항에만 나타나므로 작을수록 좋다
      n_max = group 수 × 물리 group 폭
      k_max = kUnit 수 × kUnit. allocation별 per-bank 예산 재검증을 넘으면 진단
      t_k   = min(256 * ceil(k / 256), k_max)                  (런타임)
      t_n   = min((4096/batch) * ceil(N / (4096/batch)), n_max)  (런타임)

    상한은 유효 원소 기준이고 buffer 는 물리 크기다 — block group 에 padding 이 있으면
    (block 이 실행 단위와 어긋날 때) host loop 은 유효 크기로 걷고 하드웨어는 물리
    크기를 돈다. padding 이 없으면 둘이 같다.

    n 의 올림 단위는 4096 이 아니라 **출력 block 폭 4096/batch** 다. 그것이 하드웨어가
    한 번에 소비하는 최소 단위이고, 그보다 거칠게 올리면 남는 열이 그대로 노는 lane 이
    된다 (N = 2560, batch = 8 이면 t_n 이 4096 이 아니라 2560 이다).

    host loop 은 **n → k → m** 이다. 가장 큰 buffer 인 B 가 `(k, n)` 에만 의존하므로 그
    둘을 바깥에 두고, 훨씬 작은 A 를 안쪽에서 m 마다 올린다. n 과 m 의 상대 순서는 §8
    I1(chain 을 재배치용과 계산용으로 분리)이 구현된 뒤에만 의미가 있다 — 지금은 chain
    하나에 둘 다 붙어 있어 invoke 마다 재배치가 다시 돈다.

    k Group 여러 개가 한 Execution Tile에 들어가면 reduction은 한 chain의 vACC 안에서
    끝난다. Execution Tile 경계로 k를 자르면
    부분합이 tile 마다 PIM 밖으로 나오므로 C 전송과 C drain descriptor 가 k tile 수만큼
    곱해지고 반올림도 그만큼 늘어난다. vACC 는 출력 block 하나 깊이(주소가
    `pmm + VACC_OFFSET` 하나)라 invoke 를 건너 들고 있을 수 없다. Common IR은
    `execution_tile.groups[k]`를 그대로 전달하며 이 pass가 예산을 최대한 채우도록
    값을 변경하지 않는다. 전체 reduction 범위가 그 크기보다 클 때 k loop이 두 번
    이상 돌고, 두 번째 tile부터는
    `aimx.unstage` 의 `accumulate` 가 1 이 되어 host 가 이전 부분합에 더한다 — host 가
    f32 면 그것이 fp32 누산이다.

    전치(C^T = B^T*A^T)는 아직 고르지 않는다. 유리한 쪽을 정하는 규칙이 `m` 과 `n` 의
    비교인데 둘 다 런타임 값이라 compile time 에 답이 없다. IR 은 이미 그것을 표현할 수
    있고(`aimx.stage` 의 `transpose`, `#aimx.tiling` 의 `transposed`) 이 pass 가 아직
    쓰지 않을 뿐이다.
  }];
  let constructor = "mlir::triton::createConvertPIMToAiMXPass()";
  let options = [
    Option<"batch", "batch", "int64_t", "8",
           "AiMX matmul batch. 1, 2, 4, 8 중 하나.">,
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
    (batch, t_m, t_n, t_k, padded_k, padded_n)은 전부 `#aimx.tiling` 에서 온다.
  }];
  let constructor = "mlir::triton::createAiMXLowerMatmulPass()";
  let dependentDialects = ["mlir::aimx::AiMXDialect", "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect", "mlir::func::FuncDialect"];
}

#endif
