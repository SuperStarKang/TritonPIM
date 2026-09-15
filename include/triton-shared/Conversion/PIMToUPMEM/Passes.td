#ifndef PIM_TO_UPMEM_CONVERSION_PASSES
#define PIM_TO_UPMEM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertPIMToUPMEM : Pass<"convert-pim-to-upmem", "mlir::ModuleOp"> {
  let summary = "Convert PIM Common IR into the UPMEM execution model";
  let description = [{
    `pim.device` + `pim.place` 를 UPMEM 의 host/device 구조로 옮긴다. Common IR이
    확정한 Execution Tile 경계와 resident buffer 범위는 바꾸지 않고, UPMEM의
    MRAM/WRAM·DMA·tasklet 표현만 선택한다.

    진입 즉시 legality gate 를 지난다 — target 이 upmem 이고 UPMEM topology와 bank
    용량이 기록돼 있어야 한다.

    모듈 수준:
    - `upmem.mram` = device 의 MRAM 버퍼. `pim.place.allocation_shape`를 그대로
      사용해 place마다 하나를 만든다. 모든 차원은 Common IR에서 확정된 정적 fixed
      slot이어야 하며, 그 곱으로 정확한 크기를 만들고 합이 예산에 드는지 확인한다.
      runtime trip count나 bank 예산 균등 분할 fallback은 사용하지 않는다.
    - `upmem.host_var` = device 가 실제로 도는 횟수와 MRAM 안의 축 간격.

    host:
    - `upmem.acquire`: DPU 를 확보하고 device 프로그램을 적재한다. 런타임이 쥐고 있다가
      다시 주므로 짝이 되는 반납이 없다.
    - `upmem.foreach_unit`: device 마다 하는 일. `direction = "in"` place 는
      `upmem.stage` + `upmem.to_mram` 이 되고, replicated_over 는 stage 의 dedup_over 가
      된다. stage 의 pad_to와 result shape는 place의 정적 allocation_shape다 —
      push_xfer가 모든 DPU에 같은 바이트 수를 요구하므로 실제 tail size로는 보낼 수 없다.
    - `upmem.launch`
    - `direction = "out"` place 는 fixed allocation_shape의 `upmem.gather` +
      `upmem.foreach_group` + `upmem.unstage` 가 된다.

    device(`upmem.device_func`):
    - 축 loop nest 를 그대로 옮기고 상한만 `upmem.get_host_var` 로 바꾼다.
    - 전역 좌표를 복원하던 자리(pim.device_origin)의 device 시작을 0 으로 바꾼다. 그러면
      pim.access 의 origin 이 device 안의 위치가 된다.
    - tile 은 WRAM 버퍼가 되고 계산은 그 위에서 memref 형태 linalg 로 돈다.
    - `pim.load`와 `pim.store`/`pim.combine` partial write는 타일 하나를 옮기는
      DMA가 된다. 타일이 2048 byte를 넘으면 행 묶음으로 쪼갠다.
    - tasklet 이 여럿이면 target option `nr-tasklets`에 따라 block group의 행을 연속
      구간으로 나눠 협력 실행한다. 이 값은 profile의 lanes_per_core 이하이어야 한다.
      어느 operand가 m을 따르는지는
      연산 이름이 아니라 linalg indexing map으로 판정한다. 따라서 matmul의 A/C,
      matvec의 A/y, identity pointwise의 모든 operand가 같은 규칙으로 잘리고 B/x처럼
      m을 쓰지 않는 operand는 전체 tile을 공유한다.
    - compute leaf는 fill, matmul, matvec, reduce와 identity indexing map + all-parallel
      iterator를 가진 `linalg.generic`을 지원한다. generic과 reduce의 명시적인 scalar
      `arith` body를 그대로 C loop로 내리므로 elementwise fusion과 reduction 의미가
      보존된다. M축이 없는 K-only reduce는 tasklet-local partial combine이 IR에 표현되기
      전까지 `nr-tasklets=1`이어야 한다. M축이 있는 연산은 기존처럼 output 행을 tasklet에
      나눈다.

    모든 execution-tile 반복은 PIM Common IR의 `pim.scheduling_axis` loop를 그대로
    소비한다. 특히 temporal K도 Common의 `reduction_k` scheduling loop와
    `pim.temporal_reduction` local loop가 이미 round 수, local trip 수, 전역 K 시작을
    확정한다. 이 pass는 access dependency에서 reduction 축을 찾지 않고
    `#pim.execution_tile`로 window를 다시 계산하지 않는다.

    scheduling loop의 `pim.device` 한 번은 UPMEM의 stage → launch → gather 한 번이 된다.
    launch는 DPU set 전체에서 device program을 한 번 시작하는 것이며 Block Group마다
    발생하지 않는다. device-local Block Group loop 전체가 그 launch 한 번 안에서 돈다.
    각 stage/unstage 영역과 MRAM capacity는 그 round 안의 `pim.place`를 그대로 쓴다.
    Common `pim.combine ... kind "add"`가 명시된 출력만 `reduction_k` round와
    `device_k` 좌표를 `upmem.unstage accumulate`의 첫-partial 판정에 그대로
    사용한다. 최종 output의 rank와 `pim.place` 영역은 K에 의존하지 않으므로 target이
    partial plane을 제거하거나 destination을 다시 만들지 않는다. `upmem.acquire`는
    모든 Common scheduling loop 밖에 있어 같은 DPU set과 device binary를 round
    사이에 재사용한다. 별도 temporal host window는 만들지 않는다.

    `reuse-staging`은 host의 입력 저장 공간을 round/호출 사이에 재사용한다. 내용은
    매번 다시 채우며 input residency를 가정하지 않는다. 기본값은 false이며
    Common scheduling이나 Group 크기는 바꾸지 않는다.
  }];
  let constructor = "mlir::triton::createConvertPIMToUPMEMPass()";
  let options = [
    Option<"hostThreads", "host-threads", "int64_t", "8",
           "host가 영역을 모으고 되돌릴 때 쓰는 스레드 수. 1이면 순차.">,
    Option<"nrTasklets", "nr-tasklets", "int64_t", "0",
           "DPU마다 활성화할 tasklet 수. 0이면 profile의 lanes_per_core.">,
    Option<"reuseStaging", "reuse-staging", "bool", "false",
           "Reuse host input staging allocations across rounds and calls.">,
  ];
  let dependentDialects = ["mlir::upmem::UPMEMDialect", "mlir::pim::PIMDialect",
                           "mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::func::FuncDialect", "mlir::memref::MemRefDialect",
                           "mlir::linalg::LinalgDialect", "mlir::tensor::TensorDialect"];
}

#endif
