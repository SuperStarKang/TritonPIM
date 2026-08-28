#ifndef PIM_TO_UPMEM_CONVERSION_PASSES
#define PIM_TO_UPMEM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertPIMToUPMEM : Pass<"convert-pim-to-upmem", "mlir::ModuleOp"> {
  let summary = "Convert PIM Common IR into the UPMEM execution model";
  let description = [{
    `pim.device` + `pim.place` 를 UPMEM 의 host/device 구조로 옮긴다. PIM IR 이 정하지
    않은 것을 여기서 정한다: 메모리 계층, 전송 단위, 재배치.

    진입 즉시 legality gate 를 지난다 — target 이 upmem 이고 UPMEM topology와 bank
    용량이 기록돼 있어야 한다.

    모듈 수준:
    - `upmem.mram` = device 의 MRAM 버퍼. pim.place 마다 하나다. 모든 차원이 창
      (#pim.execution_tile)이나 상수로 정적으로 묶이면 그 곱이 정확한 크기이고 합이
      예산에 드는지 컴파일 시점에 확인한다. 런타임 차원이 남으면 bank 예산을 버퍼
      수로 나눈 상한을 쓰고 host 코드가 전송 직전에 검사한다 — device 의
      __mram_noinit 은 상수 크기를 요구한다.
    - `upmem.host_var` = device 가 실제로 도는 횟수와 MRAM 안의 축 간격.

    host:
    - `upmem.acquire`: DPU 를 확보하고 device 프로그램을 적재한다. 런타임이 쥐고 있다가
      다시 주므로 짝이 되는 반납이 없다.
    - `upmem.foreach_unit`: device 마다 하는 일. `direction = "in"` place 는
      `upmem.stage` + `upmem.to_mram` 이 되고, replicated_over 는 stage 의 dedup_over 가
      된다. stage 의 pad_to 는 place 의 capacity 다 — push_xfer 가 모든 DPU 에 같은 바이트
      수를 요구하므로 실제 크기로는 보낼 수 없다.
    - `upmem.launch`
    - `direction = "out"` place 는 `upmem.gather` + `upmem.foreach_group` +
      `upmem.unstage` 가 된다.

    device(`upmem.device_func`):
    - 축 loop nest 를 그대로 옮기고 상한만 `upmem.get_host_var` 로 바꾼다.
    - 전역 좌표를 복원하던 자리(pim.device_origin)의 device 시작을 0 으로 바꾼다. 그러면
      pim.access 의 origin 이 device 안의 위치가 된다.
    - tile 은 WRAM 버퍼가 되고 계산은 그 위에서 memref 형태 linalg 로 돈다.
    - `pim.load`/`pim.store` 는 타일 하나를 옮기는 DMA 가 된다. 타일이 2048 byte 를 넘으면
      행 묶음으로 쪼갠다.
    - tasklet 이 여럿이면 target option `nr-tasklets`에 따라 block group의 행을 연속
      구간으로 나눠 협력 실행한다. 이 값은 profile의 lanes_per_core 이하이어야 한다.
      어느 operand가 m을 따르는지는
      연산 이름이 아니라 linalg indexing map으로 판정한다. 따라서 matmul의 A/C,
      matvec의 A/y, identity pointwise의 모든 operand가 같은 규칙으로 잘리고 B/x처럼
      m을 쓰지 않는 operand는 전체 tile을 공유한다.
    - compute leaf는 fill, matmul, matvec와 identity indexing map + all-parallel
      iterator를 가진 `linalg.generic`을 지원한다. generic의 scalar `arith` body는 하나의
      C loop로 내려가므로 elementwise fusion이 보존된다.

    M/N execution-tile 반복은 PIM Common IR의 `pim.scheduling_axis` loop를 그대로
    소비하며 target에서 다시 자르지 않는다. `upmem.acquire`는 그 loop 밖에 있어 같은
    DPU set과 device binary를 round 사이에 재사용하고, stage/launch/gather만 반복된다.
    `#pim.execution_tile`의 K 크기는 한 device 내부의 temporal reduction residency다.
    reduction 범위가 더 크면 K를 쓰는 입력만 window마다 다시 전송하고, launch가 낸
    부분합은 host가 `upmem.unstage accumulate`로 합친다. spatial `pim.axis = "k"`가
    있으면 각 값은 서로 다른 Execution Tile plane이며, 그 plane 내부에서만 temporal K
    window가 적용된다. spatial plane 사이의 최종 결합은 출력의 소비자가 담당한다.
    device 바이너리는 loop 상한이 host 변수로 오므로 창이 있어도 변하지 않는다.
  }];
  let constructor = "mlir::triton::createConvertPIMToUPMEMPass()";
  let options = [
    Option<"hostThreads", "host-threads", "int64_t", "8",
           "host가 영역을 모으고 되돌릴 때 쓰는 스레드 수. 1이면 순차.">,
    Option<"nrTasklets", "nr-tasklets", "int64_t", "0",
           "DPU마다 활성화할 tasklet 수. 0이면 profile의 lanes_per_core.">,
  ];
  let dependentDialects = ["mlir::upmem::UPMEMDialect", "mlir::pim::PIMDialect",
                           "mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::func::FuncDialect", "mlir::memref::MemRefDialect",
                           "mlir::linalg::LinalgDialect", "mlir::tensor::TensorDialect"];
}

#endif
