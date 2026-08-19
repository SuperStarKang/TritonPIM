#ifndef PIM_TO_UPMEM_CONVERSION_PASSES
#define PIM_TO_UPMEM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertPIMToUPMEM : Pass<"convert-pim-to-upmem", "mlir::ModuleOp"> {
  let summary = "Convert placed PIM IR into the UPMEM execution model";
  let description = [{
    `pim.device` + `pim.place` 를 UPMEM 의 host/device 구조로 옮긴다. PIM IR 이 정하지
    않은 것을 여기서 정한다: 메모리 계층, 전송 단위, 재배치.

    진입 즉시 legality gate 를 지난다 — target 이 upmem 이고, lane 이 독립 스레드이며,
    bank 용량이 기록돼 있어야 한다.

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
    - tasklet 이 여럿이면 device 가 맡은 타일을 tasklet 에 나눈다. 병렬 축 loop 들을 하나로
      합친 뒤 tasklet 번호부터 tasklet 수만큼 건너뛰며 돈다. 타일마다 쓰는 자리가 겹치지
      않으므로 barrier 가 없다. tile 버퍼는 tasklet 마다 따로 잡는다.

    시간(창) 분할은 공통 IR 의 `#pim.execution_tile` 에서 온다. 축 상한 하나가
    창 하나이고, 창의 tile 수 = 상한 / 그 축의 group tile 원소수다. host loop 은
    n → k → m 순서로 중첩되고, 창마다 [창 반복 횟수 갱신 + 그 level 이 가장 안쪽
    창인 입력의 재전송] 을 한 뒤 가장 안쪽에서 launch 와 회수를 한다 — B(k·n)는
    k level, A(m·k)는 m level 에서 재전송되고, 어느 창에도 매이지 않은 입력은
    처음 한 번만 전송된다. 배정이 상한 안이면 창 수가 런타임에 1 로 퇴화한다.
    k(reduction) 창은 launch 마다 K 의 한 구간만 누적한 부분합을 내므로 host 가
    창을 건너 누산한다(`upmem.unstage` 의 accumulate). 단 explicit_k_partition
    축이 있으면 k 상한은 공간 분할(SPLIT_K)의 눈금이라 창을 만들지 않는다.
    device 바이너리는 loop 상한이 host 변수로 오므로 창이 있어도 변하지 않는다.
  }];
  let constructor = "mlir::triton::createConvertPIMToUPMEMPass()";
  let options = [
    Option<"hostThreads", "host-threads", "int64_t", "8",
           "host가 영역을 모으고 되돌릴 때 쓰는 스레드 수. 1이면 순차.">,
  ];
  let dependentDialects = ["mlir::upmem::UPMEMDialect", "mlir::pim::PIMDialect",
                           "mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::func::FuncDialect", "mlir::memref::MemRefDialect",
                           "mlir::linalg::LinalgDialect", "mlir::tensor::TensorDialect"];
}

#endif
