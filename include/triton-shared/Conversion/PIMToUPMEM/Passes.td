#ifndef PIM_TO_UPMEM_CONVERSION_PASSES
#define PIM_TO_UPMEM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertPIMToUPMEM : Pass<"convert-pim-to-upmem", "mlir::ModuleOp"> {
  let summary = "Convert the unit form of PIM IR into the UPMEM execution model";
  let description = [{
    PIM IR의 유닛 형태(pim.units + pim.unit_data)를 UPMEM의 host/device 구조로 옮긴다.
    PIM IR이 정하지 않은 것을 여기서 정한다: 메모리 계층, 전송 단위, 재배치.

    모듈 수준:
    - `upmem.mram` = 유닛의 MRAM 버퍼. pim.unit_data마다 하나다. 크기는 유닛 메모리를
      버퍼 수로 나눈 값이다 — capacity가 런타임 값이라 컴파일 시점에 정확히 잡을 수
      없고, device의 __mram_noinit은 상수 크기를 요구하기 때문이다. 초기화하지 않는
      주소 예약이라 넉넉히 잡아도 적재 비용이 없다.
    - `upmem.host_var` = 유닛이 실제로 도는 횟수. 축마다 하나다.

    host(원래 func 몸체를 대신한다):
    - `upmem.acquire`: 유닛을 확보하고 device 프로그램을 적재한다. 커널을 부를 때마다
      다시 할 일이 아니라 런타임이 쥐고 있다가 다시 준다. 그래서 짝이 되는 반납이 없다.
    - `upmem.foreach_unit`: 유닛마다 doing 것. pim.units 몸체의 좌표 계산을 그대로
      옮겨 온다.
      - `direction = "in"` unit_data → `upmem.stage` + `upmem.to_mram`.
        stage의 pad_to는 unit_data의 capacity다. 유닛마다 같은 크기를 요구하는
        push_xfer 때문에 실제 크기(sizes)로는 보낼 수 없다.
        replicated_over는 stage의 dedup_over가 된다.
      - 축마다 `upmem.to_host_var`로 실제 반복 횟수를 보낸다. 마지막 유닛에서 작아지므로
        broadcast가 아니라 유닛별 전송이다.
    - `upmem.launch`
    - `direction = "out"` unit_data → `upmem.gather` + `upmem.foreach_group` +
      `upmem.unstage`. combine/combine_over가 있으면 그 축이 foreach_group의 축약 축이
      되고, 없으면 축약 없이 유닛 하나의 버퍼가 그대로 온다.

    device(`upmem.device_func`):
    - 축 loop nest를 그대로 옮기고 상한만 `upmem.get_host_var`로 바꾼다.
    - 전역 인덱스를 복원하던 자리(pim.unit_origin)의 유닛 시작을 0으로 바꾼다. 그러면
      pim.access의 origin이 유닛 안의 위치가 된다. Triton이 내는 `pid * BLOCK` 형태를
      전제한다 — 그 형태라야 유닛 시작에서 0이 되어 뺄셈 없이 성립한다.
    - tile은 WRAM 버퍼(`memref<..., #upmem.space<wram>>`)가 되고, 계산은 그 위에서
      memref 형태 linalg로 돈다. tensor iter_arg는 없어진다 — 버퍼가 loop를 관통하므로
      값을 나를 필요가 없다.
    - `pim.load`/`pim.store`는 **행마다** DMA 하나가 된다. MRAM DMA가 연속 영역만
      다루는데 타일의 행끼리는 축 간격만큼 떨어져 있어 한 번에 옮길 수 없다.
      MRAM 안의 위치는 access의 축별 시작에서 나온다:
        행 r = ((origin[0] + r) × pitch[1] + origin[1]) × pitch[2] + ...
      pitch는 unit_data의 capacity에서 오며 host가 `upmem.to_host_var`로 넣어 준다.
    - tasklet이 여럿이면(local-lanes > 1) 유닛이 맡은 타일을 tasklet에 나눈다.
      병렬 grid loop들을 하나로 합친 뒤 tasklet 번호부터 tasklet 수만큼 건너뛰며 돈다.
      합치는 이유는 loop 하나만 나누면 그 loop의 반복이 tasklet 수보다 적을 때 나머지
      tasklet이 놀기 때문이다. 타일마다 쓰는 자리가 겹치지 않으므로 barrier가 없다.
      대신 tile 버퍼는 tasklet마다 따로 잡는다 — 앞에 tasklet 축을 하나 더 두고
      자기 번호로 잘라 쓴다.
  }];
  let constructor = "mlir::triton::createConvertPIMToUPMEMPass()";
  let options = [
    Option<"unitMemoryBytes", "unit-memory-bytes", "int64_t", "67108864",
           "유닛 하나의 큰 메모리 바이트. MRAM 버퍼 크기를 여기서 나눈다.">,
    Option<"localLanes", "local-lanes", "int64_t", "1",
           "유닛 하나가 쓰는 tasklet 수. tile 버퍼를 이만큼 복제한다.">,
    Option<"localMemoryBytes", "local-memory-bytes", "int64_t", "49152",
           "유닛 하나의 빠른 메모리 바이트. tile 버퍼 합계가 넘으면 실패한다.">,
    Option<"hostThreads", "host-threads", "int64_t", "8",
           "host가 영역을 모으고 되돌릴 때 쓰는 스레드 수. 1이면 순차.">,
  ];
  let dependentDialects = ["mlir::upmem::UPMEMDialect", "mlir::pim::PIMDialect",
                           "mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::func::FuncDialect", "mlir::memref::MemRefDialect",
                           "mlir::linalg::LinalgDialect", "mlir::tensor::TensorDialect"];
}

#endif
