#ifndef TTSHARED_TO_PIM_CONVERSION_PASSES
#define TTSHARED_TO_PIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTTSharedToPIM : Pass<"convert-ttshared-to-pim", "mlir::ModuleOp"> {
  let summary = "Convert ttshared memref access into PIM Common (pim.access/load/store)";
  let description = [{
    --triton-to-linalg-experimental 출력의 메모리 접근 3중구조
    (reinterpret_cast + alloc + copy + to_tensor / materialize_in_destination)를
    pim.access(주소 명세) + pim.load/pim.store(전송)로 끌어올려 staging 버퍼를 없앤다.
    scf.for와 program_id 식은 그대로 둔다(SPMD, symbolic). 계산(linalg matmul/reduce/generic)은
    치환하지 않고 각 op에 pim.class 종류 태그만 단다.
    현재 지원: ① strided / ② loop-carried / store. (③ block · ④ wrap · ⑤ mask · ⑥⑦ gather는 추후)
  }];
  let constructor = "mlir::triton::createConvertTTSharedToPIMPass()";
  // pim.access/load/store와 (상수 offset/stride용) arith.constant를 만드므로 의존 dialect로 선언한다.
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::arith::ArithDialect"];
}

#endif
