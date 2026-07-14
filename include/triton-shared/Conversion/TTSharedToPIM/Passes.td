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

def RemovePIMBuffers : Pass<"pim-remove-buffers", "mlir::ModuleOp"> {
  let summary = "Remove redundant PIM transfers on value semantics (Stage B)";
  let description = [{
    pim.load(Read)/store(Write)/access(Pure) 효과 위에서 중복 전송을 없앤다.
    - passthrough: pim.store(pim.load(%ds), %dd) → pim.copy(%ds, %dd)
    - load 재사용: 같은 access의 뒤 pim.load → 앞 load 결과 / 앞 store 값으로 대체
    안전 조건: 두 op 사이(같은 블록)에 쓰기 효과 op이 없어야 한다. "같은 access"는 SSA 동일성으로
    보므로 이 pass 앞에 -cse를 돌려 Pure한 pim.access를 합쳐 두는 것을 전제한다. 죽은 pim.load
    (결과 미사용)은 -canonicalize가 정리한다.
  }];
  let constructor = "mlir::triton::createRemovePIMBuffersPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

#endif
