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

def PIMAnalyzeLogicalWork : Pass<"pim-analyze-logical-work", "mlir::ModuleOp"> {
  let summary = "Annotate SPMD program-ID structure and per-access reuse on PIM Common IR";
  let description = [{
    program ID(=SPMD tile 좌표)와 각 pim.access offset의 의존 관계를 분석해 속성으로 단다(치환 아님):
    - func: `pim.execution_model = "spmd"`, 그리고 실제로 쓰이는 program 축이 있으면
      `pim.program_dims = ["x"(,"y","z")]`.
    - pim.access: 소비자에 따라 `pim.role`(load/copy-src→input, store/copy-dst→output),
      그리고 offset이 의존하지 **않는** program 축을 `pim.invariant_program_dims`(program_dims 기준 인덱스)로 단다.
      invariant = 그 축이 바뀌어도 같은 데이터 영역을 접근 → 축 방향 program 사이 재사용 후보(§14).
    program ID는 triton-shared ABI(끝 3개 스칼라 인자 = axis 0/1/2)로 식별하며, offset backward slice를
    arith 체인·scf.for iter_arg/induction까지 따라간다. slice가 불투명(추적 불가)하면 안전하게 그 축에
    의존한다고 본다(재사용 과대주장 금지). 실제 residency는 target이 결정한다.
  }];
  let constructor = "mlir::triton::createPIMAnalyzeLogicalWorkPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

def PIMClassifyCompute : Pass<"pim-classify-compute", "mlir::ModuleOp"> {
  let summary = "Tag linalg compute ops with their target-neutral kind (pim.class)";
  let description = [{
    각 계산 op에 target 무관한 종류를 `pim.class`로 단다(치환 아님):
    matmul→gemm, matvec→gemv, transpose→transpose, reduce→reduce, fill→splat,
    generic→(reduction이면 reduce / linalg.index면 generation / 입력맵이 차원 drop이면 broadcast
    / 아니면 elementwise). 합법성·opcode·분해는 태그하지 않는다(target 몫).
    분류 태그는 스냅샷이라 op을 바꾸는 패스(fusion 등) 뒤에 낡으므로, 이 패스는 모든 구조 변형 뒤
    **맨 마지막(late)** 에 돌린다.
  }];
  let constructor = "mlir::triton::createPIMClassifyComputePass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

#endif
