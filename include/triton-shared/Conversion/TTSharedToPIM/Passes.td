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
    현재 지원: strided · loop-carried · block · masked(pad 있음/없음) 로드, row·element gather/scatter,
    tile 저장(materialize)·스칼라 저장(affine.store). wrap-around 로드는 미지원(건너뜀).
  }];
  let constructor = "mlir::triton::createConvertTTSharedToPIMPass()";
  // pim.access/load/store, (상수 offset/stride용) arith.constant, (스칼라 저장 감쌈용)
  // tensor.from_elements를 만드므로 의존 dialect로 선언한다. 입력에 그 dialect가 없어도 로드되게.
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::arith::ArithDialect",
                           "mlir::tensor::TensorDialect"];
}

def PIMBuildGrid : Pass<"pim-build-grid", "mlir::ModuleOp"> {
  let summary = "Turn SPMD program IDs into an explicit grid loop nest";
  let description = [{
    runtime가 돌던 grid 반복을 IR 안의 scf.for로 옮긴다. 그래야 서로 다른 program instance
    사이의 중복 전송을 IR에서 볼 수 있다.
    - program_id 인자 → grid loop의 induction variable. 인자는 지운다(뒤 패스가 끝 3개를
      계속 program ID로 보지 않도록).
    - num_programs 인자 → loop 상한. 그대로 남는다(호스트가 받는 grid 크기).
    - 각 loop에 `pim.grid_dim = "x"|"y"|"z"`. 이 속성이 grid loop와 instance 내부
      loop(K tile 등)를 구분하며, 이후 분석·분할이 전부 이 구분 위에서 돈다.
    triton-shared ABI(커널 인자 끝 6개 = num_programs(x,y,z) 다음 program_id(x,y,z))만 보고
    몸체는 건드리지 않는다. 그래서 커널이 grid를 어떻게 쓰든(grouped ordering 유무, GROUP_M 값)
    결과가 같다. 다만 1차원 grid + grouped ordering이면 축이 하나뿐이라 분할 후보가 안 나온다.
    실제로 쓰이는 program 축만 loop가 된다(안 쓰는 축은 범위 1). 단일 블록 func만 다룬다.
  }];
  let constructor = "mlir::triton::createPIMBuildGridPass()";
  // grid loop(scf.for)와 상한·유도변수 변환(arith.constant/index_cast)을 만든다.
  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::arith::ArithDialect"];
}

def RemovePIMBuffers : Pass<"pim-remove-buffers", "mlir::ModuleOp"> {
  let summary = "Remove redundant PIM transfers on value semantics (Stage B)";
  let description = [{
    pim.load(Read)/store(Write)/access(Pure) 효과 위에서 중복 전송을 없앤다.
    - hoisting: loop 안에서 매번 같은 영역을 읽는 pim.load를 loop 앞으로 옮긴다
    - passthrough: pim.store(pim.load(%ds), %dd) → pim.copy(%ds, %dd)
    - load 재사용: 같은 access의 뒤 pim.load → 앞 load 결과 / 앞 store 값으로 대체
    hoisting을 먼저 해야 loop 밖으로 나온 load들이 같은 블록에서 재사용 판정을 받는다.
    표준 -loop-invariant-code-motion은 pim.load가 MemRead라 옮기지 않으므로 여기서 직접 한다.
    안전 조건: 두 op 사이(같은 블록)에 쓰기 효과 op이 없어야 한다. hoisting은 loop 안에 그
    버퍼를 쓰는 op이 없어야 하며, base memref가 다르면 겹치지 않는다고 본다(가정 — Triton은
    포인터 인자가 서로 다른 버퍼임을 보장하지 않는다). scf.if 안의 load는 옮기지 않고,
    grid loop(pim.grid_dim)에서도 빼지 않는다(instance 사이 공유는 분할 구조가 정한다).
    "같은 access"는 SSA 동일성으로 보므로 이 pass 앞에 -cse를 돌려 Pure한 pim.access를 합쳐
    두는 것을, hoisting은 그 앞의 -loop-invariant-code-motion이 불변 pim.access를 loop 밖으로
    빼 둔 것을 전제한다. 죽은 pim.load(결과 미사용)은 -canonicalize가 정리한다.
  }];
  let constructor = "mlir::triton::createRemovePIMBuffersPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

def PIMAnalyzeLogicalWork : Pass<"pim-analyze-logical-work", "mlir::ModuleOp"> {
  let summary = "Annotate SPMD program-ID structure and per-access reuse on PIM Common IR";
  let description = [{
    grid 차원과 각 pim.access의 의존 관계를 분석해 속성으로 단다(치환 아님):
    - func: `pim.execution_model = "spmd"`, grid loop가 있으면 그 중첩 순서대로
      `pim.program_dims = ["x"(,"y","z")]`.
    - pim.access: 소비자에 따라 `pim.role`(load/copy-src→input, store/copy-dst→output),
      그리고 의존하지 **않는** grid 차원을 `pim.invariant_program_dims`(program_dims 기준 인덱스)로 단다.
      invariant = 그 차원이 바뀌어도 같은 데이터를 읽는다 → 그 방향 instance 사이 재사용 후보.
    - 출력 access가 어떤 차원에 불변이면 그 차원만 다른 instance들이 같은 출력에 쓴다는 뜻이라
      `pim.reduce_across_programs`로 표시한다(Split-K류). 어떤 access도 그 차원에 의존하지 않으면
      단순 중복 실행이라 결합 대상이 아니므로 표시하지 않는다.
    grid 차원은 `pim.grid_dim` 속성이 붙은 scf.for로 식별하므로 --pim-build-grid를 먼저 돌려야 한다.
    의존성은 backward slice로 구하며 arith 체인과 scf.for iter_arg를 따라간다. 추적 불가한 leaf를
    만나면 안전하게 모든 차원에 의존한다고 본다(재사용 과대주장 금지).
    영역을 정하는 것은 offset 하나가 아니므로 strides·valid·index·pad까지 함께 추적한다.
    실제 residency는 target이 결정한다.
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
