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
    치환하지 않고 각 op에 pim.op 종류 태그만 단다.
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

def PIMRemoveRedundantTransfers : Pass<"pim-remove-redundant-transfers", "mlir::ModuleOp"> {
  let summary = "Remove redundant PIM transfers on value semantics";
  let description = [{
    pim.load(Read)/store(Write)/access(Pure) 효과 위에서 중복 전송을 없앤다.
    - hoisting: loop 안에서 매번 같은 영역을 읽는 pim.load를 loop 앞으로 옮긴다
    - 읽은 값을 그대로 쓰는 경우: pim.store(pim.load(%ds), %dd) → pim.copy(%ds, %dd)
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
  let constructor = "mlir::triton::createPIMRemoveRedundantTransfersPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

def PIMAnalyzeReuse : Pass<"pim-analyze-reuse", "mlir::ModuleOp"> {
  let summary = "Annotate grid-axis dependence, reuse, and splittable axes";
  let description = [{
    grid 축과 각 pim.access의 의존 관계를 분석해 속성으로 단다(치환 아님):
    - func: `pim.execution_model = "spmd"`, grid loop가 있으면 그 중첩 순서대로
      `pim.grid_dims = ["x"(,"y","z")]`.
    - pim.access: 소비자에 따라 `pim.role`(load/copy-src→input, store/copy-dst→output),
      그리고 의존하지 **않는** grid 축을 `pim.invariant_grid_dims`(grid_dims 기준 인덱스)로 단다.
      invariant = 그 축이 바뀌어도 같은 데이터다 → 쪼갤 때 나누지 않고 복제한다.
    - pim.access: `pim.reuse_realizable` = 주소가 grid 아닌 loop 안에서 변하지 않는가.
      거짓이면 pim.access가 그 loop 안에서 만들어져 전송을 밖으로 뺄 수 없고, 불변 축이
      약속하는 재사용이 loop 순서를 바꾸지 않는 한 실현되지 않는다.
    - scf.for: 누산기(tensor iter_arg)를 계산 op의 outs로 받아 누적하면
      `pim.reduction_combine`으로 결합 연산자를 단다.
    - func: `pim.split_axes` = 쪼갤 수 있는 축 목록. 항목마다 axis · source 필드가 있고,
      쪼개면 부분 결과가 생기는 축에는 combine 필드가 붙는다. source는 grid 축이면 "grid",
      instance 내부 reduction loop면 "reduction"이다. pim.scatter가 있으면 서로 다른 유닛이
      같은 원소에 쓰지 않는다고 보장할 수 없어 grid 축을 올리지 않고, reduction loop는
      몸체에 pim.store/pim.scatter가 있으면 올리지 않는다. 결합 연산자를 못 정하는 축도
      올리지 않는다 — 모르는 채 쪼개면 target이 덮어쓰는 코드를 만든다.
    grid 축은 `pim.grid_dim` 속성이 붙은 scf.for로 식별하므로 --pim-build-grid를 먼저 돌려야 한다.
    의존성은 backward slice로 구하며 arith 체인과 scf.for iter_arg를 따라간다. 추적 불가한 leaf를
    만나면 안전하게 모든 축에 의존한다고 본다(재사용 과대주장 금지).
    영역을 정하는 것은 offset 하나가 아니므로 strides·valid·index·pad까지 함께 추적한다.
    실제 residency는 target이 결정한다.
  }];
  let constructor = "mlir::triton::createPIMAnalyzeReusePass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

def PIMAnalyzeTransfer : Pass<"pim-analyze-transfer", "mlir::ModuleOp"> {
  let summary = "Annotate the terms of the transfer, residency, and compute cost";
  let description = [{
    분할 비용 식에 들어갈 항을 계산해 속성으로 단다(치환 아님). 항을 곱해 시간으로 바꾸는
    것은 대역폭·유닛 수·용량을 아는 Target IR이다.
    - pim.access: `pim.tile_bytes` = 한 번에 옮기는 바이트(sizes 곱 × element bit width,
      올림). valid는 동적 값이라 못 쓰므로 sizes 전체를 세는 상한이다.
      `pim.transfer_count` = instance 하나 안에서 전송되는 횟수. 전송 op을 감싸는 비-grid
      loop 반복 횟수의 합이며, access op이 아니라 load/store/copy의 위치를 본다 —
      hoisting이 둘을 떼어 놓을 수 있다.
      둘을 곱해 두지 않는 이유: 곱하면 전송 건수를 복원할 수 없고, 상주량 계산에 tile
      바이트가 따로 필요하며, 감싸는 loop 반복이 비상수여도 tile 바이트는 남길 수 있다.
    - func: `pim.co_resident` = 한 순간에 유닛 메모리에 함께 있어야 하는 tile 목록.
      항목마다 access · bytes · lifetime 세 필드다. 가장 깊이 있는 pim.load의 블록에서
      그 블록의 load 결과 tensor들(streaming) + (그 블록이 scf.for 몸체면) tensor 타입
      iter_arg(persistent). access는 함수 안 pim.access의 walk 순서 인덱스다(누산기엔 없다).
      합계가 아니라 목록인 이유는 target의 메모리 계층이 하나가 아니어서 어느 tile을
      빠른 쪽에 둘지 정하려면 개별 크기가 필요하기 때문이다.
    - func: `pim.compute_ops` = instance 하나의 산술 연산 횟수. 계산 op마다
      (반복 공간 크기) × (원소당 산술 개수) × (감싸는 비-grid loop 반복 횟수)를 더한다.
      전송량은 유닛 수에 비례해 늘지만 유닛당 연산량은 반비례하므로, 이 항이 없으면
      분할의 이득이 비용 식에 나타나지 않는다.
    - func: `pim.grid_extents` = grid loop 반복 횟수. 상수가 아니면 붙이지 않는다.
      근거 없는 수를 IR에 넣지 않는다. 나머지 값은 grid 크기와 무관하므로 그때도 붙는다.
    전송량은 축 d의 grid 크기 g_d와 유닛 수 s_d에 대해
      bytes(s) = Σ_a tile_bytes_a × transfer_count_a × Π{s_j: j ∈ I_a} × Π{g_j: j ∉ I_a}
    이고 I_a는 pim.invariant_grid_dims다. s가 전부 1일 때가 전송량 하한(= 실제로 필요한
    데이터양)이다.
    --pim-remove-redundant-transfers(hoisting이 끝나야 전송 횟수가 실제와 맞는다) 뒤에
    돌려야 한다.
  }];
  let constructor = "mlir::triton::createPIMAnalyzeTransferPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

def PIMClassifyCompute : Pass<"pim-classify-compute", "mlir::ModuleOp"> {
  let summary = "Tag linalg compute ops with their target-neutral kind (pim.op)";
  let description = [{
    각 계산 op에 target 무관한 종류를 `pim.op`로 단다(치환 아님):
    matmul→gemm, matvec→gemv, transpose→transpose, reduce→reduce, fill→fill,
    generic→(reduction이면 reduce / linalg.index면 index_generation / 입력맵이 차원 drop이면
    broadcast / 아니면 elementwise). 합법성·opcode·분해는 태그하지 않는다(target 몫).
    분류 태그는 스냅샷이라 op을 바꾸는 패스(fusion 등) 뒤에 낡으므로, 이 패스는 모든 구조 변형 뒤
    **맨 마지막(late)** 에 돌린다.
  }];
  let constructor = "mlir::triton::createPIMClassifyComputePass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

#endif
