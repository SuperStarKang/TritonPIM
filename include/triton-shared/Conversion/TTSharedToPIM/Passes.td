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

def PIMNormalizeAccumulator : Pass<"pim-normalize-accumulator", "mlir::ModuleOp"> {
  let summary = "Accumulate into the accumulator instead of into a zero-filled temporary";
  let description = [{
    `matmul(outs=0으로 채운 tile)` + `generic(add)(누산기, 그 결과)`를
    `matmul(outs=누산기)`로 접는다. 0에 더한 것을 누산기에 더하는 것과 누산기에 바로
    더하는 것이 같으므로 결과가 바뀌지 않는다.
    임시 tile 하나와 tile 크기만큼의 덧셈이 사라진다. 유닛 메모리가 작을수록 크게 영향을
    주며, 접지 않으면 pim.co_resident가 실제 필요량보다 tile 하나만큼 작게 나온다
    (임시 tile은 pim.load 결과가 아니라 목록에 안 들어간다).
    조건: generic이 두 입력을 그대로 더하기만 하고(축 전부 parallel, map 전부 항등),
    outs가 그 누산기이고, 계산 결과가 그 generic에서만 쓰이고, 계산의 outs가 0이어야 한다.
    하나라도 어긋나면 그대로 둔다.
    이어서 누산기 초기값을 만드는 op을 그 loop 바로 앞으로 내린다. tensor는 값 의미라
    -loop-invariant-code-motion이 grid loop 밖까지 빼 두는데, 값으로는 맞아도 버퍼로
    내리면 초기화가 한 번만 돌아 두 번째 instance부터 앞 결과가 남는다. 접은 뒤라야
    초기값이 iter_args에서만 쓰여 옮길 수 있다.
    구조를 바꾸므로 분석 패스들보다 앞에 둔다 — 접은 뒤라야 상주량과 연산량이 실제와 맞는다.
    --pim-partition보다도 앞이라야 초기값이 유닛 region 안에 들어간다.
  }];
  let constructor = "mlir::triton::createPIMNormalizeAccumulatorPass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def PIMAnalyzeReuse : Pass<"pim-analyze-reuse", "mlir::ModuleOp"> {
  let summary = "Annotate grid-axis dependence, reuse, and splittable axes";
  let description = [{
    grid 축과 각 pim.access의 의존 관계를 분석해 속성으로 단다(치환 아님):
    - func: `pim.execution_model = "spmd"`, grid loop가 있으면 그 중첩 순서대로
      `pim.grid_dims = ["x"(,"y","z")]`.
    - pim.access: `pim.access_id` = 이 함수 안에서 이 access를 가리키는 이름. 뒤 패스가
      자리 번호 대신 이 값으로 참조한다 — 자리 번호는 access가 하나 늘거나 줄면 어긋난다.
    - pim.access: 소비자에 따라 `pim.role`(load/copy-src→input, store/copy-dst→output),
      그리고 의존하지 **않는** grid 축을 `pim.invariant_grid_dims`(grid_dims 기준 인덱스)로 단다.
      invariant = 그 축이 바뀌어도 같은 데이터다 → 쪼갤 때 나누지 않고 복제한다.
    - scf.for: 누산기(tensor iter_arg)를 계산 op의 outs로 받아 누적하면
      `pim.reduction_combine`으로 결합 연산자를 단다.
    - func: `pim.split_axes` = 쪼갤 수 있는 축 목록. 항목마다 axis · source 필드가 있고,
      쪼개면 부분 결과가 생기는 축에는 combine 필드가 붙는다. source는 grid 축이면 "grid",
      instance 내부 reduction loop면 "reduction"이다. pim.scatter가 있으면 서로 다른 유닛이
      같은 원소에 쓰지 않는다고 보장할 수 없어 grid 축을 올리지 않고, reduction loop는
      몸체에 pim.store/pim.scatter가 있으면 올리지 않는다. 결합 연산자를 못 정하는 축도
      올리지 않는다 — 모르는 채 쪼개면 target이 덮어쓰는 코드를 만든다.
      grid 축에 결합이 필요한지는 출력에 **저장되는 값**이 그 축에 의존하는지로 정한다.
      위치가 그 축에 불변인 것만으로는 부족하다 — 값도 불변이면 모든 instance가 같은 것을
      쓰는 중복 실행이라 합치면 결과가 유닛 수만큼 커진다. 값을 추적하지 못한 축은
      어느 쪽인지 모르므로 아예 올리지 않는다.
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
  let summary = "Annotate transfer size and per-unit residency";
  let description = [{
    전송 한 번의 크기와 유닛 하나가 동시에 들고 있어야 하는 양을 속성으로 단다(치환 아님).
    - pim.access: `pim.tile_bytes` = 한 번에 옮기는 바이트(sizes 곱 × element bit width,
      올림). valid는 동적 값이라 못 쓰므로 sizes 전체를 세는 상한이다.
    - func: `pim.co_resident` = 한 순간에 유닛 메모리에 함께 있어야 하는 tile 목록.
      항목마다 access · bytes · lifetime 세 필드다. 가장 깊이 있는 전송의 블록을 잡아,
      그 블록에서 매 반복 새로 읽는 것(streaming)과 그 블록을 감싸는 loop를 관통하는
      것(persistent = 누산기 iter_arg, 그리고 밖에서 한 번 읽어 안에서 쓰는 tile)을 모은다.
      뒤엣것은 --pim-remove-redundant-transfers가 loop 밖으로 뺀 load다. 빠뜨리면 용량을
      실제보다 작게 보고 들어가지 않는 분할을 고른다.
      access는 pim.access_id이며, access를 거치지 않는 gather 결과와 누산기에는 없다.
      gather는 전송량을 못 세도(index 텐서 값에 달렸다) 상주량은 결과 tensor 타입에서
      나오므로 목록에 넣는다.
      합계가 아니라 목록인 이유는 target의 메모리 계층이 하나가 아니어서 어느 tile을
      빠른 쪽에 둘지 정하려면 개별 크기가 필요하기 때문이다.
    hoisting이 끝난 뒤라야 어느 tile이 어느 블록에 있는지가 확정되므로
    --pim-remove-redundant-transfers 뒤에 돌린다.

    비용 모델(전송 횟수·연산량·grid 크기)은 여기서 내지 않는다. 셋 다 감싸는 loop 반복
    횟수를 컴파일 시점 상수로 요구하는데, 실제 커널에서는 K loop 상한이 런타임 인자이고
    grid 상한은 num_programs 인자라 상수인 적이 없다. 값이 필요해지면(autotune) 반복
    횟수와 반복 1회분을 나눠서, 곱하는 쪽은 그 수를 아는 launch 계층에 두어야 한다.
  }];
  let constructor = "mlir::triton::createPIMAnalyzeTransferPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

def PIMPartition : Pass<"pim-partition", "mlir::ModuleOp"> {
  let summary = "Assign grid instances to units and derive each unit's data region";
  let description = [{
    반복 공간을 유닛 수만큼 나눈다. 여기까지가 target 무관이고, 이 결과를 어느 메모리에
    어떤 단위로 올릴지부터가 target 몫이다.
    - `pim.units`: 유닛 하나가 하는 일을 담는 region. 축 loop nest를 이 안으로 옮긴다.
      region 밖 = host(데이터 준비·배포·회수), region 안 = 유닛 하나. block argument는
      이 유닛의 좌표다. 속성이 아니라 region인 이유는 target이 무시할 수 없게 하기 위해서다.
    - 각 축 loop의 하한·상한을 그 유닛 몫으로 좁힌다. 유닛당 반복 수는 올림 나눗셈이고
      상한은 원래 상한으로 자르므로, grid 크기가 유닛 수로 나누어떨어지지 않아도 된다.
      타일 안의 나머지는 pim.access의 valid가 이미 담당한다.
    - `pim.unit_data`: 유닛 하나가 자기 메모리에 갖고 있어야 하는 영역. pim.access의
      origin에 그 유닛의 loop 하한을 대입해 시작 위치를, 그 유닛이 실제로 도는 횟수 ×
      타일 크기로 크기를 얻는다. 배정량이 아니라 실제 횟수를 쓰는 이유는, 반복이 유닛
      수로 나누어떨어지지 않을 때 마지막 유닛이 덜 맡기 때문이다 — 배정량을 쓰면 원본
      버퍼 밖을 가리킨다. origin이 없는 access(주소를 loop iter_arg로 나르는 형태)는
      건너뛴다. `access` 필드에 유도한 pim.access_id를 적어, 이 영역 안에서 타일이
      어디인지 target이 되짚을 수 있게 한다.
      입력이 어느 축에 불변이면 그 축만 다른 유닛들이 같은 데이터를 받으므로
      `replicated_over`로 적고, 출력은 쪼갠 reduction 축마다 부분 결과가 생기므로
      `combine`/`combine_over`로 적는다.
    - num-units / local-memory-bytes를 주면 요청한 유닛 수와 pim.co_resident 합계를
      확인한다. 0이면 확인하지 않는다.
    쪼갤 축과 결합 연산자는 --pim-analyze-reuse가 `pim.split_axes`로 내므로 그 뒤에 돌린다.
    splits는 그 목록 순서대로 축마다 하나씩 주며, 1이면 그 축은 쪼개지 않는다.
  }];
  let constructor = "mlir::triton::createPIMPartitionPass()";
  let options = [
    ListOption<"splits", "splits", "int64_t",
               "축별 유닛 수. pim.split_axes 순서대로 하나씩.">,
    Option<"numUnits", "num-units", "int64_t", "0",
           "기계의 유닛 개수. 0이면 확인하지 않는다.">,
    Option<"localMemoryBytes", "local-memory-bytes", "int64_t", "0",
           "유닛당 빠른 메모리 바이트. 0이면 확인하지 않는다.">,
  ];
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
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
