#ifndef TTSHARED_TO_PIM_CONVERSION_PASSES
#define TTSHARED_TO_PIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def PIMNormalizeAddress : Pass<"pim-normalize-address", "mlir::ModuleOp"> {
  let summary = "Put loop-carried address recurrences into closed form";
  let description = [{
    주소를 loop iter_arg 로 나르는 형태(`a_ptrs += BLOCK_K * stride_ak`)를
    `offset(i) = initial + i * advance` 로 다시 쓴다. 그래야 뒤의 배치 pass 가
    축별 시작 위치를 유닛 좌표로 되돌릴 수 있다.
    iteration = (유도변수 - 하한) / step 이고, yield 가
    `addi(<iter_arg>, advance)` 이며 advance 가 loop 밖에서 정의될 때만 성립한다.
    주소를 나르는 iter_arg 가 그 형태가 아니면 진단을 낸다.
  }];
  let constructor = "mlir::triton::createPIMNormalizeAddressPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::scf::SCFDialect"];
}

def ConvertTTSharedToPIM : Pass<"convert-ttshared-to-pim", "mlir::ModuleOp"> {
  let summary = "Convert ttshared memref access into PIM Common (pim.access/load/store)";
  let description = [{
    --triton-to-linalg-experimental 출력의 메모리 접근 관용구
    (reinterpret_cast + alloc + copy + to_tensor / materialize_in_destination)를
    pim.access(주소 명세) + pim.load/pim.store(전송)로 끌어올려 staging 버퍼를 없앤다.
    지원하는 것은 strided·loop-carried·block·masked(pad 있음/없음) 로드와 tile 저장이다.
    각 access 에 이 함수 안에서 유일한 id 를 매긴다.
    offset 을 축별 origin 으로 분해하지 못하면 origin 을 비워 두고
    --pim-recover-access-origin 이 채운다.
    변환 뒤 남은 memref/bufferization op 은 진단을 낸다 — 조용히 두면 배치 pass 가
    그 버퍼를 보지 못한 채 부분 배치를 만든다.
  }];
  let constructor = "mlir::triton::createConvertTTSharedToPIMPass()";
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::arith::ArithDialect"];
}

def PIMBuildLogicalGrid : Pass<"pim-build-logical-grid", "mlir::ModuleOp"> {
  let summary = "Turn SPMD program IDs into a logical tile loop nest";
  let description = [{
    source grid 축을 logical tile 좌표로 되돌려 scf.for nest 를 세운다. source grid 축은
    사용자가 launch 편의로 고른 것이라 데이터가 어떻게 쪼개지는지를 말하지 않는다.
    - 2차원 이상 grid: 축이 그대로 logical 축이다.
    - `pid / d` 와 `pid % d` 쌍: 한 축이 두 logical 축으로 갈라진다.
    - GROUP_M 순서: 그룹 좌표 사슬을 (m, n) 로 되돌린다.
      순서 자체는 배치에 영향이 없으므로 사슬은 통째로 죽는다.
    복원된 logical 축은 output tile 좌표인 m 과 n 뿐이며 각 loop에 `pim.axis`로
    이름을 단다. source program_id가 별도의 split-K plane을 이미 만들었다면
    진단한다. K는 이 pass가 source grid에서 만들지 않고, `pim-form-compute`가
    linalg accumulator 의미에서 찾은 하나의 `pim.reduction` domain으로 남긴다.
    복원할 수 없는 형태는 진단을 낸다.
  }];
  let constructor = "mlir::triton::createPIMBuildLogicalGridPass()";
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def PIMRecoverAccessOrigin : Pass<"pim-recover-access-origin", "mlir::ModuleOp"> {
  let summary = "Fill in the per-axis origin of each pim.access";
  let description = [{
    offset 의 각 항을 축에 배정해 `offset = Σ_d origin[d]·strides[d]` 를 복원한다.
    항을 좌표 쪽과 간격 쪽으로 가르고, 간격 쪽을 index_cast 와 상수 곱을 벗겨 낸
    기준 값으로 access 의 strides 와 대조한다. SSA 동일성으로 대조하면 같은 간격을
    다른 식으로 쓴 항이 어긋난다.
    어느 축과도 맞지 않는 간격이면 크기 1 의 축을 앞에 새로 만든다.
  }];
  let constructor = "mlir::triton::createPIMRecoverAccessOriginPass()";
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::arith::ArithDialect"];
}

def PIMFormCompute : Pass<"pim-form-compute", "mlir::ModuleOp"> {
  let summary = "Accumulate into the accumulator instead of into a zero-filled temporary";
  let description = [{
    `matmul(outs=0으로 채운 tile)` + `generic(add)(누산기, 그 결과)` 를
    `matmul(outs=누산기)` 로 접는다. 임시 tile 하나와 tile 크기만큼의 덧셈이 사라지고,
    상주량이 실제와 맞게 된다.
    이어서 linalg scalar body의 add가 loop-carried accumulator를 직접 갱신하고 같은
    iter_arg 위치로 yield되는지를 읽어 해당 loop에 `pim.reduction = "add"`를 기록한다.
    add의 반대 operand는 이전 accumulator에 의존하지 않는 독립 contribution이어야
    하며, 정규화된 reduction loop는 정확히 하나의 accumulator/result만 가진다.
    op 이름을 열거하지 않으므로 같은 계약의 matmul, matvec, reduce, convolution,
    generic에 공통으로 적용된다. access origin은 주소 정보이므로 reduction 판정에
    사용하지 않는다.
    이어서 누산기 초기값을 만드는 op 을 그 loop 바로 앞으로 내린다. 값 의미로는 밖에
    있어도 맞지만 버퍼로 내리면 초기화가 한 번만 돌아 두 번째 instance 부터 앞 결과가 남는다.
  }];
  let constructor = "mlir::triton::createPIMFormComputePass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def PIMSetTargetProfile : Pass<"pim-set-target-profile", "mlir::ModuleOp"> {
  let summary = "Attach the common machine profile to the module";
  let description = [{
    `#pim.target_profile` 을 module 에 붙인다. 이 pass 앞의 어떤 pass 도 profile 을 읽지
    않고, 뒤의 pass 는 전부 이 attribute 에서만 기계 값을 얻는다.
    topology와 capacity 상한만 공통 정보로 기록한다. capacity 0은 해당 저장소가 없거나
    공통 단계에서 그 상한을 검사하지 않는다는 뜻이다.
  }];
  let constructor = "mlir::triton::createPIMSetTargetProfilePass()";
  let options = [
    Option<"target", "target", "std::string", "\"upmem\"", "target 이름.">,
    Option<"numUnits", "num-units", "int64_t", "2048",
           "PIM 실행 unit 개수.">,
    Option<"banksPerUnit", "banks-per-unit", "int64_t", "1",
           "PIM 실행 unit 하나의 bank 수.">,
    Option<"coresPerBank", "cores-per-bank", "int64_t", "1",
           "bank 하나의 PIM Core 수.">,
    Option<"lanesPerCore", "lanes-per-core", "int64_t", "16",
           "PIM Core 하나의 병렬 실행 lane 능력값.">,
    Option<"bankCapacityLimitBytes", "bank-capacity-limit-bytes", "int64_t",
           "62914560", "bank 하나가 담을 수 있는 최대 byte 수.">,
    Option<"localCapacityLimitBytes", "local-capacity-limit-bytes", "int64_t",
           "49152", "PIM Core local storage의 최대 byte 수.">,
  ];
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

def PIMFormBlockGroups : Pass<"pim-form-block-groups", "mlir::ModuleOp"> {
  let summary = "Bundle neighbouring logical tiles into one block group";
  let description = [{
    이웃한 logical tile 을 출력 축 방향으로 묶어 큰 tile 하나처럼 만든다. 축 loop 는
    group 좌표가 되고(상한 = ceildiv(tiles, g)), 몸체 첫머리에서 첫 tile 좌표(iv·g)를
    복원해 기존 origin 산술에 잇는다. pim.access 의 sizes 와 tile tensor 는 group
    크기로 커진다. 묶음의 경계는 언제나 tile 경계 위다 — tile 을 가르지 않는다.

    왜 묶는가: 기계의 상주 단위(local storage 에 한 번에 올라가는 working set)는
    Triton BLOCK 과 무관하게 크고, 묶인 tile 들은 같은 입력 조각을 공유하므로
    (matmul 의 행 방향 이웃은 B 를, 열 방향 이웃은 A 를) 상주 중 재사용이 늘어난다.
    몇 개를 묶을지는 target 의 정책이라 옵션으로 받고, 기본 1 이면 항등이다.

    마지막 group이 부분이면 기존 valid 산술에서 원본 extent를 복원하고
    `min(max(extent - group_origin, 0), group_size)`로 다시 만든다. 따라서
    pim.access의 sizes, valid, tensor shape가 모두 같은 Group을 표현하며 pad 값은
    확대된 invalid tail에도 그대로 적용된다. pim-place-data도 같은 extent 복원
    함수를 사용해 resident transfer 범위를 자른다.
    valid가 없는 access는 해당 loop의 Block 수가 Group 크기로 정적으로 나누어질 때만
    확대할 수 있다. runtime Block 수 또는 나머지가 있는 정적 Block 수에는 원본 extent를
    복원할 dynamic valid가 필요하며, 그렇지 않으면 phantom Block 접근을 진단한다.

    group-m/n/k는 각 축에서 묶을 Block 개수다. 양수면 그대로 쓰고, 0이면
    기본값인 Block 하나를 쓴다. 이 pass는 비용식이나 profile 탐색을 수행하지 않는다.

    결과는 target-neutral `#pim.block<shape>`와
    `#pim.block_group<blocks>`로 func에 붙는다. Group 원소 shape는 두 값의 곱으로
    파생하며 중복 저장하지 않는다. 하드웨어 실행 눈금에 대한 padding은
    target lowering이 target IR에 materialize한다.
  }];
  let options = [
    Option<"groupM", "group-m", "int64_t", "0",
           "m 방향으로 묶는 Block 수. 0 또는 1 = 항등">,
    Option<"groupN", "group-n", "int64_t", "0",
           "n 방향으로 묶는 Block 수. 0 또는 1 = 항등">,
    Option<"groupK", "group-k", "int64_t", "0",
           "reduction 방향으로 묶는 Block 수. 0 또는 1 = 항등. 출력 축이 아니라 "
           "reduction loop 를 넓힌다 — 누산기는 그대로고 A·B strip 의 K 변만 "
           "커져서, local 적재 한 번이 담는 K 조각이 길어지고 DMA 준비·동기화 "
           "횟수가 준다.">,
  ];
  let constructor = "mlir::triton::createPIMFormBlockGroupsPass()";
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PIMMapDevices : Pass<"pim-map-devices", "mlir::ModuleOp"> {
  let summary = "Schedule execution tiles over PIM Devices and build the pim.device region";
  let description = [{
    `#pim.execution_tile` 이 먼저 결정된 뒤 logical 축마다 동시에 사용할 PIM Device
    수 `[p_m,p_n,p_k]`를 받아 loop nest 를 `pim.device` region 으로 감싼다. Execution Tile은 device
    하나의 resident 실행 단위이며, Device Grid의 각 좌표는 서로 다른 Execution Tile
    instance를 맡는다. m/n은 `pim.axis` output domain이고, K는 하나의
    `pim.reduction` semantic domain이다. source-level `pim.axis="k"`는 허용하지 않는다.
    `p_k`는 전체 semantic K Block Group을 동시에 몇 device에 나눌지 정하고,
    `execution_tile.groups[k]`는 device 하나가 round 한 번에 처리할 K Block Group 수다.
    없는 M/N 축과 semantic reduction이 없는 K의 partition은 반드시 1이다.
    Common IR은 비용식이나 자동 device 수 선택을
    수행하지 않는다. 형상을 아는 호출자나 autotuner가 정한 정수 세 개만 materialize한다.
    전체 block-group grid 를 execution-tile grid 로 먼저 자른 뒤 각 축에 대해
      execution_tile_index = round * active_devices + device_coordinate
      tile_begin = execution_tile_index * tile_groups
    로 배치한다. Execution Tile 수가 active device 수보다 많으면 같은 device set을
    여러 round에서 재사용한다. 마지막 round에 Tile이 없는 device는 0회 loop인 no-op이다.
    device 안에서 좌표를 0부터 세고, 전역 좌표는 몸체 첫머리에서 다시 만들어
    `pim.device_origin` 표시를 단다. round loop는 pim.device 바깥 host IR에 남는다.
    `pim.reduction`으로 명시된 loop는 다음 식으로 직접 나누고
      k_tiles = ceildiv(total_k_groups, execution_tile.groups[k])
      k_rounds = ceildiv(k_tiles, p_k)
      first_k_group = (round * p_k + device_k) * execution_tile.groups[k]
      local_k_groups = min(execution_tile.groups[k],
                           max(total_k_groups - first_k_group, 0))
    `pim.scheduling_axis = "reduction_k"` round를 device 바깥에 만든다. 따라서
    `pim.device` 한 번은 정확히 한 resident Execution Tile만 실행한다. 여러 temporal
    round와 K device가 만드는 partial contribution은
    `pim.combine partial into final_access`로 명시한다. 따라서 target은 명시된
    semantic K schedule, device mapping, combiner와 destination을
    소비하며 결합 범위나 연산을 access rank에서 다시 추론하지 않는다.

    M/N 축이 없는 K-only reduction에서는 zero-init tensor chain과 direct output
    access/store가 reduction loop의 형제다. 이 pass가 single-use init chain을
    reduction 앞으로, output access를 reduction 뒤로 같은 `pim.device`에 함께 옮긴
    다음 store를 partial-to-final `pim.combine`으로 바꾼다. 따라서 init은 Execution
    Tile마다 다시 실행되고 output도 host에 남지 않는다.
  }];
  let options = [
    Option<"pM", "p-m", "int64_t", "1", "m축 device partition 수.">,
    Option<"pN", "p-n", "int64_t", "1", "n축 device partition 수.">,
    Option<"pK", "p-k", "int64_t", "1", "k축 device partition 수.">,
  ];
  let constructor = "mlir::triton::createPIMMapDevicesPass()";
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PIMFormExecutionTiles : Pass<"pim-form-execution-tiles", "mlir::ModuleOp"> {
  let summary = "Choose the resident execution-tile shape of one PIM Device";
  let description = [{
    execution tile(PIM Device 하나가 한 번에 실행하는 resident Group domain)을
    `#pim.execution_tile` 로 임시로 func에 부착한다 — 데이터 3계층의 셋째 층을 공통
    IR에서 결정하는 pass다. pim-map-devices가 전체 block-group grid를 이 모양으로
    자르고 attribute를 `pim.device`로 옮긴 뒤 func의 중복 값은 지운다. target lowering은
    선택된 모양과 pim.place capacity를 소비할 뿐 전체 문제를 다시 분할하지 않는다.

    tile-m/n/k는 각 축에서 상주시킬 Block Group 개수다. 양수면 그대로 쓰고,
    option 기본값 0은 Group 하나로 정규화한다. 존재하지 않는 축도 곱의 항등원인
    1로 기록하며 명시적인 1은 허용하고 2 이상은 거부한다. 따라서
    기본 lowering은 항상 가장 작은 Execution Tile을 만들며, 문제 크기가 크면
    pim-map-devices가 이 Tile을 여러 scheduling round에서 반복 실행한다.
    PIM DRAM 용량은 Tile을 키우는 목표가 아니며 target lowering의 legality 상한이다.
  }];
  let options = [
    Option<"tileM", "tile-m", "int64_t", "0",
           "m 방향 Block Group 수. 0 또는 1 = 한 Group">,
    Option<"tileN", "tile-n", "int64_t", "0",
           "n 방향 Block Group 수. 0 또는 1 = 한 Group">,
    Option<"tileK", "tile-k", "int64_t", "0",
           "device 하나가 K round 한 번에 처리할 semantic reduction Block Group 수. "
           "0 또는 1 = 한 Group. 여러 (round, device_k) partial의 결합 의미는 Common "
           "IR에 기록하고 실제 host/device 구현만 target lowering이 정한다.">,
  ];
  let constructor = "mlir::triton::createPIMFormExecutionTilesPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

def PIMPlaceData : Pass<"pim-place-data", "mlir::ModuleOp"> {
  let summary = "Materialize the final region, replication, and capacity of each buffer";
  let description = [{
    access 마다 `pim.place` 를 만든다.
    - start  = 축별 origin 을 이 device 의 첫 좌표에서 본 값
    - size   = (tile 수 - 1)·tile + 마지막 tile 의 valid. 경계 tile 이 원본 버퍼를
               넘지 않도록 실제 유효 범위까지 반영한 값이다.
    - capacity = execution tile 한 개의 고정 용량. scheduling round가 바뀌어도 같은
                 PIM Device buffer를 재사용하므로 전체 문제 크기와 무관하다.
      allocation_shape는 항상 Block, Block Group, Execution Tile에서 정적으로
      파생한다. 계층 밖 loop가 resident origin을 움직이는 경우 runtime trip count로
      대체하지 않고 진단한다.
    입력의 `pim.access` origin이 어느 logical 축에 불변이면 그 축만 다른 device들은
    같은 데이터를 받으므로 `replicated_over`로 적는다. 중복 선형화 값인 offset에서
    축을 다시 추론하지 않는다. 끊기면 host staging이 device 수만큼 부풀어 오른다.
  }];
  let constructor = "mlir::triton::createPIMPlaceDataPass()";
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::arith::ArithDialect"];
}

def PIMVerify : Pass<"pim-verify", "mlir::ModuleOp"> {
  let summary = "Check the minimal target-independent PIM contract";
  let description = [{
    이 pass는 target lowering이 직접 소비하는 최소 계약만 검사한다.
    - device partition의 곱이 profile의 num_units를 넘지 않는다.
    - Block, Block Group, Execution Tile이 존재하고 실제 local tile 크기와 맞는다.
    - access와 place가 id로 양방향 일대일이며 rank가 일치한다.
    - allocation_shape는 정적인 Execution Tile slot이고 local/unit capacity를 넘지 않는다.
    - 현재 지원하지 않는 0 이외의 padding을 거부한다.
    - M/N/K scheduling marker의 개수와 round/loop 대응만 검사한다.
      scheduling 산술의 정확한 표현은 각 생성 pass의 단위 테스트가 담당한다.
    - 최종 output placement는 out이고 K scheduling에 의존하지 않는다.
    - semantic K reduction과 pim.combine은 일대일이며 combiner가 일치한다.

    pim.access.valid는 PIMPlaceData가 pim.place.size로 materialize한 뒤 target에서
    소비하지 않으므로 이 pass에서 산술 형태를 다시 검사하지 않는다. 동적 start와
    size 역시 생성 pass의 결과를 신뢰하며, 정적으로 아는 범위만 확인한다.
  }];
  let constructor = "mlir::triton::createPIMVerifyPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

#endif
