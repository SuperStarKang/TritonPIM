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
    앞의 두 logical 축이 m 과 n 이고, 남는 source 축은
    k 이다. 각 loop 에 `pim.axis` 로 이름을 단다.
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
    어느 축과도 맞지 않는 간격이면 크기 1 의 축을 앞에 새로 만든다 — split-K 의
    출력면처럼 원본 버퍼에는 있는데 tile 에는 없는 축이 이 경우다.
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
           "33554432", "bank 하나가 담을 수 있는 최대 byte 수.">,
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

    마지막 group 이 부분이면 기존 부분 tile 과 같은 방식으로 처리된다 — 배치가
    원본 크기로 잘리고(pim-place-data 의 clamp) host 가 남는 자리에 0 을 채운다.
    valid(mask)는 건드리지 않는다 — 소비자는 pim-place-data 의 원본 크기 복원뿐이고
    그 판정은 식의 구조만 본다.

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
    instance를 맡는다. reduction 축의 서로 다른 Tile은 같은 출력 좌표의 partial을 만든다.
    현재 Common IR에서 pim.axis="k"는 spatial split-K Tile 좌표이고,
    execution_tile.groups[k]는 그 Tile 내부 temporal reduction window다.
    없는 축의 partition은 반드시 1이다. Common IR은 비용식이나 자동 device 수 선택을
    수행하지 않는다. 형상을 아는 호출자나 autotuner가 정한 정수 세 개만 materialize한다.
    전체 block-group grid 를 execution-tile grid 로 먼저 자른 뒤 각 축에 대해
      execution_tile_index = round * active_devices + device_coordinate
      tile_begin = execution_tile_index * tile_groups
    로 배치한다. Execution Tile 수가 active device 수보다 많으면 같은 device set을
    여러 round에서 재사용한다. 마지막 round에 Tile이 없는 device는 0회 loop인 no-op이다.
    device 안에서 좌표를 0부터 세고, 전역 좌표는 몸체 첫머리에서 다시 만들어
    `pim.device_origin` 표시를 단다. round loop는 pim.device 바깥 host IR에 남는다.
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
    0이면 존재하는 축은 Group 하나, 존재하지 않는 축은 0으로 기록한다. 따라서
    기본 lowering은 항상 가장 작은 Execution Tile을 만들며, 문제 크기가 크면
    pim-map-devices가 이 Tile을 여러 scheduling round에서 반복 실행한다.
    PIM DRAM 용량은 Tile을 키우는 목표가 아니며 target lowering의 legality 상한이다.
  }];
  let options = [
    Option<"tileM", "tile-m", "int64_t", "0",
           "m 방향 Block Group 수. 0 = 한 Group">,
    Option<"tileN", "tile-n", "int64_t", "0",
           "n 방향 Block Group 수. 0 = 한 Group">,
    Option<"tileK", "tile-k", "int64_t", "0",
           "각 spatial K plane 내부 reduction 방향 Block Group 수. 0 = 한 Group. "
           "k은 독립된 spatial output 축이고 tile-k는 그 plane "
           "내부의 temporal window다. 여러 window의 partial 결합 방식은 target "
           "lowering이 정한다.">,
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
    입력이 어느 logical 축에 불변이면 그 축만 다른 device 들은 같은 데이터를 받으므로
    `replicated_over` 로 적는다. 끊기면 host staging 이 device 수만큼 부풀어 오른다.
  }];
  let constructor = "mlir::triton::createPIMPlaceDataPass()";
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::arith::ArithDialect"];
}

def PIMVerify : Pass<"pim-verify", "mlir::ModuleOp"> {
  let summary = "Fail fast on the target-independent placement contract";
  let description = [{
    - access 와 place 가 id 로 양방향 일대일이다.
    - origin 이 비어 있지 않고 rank 가 sizes 와 같다.
    - pad 가 0 이 아니면 진단(단계 1 은 pad = 0 만 지원).
    - device partition 의 곱이 profile의 num_units를 넘지 않는다.
    - `pim.device`에 `#pim.execution_tile`이 존재하고 Block×Group에서 파생한
      shape 및 device mapping과 일치한다.
    - 한 Group의 local buffer 한 벌이 local_capacity_limit_bytes 안이다. lane들은
      같은 Group을 협력 실행하므로 buffer를 lane 수만큼 복제하지 않는다.
    - 정적으로 아는 값끼리는 start·size·allocation_shape 부등식을 직접 검사한다. 런타임 값이면
      배치 pass 가 clamp 식을 만들었는지 구조로 확인한다.
    버퍼별 용량 검사는 버퍼를 어떻게 나누는지가 target lowering policy 라 여기 두지 않는다.
  }];
  let constructor = "mlir::triton::createPIMVerifyPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

#endif
