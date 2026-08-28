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
    usable-*-capacity-bytes 를 음수로 주면 그 필드를 만들지 않는다(미측정). 0 은 그
    저장소가 존재하지 않는다는 뜻이라 해당 검사가 생략되고, 미측정은 그 값을 필요로
    하는 검사에 도달하는 순간 진단이 된다.
  }];
  let constructor = "mlir::triton::createPIMSetTargetProfilePass()";
  let options = [
    Option<"target", "target", "std::string", "\"upmem\"", "target 이름.">,
    Option<"numberOfPIMDevices", "number-of-pim-devices", "int64_t", "2048",
           "PIM Device 개수.">,
    Option<"banksPerDevice", "number-of-banks-per-pim-device", "int64_t", "1",
           "PIM Device 하나의 bank 수.">,
    Option<"coresPerBank", "number-of-pim-cores-per-bank", "int64_t", "1",
           "bank 하나의 PIM Core 수.">,
    Option<"lanesPerCore", "number-of-parallel-lanes-per-pim-core", "int64_t", "16",
           "PIM Core 하나의 병렬 실행 lane 능력값.">,
    Option<"parallelExecutionKind", "parallel-execution-kind", "std::string",
           "\"multithread\"", "lane 의 성격. multithread 또는 simd.">,
    Option<"usableBankCapacityBytes", "usable-bank-capacity-bytes", "int64_t",
           "33554432", "bank 하나에서 커널이 사용할 수 있는 용량. 음수면 미측정.">,
    Option<"usableLocalCapacityBytes", "usable-local-capacity-bytes", "int64_t",
           "49152", "PIM Core 하나의 local storage 중 커널 몫. 음수면 미측정.">,
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

def PIMPlanIntraDevice : Pass<"pim-plan-intra-device", "mlir::ModuleOp"> {
  let summary = "Plan one block group's bank, core, and lane decomposition";
  let description = [{
    `#pim.intra_device_plan` 을 func 에 붙인다. bank_partitions는 하나의 block
    group을 [m,n,k] bank grid로 나누고, core_partitions는 각 bank subtile을 다시
    [m,n,k] core grid로 나눈다. lanes는 각 core subtile을 lane_axis에서 나눈다.
    세 분할 모두 연속 sequential 분할이며 별도의 assignment 필드는 없다.
  }];
  let options = [
    ListOption<"bankPartitions", "bank-partitions", "int64_t",
               "하나의 block group을 나누는 bank grid [m,n,k]. 기본 [1,1,1].">,
    ListOption<"corePartitions", "core-partitions", "int64_t",
               "각 bank subtile을 나누는 core grid [m,n,k]. 기본 [1,1,1].">,
    Option<"lanes", "lanes", "int64_t", "0",
           "core당 사용할 lane 수. 0이면 profile 능력값에서 선택.">,
    Option<"laneAxis", "lane-axis", "std::string", "\"m\"",
           "lane이 core subtile을 나눌 축: m, n, 또는 k.">,
  ];
  let constructor = "mlir::triton::createPIMPlanIntraDevicePass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

def PIMMapDevices : Pass<"pim-map-devices", "mlir::ModuleOp"> {
  let summary = "Schedule execution tiles over PIM Devices and build the pim.device region";
  let description = [{
    `#pim.execution_tile` 이 먼저 결정된 뒤 logical 축마다 동시에 사용할 PIM Device
    수를 정하고 loop nest 를 `pim.device` region 으로 감싼다. Execution Tile은 device
    하나의 resident 실행 단위이며, Device Grid의 각 좌표는 서로 다른 Execution Tile
    instance를 맡는다. reduction 축의 서로 다른 Tile은 같은 출력 좌표의 partial을 만든다.
    현재 Common IR에서 pim.axis="k"는 spatial split-K Tile 좌표이고,
    execution_tile.groups[k]는 그 Tile 내부 temporal reduction window다.
    k 은 자동 인수분해에서는 나누지 않지만, device-axis-partitions 로
    명시하면 나눌 수 있다 — 그때 K 영역이 연속인지는 pim-place-data 가 판정한다.
    자동 인수분해는 device 수를 정사각에 가깝게 가른다 — tile 개수가 런타임이라
    컴파일 시점에는 그것 말고 고를 근거가 없다.
    다만 **어느 축이 싼지는 tile 개수와 무관하게 정해진다.** 축 하나를 쪼개면 그 축에
    불변인 입력이 device 수만큼 복제되기 때문이다(§pim-place-data 의 replicated_over).
    그 순서를 device-axis-priority 로 받는다.
    전체 block-group grid 를 execution-tile grid 로 먼저 자른 뒤 각 축에 대해
      execution_tile_index = round * active_devices + device_coordinate
      tile_begin = execution_tile_index * tile_groups
    로 배치한다. Execution Tile 수가 active device 수보다 많으면 같은 device set을
    여러 round에서 재사용한다. 마지막 round에 Tile이 없는 device는 0회 loop인 no-op이다.
    device 안에서 좌표를 0부터 세고, 전역 좌표는 몸체 첫머리에서 다시 만들어
    `pim.device_origin` 표시를 단다. round loop는 pim.device 바깥 host IR에 남는다.
  }];
  let options = [
    Option<"deviceAxisPriority", "device-axis-priority", "std::string", "\"\"",
           "먼저 쪼갤 logical 축 이름을 쉼표로 나열한다. 앞에 적힌 축이 인수분해의 "
           "큰 인수를 받는다. 축마다 복제되는 피연산자가 다르므로 어느 축이 싼지는 "
           "target 이 안다 — matmul 은 m 을 쪼개면 B 가, "
           "n 을 쪼개면 A 가 device 수만큼 복제된다. "
           "비우면 logical 축 순서 그대로 정사각에 가깝게 나눈다.">,
    Option<"deviceAxisGranularity", "device-axis-granularity", "std::string",
           "\"\"", "축을 자를 때 지킬 최소 단위를 <축이름>:<원소수> 로, 쉼표로 "
           "나열한다. 눈금보다 잘게 자르면 경계에 걸친 실행 단위가 양쪽 device 에서 "
           "각각 통째로 실행되어 총 일감이 늘어난다. 눈금을 주면 균등 대신 눈금 "
           "배수로 나누고 나머지는 앞쪽 device 가 하나씩 더 맡는다. Group 원소 수는 "
           "#pim.block × #pim.block_group에서 파생한다.">,
    Option<"deviceAxisPartitions", "device-axis-partitions", "std::string",
           "\"\"", "축별 device partition 수를 <축이름>:<개수> 로, 쉼표로 나열한다. "
           "주면 자동 인수분해를 통째로 대체한다. k 도 나눌 수 "
           "있다 — 단 kernel 이 K 를 연속 구간으로 갈라야 하며(blocked 분할), 그 "
           "판정은 pim-place-data 가 한다. 나열되지 않은 축은 1 이고, 곱이 PIM "
           "Device 수를 넘으면 진단이다. tile 수는 런타임이라 어느 축을 얼마나 "
           "나눌지는 형상을 아는 쪽(사용자·autotuner)이 정한다.">,
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
    - device partition 의 곱이 PIM Device 개수를 넘지 않는다.
    - `pim.device`에 `#pim.execution_tile`이 존재하고 Block×Group에서 파생한
      shape 및 device mapping과 일치한다.
    - 한 Group의 local buffer 한 벌이 usable_local_capacity_bytes 안이다. lane들은
      같은 Group을 협력 실행하므로 buffer를 lane 수만큼 복제하지 않는다.
    - 정적으로 아는 값끼리는 start·size·capacity 부등식을 직접 검사한다. 런타임 값이면
      배치 pass 가 clamp 식을 만들었는지 구조로 확인한다.
    - 같은 logical 축을 나누는 두 place 가 같은 분할 수를 쓴다.
    버퍼별 용량 검사는 버퍼를 어떻게 나누는지가 target lowering policy 라 여기 두지 않는다.
  }];
  let constructor = "mlir::triton::createPIMVerifyPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

#endif
