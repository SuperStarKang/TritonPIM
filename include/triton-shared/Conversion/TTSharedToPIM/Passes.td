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
    - GROUP_M 순서: 그룹 좌표 사슬을 (output_row_tile, output_column_tile) 로 되돌린다.
      순서 자체는 배치에 영향이 없으므로 사슬은 통째로 죽는다.
    앞의 두 logical 축이 output_row_tile 과 output_column_tile 이고, 남는 source 축은
    explicit_k_partition 이다. 각 loop 에 `pim.axis` 로 이름을 단다.
    복원할 수 없는 형태는 진단을 낸다.
  }];
  let constructor = "mlir::triton::createPIMBuildLogicalGridPass()";
  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::arith::ArithDialect"];
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
    ListOption<"pimCoreShape", "pim-core-shape", "int64_t",
               "PIM Group 하나의 PIM Core topology shape.">,
    Option<"lanesPerCore", "number-of-parallel-lanes-per-pim-core", "int64_t", "16",
           "PIM Core 하나의 병렬 실행 lane 능력값.">,
    Option<"parallelExecutionKind", "parallel-execution-kind", "std::string",
           "\"multithread\"", "lane 의 성격. multithread 또는 simd.">,
    Option<"uniformInstructionSequence", "requires-uniform-instruction-sequence",
           "bool", "false",
           "한 PIM Group 안의 PIM Core 들이 같은 command sequence 를 요구하는가.">,
    Option<"usableBankCapacityBytes", "usable-bank-capacity-bytes", "int64_t",
           "33554432", "PIM Core 하나의 attached bank 중 커널 몫. 음수면 미측정.">,
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
  }];
  let options = [
    Option<"groupM", "group-m", "int64_t", "1",
           "output_row_tile 방향으로 묶는 tile 수">,
    Option<"groupN", "group-n", "int64_t", "1",
           "output_column_tile 방향으로 묶는 tile 수">,
    Option<"groupK", "group-k", "int64_t", "1",
           "reduction 방향으로 묶는 tile 수. 출력 축이 아니라 reduction loop 를 "
           "넓힌다 — 누산기는 그대로고 A·B strip 의 K 변만 커져서, local 적재 "
           "한 번이 담는 K 조각이 길어지고 DMA 준비·동기화 횟수가 준다.">,
  ];
  let constructor = "mlir::triton::createPIMFormBlockGroupsPass()";
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PIMPlanGroupExecution : Pass<"pim-plan-group-execution", "mlir::ModuleOp"> {
  let summary = "Decide how the lanes of a PIM Core execute one block group";
  let description = [{
    `#pim.group_execution` 을 func 에 붙인다 — lane 수와, lane 이 block group 의
    무엇을 나눠 갖는가다. profile 의 lane 수는 능력값이고, whole_tile(복제) 모드의
    실제 값은 local storage 예산이 정한다:
    lanes = max(1, min(능력값, usable_local_capacity_bytes / lane당 group 바이트)).
    parallel_execution_kind 가 simd 면 버퍼가 lane 마다 복제되지 않으므로 능력값을
    그대로 쓰고 역할은 언제나 output_row 다.
    다중 core 로의 group 분배는 아직 없으므로 PIM Core 가 여럿인 topology 는 진단을
    낸다 — 그런 target 이 오면 group 계층의 확장으로 만든다.
  }];
  let options = [
    Option<"laneMode", "lane-mode", "std::string", "\"whole_tile\"",
           "multithread target 의 lane 실행 형태. whole_tile 은 lane 이 서로 다른 "
           "tile 을 통째로 맡고 tile 버퍼가 lane 수만큼 복제된다. output_row 는 "
           "lane 들이 한 tile 을 출력 행으로 나눠 함께 계산하고 버퍼는 한 벌만 "
           "둔다 — lane 수가 버퍼 예산과 무관해지므로 tile 을 키울 때 쓴다. "
           "simd target 은 이 옵션을 읽지 않는다(lane 협력이 하드웨어에 내장돼 "
           "있어 언제나 output_row 다).">,
  ];
  let constructor = "mlir::triton::createPIMPlanGroupExecutionPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

def PIMMapDevices : Pass<"pim-map-devices", "mlir::ModuleOp"> {
  let summary = "Distribute logical tiles over PIM Devices and build the pim.device region";
  let description = [{
    logical 축마다 device partition 수를 정하고 loop nest 를 `pim.device` region 으로 감싼다.
    explicit_k_partition 은 자동 인수분해에서는 나누지 않지만, device-axis-partitions 로
    명시하면 나눌 수 있다 — 그때 K 영역이 연속인지는 pim-place-data 가 판정한다.
    자동 인수분해는 device 수를 정사각에 가깝게 가른다 — tile 개수가 런타임이라
    컴파일 시점에는 그것 말고 고를 근거가 없다.
    다만 **어느 축이 싼지는 tile 개수와 무관하게 정해진다.** 축 하나를 쪼개면 그 축에
    불변인 입력이 device 수만큼 복제되기 때문이다(§pim-place-data 의 replicated_over).
    그 순서를 device-axis-priority 로 받는다.
    각 loop 를 balanced block 으로 좁힌다. 몫과 나머지로 나누어 나머지 partition 이 하나씩
    더 맡으므로 꼬리에서 device 가 놀지 않고, 축마다 연속 구간 하나가 유지된다.
    device 안에서 좌표를 0 부터 세고, 전역 좌표는 몸체 첫머리에서 다시 만들어
    `pim.device_origin` 표시를 단다.
  }];
  let options = [
    Option<"deviceAxisPriority", "device-axis-priority", "std::string", "\"\"",
           "먼저 쪼갤 logical 축 이름을 쉼표로 나열한다. 앞에 적힌 축이 인수분해의 "
           "큰 인수를 받는다. 축마다 복제되는 피연산자가 다르므로 어느 축이 싼지는 "
           "target 이 안다 — matmul 은 output_row_tile 을 쪼개면 B 가, "
           "output_column_tile 을 쪼개면 A 가 device 수만큼 복제된다. "
           "비우면 logical 축 순서 그대로 정사각에 가깝게 나눈다.">,
    Option<"deviceAxisGranularity", "device-axis-granularity", "std::string",
           "\"\"", "축을 자를 때 지킬 최소 단위를 <축이름>:<원소수> 로, 쉼표로 "
           "나열한다. 눈금보다 잘게 자르면 경계에 걸친 실행 단위가 양쪽 device 에서 "
           "각각 통째로 실행되어 총 일감이 늘어난다. 눈금을 주면 균등 대신 눈금 "
           "배수로 나누고 나머지는 앞쪽 device 가 하나씩 더 맡는다. tile 하나가 몇 "
           "원소인지는 pim.access 의 sizes 에서 읽는다.">,
    Option<"deviceAxisPartitions", "device-axis-partitions", "std::string",
           "\"\"", "축별 device partition 수를 <축이름>:<개수> 로, 쉼표로 나열한다. "
           "주면 자동 인수분해를 통째로 대체한다. explicit_k_partition 도 나눌 수 "
           "있다 — 단 kernel 이 K 를 연속 구간으로 갈라야 하며(blocked 분할), 그 "
           "판정은 pim-place-data 가 한다. 나열되지 않은 축은 1 이고, 곱이 PIM "
           "Device 수를 넘으면 진단이다. tile 수는 런타임이라 어느 축을 얼마나 "
           "나눌지는 형상을 아는 쪽(사용자·autotuner)이 정한다.">,
  ];
  let constructor = "mlir::triton::createPIMMapDevicesPass()";
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PIMPlaceData : Pass<"pim-place-data", "mlir::ModuleOp"> {
  let summary = "Materialize the final region, replication, and capacity of each buffer";
  let description = [{
    access 마다 `pim.place` 를 만든다.
    - start  = 축별 origin 을 이 device 의 첫 좌표에서 본 값
    - size   = (tile 수 - 1)·tile + 마지막 tile 의 valid. 경계 tile 이 원본 버퍼를
               넘지 않도록 실제 유효 범위까지 반영한 값이다.
    - capacity = 배정량 × tile. device 불변이라 버퍼 크기가 여기서 나온다.
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
    - lane 당 group 바이트 × lane 수가 usable_local_capacity_bytes 안이다(simd 와
      협력 모드는 × 없이 한 벌, 0 이면 생략, 미측정이면 진단).
    - 정적으로 아는 값끼리는 start·size·capacity 부등식을 직접 검사한다. 런타임 값이면
      배치 pass 가 clamp 식을 만들었는지 구조로 확인한다.
    - 같은 logical 축을 나누는 두 place 가 같은 분할 수를 쓴다.
    버퍼별 용량 검사는 버퍼를 어떻게 나누는지가 target lowering policy 라 여기 두지 않는다.
  }];
  let constructor = "mlir::triton::createPIMVerifyPass()";
  let dependentDialects = ["mlir::pim::PIMDialect"];
}

#endif
