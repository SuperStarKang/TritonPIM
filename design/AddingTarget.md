# PIM target 추가 방법

새 target은 PIM Common IR의 의미를 다시 정의하지 않고, 동일한
Block–Group–Execution Tile과 device mapping을 target IR로 옮긴다.

## 1. 공유 target spec

`pim_backend/backend/target.py`에 runtime config와 builder를 추가한다.

```text
PIMTargetSpec
  common_profile: PIMTargetProfile
    target
    num_units
    banks_per_unit
    cores_per_bank
    lanes_per_core
    bank_capacity_limit_bytes
    local_capacity_limit_bytes

  runtime_config: <Target>RuntimeConfig
```

Common profile 필드는 추가하지 않는다. 새 target만 요구하는 값은
`<Target>RuntimeConfig`에 둔다. compiler와 autotuner가 같은 builder를 사용하므로
profile 값을 두 군데에 복사하지 않는다.

`local_capacity_limit_bytes=0`은 compiler가 배치할 local memory가 없다는 뜻이다.
`bank_capacity_limit_bytes`는 bank 하나의 양수 byte 상한이며, unit 전체 상한은
`banks_per_unit * bank_capacity_limit_bytes`로 파생한다.

## 2. Compiler pipeline 연결

`pim_backend/backend/compiler.py`에 다음 두 단계를 추가한다.

```text
PIM Common IR -> <Target> IR
<Target> IR   -> runtime artifact
```

TTIR→TTShared IR과 TTShared IR→PIM Common IR pipeline은 수정하지 않는다. 새 target
lowering은 Common IR의 다음 값만 읽는다.

- `#pim.target_profile`
- `#pim.block`
- `#pim.block_group`
- `#pim.execution_tile`
- `#pim.device_mapping<partitions=[p_m,p_n,p_k]>`
- `pim.access/load/store/place`
- linalg compute

Target의 물리 padding, instruction/descriptor 생성, reduction primitive, 실제 lane
활성화 방식은 target lowering에서 결정한다.

## 3. IR과 pass

필요한 구성은 target마다 다음과 같다.

```text
include/triton-shared/Dialect/<Target>/
lib/Dialect/<Target>/
include/triton-shared/Conversion/PIMTo<Target>/
lib/Conversion/PIMTo<Target>/
lib/Target/<Target>/
test/Conversion/PIMTo<Target>/
test/Target/<Target>/
```

Target IR에는 Common IR에 이미 있는 hierarchy나 mapping을 다른 이름으로 복제하지
않는다. Target operation이 실제 codegen에 필요한 물리 정보만 추가한다.

## 4. Autotuner 연결

공통 autotuner Config는 target을 추가해도 바뀌지 않는다.

```text
block_m/n/k
group_m/n/k
tile_m/n/k
p_m/n/k
```

새 target의 capacity 검사가 Common IR의 일곱 profile 필드만으로 표현되지 않는다면
`autotuner/constraints.py`에 target runtime config를 읽는 물리 검사를 추가한다. 그
검사 결과나 target policy를 `PIMConfig` 필드로 추가하지 않는다.

성능 feature도 Block–Group–Tile, mapping, access dependency에서 파생한다. Target의
고정 reduction/persistence 동작은 cost 계산에 반영할 수 있지만 search boolean으로
노출하지 않는다.

## 5. Backend option

공통 option은 다음으로 고정한다.

```text
pim_target
num_units
banks_per_unit
cores_per_bank
lanes_per_core
bank_capacity_limit_bytes
local_capacity_limit_bytes
group_m/n/k
tile_m/n/k
p_m/n/k
```

Target runtime option만 `PIMOptions`와 `<Target>RuntimeConfig`에 추가한다. UPMEM의
`nr_tasklets`, AiMX의 `batch`가 예다.

## 6. 확인 항목

사용자가 빌드한 뒤 다음 경계를 각각 확인한다.

1. target profile이 일곱 필드로 PIM Common IR에 기록되는가
2. Block–Group–Execution Tile attribute가 target lowering까지 보존되는가
3. `[p_m,p_n,p_k]` mapping과 scheduling round가 target IR에 반영되는가
4. `pim.place.allocation_shape`와 `replicated_over`를 target transfer가 사용하는가
5. target 물리 capacity/padding 오류가 target lowering에서 명확히 진단되는가
6. generated artifact가 target runtime config를 그대로 사용하는가
