# PIM Common IR AutoTuner 설계

## 1. 역할

AutoTuner는 PIM 실행 의미를 새로 정의하지 않는다. 후보마다 PIM Common IR에
기록될 값만 선택하고, 같은 lowering을 실제로 실행해 end-to-end 시간을 비교한다.

```text
PIMProblem(access dependency) + PIMTargetSpec
             │
             ▼
  [Block, Group, Execution Tile, p_m/p_n/p_k]
             │
             ▼
       PIM Common IR lowering
             │
             ▼
       target IR lowering
             │
             ▼
          real execution
```

## 2. 유일한 공통 configuration

```text
PIMConfig
  block_m, block_n, block_k
  group_m, group_n, group_k
  tile_m, tile_n, tile_k
  p_m, p_n, p_k
```

- `block_*`: Triton block 크기. TTShared→PIM에서 `#pim.block`으로 복원한다.
- `group_*`: Block Group 하나에 묶는 Block 수.
- `tile_*`: Execution Tile 하나에 묶는 Block Group 수.
- `p_*`: `[p_m,p_n,p_k]` PIM Device mapping.

Execution Tile의 논리 원소 크기는 저장하지 않고 다음처럼 파생한다.

```text
extent_axis = block_axis * group_axis * tile_axis
```

`distribution_policy`, `reduction_policy`, `capacity_limit_*`, residency,
streaming, persistence, bank/core partition은 Config가 아니다.

`PIMProblem`은 별도 실행 policy가 아니라 PIM Common IR에서 읽은 다음 의미만
정규화한다.

```text
compute_kind
axis extents [m,n,k]
unique pim.place buffers
  direction, dtype, layout, access-dependent axes, irregular 여부
```

입력 access가 사용하지 않는 logical axis가 `replicated_over`다. Reduction axis는
input access에는 있지만 output access에는 없는 `k`로 파생한다. `axis_sources`,
residency flag, reduction policy는 추가하지 않는다. 연산 종류는 target별 선택 knob가
아니라 Common IR의 compute op를 식별하기 위해서만 사용한다.

## 3. Target 정보

compiler와 autotuner는 하나의 정의를 공유한다.

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

  runtime_config
    UPMEM: nr_tasklets, host_threads
    AiMX:  batch, base_fpga
```

공통 profile은 그대로 `#pim.target_profile`이 된다. runtime config는 target
lowering/runtime에만 전달된다.

## 4. 초기 후보

기본 Group은 Block 하나인 `[1,1,1]`에서 시작한다. 이것이 Common IR의 최소
Block Group이다. 더 큰 Group은 search가 실제 성능을 비교해 선택한다.

Execution Tile seed는 다음 경계를 지킨다.

- 작은 문제: 해당 device partition에 필요한 Block Group 수까지만 묶는다.
- 큰 문제: unit/bank capacity를 넘기 전까지만 묶는다.
- 전체 문제를 PIM DRAM 크기에 맞춰 padding하지 않는다.

`p_m/p_n/p_k`는 policy 이름에서 계산하지 않고 후보의 정수 값으로 직접 탐색한다.
첫 seed는 problem grid가 허용하는 범위에서 `num_units`에 가장 가까운
active-unit 수를 사용한다. 그러나 이것은 고정 정책이 아니며, 축별
seed와 random/LFBO 후보도 함께 측정해 한 device의 utilization과 여러 device의
parallelism을 실제 end-to-end 시간으로 비교한다.
`(1,1,1)`은 강제 baseline이 아니다. problem grid가 unit 하나만 허용할 때만
최대 feasible seed 자체가 `(1,1,1)`이 된다.

UPMEM 초기 후보에는 유효한 각 `p_k` 값에 대해 `p_m*p_n*p_k`가 최대인
partition 조합을 추가한다. 이는 K-only 분할과 spatial 분할을 별도의 낮은
parallelism으로 비교하지 않고, 가능한 한 같은 전체 active-unit 범위에서
split-K 비율을 실제 측정하기 위한 seed다. 최종 선택은 여전히 end-to-end
측정값으로 결정한다.

## 5. Constraint

compile 전에 수학적으로 실행 불가능한 후보만 제거한다.

1. Block, Group, Tile, partition은 모두 양수다.
2. Block Group working set은 `local_capacity_limit_bytes` 이하다.
3. Execution Tile working set은 unit의 bank capacity 이하다.
4. `p_m*p_n*p_k <= num_units`다.
5. 각 partition은 해당 Execution Tile grid보다 크지 않다.
6. 한 K Execution Tile window는 한 `p_k` plane 안에 들어간다.
7. AiMX의 물리 padding과 allocation 상한은 AiMX runtime config의 `batch`로
   계산한다.

Reduction이 있는 현재 Triton kernel legality로
`K % (p_k * block_k) == 0`도 확인한다. 이것은 Common IR 속성이 아니라 각 spatial
K plane이 Block K 경계에서 시작해야 하는 현재 kernel template의 지원 조건이다.
사용하지 않는 축은 Block, Group, Tile, partition을 모두 1로 고정한다.

Target lowering이 이미 검사하는 alignment나 reduction 위치를 공통 Config의
정책으로 복제하지 않는다.

## 6. 내부 feature

다음 값은 search 모델 입력을 위해 Common IR 값에서 일시적으로 파생한다.

- active units와 scheduling rounds
- effective utilization과 load imbalance
- 모든 input placement의 transfer 및 `replicated_over`에 따른 reuse 양
- p_k partial output과 reduction data 양
- padding ratio
- local/unit memory pressure

이 값들은 compiler option, PIM attribute, cache의 Config에 들어가지 않는다.
최종 objective는 항상 실제 측정한 end-to-end 시간이다.

## 7. LFBO search

최대 active-unit seed와 cache warm start를 먼저 실측한다. 이후 UPMEM의
`p_k` 단계별 최대 active-unit seed, 축별·균형 seed, random valid candidate를
측정한 후 상위 quantile을 GOOD으로 표시한다.
Random Forest의 decision path에서 중요하게 사용된 feature를 Block, Group, Tile,
`p_m/p_n/p_k`에 다시 연결하고, 해당 값의 이웃 후보만 생성한다. Constraint를
통과한 후보를 실제 실행하고 다시 학습한다. partition 이웃으로 이동할 때는
같은 축의 Execution Tile 크기만 새 device plane의 group 수 이하로 줄인다.
그렇지 않으면 유효한 `p_k` 증가도 이전 `tile_k` 때문에 생성 직후 제거되기
때문이다.

Cache key에는 problem shape, compute kind, access별 dtype/layout/dependency, 일곱
common profile 값, target runtime config, compiler/template version을 포함한다. 이전
policy 기반 Config cache는 새
schema와 key가 달라 재사용하지 않는다. Access 기반 공통 분석을 사용하는
search-v3도 이전 GEMM 전용 feature 결과를 best로 곧바로 반환하지 않도록 별도
cache schema를 사용한다.

## 8. Target lowering과의 경계

- UPMEM은 Common IR의 계층과 mapping을 DPU MRAM/WRAM 실행으로 옮긴다.
- AiMX는 같은 계층을 physical group과 descriptor chain으로 옮긴다.
- reduction 위치, descriptor 재사용, physical buffer 재사용은 target lowering의
  고정 의미이며 autotuner boolean knob가 아니다.
- autotuner는 `p_k`를 포함한 후보 전체를 실제로 실행한 시간으로 비교한다.

공통 problem/footprint 모델은 GEMV/GEMM, elementwise, reduction, flattened
convolution, fused placement, irregular access를 같은 access dependency로 표현한다.
각 연산의 frontend lowering과 target compute lowering은 별도 구현 범위지만 Config
schema와 LFBO search는 바꾸지 않는다.
