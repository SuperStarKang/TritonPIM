# PIM Common IR AutoTuner 설계

구현: [pim_backend/autotuner](../pim_backend/autotuner). AutoTuner가 고르는 값의 의미는
[PIM Common IR 설계](PIMCommonIR_Design.md) 3.1절과 7절, UPMEM에서의 쓰임은
[UPMEM Target 설계](UPMEMTarget_Implementation.md) 4절, AiMX에서의 쓰임은
[Silent-PIM 설계](Silent-PIM_Design.md) 9절을 따른다.

## 1. 역할

**내용**

AutoTuner는 PIM 실행 의미를 새로 정의하지 않는다. 후보마다 Common IR에 기록될 4단계
값(Block, Group, Execution Tile, partition)만 고르고, 같은 lowering을 실제로 실행해
end-to-end 시간이 가장 짧은 후보를 선택한다.

**근거**

- seed 알고리즘은 하나다. target 이름으로 분기하지 않는다(`space.py`).
- target 차이는 `TargetModel`이 내는 숫자(Block, 용량, padding, 허용 mapping)로만 들어온다
  (`model.py`).
- seed는 측정 없이 계산한다. 단일 unit profiling 결과(`performance.py`)는 seed에 쓰지
  않는다. 최종 선택은 실제 실행 시간으로만 한다(`search.py`).

**예시**

```text
PIMProblem(access dependency) + PIMTargetSpec
             │
             ▼
  [Block, Group, Execution Tile, p_m/p_n/p_k]   (PIMConfig)
             │  pim_launch_options
             ▼
  Common IR lowering → target lowering → 실제 실행 → total_ms
```

## 2. 문제와 configuration 표현

**내용**

`PIMProblem`은 축 extent `[M,N,K]`, `compute_kind`, 입출력 `PIMAccess` 목록만 가진다.
access가 쓰지 않는 축이 `shared_over`이고, 입력에는 있고 출력에는 없는 축이 reduction
축이다. `PIMConfig`는 Common IR 속성과 일대일로 대응한다.

**근거**

- [types.py](../pim_backend/autotuner/types.py): `accesses`가 비어 있으면 GEMM
  `A(m,k)`, `B(k,n)`, `C(m,n)`으로 채운다. `matvec`, `elementwise`, `reduction`,
  `convolution` 생성자가 같은 표현으로 다른 연산을 만든다.
- `pim_hierarchy`는 Common IR의 `pim-map-devices`와 같은 식을 쓴다.

**예시**

```text
PIMConfig
  block_m, block_n, block_k   Triton Block     → #pim.block
  group_m, group_n, group_k   Block Group      → #pim.block_group
  tile_m,  tile_n,  tile_k    Execution Tile   → #pim.execution_tile
  p_m,     p_n,     p_k       Device Partition → #pim.device_mapping

block_group_extent[d] = block[d] × group[d]
total_groups[d]       = ceildiv(problem[d], block_group_extent[d])
execution_tiles[d]    = ceildiv(total_groups[d], tile[d])
rounds[d]             = ceildiv(execution_tiles[d], p[d])
```

| 생성자 | accesses | 비고 |
|---|---|---|
| `PIMProblem(M,N,K)` | A(m,k), B(k,n), C(m,n) | matmul |
| `PIMProblem.matvec(M,K)` | A(m,k), x(k), y(m) | N=1 |
| `PIMProblem.elementwise(M,N)` | 입력들(m,n), 출력(m,n) | K=1 |
| `PIMProblem.reduction(M,K,N)` | 입력(m,n,k), 출력(m,n) | |
| `PIMProblem.convolution(S,C,R)` | GEMM과 같은 축 구조 | im2col 관점 |

## 3. 계층별 역할

**내용**

각 계층이 다루는 자원은 하나씩 정해져 있고, 앞 계층이 정한 단위를 다음 계층이 묶는다.
Block 크기는 비교하지 않는다. 같은 Block 위에서 Group이 device의 연산 폭을, Execution
Tile이 bank 상주량을 만든다.

**근거**

- Block은 `TargetModel.block(problem)` 하나다. 기본은 `TRITON_BLOCK = (8, 16, 16)`이고
  AiMX matmul만 Block m을 `min(8, batch)`로 둔다(`types.py`, `model.py`).
- `GROUP_VALUES`는 4096까지 열려 있다. AiMX의 n 4096 폭도 Block 16 × Group 256으로
  만들 수 있다(`space.py`).

**예시**

| 계층 | 다루는 자원 | 결정 방식 |
|---|---|---|
| Triton Block | 프로그램 단위 | `TargetModel.block` 고정. extent 1인 축은 1 |
| Block Group | device 1개의 연산 능력, local memory | 순위 규칙 상위 `block_group_top_k`(2)개 |
| Execution Tile | device 1개의 bank 용량 | seed: 병렬도 단계와 round 패턴에서 역산. LFBO: ladder 이웃 |
| Device Partition | device 여러 개의 병렬성 | 병렬도 단계 seed + 측정 |

## 4. 전체 동작

**내용**

seed 계산(A) → 정적 제약(B) → 초기 측정(C) → LFBO 탐색(D) → 선택과 cache 기록(E)
순서로 동작한다. A와 B는 장치를 쓰지 않는다.

**근거**

- `tune`([search.py](../pim_backend/autotuner/search.py))이 C~E를, `heuristic_seeds`
  ([space.py](../pim_backend/autotuner/space.py))가 A를,
  `ConstraintEngine`([constraints.py](../pim_backend/autotuner/constraints.py))이 B를 맡는다.

**예시**

```text
 Triton kernel + tensors → PIMProblem          PIMTargetSpec → TargetModel
                    │                                     │
 ═════════ Stage A: seed (측정 없음) ══════════════════════╪═══════════
   A1 Triton Block      TargetModel.block ◄──────────────────┤
   A2 Block Group       rank_block_groups → 상위 2개 ◄──────┤ group_reason, local capacity,
                    │                                       │ scheduled_compute, access_allocations
   A3 병렬도 단계        A_max, A_max/2, … , A_max/16 ◄──────┤ num_units, mapping_reason
   A4 round·P·Tile      round 패턴 → P → tile = ceil(grid/(P·round))
                    │   → 안전 경계 → 비용 순 선택 ◄────────┘ allocations, minimum_compute
 ═══════════════════╪═══════════════════════════════════════════════
   B  Constraint        ConstraintEngine.check
   C  초기 측정          initial → seed[0] → warm-cache → seed[1:] → structured
   D  LFBO              Random Forest → 계층 일관 mutation → 측정 반복
   E  선택 + cache       end-to-end 시간 최소, PIMConfigCache.store
```

## 5. TargetModel

**내용**

seed, constraint, feature는 [model.py](../pim_backend/autotuner/model.py)의 `TargetModel`만
호출한다. target별 규칙은 subclass(`UPMEMModel`, `AiMXModel`)에만 있고 `target_model(spec)`이
`common_profile.target`으로 하나를 고른다.

**근거**

- UPMEM lane 규칙은 모든 tasklet이 Block Group 하나의 출력 m 행을 똑같이 나눠 계산하기
  때문이다. `lanes = nr_tasklets` 또는 `lanes_per_core`. Block m이 8이어도 Group m이 2
  이상이면 16 tasklet 조건을 만족한다.
- AiMX Block Group 상한(`_compute_unit`)은 저장 padding 없이 device 하나가 계산하는
  단위다. 그보다 큰 범위는 Execution Tile이 bank에 상주시킨다.
- AiMX 할당(`aimx_runtime_allocations`)은 GEMM/GEMV 상주량의 하한 추정이다. scratch,
  상수, fusion 할당은 compile된 Common IR과 lowering의 할당 보고가 최종이다.
- access 할당은 전송량 feature와 compute/byte에, overhead 할당은 용량 검사에만 쓴다.
- `stage_target`: AiMX GEMM은 한 B 행을 batch 행이 함께 읽으므로 M ≥ batch이면 batch가
  클수록 B 재읽기가 줄고, M < 8이면 batch를 M에 맞춰야 output block 행이 0으로 채워지지
  않는다. 이 target이 같은 문제의 탐색, cache key, compile에 함께 쓰인다(9.2~9.4절).
- `minimum_compute`는 target 연산 폭이 문제 전체에 강제하는 padding만 포함한다. 어떤
  config로도 줄일 수 없는 padding이므로 seed 안전 경계(6.4절)에서 뺀다.

**예시**

| 질의 | 쓰는 곳 | UPMEM | AiMX |
|---|---|---|---|
| `stage_target(problem)` | 모델 stage, torch 실행, packing 비용 | spec 그대로 | matmul: `batch = min(spec batch, M 이상인 가장 작은 2의 거듭제곱)`. 그 밖은 spec 그대로 |
| `block(problem)` | A1 | `(8,16,16)` | matmul: `(min(8, batch),16,16)`, 그 밖: `(8,16,16)` |
| `problem_reason` | 제약 | 없음 | matmul·matvec, bf16/fp16/fp32 |
| `group_reason(extent)` | Block Group | `extent_m % lanes ≠ 0`이면 거부 | matmul: extent > `(batch, 4096, max(256, matmul_k_alignment 또는 4096/batch))`, matvec: extent > `(4096, 1, 256)`이면 거부 |
| `local_capacity` | Block Group | `local_capacity_limit_bytes`(기본 46,680 B) | 0(검사 생략) |
| `scheduled_compute` | 순위, feature, 안전 경계 | 논리 compute | matmul: m→batch, n→4096/batch, k→Group마다 256/batch로 올린 뒤 Tile 합을 256으로 올림. matvec: m→4096, k→256 올림 |
| `minimum_compute` | 안전 경계 | `Π problem` | matmul: M→batch, N→4096/batch, K→256 올림. matvec: M→4096, K→256 올림 |
| `access_allocations` | 순위, 제약, feature | 논리 footprint | A/B/C(물리 padding 포함). matvec은 W→B, x→A, y→C |
| `overhead_allocations` | 제약, feature | 없음 | PMM, Arel, descriptor chain |
| `allocation_reason` | 제약 | 없음 | 단일 할당 ≤ 1023×2 MiB, bank interleave footprint ≤ bank 용량 |
| `unit_capacity` | 제약 | `banks_per_unit × bank_capacity`(기본 60 MiB) | 같음 |
| `num_units` | Partition | profile 값(기본 2048) | profile 값(기본 `/dev/pl_dma*` 수) |
| `mapping_reason(p)` | Partition | 없음 | 없음(split-K `p_k > 1` 허용) |
| `resident_operand_reason` | 모델 kv stage 제약 | 없음 | matmul: 상주 B의 저장 크기 `tile_k·⌈group_k⌉₂₅₆ × tile_n·⌈group_n⌉₄₀₉₆`가 `⌈K⌉₂₅₆ × ⌈N⌉₄₀₉₆`와 다르면 거부 |
| `resident_operand_config` | 모델 kv stage 초기 후보 | None | Group `(batch, min(4096, ⌈N⌉_block_n), 256)`/Block, Tile이 문제 전체를 덮고 P=(1,1,1) |
| `physical_pressure(plans)` | 측정 후 feature | 할당 합 / unit 용량 | bank interleave 바이트 / bank 용량 |
| `launch_timing_name` | 단일 unit 측정 | `kernel` | `invoke` |
| `shared_activation_scratch` | 모델 runtime | True | False |
| `gemm_profile_spec` | 단일 unit 측정 | Block (8,16,16), int8→int32, Group (lanes/8, 2, 1) | Block `(min(8,batch),16,16)`, fp16→fp32, Group = 연산 단위 / Block |
| `gemv_profile_spec` | 단일 unit 측정 | Block (2^a,1,64), int32→int32, Group (lanes/2^a,1,1) | Block (8,1,16), fp16→fp32, Group (512,1,16) |

## 6. Stage A: seed

구현: `space.py` `heuristic_seeds`. 측정 없이 계산만 한다.

### 6.1 A1 Triton Block

**내용**

`build_config_space`가 `block_values = TargetModel.block(problem)`으로 두고, extent가 1인
축은 1로 고정한다.

**근거**

- `axis_values(problem, spec, "block", axis)`.

**예시**

```text
UPMEM GEMM (1024,6144,4096)  → Block (8,16,16)
GEMV (4096,1,4096)           → Block (8,1,16)
AiMX GEMM M=4 (batch 4)      → Block (4,16,16)
```

### 6.2 A2 Block Group — `rank_block_groups`

**내용**

Group 축마다 `GROUP_VALUES = (1, 2, 4, …, 4096)`를 조합하고 tile=(1,1,1), p=(1,1,1)로
`ConstraintEngine`을 통과한 것만 남긴다. extent마다 가장 좋은 항목 하나를 남긴 뒤 아래 키의
내림차순으로 정렬하고 상위 `block_group_top_k`(기본 2)개를 쓴다.

**근거**

- key 1은 device 내부 연산 활용도다. 문제 경계 padding과 target 고유 padding을 모두 포함한다.
- key 2는 뒤 계층의 선택지를 지킨다. Block Group 하나가 문제 전체가 되면 Partition 단계에서
  비교할 대상이 사라진다. `legal_partitions`(`mapping_reason` 포함) 기준이다.
- key 3은 할당 바이트 기준이므로 저장 padding(AiMX의 n 4096 폭)을 반영한다.
- 모든 키가 같으면 정렬 전 순서(Group m → n → k 오름차순 열거)를 유지한다. Python 정렬은
  `reverse=True`에서도 같은 키의 원래 순서를 유지한다.

**예시**

```text
key 1  compute efficiency = Π problem / (Π total_groups × scheduled_compute)    ↑
key 2  reachable units    = Block Group grid가 채울 수 있는 최대 active unit     ↑
key 3  compute per byte   = Π block_group_extent / Σ access_allocations          ↑
key 4  Π block (Block이 고정이므로 실제로는 동률)
```

### 6.3 A3 병렬도 단계

**내용**

1위·2위 Block Group마다 tile=1의 grid가 채울 수 있는 최대 active unit `A_max`에서 시작해
절반씩 줄인 단계를 만든다. 모든 데이터를 소수의 device에 몰지 않도록 하한을 둔다.

**근거**

- legal partition(`legal_partitions`)은 축마다 2의 거듭제곱(`partition_values`, num_units까지)의
  조합 중 `p_m·p_n·p_k ≤ min(instance_units, num_units)`이고, `partition_axes`가 허용한 축만
  1보다 크며, `mapping_reason`이 `None`인 것이다.
- `reachable_units(grid, partitions)` = 축마다 `p[d] ≤ grid[d]`인 partition의 최대 곱.

**예시**

```text
grid   = total_groups (tile = 1)
A_max  = reachable_units(grid, legal partitions)
level  = 0 … parallel_levels−1 (기본 5)
units  = A_max >> level      (level ≥ 1에서 units < A_max / parallel_floor(16)이면 중단)
```

### 6.4 A4 round 패턴, Partition, Tile

**내용**

단계마다 round 패턴을 정하고, `p_m·p_n·p_k == units`인 partition에서 tile을 역산한다. 안전
경계를 통과한 후보를 비용 순으로 고른다.

**근거**

- level 0만 round가 2인 패턴을 허용하는 이유는 seed가 병렬도 축을 넓게 덮으면서 round 수를
  작게 유지하기 위함이다.
- Block Group 하나에서 나오는 seed는 최대 9개(level 0: (1,1,1)에서 2개 + round 2 패턴
  3개, level 1~4: 1개씩)이고 두 Block Group을 합치면 최대 18개다.
- padding ratio의 기준은 `minimum_compute`다. AiMX처럼 연산 폭이 문제보다 큰 target에서도
  config가 더하는 padding만 잰다.

**예시**

| 단계 | 규칙 |
|---|---|
| round 패턴 | `initial_round_patterns` 중 level 0은 round 곱 ≤ 2, level ≥ 1은 1만. 기본 패턴: (1,1,1), (2,1,1), (1,2,1), (1,1,2), (4,1,1), (1,4,1), (1,1,4) |
| Partition | `p_m·p_n·p_k == units`, 축마다 `p[d] ≤ grid[d]` |
| Tile | `tile[d] = ceil(grid[d] / (p[d] · round[d]))`. `TILE_VALUES`로 제한하지 않는다 |
| 필터 | `engine.valid`(호출자 제약 포함)와 `_safe_initial_schedule` |
| 선택 | `(scheduling_rounds, reduction_bytes, stage_bytes)` 오름차순. level 0의 (1,1,1)은 2개, 그 밖의 (level, 패턴)은 1개 |
| 병합 | 두 Block Group의 seed를 같은 순번끼리 번갈아 놓고 중복을 제거 |

| 안전 경계 (`_safe_initial_schedule`) | 기본값 |
|---|---|
| schedule utilization = useful tiles / (rounds × active units) | ≥ 0.5 |
| padding ratio = (rounds × active × `scheduled_compute` − `minimum_compute`) / (rounds × active × `scheduled_compute`) | ≤ 0.5 |
| scheduling rounds | ≤ 4 |
| reduction contributions (`rounds_k · p_k`, reduction이 있을 때) | ≤ 64 |

### 6.5 계산 예시

**내용**

UPMEM, `(M,N,K) = (1024, 6144, 4096)`, int8→int32, DPU 2048개, tasklet 16,
local 용량 46,680 B로 6.2~6.4절 규칙에 대입한다.

**근거**

- UPMEM `access_allocations`는 논리 footprint이므로 Block Group local 바이트는
  `m·k + k·n + 4·m·n`이다.
- 모든 후보가 나누어떨어지므로 key 1 = 1, partition 상한 2048에 도달하므로 key 2 = 2048이다.

**예시**

A2:

| extent (m,n,k) | Group | local bytes ≤ 46,680 | compute/byte |
|---|---|---|---|
| (32,64,256) | (4,4,16) | 32,768 | 16.0 |
| (64,32,256) | (8,2,16) | 32,768 | 16.0 |
| (64,64,128) | (8,4,8) | 32,768 | 16.0 |
| (32,32,512) | (4,2,32) | 36,864 | 14.2 |
| (64,64,256) | (8,4,16) | 49,152 ✗ | — |
| (8,…) | (1,…) | lane 규칙 ✗ | — |

키 1~4가 같은 세 후보는 열거 순서를 따르므로 1위 (4,4,16), 2위 (8,2,16)이다.

A3: 1위의 grid는 `(1024/32, 6144/64, 4096/256) = (32, 96, 16)`이다. `p_m ≤ 32`,
`p_n ≤ 64`, `p_k ≤ 16`이므로 `A_max = 2048`이고 단계는 2048 → 1024 → 512 → 256 → 128이다.

A4 (level 0, round (1,1,1)): 곱이 2048인 partition 중 `p_k=1`인 것은 P=(32,64,1)뿐이지만
7절의 `partition ≤ execution tiles`에 걸린다.

```text
P=(32,64,1): tile_n = ceil(96/64) = 2 → execution tiles_n = 48 < p_n = 64  ✗
```

n grid 96에서 `p_n ≤ 32`여야 하므로 남는 후보는 `p_m ≤ 32, p_n ≤ 32, p_k ≤ 16`이다. 모두
grid를 정확히 덮어 utilization 1, padding 0이다. 비용 첫 키(round)는 같고, 둘째 키
`reduction_bytes = C 바이트 × (p_k − 1)`이 가장 작은 `p_k=2`는 P=(32,32,2) 하나다.

```text
P=(32,32,2): tile = (ceil(32/32), ceil(96/32), ceil(16/2)) = (1, 3, 8)
             execution tiles = (32, 32, 2), utilization = 2048 / 2048 = 1
             execution extent = (32, 192, 2048)
2위 (p_k=4): (16,32,4)와 (32,16,4)의 stage_bytes가 28 MiB로 같아 열거 순서상 (16,32,4)
```

tile과 P의 곱이 grid를 정확히 덮으면 utilization은 1이다. 예를 들어 Group (4,4,1)의 grid
(32,96,256)은 `T(4,3,32) × P(8,32,8)`이 `(8·4, 32·3, 8·32)`로 정확히 덮는다. Group k=1은
1위 Block Group이 아니므로 이런 후보는 seed가 아니라 structured random이나 LFBO에서 나온다.

## 7. Stage B: Constraint

**내용**

`ConstraintEngine.check`는 compile 전에 실행할 수 없는 후보만 거부하고 거부 사유 문자열을
돌려준다. Block 값이 `TargetModel.block`인지는 검사하지 않는다. 다른 Block으로 측정한 cache
기록도 유효한 후보로 남는다.

**근거**

- 각 계층은 아래 계층 단위 수를 넘지 않는다. 완전히 빈 Block, Block Group, Execution Tile을
  만들지 않는다. 문제 경계의 나머지는 lowering이 mask와 padding으로 처리하고, 그 비용은
  compute efficiency와 padding ratio에 반영된다.
- 호출자 제약은 정적 검사를 모두 통과한 config에만 부른다.

**예시**

| 순서 | 검사 |
|---|---|
| 1 | Block, Group, Tile, partition, axis extent가 모두 양수 |
| 2 | `TargetModel.problem_reason` |
| 3 | output access 1개 이상, direction은 in/out/inout, axis는 m/n/k |
| 4 | reduction 축은 k만, reduction이 없으면 K=1 |
| 5 | extent 1인 축은 Block/Group/Tile/partition이 모두 1 |
| 6 | group ≤ ceil(extent/block), tile ≤ total groups, partition ≤ execution tiles |
| 7 | Block Group: `group_reason`, logical footprint ≤ `local_capacity`(0이면 생략) |
| 8 | Execution Tile: `allocation_reason`, access + overhead 할당 ≤ `unit_capacity` |
| 9 | Partition: `p_m·p_n·p_k ≤ num_units`, `mapping_reason` |
| 10 | 호출자 제약 `constraint(problem, config)` |

## 8. Stage C~E: 측정과 탐색

구현: `search.py` `tune`.

### 8.1 초기 측정

**내용**

cache에 완료된 entry가 있으면 측정 없이 그 best를 돌려준다(`retune=False`). 없으면 아래
순서로 후보를 모아 측정한다. 제약을 통과하지 못했거나 이미 본 후보는 건너뛰고, budget이
끝나면 cache에 측정 기록이 있는 후보만 더 읽는다.

**근거**

- seed는 `heuristic_seeds(...)[:max(1, max_benchmarks // 2)]`, structured는
  `max(1, max_benchmarks // 4)`개를 만든다.
- structured random(`structured_valid_configs`)은 축마다 Group을 무작위로 고르고, Block Group
  grid의 최대 active partition 중 하나와 round 패턴 하나를 고른다. tile은
  `max(1, grid // (p · round))`로 역산하고 6.4절 안전 경계를 통과한 것만 쓴다.

**예시**

| 순서 | 출처(`source=`) | 개수 |
|---|---|---|
| 1 | `initial`(호출자가 준 후보) | 전부 |
| 2 | `seed` 첫 번째 | 1 |
| 3 | `warm-cache`(`lookup_similar`: 같은 context의 비슷한 shape best) | 최대 3 |
| 4 | `seed` 나머지 | `max_benchmarks // 2`까지 |
| 5 | `structured` | `max_benchmarks // 4`까지 |

### 8.2 Feature

**내용**

아래 값을 Common IR 값과 `TargetModel`에서 파생해 Random Forest 입력으로만 쓴다. compiler
option이나 IR 속성이 되지 않는다. 입력 벡터는 knob 12개(`log2` Block/Group/Tile, partition
그대로)와 아래 feature 14개다.

**근거**

- [features.py](../pim_backend/autotuner/features.py) `extract_features`.
- 측정 결과에 lowering의 `physical_allocations`가 있으면 `unit_memory_pressure`를
  `physical_pressure`로 교체한다.

**예시**

| feature | 정의 |
|---|---|
| `block_group_compute_per_byte` | Π group extent / Block Group access 바이트 |
| `execution_tile_compute_per_byte` | Π execution extent / (access + overhead 바이트) |
| `active_units` | `p_m·p_n·p_k` |
| `useful/scheduled_execution_tiles`, `schedule_utilization` | Π execution tiles, rounds × active, 둘의 비 |
| `scheduling_rounds` | Π rounds |
| `stage_bytes` | Σ_입력 rounds × (access 축의 partition 곱) × tile 바이트 |
| `h2d_bytes`, `d2h_bytes` | rounds × active × 입력/출력 tile 바이트 |
| `reduction_bytes` | 최종 출력 바이트 × (`rounds_k · p_k` − 1) |
| `padding_ratio` | 1 − Π problem / scheduled compute |
| `local_memory_pressure`, `unit_memory_pressure` | Block Group 바이트 / local 용량, Execution Tile 바이트 / unit 용량 |

### 8.3 LFBO

**내용**

측정 기록의 상위 quantile(기본 10%)을 GOOD으로 표시해 Random Forest를 학습한다. 시간이 가장
짧은 기록 `search_copies`개(기본 5)마다 무작위 tree 하나의 decision path에 쓰인 feature를
Block/Group/Tile/partition fragment로 되돌리고, 그 fragment의 이웃을 만든다.

**근거**

- Random Forest([forest.py](../pim_backend/autotuner/forest.py)): tree 25개, 깊이 8,
  leaf 최소 2, 분할마다 √(feature 수)개 feature, Gini 기준.
- `FEATURE_INFLUENCE`가 feature → fragment 대응을 정한다(예: `reduction_bytes` →
  `block_k, group_k, tile_k, p_k`).
- `tile_ladder`는 tile (1,1,1)에서 unit 용량이 허용하는 동안 compute/byte가 가장 좋은 축을
  키운 뒤(`tile_growth`), 나눌 수 있는 축 중 compute/byte 손실이 가장 작은 축을 줄여 grid가
  최대 active unit을 채울 때까지 내려가는 점의 열이다. seed가 아니라 LFBO 이웃 생성과
  packing 정적 비용(9.4절)에 쓴다.

**예시**

| 바뀐 fragment | 이웃 생성(`semantic_mutate`) |
|---|---|
| block, group | 값 한 단계 이동 후 새 Block Group의 `tile_ladder`에서 Execution Tile 수가 가장 가까운 점으로 이동(`regroup`). partition이 들어가면 유지, 아니면 `best_partitions` 첫 항목 |
| tile, partition | 해당 값 한 단계 이동 + 같은 Block Group ladder의 인접 점(`ladder_neighbors`) |

```text
generation마다:
  label = 상위 10% GOOD
  forest.fit → copies 5개의 decision path → fragment → 이웃
  제약 통과 이웃 → forest 확률 − 0.2 × 최대 유사도(diversity penalty)
  상위 select_per_generation(8)개 측정
종료: max_generations(10) | max_benchmarks(100) | max_autotune_seconds
      | label이 한 종류 | 개선 < improvement_epsilon(0.002)이 patience(3) generation 연속
```

### 8.4 Cache

**내용**

[cache.py](../pim_backend/autotuner/cache.py)의 `PIMConfigCache`가 best와 모든 측정 기록을
JSON 파일 하나에 저장한다. 기본 경로는 `workspace/Experiment/Autotuner/autotune_cache.json`
(`artifacts.py` `DEFAULT_CACHE_PATH`)이다.

**근거**

- 측정할 때마다 `store_partial`로 미완료 entry를 저장하므로 탐색이 중간에 끊겨도 다음 탐색은
  같은 후보를 다시 측정하지 않는다.
- 미완료 entry는 lookup에서 miss로 취급한다.
- 파일은 임시 파일에 쓴 뒤 `os.replace`로 교체한다.
- key에 Block 정책이나 seed 규칙은 들어가지 않는다. 같은 key로 다른 Block을 써서 측정한
  entry가 있으면 그 entry가 hit한다.

**예시**

| 항목 | 내용 |
|---|---|
| key | `M×N×K` + context |
| context | `CACHE_SCHEMA`, input/output dtype, compute kind, access별 direction/axes/dtype/layout/irregular, common profile 값(target, num_units, banks, cores, lanes, 용량 2개), runtime config 타입과 값(AiMX batch 포함), template version, compiler version |
| hit | `complete=True` entry가 있으면 `tune`은 탐색 없이 저장된 best를 반환 |
| 부분 기록 | `complete=False`, records만 보존 |
| warm-start | `lookup_similar`: 같은 context에서 `Σ |log2(shape 비)|`가 작은 best 최대 3개 |

## 9. 호출 경로

### 9.1 Triton 커널 decorator — `pim_autotune`

**내용**

[decorator.py](../pim_backend/autotuner/decorator.py)의 `pim_autotune`은 `fn(a, b, *, config)`
형태의 GEMM 함수를 감싼다. cache hit이면 그 config로 실행하고, 탐색이 꺼져 있으면 첫 유효
seed로 실행하고, 그 밖에는 `tune`을 돌린다.

**근거**

- 환경 변수 `PIM_AUTOTUNE`, `PIM_RETUNE`이 `search`, `retune` 인자를 덮는다(`"1"`만 참).
- 측정은 warmup 1회 뒤 `reps`(기본 3)회의 median이다. `pim_driver.timing` 단계별 median과
  `pim_driver.aimx_allocations`(`physical_allocations`)도 함께 기록한다.
- 정확성 검사: `ref_fn`이 있으면 전체 `allclose`, 없으면 행·열을 최대 32개씩 표본 추출해
  `rtol/atol` 또는 `rel_l2`로 비교, `check_correctness=False`이면 `correctness=None`(성능
  후보로는 사용). 검사 방식과 허용 오차는 cache template version에 들어간다.
- `pim_launch_options`가 config와 target spec을 compiler 옵션으로 옮긴다.

**예시**

```python
@pim_autotune(target="upmem", budget=SearchBudget(max_benchmarks=30))
def gemm(a, b, *, config):
    ...  # gemm_kernel[grid](..., **pim_launch_options(config, spec))
```

### 9.2 torch 단일 GEMM 경로

**내용**

`torch_backend/execution.py` `resolve_execution`은 target을 `stage_target(problem)`으로 바꾼
뒤 config를 다음 우선순위로 정한다.

**근거**

- `PIMTorchConfig` 주석(`config.py`): cache miss는 seed로 대체하지 않고 오류를 낸다. 무엇으로
  실행됐는지 알 수 없게 되기 때문이다.
- 선택한 config도 `ConstraintEngine.check`를 다시 통과해야 한다.

**예시**

| 순위 | 출처 |
|---|---|
| 1 | `PIMTorchConfig.config` |
| 2 | `autotune_cache_path`의 best (template `TORCH_GEMM_TEMPLATE_VERSION`, miss면 오류) |
| 3 | `_seed_config`: `heuristic_seeds`의 첫 유효 후보 |

### 9.3 모델 stage 경로

**내용**

`torch_backend/model_compile.py` `compile_model`은 phase마다 첫 layer를 stage class로 나누고
(`model_policy.py` `analyze_layer`), class마다 측정 후보를 모은 뒤(`tune_phase`), packing과
class별 config를 함께 고르고(`select_phase`), 모든 layer를 그 config로 compile한다. 측정은
weight 또는 key/value cache가 이미 상주한 상태의 실행 시간 E와 weight 업로드 시간 U다.

**근거**

- `StageKey` 종류(`label`): weight operand가 있으면 weight, 장치 key/value cache를 그대로
  읽으면 kv, 둘 다 아니면 activation.
- `tune_stage_class`: `stage_target(problem)`으로 탐색·기록하고, `initial` =
  `widened_configs` + `fitted_configs` + (kv class면) `resident_operand_config`. cache hit으로
  `tune`이 탐색하지 않으면 아직 재지 않은 `initial` 후보를 따로 재서 기록에 더한다. 측정
  프로세스가 실패한 기록(`correctness=False`)과 현재 제약을 통과하지 못한 기록은 후보에서 뺀다.
- `stage_config_space`: `partition_axes = (True, True, mapping_reason(p=(1,1,2)) is None)`,
  `instance_units = max(1, num_units // instances)`, weight/kv class는
  `initial_round_patterns = ((1,1,1),)`.
- `stage_constraint`(`model_policy.py`): weight·kv stage는 M/N/K round 1개, kv stage는
  `instances × p_m·p_n·p_k ≤ num_units`(모든 instance가 한 batch round)와
  `resident_operand_reason` 통과, 모든 stage는 round ≤ `max_initial_scheduling_rounds`(4).
  `p_k`는 target의 `mapping_reason`에 맡긴다.
- `widened_configs`: packed class가 이어 붙인 member class의 best에서 packed 축 tile만 늘린
  후보. Block m이 다른 member(M이 작아 batch가 다른 경우)는 쓰지 않는다. `fitted_configs`:
  같은 context의 비슷한 shape best 최대 4개의 tile을 round 1개가 되도록 맞춘 후보.
- `cache_version`: `TORCH_GEMM_TEMPLATE_VERSION|model|op|w<weight mask>|x<instances>`에
  kv class는 `|kv<mask>`(전치면 `t`)를 붙인다.
- 기록: `correctness=None`, `timings=(("weight_upload", U),)`, `physical_allocations` =
  (weight, state, input, output[, kv]) 바이트(`Footprint`).

**예시**

```text
compile_model(phases, repeats={phase: 실행 횟수})     # None이면 모든 phase 1
  1 tune_phase       class마다 tune_stage_class (packed class는 member class를 먼저)
  2 kv_bytes_bound   layers × Σ kv class (count × 측정 후보 중 최대 kv 바이트)
                     → 첫 kv phase부터 capacity = unit 용량 − 모든 phase의 최대 bound
  3 select_phase     packed class(origin이 있는 class: 같은 activation의 weight를 n으로
                     이어 붙인 QKV·gate/up, 같은 B를 읽는 GEMM을 m으로 쌓은 것)마다
                     pack/separate 2^n 조합 전부에 select_configs → 한 layer 시간 최소
  4 session.compile  configure_with(choice): class의 config, pack=GEMM이 separate가 아님
                     compile된 layer 0의 stage가 고른 class와 같은지 검사
```

`select_configs(classes, candidates, capacity, layers, repeats)`는 class마다 측정 후보 하나를
고르는 multiple-choice knapsack이다. `repeats`는 한 request에서 phase를 실행하는 횟수다
(prefill 1, decode는 생성 token 수 − 1).

| plan | 한 layer 시간 | 용량 조건 |
|---|---|---|
| reloaded | `Σ count × repeats × (E + U)` | `slot + max input + max output ≤ capacity` |
| resident | `Σ count × (repeats × E + U)` | `layers × slot + max input + max output ≤ capacity` |

```text
slot   = Σ count × (weight + state)          scratch = max input + max output
frontier: class를 하나씩 더하며 (시간, slot) 순으로 정렬하고,
          앞선(더 빠른) 선택이 slot·input·output 모두 같거나 작으면 버린다 (Pareto)
plan마다 용량을 만족하는 가장 빠른 선택 → 두 plan 중 엄격히 더 짧은(<) 쪽
repeats = 1이면 두 시간이 같아 reloaded가 남는다
```

session 실행 시 `select_layer_config`(`execution.py`)는 `stage_target`으로 바꾼 target에서
config를 정한다. compile된 layer는 고른 config를 그대로 받고, `autotune_cache_path`만 주면
측정 기록 중 실제 unit 수에 맞는 것을 `batch rounds × total_ms` 최소로 고른다. weight stage는
instance 1개·round 1개, kv stage는 M/N/K round 1개·batch round 1개여야 한다
(`p_b = min(instances, num_units // units)`, `batch rounds = ceil(instances / p_b)`).

### 9.4 GEMM packing 비용 — `fx_packing._cost`

**내용**

`torch_backend/fx_packing.py`의 block-diagonal packing(`_block_diagonal`)은 member p개를
`[A1 | … | Ap] @ blockdiag(B1 … Bp)`로 묶을지 AutoTuner의 정적 모델로 정한다. 실행하지 않는다.

**근거**

- `_cost(spec, problem)`: `stage_target(problem)`에서 `rank_block_groups` 상위 Block Group의
  `tile_ladder` 마지막 점(가장 병렬인 점)과 `best_partitions` 첫 유효 partition으로
  `(tiles × scheduled_compute, tiles × Σ access_allocations)`를 낸다.
- 묶음 수 count마다 비용 × (member 수 / count)를 비교한다. compute가 묶지 않을 때 이하인
  count 중 access 바이트가 가장 작은(같으면 큰) count를 고르고, 1이면 묶지 않는다.

**예시**

```text
count ∈ {member 수의 약수}:  problem = (m, count·n, count·k)
free  = {count | compute(count) ≤ compute(1)}
count = argmin over free of (bytes(count), −count)
```

## 10. 단일 unit 측정 도구

**내용**

`autotuner/gemm.py`(`profile_gemm`), `autotuner/gemv.py`(`profile_gemv`, UPMEM·AiMX),
`performance.py`(`calibrate_performance`, `profile_single_unit`)는 device 1개에서 Group과
Tile을 키우며 throughput 포화점을 측정하는 독립 도구다. seed 생성과 `tune`에는 쓰지 않는다.

**근거**

- `num_units=1`로 바꾼 target에서 실제 커널을 실행하고 `launch_timing_name` 시간만 쓴다.
- matmul/matvec: profile spec의 Block 후보마다 `group_k`를 2배씩 키워 throughput이 안정되면
  멈추고, 가장 빠른 Group에서 `tile_k`를 2배씩 키운다. 안정 판정은 마지막 4점이 직전 plateau의
  ±5%, 마지막 3점이 plateau의 ±5% 안에 있고 어떤 점도 plateau/0.95를 넘지 않는 것이다. Tile이
  안정되지 않으면 오류를 낸다.
- 그 밖의 연산은 `sweep_extents`가 필요하며 `rank_block_groups` 상위 Block Group마다
  `tile_growth`를 따라 측정한다.
- 결과는 `single_unit.csv`(측정마다 즉시 기록)와 `performance.json`(schema
  `pim-single-unit-saturation-v2`)이다. 저장 경로는 프로젝트 안이어야 한다(`project_path`).
- GEMV profiling의 dtype은 target의 `gemv_profile_spec`을 따른다(UPMEM int32→int32, AiMX
  fp16→fp32). AiMX의 정확도 검사는 vACC 수치 특성 때문에 `x[::256] = 1`, 기대값
  `ceil(K/256)`이다([Silent-PIM 설계](Silent-PIM_Design.md) 11.3절).

**예시**

```text
Block 후보 b:
  group_k = 1, 2, 4, …   (제약 또는 측정 상한까지, 안정되면 중단) → 가장 빠른 Group g
  tile_k  = 2, 4, 8, …   (g 고정, 안정될 때까지)                  → saturation work
```

## 11. Target lowering과의 경계

**내용**

AutoTuner는 Common IR 값만 정한다. reduction 위치, descriptor 재사용, physical buffer 재사용,
물리 DPU 배정 묶음(`device_mapping`)은 target lowering의 고정 의미이며 AutoTuner knob가
아니다.

**근거**

- UPMEM 물리 배정 묶음 G는 `convert-pim-to-upmem`이 shared_over와 고정 slot 바이트로 정한다
  (UPMEM 설계 6.1절).
- 새 target은 `TargetModel` subclass 하나를 추가하고 `target_model`에 등록한다. seed, constraint,
  feature 코드는 바꾸지 않는다.

**예시**

| 결정 | 주체 |
|---|---|
| Block/Group/Tile/partition | AutoTuner |
| MRAM slot 배치, tasklet 분할, DMA chunk | UPMEM lowering |
| 물리 DPU 묶음 G, assignment | UPMEM lowering + runtime |
| descriptor chain, physical group | AiMX lowering |

## 12. 확인 방법

**내용**

seed 계산은 장치 없이 확인할 수 있다. 모델 경로의 측정과 선택은 예제 스크립트로 확인한다.

**근거**

- `heuristic_seeds`, `rank_block_groups`는 측정 함수를 호출하지 않는다.

**예시**

```sh
# triton_shared/pim_backend 에서
python -c "
from autotuner import PIMProblem, build_config_space, heuristic_seeds, target_spec
from autotuner.space import rank_block_groups
problem = PIMProblem(1024, 6144, 4096, input_dtype='int8', output_dtype='int32')
target = target_spec('upmem', num_units=2048, nr_tasklets=16)
space = build_config_space(problem, target)
print(rank_block_groups(problem, target, space)[:2])
for config in heuristic_seeds(problem, target, space):
    print(config.key())
"
```

기대 출력: 첫 줄은 `[((8, 16, 16), (4, 4, 16)), ((8, 16, 16), (8, 2, 16))]`, 첫 seed는
`(8, 16, 16, 4, 4, 16, 1, 3, 8, 32, 32, 2)`이다(6.5절).

모델 경로: `torch_backend/examples/upmem/e2e/verify_llama_layer.py`(정확도),
`benchmark_model_llama_ttft.py`(TTFT), `benchmark_model_llama_generate.py`(generation).
