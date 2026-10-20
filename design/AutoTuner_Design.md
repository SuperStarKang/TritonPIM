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
  PIM Common IR lowering → target IR lowering → real execution
```

설계 원칙은 세 가지다.

- **seed 알고리즘은 하나다.** target 이름으로 분기하지 않는다.
- **target 차이는 숫자로만 들어온다.** target은 `TargetModel`을 통해 용량, padding,
  허용 mapping만 알려 준다. UPMEM의 작은 Block Group과 AiMX의 큰 Block Group은
  이 숫자에서 자동으로 갈린다.
- **seed는 측정하지 않고 계산한다.** 단일 unit profiling 자료를 seed에 쓰지 않는다.
  최종 선택은 실제 실행 시간으로만 한다.

## 2. Configuration

```text
PIMConfig
  block_m, block_n, block_k      Triton Block        → #pim.block
  group_m, group_n, group_k      Block Group         → #pim.block_group
  tile_m,  tile_n,  tile_k       Execution Tile      → #pim.execution_tile
  p_m,     p_n,     p_k          Device Partition    → #pim.device_mapping
```

모든 후보는 Common IR과 같은 식으로 해석한다.

```text
block_group_extent[d] = block[d] * group[d]
total_groups[d]       = ceildiv(problem[d], block_group_extent[d])
execution_tiles[d]    = ceildiv(total_groups[d], tile[d])
rounds[d]             = ceildiv(execution_tiles[d], p[d])
```

`PIMProblem`은 Common IR에서 읽은 의미만 정규화한다. 담는 값은 `compute_kind`,
axis extent `[m,n,k]`, unique `pim.place` buffer(direction, dtype, layout,
access axes, irregular)다. 입력 access가 쓰지 않는 축이 `shared_over`이고,
input에는 있고 output에는 없는 `k`가 reduction axis다.

## 3. 계층별 역할

각 계층이 고려하는 자원은 하나씩 정해져 있다. 앞 계층이 정한 단위를 다음 계층이 묶는다.

| 계층 | 고려하는 자원 | 결정 방식 |
|---|---|---|
| Triton Block | 프로그램 단위 | 모든 target 공통 `TRITON_BLOCK = (8, 16, 16)`으로 고정 |
| Block Group | **device 1개의 연산 능력** | 규칙으로 순위를 매기고 상위 2개를 사용 (target 숫자) |
| Execution Tile | **device 1개의 bank 용량** | seed: 병렬도 단계와 round 패턴에서 역산. LFBO: ladder 이웃 |
| Device Partition | **device 여러 개의 병렬성** | 병렬도 5단계 seed와 측정으로 선택 (seed + LFBO) |

Block 크기를 비교하지 않는다. 같은 Block 위에서 Block Group이 target의 연산 폭을,
Execution Tile이 bank 상주량을 만든다. Group 값은 4096까지 열어 두므로, AiMX의 n 4096 폭도
Block 16 × Group 256으로 만들 수 있다.

## 4. 전체 동작

```text
 Triton kernel + tensors → PIMProblem          PIMTargetSpec → TargetModel
                    │                                     │
 ═════════ Stage A: seed (공통 알고리즘, 측정 없음) ═══════╪═══════════
   A1 Triton Block      (8,16,16) 고정                      │
   A2 Block Group       rank_block_groups → 상위 2개 ◄──────┤ padding, lanes,
                    │                                       │ local capacity, compute unit
   A3 병렬도 단계        A_max, A_max/2, … , A_max/16 ◄──────┤ num_units, mapping
   A4 round·P·Tile      round 패턴 → P → tile = ceil(grid/(P·round))
                    │   → 안전 경계 → 비용 순 선택 ◄────────┘ allocations, bank capacity
 ═══════════════════╪═══════════════════════════════════════════════
   B  Constraint        ConstraintEngine (TargetModel 경유)
   C  초기 측정          initial → seed[0] → warm-cache → seed[1:] → structured random
   D  LFBO              Random Forest → 계층 일관 mutation → 측정 반복
   E  선택 + cache       end-to-end 시간 최소
```

## 5. TargetModel

seed, constraint, feature는 `autotuner/model.py`의 `TargetModel`만 호출한다.
target별 규칙은 subclass(`UPMEMModel`, `AiMXModel`)에만 있고,
`target_model(spec)`이 `common_profile.target`으로 하나를 고른다.

| 질의 | 쓰는 계층 | UPMEM | AiMX |
|---|---|---|---|
| `problem_reason` | 전체 | 없음 | matmul만, bf16/fp16/fp32 |
| `group_reason(extent)` | Block Group | `extent_m % lanes != 0`이면 거부 | extent가 연산 단위 `(batch, 4096, max(256, matmul_k_alignment 또는 4096/batch))`를 넘으면 거부 |
| `local_capacity` | Block Group | `local_capacity_limit_bytes` (기본 46,680 B) | 0 (local 없음) |
| `scheduled_compute` | Block Group, feature | 논리 compute | m→batch, n→4096/batch, k→256 올림 |
| `access_allocations` | Block Group, Tile, feature | 논리 footprint | A/B/C (물리 padding 포함) |
| `overhead_allocations` | Tile | 없음 | PMM, Arel, descriptor chain |
| `allocation_reason` | Tile | 없음 | 단일 할당 ≤ 1023×2 MiB, bank interleave footprint |
| `unit_capacity` | Tile | `banks_per_unit × bank_capacity` (기본 60 MiB) | 같음 |
| `num_units`, `mapping_reason(p)` | Partition | profile 값, 없음 | `/dev/pl_dma*` 수, `p_k≠1` 거부 |
| `physical_pressure(plans)` | 측정 후 feature | 합 / unit capacity | bank interleave / bank capacity |

- UPMEM lane 규칙은 lowering이 Block Group의 출력 m 행을 모든 tasklet에 똑같이
  나누기 때문에 생긴다(`PIMToUPMEMPass.cpp`). `lanes = nr_tasklets` 또는
  `lanes_per_core`다. Block m이 8이어도 Group m이 2 이상이면 16 tasklet 조건을 만족한다.
- AiMX Block Group 상한은 저장 padding 없이 device 하나가 계산하는 단위다. 그보다 큰
  범위는 Execution Tile이 bank에 상주시킨다. UPMEM에서 WRAM이 Block Group을 제한하는 것과
  같은 역할이다.
- AiMX 할당은 GEMM의 **하한 추정**이다. scratch, 상수, fusion 할당은 lowering이
  최종 검사한다. 따라서 constraint를 통과해도 lowering이 용량 초과 오류를 낼 수 있고,
  그 오류를 다른 config로 대체하지 않는다.
- access 할당과 overhead 할당을 나누는 이유가 있다. access 할당은 buffer별 전송량
  feature와 Block Group compute/byte에, overhead 할당은 용량 검사에만 쓴다.
- `gemm_profile_spec`/`gemv_profile_spec`은 독립 단일 unit 측정 도구(`autotuner/performance.py`,
  `gemm.py`, `gemv.py`)만 사용한다. seed 생성에는 쓰지 않는다.

## 6. Stage A: seed

구현은 `autotuner/space.py`의 `heuristic_seeds`다. 측정 없이 계산만 한다.

### 6.1 A1 Triton Block

`block_values = TRITON_BLOCK = (8, 16, 16)`. extent가 1인 축은 1로 고정한다.

### 6.2 A2 Block Group — `rank_block_groups`

Group 축마다 `GROUP_VALUES = (1, 2, 4, …, 4096)`을 조합하고, tile=(1,1,1), p=(1,1,1)로 만든 뒤
`ConstraintEngine`을 통과한 것만 남긴다. 통과 조건에는 lane 규칙, AiMX 연산 단위,
local capacity, 단일 Block Group의 bank 용량이 포함된다. 남은 후보를 다음 키로 정렬한다.

```text
key 1  compute efficiency = useful compute / (Π total_groups × scheduled_compute)  ↑
key 2  reachable units    = Block Group grid가 채울 수 있는 최대 active unit      ↑
key 3  compute per byte   = Π block_group_extent / Σ access_allocations             ↑
key 4  Π block 큰 쪽 (Block이 고정이므로 실제로는 동률)
→ extent마다 하나씩 남기고 상위 block_group_top_k개(기본 2)를 사용
```

- key 1은 device 내부 연산 활용도다. 문제 경계 padding과 target 고유 padding을
  모두 포함한다.
- key 2는 뒤 계층의 선택지를 지킨다. Block Group 하나가 문제 전체가 되면 Partition
  단계에서 비교할 대상이 사라진다. `mapping_reason`을 반영하므로 AiMX에서는 k 방향
  grid가 병렬성으로 계산되지 않는다.
- key 3은 할당 bytes 기준이다. 따라서 저장 padding(AiMX의 n 4096 폭)도 여기서
  반영된다.
- 모든 key가 같으면 정렬 전 순서(Group m → n → k 오름차순 열거)를 유지한다.

### 6.3 A3 병렬도 단계

```text
grid   = total_groups (tile = 1)
A_max  = reachable_units(grid, legal partitions)
level  = 0 … parallel_levels−1 (기본 5)
units  = A_max >> level          level ≥ 1에서 units < A_max / parallel_floor(16)이면 중단
```

legal partition은 축마다 2의 거듭제곱(`partition_values`, num_units까지)의 조합 중
`p_m·p_n·p_k ≤ min(instance_units, num_units)`이고, `partition_axes`가 허용한 축만 1보다 크며,
`mapping_reason`이 `None`인 것이다.

모든 데이터를 소수의 device에 몰지 않도록 병렬도에 하한을 둔다. 측정이 병렬도 축을
넓게 덮고 시작하게 하려는 것이다.

### 6.4 A4 round 패턴, Partition, Tile

| 단계 | 규칙 |
|---|---|
| round 패턴 | `initial_round_patterns` 중 level 0은 round 곱 ≤ 2, level ≥ 1은 1만. 기본 패턴은 (1,1,1), (2,1,1), (1,2,1), (1,1,2), (4,1,1), (1,4,1), (1,1,4) |
| Partition | `p_m·p_n·p_k == units`이고 축마다 `p[d] ≤ grid[d]` |
| Tile | `tile[d] = ceil(grid[d] / (p[d] · round[d]))`. `TILE_VALUES`로 제한하지 않는다 |
| 필터 | `engine.valid`(호출자 제약 포함)와 `_safe_initial_schedule` (아래 표) |
| 선택 | `(scheduling_rounds, reduction_bytes, stage_bytes)` 오름차순. level 0의 round (1,1,1)은 2개, 그 밖의 (level, 패턴)은 1개 |
| 병합 | 두 Block Group의 seed를 같은 순번끼리 번갈아 놓고 중복을 제거한다 |

| 안전 경계 (`_safe_initial_schedule`) | 기본값 |
|---|---|
| schedule utilization = useful tiles / (rounds × active units) | ≥ 0.5 |
| padding ratio (`scheduled_compute` 기준) | ≤ 0.5 |
| scheduling rounds | ≤ 4 |
| reduction contributions (`rounds_k · p_k`) | ≤ 64 |

Block Group 하나에서 나오는 seed는 최대 9개(level 0: 2 + 3, level 1~4: 1씩)이고,
두 Block Group을 합하면 최대 18개다. `tune`은 이 중 앞에서부터 `max(1, max_benchmarks // 2)`개를 쓴다.
search 없이 실행하는 경로(`torch_backend/execution.py:_seed_config`)는 첫 유효 seed를 쓴다.

### 6.5 Worked example

**UPMEM, Llama-3-8B qkv (M, N, K) = (1024, 6144, 4096), int8→int32, 2048 DPU, 16 tasklet,
WRAM 46,680 B** (손 계산)

A2: Block Group의 local footprint는 `m·k + k·n + 4·m·n` bytes다.

| extent (m,n,k) | Group | local bytes ≤ 46,680 | compute/byte |
|---|---|---|---|
| **(32,64,256)** | (4,4,16) | 32,768 | **16.0** |
| **(64,32,256)** | (8,2,16) | 32,768 | **16.0** |
| (64,64,128) | (8,4,8) | 32,768 | 16.0 |
| (32,32,512) | (4,2,32) | 36,864 | 14.2 |
| (64,64,256) | (8,4,16) | 49,152 ✗ | — |
| (8,…) | (1,…) | lane 규칙 ✗ | — |

세 후보 모두 key 1 = 1(나누어떨어짐), key 2 = 2048이다. key 3도 같으므로 열거 순서에서
Group m이 작은 (4,4,16)이 1위, (8,2,16)이 2위가 된다.

A3: 1위의 grid는 `(1024/32, 6144/64, 4096/256) = (32, 96, 16)`이다.
`p_m ≤ 32`, `p_n ≤ 64`, `p_k ≤ 16`이므로 A_max = 2048이고, 단계는 2048 → 1024 → 512 → 256 → 128이다.

A4: weight stage는 round (1,1,1)만 허용한다(모델 경로, §8.1). P = (32,64,1)이면
tile = `(ceil(32/32), ceil(96/64), ceil(16/1)) = (1, 2, 16)`이고, execution tile은 (32,48,1)개다.
useful tile 1536 / scheduled 2048이므로 utilization 0.75로 경계를 통과한다.

같은 규칙에서 tile과 P의 곱이 grid를 정확히 덮으면 utilization은 1이다. 예를 들어
2026-10-20 측정(`verify_llama_layer.py`, Llama-3 layer 0)에서 선택된 qkv config `G(4,4,1) T(4,3,32) P(8,32,8)`은 grid (32,96,256)를
`(8·4, 32·3, 8·32)`로 정확히 덮는다. 이 config는 1위 Block Group이 아니므로 seed가 아니라
structured random 또는 LFBO에서 나왔다.

## 7. Stage B: Constraint

`ConstraintEngine.check`는 compile 전에 실행할 수 없는 후보만 제거한다. 검사 순서는
다음과 같다.

1. Block, Group, Tile, partition과 axis extent는 모두 양수
2. `TargetModel.problem_reason`
3. access direction/axis, reduction axis는 `k`, reduction이 없으면 K=1
4. extent가 1인 축은 Block/Group/Tile/partition이 모두 1
5. 각 계층은 아래 계층 단위 수를 넘지 않는다: group ≤ ceil(extent/block),
   tile ≤ total groups, partition ≤ execution tiles. 완전히 빈 Block, Block Group,
   Execution Tile을 만들지 않는다. 문제 경계의 나머지는 lowering이 mask와 padding으로
   처리하고, 그 비용은 compute efficiency에 반영된다
6. Block Group: `group_reason`, logical footprint ≤ `local_capacity`(0이면 생략)
7. Execution Tile: `allocation_reason`, access + overhead 할당 ≤ `unit_capacity`
8. Partition: `p_m·p_n·p_k ≤ num_units`, `mapping_reason`
9. 호출자 제약 `constraint(problem, config)` (예: 모델 stage의 round 제약)

Block 값이 `TRITON_BLOCK`인지는 검사하지 않는다. 다른 Block으로 측정한 cache 기록도 유효한 후보로 남는다.

## 8. Stage C~E: 측정과 탐색

구현은 `autotuner/search.py`의 `tune`이다.

### 8.1 초기 측정

| 순서 | 출처 (`source=`) | 개수 |
|---|---|---|
| 1 | `initial` (호출자가 준 후보) | 전부 |
| 2 | `seed` 첫 번째 | 1 |
| 3 | `warm-cache` (`lookup_similar`: 같은 context의 비슷한 shape best) | cache에 있는 만큼 |
| 4 | `seed` 나머지 | `max_benchmarks // 2`까지 |
| 5 | `structured` | `max_benchmarks // 4`까지 |

constraint를 통과하지 못했거나 이미 본 후보는 건너뛴다. budget이 끝나면 cache에 측정 기록이
있는 후보만 더 읽는다.

structured random은 축마다 Group을 무작위로 고르고, Block Group grid의 최대 active partition 중
하나와 round 패턴 하나를 고른다. tile은 `max(1, grid // (p · round))`로 역산하고, §6.4의 안전 경계를
통과한 것만 측정한다. 경계 값은 `PIMConfigSpace` 필드로 바꿀 수 있다.

모델 stage 경로(`torch_backend/model_compile.py:stage_config_space`)는 공간을 다음과 같이 덮어쓴다.

| 필드 | 값 |
|---|---|
| `partition_axes` | (True, True, target이 p_k > 1을 허용하는가) |
| `instance_units` | `max(1, num_units // instances)` |
| `initial_round_patterns` | weight stage는 ((1,1,1),) |

### 8.2 Feature

다음 값을 Common IR 값과 `TargetModel`에서 파생해 Random Forest 입력으로만 쓴다.
compiler option이나 IR attribute가 되지 않는다.

| feature | 계층 |
|---|---|
| block_group_compute_per_byte, local_memory_pressure | Block Group |
| execution_tile_compute_per_byte, unit_memory_pressure | Execution Tile |
| stage/h2d/d2h bytes | Execution Tile + Partition |
| active_units, useful/scheduled tiles, schedule_utilization, rounds, reduction_bytes | Partition |
| padding_ratio | 전체 (`scheduled_compute` 기준) |

측정 결과에 lowering의 `physical_allocations`가 있으면 `unit_memory_pressure`를
`physical_pressure`로 교체한다.

### 8.3 LFBO

상위 quantile(기본 10%)을 GOOD으로 표시해 Random Forest를 학습한다. 시간이 가장 짧은 기록
`search_copies`개(기본 5)에서 decision path에 쓰인 feature를 Block/Group/Tile/partition fragment로
되돌리고, 그 이웃을 만든다.

| 바뀐 fragment | 이웃 생성 (`semantic_mutate`) |
|---|---|
| block, group | 새 Block Group의 `tile_ladder`에서 **Execution Tile 수가 가장 가까운 점**으로 이동(`regroup`). partition이 들어가면 유지하고, 아니면 최대 active로 교체 |
| tile, partition | 해당 값 한 단계 이동 + 같은 Block Group ladder의 인접 점(`ladder_neighbors`) |

`tile_ladder`는 tile (1,1,1)에서 bank 용량이 허용하는 동안 compute/byte가 가장 좋은 축을 키운 뒤,
나눌 수 있는 축 중 compute/byte 손실이 가장 작은 축을 줄여 grid가 최대 active unit을 채울 때까지
내려가는 점의 열이다. seed가 아니라 LFBO 이웃 생성과 FX packing 비용 계산(`fx_packing._cost`)에 쓴다.

이웃은 constraint를 통과한 것만 Random Forest 확률과 다양성 penalty(기본 0.2)로 generation마다
`select_per_generation`개(기본 8)를 고른 뒤 측정한다. budget(`max_benchmarks`,
`max_autotune_seconds`)이 끝나거나, GOOD/나머지 label이 한 종류뿐이거나, `patience`(기본 3)
generation 연속으로 개선이 `improvement_epsilon`(기본 0.002) 미만이면 종료한다. 개선이 epsilon
이상인 generation은 stale count를 0으로 되돌린다.

### 8.4 Cache

`PIMConfigCache`가 best와 측정 기록을 JSON 하나에 저장한다. 라이브러리 기본 경로는
`workspace/Experiment/Autotuner/autotune_cache.json`(`autotuner/artifacts.py`의 `DEFAULT_CACHE_PATH`)이다.

| 항목 | 내용 |
|---|---|
| key | `M×N×K`, `CACHE_SCHEMA`, dtype, compute kind, access별 dtype/layout/dependency, common profile 값, target runtime config, template version, compiler version |
| hit | 완료된 entry가 있으면 `tune`은 탐색 없이 저장된 best를 반환한다 (`retune=False`) |
| 부분 기록 | 측정할 때마다 `store_partial`로 저장한다. 탐색이 중간에 끊겨도 다음 탐색은 같은 후보를 다시 재지 않는다 |
| warm-start | `lookup_similar`가 같은 context의 비슷한 shape best를 초기 후보로 준다 |

key에 Block 정책이나 seed 규칙은 들어가지 않는다. 같은 key로 이전 Block을 써서 측정한 entry가 있으면 그 entry가 그대로 hit한다.

## 9. Target lowering과의 경계

- UPMEM은 Common IR 계층과 mapping을 DPU MRAM/WRAM 실행으로 옮긴다.
- AiMX는 같은 계층을 physical group과 descriptor chain으로 옮긴다.
- reduction 위치, descriptor 재사용, physical buffer 재사용은 target lowering의 고정
  의미이며 autotuner knob가 아니다.
- 새 target은 `TargetModel` subclass 하나를 추가하고 `target_model`에 등록한다. seed,
  constraint, feature 코드는 바꾸지 않는다.

## 10. 검증

검증은 실제 실행 경로로 한다. seed 계산 결과는 장치 없이 다음 명령으로 확인한다.

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

모델 경로의 측정과 선택은 `torch_backend/examples/upmem/e2e/verify_llama_layer.py`
(정확도)와 `benchmark_model_llama_ttft.py`(TTFT)로 확인한다.
