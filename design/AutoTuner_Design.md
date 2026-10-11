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

설계 원칙은 두 가지다.

- **seed 알고리즘은 하나다.** target 이름으로 분기하지 않는다.
- **target 차이는 숫자로만 들어온다.** target은 `TargetModel`을 통해 용량, padding,
  허용 mapping만 알려 준다. UPMEM의 작은 Block Group과 AiMX의 큰 Block Group은
  이 숫자에서 자동으로 갈린다.

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

각 계층은 고려하는 자원이 하나씩 정해져 있다. 앞 계층이 정한 단위를 다음 계층이 묶는다.

| 계층 | 고려하는 자원 | 결정 방식 |
|---|---|---|
| Triton Block | 프로그램 단위 | 후보 열거 |
| Block Group | **device 1개의 연산 능력** | 규칙 (target 숫자) |
| Execution Tile | **device 1개의 bank 용량** | 규칙으로 상한을 구하고 ladder 생성 |
| Device Partition | **device 여러 개의 병렬성** | **측정으로 선택** (frontier + LFBO) |

마지막 계층에서 "device 하나가 처리하는 양을 크게"와 "여러 device로 나누기" 사이의
trade-off를 실제 실행 시간으로 결정한다.

## 4. 전체 동작

```text
 Triton kernel + tensors → PIMProblem          PIMTargetSpec → TargetModel
                    │                                     │
 ═════════ Stage A: frontier seed (공통 알고리즘) ═════════╪═══════════
   A1 Triton Block      block ∈ block_values                │
   A2 Block Group       device 1개 연산 포화 ◄──────────────┤ padding, lanes,
                    │   → 상위 K개                          │ local capacity
   A3 Execution Tile    T_max(bank 용량) → … → T_min ◄──────┤ allocations,
                    │   (device당 처리량 大 → 小)           │ bank capacity
   A4 Partition         ladder 단계마다 active 최대 p ◄─────┘ num_units, mapping
                    │   → frontier
 ═══════════════════╪═══════════════════════════════════════════════
   B  Constraint        ConstraintEngine (TargetModel 경유)
   C  초기 측정          frontier → cache warm-start → structured random
   D  LFBO              Random Forest → 계층 일관 mutation → 측정 반복
   E  선택 + cache       end-to-end 시간 최소
```

## 5. TargetModel

seed, constraint, feature는 `autotuner/model.py`의 `TargetModel`만 호출한다.
target별 규칙은 subclass(`UPMEMModel`, `AiMXModel`)에만 존재하고,
`target_model(spec)`이 `common_profile.target`으로 하나를 고른다.

| 질의 | 쓰는 계층 | UPMEM | AiMX |
|---|---|---|---|
| `problem_reason` | 전체 | 없음 | matmul만, bf16/fp16/fp32 |
| `group_reason(extent)` | Block Group | `extent_m % lanes != 0`이면 거부 | 없음 |
| `local_capacity` | Block Group | 48 KiB | 0 (local 없음) |
| `scheduled_compute` | Block Group, feature | 논리 compute | m→batch, n→4096/batch, k→256 올림 |
| `access_allocations` | Block Group, Tile, feature | 논리 footprint | A/B/C (물리 padding 포함) |
| `overhead_allocations` | Tile | 없음 | PMM, Arel, descriptor chain |
| `allocation_reason` | Tile | 없음 | 단일 할당 ≤ 1023×2 MiB, bank interleave footprint |
| `unit_capacity` | Tile | `banks_per_unit × bank_capacity` | 같음 |
| `num_units`, `mapping_reason(p)` | Partition | profile 값, 없음 | `/dev/pl_dma*` 수, `p_k≠1` 거부 |
| `physical_pressure(plans)` | 측정 후 feature | 합 / unit capacity | bank interleave / bank capacity |

- UPMEM lane 규칙은 lowering이 Block Group의 출력 m 행을 모든 tasklet에 똑같이
  나누기 때문에 생긴다(`PIMToUPMEMPass.cpp`). `lanes = nr_tasklets` 또는
  `lanes_per_core`다.
- AiMX 할당은 GEMM의 **하한 추정**이다. scratch, 상수, fusion 할당은 lowering이
  최종 검사한다. 따라서 constraint를 통과해도 lowering이 용량 초과 오류를 낼 수 있고,
  그 오류를 다른 config로 대체하지 않는다.
- access 할당과 overhead 할당을 나누는 이유가 있다. access 할당은 buffer별 전송량
  feature와 Block Group compute/byte에, overhead 할당은 용량 검사에만 쓴다.

## 6. Stage A: frontier seed

구현은 `autotuner/space.py`에 있다.

### 6.1 A1 Triton Block

`block_values`의 모든 값을 후보로 둔다. extent가 1인 축은 1로 고정한다.
Block을 하나로 고정하지 않는다. `group_values`의 최대값이 32이므로, Block이 작으면
도달할 수 있는 Block Group extent의 상한도 작아지기 때문이다. 예를 들어 AiMX의
n=4096은 block 128 × group 32로만 만들 수 있다.

### 6.2 A2 Block Group — `rank_block_groups`

모든 (block, group) 조합을 tile=1, p=1로 만든 뒤 `ConstraintEngine`을 통과한 것만
남긴다. 통과 조건에는 lane 규칙, local capacity, 단일 Block Group의 bank 용량이
포함된다. 남은 후보를 다음 키로 정렬한다.

```text
key 1  compute efficiency = useful compute / (Π total_groups × scheduled_compute)  ↑
key 2  reachable units    = Block Group grid가 채울 수 있는 최대 active unit      ↑
key 3  compute per byte   = Π block_group_extent / Σ access_allocations             ↑
tie    Π block 큰 쪽
→ extent마다 하나씩 남기고 상위 block_group_top_k개(기본 3)를 사용
```

- key 1은 device 내부 연산 활용도다. 문제 경계 padding과 target 고유 padding을
  모두 포함한다.
- key 2는 뒤 계층의 선택지를 지킨다. Block Group 하나가 문제 전체가 되면 Partition
  단계에서 비교할 대상이 사라진다. `mapping_reason`을 반영하므로 AiMX에서는 k 방향
  grid가 병렬성으로 계산되지 않는다.
- key 3은 할당 bytes 기준이다. 따라서 저장 padding(AiMX의 n 4096 폭)도 여기서
  반영된다.

### 6.3 A3 Execution Tile — `tile_ladder`

```text
T_max  tile=(1,1,1)에서 시작해 축마다 다음 tile 값 후보를 만들고,
       constraint를 통과한 것 중 execution tile compute/resident byte가
       가장 큰 축 하나를 키운다. 더 키울 수 없으면 멈춘다.
ladder T_max에서 compute/byte 손실이 가장 작은 축을 한 단계씩 줄인다.
       Execution Tile grid가 도달 가능한 최대 active unit을 채우면 멈춘다.
```

- 축별 tile 값은 `tile_values` 중 total groups보다 작은 값에 total groups를 더한 것이다.
- ladder는 "device 하나가 많이 처리"하는 쪽부터 "여러 device로 나누기 쉬운" 쪽까지의
  후보 축이다. 길면 양 끝을 포함해 `max_ladder_steps`개(기본 4, 2 이상)를 log 간격으로
  고른다.

### 6.4 A4 Partition — `best_partitions`

ladder 단계마다 Execution Tile grid에 들어가는 legal partition 중 active unit이 최대인
것을 고른다. 동률이면 rounds, reduction bytes, stage bytes 순으로 작은 것을 고르고
최대 2개를 남긴다. legal partition은 `partition_values`의 조합 중
`p_m·p_n·p_k ≤ num_units`이고 `mapping_reason`이 `None`인 것이다.

`heuristic_seeds`는 Block Group 순위 순서로, 각 ladder를 **최대 병렬 쪽부터** 나열한다.
따라서 첫 seed는 최상위 Block Group의 최대 병렬 frontier 점이고, search 없이 실행할 때의
기본 config가 된다.

### 6.5 Worked example

**AiMX, M=N=K=4096, fp16, num_units=4, batch=8**

| 단계 | 결과 | 근거 |
|---|---|---|
| A2 | extent (1024,4096,4096) = block (128,128,128) × group (8,32,32) | padding이 없고(key 1 = 1), grid (4,1,1)이 4 unit을 채우며, 같은 조건에서 compute/byte가 최대 |
| A3 | ladder (4,1,1) → (2,1,1) → (1,1,1) | tile_m=4일 때 약 0.75 GiB로 heap 8 GiB 안. Execution Tile 수 1 → 2 → 4 |
| A4 | p (1,1,1) / (2,1,1) / (4,1,1) | p_k=1만 legal |

```text
frontier   tile      p        device당 연산   H2D (A + B×복제)
F1         (1,1,1)   (4,1,1)  25%             32 + 32×4 MiB
F2         (2,1,1)   (2,1,1)  50%             32 + 32×2 MiB
F3         (4,1,1)   (1,1,1)  100%            32 + 32 MiB
```

B는 m 방향 `shared_over`이므로 p_m만큼 복제된다. F1(device 4개)과 F3(device 1개)
중 어느 쪽이 빠른지는 측정으로 정한다.

**UPMEM, M=N=K=4096, int8→int32, num_units=2048, lanes=16**

| extent (m,n,k) | local bytes ≤ 49,152 | compute/byte |
|---|---|---|
| (64,64,128) | 32,768 | 16.0 |
| **(64,64,256)** | 49,152 | **21.3** |
| (128,128,128) | 98,304 ✗ | — |
| (8,…) | lane 규칙 ✗ | — |

A2는 block (64,64,128) × group (1,1,2)를 고른다. ladder는 MRAM 60 MiB 상한에서 시작해
Execution Tile 수가 2048 이상이 될 때까지 줄어든다.

같은 알고리즘인데 AiMX는 (1024,4096,4096), UPMEM은 (64,64,256)이 나온다. 결과를 가른
것은 target 이름이 아니라 padding 단위와 local 48 KiB라는 숫자다.

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

## 8. Stage C~E: 측정과 탐색

### 8.1 초기 측정

순서는 frontier 첫 seed → cache warm-start(비슷한 shape의 best) → 나머지 frontier →
structured random이다.

structured random은 block/group을 무작위로 고른다. partition은 Block Group grid의
최대 active 조합 중에서 고르고, tile은 목표 round 수에서 역산한다. 이 후보는 다음
초기 안전 경계를 통과해야 측정한다.

| 경계 | 기본값 |
|---|---|
| schedule utilization | ≥ 0.5 |
| padding ratio (`scheduled_compute` 기준) | ≤ 0.5 |
| scheduling rounds | ≤ 4 |
| reduction contributions | ≤ 64 |

frontier seed는 용량과 grid에서 구성되므로 이 경계를 적용하지 않는다. 경계 값은
`PIMConfigSpace` 또는 `space_overrides`로 바꿀 수 있다.

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

상위 quantile을 GOOD으로 표시해 Random Forest를 학습한다. 좋은 기록의 decision path에
쓰인 feature를 Block/Group/Tile/partition fragment로 되돌리고, 그 이웃을 만든다.

| 바뀐 fragment | 이웃 생성 (`semantic_mutate`) |
|---|---|
| block, group | 새 Block Group의 ladder에서 **Execution Tile 수가 가장 가까운 점**으로 이동(`regroup`). partition이 들어가면 유지하고, 아니면 최대 active로 교체 |
| tile, partition | 해당 값 한 단계 이동 + 같은 Block Group ladder의 인접 frontier 점(`ladder_neighbors`) |

이웃은 constraint를 통과한 것만 Random Forest 확률과 다양성 penalty로 고른 뒤 측정한다.
budget(`max_benchmarks`, `max_autotune_seconds`)이 끝나거나 `patience`(기본 3) generation
연속으로 개선이 `improvement_epsilon`(기본 0.002) 미만이면 종료한다. 개선이 epsilon 이상인
generation은 stale count를 0으로 되돌린다. epsilon은 측정 noise 수준이다. 1 device처럼
config 간 차이가 작은 지형에서 0.5%씩 이어지는 개선 추세를 끊지 않기 위해서다.

### 8.4 Cache

best와 측정 기록을 `TRITON_CACHE_DIR/pim_autotune_cache.json`에 저장한다. Key에는
problem shape, compute kind, access별 dtype/layout/dependency, common profile 일곱 값,
target runtime config, template version이 들어간다.

## 9. Target lowering과의 경계

- UPMEM은 Common IR 계층과 mapping을 DPU MRAM/WRAM 실행으로 옮긴다.
- AiMX는 같은 계층을 physical group과 descriptor chain으로 옮긴다.
- reduction 위치, descriptor 재사용, physical buffer 재사용은 target lowering의 고정
  의미이며 autotuner knob가 아니다.
- 새 target은 `TargetModel` subclass 하나를 추가하고 `target_model`에 등록한다. seed,
  constraint, feature 코드는 바꾸지 않는다.

## 10. 검증

```sh
# triton_shared 에서
python -m unittest pim_backend.autotuner.test_aimx_capacity \
  pim_backend.autotuner.test_seed_frontier
```

`test_seed_frontier`는 6.5절의 결과(Block Group 순위, ladder, frontier seed 순서,
p_k 제외, UPMEM lane 규칙)를 고정한다.
