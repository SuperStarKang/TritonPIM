# PIM AutoTuner

`design/AutoTuner_Design.md`의 구현. Phase 1(§53) 범위 — block / block group /
execution tile / device distribution / reduction을 하나의 계층적 configuration
space에서 탐색한다.

## 모듈 ↔ 설계 단계

| 모듈 | 설계 단계 |
|---|---|
| `types.py` | §2, §51 — PIMProblem / PIMHardwareProfile / PIMConfig / PIMPlan / PIMPlanFeatures / PIMBenchmarkResult |
| `space.py` | §4~§7 — config space, derived tile(§5), KSplitFragment(§6), DistributionFragment(§6.1), seed A~F(§7) |
| `constraints.py` | §8~§13 — Constraint Engine (구조 / local memory / device memory / active grid / target) |
| `planner.py` | §14~§19 — PIM Planner (tile grid, active grid, scheduling, transfer, reduction plan) |
| `features.py` | §20~§27 — Plan Feature Extractor와 feature vector |
| `forest.py` | §36 — Random Forest surrogate (numpy만 사용, decision path 노출) |
| `search.py` | §33, §37~§44, §52 — LFBO loop: quantile label, search copies, tree-guided neighbor, scoring, diversity, termination |
| `cache.py` | §45~§46 — config cache, warm start, 측정 이력 저장·재사용 |

## 사용

```python
from autotuner import PIMProblem, PIMHardwareProfile, PIMBenchmarkResult, tune

problem = PIMProblem(M=1024, N=8192, K=28672)
hardware = PIMHardwareProfile(
    target="upmem", num_units=2048,
    local_memory_bytes=48 * 1024, device_memory_bytes=64 * 1024 * 1024,
    alignment_bytes=8, execution_units_per_device=16,
    # 한 device의 실행 자원을 가득 채우는 작업 묶음 — target 실측으로 지정
    utilization_tile_m=..., utilization_tile_n=..., utilization_tile_k=...,
    target_constraints={"k_split_block_multiple": True},
)

def benchmark_fn(problem, hardware, config, plan):
    # Stage 6~7: config/plan을 실제 커널 launch로 옮기고 wall-clock을 잰다.
    ...
    return PIMBenchmarkResult(total_ms=..., correctness=...)

result = tune(problem, hardware, benchmark_fn)
result.best.config    # PIMConfig
result.best.plan      # PIMPlan (active_device_grid, rounds, transfer, reduction)
result.best.result    # PIMBenchmarkResult
```

실행 가능한 예시는 `../examples/autotune_gemm.py`(UPMEM split-K GEMM 커널 연결).

## 설계 문서와 다른/구체화한 결정

- **P_M/P_N의 소유**: §51의 PIMConfig는 p_m/p_n/p_k를 갖지만, §6.1대로
  distribution policy가 tile grid를 보고 P_M/P_N을 결정한다. 구현에서는
  candidate 생성 시점에 `resolve_distribution`으로 확정해 Config에 넣는다.
  그래야 §52 pseudocode의 순서(Constraint → Planner)대로 Constraint 4(§12)를
  Planner 이전에 검사할 수 있다.
- **DistributionFragment**: §6.1의 6가지 중 K축을 포함하는 policy
  (NK/MK/MNK-balanced)는 P_K를 KSplitFragment(§6)가 소유하는 것과 겹치므로
  Phase 1에서는 M-major / N-major / MN-balanced만 둔다.
- **Random Forest**: pim_triton 환경에 scikit-learn이 없어 numpy만으로
  구현했다 (`forest.py`). tree-guided search(§38)가 요구하는 per-tree
  decision path를 직접 노출한다.
- **S_runtime / S_metadata**: §10, §11의 runtime·metadata 항은 0으로 두었다.
  배포 환경의 예약량은 `local_memory_bytes` / `device_memory_bytes` 예산에
  반영해서 준다.
- **seed는 상수가 아니라 profile 예산에서 유도한다** (target-중립):
  - block/group — local memory 예산을 채울 때까지 m→n→k 번갈아 2배
    (MLIR 공통 pass의 block group 자동 산출과 같은 철학).
  - execution tile — profile의 `utilization_tile_*`(한 device의 실행 자원을
    가득 채우는 단위)로 묶는다. device memory 용량(§11)은 GEMM이 PIM DRAM보다
    클 때만 상한으로 작동한다. UPMEM처럼 memory가 작으면 용량 상한이 먼저
    걸려 채우기가 되고, AiMX처럼 실행 단위·memory가 크면 활용 단위에서 멈춘다.
  - seed는 전략 축(PK / distribution / tile 방향)만 서로 다르게 열고,
    utilization은 seed 유도에만 쓰며 objective는 항상 실측 total_ms다 (§31).
- **KSplitFragment 범위**: 문서 예시는 PK≤8이지만 UPMEM 실측 최적점
  (split_k.py, SPLIT_K=16)이 탐색 범위 안에 있도록 PK=16까지 연다.
- **reduction 위치는 고정 template 기간 동안 target이 정한다**:
  커널이 하나뿐이라 reduction 위치를 config가 바꿀 수 없으므로,
  `build_config_space`가 `TARGET_REDUCTION_POLICY`(upmem→host, aimx→device)로
  KSplit 전략을 구성해 plan·feature가 실제 실행과 일치하게 한다.
  표에 없는 target은 KeyError로 명시적으로 실패한다. codegen이
  `pim.reduction_plan`을 소비해 policy대로 코드를 생성하게 되면 이 표를
  제거하고 reduction 위치를 다시 탐색 dimension으로 연다. seed는 전략을
  하드코딩하지 않고 spec에서 P_K 사다리(1·2·4·8·16)로 고른다.

## Phase 2 — Residency / Replication (§53)

Config에 `a_residency`/`b_residency`(stream|resident),
`replication_policy`(slice|full), `streaming_policy`(per_round|all_rounds)를
추가했다.

- resident operand는 steady-state H2D(`a_bytes`/`b_bytes`)에서 빠지고 1회
  placement(`a_resident_bytes`/`b_resident_bytes`)로 계산된다 — transfer
  minimization이 search space에 반영되는 지점.
- device memory 검사(§11)가 residency/streaming을 반영한다: resident+slice와
  stream+all_rounds는 자기 몫 전체(자기 round 조각 × K plane), resident+full은
  operand 전체 사본, stream+per_round는 temporal window 하나.
- legality: resident operand가 없으면 replication은 slice여야 한다.
- Seed G(weight-stationary, B resident)는 space가 resident를 허용할 때만
  생성된다. residency를 실행에 반영하지 못하는 runtime(현 split_k 러너)은
  `PIMConfigSpace(residency_policies=("stream",), ...)`로 제한해서 쓴다.

## Phase 3 — Persistent Execution (§53)

PersistenceFragment `(device_set_reuse, descriptor_reuse, command_reuse,
persistent_execution)` 을 추가했다. 재사용은 단계적으로만 성립하므로
(descriptor ⇒ device set, command ⇒ descriptor, persistent ⇒ command)
legal 조합 5개만 나열하고 하나의 categorical knob으로 탐색한다.

- Planner가 descriptor 계획을 산출한다: descriptor는 (device, round) 작업마다
  하나이고, 재사용이 켜지면 같은 active 모양의 round는 다시 만들지 않는다.
- Plan feature: `descriptor_build_count`, `descriptor_transfer_bytes`
  (h2d의 §23 metadata 항으로도 가산), `descriptor_reuse_factor`,
  `command_build_count`, `kernel_launch_count`, `device_alloc_count`.
- descriptor 크기는 profile의 `descriptor_bytes`(기본 0 = 무시)로 준다.
- persistence를 실행에 반영하지 못하는 runtime은 space를 기본 전략
  `(True, False, False, False)` 하나로 제한해서 쓴다 (현 split_k 러너).

## Cache 동작 (§45~§46 확장)

같은 tuning key(shape·dtype·layout·hardware·version)에 대해 best와
**측정 이력 전체**를 저장한다. `tune(cache=...)`의 동작:

1. **정확 일치**: 같은 key의 항목이 있으면 탐색 없이 저장된 best를 반환한다
   (`TuningResult.from_cache=True`). 재탐색은 `retune=True`.
2. **재탐색(retune)**: 이미 측정한 configuration은 저장된 결과를 재사용하고
   (`(cache)` 표기, 실측 예산 미소모), 실측은 새 configuration에만 쓴다 —
   이전 측정 전체가 학습 데이터로 복원되므로 탐색은 이전 best 주변의
   neighbor 생성부터 사실상 이어진다.
3. **유사 shape warm start**: context(hardware·dtype·version)가 정확히 같고
   shape만 다른 항목의 best knob을 seed로 주입한다. context가 하나라도
   다르면(예: local_memory 변경) 재사용하지 않는다.
