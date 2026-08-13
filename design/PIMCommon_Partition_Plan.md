# PIM Common — 분할 · Split-K 변환과 비용 모델 구현 계획

> 목적: `PIMCommon_Design.md` §16(작업 분할·K loop·Split-K)과 §20.3(target이 호출하는 PIM Common 패스)에서
> "설계했으나 미구현"으로 남은 부분을, **target 하드웨어 정보를 배제한 공통(target-neutral) 변환·비용 계층**으로 구현한다.
> 원칙: **구조는 PIM Common이 만들고, 파라미터(유닛 수·factor·BW·rate)는 나중에 Target/`perf_model`이 준다** (§16.3, §22).
> 전제: 지금은 PIM Common IR만 완성한다. Target IR(UPMEM/AiMX legalization)은 이후 작업.

---

## 0. 현재 상태 (2026-07 점검 결과)

구현됨: `pim` dialect(6 op + `!pim.access`), `--convert-ttshared-to-pim`, `--pim-build-grid`,
`--pim-remove-buffers`(§15.6 hoisting 포함, grid-loop skip), `--pim-analyze-logical-work`
(offset/strides/valid/pad/index 5개 root 추적, `invariant_program_dims`, `reduction_combine`,
`reduce_across_programs`), `--pim-partition`(`access_bytes`/`working_set_bytes`/cost 표), `--pim-classify-compute`.
파이프라인은 `pim_backend/backend/compiler.py`의 `PIM_COMMON_PASSES`가 §20.1 순서 그대로.

미구현(이 문서의 대상):
- `--pim-split-reduction` (§16.5) — reduction loop를 grid 축으로 승격
- `--pim-apply-partition` (§20.3) — grid 축을 유닛/순차로 쪼개고 재배치 + 재-hoisting
- `--pim-partition`의 **reduction 비용 동인** 확장 — `final_output_bytes` / `partial_combine_bytes` 분리
- host `perf_model` 계약 (Triton `prune_configs_by` 연결) — 이번엔 중립 emit + 스텁까지만

병행 정리(소):
- `pim.grid_permutable` 항상-emit 위반 (partition이 `grid.empty()` return 뒤에서 설정) — apply-partition의 재배치 합법성 판단이 이 속성을 읽으므로 **선행 수정 필요**.
- (낮은 우선순위, 분할과 무관) `pim.copy` verifier 부재, classify의 `scan` 미구현 — 이 계획의 blocker 아님. 별도 처리.

---

## 1. 범위와 순서

의존성 순서로 4단계 + 선행 정리.

| 단계 | 산출물 | 의존 |
| --- | --- | --- |
| Phase 0 | `grid_permutable` 항상 emit 수정 | 없음 |
| Phase 1 | `--pim-split-reduction=factor=S` | analyze의 `reduction_combine` |
| Phase 2 | `--pim-apply-partition=splits=[...]` | Phase 0(`grid_permutable`), remove-buffers hoisting |
| Phase 3 | `--pim-partition` reduction 비용 확장 | Phase 1(승격 후 `reduce_across_programs`) |
| Phase 4 | host `perf_model` 계약 + `prune_configs_by` 스텁 | Phase 3 emit |

두 변환(Phase 1·2)은 §20.3대로 **기본 파이프라인에 넣지 않고**, `triton-shared-opt`에서 옵션을 주어 직접 호출·테스트한다(=target 없이 검증 가능).

---

## Phase 0 — `pim.grid_permutable` 항상 emit (선행 정리)

**파일:** `lib/Conversion/TTSharedToPIM/PIMPartitionPass.cpp`

현재 `grid_permutable`이 `if (grid.empty()) return;` **뒤**에서 설정된다. 설계 §16.3은 이를 grid 무관·항상 emit으로 분류.
수정: `grid_permutable` 계산·부착을 `grid.empty()` early-return **앞**으로 이동. 값은 기존대로 `reduce_across_programs`가 비면 `true`.
`access_bytes`/`working_set_bytes`와 같은 "항상 emit" 블록에 합류시킨다.

**검증(직접):** `test/Conversion/TTSharedToPIM/`에 grid loop 없는 `pim.access`만 있는 함수 케이스 추가 → `pim.grid_permutable`이 붙는지 CHECK.

---

## Phase 1 — `--pim-split-reduction=factor=S` (§16.5)

**새 파일:** `lib/Conversion/TTSharedToPIM/PIMSplitReductionPass.cpp`
**옵션:** `factor` (i64, 필수). Passes.td에 `Option<"factor", "factor", "int64_t", "1">`.

### 하는 일
`pim.reduction_combine`이 붙은 순차 reduction `scf.for`(K-loop)를 `S × (trip/S)`로 잘라 **바깥 조각을 새 grid 축으로 승격**.

### 변환 전 / 후

```mlir
// 전
scf.for %gm ... {pim.grid_dim="x"} {
 scf.for %gn ... {pim.grid_dim="y"} {
   %init = linalg.fill ins(%zero) outs(%empty)
   %acc = scf.for %k = %c0 to %Kt step %c1 iter_args(%a=%init) {
     ... 누적 ...
     scf.yield %next
   } {pim.reduction_combine="add"}
   pim.store %acc, %dC
 }}
}
```

```mlir
// 후 (factor=S,  inner_trip = Kt/S)
scf.for %gm ... {pim.grid_dim="x"} {
 scf.for %gn ... {pim.grid_dim="y"} {
  scf.for %gk = %c0 to %S step %c1 {                 // 새 grid 축
   %init = linalg.fill ins(%zero) outs(%empty)       // init 이 안으로 들어옴
   %acc = scf.for %k = %c0 to %inner_trip step %c1 iter_args(%a=%init) {
     %kk = arith.addi (arith.muli %gk, %inner_trip), %k    // 원래 k 인덱스 복원
     ... 누적(%kk 사용) ...
     scf.yield %next
   } {pim.reduction_combine="add"}
   pim.store %acc, %dC_partial     // 주소는 gm,gn 의존, gk 무관
  } {pim.grid_dim="k"}
 }}
}
```

핵심: **init(`linalg.fill`)을 새 loop 안으로 복제** — 각 `%gk`가 0에서 시작해야 함. 새 축 이름은 `pim.grid_dim`에서 미사용 문자(`"k"`)를 고른다(build-grid의 `axisName` 규칙 재사용).

### 합법성 (§16.5, 전부 IR로 판정)
1. loop에 `pim.reduction_combine`이 있다 (결합 가능 누적)
2. loop 몸체에 `pim.store`/`pim.scatter`가 없다 (반복마다 부수효과 없음)
3. 누산기 초기값이 `linalg.fill` 상수 채움이다 (rematerialize 가능)

셋 중 하나라도 불만족이면 `emitError`로 실패(no silent skip — convert 패스의 wraparound 처리와 같은 방식).

### 초기 제약
- `factor`는 `Kt`의 약수만 허용(`Kt % S == 0`). 아니면 실패. 나머지(remainder) 조각은 padding 영역(target, §1)이라 **초기 구현에서 제외**.
- 다중 reduction loop가 한 인스턴스에 여러 개면, 초기엔 가장 바깥 하나만 대상으로(문서화).

### 재사용
- 인덱스 산술/`index_cast`: `PIMBuildGridPass.cpp`의 `castTo`.
- 축 이름 부여: `axisName`.
- 새 loop 생성/이동: 표준 `scf.ForOp` builder + `IRMapping`으로 body clone.

### 승격 후 분석 재실행
`reduce_across_programs`는 **split-reduction이 직접 달지 않는다.** 승격 후 `%dC_partial`이 `gk`에 무관해지므로,
`--pim-analyze-logical-work`를 **다시 돌리면** 기존 로직이 `invariant_program_dims`에 k를 넣고 `reduce_across_programs=["k"]`를 자동 도출한다(§16.5 "기존 분석이 그대로 동작"). → 권장 호출 순서: `--pim-split-reduction=factor=S --pim-analyze-logical-work`.

### lit 테스트 (`test/Conversion/TTSharedToPIM/split_reduction.mlir`)
- RUN: `triton-shared-opt --pim-split-reduction=factor=64 --pim-analyze-logical-work %s`
- CHECK: 새 `scf.for ... {pim.grid_dim = "k"}`, `linalg.fill`이 그 안, `%dC_partial` access에 `invariant_program_dims`에 k 인덱스 포함, func attr `pim.reduce_across_programs = ["k"]`.
- 음성 케이스(`-verify-diagnostics`): 몸체에 `pim.store` 있는 loop → 실패.

---

## Phase 2 — `--pim-apply-partition=splits=[...]` (§16.1, §20.3)

**새 파일:** `lib/Conversion/TTSharedToPIM/PIMApplyPartitionPass.cpp`
**옵션:** `splits` (`ListOption<int64_t>`), grid 축 순서(바깥→안, `pim.program_dims` 순)대로 각 축의 유닛 분할 `s_d`.

### 하는 일
각 `pim.grid_dim` 축 `scf.for %g = 0 to G`를 유닛 축(바깥)과 순차 축(안)으로 쪼개고, 유닛 축들을 바깥으로 재배치.

```mlir
// 전
scf.for %gm = 0 to 32 {pim.grid_dim="x"} { scf.for %gn = 0 to 16 {pim.grid_dim="y"} { body(%gm,%gn) }}

// 후 (splits=[1,16] → s_m=1, s_n=16;  유닛 축 = s>1 인 축)
scf.for %un = 0 to 16 {pim.unit_dim="y"} {          // 유닛 축(바깥, 동시). n 을 16 유닛으로
  // sm 은 s_m=1 이라 순차 loop 1개(생략 가능), gm = %sm
  scf.for %sm = 0 to 32 {                            // 순차 축(안). pim.grid_dim 제거됨
    // gm = %sm ,  gn = %un * (16/16) + 0 = %un
    body(%sm, %un)
  }
}
```

규칙:
- 축 `d`(extent `G`, split `s`)를 `for %u=0..s { for %seq=0..G/s { %g = %u*(G/s) + %seq } }`로 치환. `s | G` 필요(분할 후보가 약수라 보장, §16.3).
- **유닛 loop**(`%u`)에는 `pim.unit_dim` 부착, **순차 loop**(`%seq`)는 `pim.grid_dim`을 **제거**(평범한 loop가 되어 hoisting 대상이 됨).
- **재배치**: 모든 유닛 loop를 바깥으로, 순차 loop를 안으로 모은다. 합법성은 `pim.grid_permutable == true` 확인(Phase 0). `false`(=`reduce_across_programs` 있음)면 해당 축은 재배치 대상에서 제외하거나 실패(초기엔 실패 + 진단).

### 재-hoisting (전송 절감의 실체, §15.6·§20.3)
치환·재배치 후 **`--pim-remove-buffers`를 다시 호출**한다. 순차 loop가 `pim.grid_dim`을 잃었으므로, 그 순차 축에 불변인 `pim.load`(예: n으로 쪼갠 경우의 B)가 순차 loop 밖으로 hoist되어 "유닛당 한 번만 로드"가 된다. **패스를 새로 구현하지 않고 기존 `hoistLoads`를 재사용**하는 것이 §16.3 step 4의 핵심.

권장 호출 순서: `--pim-apply-partition=splits=1,16 --pim-remove-buffers -canonicalize`.

### 재사용
- 인덱스 산술: `castTo`(build-grid).
- hoisting: `PIMRemoveBuffersPass`의 `hoistLoads`/`writesInLoop`/`mayAlias`(그대로, grid-loop skip은 순차 loop에 `pim.grid_dim`이 없으니 자연히 통과).
- 재배치: `scf` loop 재중첩은 body clone + `IRMapping`.

### 초기 제약
- `s | G` 필수(비약수 제외). 동적 grid(비상수 extent)는 치환은 가능하나 `G/s`가 SSA가 됨 — 초기엔 상수 extent만 지원(문서화), 동적은 후속.
- `pim.unit_dim`은 target이 유닛 dispatch로 바꾸기 위한 표식(§16.3 step 5). PIM Common은 표식만 단다.

### lit 테스트 (`test/Conversion/TTSharedToPIM/apply_partition.mlir`)
- RUN: `triton-shared-opt --pim-analyze-logical-work --pim-apply-partition=splits=1,16 --pim-remove-buffers %s`
- CHECK: 바깥 `pim.unit_dim="y"` loop, 안쪽 순차 loop(`pim.grid_dim` 없음), B의 `pim.load`가 순차 loop **밖**으로 hoist됨, 몸체 의미 보존.

---

## Phase 3 — `--pim-partition` reduction 비용 동인 확장 (§16.5 비용)

**파일:** `lib/Conversion/TTSharedToPIM/PIMPartitionPass.cpp`

### 추가할 것
`reduce_across_programs`가 비어 있지 않은(=Split-K 승격된) 함수에서, 출력 access의 전송을 **두 조각으로 분리**해 emit.

```text
final_output_bytes    = 출력 타일 bytes × Π g_(non-reduce 축)        // 불가피한 최종 쓰기 (S 무관)
partial_combine_bytes = (Π s_(reduce 축) − 1) × final_output_bytes    // host 결합일 때만 발생
partial_combine_ops   = (Π s_(reduce 축) − 1) × 출력 원소 수           // 결합 덧셈 횟수
```

- reduce 축의 유닛 분할 `s_k`는 승격으로 생긴 축의 split. 전액 병렬(`s_k = g_k = S`)이면 `partial_combine_bytes = (S−1)×final`.
- 이 셋을 func 속성으로: `pim.final_output_bytes`, `pim.partial_combine_bytes`, `pim.partial_combine_ops`, 그리고 combine 연산자는 이미 있는 `pim.reduction_combine`을 참조.
- **의도:** 지금 `access_bytes`가 "C ∝ S"로 뭉뚱그리던 것을, target이 "이 부분합을 전송으로 볼지(UPMEM) vACC 누적으로 볼지(AiMX)" 고를 수 있도록 **라벨로 분리**. 전송 시간·reduction 시간의 하드웨어 의존성은 여기서 넣지 않는다(Part 3의 중립 동인만).

### 후보 공간
Split-K factor `S`는 승격 후 `s_k`로 흡수되므로, **기존 `buildCostTable`의 축 집합에 k 축이 자동 포함**된다. 전송 표는 §16.2 공식 그대로(수정 최소). 위 세 동인만 추가 emit.

### 재사용
- `accessBytes`/`tileBytes`/`bytesOf`(element bytes, ceil)/`workingSetBytes`/`divisorsOf`/`buildCostTable` 그대로.
- 출력 access 식별: 이미 있는 `pim.role == "output"` + `reduce_across_programs` 축 교차.

### lit 테스트 (`test/Conversion/TTSharedToPIM/partition_reduction_cost.mlir`)
- 입력: 상수 grid로 특수화 + split-reduction 통과한 matmul.
- RUN: `triton-shared-opt --pim-analyze-logical-work --pim-partition %s`
- CHECK: `pim.final_output_bytes`, `pim.partial_combine_bytes`, `pim.partial_combine_ops`가 예상 수치로 emit (예: `M=N=256, S=64` → ops = 63×65536 = 4128768).

---

## Phase 4 — host `perf_model` 계약 + Triton `prune_configs_by` (스텁)

**파일:** `pim_backend/backend/`(host Python). 지금은 **중립 emit + 얇은 스텁**까지만(진짜 rate는 Target IR 단계에서).

### 계약 (하드웨어 무관 속성 → host)
컴파일러가 남기는 중립 동인(access별 `pim.access_bytes`·`pim.invariant_program_dims`, func의 `pim.working_set_bytes`·`pim.final_output_bytes`·`pim.partial_combine_bytes`·`pim.partial_combine_ops`·`pim.reduction_combine`)을 host가 읽는다. (MLIR module attribute를 파싱하거나, 컴파일 산출물 옆 작은 JSON으로 덤프 — 방식은 Phase 4에서 확정.)

### `perf_model` 형태 (Part 1·Part 3 설계)
```python
def pim_perf_model(**kw):        # kw: config(BLOCK_*), M,N,K, + 후보 splits/S
    # 중립 동인(위 속성 or config+shape에서 재계산) + target rate(하드웨어) 주입
    t_xfer   = total_bytes / hw.BW
    t_comp   = total_mac  / (U * hw.f)
    t_reduce = hw.reduce_cost(final_bytes, partial_bytes, partial_ops)   # target 함수(vACC vs host)
    return t_xfer + t_comp + t_reduce

@triton.autotune(configs=[...], key=['M','N','K'],
    prune_configs_by={'perf_model': pim_perf_model, 'top_k': 3,
                      'early_config_prune': working_set_prune})   # §16.7
```
- `hw`(BW, f, reduce_cost)는 **placeholder**로 두고, Target IR이 생기면 실제 값/함수로 교체.
- `reduce_cost`가 vACC(≈0 전송)/host(전송+덧셈)를 흡수 — Part 3의 `α·bytes/BW + β·ops/rate`.
- 런타임 grid 유지(캐싱 재사용 보존). 고정 shape만 opt-in 특수화(`pim.partition_cost` 표 캐싱).

---

## 5. 파이프라인 배선 · 등록

- `Passes.td`에 `PIMSplitReduction`(opt `factor`), `PIMApplyPartition`(opt `splits`) 추가.
- `TTSharedToPIM.h`에 생성자 선언, `RegisterTritonSharedDialects.h`에 등록, `CMakeLists.txt`에 소스 추가.
- **기본 `PIM_COMMON_PASSES`에는 넣지 않는다**(§20.3, target-called). `compiler.py`는 그대로.
- 상수 grid 특수화(launch-grid 상수 주입)는 별도 소훅으로 뒤에(§16.3) — 이 계획 범위 밖, Phase 3 표를 실제로 살릴 때.

---

## 6. 검증 계획 (전부 사용자가 직접 실행)

빌드 후:
```
lit test/Conversion/TTSharedToPIM test/Dialect/PIM -v
```
개별 확인:
```
triton-shared-opt --pim-split-reduction=factor=64 --pim-analyze-logical-work test/.../split_reduction.mlir
triton-shared-opt --pim-analyze-logical-work --pim-apply-partition=splits=1,16 --pim-remove-buffers test/.../apply_partition.mlir
triton-shared-opt --pim-analyze-logical-work --pim-partition test/.../partition_reduction_cost.mlir
```
기대: 위 각 Phase의 CHECK 항목. end-to-end로 matmul에 `PIM_COMMON_PASSES` → `split-reduction` → `apply-partition` → `remove-buffers`를 순서대로 걸어 의미 보존 + 속성 산출 확인.

---

## 7. 위험 · 미결정

- **FP 비결합**: Split-K는 결과를 미세하게 바꿈(§16.5). 알려진 trade-off. 테스트는 구조/속성 위주, 수치 동치는 요구하지 않음.
- **나머지 tile**: `S∤Kt`, `s∤G` 초기 미지원(padding은 target). 동적 grid도 후속.
- **상수 grid 전제**: `partition_cost` 표·Phase 3 수치는 grid 상수일 때만 완전. 비상수면 동인만 남고 host가 `g` 주입.
- **alias 가정**: hoisting은 base memref 단위 보수적 판정(§15.7) — 그대로 승계.
- **`grid_permutable` false 축**: reduce 축은 재배치 불가 — apply-partition에서 제외/실패로 처리.
