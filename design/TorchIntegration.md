# Torch → PIM integration

Torch 모델을 PIM에서 실행하는 경로는 **compile time**과 **runtime** 두 단계로 나뉜다.
compile time은 residency policy 결정, policy 제약 안의 autotuning, kernel 컴파일,
메모리 배치를 장치 할당 없이 끝낸다. runtime은 장치를 할당하고, 미리 컴파일된 stage를
장치에 결합하고, 분할된 FX graph를 순서대로 실행한 뒤 해제한다. 두 단계는 같은
프로세스에서 실행한다.

```text
입력: Torch layers, example inputs, target 이름, unit 수, target runtime option
                                │
════════════════════ COMPILE TIME (장치 할당 없음) ═════════════════════════════
 1. Target spec      등록된 target profile + 입력 unit 수 → PIMTargetSpec
 2. Stage 분석       torch.export → FX GEMM 분석 → layer별 stage와 stage class
 3. 가능한 정책     stage class별 최소 footprint → F ⊆ {P-1(모든 layer), P-2(layer 하나)}
 4. 제약 autotuning  stage class별 tune() 1회: P-2 예산 안의 후보만 측정 (정책 무관 cache)
                     측정 = weight 1회 upload 후 반복 실행 시간 E (+ U, footprint)
 5. 정책·조합 선택   joint: F의 정책마다 최속 조합 또는 DP → 요청당 시간 최소 정책
                     layer_first: layer 하나 기준 최속 조합 → P-1에 맞으면 P-1, 아니면 P-2
 6. 컴파일·배치      PIM stage: Triton → TTIR → TTShared → PIM Common IR → Target IR
                     CPU partition: Inductor, residency plan, target 주소 배치
════════════════════ RUNTIME (요청마다) ═══════════════════════════════════════
 7. open             장치 할당, 실제 unit 수 == compile time unit 수 확인
                     stage를 장치에 결합 (target context, DPU program handle)
 8. 실행             policy대로 weight 배치, 분할된 FX graph 실행
 9. close            해제
```

4단계의 측정은 후보마다 짧은 `PIMSession`을 열고 닫는다. 이것은 측정을 위한 임시 장치
사용이며 runtime의 할당과 수명이 다르다.

---

## 1. 용어

| 용어 | 정의 |
|---|---|
| stage | `LayerPlan` 안의 PIM 실행 단위 하나 (`_LayerGemm`). weight GEMM 하나, 또는 같은 shape의 독립 activation GEMM 묶음 하나 |
| weight stage | weight operand를 장치에 상주시키는 stage. instance 1, M/N/K round 1, `p_k=1` |
| activation stage | 두 operand가 모두 activation인 stage (QKᵀ, PV). `p_k=1`, M/N/K round ≤ 4 |
| stage class | `(M,N,K)`, dtype, op, weight mask, instance 수가 같은 stage의 모음. autotuning과 config 선택의 단위 |
| footprint | stage 하나가 unit 하나에 상주시키는 bytes: weight / state / input scratch / output scratch |
| residency policy | P-1 `model`: 모든 layer의 weight와 state 상주. P-2 `layer`: layer 하나의 weight와 state만 상주, layer마다 교체 |
| E | weight가 이미 상주한 상태에서 stage 한 번의 실행 시간 |
| U | stage weight의 upload 시간 |
| Cap | unit 하나의 용량 = `banks_per_unit × bank_capacity_limit_bytes` |

---

## 2. Target spec: 입력과 등록 profile

target 이름과 unit 수는 입력으로 받는다. 용량, bank 구조, lane 수는 `backend/target.py`에
등록된 profile을 쓴다. compile time은 하드웨어를 조회하지 않는다.

```python
spec = target_spec("aimx", num_units=1, base_fpga=0)
spec = target_spec("upmem", num_units=2048, nr_tasklets=16)
```

| 값 | 소유자 |
|---|---|
| M/N/K, compute op, dtype/layout별 access dependency | `PIMProblem` |
| target, unit 수, bank 구조, 용량 | `PIMTargetSpec.common_profile` |
| `nr_tasklets`, `host_threads` / `batch`, `base_fpga` | target runtime config |
| Block–Group–Execution Tile, `[p_m,p_n,p_k]` | `PIMConfig` |
| total groups, Execution Tiles, scheduling rounds | `pim_hierarchy()` |

runtime open에서 adapter가 실제 unit 수를 보고한다. compile time의 unit 수와 다르면
예외로 멈춘다. 컴파일된 artifact, policy, 주소 배치가 모두 그 unit 수를 전제로 하기 때문이다.

---

## 3. Stage 분석

각 layer를 `torch.export`로 내보내고 `analyze_fx_gemms`로 GEMM을 찾는다. stage 구성은
`build_layer_plan`이 정한다. 같은 깊이, 같은 op, 같은 `(M,N,K)`와 dtype, 같은 config의
독립 activation GEMM은 instance로 묶이고, weight GEMM은 단일 instance stage가 된다.
분석 단계는 stage class마다 같은 임시 config를 넘겨 이 묶음을 그대로 얻는다. 최종 config도
stage class 단위로 정하므로 묶음이 바뀌지 않는다. 다음 layer의 example input은 CPU eager
실행으로 얻는다.

Transformer layer 하나의 예:

| stage | stage class `(M,N,K)` | weight | instances |
|---|---|---|---:|
| q/k/v/o_proj | (1024,4096,4096) | O | 1 |
| QKᵀ | (1024,1024,128) | X | 32 |
| PV | (1024,128,1024) | X | 32 |
| gate/up_proj | (1024,11264,4096) | O | 1 |
| down_proj | (1024,4096,11008) | O | 1 |

stage 9개, stage class 5개. 측정은 stage class마다 한 번이고, 같은 class의 stage는
측정 결과와 config를 공유한다.

---

## 4. Residency policy 결정

### 4.1 Footprint 추정

config 하나의 footprint는 target model이 추정한다 (`autotuner/model.py`).

```text
access bytes   = TargetModel.access_allocations(problem, config)   A(m,k), B(k,n), C(m,n)
overhead bytes = TargetModel.overhead_allocations(problem, config) (AiMX: PMM, Arel, descriptor)

weight  = weight operand access의 bytes
state   = overhead + (activation scratch가 stage 전용인 target이면) 나머지 access
input   = (activation scratch를 stage끼리 공유하는 target이면) weight 아닌 input access
output  = (공유하는 target이면) output access
```

activation scratch를 stage끼리 공유하는지는 target model의 속성이다. UPMEM은 모든 stage가
같은 MRAM input/output 영역을 쓰고, AiMX는 stage마다 자기 buffer를 가진다. 이 분류는 각
adapter가 residency planner에 보고하는 `StageRequirement`와 같다.

### 4.2 최소 footprint와 policy 판정

stage class마다 policy 제약을 만족하는 config 중 footprint 합이 가장 작은 것을 찾는다.
weight stage는 round 1이므로 block, group, partition을 정하면 tile이
`ceil(total_groups / p)`로 정해진다. activation stage는 M/N/K round 곱이
`max_initial_scheduling_rounds`(기본 4) 이하인 tile만 본다. 이 상한이 없으면 최소
footprint가 "가장 작은 Block Group 하나씩, round 수천 번"인 config가 되어 측정할 수 없다.
이 상한은 tuning의 모든 후보에도 같은 제약으로 적용된다. 탐색은 계산만 하고 측정하지 않는다.

```text
m_min(class) = min_{c ∈ C_ws(class)} footprint(c)
```

정책마다 아래 식에 `m_min`을 넣어 **가능한 정책 집합 F**를 구한다. 정책을 여기서 하나로
정하지 않는다. 최종 정책은 측정 뒤 §6의 전략이 정한다.

```text
P-1: Σ_layer Σ_stage (weight + state) + max input + max output ≤ Cap
P-2: max_layer Σ_stage (weight + state) + max input + max output ≤ Cap
```

최소 footprint config가 실제로 존재하므로 F에 든 정책은 실현 가능하다. 다만 target model은
추정치이다. 실측 footprint로 고른 조합이 어느 정책에도 맞지 않으면 §6에서 예외가 난다.
F가 비면 tuning 전에 예외가 난다.

### 4.3 Policy가 정하는 것

| policy | weight stage 제약 | 목적함수 (요청 하나) |
|---|---|---|
| P-1 `model` | round 1, `p_k=1`, footprint ≤ 예산 | Σ count·(E + U/R) |
| P-2 `layer` | 같음 | Σ count·(E + U) |

R은 session 하나가 처리하는 요청 수다. P-1은 weight를 session에 한 번 올리므로 요청 하나는
U의 1/R만 낸다.

activation stage는 `p_k=1`, M/N/K round ≤ 4, footprint ≤ 예산을 지킨다. weight가 없으므로 U=0이다.

tuning 예산은 **P-2 기준**이다. P-1의 계수는 P-2 이상이므로 P-1에서 가능한 config는 P-2에서도
가능하다. 그래서 P-2 예산으로 한 번 측정한 후보 집합이 두 정책 모두의 후보를 포함한다. stage
class j의 예산은 다른 class가 모두 최소 footprint일 때 남는 용량이다.

```text
budget_j = Cap − Σ_{i≠j} bytes_policy(i, m_min(i)) − scratch_min
```

---

## 5. 제약 autotuning

### 5.1 같은 autotuner, policy가 바꾸는 입력

autotuner 엔진(`autotuner/`: space, `ConstraintEngine`, search, cache)은 하나이다. policy는
엔진에 넘기는 네 가지를 바꾼다.

| 입력 | 기존 kernel autotuning (`autotune_torch.py`) | model compile |
|---|---|---|
| 제약 | 정적 검사 | 정적 검사 + `tune(constraint=...)` |
| 초기 round 패턴 | (1,1,1), (2,1,1), … | weight stage는 (1,1,1)만 |
| 초기 후보 | heuristic seed, cache 유사 기록 | + 최소 footprint config (`tune(initial=...)`) |
| `benchmark_fn` | `torch.compile(mm)` 한 번 호출 시간 | `PIMSession` stage, weight 1회 upload 후 E |
| cache context | `torch-gemm-semantic-k-v2` | `torch-gemm-semantic-k-v2\|model\|<op>\|w<weight mask>\|x<instances>` (정책 무관, 측정 공유) |

`ConstraintEngine.check`는 기존 검사를 모두 통과한 config에 대해 마지막으로
`constraint(problem, config)`를 부른다. 문자열을 돌려주면 거부 사유이다. `tune()`은
초기 후보, structured 후보, mutation 후보 모두에 이 검사를 적용하므로 측정되는 후보는
전부 제약을 만족한다.

### 5.2 측정

stage class마다 stage 하나만 담은 Torch module을 만든다. weight stage는 weight buffer와
`mm` 하나, activation stage는 instance 수만큼의 독립 `mm`이다. 후보 config마다:

```text
with PIMSession(spec) as session:
    session.prepare([module], inputs, configure=→ 후보 config)
    footprint = session.backend.requirements(plans)          실측 slot bytes
    U = time(session.backend.model.enter_layer(plan))        weight 1회 upload
    warmup × w, E = median(run_model × r)                    activation만 전송
→ PIMBenchmarkResult(total_ms=E,
                     timings=(("weight_upload", U),),
                     physical_allocations=((weight, state, input, output),))
```

`physical_allocations`는 search feature의 memory pressure와 조합 선택에 함께 쓰인다.
같은 config가 이미 cache에 있으면 측정하지 않는다.

**후보마다 새 프로세스.** 위 측정은 `multiprocessing` spawn으로 만든 새 프로세스에서 한다
(`measure_isolated`). weight는 shape만 넘기고 자식이 같은 shape의 난수로 만든다(E, U,
footprint는 값과 무관하다). device context, stub 라이브러리, host DMA buffer가 자식과 함께
사라지므로 후보가 늘어도 누적되지 않는다. runtime이 `exit`나 segfault로 끝나면 그 후보만
`correctness=False`(timings에 exit code)로 기록하고 탐색을 계속한다. 이 후보는 선택에서 제외된다.

**후보마다 cache 저장.** `tune()`은 새로 잰 후보마다 `PIMConfigCache.store_partial`로 기록을
남긴다. 이 entry는 `complete: false`라 lookup에서는 miss이고, 다음 탐색은 남은 기록을 다시 재지
않고 이어서 진행한다. 탐색이 끝나면 `store`가 best와 함께 `complete: true`로 덮는다.

---

## 6. 조합 선택

stage class j가 config c를 고르면 policy 식에 들어가는 bytes와 시간은 다음과 같다.
`n_j`는 모든 layer의 stage 수 합, `l_j`는 layer 하나의 최대 stage 수이다.

```text
                 weight 계수   state 계수
P-1 model           n_j          n_j
P-2 layer           l_j          l_j

bytes_policy(j, c) = a_j·weight(c) + b_j·state(c)
scratch(c_*)       = max_j input(c_j) + max_j output(c_j)
time(j, c)         = n_j·(E(c) + share·U(c)),  share = 1/R (P-1), 1 (P-2)
```

layer 구성이 같으면 이 식은 `plan_residency`와 정확히 같다. layer마다 구성이 다르면
P-2의 `l_j`가 layer별 최대치라서 실제보다 크게(안전하게) 잡힌다.

```text
1. stage class마다 측정 기록 중 time 최소 config → 합이 Cap 이하이면 그대로 사용
2. 아니면 multiple-choice knapsack DP:
     scratch 상한 (I, O)을 기록에 나타난 값들로 열거하고,
     input ≤ I, output ≤ O인 후보만으로
     Σ_j bytes_policy(j, c_j) ≤ Cap − I − O 에서 Σ_j time(j, c_j) 최소를 DP로 구한다
     (bytes는 MiB 단위로 올림)
3. 가능한 조합이 없으면 그 정책은 제외
```

용량이 남으면 1단계에서 끝나므로 DP를 쓰지 않는다. DP는 빠른 config가 더 큰 footprint를
쓰고, 그 합이 용량에 닿을 때만 결과를 바꾼다.

### 6.1 정책 선택 전략

| 전략 | 동작 | 최적성 |
|---|---|---|
| `joint` | F의 정책마다 위 1–3을 풀고 요청당 시간이 작은 정책을 고른다. 같으면 P-1 | 측정된 후보 안에서 (정책, config) 전체 최적 |
| `layer_first` | P-2 예산으로 share=0(실행 시간만) 최속 조합을 고른 뒤, 그 조합이 P-1 식에 맞으면 P-1, 아니면 P-2 | P-1에 맞추려고 config를 줄이는 선택은 하지 않는다 |

```text
joint가 layer_first보다 나아지는 경우 (같은 측정 후보 기준)
  layer_first의 조합이 P-1 용량을 넘지만, 조금 느린 조합은 P-1에 들어가고
  Σ_j n_j·[E_j(느린) − E_j(빠른)]  <  Σ_j n_j·U_j·(1 − 1/R)
  → layer_first는 P-2(매 요청 upload), joint는 P-1

두 전략이 같은 결과를 내는 경우
  · layer_first 조합이 P-1에 들어감 (용량 여유)
  · 최소 footprint로도 P-1이 안 됨 (P-2만 가능)
  즉 차이는  Cap / M_fast < L ≤ Cap / M_min  인 layer 수 L에서만 생긴다
  (M_fast: layer_first 조합의 layer당 bytes, M_min: 최소 footprint의 layer당 bytes)
```

---

## 7. 컴파일과 runtime 분리

`PIMSession`은 두 가지 순서를 지원한다.

```text
compile time 경로 (model compile)
  session = PIMSession(spec)
  session.compile(layers, inputs, configure=선택 결과)     장치 할당 없음
  with session:                                           open: 할당 + unit 수 확인 + stage 결합
      session.run_model(*inputs)

session-first 경로 (기존 예제)
  with PIMSession(spec) as session:                       open: 할당
      session.prepare(layers, inputs, configure=...)      컴파일 + 배치 + stage 결합
      session.run_model(*inputs)
```

adapter 계약:

| 메서드 | 시점 | 장치 사용 | 하는 일 |
|---|---|---|---|
| `begin_model()` | compile | 없음 | target model runtime 생성 |
| `compile_invocation(stage, inputs)` | compile | 없음 | Target IR → host/stub 라이브러리 |
| `resources()` | compile | 없음 | spec의 unit 수와 용량 |
| `requirements(plans)` | compile | 없음 | stage footprint 보고 |
| `prepare_model(plans, residency)` | compile | 없음 | policy 기록, 주소 배치, DPU program 링크 |
| `open()` | runtime | 할당 | 장치 확보, 실제 unit 수 |
| `bind_model(plans)` | runtime | 있음 | stage context / handle 생성 |
| `weight_table(plans)` | runtime | 없음 | weight 배치표 |
| `enter_layer` / `run` / `exit_layer` | runtime | 있음 | policy대로 upload, 실행, 해제 |
| `close()` | runtime | 해제 | context와 장치 해제 |

`session.compile`이 고른 policy가 model compile의 policy와 다르면 예외로 멈춘다.

---

## 8. API

```python
from torch_backend import compile_model

compiled = compile_model(
    layers, example_inputs,
    target="aimx", units=1, requests=R, strategy="joint",   # 또는 "layer_first"
    runtime_options={"base_fpga": 0}, cache_path=..., max_benchmarks=20, warmup=1, repeat=3,
)
compiled.report          # policy, stage class별 측정 기록과 선택, 예산
with compiled.session as session:
    output = session.run_model(*inputs)
```

| 모듈 | 역할 |
|---|---|
| `autotuner/constraints.py` | `ConstraintEngine(constraint=...)` |
| `autotuner/search.py` | `tune(constraint=..., initial=...)` |
| `autotuner/model.py` | `TargetModel.shared_activation_scratch` |
| `torch_backend/model_policy.py` | stage 분석, footprint, 최소 footprint, policy 판정, 제약 |
| `torch_backend/model_compile.py` | stage 측정, 제약 tuning, 조합 선택, `compile_model` |
| `backend/runtime.py` | `PIMSession.compile`, open 시 stage 결합 |
| `backend/aimx_adapter.py`, `aimx_model.py` | `prepare_model`/`bind_model` 분리 |
| `backend/upmem_adapter.py`, `upmem_model.py` | `prepare_model`/`bind_model` 분리 |

---

## 9. Kernel lowering

stage 하나의 컴파일 경로는 기존과 같다.

```text
aten.mm / aten._int_mm / aten.bmm
  -> PIMProblem(M,N,K, compute op, pim.access dependencies)
  -> PIMConfig (stage class 선택 결과)
  -> semantic GEMM TritonTemplate   source grid = M/N, source reduction = complete K
  -> TTIR -> TTShared IR -> PIM Common IR
       Block -> Block Group -> Execution Tile, [p_m,p_n,p_k], pim.place -> pim.combine
  -> UPMEM IR | AiMX IR -> host/stub
```

`PIMConfig`는 보정 없이 backend option으로 이동한다. `block_*`는 template constexpr,
나머지 계층과 mapping 값은 PIM compiler option이다. Torch source에는 `SPLIT_K`, `pid_k`,
`partial[p_k,M,N]`가 없고, `p_k`와 `tile_k`가 하나의 semantic reduction domain을 나눈다.
model 경로는 `p_k=1`만 쓴다.

template 조건:

- 2-D CPU `float32` 또는 `int8` 입력 두 개, 출력은 `float32` 또는 `int32`
- static M/N/K
- A/B row-major contiguous; 아니면 lowering이 copy를 삽입
- M/N/K tail은 masked access와 Common local trip count로 처리
- template 주변 fusion은 지원하지 않음

---

## 10. 제한

- stage class 단위로 config를 고른다. 같은 class의 stage들은 같은 config를 쓴다.
- policy 판정은 target model 추정 footprint를 쓴다. 선택은 실측 footprint를 쓴다.
- 측정은 stage를 하나씩 실행한다. stage 사이의 host/CPU 겹침은 반영하지 않는다.
- compile time artifact는 프로세스 메모리에 있다. 프로세스 사이의 저장은 지원하지 않는다.
