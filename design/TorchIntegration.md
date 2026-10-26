# Torch → PIM integration

Torch 모델을 PIM에서 실행하는 경로의 Torch 쪽 설계를 다룬다. 범위는 두 가지다.

1. **Torch FX graph 분석**: layer 하나의 FX graph에서 GEMM을 찾고, 독립 GEMM들을 더 큰
   GEMM으로 합치는 packing 변환(규칙 ①~⑤)을 적용한다.
2. **Torch runtime memory management**: 요청 단계(phase)마다 layer를 slot과 group에 대응시키고,
   key/value cache를 장치에 상주시키는 메모리 table과 그 실행 규칙.

TritonPIM의 compile과 lowering(Triton → TTIR → TTShared → PIM Common IR → Target IR,
stub·descriptor·program 생성)은 이 문서의 범위 밖이다. 대상 코드는 `torch_backend/`와 session
runtime(`backend/runtime.py`, `backend/residency.py`, `backend/*_model.py`)이다.
target별 차이는 autotuner의 target model(`autotuner/model.py`)이 공개하는 값과 adapter가 보고하는
숫자로만 들어온다.

```text
입력: phase마다 (Torch layers, example inputs), target, unit 수, kv_inputs, repeats
                                   │
═══════════════════════ COMPILE TIME (장치 할당 없음) ═══════════════════════════
 1. Target spec        target_spec(target, num_units, **runtime_options), 용량 C
 2. FX 분석            phase마다 첫 layer: export → normalize → freeze → packing (§2) → stage class
 3. 튜닝               class마다 weight(또는 cache) 상주 E, 첫 업로드 U, footprint 측정 (§4.1)
 4. cache 예약         측정 후보의 가장 큰 cache footprint로 KV 상한 (kv_bytes_bound)
 5. 선택               phase마다 packing과 class별 config (§4.2)
 6. 컴파일             PIMSession.compile: 모든 phase의 모든 layer, plan_memory (§3)
═══════════════════════ RUNTIME ═════════════════════════════════════════════════
 7. open / bind        장치 할당, unit 수 확인, stage handle/context 생성
 8. write_kv           host cache 위치 범위 → 다음 cache stage 실행 직전에 장치로 전송
 9. run_model(phase)   layer 진입: (phase, group)이 바뀌면 그 group의 weight 업로드
10. close              해제
```

---

## 1. 용어

| 용어 | 정의 |
|---|---|
| unit | 장치 하나. AiMX는 FPGA, UPMEM은 DPU |
| phase | 요청 단계(prefill, decode). `PIMSession.plans[phase]`, `LayerPlan.phase` |
| stage | `LayerPlan` 안의 PIM 실행 단위(`_LayerGemm`). weight GEMM 하나, 또는 같은 shape의 독립 activation/cache GEMM 묶음 하나 |
| instance | stage 안의 독립 GEMM 하나. `bmm`은 batch 차원이 instance |
| stage class | `StageKey`(`(m, n, k)`, dtype, op, weight mask, instance 수, cache mask, cache 전치)가 같은 stage의 모음. 튜닝과 config의 단위 |
| packing | 독립 GEMM 여러 개를 GEMM 하나로 합치는 FX 변환 (§2) |
| slot | layer 하나의 stage들이 쓰는 weight + state 영역 |
| group | 함께 업로드되는 layer 묶음. layer l은 slot `l mod slots`, group `⌊l / slots⌋` |
| scratch | 모든 stage가 처음부터 함께 쓰는 input/output 영역 |
| KV 영역 | layer마다 KEY, VALUE 영역. phase와 무관하게 같은 offset |
| E | weight(또는 cache)가 상주한 상태에서 stage 한 번의 실행 시간 |
| U | stage weight를 한 번 올리는 시간 (pack + 전송) |
| repeats | 한 요청이 그 phase를 실행하는 횟수. prefill 1, decode 생성 token 수 − 1 |
| C | unit 하나의 용량 = `banks_per_unit × bank_capacity_limit_bytes` |

---

## 2. Torch FX graph 분석

### 2.1 흐름

```text
torch.export(layer)
  → normalize_fx_gemms        linear / addmm → mm
  → _freeze_graph             weight와 weight만 읽는 view를 상수로
  → pack_gemms                CSE → ② n 잇기 → ① m 쌓기 → ③ block-diagonal → ④/⑤ cache window
  → build_layer_plan          같은 depth·shape·config·cache 읽기의 독립 activation GEMM → instance 묶음 stage
  → analyze_layer             stage class와 원래 GEMM 이름 → class 대응
```

모델 코드는 head마다 `mm`을 쓰거나 Hugging Face 방식(`repeat_kv` + `matmul`)을 그대로
쓴다. block-diagonal 같은 배치는 모델 코드가 아니라 compiler가 정한다
(`torch_backend/fx_packing.py`, `torch_backend/model.py`).

출력 행이나 열이 하나인 GEMM(`M = 1` 또는 `N = 1`)은 GEMV(matvec)로 실행한다
(`execution.gemm_problem`). decode의 `x·W`, `q·Kᵀ`, `p·V`가 여기에 해당한다.

### 2.2 왜 세 가지 변환인가

GEMM 두 개 `C₁ = A₁B₁`, `C₂ = A₂B₂`를 GEMM 하나로 합치려면 결과를 어느 축으로 이어 붙일지
정해야 한다. 축마다 필요한 조건이 다르고, 이 셋이 전부이다.

| 변환 | 공유하는 operand | 합치는 축 | 0 블록 | 계산량 | 모델에서 나오는 곳 |
|---|---|---|---|---|---|
| ① m 쌓기 | B | m | 없음 | 같음 | GQA의 QKᵀ, PV (query 여러 개가 KV 하나를 공유) |
| ② n 잇기 | A | n | 없음 | 같음 | q/k/v projection, gate/up projection (같은 입력) |
| ③ block-diagonal | 없음 | k (대각) | 있음 | p배, target padding이 흡수할 수 있음 | MHA의 head끼리, GQA의 KV group끼리 |

적용 순서는 ② → ① → ③이다(`pack_gemms`). 0이 없는 변환을 먼저 적용해야 ③이 다루는 instance가
줄고 0 블록이 늘지 않는다.

### 2.3 ① m 쌓기 (GQA)

GQA에서는 query head g개가 같은 K, V를 쓴다. query들을 m 방향으로 쌓으면 K를 한 번만 쓰는
dense GEMM 하나가 된다.

```text
repeat_kv (K를 g번 복제)                  ① m 쌓기
  Q0 @ K0ᵀ = S0                             ┌────┐                ┌────┐
  Q1 @ K0ᵀ = S1                             │ Q0 │                │ S0 │
  Q2 @ K0ᵀ = S2                             │ Q1 │  @  K0ᵀ    =   │ S1 │
  Q3 @ K0ᵀ = S3                             │ Q2 │    (D × S)     │ S2 │
                                            │ Q3 │                │ S3 │
                                            └────┘                └────┘
                                           (gS × D)              (gS × S)

PV:  [P0; P1; P2; P3] (gS × S) @ V0 (S × D) = [O0; O1; O2; O3] (gS × D)
```

**판정.** 같은 depth·shape·dtype의 독립 activation GEMM 묶음(`_instance_groups`) 안에서
B operand가 **같은 FX node**인 GEMM들을 쌓는다(`_stack_shared_rights`). view 계열 op를 따라
올라가 원본을 비교하지 않는다. 따라서 모델이 query head마다 같은 KV node를 쓰면(예제의
`--kv share`) 쌓이고, Hugging Face `repeat_kv`처럼 head마다 별도의 B node를 만들면(`--kv repeat`)
쌓이지 않는다. `bmm`과 cache를 읽는 GEMM은 묶음 대상이 아니다.

### 2.4 ② n 잇기 (q/k/v, gate/up)와 weight 사본

같은 입력 x를 쓰는 weight GEMM은 weight를 n 방향으로 이어 GEMM 하나로 만들고, 출력을 열
방향으로 나눈다(`_concatenate_weights`). host에는 이어 붙인 행렬을 두지 않는다: packed B는 원래
frozen weight들의 `ConcatenatedWeight`이고, 연속 행렬은 weight 업로드(또는 reference 계산) 동안에만
`tensor()`로 만든다. stage 실행 인자로는 weight를 넘기지 않는다.

```text
                ┌──────────┬──────┬──────┐       ┌──────────┬──────┬──────┐
  x (S × H)  @  │    Wq    │  Wk  │  Wv  │   =   │    q     │  k   │  v   │
                └──────────┴──────┴──────┘       └──────────┴──────┴──────┘
                     (H × (n_q + n_k + n_v))             → 열 방향으로 split

gate/up:  x @ [Wg | Wu] → [g | u] → silu(g) * u
```

입력 A의 전송이 1/3(q/k/v), 1/2(gate/up)로 줄고 stage 수가 준다. AiMX는 B를 n 4096 폭으로
저장하므로(`AiMXModel._compute_unit`) n이 작은 GQA의 Wk, Wv의 저장 낭비가 줄어든다.

frozen weight는 checkpoint tensor 또는 그 전치 view일 수 있다. 사본 여부는 stage 종류가 정한다.

| stage | weight operand | 근거 |
|---|---|---|
| GEMM (matmul) | view를 `contiguous()` 사본으로 바꿔 compile·업로드 | `_LayerGemm.prepare`, lowering의 `require_contiguous` |
| GEMV (matvec) | view를 stride대로 compile하고 업로드 때 그대로 읽음 | `lowering.make_pim_gemm_lowering` |

GEMM stage의 사본은 frozen graph가 가진 원본과 별도로 남으므로 view weight는 host에 두 벌이 된다.

### 2.5 ③ block-diagonal

공유가 없는 instance는 A를 k 방향으로 나란히 두고 B를 대각에 둔다(`_block_diagonal`).

```text
                                ┌──────┬──────┐
  ┌──────┬──────┐               │  B1  │  0   │          ┌──────┬──────┐
  │  A1  │  A2  │       @       ├──────┼──────┤    =     │  C1  │  C2  │
  └──────┴──────┘               │  0   │  B2  │          └──────┴──────┘
     m × p·k                    └──────┴──────┘             m × p·n
                                  p·k × p·n
```

0 블록은 실제로 곱해진다. target이 그 자리를 어차피 padding으로 계산할 때만 공짜이다.
A'와 B'는 CPU partition에서 만든다(`constant_pad_nd` + `cat`).

### 2.6 ③의 판정: target model

①과 ②는 계산이 늘지 않으므로 공유가 있으면 항상 적용한다(`PIMTorchConfig(pack=False)`인
GEMM은 제외). ③의 묶음 수 p는 autotuner 공개 API로 계산한 정적 비용으로 정한다
(`fx_packing._cost`). 실행 seed를 고르지 않으며 측정도 하지 않는다.

```text
for p in instance 수의 약수:
    problem_p  = gemm_problem(m, p·n, p·k)        (stage_target spec으로)
    config_p   = rank_block_groups 상위 Block Group의 tile_ladder 마지막 점(최대 병렬)
                 + best_partitions 첫 partition
    compute(p) = (I / p) · useful_tiles · scheduled_compute(problem_p, config_p)
    bytes(p)   = (I / p) · useful_tiles · Σ access_allocations(problem_p, config_p)
p* = argmin (bytes(p), −p)     단, compute(p) ≤ compute(1)
```

| target | `scheduled_compute` | 결과 |
|---|---|---|
| AiMX | m은 batch, n은 4096/batch, k는 MAC 단위로 올린 값 | 0 블록이 padding 자리를 채우면 p > 1 |
| UPMEM | `prod(execution_extent)` (기본 구현) | 0도 계산으로 셈 → 대개 p = 1 |

### 2.7 key/value cache를 읽는 GEMM: ④ k-window, ⑤ n-window

**내용**
host key/value cache는 layer마다 `[kv_heads, positions, head_dim]`이다. B가 KV head 하나 `cache[h]`
(또는 그 전치)인 GEMM에는 ①~③을 적용하지 않는다. cache는 장치에 그대로 있어야 하므로 B를
잇거나 대각으로 만들 수 없다. cache stage는 instance마다 unit 하나 이상을 쓰고 모든 instance가 한
round에 돌아야 하므로, 한 묶음의 GEMM 수가 unit 수보다 많을 때만 이어진 KV head를 읽는 GEMM끼리
묶어 B를 그 head들의 합집합 window(`cat(cache[h], dim=1)`)로 만든다. 묶음 수는 unit 수 이하의 가장
큰 약수다. window는 token-row view `[positions, kv_heads·head_dim]`의 열 범위로 적는다
(`stage.kv_windows`, `residency.token_rows`). stage는 cache operand 자리에 host cache 자체를 받으므로
합집합 window는 host에서 만들지 않는다(reference 계산만 만든다).

**근거**
- `fx_packing._pack_cache_windows`, `cache_window`. 묶은 GEMM의 origin은 None이라 member로 나눠 재지 않는다.
- packed GEMM(origin이 있는 것)이 cache를 읽으면 `build_layer_plan`이 예외를 낸다.
- 같은 KV head를 읽는 query head(GQA)는 CSE 뒤 같은 window를 읽으므로 같은 묶음 안의 행이 된다.

**예시**

| 규칙 | window가 놓인 B 축 | 묶은 GEMM | 버리는 것 | 나오는 곳 |
|---|---|---|---|---|
| ④ k-window | k (B = windowᵀ) | `[A₁ 0; 0 A₂] @ [W₁ \| W₂]ᵀ` | A의 0 | QK (`q_h @ K_jᵀ`) |
| ⑤ n-window | n (B = window) | `[A₁; A₂] @ [W₁ \| W₂]` → 행마다 자기 열 | 교차 항 | PV (`p_h @ V_j`) |

| target (head 32, m = 1) | unit | 묶음 | 결과 |
|---|---|---|---|
| UPMEM | DPU 2048 | 32 ≤ 2048 → 묶지 않음 | head마다 GEMV, 0 없음 |
| AiMX | FPGA 2 | 32 → FPGA당 16 head | QK `(16, C, 16·head_dim)`, PV `(16, 16·head_dim, C)` × 2 |

두 target의 decode layer는 같은 graph다: query head h마다 `q_h @ K[h // g]ᵀ`, `p_h @ V[h // g]`,
cache는 오른쪽 operand(`examples/aimx/llama_decode.py`, `examples/upmem/e2e/llama_workloads.py`).
host cache의 위치는 토큰 위치 그대로이고 물리 배치는 adapter가 정한다(§3.4).

### 2.8 config 대응

packing 후에는 원래 GEMM 여러 개가 packed GEMM 하나가 된다. `build_layer_plan`은 원래 GEMM
이름마다 config를 요구하고, 한 packing에 속한 원래 GEMM은 같은 config여야 한다(다르면 예외).
`analyze_layer`의 대응표(원래 이름 → stage class)도 packed 기준으로 만든다. 모든 layer의
GEMM 이름이 분석한 layer와 같아야 하며, 없는 이름은 `KeyError`로 드러난다(`compile_model.configure_with`).

### 2.9 UPMEM에서의 instance 배치

UPMEM은 DPU를 instance 수만큼 나눠 실행한다. instance 하나가 쓰는 DPU 수는 config의
`p_m·p_n·p_k`이고, 동시에 실행되는 instance 수는 `p_b = min(instances, units / (p_m·p_n·p_k))`이다
(`execution.select_layer_config`). DPU 하나는 Execution Tile 하나를 맡고, 그 안에서 tasklet이 행을 나눈다.

AiMX와 packing 효과가 다른 이유는 instance를 맡는 장치 수와 계산 단위이다. AiMX는 instance 하나를
FPGA 하나가 계산할 수 있어 ①에서 B를 한 번만 받고, MAC 단위가 넓어 ③의 0이 padding에 흡수된다.
UPMEM은 instance를 DPU 여러 개로 나누므로 DPU마다 operand 사본이 필요하고, 0이 그대로 계산된다.

---

## 3. Torch runtime memory management

### 3.1 메모리 table

모든 unit이 같은 배치이므로 unit 메모리 base로부터의 offset table 하나를 쓴다
(`backend/residency.py:plan_memory`, `MemoryTable`). phase 영역은 offset 0부터 위로, key/value cache는
용량에서 아래로 쌓는다.

```text
offset 0                                                   kv_base                    C
│ phase 영역 (phase마다 offset 0부터, 서로 덮어씀)          │ key/value cache            │
│ slot0 W │ slot1 W │ … │ slot0 S │ slot1 S │ … │ in │ out │ … │ L1 V │ L1 K │ L0 V │ L0 K │
```

| 항목 | key (`MemoryTable.entries`) | 규칙 |
|---|---|---|
| weight / state | `(phase, layer, stage, operand)` | slot과 group에 속함. 같은 slot의 layer는 같은 offset |
| input / output | `(phase, layer, stage, operand)` | phase의 scratch. 크기는 stage 최댓값 |
| KEY / VALUE operand | `(phase, layer, stage, operand)` | 그 layer cache 항목의 offset을 그대로 씀 |
| cache | `("kv", layer, KEY/VALUE)` | phase 없음. layer마다 key, value 순서로 용량에서 아래로 |

- cache 영역 크기는 그 영역을 읽는 stage operand가 보고한 unit당 bytes이며, 같은 영역을 읽는 stage는 bytes가 같아야 한다(다르면 예외).
- cache를 처음 읽는 phase 전의 phase는 cache 범위까지 unit 전체를 쓰고, 그 phase부터는 `kv_base` 아래에서 끝난다.
- `bytes_per_unit`은 한 시점에 unit이 담는 최대 bytes(cache 전 phase 영역, 또는 cache + 그 뒤 phase 영역)이다.

| | AiMX | UPMEM |
|---|---|---|
| state | slot마다 descriptor chain, PMM | 0 (program은 IRAM) |
| operand bytes | 256 KiB 배수로 올림 (`OPERAND_ALIGNMENT`) | binding 크기 그대로 |
| 주소 | table을 slot 경계에서 잘라 region(≤ `1023 × 2 MiB`)으로 묶고 FPGA마다 `aimx_model_region_create`. cache 항목은 별도 region | `MRAM_BASE + offset`으로 stage program을 link (`_build_program`) |
| phase | 하나만 (`place`가 `NotImplementedError`) | 모두 |

### 3.2 slot, group과 대응표

phase마다 slot 수를 정하고 layer를 slot과 group에 대응시킨다(`_plan_phase`).

```text
A   = C                    cache를 처음 읽는 phase 전
      kv_base              그 phase부터
F_l = Σ_stage (weight + state)                 layer l
Sh  = max input bytes + max output bytes       scratch

Σ_l F_l + Sh ≤ A   slots = L, groups = 1, slot_bytes = (F_0, …, F_{L−1})
그 밖              모든 layer의 operand별 (kind, bytes)가 첫 layer와 같아야 한다 (다르면 예외)
                   slots = ⌊(A − Sh) / F_0⌋, slots < 1이면 예외

slot(l)  = l mod slots
group(l) = ⌊l / slots⌋        groups = ⌈L / slots⌉
```

phase 영역은 모든 slot의 weight, 모든 slot의 state, scratch 순서다. 한 group의 weight는 연속 범위이므로
group 업로드는 unit마다 전송 한 번(UPMEM) 또는 region마다 전송 한 번(AiMX)이다.

### 3.3 runtime 동작

```text
compile (prepare_model)
  UPMEM: stage마다 table offset으로 program link
  AiMX : table을 region으로 자름 (chunks)

open / bind (bind_model)
  UPMEM: stage마다 native handle (upmem_model_create)
  AiMX : FPGA마다 region 할당 → slot 첫 layer의 stage마다 job별 context (aimx_model_create)
         같은 slot의 다른 layer stage는 context를 공유. slot_bytes, 종류, offset, job 수가 다르면 예외
         모든 slot의 descriptor chain을 만들고 업로드 (upload=3, chain_build_ms)

run_model(phase) — layer마다
  enter_layer: GroupLoader.enter((phase, group(l)))
    UPMEM pack: stage pool에 weight 배치(PACK, DPU 접근 없음) + SG block 표
          push: 첫 stage program 적재 → upmem_model_group_push (dpu_push_sg_xfer_symbol 1회)
    AiMX  pack: FPGA마다 group weight 범위의 host image에 stub이 직접 조립 (upload=2)
          push: region마다 aimx_model_push
          groups = 1: pack을 push로 미뤄 region 하나씩 pack → push → 해제 (host에 region 하나만)
  stage 실행
  exit_layer: AiMX 없음, UPMEM은 그 layer stage의 host pool 해제

release (측정 경로, §4.1)
  model runtime만 해제. 장치는 남고 다시 prepare할 수 있다

close
  background pack 종료 → context/handle, region 해제 → 장치 반납
```

`GroupLoader`는 한 group이 실행되는 동안 실행 순서의 다음 group을 thread에서 pack한다
(`backend/runtime.py`). pack은 host 메모리만 쓰고, push는 같은 장치의 PIM 연산과 겹칠 수 없다.
마지막 phase의 마지막 group 다음에는 그 phase의 첫 group을 pack한다(decode step 반복).

AiMX에서 cache stage의 KEY/VALUE operand offset은 layer마다 다르므로 slot을 공유하면 bind의 배치 비교에서
예외가 난다. 따라서 AiMX decode는 groups = 1이어야 한다.

### 3.4 key/value cache 규칙

**내용**
host cache가 원본이다. `PIMSession.write_kv(layer, kind, cache, start, stop)`는 위치 범위를 그 layer에서
해당 cache를 읽는 stage의 adapter에 넘기고, adapter는 대기열에 넣었다가 cache stage가 다음에 실행되기
직전(`run`)에 모든 layer의 대기 위치를 보낸다. 처음 보낼 때 host 배치식을 생성 코드의 pack과 한 번 대조한다.

**근거**
- `aimx_model.write_kv`/`_push_kv`/`_check_kv_layout`, `upmem_model.write_kv`/`_push_kv`/`_check_kv_layout`.
- 측정(`measure_stage`)은 cache를 쓰지 않는다. E는 cache 값과 무관하다.

**예시**

```text
공통 host KV  K[h, t, f], V[h, t, f] ── write_kv(t0, t1) ──┬── AiMX : B 패널 배치(pim_model_slot_layout), BF16
                                                           │         region host shadow에 쓰고 항목·FPGA마다
                                                           │         바뀐 byte 구간만 aimx_model_push_ranges
                                                           └── UPMEM: matvec 행렬 tile 배치(_kv_layout), INT8
                                                                     DPU별 host image에 쓰고 [kv_base, C) 전체를
                                                                     같은 offset으로 한 번 전송 (kv_push_region)
```

| target | cache operand 조건 |
|---|---|
| AiMX | GEMM의 B operand, instance마다 FPGA 하나(devices 1), batch round 1 |
| UPMEM | slot이 execution extent `(m, k)`와 같고 `p_n = 1`, slot bytes = rows × cols (INT8) |

UPMEM GEMV의 행렬은 QK에서 `K[h]`(행 = 위치), PV에서 `V[h]ᵀ`(열 = 위치)이다(`_positions_on_rows`).
두 생성 예제는 장치에 위치 `[0, flushed)`를 두고, 그 뒤 위치(최대 `TAIL − 1` = 15개)와 새 token은 decode
layer가 host cache에서 읽어 CPU에서 attention에 합친다. tail이 `TAIL`개가 되면 `write_kv`로 한 번에 보낸다
(`examples/aimx/llama_decode.py`, `examples/upmem/e2e/llama_generation.py`).

### 3.5 성능 식

phase 하나에서 class별 stage 수를 `count`, 선택 후보의 E와 U를 쓰면 한 요청의 layer 하나 시간은 다음과
같다(`model_compile.select_configs`).

```text
reloaded   Σ count · repeats · (E + U)     layer group이 slot을 번갈아 씀, 실행마다 weight 재업로드
           조건: slot_bytes + Sh ≤ A
resident   Σ count · (repeats · E + U)     모든 layer가 자기 slot, weight는 한 번
           조건: L · slot_bytes + Sh ≤ A
```

repeats = 1이면 두 식이 모두 E + U이고, 더 많은 선택이 들어가는 reloaded를 쓴다.
runtime에서 groups > 1이면 phase를 실행할 때마다 모든 group을 다시 올리며, 다음 group의 pack만 계산과
겹치고 push는 겹치지 않는다. `group0_pack_ms`, `group0_upload_ms`는 가릴 앞 실행이 없는 첫 group의 비용이다.

---

## 4. Compile time

```text
compile_model(phases, target, units, kv_inputs, repeats, runtime_options, cache_path, …)
  1. spec = target_spec(target, num_units=units, **runtime_options)
  2. tune_phase (phase마다)     analyze_layer(첫 layer) → class마다 tune_stage_class
                                packed class는 member class를 먼저 tune
  3. kv_bytes_bound             layers × Σ count × max(후보 cache footprint), phase 중 최댓값
  4. capacity                   첫 cache reader 전 phase: C, 그 뒤: C − KV 상한
  5. select_phase (phase마다)   packing 유지/분리 조합마다 select_configs, 가장 짧은 시간
  6. PIMSession.compile         모든 phase의 모든 layer를 고른 config로 컴파일 (장치 할당 없음)
                                phase마다 layer 0이 tune한 class와 같은 stage인지,
                                모든 layer의 stage 구성이 layer 0과 같은지 검사
  7. plan_memory                slot, group, 대응표, KV 항목 (§3)
  8. CompiledModel(session, report)
```

### 4.1 측정

stage class마다 stage 하나만 담은 Torch module(`StageModule`)을 만들고, class 하나의 후보들을
측정 프로세스 하나에서 잰다(`StageMeasurer`). 측정 프로세스는 spawn으로 띄우고 장치를 한 번만 연다.

```text
측정 프로세스 (class마다 1개, 첫 measure 호출에서 시작)
  weight 난수 생성 1회, PIMSession open 1회
  후보마다 (Pipe로 config 수신)
    torch._dynamo.reset → session.prepare({"stage": ...}, kv_inputs)   cache class는 첫 입력이 cache
    stage shape와 instance 수가 class와 같은지 검사
    enter_layer (첫 업로드 = U)
    run_model("stage") × warmup(기본 1) → × repeat(기본 3) → 중앙값 = E
    session.release → 결과 송신
  config = None 수신 → close
```

| 항목 | 값 |
|---|---|
| 측정값 | E, 첫 업로드(`weight_upload`), footprint(weight, state, input, output, cache class는 cache bytes) |
| 실패 | 결과 전에 측정 프로세스가 끝나면 `correctness=False`, `exit_code`로 기록하고 선택에서 제외. 다음 후보는 새 프로세스 |
| 제약 | `stage_constraint`: weight stage와 cache stage는 M/N/K round 1, cache stage는 모든 instance가 한 batch round(`instances × P ≤ units`)이고 `resident_operand_reason`을 만족, 나머지는 round ≤ `max_initial_scheduling_rounds`(기본 4) |
| seed 공간 | `partition_axes`: target이 k를 나눌 수 없으면 k partition은 1. `instance_units = max(1, units / instances)`. weight/cache stage의 초기 round 패턴은 (1, 1, 1)만 |
| budget | class마다 `max_benchmarks`(기본 20), `max_generations`(기본 10) |
| cache | 기본 `workspace/Experiment/Autotuner/autotune_cache.json`. context는 `<template>\|model\|<op>\|w<weight mask>\|x<instances>`, cache class는 `\|kv<mask>[t]`를 덧붙임 |

cache에 있는 후보만 쓰면 측정 프로세스를 띄우지 않는다. 측정 대상과 조회 key는
`target_model(spec).stage_target(problem)`(AiMX GEMM batch)이다.

### 4.2 선택

phase마다 packed class의 유지/분리 조합(`product((True, False), …)`)을 모두 시도한다. 분리하면 그 class를
member class로 바꾸고, member key가 다른 class와 같으면 하나로 합친다. 조합마다 `select_configs`가 class별
후보 하나를 고른다.

- reloaded와 resident 두 계획(§3.5)을 각각 푼다. slot bytes는 stage마다 더하고 scratch는 최대 input + 최대 output이므로, slot bytes·input·output·시간 모두에서 지지 않는 부분 선택만 남기는 multiple-choice knapsack이다.
- 용량 조건을 만족하는 것 중 시간이 가장 짧은 계획을 쓴다. 들어가는 선택이 없으면 그 조합은 제외하고, 모든 조합이 제외되면 예외.
- 분리를 고른 GEMM은 `PIMTorchConfig(pack=False)`로 컴파일한다. 조합별 결과는 `report["phases"][phase]["packing"]`에 남는다.

튜닝의 초기 후보는 세 가지다.

| 후보 | 재료 | 규칙 |
|---|---|---|
| 넓힌 후보 (`widened_configs`) | packing member class의 cache 최선 config | 합친 축의 tile만 packed 크기를 덮도록 키움 (member round 유지). Block m이 다르면 제외 |
| 맞춘 후보 (`fitted_configs`) | 같은 context의 비슷한 shape cache 최선 config 4개 | block, group, partition 유지. 모든 tile을 이 shape를 1 round로 덮도록 재계산 |
| 상주 후보 (`resident_operand_config`) | cache class만 | target이 정한 padding 없는 상주 B config |

---

## 5. 인터페이스

### 5.1 API

```python
from torch_backend import compile_model
from triton.backends.pim.residency import KEY, VALUE

compiled = compile_model(
    {"prefill": (prefill_layers, prefill_example), "decode": (decode_layers, decode_example)},
    target="upmem", units=2048,
    kv_inputs={"decode": decode_kv_inputs},          # kv(index, inputs) -> {KEY: cache, VALUE: cache}
    repeats={"prefill": 1, "decode": new_tokens - 1},
    runtime_options={"nr_tasklets": 16}, cache_path=..., max_benchmarks=20,
)
compiled.report   # target, units, spec, memory, compile_ms, phases
with compiled.session as session:
    session.write_kv(layer, KEY, key_cache, start, stop)
    hidden = session.run_model("prefill", *inputs)
    hidden = session.run_model("decode", *inputs)
    rows = session.memory_table()
```

| `report` 항목 | 내용 |
|---|---|
| `memory` | `kv_bytes`, `kv_base`, `kv_bytes_bound`, `phase_capacity_bytes`, `kv_regions`, `bytes_per_unit`, `capacity_bytes`, `phases`(slots, groups, slot_of_layer, group_of_layer, slot_bytes, scratch_bytes, bytes) |
| `phases[phase]` | `layers`, `repeats`, `estimated_execution_ms`(상주 실행 시간 합), `analysis_ms`, `tuning_ms`, `selection_ms`, `packing`, `stage_classes`(선택 config, E, 첫 업로드, footprint) |

`memory_table()` 행: UPMEM은 table 항목마다 `kind, phase, layer, stage, operand, slot, group, pa, bytes`,
AiMX는 slot 첫 layer의 stage 항목마다 FPGA별 `slot, layers, stage, unit, buffer, kind, offset, pa, bytes, device_pa`.

### 5.2 adapter 계약

| 메서드 | 시점 | 하는 일 |
|---|---|---|
| `begin_model()` | compile | target model runtime 생성 (Session마다 하나) |
| `resources()` | compile | unit 수와 용량 C |
| `requirements(plans)` | compile | stage operand별 종류와 unit당 bytes |
| `prepare_model(plans, memory)` | compile | table offset 배치. UPMEM은 program link, AiMX는 region 분할 |
| `model.compile_invocation(stage, inputs)` | compile | stage의 native artifact 생성 (lowering 경로) |
| `open()` | runtime | 장치 확보, `num_units` |
| `bind_model(plans)` | runtime | stage handle/context 생성. AiMX는 region 할당과 descriptor chain 업로드 |
| `weight_table(plans)` / `memory_table(plans)` | runtime | weight 배치, table 행과 장치 주소 |
| `model.enter_layer` / `run` / `exit_layer` | runtime | group이 바뀔 때만 업로드, 실행 |
| `model.write_kv(stage, cache, start, stop)` | runtime | cache 위치 대기열, 다음 cache stage 실행 전 전송 |
| `end_model()` | runtime | model runtime 해제, 장치 유지 (`PIMSession.release()`) |
| `stats()` / `close()` | runtime | 누적 통계 / 해제 |

### 5.3 모듈

| 모듈 | 역할 |
|---|---|
| `torch_backend/fx_analysis.py`, `fx_normalization.py` | GEMM 찾기, `linear`/`addmm` → `mm` |
| `torch_backend/fx_packing.py` | CSE, 규칙 ①~⑤, ③의 target model 판정, `ConcatenatedWeight` |
| `torch_backend/model.py` | `_freeze_graph`, `build_layer_plan`, `LayerPlan`, `_LayerGemm`, `CompileModelCPU` |
| `torch_backend/model_policy.py` | `StageKey`, `analyze_layer`, `stage_constraint` |
| `torch_backend/model_compile.py` | `StageMeasurer`, `select_configs`, `select_phase`, `compile_model` |
| `torch_backend/execution.py` | `gemm_problem`, `select_layer_config` |
| `torch_backend/lowering.py`, `template.py`, `backend.py` | `mm`/`_int_mm` → PIM GEMM/GEMV template, Inductor 설정 override |
| `backend/residency.py` | `plan_memory`, `MemoryTable`, `token_rows` |
| `backend/runtime.py` | `PIMSession`, `GroupLoader`, `group_order` |
| `backend/upmem_adapter.py`, `aimx_adapter.py` | 장치 open/close, `begin_model`/`end_model`, 통계 |
| `backend/upmem_model.py`, `aimx_model.py` | 배치, bind, group pack/push, cache 쓰기, `memory_table` |

---

## 6. 제한

- 한 phase의 모든 layer는 분석한 첫 layer와 stage 구성이 같아야 한다. 같은 class의 stage는 같은 config를 쓴다.
- slot을 공유하는(groups > 1) layer는 operand별 종류와 bytes가 같아야 한다.
- AiMX는 Session 하나에 phase 하나만 배치한다. cache stage는 slot을 공유할 수 없으므로 AiMX decode는 groups = 1이어야 한다.
- cache stage는 M/N/K round 1, 모든 instance가 batch round 1이어야 한다. AiMX는 instance마다 FPGA 하나를 쓴다.
- weight를 가진 `bmm`은 지원하지 않는다.
- 측정은 stage를 하나씩 실행한다. stage 사이의 host 작업 겹침은 반영하지 않는다.
- group의 push는 PIM 연산과 겹치지 않는다. 겹치는 것은 다음 group의 host pack뿐이다.
- compile 결과는 프로세스 메모리(`CompiledModel.session`)에 있다. 프로세스 사이의 저장은 지원하지 않는다.
