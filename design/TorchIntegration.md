# Torch → PIM integration

Torch 모델을 PIM에서 실행하는 경로의 Torch 쪽 설계를 다룬다. 범위는 두 가지다.

1. **Torch FX graph 분석**: layer 하나의 FX graph에서 GEMM을 찾고, 독립 GEMM들을 더 큰
   GEMM으로 합치는 packing 변환(m 쌓기, n 잇기, block-diagonal)을 적용한다.
2. **Torch runtime memory management**: layer 하나의 메모리 관리 테이블을 용량만큼 복제한
   slot에 layer를 대응시키고, slot의 buffer와 명령(descriptor, program)을 계속 재사용한다.

TritonPIM의 compile과 lowering(Triton → TTIR → TTShared → PIM Common IR → Target IR,
stub·descriptor·program 생성)은 바꾸지 않는다. 바뀌는 것은 `torch_backend/`와 session
runtime의 Python 코드(`backend/runtime.py`, `backend/residency.py`, `backend/*_model.py`)이다.
모든 설계는 target 중립이며, target별 차이는 autotuner의 target model
(`autotuner/model.py`)이 공개하는 값으로만 결정한다.

```text
입력: Torch layers (같은 구조), example inputs, target 이름, unit 수
                                   │
═══════════════════════ COMPILE TIME (장치 할당 없음) ═══════════════════════════
 1. Target spec        등록 profile + unit 수 → PIMTargetSpec, 용량 C
 2. FX 분석            첫 layer torch.export → normalize → packing (§2) → stage 묶음
 3. 튜닝               stage class마다 weight 상주 E 측정 → 가장 빠른 E의 10% 이내에서 footprint 최소
 4. layer 테이블       layer 하나의 영역 크기: weight, KV, activation, state (§3)
 5. slot과 group       모두 들어가면 layer마다 slot, 아니면 slot(l) = l mod N, group(l) = l div N
 6. 컴파일             모든 layer를 같은 config로 컴파일, 테이블 배치
═══════════════════════ RUNTIME ═════════════════════════════════════════════════
 7. open / bind        장치 할당, slot마다 한 번 할당하고 PA를 테이블에 기록
 8. run                layer 진입: group이 바뀌면 그 group의 weight를 함께 올림, 해제 없음
 9. close              해제
```

---

## 1. 용어

| 용어 | 정의 |
|---|---|
| unit | 장치 하나. AiMX는 FPGA, UPMEM은 DPU |
| stage | `LayerPlan` 안의 PIM 실행 단위 하나(`_LayerGemm`). weight GEMM 하나, 또는 같은 shape의 독립 activation GEMM 묶음 하나 |
| instance | stage 안의 독립 GEMM 하나. `bmm`은 batch 차원이 instance |
| stage class | `(m, n, k)`, dtype, op, weight mask, instance 수가 같은 stage의 모음. 튜닝과 config의 단위 |
| packing | 독립 GEMM 여러 개를 GEMM 하나로 합치는 FX 변환 (§2) |
| slot | layer 하나가 unit에 차지하는 영역 묶음(weight, KV, activation, state). 물리 주소가 고정됨 |
| 대응표 | layer → slot (`slot_of_layer`)과 layer → group (`group_of_layer`) |
| group | 함께 상주하고 함께 업로드되는 layer 묶음. layer l은 group ⌊l / N⌋ |
| E | weight가 상주한 상태에서 layer(또는 stage) 한 번의 실행 시간 |
| U | group 하나의 weight를 올리는 시간 |
| C | unit 하나의 용량 = `banks_per_unit × bank_capacity_limit_bytes` |

---

## 2. Torch FX graph 분석

### 2.1 흐름

```text
torch.export(layer)
  → normalize_fx_gemms        linear / addmm → mm
  → pack_gemm_groups          ② n 잇기 → ① m 쌓기 → ③ block-diagonal   (이 절)
  → freeze                    weight와 weight만 읽는 view를 상수로
  → build_layer_plan          같은 depth·shape의 독립 activation GEMM → instance 묶음 stage
  → analyze_layer             stage class와 원래 GEMM 이름 → class 대응
```

모델 코드는 head마다 `mm`을 쓰거나 Hugging Face 방식(`repeat_kv` + `matmul`)을 그대로
쓴다. block-diagonal 같은 배치는 모델 코드가 아니라 compiler가 정한다.

### 2.2 왜 세 가지 변환인가

GEMM 두 개 `C₁ = A₁B₁`, `C₂ = A₂B₂`를 GEMM 하나로 합치려면 결과를 어느 축으로 이어 붙일지
정해야 한다. 축마다 필요한 조건이 다르고, 이 셋이 전부이다.

| 변환 | 공유하는 operand | 합치는 축 | 0 블록 | 계산량 | 모델에서 나오는 곳 |
|---|---|---|---|---|---|
| ① m 쌓기 | B | m | 없음 | 같음 | GQA의 QKᵀ, PV (query 여러 개가 KV 하나를 공유) |
| ② n 잇기 | A | n | 없음 | 같음 | q/k/v projection, gate/up projection (같은 입력) |
| ③ block-diagonal | 없음 | k (대각) | 있음 | p배, target padding이 흡수할 수 있음 | MHA의 head끼리, GQA의 KV group끼리 |

적용 순서는 ② → ① → ③이다. 0이 없는 변환을 먼저 적용해야 ③이 다루는 instance가 줄고
0 블록이 늘지 않는다.

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

Q의 head 차원은 (KV head, g) 순서로 붙어 있으므로 쌓기는 복사 없는 view이다.

```text
q (H, S, D) → view (H/g, g·S, D)            k (H/g, D, S)  (repeat 없음)
bmm → (H/g, g·S, S) → view (H, S, S)
```

**판정.** 같은 depth·shape·dtype의 독립 activation GEMM 묶음(`_instance_groups`) 안에서
B operand가 **같은 FX node**인 GEMM들을 쌓는다(`_stack_shared_rights`). view 계열 op를 따라
올라가 원본을 비교하지 않는다. 따라서 모델이 query head마다 같은 KV node를 쓰면(e2e 예제의
`--kv share`) 쌓이고, Hugging Face `repeat_kv`처럼 head마다 별도의 B node를 만들면(`--kv repeat`)
쌓이지 않는다. `bmm`은 묶음 대상이 아니다.

### 2.4 ② n 잇기 (q/k/v, gate/up)

같은 입력 x를 쓰는 weight GEMM은 weight를 n 방향으로 이어 GEMM 하나로 만들고, 출력을 열
방향으로 나눈다. host에는 이어 붙인 행렬을 만들지 않는다: packed B는 원래 frozen weight들의
`fx_packing.ConcatenatedWeight`이고, 연속 행렬은 weight를 장치에 올리는 동안(또는 reference
계산)에만 `tensor()`로 만든다. 다른 weight도 checkpoint tensor(또는 그 전치 view)를 복사 없이
stage가 그 stride대로 읽는다(`_LayerGemm.prepare`). stage 인자로는 weight를 넘기지 않는다.

```text
                ┌──────────┬──────┬──────┐       ┌──────────┬──────┬──────┐
  x (S × H)  @  │    Wq    │  Wk  │  Wv  │   =   │    q     │  k   │  v   │
                └──────────┴──────┴──────┘       └──────────┴──────┴──────┘
                     (H × (n_q + n_k + n_v))             → 열 방향으로 split

gate/up:  x @ [Wg | Wu] → [g | u] → silu(g) * u
```

입력 A의 전송이 1/3(q/k/v), 1/2(gate/up)로 줄고 stage 수가 준다. AiMX는 B를 4096 폭으로
저장하므로 n이 작은 GQA의 Wk, Wv(n = 1024)의 저장 낭비가 줄어든다.

### 2.5 ③ block-diagonal

공유가 없는 instance는 A를 k 방향으로 나란히 두고 B를 대각에 둔다.

```text
                                ┌──────┬──────┐
  ┌──────┬──────┐               │  B1  │  0   │          ┌──────┬──────┐
  │  A1  │  A2  │       @       ├──────┼──────┤    =     │  C1  │  C2  │
  └──────┴──────┘               │  0   │  B2  │          └──────┴──────┘
     m × p·k                    └──────┴──────┘             m × p·n
                                  p·k × p·n

bmm 형태:  A' = (I/p, m, p·k),  B' = blockdiag (I/p, p·k, p·n),  C' = (I/p, m, p·n) → 열 split
```

0 블록은 실제로 곱해진다. target이 그 자리를 어차피 padding으로 계산할 때만 공짜이다.
A'와 B'는 CPU partition에서 만든다(`pad` + `cat`).

### 2.6 적용 판정: target model

①과 ②는 계산이 늘지 않으므로 공유가 있으면 항상 적용한다(`PIMTorchConfig(pack=False)`인
GEMM은 제외). ③의 묶음 수 p는 autotuner의 공개 API로 계산한 정적 비용으로 정한다
(`fx_packing._cost`). 실행 seed를 고르지 않으며 측정도 하지 않는다.

```text
for p in instance 수의 약수:
    problem_p  = PIMProblem(m, p·n, p·k, dtype)
    config_p   = rank_block_groups 상위 Block Group의 tile_ladder 마지막 점(최대 병렬)
                 + best_partitions 첫 partition
    compute(p) = (I / p) · useful_tiles · scheduled_compute(problem_p, config_p)
    bytes(p)   = (I / p) · useful_tiles · Σ access_allocations(problem_p, config_p)
p* = argmin (bytes(p), −p)     단, compute(p) ≤ compute(1)
```

| target | `scheduled_compute` | ③ 결과 (head_dim 128, S = 1024) |
|---|---|---|
| AiMX | m을 8, n을 512, k를 256 단위로 올림 | QKᵀ p = 2 (k 128 → 256을 0이 채움), PV p = 4 (n 128 → 512) |
| UPMEM | `prod(execution_extent)` (원소 그대로) | p = 1 (0이 실제 계산) |

AiMX에서 ③은 계산이 같고 instance가 줄어 빨라진다. Llama-2-7B layer 하나에서 QKᵀ
invoke는 77.9 → 62.2 ms, PV invoke는 115.3 → 86.7 ms, activation GEMM 전체는 −35%였다.

### 2.6.1 key/value cache를 읽는 GEMM: ④ k-window, ⑤ n-window

**내용**
host key/value cache는 layer마다 `[kv_heads, positions, head_dim]`이다(Hugging Face cache와 같은
배치). B가 KV head 하나 `cache[h]`(또는 그 전치)인 GEMM에는 ①~③을 적용하지 않는다. cache는 장치에
그대로 있어야 하므로 B를 잇거나 대각으로 만들 수 없다. 대신 cache stage는 instance마다 unit 하나
이상을 쓰고 모든 instance가 한 round에 돌아야 한다. 그래서 한 묶음의 GEMM 수가 unit 수보다 많을
때만, 이어진 KV head를 읽는 GEMM끼리 묶어 B를 그 head들을 나란히 둔 합집합 window(`cat(cache[h],
dim=1)`)로 만든다. 묶음 수는 unit 수 이하의 가장 큰 약수다. window는 token-row view
`[positions, kv_heads·head_dim]`의 열 범위로 적는다(`stage.kv_windows`, `residency.token_rows`).
stage는 cache operand 자리에 host cache 자체를 받으므로, 합집합 window는 host에서 만들지 않는다
(장치 영역을 읽는다; reference만 만든다).

**근거**
- `fx_packing._pack_cache_windows`, `cache_window`. 묶은 GEMM의 origin은 None이라 member로
  나눠 재지 않는다(나누면 한 round 조건을 어긴다).
- 같은 KV head를 읽는 query head(GQA)는 CSE 뒤 같은 window를 읽으므로, 같은 묶음 안의 행이 된다.

**예시**

| 규칙 | window가 놓인 B 축 | 묶은 GEMM | 버리는 것 | 나오는 곳 |
|---|---|---|---|---|
| ④ k-window | k (B = windowᵀ) | `[A₁ 0; 0 A₂] @ [W₁ \| W₂]ᵀ` | A의 0 | QK (`q_h @ K_jᵀ`) |
| ⑤ n-window | n (B = window) | `[A₁; A₂] @ [W₁ \| W₂]` → 행마다 자기 열 | 교차 항 | PV (`p_h @ V_j`) |

| target (Llama-2, head 32) | unit | 묶음 | 결과 |
|---|---|---|---|
| UPMEM | DPU 2048 | 32 ≤ 2048 → 묶지 않음 | head마다 GEMV, 0 없음 |
| AiMX | FPGA 2 | 32 → FPGA당 16 head | QK `(16, C, 2048)`, PV `(16, 2048, C)` ×2 |

두 target의 decode layer는 같은 graph다: query head h마다 `q_h @ K[h // g]ᵀ`, `p_h @ V[h // g]`,
cache는 오른쪽 operand(`examples/aimx/llama_decode.py`, `examples/upmem/e2e/llama_workloads.py`).
host cache의 위치는 토큰 위치 그대로이고 `write_kv(layer, kind, cache, start, stop)`로 위치 범위를
알린다. 물리 배치는 adapter가 각자 정한다.

```text
공통 host KV   K[h, t, f], V[h, t, f]  ── write_kv(t0, t1) ──┬── AiMX adapter : B = Kᵀ / V panel, 위치 → (FPGA, 행)
                                                               └── UPMEM adapter: matvec 행렬 K / Vᵀ tile, 위치 → (DPU, local 행)
```

UPMEM의 head별 GEMV는 matvec이라 QK의 행렬이 `(K[h]ᵀ)ᵀ` = K[h] 그대로(행 = 위치)이고, PV의 행렬은
V[h]ᵀ(열 = 위치)다(`upmem_model._positions_on_rows`). unit c는 위치 `[c·S, (c+1)·S)`를 가지므로
장치 출력 순서가 토큰 순서와 같다.

### 2.7 config 대응

packing 후에는 원래 GEMM 여러 개가 packed GEMM 하나가 된다. `build_layer_plan`은 원래 GEMM
이름마다 config를 요구하므로, 한 packing에 속한 원래 GEMM에는 같은 config를 준다.
`analyze_layer`의 대응표(원래 이름 → stage class)도 packed 기준으로 만든다. 모든 layer의
GEMM 이름이 분석한 layer와 같아야 하며, 없는 이름은 `KeyError`로 드러난다.

### 2.8 예: Llama-3-8B attention

query head 32, KV head 8(g = 4), head_dim 128, S = 1024.

```text
x ──② [Wq | Wk | Wv]──▶ q (32 head), k (8), v (8) ── RoPE (CPU)
                │
   ① q (32, S, D) → (8, 4S, D)                k (8, D, S)
   ③ AiMX p = 2  → QKᵀ (4096, 2048, 256) × 4   UPMEM p = 1 → (4096, 1024, 128) × 8
                │
          softmax (CPU, head마다)
                │
   ① P (32, S, S) → (8, 4S, S)                v (8, S, D)
   ③ AiMX p = 4  → PV (4096, 512, 4096) × 2    UPMEM p = 1 → (4096, 128, 1024) × 8
                │
          o_proj ──▶ ② [Wg | Wu] ──▶ down
```

| AiMX, attention GEMM | repeat_kv + head 묶음 | ① + ③ |
|---|---|---|
| QKᵀ instance | 16 × (1024, 2048, 256) | 4 × (4096, 2048, 256) |
| PV instance | 8 × (1024, 512, 4096) | 2 × (4096, 512, 4096) |
| B 전송 (QKᵀ + PV) | 32 + 256 MiB | 8 + 64 MiB |
| 하드웨어 계산 | 25.8 G | 25.8 G |

### 2.9 UPMEM에서의 실행

UPMEM은 DPU를 instance 수만큼 나눠 한 round에 실행한다. instance 하나가 쓰는 DPU 수는
config의 `p_m·p_n·p_k`이고, 동시에 실행되는 instance 수는
`p_b = min(instances, units / (p_m·p_n·p_k))`이다(`execution.py` `select_layer_config`).
DPU 하나는 Execution Tile 하나를 맡고, 그 안에서 tasklet이 행을 나눈다.

```text
Llama-3 QKᵀ (4096, 1024, 128) × 8 (① 뒤),  2026-10-20 선택 config
  B (8,16,16) G (4,4,1) T (4,2,8) P (32,8,1) → Execution Tile (128, 128, 128)
DPU 2048 = 8 instance × 256 DPU  → p_b = 8, 1 round

┌── KV head 0: 256 DPU ──┐┌── KV head 1 ──┐ … ┌── KV head 7 ──┐
│ 32 × 8 출력 조각        ││               │   │               │
│ DPU 하나 = 128×128      ││               │   │               │
└────────────────────────┘└───────────────┘   └───────────────┘
```

세 변환 뒤에도 같은 규칙이 적용된다. 바뀌는 것은 instance 수와 instance당 DPU 수이다.

| UPMEM | head마다 | ① m 쌓기 | ② q/k/v 잇기 | ③ p = 2 |
|---|---|---|---|---|
| instance × instance당 DPU | 32 × 64 | 8 × 256 | 1 × 2048 | 16 × 128 |
| DPU당 계산 | 128×128×128 | 같음 | 3 GEMM → 1 | 2배 (절반이 0) |
| DPU로 가는 A | 기준 | 같음 | 1/3 | 같음 |
| DPU로 가는 B | 기준 | 같음 (DPU마다 사본 필요) | 같음 | 2배 |
| 결과 | – | host의 repeat_kv 복사만 제거 | 이득 | 손해 → p = 1 선택 |

AiMX와 효과가 다른 이유는 instance를 맡는 장치 수와 계산 단위이다. AiMX는 instance 하나를
FPGA 하나가 통째로 계산하므로 ①에서 B를 한 번만 받고, MAC 단위가 넓어 ③의 0이 padding에
흡수된다. UPMEM은 instance를 DPU 수십–수백 개로 나누므로 DPU마다 operand 사본이 필요하고,
계산 단위가 작아 0이 그대로 계산된다.

---

## 3. Torch runtime memory management

### 3.1 메모리 관리 테이블

unit마다 테이블 하나를 둔다. 행은 slot, 열은 영역 종류이고, 칸마다 (PA, 크기)를 기록한다.
명령(AiMX descriptor, UPMEM program)은 테이블의 PA를 담으므로, 테이블이 바뀌지 않으면
명령도 바뀌지 않는다.

```text
slot │ stage │ Weight          │ KV              │ Activation            │ State
─────┼───────┼─────────────────┼─────────────────┼───────────────────────┼─────────────────────
 0   │ qkv   │ W_qkv (PA, size)│                 │ A, Arel, C  (AiMX)     │ descriptor (AiMX)
     │ QKᵀ   │                 │ K  (capacity)   │                       │ descriptor
     │ PV    │                 │ V  (capacity)   │                       │ descriptor
     │ o     │ Wo              │                 │                       │ descriptor
     │ gu    │ W_gu            │                 │                       │ descriptor
     │ d     │ Wd              │                 │                       │ descriptor
 1   │ …     │ slot 0과 같은 배치, 다른 PA                                                 │
 …   │       │                 │                 │                       │
─────┴───────┴─────────────────┴─────────────────┼───────────────────────┼─────────────────────
 공유 영역                                         │ input/output scratch  │
                                                   │ (UPMEM, stage 최대값)  │
```

| 열 | AiMX (Silent-PIM) | UPMEM |
|---|---|---|
| Weight | slot마다 | slot마다 |
| KV | slot마다 (decode) | slot마다 (decode) |
| Activation | slot마다 (stage별 A, Arel, C) | 공유 1개 |
| State | slot마다 descriptor chain | 없음 (program은 stage마다 IRAM 적재, 주소는 link 때 고정) |
| PA를 정하는 쪽 | runtime 할당 결과(`pim_malloc`, `VA2PA`)를 bind 후 기록 | torch가 MRAM offset을 직접 정함 |

target model의 `shared_activation_scratch`가 Activation 열의 공유 여부를 정한다
(AiMX `False`, UPMEM `True`).

AiMX는 할당 하나의 최대 크기가 `1023 × 2 MiB`이고, descriptor가 33-bit 물리 주소를 쓴다.
그래서 slot은 물리 연속인 할당 여러 개로 구성되며, 테이블은 할당마다의 PA를 기록한다.

### 3.2 slot, group과 대응표

layer 하나의 크기에서 slot 수를 정하고, layer를 slot과 group에 대응시킨다
(`backend/residency.py:plan_memory`).

```text
F_l = Σ_stage (weight + state)                     layer l (KV는 아직 CPU, §3.4)
Sh  = max input bytes + max output bytes           공유 scratch (stage 최대값)

Σ_l F_l + Sh ≤ C   모든 layer 상주     N = L, slot_bytes = (F_0, …, F_{L−1})
그 밖             N = ⌊(C − Sh) / F_0⌋  N < 1이면 예외
                   모든 layer의 stage별 크기가 layer 0과 같아야 한다 (다르면 예외)

slot(l)  = l mod N
group(l) = ⌊l / N⌋        groups = ⌈L / N⌉
```

한 group의 layer들은 모든 slot을 채우고 함께 올라간다. 다음 group은 같은 slot의 weight만
덮어쓰고, buffer와 명령(descriptor, program)은 그대로 둔다. groups = 1이면 첫 요청에서 한 번만
올리고, 이후 요청은 업로드가 없다.

### 3.3 runtime 동작

```text
open / bind
  slot마다 한 번 할당
    AiMX : stage마다 context 묶음(aimx_model_create). slot을 쓰는 layer의 같은 위치
           stage는 같은 context를 공유. 처음 실행할 때 buffer 할당과 descriptor 생성
    UPMEM: compile 때 MRAM offset으로 weight 영역을 배치하고 stage program을 그 주소로 link.
           bind에서 stage handle을 만들고, weight를 stage pool에 배치만 한다(PACK, 전송 없음).
           group마다 MRAM 주소 순서의 분산-수집(SG) block 표를 만든다
  bind 후 PA를 테이블에 기록

run_model (요청마다)
  layer l 진입:
    group(l) == 올라와 있는 group  → 바로 실행
    group(l) != 올라와 있는 group  → group(l)의 모든 weight 업로드 → 올라와 있는 group = group(l)
      AiMX : group의 모든 layer, 모든 weight stage를 upload=True로 invoke
      UPMEM: program 적재 후 dpu_push_sg_xfer_symbol 1회 (group당 전송 1회)
  layer l 퇴장: 해제 없음

release (측정 경로, §4.1)
  model runtime만 해제 (stage handle, group, context). 장치는 할당된 채로 남고 다시 prepare할 수 있다

close
  slot 해제, 장치 반납
```

같은 slot을 쓰는 layer는 stage별 크기(weight, state, input, output)가 같아야 한다. 다르면
`plan_memory`가 compile 때 예외를 낸다. AiMX runtime도 같은 slot에 다른 크기를 할당하면 예외를 낸다
(`aimx_resident_alloc`: "resident allocation signature changed").

### 3.4 KV 규칙

device의 slot 내용은 항상 host 원본의 사본으로 둔다.

- weight: 바뀌지 않는다.
- KV: RoPE가 CPU에 있으므로 새 K, V는 host에서 먼저 만들어진다. host의 `kv_table`이 원본이고,
  device에는 token마다 append한다.

그래서 slot을 비울 때 다시 써 보낼 필요가 없다. KV를 device에 두게 되면, group이 바뀌어 다시
올라오는 layer는 weight와 함께 그 layer의 KV 전체를 다시 올린다. 이 비용은 KV 길이에 비례한다(token당 layer당 K, V 각
`hidden × dtype bytes`). KV 영역은 capacity 크기로 잡고, 명령은 capacity 기준으로 만든다.

| attention (decode) | KV 길이가 놓이는 축 | 고정 명령으로 처리하는 법 |
|---|---|---|
| QKᵀ | n (MAC의 바깥 loop) | capacity 크기 chain에서 ⌈len / 512⌉ 열 block만큼 앞부분만 실행 (`pim_exec`의 `desc_idx`) |
| PV | k (MAC의 안쪽 loop) | capacity 전체를 계산하고 len 뒤의 P를 0으로 두거나, KV block마다 chain을 두고 부분합을 host에서 더함 |

KV의 부분 append는 stub의 범위 staging이 필요하다. 이는 lowering 변경이므로 이 문서의
범위 밖이다. 현재 KV는 CPU `kv_table`에 FP32로 두고, attention GEMM의 operand로 activation처럼
매 실행 staging한다.

### 3.5 prefill과 decode

명령과 weight 중 큰 쪽을 고정하고 작은 쪽을 옮기는 것이 최적이다. AiMX descriptor의 크기는
`m · n · k / 128` bytes, weight는 `n · k · 2` bytes이므로 비율은 `m / 256`이다.

| 단계 | m (unit당) | descriptor / layer | weight / layer | 고정할 것 |
|---|---:|---:|---:|---|
| prefill 1024 token, m-split | 512 | 818 MB | 428 MB | 명령 (slot을 고정하고 weight를 옮김) |
| decode 1 token (8행 padding) | 8 | 12.8 MB | 428 MB | weight, KV, 명령 모두 |

prefill에서 명령을 옮기면 layer마다 descriptor 생성 0.27 s와 전송 0.25 s가 든다. weight를
옮기면 0.166 s이다(Llama-2-7B, AiMX FPGA 2개 실측). decode에서는 GEMM을 FPGA끼리 나누지
않고 layer를 FPGA에 나눠 두면(FPGA 하나에 16 layer) weight 6.85 GB, descriptor 0.20 GB,
KV 0.26 GB(1024 token)로 용량 안에 들어간다(추정). n-split은 B의 4096 폭 저장 때문에
q/k/v/o weight를 줄이지 못한다.

### 3.6 성능 식

E는 layer 실행(PIM + CPU), U_g는 group g의 weight 업로드, L은 layer 수, G는 group 수이다.

```text
G = 1   첫 요청   T = L·E + U_0          이후 요청   T = L·E
G > 1   매 요청   T = L·E + Σ_g U_g      (group 업로드 G회)
```

group 업로드와 계산을 겹치는 경로는 없다(§6).

| 측정 (UPMEM 2048 DPU, prefill 1024 token, 2026-10-20) | slots / groups | cold TTFT | warm TTFT | first token |
|---|---|---:|---:|---:|
| Llama-2-7b | 32 / 1 | 80.8 s | 77.9 s | 29949 |
| Llama-3-8B (`--kv share`) | 32 / 1 | 83.3 s | 77.6 s | 1404 |

두 측정은 Block (8,16,16) 이전의 config로 실행했다. G > 1인 회전 경로는 Llama에서 아직
실행하지 않았다.

---

## 4. Compile time

```text
compile_model(layers, example_inputs, target, units, runtime_options, cache_path, …)
  1. spec = target_spec(target, num_units=units, **runtime_options)
  2. analyze_layer(layers[0])         첫 layer만 export, packing, stage class
  3. tune_stage_class                 class마다 tune(), weight 상주 측정 (§4.1), 선택 (§4.2)
                                      packed class는 member class를 먼저 tune하고 packing 유지 여부를 정함
  4. PIMSession.compile               모든 layer를 같은 config로 컴파일 (장치 할당 없음)
                                      layer 0이 tune한 class와 같은 stage로 나뉘었는지,
                                      모든 layer의 stage 구성이 layer 0과 같은지 검사
  5. plan_memory                      slot, group, 대응표 (§3.2)
  6. CompiledModel(session, report)
```

### 4.1 측정

stage class마다 stage 하나만 담은 Torch module(`StageModule`)을 만들고, class 하나의 후보들을
측정 프로세스 하나에서 잰다(`StageMeasurer`). 측정 프로세스는 spawn으로 띄우고, 장치(UPMEM DPU set,
AiMX FPGA)를 한 번만 연다. 후보마다 다음을 반복한다.

```text
측정 프로세스 (class마다 1개, 첫 measure 호출에서 시작)
  weight 난수 생성 1회, PIMSession open 1회
  후보마다 (Pipe로 config 수신)
    torch._dynamo.reset → session.prepare (export, Inductor, lowering, native 빌드, bind)
    enter_layer (첫 업로드, U 기록)
    run_model × warmup(기본 1) → run_model × repeat(기본 3) → 중앙값 = E
    session.release (model runtime 해제, 장치 유지) → 결과 송신
  config = None 수신 → close
```

| 항목 | 값 |
|---|---|
| 측정값 | E(weight 상주 후 실행의 중앙값), 첫 업로드 시간(`weight_upload`), 실측 footprint(`physical_allocations`) |
| 실패 | 결과를 보내기 전에 측정 프로세스가 끝나면(`wait([connection, sentinel])`) 그 후보를 `correctness=False`, `exit_code`로 기록하고 선택에서 제외한다. 다음 후보는 새 측정 프로세스에서 잰다 |
| 제약 | `stage_constraint`: weight stage는 M/N/K round 1, activation stage는 round ≤ `max_initial_scheduling_rounds`(4). p_k는 target의 `mapping_reason`이 정한다 |
| seed 공간 | `partition_axes`: target이 k를 나눌 수 없으면 k partition은 1뿐이다. `instance_units = units / instances`: instance 하나가 채울 unit 수 |
| 초기 round 패턴 | weight stage는 (1, 1, 1)만 |
| budget | class마다 `max_benchmarks`(기본 20), `max_generations`(기본 10) |
| cache | 경로는 `compile_model(cache_path=…)`, 기본은 `workspace/Experiment/Autotuner/autotune_cache.json`. context는 `<template>\|model\|<op>\|w<weight mask>\|x<instances>` |

cache에 있는 후보만 쓰면(cache hit) 측정 프로세스를 띄우지 않는다. 컴파일은 config마다 다르므로
후보마다 남는다. 측정 프로세스 안에는 후보별 host 라이브러리(`ctypes.CDLL`)와 Inductor 결과가 쌓인다.

### 4.2 선택

stage class마다 측정 기록 중 E가 가장 빠른 값의 (1 + `FOOTPRINT_TOLERANCE`, 0.10) 이내인
config들에서 footprint(weight + state + input + output)가 가장 작은 것을 쓴다. E만 보면 Block
Group의 n이나 k가 저장 단위(AiMX: n 4096, k 256)보다 작아 weight가 3–4배로 저장되는 config가
몇 % 빠르다는 이유로 선택된다(Llama-3-8B: layer 3.61 GB). packing 유지 여부도 같은 규칙으로 정한다.
packed 쪽은 `count × E_packed`와 packed footprint, 분리 쪽은 `Σ member count × E_member`와 member
footprint 합을 비교한다. 분리를 고르면 그 GEMM들에 `PIMTorchConfig(pack=False)`를 주고, 결과는
`report["packing"]`에 남는다.

튜닝의 초기 후보는 두 가지다.

| 후보 | 재료 | 규칙 |
|---|---|---|
| 넓힌 후보 (`widened_configs`) | packing member class의 cache 최선 config | 합친 축의 tile만 packed 크기를 덮도록 키움 (member의 round 유지) |
| 맞춘 후보 (`fitted_configs`) | 같은 context(op, weight mask, instance 수)의 비슷한 shape cache 최선 config 4개 | block, group, partition 유지. 모든 tile을 이 shape를 1 round로 덮도록 재계산 |

---

## 5. 인터페이스

### 5.1 API

```python
from torch_backend import compile_model

compiled = compile_model(
    layers, example_inputs, target="aimx", units=2,
    runtime_options={"base_fpga": 0}, cache_path=..., max_benchmarks=20,
)
compiled.report       # stage class별 선택, packing 판단, memory (slots, groups, slot_of_layer, group_of_layer)
with compiled.session as session:
    output = session.run_model(*inputs)
    rows = session.memory_table()   # slot, layers, stage, unit, column, pa, bytes
```

### 5.2 adapter 계약

| 메서드 | 시점 | 하는 일 |
|---|---|---|
| `begin_model()` | compile | target model runtime 생성 (session마다 동시에 하나) |
| `end_model()` | runtime | model runtime 해제, 장치는 유지. `PIMSession.release()`가 부르고 다음 `prepare`를 허용 |
| `compile_invocation(stage, inputs)` | compile | Target IR → host/stub 라이브러리 (lowering 경로, 변경 없음) |
| `resources()` | compile | unit 수와 용량 C |
| `requirements(plans)` | compile | stage별 weight, state, input, output bytes |
| `prepare_model(plans, memory)` | compile | 메모리 계획(slot, 대응표) 기록. UPMEM은 MRAM offset 배치와 program link |
| `open()` | runtime | 장치 확보, 실제 unit 수 확인 |
| `bind_model(plans)` | runtime | slot마다 context/handle 생성, 같은 slot의 stage는 공유. UPMEM은 weight를 pool에 배치하고 group SG 표 생성 |
| `memory_table(plans)` | runtime | slot·stage·unit·열·PA·bytes 행 (AiMX는 할당된 buffer만) |
| `enter_layer` / `run` / `exit_layer` | runtime | group이 바뀔 때만 group 전체 업로드, 실행, 해제 없음 |
| `close()` | runtime | model runtime과 slot 해제, 장치 반납 |

### 5.3 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `torch_backend/fx_packing.py` | CSE, ② n 잇기, ① m 쌓기, ③ block-diagonal과 target model 판정 | 구현됨 (`mm`/`_int_mm` instance, `bmm`은 예정) |
| `torch_backend/model.py` | `build_layer_plan(spec=…)`: freeze 뒤 packing, stage의 `sources` | 구현됨 |
| `torch_backend/model_policy.py` | `analyze_layer`(packing 포함), `stage_constraint` | 구현됨 |
| `torch_backend/model_compile.py` | `StageMeasurer` 측정, class별 선택, packing 판단, `compile_model` | 구현됨 |
| `backend/residency.py` | `plan_memory`: slot 수, group 수, layer → slot/group | 구현됨 |
| `backend/runtime.py` | `PIMSession.memory`, `memory_table()`, `release()` | 구현됨 |
| `backend/upmem_adapter.py`, `aimx_adapter.py` | 장치 open/close, `begin_model`/`end_model` | 구현됨 |
| `backend/aimx_model.py`, `upmem_model.py` | slot별 bind/배치, group이 바뀔 때만 업로드, `memory_table` (AiMX는 context의 PA, UPMEM은 MRAM 주소) | 구현됨 |
| group 업로드와 계산 겹치기 | prefetch | 예정 (DMA·PIM 동시 동작 측정 후) |
| KV 열 | decode의 KV slot | 예정 (부분 append는 lowering 변경 필요) |

---

## 6. 제한

- 모든 layer는 분석한 첫 layer와 stage 구성이 같아야 한다.
- stage class 단위로 config를 고른다. 같은 class의 stage는 같은 config를 쓴다.
- 측정은 stage를 하나씩 실행한다. stage 사이의 host 작업 겹침은 반영하지 않는다.
- group 업로드의 prefetch는 DMA와 PIM 실행이 동시에 진행되는지 측정한 뒤 적용한다.
- UPMEM group 전송(`dpu_push_sg_xfer_symbol`)은 rank 전체를 할당했을 때만 정답이 확인되었다.
  rank 일부(예: 8 DPU)만 할당하면 첫 전송 조각만 MRAM에 쓰인다.
- groups > 1인 회전 경로는 Llama 모델에서 아직 실행하지 않았다.
- KV의 부분 append는 lowering 변경이 필요하므로 이 설계 밖이다.
- compile 결과는 프로세스 메모리에 있다. 프로세스 사이의 저장은 지원하지 않는다.
