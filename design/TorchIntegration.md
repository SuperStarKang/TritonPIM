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
 3. 튜닝               stage class마다 weight 상주 E 측정 → 가장 빠른 config
 4. layer 테이블       layer 하나의 영역 크기: weight, KV, activation, state (§3)
 5. slot 수와 대응표   N = min(L, ⌊(C − 공유 영역) / layer 크기⌋), layer → slot
 6. 컴파일             모든 layer를 같은 config로 컴파일, 테이블 배치
═══════════════════════ RUNTIME ═════════════════════════════════════════════════
 7. open / bind        장치 할당, slot마다 한 번 할당하고 PA를 테이블에 기록
 8. run                layer 진입: slot의 주인이 다르면 weight(와 KV)만 올림, 해제 없음
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
| 대응표 | layer → slot. 고정 slot과 교체 slot으로 이루어짐 |
| E | weight가 상주한 상태에서 layer(또는 stage) 한 번의 실행 시간 |
| U | slot에 다른 layer의 weight를 올리는 시간 |
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

**판정.** B operand에서 view 계열 op(`view`, `reshape`, `expand`, `transpose`, `slice`,
`select`, `unsqueeze`, `contiguous`, `clone`)를 따라 올라가 원본 tensor와 영역을 비교한다.
Hugging Face의 `repeat_kv`는 `unsqueeze → expand(…, n_rep, …) → reshape`이므로, `expand`가
늘린 차원에서 "instance h는 원본 KV ⌊h / g⌋를 쓴다"는 구조가 그래프에 그대로 나타난다.

### 2.4 ② n 잇기 (q/k/v, gate/up)

같은 입력 x를 쓰는 weight GEMM은 weight를 n 방향으로 이어 GEMM 하나로 만들고, 출력을 열
방향으로 나눈다. weight는 상수이므로 이어 붙이기는 freeze 때 한 번이다.

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

①과 ②는 계산이 늘지 않으므로 공유가 있으면 항상 적용한다. ③의 묶음 수 p는 autotuner의
공개 API로 정한다. autotuner는 바꾸지 않는다.

```text
for p in instance 수의 약수:
    problem_p  = PIMProblem(m, p·n, p·k, dtype)
    config_p   = heuristic_seeds(problem_p, spec)의 첫 유효 config
    compute(p) = (I / p) · target_model(spec).scheduled_compute(problem_p, config_p)
    bytes(p)   = (I / p) · Σ target_model(spec).access_allocations(problem_p, config_p)
p* = argmin (bytes(p), I / p)     단, compute(p) ≤ compute(1)
```

| target | `scheduled_compute` | ③ 결과 (head_dim 128, S = 1024) |
|---|---|---|
| AiMX | m을 8, n을 512, k를 256 단위로 올림 | QKᵀ p = 2 (k 128 → 256을 0이 채움), PV p = 4 (n 128 → 512) |
| UPMEM | `prod(execution_extent)` (원소 그대로) | p = 1 (0이 실제 계산) |

AiMX에서 ③은 계산이 같고 instance가 줄어 빨라진다. Llama-2-7B layer 하나에서 QKᵀ
invoke는 77.9 → 62.2 ms, PV invoke는 115.3 → 86.7 ms, activation GEMM 전체는 −35%였다.

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
config의 `p_m·p_n`이고, 동시에 실행되는 instance 수는
`p_b = min(instances, units / (p_m·p_n·p_k))`이다(`execution.py` `select_layer_config`).
DPU 하나는 Execution Tile 하나를 맡고, 그 안에서 tasklet이 행을 나눈다.

```text
Llama-2 QKᵀ (1024, 1024, 128) × 32,  config p = (8, 8, 1), Execution Tile (128, 128, 128)
DPU 2048 = 32 instance × 64 DPU  → p_b = 32, 1 round

┌── head 0: 64 DPU ──┐┌── head 1 ──┐ … ┌── head 31 ──┐
│ 8 × 8 출력 조각     ││            │   │             │
│ DPU 하나 = 128×128  ││            │   │             │
└────────────────────┘└────────────┘   └─────────────┘
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

### 3.2 slot 수와 대응표

layer 하나의 크기에서 slot 수를 정하고, layer를 slot에 대응시킨다.

```text
F  = Σ_stage (weight + KV_capacity + activation_private + state)    layer 하나
Sh = 공유 영역 (UPMEM scratch, AiMX 0)
N  = min(L, ⌊(C − Sh) / F⌋)                                         N = 0이면 예외

N ≥ L   모든 layer 고정                     slot(l) = l
N < L   R개 교체 slot, K = N − R개 고정      slot(l) = l                 (l < K)
                                                     = K + (l − K) mod R  (l ≥ K)
        R = 1 (기본), R = 2 (prefetch를 쓸 때)
```

이전의 두 정책은 양 끝의 특수한 경우로 들어간다.

| 경우 | 대응표 | 이전 이름 |
|---|---|---|
| N ≥ L | 모든 layer 고정 | P-1 |
| N = 1 | 모든 layer가 slot 하나 공유 | P-2 |
| 1 < N < L | 고정 K + 교체 R | (새로 생김) |

순환 배치(`slot(l) = l mod N`)는 쓰지 않는다. 요청은 layer 0 → L−1을 차례로 접근하므로,
N < L인 cache에 순환 배치하면 hit가 0이다. 순환 접근에서 최적 교체(Belady)의 miss 수는
`L − N + 1`이고, 이는 "고정 N−1 + 교체 1"과 같다.

### 3.3 runtime 동작

```text
open / bind
  slot마다 한 번 할당
    AiMX : stage마다 context 묶음(aimx_model_create). slot을 쓰는 layer의 같은 위치
           stage는 같은 context를 공유. 처음 실행할 때 buffer 할당과 descriptor 생성
    UPMEM: MRAM offset으로 weight 영역 배치, stage program을 그 주소로 link
  bind 후 PA를 테이블에 기록

run_model (요청마다)
  layer l 진입:
    slot(l)의 주인 == l  → 바로 실행
    slot(l)의 주인 != l  → weight 업로드 (KV를 쓰면 KV도) → 주인 = l
  layer l 퇴장: 해제 없음

close
  slot 해제
```

같은 slot을 쓰는 layer는 slot 표(slot별 크기, 정렬, 종류)가 같아야 한다. 다르면 bind에서
예외가 난다. AiMX runtime도 같은 slot에 다른 크기를 할당하면 예외를 낸다
(`aimx_resident_alloc`: "resident allocation signature changed").

### 3.4 KV 규칙

device의 slot 내용은 항상 host 원본의 사본으로 둔다.

- weight: 바뀌지 않는다.
- KV: RoPE가 CPU에 있으므로 새 K, V는 host에서 먼저 만들어진다. host의 `kv_table`이 원본이고,
  device에는 token마다 append한다.

그래서 slot을 비울 때 다시 써 보낼 필요가 없다. 교체 slot에 들어오는 layer는 weight와 함께
그 layer의 KV 전체를 다시 올린다. 이 비용은 KV 길이에 비례한다(token당 layer당 K, V 각
`hidden × dtype bytes`). KV 영역은 capacity 크기로 잡고, 명령은 capacity 기준으로 만든다.

| attention (decode) | KV 길이가 놓이는 축 | 고정 명령으로 처리하는 법 |
|---|---|---|
| QKᵀ | n (MAC의 바깥 loop) | capacity 크기 chain에서 ⌈len / 512⌉ 열 block만큼 앞부분만 실행 (`pim_exec`의 `desc_idx`) |
| PV | k (MAC의 안쪽 loop) | capacity 전체를 계산하고 len 뒤의 P를 0으로 두거나, KV block마다 chain을 두고 부분합을 host에서 더함 |

KV의 부분 append는 stub의 범위 staging이 필요하다. 이는 lowering 변경이므로 이 문서의
범위 밖이다. 그동안 KV는 activation처럼 매 실행 staging한다.

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

E는 layer 실행(PIM + CPU), U는 weight 교체, L은 layer 수이다.

```text
요청당 업로드 = L − K
T_request    = L·E + (L − K)·U − (겹쳐 숨긴 업로드)

고정 slot의 이득    Δ = K·U / (L·(E + U))
겹침의 상한         Δ ≤ U / (E + U)          (교체 slot 2개와 prefetch, DMA와 PIM 실행의 동시 동작 필요)
```

| AiMX 2 FPGA, Llama-2-7B prefill (E = 1.351 s, U = 0.166 s, F = 1.64 GB) | 요청당 업로드 | T |
|---|---:|---:|
| N = 1 (이전 P-2, slot 공유) | 32 | 48.5 s |
| N = 5, 고정 4 + 교체 1 | 28 | 47.9 s |
| N = 5, 고정 3 + 교체 2, 완전 겹침 | 29 | 43.4 s |
| 상한 (업로드 0) | 0 | 43.2 s |

| 측정 (Llama-2-7B, AiMX FPGA 2개) | 이전 P-2 (layer마다 해제) | slot 공유 |
|---|---:|---:|
| layer 하나의 업로드 U | 848 ms | 166 ms |
| layer 하나 (요청당) | 2261 ms | 1517 ms |
| 32 layer TTFT | 77.5 s | 52.4 s |

slot을 공유하면 layer가 바뀔 때 alloc, descriptor 생성, descriptor 전송, 해제가 사라지고
weight 전송만 남는다.

---

## 4. Compile time

```text
compile_model(layers, example_inputs, target, units, runtime_options, cache_path, …)
  1. spec = target_spec(target, num_units=units, **runtime_options)
  2. analyze_layer(layers[0])         첫 layer만 export, packing, stage class
  3. tune_stage_class                 class마다 tune(), weight 상주 측정, 가장 빠른 E의 config
  4. layer 테이블 크기, N, 대응표       (§3.2)
  5. PIMSession.compile               모든 layer를 같은 config로 컴파일
                                      모든 layer의 stage 구성이 분석한 layer와 같은지 검사
  6. CompiledModel(session, table, report)
```

### 4.1 측정

stage class마다 stage 하나만 담은 Torch module을 만들고, 후보 config마다 새 프로세스에서
측정한다(`measure_isolated`). 측정값은 E(weight 상주 후 반복 실행), 첫 업로드 시간,
실측 footprint이다. 실패한 후보는 `correctness=False`로 기록하고 선택에서 제외한다.

| 입력 | 값 |
|---|---|
| 제약 | `stage_constraint`: `p_k = 1`, weight stage는 M/N/K round 1, activation stage는 round ≤ `max_initial_scheduling_rounds` |
| 초기 round 패턴 | weight stage는 (1, 1, 1)만 |
| cache context | `<template>\|model\|<op>\|w<weight mask>\|x<instances>` |

### 4.2 선택

stage class마다 측정 기록 중 E가 가장 작은 config 하나를 쓴다. 정책별 비용 비교나 knapsack
DP는 하지 않는다. 측정상 weight stage는 가장 빠른 config가 곧 가장 작은 config였고, layer
하나의 footprint는 descriptor(∝ m·n·k)와 weight가 대부분이라 config로 바꿀 여지가 작다.

---

## 5. 인터페이스

### 5.1 API

```python
from torch_backend import compile_model

compiled = compile_model(
    layers, example_inputs, target="aimx", units=2,
    runtime_options={"base_fpga": 0}, cache_path=..., max_benchmarks=20,
)
compiled.report       # stage class별 선택, memory (slot 수, 고정 slot 수, 대응표)
with compiled.session as session:
    output = session.run_model(*inputs)
    rows = session.memory_table()   # slot, layers, stage, unit, column, pa, bytes
```

### 5.2 adapter 계약

| 메서드 | 시점 | 하는 일 |
|---|---|---|
| `begin_model()` | compile | target model runtime 생성 |
| `compile_invocation(stage, inputs)` | compile | Target IR → host/stub 라이브러리 (lowering 경로, 변경 없음) |
| `resources()` | compile | unit 수와 용량 C |
| `requirements(plans)` | compile | stage별 weight, state, input, output bytes |
| `prepare_model(plans, memory)` | compile | 메모리 계획(slot, 대응표) 기록. UPMEM은 MRAM offset 배치와 program link |
| `open()` | runtime | 장치 확보, 실제 unit 수 확인 |
| `bind_model(plans)` | runtime | slot마다 context/handle 생성, 같은 slot의 stage는 공유 |
| `memory_table(plans)` | runtime | slot·stage·unit·열·PA·bytes 행 (AiMX는 할당된 buffer만) |
| `enter_layer` / `run` / `exit_layer` | runtime | slot 주인이 다를 때만 업로드, 실행, 해제 없음 |
| `close()` | runtime | slot 해제 |

### 5.3 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `torch_backend/fx_packing.py` | CSE, ② n 잇기, ① m 쌓기, ③ block-diagonal과 target model 판정 | 구현됨 (`mm`/`_int_mm` instance, `bmm`은 예정) |
| `torch_backend/model.py` | `build_layer_plan(spec=…)`: freeze 뒤 packing, stage의 `sources` | 구현됨 |
| `torch_backend/model_policy.py` | `analyze_layer`(packing 포함), `stage_constraint` | 구현됨 |
| `torch_backend/model_compile.py` | 측정, class별 가장 빠른 config, `compile_model` | 구현됨 |
| `backend/residency.py` | `plan_memory`: slot 수, 고정 slot 수, layer → slot. `SlotOwners`: slot 주인 | 구현됨 (교체 slot R = 1) |
| `backend/runtime.py` | `PIMSession.memory`, `PIMSession.memory_table()` | 구현됨 |
| `backend/aimx_model.py`, `upmem_model.py` | slot별 bind/배치, 주인이 바뀔 때만 업로드, `memory_table` (AiMX는 context의 PA, UPMEM은 MRAM 주소) | 구현됨 |
| 교체 slot 2개와 prefetch | 업로드와 계산 겹치기 | 예정 (DMA·PIM 동시 동작 측정 후) |
| KV 열 | decode의 KV slot | 예정 (부분 append는 lowering 변경 필요) |

---

## 6. 제한

- 모든 layer는 분석한 첫 layer와 stage 구성이 같아야 한다.
- stage class 단위로 config를 고른다. 같은 class의 stage는 같은 config를 쓴다.
- 측정은 stage를 하나씩 실행한다. stage 사이의 host 작업 겹침은 반영하지 않는다.
- 교체 slot의 prefetch는 DMA와 PIM 실행이 동시에 진행되는지 측정한 뒤 적용한다.
- KV의 부분 append는 lowering 변경이 필요하므로 이 설계 밖이다.
- compile 결과는 프로세스 메모리에 있다. 프로세스 사이의 저장은 지원하지 않는다.
