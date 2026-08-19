# Autotuning 설계 — pooled multi-PIM 실행 계획 탐색

> 중심 질문: **서로 다른 실행 모델을 가진 여러 PIM device를 하나의 compiler/runtime에서
> 공통적으로 표현하고, pooled multi-PIM 환경에서 데이터 이동과 device utilization을 함께
> 고려해 효율적으로 실행할 것인가.**
>
> 이 문서는 그 질문의 탐색(autotuning) 부분을 설계한다. 탐색 대상은 tile 크기 하나가
> 아니라 **BLOCK × block group × execution tile × pool mapping**의 결합 공간이고,
> 평가 기준은 "한 device에서 빠른가"가 아니라 "D개의 PIM device에 나눴을 때
> reuse·전송량·가동률·균형이 함께 좋은가"다 — single-device kernel tuning이 아니라
> **pool-level workload shaping**이다.
>
> 원칙: **탐색 층에 target 이름 분기를 두지 않는다.** 컴파일러가 이미 지키는 규칙
> (`pim_backend/backend/compiler.py:191` — "target 고유 값은 전부 위 두 함수에 있다")을
> Python 층으로 연장한다.
>
> 관련 문서:
> - 3계층(block / block group / execution tile)의 IR 설계 → `PIM_Lowering_Walkthrough.md`
> - AiMX target의 계층 소비 방식 → `AiMXTarget_Design.md`
> - `BLOCK`과 분할을 함께 탐색해야 하는 이유 → `PIMCommon_Design.md` §16.7

---

## 0. 개정 이력과 현재 상태

- **2026-08-07 초판**: 탐색 축이 `BLOCK` + `num_units` + `local_lanes`뿐이었고, 당시
  파이프라인(`pim-map-cores`, `pim-form-execution-waves` — 이후 삭제됨)을 전제했다.
- **2026-08-19 개정(이 판)**: 3계층 파이프라인(`pim-form-block-groups`,
  `pim-plan-group-execution`, `pim-map-devices`, `pim-form-execution-tiles`) 기준으로
  다시 쓰고, multi-PIM(pool) 축과 비용 모델의 utilization 특징을 추가했다.

### 동작함

- `PIMDriver.get_benchmarker()`가 `_pim_do_bench`를 돌려준다
  (`pim_backend/backend/driver.py:297`). launch 전체(staging + h2d + kernel + d2h +
  combine) 벽시계를 warmup 1회 + rep 3회로 잰다. 단계별 시간은
  `pim_timing_ms` 5칸(`driver.py:46,259`)에 남는다.
- 3계층 손잡이가 전부 backend 옵션으로 열려 있다: `group_m/n/k`, `tile_m/n/k`,
  `num_units`, `lane_mode`, `local_lanes`, `device_axis_partitions`
  (`compiler.py:345-402`). aimx 전용으로 `batch`, `bank_capacity_bytes`가 있다.
- `pim_backend/backend/tiling.py:20`의 `execution_tile()` helper가 execution tile
  크기(원소)로부터 launch kwargs(`device_axis_partitions`, `SPLIT_K`, `tile_*`)를
  유도한다. 경계 조건(assert)이 곧 버림 규칙의 사본이다.
- 기계 서술이 이미 한 곳에 있다: `TargetProfile` dataclass(`compiler.py:139`)와
  `_TARGET_PROFILE`(`compiler.py:192`). 탐색 층이 새 서술을 만들 필요가 없다.
- Config kwargs로 커널 인자가 아닌 키를 넘기는 경로가 열려 있다. `jit.py`의
  `**options`를 타고 `PIMOptions`로 흘러가고, `parse_options`가
  `PIMOptions.__dataclass_fields__`에 없는 키를 버리므로 GPU 전용 키는 무해하다.

### 동작하지 않음 (이 문서의 대상)

| # | 문제 | 결과 |
|---|---|---|
| 1 | 용량을 넘는 config가 `triton-shared-opt` subprocess 실패로 올라옴 | `Autotuner._bench`가 `OutOfResources` 계열만 잡으므로 **탐색 전체가 중단**. 지금은 사람이 손으로 후보를 골라 회피 중 |
| 2 | 탐색 공간이 tile 크기뿐 | group·execution tile·pool mapping 축이 열려 있는데 탐색이 쓰지 않는다. §2의 trade-off가 통째로 탐색 밖에 있다 |
| 3 | 추정 없이 전수 측정 | config 하나당 launch 수 회 + target 빌드. 공간이 4축으로 늘면 비현실적 |
| 4 | 측정이 총 시간 1종 | `pim_timing_ms` 5단계가 이미 있으나 autotune이 쓰지 않는다 |
| 5 | 컴파일러 → Python 되돌림 채널 없음 | 실제 group 크기·lane 수·분할 수·활성 device 수를 Python이 다시 계산해야 한다 |

---

## 1. 위치 — 세 계층의 역할과 탐색 손잡이

세 단위의 역할은 겹치지 않는다. 이 구분이 탐색 공간의 축 구성이다.

| 계층 | 역할 | 단위 | 손잡이 → 소비하는 pass |
|---|---|---|---|
| **Triton BLOCK** | software computation granularity — frontend(GPU·CPU와 동일한 표현)를 유지하는 컴파일 단위 | `BLOCK_M/N/K` (커널 constexpr) | 커널 정의 |
| **Block group** | PIM compute granularity — 이웃 tile을 target이 효율적으로 처리하는 단위로 묶는다. local 상주 단위 | `group_m/n/k` (tile 개수) | `--pim-form-block-groups` |
| **Execution tile** | residency / launch granularity — device 하나가 launch 1회에 상주시키는 working set의 상한 | `tile_m/n/k` (원소) | `--pim-form-execution-tiles` |

여기에 pool mapping이 얹힌다.

| pool mapping 손잡이 | 의미 | 소비하는 곳 |
|---|---|---|
| `num_units` (D) | 쓸 PIM Device 수 | profile → `--pim-map-devices` |
| `device_axis_partitions` | 축별 device 분할 수. `tiling.execution_tile()`이 tile 크기에서 유도 | `--pim-map-devices` |
| `lane_mode`, `local_lanes` | PIM Core 내부 lane 실행 형태·능력값 | profile → `--pim-plan-group-execution` |
| `SPLIT_K` (커널 constexpr) | K를 연속 구간으로 가르는 blocked 분할 | 커널 + `--pim-place-data` 판정 |

target 고유 눈금은 profile에만 있다 — aimx의 `batch`는
`execution_unit = [batch, 4096/batch, 256/batch]`로 profile에 실리고
(`compiler.py:185`), group 채움 규칙이 그 눈금을 소비한다. upmem은 눈금이 없어
(`execution_unit` 빈 값) group을 WRAM 예산 자동 산출이 정한다.

**PIM 공통 성질에 대한 서술 규약.** "모든 PIM의 실행 모델이 같다"고 쓰지 않는다.
서로 다른 PIM architecture에서도 **반복적으로 나타나는 memory-centric 실행 특성**
(locality, explicit partitioning, placement, local execution, explicit movement,
capacity constraint, repeated local execution)이 존재한다고 쓴다. 이 특성들이
위 계층 구조와 아래 탐색 공간의 존재 근거다.

---

## 2. 핵심 trade-off — group 크기와 pool 가동률

묶음 크기 `g`를 키우면 reuse가 늘고 전송이 줄지만, 독립적으로 배분할 수 있는
group 수가 줄어 device가 논다. 이것이 pooled multi-PIM autotuning의 존재 이유다.

### 2.1 수식

M 방향 한 축으로 단순화하면:

```
N_B(형상)  = ceil(M / BLOCK_M)                  Triton block 수
N_G(g)     = ceil(N_B / g)                      block group 수
U_D(g)     = min(D, N_G) / D                    device 가동률 (D = device 수)
R(g)       ∝ g                                  group 내 operand reuse
Bytes_B(g) ≈ N_G(g) × |B_tile|  ∝ 1/g           B의 총 전송량 (m 방향 묶음일 때)
```

m 방향 이웃 tile은 B를, n 방향 이웃 tile은 A를 공유한다
(`Passes.td`의 `pim-form-block-groups` 설명). 그래서:

```
g ↑
 ├── reuse ↑, 유효 일감당 전송 ↓
 └── N_G ↓ ──── U_D ↓  (N_G < D가 되는 순간부터)
```

### 2.2 대입 예시 — M = 128, BLOCK_M = 32, D = 4

`N_B = 4`.

| group_m | N_G | 배분 | U_D | reuse |
|---|---|---|---|---|
| 1 | 4 | PIM0~3에 1개씩 | 4/4 = 100% | 최소 |
| 2 | 2 | PIM0·1만, PIM2·3 유휴 | 2/4 = 50% | 2배 |
| 4 | 1 | PIM0만, 나머지 유휴 | 1/4 = 25% | 최대 |

단일 device 관점의 최적(g 최대)과 pool 관점의 최적이 다르다. **최적 tile은 한 PIM에서
가장 빠른 tile이 아니라, operand reuse와 여러 PIM의 동시 가동률을 함께 최대화하는
tile이다.** sweet spot의 위치는 형상·device 수·전송 계수에 따라 움직이므로 추정으로
순위를 정하고 측정으로 확정한다(§3.2의 3층).

### 2.3 utilization은 목표 함수가 아니라 cost feature다

`U_D = 100%`가 반드시 빠른 것은 아니다 — 너무 잘게 나누면 가동률이 높아도 launch·
전송이 늘고 reuse가 준다. 그래서 U_D를 최적화 목표로 두지 않고, **multi-PIM 성능을
예측하는 비용 모델의 특징(feature)으로 쓴다**:

```
T = T_compute + T_transfer + T_launch + T_padding + T_imbalance
```

이 분해는 개념 구조이고, 실제 적합 형태는 §8이 정한다 — `T_padding`과
`T_imbalance`는 독립 항이 아니라 compute·transfer 항의 특징 계산 방식(가장 많이 맡은
device 기준, 하드웨어가 도는 크기 기준)에 흡수된다.

---

## 3. 원칙

### 3.1 target 이름 분기 금지

탐색 층은 `_TARGET_PROFILE[target](options)`가 돌려주는 `TargetProfile`
(`compiler.py:139-150`)의 **값**만 읽는다. `if target == "upmem"`을 쓰지 않는다.
target 추가 비용을 profile 함수 하나로 유지한다.

### 3.2 3층 구조 — 섞지 않는다

| 층 | 판단 | 근거 | 틀렸을 때 |
|---|---|---|---|
| **버림** | 실행 가능 여부 | 부등식 (컴파일 전) | 놓쳐도 아래 층이 받는다 |
| **순위** | 추정 | 비용 모델 (컴파일 전) | 최적을 놓친다 |
| **측정** | 사실 | 실행 | — |

전송량만으로 순위를 매기면 항상 유닛 하나짜리 설정이 1등이 되는데, 그것은 §2의
trade-off에서 U_D를 무시한 것이다. 반대로 U_D가 낮다고 버리면 전송이 줄어 더 빠른
config를 놓친다. **그래서 추정치는 전부 순위에만 쓰고 버리지 않는다.**

### 3.3 실패 방향이 안전한 쪽

인식하지 못한 실패는 "지금과 같은 동작"(탐색 중단)으로 떨어진다. 조용히 틀린 config를
고르는 경로를 만들지 않는다.

---

## 4. 기계 서술 (0단계)

초판은 탐색용 dict 신설을 제안했으나, 병합 후에는 `TargetProfile` dataclass가 이미
공통 서술이다. 탐색 층은 그것을 그대로 읽는다.

```python
# pim_backend/backend/tuning.py (신규)
from triton.backends.pim.compiler import _TARGET_PROFILE

def machine_profile(options):
    return _TARGET_PROFILE[options.pim_target](options)   # TargetProfile
```

**이 아래 모든 코드는 이 값만 읽는다.** 쓰는 필드:

| 필드 | 의미 | upmem | aimx |
|---|---|---|---|
| `devices` | PIM Device 수 D_phys | `num_units` or 2048 | `num_units` or `/dev/pl_dma*` 개수 |
| `lanes` | Core 하나의 lane 능력값 | `local_lanes` (기본 16) | 4096 (고정) |
| `parallel_execution_kind` | lane의 성격 | multithread | simd |
| `lane_mode` | lane 실행 형태 | whole_tile / output_row | (없음) |
| `execution_unit` | 하드웨어 실행 눈금 [m,n,k] | (없음) | [batch, 4096/b, 256/b] |
| `bank_bytes` | 큰 메모리 예산 | MRAM 몫 (기본 60 MiB) | PIM DRAM (선언 1 GiB) |
| `local_bytes` | 빠른 메모리 예산 | WRAM 몫 (기본 48 KiB) | 0 (저장소 없음) |

`bank_bytes < 0`(미측정)과 `local_bytes == 0`(없음)의 구분은 `compiler.py:136-138`의
규약을 따른다. 이 두 값이 aimx를 예외 처리 없이 흡수하는 열쇠다.

---

## 5. 실패를 복구 가능하게 만든다 (1단계)

### 5.1 PIM의 공통 성질

PIM에서는 tile·group·execution tile 크기와 유닛 수가 전부 컴파일 시점 상수이고
기계는 profile로 서술된다. 그래서 **config가 바꿀 수 있는 실패는 언제나 두 상수의
비교로 나온다.** 층이 달라도 문구의 형태가 같다.

| 진단 | 위치 | 층 |
|---|---|---|
| `tile buffers need <N> byte of local storage but a PIM Core has <M>` | `PIMVerifyPass.cpp:74` | PIM Common |
| `the partition counts need <N> PIM Devices but the machine has <M>` | `PIMVerifyPass.cpp:90` | PIM Common |
| `needs <N> elements but only <M> ...` (배정량 > capacity) | `PIMVerifyPass.cpp:150` | PIM Common |
| `the execution tile needs <N> ...` | `PIMToUPMEMPass.cpp:436` | target |
| `this buffer needs <N> byte of MRAM but ...` | `PIMToUPMEMPass.cpp:809` | target |
| `(m_t=…, k_max=…, n_max=…) needs <N> ...` (bank 예산 초과) | `PIMToAiMXPass.cpp:154` | target |

반면 config를 바꿔도 사라지지 않는 진단은 이 형태가 아니다.

- `a multithread profile carries no lane_mode` (`PIMPlanGroupExecutionPass.cpp:56`) — 옵션 배관 오류
- 다중 PIM Core topology의 group 분배 미구현 진단 (`Passes.td`의 `pim-plan-group-execution`)
- `origin rank does not match sizes rank` 등 IR 불변식

**즉 메시지 형태가 곧 분류다.** 진단 목록을 표로 유지할 필요가 없다.

### 5.2 설계

`_run_opt`(`compiler.py:45`)가 **모든 MLIR pass의 단일 통로**다 — PIM Common,
UPMEM 변환, AiMX 변환이 전부 지나간다. 여기 한 곳만 고친다.

```python
from triton.runtime.errors import OutOfResources

# PIM은 용량이 전부 컴파일 시점 상수라, config 때문에 나는 실패는 항상 "필요 N, 보유 M"
# 형태로 나온다. 그 형태만 autotune이 아는 예외로 바꾸고 나머지(미구현·IR 불변식)는
# 그대로 올려 hard fail로 둔다.
_NEED = re.compile(r"needs? (\d+) (.+?) but .+? (?:has|is|only) (\d+)")

def _run_opt(mlir_text: str, passes: list) -> str:
    opt = cpu_compiler._get_triton_shared_opt_path()
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.mlir"), os.path.join(tmp, "out.mlir")
        Path(src).write_text(mlir_text)
        r = subprocess.run([opt, src, *passes, "-o", dst], capture_output=True, text=True)
        if r.returncode:
            m = _NEED.search(r.stderr)
            if m:
                raise OutOfResources(int(m.group(1)), int(m.group(3)), m.group(2))
            raise RuntimeError(r.stderr)
        if r.stderr:
            sys.stderr.write(r.stderr)   # 경고·remark는 지금처럼 그대로 보인다
        return Path(dst).read_text()
```

`OutOfResources(required, limit, name)`의 세 인자가 정규식의 세 그룹에 1:1로 대응한다.
`Autotuner._bench`가 이 예외를 잡아 해당 config를 `inf`로 두고 탐색을 계속한다.

### 5.3 규약으로 승격

문구에 기대는 취약함은 규약으로 없앤다.

> **PIM 계열 pass는 용량 부족을 `... need(s) <N> <자원> but <보유자> has/is/only <M>`
> 형태로 낸다.** autotuning이 이 형태를 config 단위 실패로 해석한다.

새 target의 변환 pass가 이 규약만 지키면 탐색 층은 한 줄도 고치지 않는다.

---

## 6. Early pruning (2단계)

1단계와 **중복이 아니라 계층**이다. early pruning은 값싸고 확실한 부등식으로 컴파일을
아끼고, 1단계는 나머지를 안전하게 흡수한다. **그래서 early pruning은 정확할 필요가
없다.** 규칙은 전부 현재 pass·helper의 사본이다.

### 6.1 기호

```
기계 (TargetProfile)
  D_phys  물리 device 수
  S       local 예산 (bytes)      upmem 49152,  aimx 0 (없음)
  Bnk     bank 예산 (bytes)       upmem 60 MiB, aimx 1 GiB (선언값)
  kind    multithread | simd
  mode    lane_mode (multithread 전용)
  unit    execution_unit [um,un,uk] (없으면 눈금 없는 기계)

config
  bm, bn, bk     BLOCK_M / BLOCK_N / BLOCK_K
  gm, gn, gk     group_m / group_n / group_k    (0 = 자동)
  tm, tn, tk     tile_m / tile_n / tile_k (원소, 0 = 자동)
  D              num_units
  Lreq           local_lanes
  SK             SPLIT_K

형상 (nargs)   M, N, K

커널 (target 상수가 들어가지 않는 유일한 커널 지식)
  W_group(gm·bm, gn·bn, gk·bk) = 한 벌 기준 local 상주 바이트
    (예: i8 GEMM  = gm·bm·gk·bk + gk·bk·gn·bn + gm·bm·gn·bn·4)
  W_exec(tm, tn, tk) = execution tile의 bank 상주 바이트
    (예: GEMM     = A(tm×tk) + B(tk×tn) + C(tm×tn) 원소×바이트, tiling.py:14-16)
```

### 6.2 규칙 — 전부 pass·helper의 사본

**R1. tile·group 수** (`pim-build-logical-grid`, `pim-form-block-groups`)

```
Tm = ceil(M / bm)   Tn = ceil(N / bn)   Tk = ceil(K / bk)
Gm = ceil(Tm / gm)  Gn = ceil(Tn / gn)                    group 좌표 수
```

**R2. group 자동 산출** (`pim-form-block-groups`, `Passes.td` 출처 규칙)

축의 옵션이 0일 때:

- 눈금 없는 기계 + 협력 모드(simd 또는 `mode = output_row`): local 예산 `S`에 들어갈
  때까지 m → n → k 순서로 번갈아 1씩 키운다.
- 눈금 있는 기계: 실행 눈금 채움이 담당한다 —
  ```
  tiles  = max(1, floor(unit / block))      block 정수 개를 눈금에 채운다
  useful = tiles · block                    실제 데이터
  phys   = ceil(useful / unit) · unit       하드웨어가 도는 크기
  ```
  `phys − useful`이 padding이고(pass가 remark로 알린다), 배수면 padding이 없다.
- 조건 밖(whole_tile, 예산 미측정)은 항등(1)이다.

**R3. lane 해석** (`PIMPlanGroupExecutionPass.cpp:65-94`)

```
        | Lreq                                        kind = simd
L    =  | max(1, min(Lreq, floor(S / W_group)))       mode = whole_tile (lane마다 복제)
        | max { l <= Lreq : (group 출력 행 수) mod l == 0 }   mode = output_row (한 벌 협력)
```

요청값이 아니라 이 `L`이 실제 값이며, upmem에서 `-DNR_TASKLETS`로 들어가는 수다
(`compiler.py:279-283`의 재파싱 근거).

**R4. device 분할** (`pim-map-devices`, `tiling.py:35-45`)

`device_axis_partitions`를 주면 그대로(곱이 D를 넘으면 진단), 주지 않으면 정사각에
가까운 자동 인수분해. tile 크기로 생각할 때는 helper가 유도한다:

```
Pm = ceil(M / tm)    Pn = ceil(N / tn)    Pk = K / tk       (tiling.execution_tile)
```

balanced block이므로 partition `i`는 `floor(G/P) + [i < G mod P]`개를 맡는다. 시간은
가장 많이 맡은 device가 정하므로 특징은 올림 기준으로 센다.

**R5. 활성 device와 가동률** (§2의 U_D를 2축으로 일반화)

```
gm·gn 단위 배분이므로   A = min(Gm, Pm) · min(Gn, Pn) · min(Gk_parts, Pk)
u_dev = A / D
imb   = ceil(Gm/Pm)·ceil(Gn/Pn) / (Gm/Pm · Gn/Pn)          부하 불균형 (>= 1)
```

**R6. execution tile과 round** (`pim-form-execution-tiles`, target lowering)

배정량이 상한 안이면 창(round)은 1개로 퇴화한다. 상한이 배정량보다 작으면:

```
R = Π_축 ceil(배정량_축 / 상한_축)          시간 창 수
```

upmem은 출력 상한을 시간 창으로 소비하고, k 창은 부분합을 host가 누산한다
(unstage accumulate). aimx는 상한이 담는 block group 수가 buffer 크기다. 상한 자동
산출은 m = group 1개, n = target 기본 폭, k = 남는 bank 예산 전부(이분 탐색)다.

**R7. 복제와 전송량** (`pim-place-data`의 `replicated_over`)

축 하나를 device로 나누면 그 축에 불변인 피연산자가 그 분할 수만큼 복제된다 —
matmul은 `output_row_tile`을 나누면 B가, `output_column_tile`을 나누면 A가 복제된다.

```
bytes_in(총) = |A|·Pn_eff + |B|·Pm_eff + ...               § 2.1의 Bytes_B(g)의 일반형
```

### 6.3 버림 — 부등식

```
[1] local 용량     S > 0 이고 W_group > S  (whole_tile은 L=1 기준)      → 버린다
[2] device 초과    Pm·Pn·Pk > D_phys                                    → 버린다
[3] 경계 정합      tm mod (gm·bm) != 0  (tn, tk 동일)                   → 버린다
                   tk 있음: K mod tk != 0, SK mod (K/tk) != 0,
                   (K/SK) mod bk != 0                                   → 버린다
[4] bank 용량      Bnk > 0 이고 W_exec > Bnk                            → 버린다
[5] 중복           해석된 키 (bm,bn,bk, gm,gn,gk 해석값, tm,tn,tk,
                   D, L, SK) 가 이미 나옴                                → 버린다
```

- **[1]** R3이 whole_tile의 `L`을 낮춰 맞추므로, 실패하는 유일한 경로는 `L = 1`인데도
  `W_group > S`인 경우다. output_row·simd는 한 벌 그대로 비교한다.
- **[3]** 은 `tiling.execution_tile()`의 assert 사본이다 — helper를 통과하지 못하는
  config는 컴파일에 도달하지 못하므로 미리 거른다.
- **[4]** 가 §2의 capacity 조건이다: 요구 조건이 "전체 문제 크기 ≤ PIM 메모리"가
  아니라 **"현재 execution tile의 working set ≤ PIM 메모리"** 다. 전체 GEMM은 PIM
  메모리보다 커도 된다 — 창 반복(R6)이 받는다.
- **[5]** 실질적인 절약. `Lreq ∈ {1,4,8}`이어도 R3의 해석값이 같으면 컴파일 결과가
  같은 config를 반복 측정하게 된다. group 자동(0)도 R2로 해석한 값을 키에 쓴다.

**MRAM 버퍼별 몫·descriptor 예산은 일부러 넣지 않는다.** target lowering이 버퍼를
어떻게 나누는지에 의존하므로 1단계가 잡게 둔다 — 계층 구조의 이득이다.

### 6.4 순위로 넘기는 것

`u_dev`, `imb`, `R`, 복제 바이트, padding 비율은 **추정**이므로 순위(§8)에만 쓴다.
어느 하나로 잘라내면 §2의 trade-off 한쪽을 잃는다 — U_D가 낮은 config가 전송 절감으로
더 빠를 수 있고, 그 반대도 있다.

---

## 7. 탐색 공간 생성 (3단계)

후보를 손으로 쓰지 않고 profile과 helper에서 만든다. 탐색 공간은

```
Software Tile (BLOCK) × PIM Granularity (group) × Residency (execution tile) × Pool Mapping (D, 분할)
```

이고, 축의 존재 여부 자체가 profile 값으로 결정된다.

```python
def candidates(blocks, options, shapes=None):
    p = machine_profile(options)
    devices = _halvings(p.devices)                       # D_phys, /2, /4, ...
    # group 축: 눈금 있는 기계는 채움 규칙이 정하므로 자동(0) 하나. 눈금 없는 기계는
    # 자동(0 = 예산 채움) + 명시값 몇 개 — 자동값은 reuse만 보고 u_dev를 보지 않으므로
    # 자동값 이하의 명시값들이 pool 관점의 후보가 된다 (§2).
    groups = [(0, 0, 0)] if p.execution_unit else [(0, 0, 0), (1, 1, 1), (2, 2, 1), (4, 2, 1)]
    # lane 축: simd 또는 output_row(약수 정합)는 요청값 축이 사실상 없다.
    lanes = [p.lanes] if (p.parallel_execution_kind == "simd"
                          or p.lane_mode == "output_row") else [1, 4, 8, 16]
    # execution tile 축: 자동(0) + k 상한 후보 — k는 bank 예산 이분 탐색의 자동값이
    # 최대이므로, 그 이하 후보는 round 수와 상주량의 교환을 재는 자리다.
    tiles = [(0, 0, 0)] + (shapes and _k_caps(shapes) or [])
    return [triton.Config({...BLOCK..., "group_m": g[0], "group_n": g[1], "group_k": g[2],
                           "tile_m": t[0], "tile_n": t[1], "tile_k": t[2],
                           "num_units": d, "local_lanes": l})
            for b in blocks for g in groups for t in tiles for d in devices for l in lanes]
```

- **tile 크기 → 분할 수 변환은 `tiling.execution_tile()`을 그대로 쓴다.** 문제 크기를
  아는 launch 지점에서 kwargs를 유도하는 helper가 이미 있으므로 탐색 층이 재구현하지
  않는다.
- target 무관성의 구체적 형태가 여기다: aimx는 `execution_unit` + `simd` +
  `local_bytes = 0`으로 group·lane 축이 자동으로 소거되고 후보 수가 크게 준다. 새
  target도 profile 값에 따라 축이 켜지고 꺼질 뿐, 생성 코드는 같다.
- `num_units`는 "물리 유닛 개수"와 "이 커널이 쓸 유닛 수"를 겸한다(§12의 위험).

---

## 8. 측정 기록 기반 비용 모델 (4단계)

**구조(target 무관)와 계수(target별 측정)를 분리한다.** 계수를 코드에 박아 넣으면
target 의존을 Python으로 옮겨 심는 셈이 된다.

### 8.1 특징 — 컴파일 없이, target 상수 없이

§6.2의 R1~R7이 그대로 특징이다. 전부 pass 규칙의 사본이므로 target에 묶이지 않는다.
§2.3의 개념 분해 `T = T_compute + T_transfer + T_launch + T_padding + T_imbalance`는
특징 계산 방식으로 실현된다:

| 개념 항 | 특징으로의 실현 |
|---|---|
| `T_compute` | device당 MAC 수 — **가장 많이 맡은 device 기준**(olim). 눈금 기계는 `phys` 기준으로 세면 padding 낭비가 자동 반영된다 |
| `T_transfer` | device당 in/out 바이트 — **복제(R7) 포함**. group이 크면 1/g로 주는 값 |
| `T_launch` | round 수 R × launch 1회 상수 + 상수항 |
| `T_padding` | 독립 항이 아님 — compute·transfer를 phys 기준으로 센 것에 흡수 |
| `T_imbalance` | 독립 항이 아님 — 올림 기준으로 센 것에 흡수. `u_dev`·`imb`는 진단·기록용으로 함께 저장 |

### 8.2 기록 — target별 JSON 한 파일

```
pim_backend/backend/profiles/<target>.json

{"target": "upmem",
 "samples": [
   {"shape": [1024,1024,1024],
    "config": {"BLOCK_M":32, ..., "group_m":2, "tile_k":4096, "num_units":2048, ...},
    "features": {"macs_per_dev": ..., "bytes_in_per_dev": ..., "rounds": 1,
                 "lanes": 8, "u_dev": 1.0, "imb": 1.0, "padding": 0.0},
    "ms": 159.44,
    "stages": {"stage":12.00, "h2d":25.12, "kernel":106.49, "d2h":5.65, "combine":10.19}},
   ...]}
```

채우는 방법은 둘이고, 같은 파일을 채우므로 따로 만들지 않는다.

- **미리** — `workspace/`의 sweep 결과를 seed로 넣는다. 수행된 프로파일링이 그대로
  첫 데이터가 된다.
- **쓰면서** — 평소 autotune 실행이 끝날 때 `configs_timings`를 특징과 함께 추가한다.

**`stages`를 함께 저장하는 것이 중요하다.** 총 시간 하나로 회귀하면 전송 계수와 계산
계수가 서로를 흡수한다. 단계별로 저장하면 `h2d`로 전송 계수만, `kernel`로 계산 계수만
따로 적합할 수 있어 표본이 훨씬 적게 든다. `pim_timing_ms`는 이미 있고, 이것을 내지
않는 target에서는 0으로 채워지므로 "있으면 저장"으로 두면 target 무관하다.

### 8.3 적합과 예측

계수에 대해 선형인 모델로 두어 최소자승으로 푼다.

```
kernel_ms  ≈ a0 + a1 · macs_per_dev / L
h2d_ms     ≈ b0 + b1 · bytes_in_per_dev + b2 · R        (round마다 stage 반복)
d2h_ms     ≈ c0 + c1 · bytes_out_per_dev
combine_ms ≈ d0 + d1 · combine_ops

perf_model = kernel_ms + h2d_ms + d2h_ms + combine_ms + stage_ms
```

- `torch.linalg.lstsq`로 푼다 (torch는 이미 의존성이다).
- **cold start가 곧 fallback이다.** 표본이 임계치 미만이면 `perf_model`을 달지 않는다
  = 전수 측정 = 현재 동작. 새 target은 자동으로 여기서 시작한다.
- `top_k`는 신뢰도에 따라 움직인다. leave-one-out으로 예측 순위와 실측 순위의 상관을
  재서 낮으면 `top_k`를 키운다 — "추정은 순위만 정하고 최종 판단은 측정"(§3.2)의 구현.

---

## 9. Persistent execution과의 관계

Persistent execution은 별도 기능이 아니라 execution tile 추상화의 결과다:

```
Execution tile (동일 shape)
    → 동일한 device-local 물리 buffer, 동일한 실행 순서
    → 동일 kernel / descriptor / 물리 buffer 재사용     (address-stable execution footprint)
    → problem-sized 상주를 tile-sized 상주로 대체       (streaming — 큰 workload 지원)
```

autotuning 관점에서 두 가지가 따라온다.

1. **측정 대상은 정상 상태(steady state)다.** 초회 비용(device 할당, kernel 빌드,
   ELF/descriptor 적재)은 재사용으로 상각되므로, 반복 실행의 비용은
   `T_repeat ≈ T_stage + T_transfer + T_launch + T_compute`다. `_pim_do_bench`의
   warmup 1회가 이미 이 구분이다 — 같은 `num_units`면 upmem runtime의 set 캐시가
   재할당을 생략하므로 warmup이 초회 비용을 흡수한다. 비용 모델(§8)이 적합하는 것도
   정상 상태다.
2. **round 수 R이 특징에 있는 이유.** 창을 반복하면 실행 계획은 재사용되지만
   stage·transfer·launch는 R배가 된다. execution tile을 줄여 상주량을 아끼는 것과
   R이 늘어나는 것의 교환을 §8이 계수로 배운다. 실행 계획 생성·전송량이 문제 크기가
   아니라 tile 크기에 비례하는 것 — 이것이 persistent 실행의 이득이고, R6·R7이 그
   구조를 특징으로 옮긴다.

---

## 10. Common / target-specific 분리

탐색·비용 모델은 공통 층에서 동작하고, target 층은 제약과 계수로만 나타난다.

| | 공통 층 (모든 target) | target 층 |
|---|---|---|
| **표현** | partition, placement, block grouping, execution tiling, residency, multi-device 분배, persistent 실행 | — |
| **탐색축** | BLOCK, group, execution tile, D, 분할 | 없음 (축의 존재 여부만 profile 값이 정함) |
| **제약** | §6.3 부등식 | upmem: DPU 수·tasklet·WRAM·8-byte 정렬·DMA 단위 / aimx: batch·실행 눈금·padding·PIM DRAM 예산·descriptor |
| **계수** | 없음 | `<target>.json`의 측정 기록 |

### target 추가 비용

| 층 | target 상수를 쓰는가 | 근거 |
|---|---|---|
| `machine_profile()` | **여기만** | `_TARGET_PROFILE`의 profile 함수 하나 |
| §5 예외 번역 | 아니오 | 진단 형태가 규약 (§5.3) |
| §6 R1·R4·R5 | 아니오 | 형상·config 산술 |
| §6 R2·R3·R6·R7 | `S`, `Bnk`, `kind`, `mode`, `unit` **값만** | pass 규칙의 사본 |
| §7 후보 생성 | 같은 값들 **만** | 축의 소거가 값으로 결정 |
| §8 계수 | **측정으로 흡수** | target별 JSON. 새 target은 빈 파일 → 전수 측정으로 시작 |

**새 target을 추가하는 비용은 profile 함수 하나와 `<target>.json` 빈 파일이다.**

---

## 11. 평가 계획 — research question 중심

실험을 기능별이 아니라 질문별로 구성한다. 검증은 전부 사용자가 직접 실행한다.

- **RQ1. 공통 abstraction이 서로 다른 PIM target을 실제로 커버하는가.**
  동일 Triton 커널 → 동일 PIM Common IR → upmem/aimx 두 lowering.
- **RQ2. Multi-PIM-aware grouping이 왜 필요한가.** 1/2/4/8 device에서 group 크기
  sweep. 그래프 3종을 같은 x축(group 크기)으로 겹친다:
  1. latency/throughput (device 수별 곡선 — sweet spot의 이동을 보인다)
  2. active device 비율 (§2.2 표의 실측판)
  3. operand 전송량 (`Bytes_B(g) ∝ 1/g`의 실측판)
  두 곡선의 교차가 §2의 trade-off이고, §6.2 예측식(R5·R7)의 검증 자리다.
- **RQ3. 공통 최적화 + target-specific 최적화의 누적 효과.**
  baseline → +common mapping → +multi-PIM planning → +target-specific 제약 →
  +persistent 실행의 누적 막대.
- **RQ4. Persistent 실행이 반복 비용을 얼마나 줄이는가.** upmem: compile/load/stage/
  h2d/launch 단계별. aimx: descriptor 생성량·전송량·`pim_exec` 횟수.
- **RQ5. PIM 메모리보다 큰 workload를 실행할 수 있는가.** 문제 크기 / PIM 용량 =
  0.5×~8×에서 실행 가능 여부와 성능 — [4]·R6이 설계상 보장하는 것의 실측.

---

## 12. 착수 순서와 검증 계획

| 단계 | 내용 | 확인 방법 |
|---|---|---|
| 1 | §5 예외 번역 | configs에 local 용량 초과 BLOCK 추가 → **그 config만 건너뛰고** sweep이 끝나면 성공 |
| 2 | §6 early pruning | 같은 config가 **컴파일도 되지 않고** 사라지는지. 해석값 중복([5])이 후보를 줄이는지 |
| 3 | §7 후보 생성 | group·tile·device 축을 포함한 sweep. §2.2·§6.2의 예측(u_dev·전송량)과 실측 순위 대조 — **이 sweep이 그대로 RQ2의 데이터가 된다** |
| 4 | §8.2 기록만 | JSON에 표본이 쌓이는지. 모델 없이 동작이 바뀌지 않아야 한다 |
| 5 | §8.3 적합 | leave-one-out 상관. 낮으면 `top_k`를 키우고 표본을 더 모은다 |

**4단계(기록)를 5단계(적합)보다 먼저 떼어 놓는 것이 요점이다.** 기록은 모델이 없어도
지금 켤 수 있고, 데이터가 쌓여야 모델을 만들 수 있으며, 만들어도 데이터가 없으면
검증할 수 없다.

---

## 13. 미결정 · 위험

- **진단 문구 의존** (§5). 규약(§5.3)으로 완화하지만 컴파일러 쪽 문구가 바뀌면 조용히
  매칭이 풀린다. 그때의 동작은 "탐색 중단"이라 안전한 방향이다. lit 테스트로 문구를
  고정하는 것을 검토한다.
- **Python 쪽 특징 산술이 pass와 어긋날 위험** (§6). `W_group`이
  `pim::localTileBytes`와, R2가 실제 자동 산출과 어긋나면 [1]·[5]가 조용히 틀린다.
  근본 해결은 되돌림 채널(아래)이고, 그 전에는 컴파일 성공 시 번들의 `lanes`와 R3의
  `L`을 대조해 경고한다.
- **`pim.metrics` 되돌림 채널 (나중).** PIM Common pass가 module attribute로 실측치
  (group 크기·lanes·분할·활성 device·MAC·전송 바이트)를 남기고 metadata로 실어 보내면,
  커널별 바이트 함수와 Python 쪽 규칙 사본이 실측치로 대체된다. 무거우므로 §8이 실제로
  유효하다고 확인된 뒤에 붙인다.
- **group 자동 산출과 탐색의 긴장** (§7). R2의 자동값은 local 예산 채움 — reuse만
  보고 `u_dev`를 보지 않는다. pool 관점에서는 자동값 이하가 최적일 수 있으므로 탐색이
  자동값을 상한 후보로 다뤄야 한다. 자동값이 항상 이기면 group 축을 접어도 된다는
  뜻이고, 그것도 RQ2가 판정한다.
- **`num_units`의 이중 의미** (§7). "물리 유닛 개수"와 "이 커널이 쓸 유닛 수"를 한
  필드가 겸한다. 탐색 축으로 쓰면 상한을 별도로 관리해야 한다. `PIMOptions`에 물리
  개수를 따로 두는 편이 나은지 검토한다.
- **U_D·imb 식이 group당 균일 비용을 가정한다** (§6.2 R5). round·경계 group(부분
  tile)과의 상호작용은 미검증 — §12의 3단계(= RQ2)가 재는 자리다.
