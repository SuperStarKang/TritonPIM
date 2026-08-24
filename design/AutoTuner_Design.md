# PIM-aware Hierarchical AutoTuner Design

## 1. 전체 동작 과정

본 AutoTuner의 목적은 PyTorch `aten::`에서 PIM 실행이 가능한 GEMM을 인식한 뒤, 단순히 `BLOCK_M/N/K`만 선택하는 것이 아니라 다음 요소를 **하나의 계층적 configuration space**에서 함께 탐색하는 것이다.

- Triton block shape
- block grouping
- execution tile shape
- active PIM device distribution
- device scheduling
- K-axis 분할
- reduction strategy

최종 목표는 특정 parameter 하나를 최대화하거나 최소화하는 것이 아니다.

예를 들어 device utilization을 최대화하기 위해 `P_K`를 크게 하면 더 많은 PIM device를 동시에 사용할 수 있지만, partial output이 증가하여 reduction과 transfer가 증가할 수 있다.

따라서 AutoTuner가 풀어야 할 문제는 다음과 같다.

$$
c^*
=
\operatorname*{arg\,min}_{c\in\mathcal{C}_{\mathrm{valid}}}
T_{\mathrm{end-to-end}}(c)
$$

여기서 $c$는 PIM configuration이며, $\mathcal{C}_{valid}$는 하드웨어 제약을 만족하는 configuration 집합이다.

---

## 1.1 전체 구조

```text
                           PyTorch Model
                                │
                                ▼
                           torch.compile
                                │
                                ▼
                        Dynamo / FX Graph
                                │
                                ▼
                           ATen Graph
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                           Inductor                            │
│                                                               │
│                        aten::_int_mm                          │
│                             │                                 │
│           ┌─────────────────┼─────────────────┐               │
│           │                 │                 │               │
│           ▼                 ▼                 ▼               │
│       CPU Choice        GPU Choice       PIM Choice           │
│                                               │               │
│                                               ▼               │
│                                ┌────────────────────────┐     │
│                                │   PIM AutoTuner        │     │
│                                │                        │     │
│                                │ Helion-inspired        │     │
│                                │ LFBOTreeSearch         │     │
│                                └───────────┬────────────┘     │
│                                            │                  │
│                                            ▼                  │
│                                   ① PIM Config Space          │
│                                            │                  │
│               ┌────────────────────────────┼──────────────┐   │
│               │            │               │              │   │
│               ▼            ▼               ▼              ▼   │
│             Block        Group        Execution Tile   Distribution
│           BM BN BK      GM GN GK        TM TN TK       PM PN PK
│                                                            │ │
│                                                            │ ▼
│                                                            │ Reduction
│                                                            │ Strategy
│                                            │               │
│                                            ▼               │
│                                ② Constraint Engine          │
│                                            │               │
│                                invalid ────┤               │
│                                            │ valid         │
│                                            ▼               │
│                                      ③ PIM Planner         │
│                                            │               │
│                          ┌─────────────────┼────────────┐   │
│                          ▼                 ▼            ▼   │
│                    Tile Mapping        Transfer      Reduction
│                          │                 │            │   │
│                          └─────────────────┼────────────┘   │
│                                            ▼               │
│                                 pim.device_plan            │
│                                 pim.reduction_plan         │
│                                            │               │
│                                            ▼               │
│                                ④ Plan Feature Extractor    │
│                                            │               │
│                       ┌────────────────────┼────────────────┐
│                       ▼                    ▼                ▼
│                  Utilization          Transfer         Reduction
│                  / Imbalance           Bytes             Bytes
│                       │                    │                │
│                       └────────────────────┼────────────────┘
│                                            │
│                                            ▼
│                                   ⑤ Candidate Lowering
│                                            │
│                                            ▼
│                                       PIMTemplate
│                                            │
│                                            ▼
│                                           TTIR
│                                            │
│                                            ▼
│                                   Triton Shared IR
│                                            │
│                                            ▼
│                                          PIM IR
│                                            │
│                    ┌───────────────────────┼─────────────────────┐
│                    ▼                       ▼                     ▼
│                 UPMEM IR                AiMX IR              HBM-PIM IR
│                    │                       │                     │
│                    └───────────────────────┼─────────────────────┘
│                                            ▼
│                                    ⑥ Real Execution
│                                            │
│                                            ▼
│                                  Benchmark / Breakdown
│                                            │
│                 ┌──────────────────────────┼───────────────────┐
│                 ▼              ▼           ▼         ▼         ▼
│              Transfer       Compute    Reduction   Launch   Total
│                                            │
│                                            ▼
│                                   ⑦ LFBO Training
│                                            │
│                                  Random Forest Classifier
│                                            │
│                                            ▼
│                                      P(Good Config)
│                                            │
│                                            ▼
│                              ⑧ Tree-guided Neighbor Search
│                                            │
│                                            ▼
│                                New Candidate Configs
│                                            │
│                                            └───────────────┐
│                                                            │
│                                  Constraint → Plan → Run ←─┘
│
│                                            │
│                                   Search Convergence
│                                            ▼
│                                    Best PIM Config
│                                            │
│                ┌───────────────────────────┼───────────────┐
│                ▼                           ▼               ▼
│              CPU                         GPU          Best PIM
│                └───────────────────────────┼───────────────┘
│                                            ▼
│                                autotune_select_algorithm
│                                            │
│                                            ▼
│                                          Winner
└────────────────────────────────────────────┬──────────────────┘
                                             │
                                             ▼
                                          Scheduler
                                             │
                                             ▼
                                           Codegen
```

이 구조의 가장 중요한 특징은 **두 단계의 autotuning**이다.

```text
Inner Autotuning
─────────────────────────────
여러 PIM Config
       ↓
Best PIM Config


Outer Autotuning
─────────────────────────────
CPU
GPU
Best PIM
       ↓
Inductor selection
```

PIM 내부의 수많은 configuration을 직접 Inductor에 모두 노출하기보다, 먼저 PIM 내부 search를 수행하여 우수한 PIM choice를 만든 뒤 CPU/GPU choice와 비교한다.

---

# 2. 핵심 설계 원칙

AutoTuner는 다음 네 가지 객체를 명확히 분리한다.

```text
Problem
   +
Hardware
   +
Config
   │
   ▼
Planner
   │
   ▼
Plan
   │
   ▼
Execution
   │
   ▼
Measurement
```

즉,

$$
Plan =
Planner(Problem,\ Hardware,\ Config)
$$

이다.

각 객체의 역할은 다음과 같다.

| 객체 | 의미 |
|---|---|
| `PIMProblem` | 실행해야 하는 GEMM |
| `PIMHardwareProfile` | 사용 가능한 PIM 자원과 제약 |
| `PIMConfig` | AutoTuner가 변경하는 선택 변수 |
| `PIMPlan` | 해당 Config를 실제 hardware에 배치한 결과 |
| `PIMBenchmarkResult` | 실제 실행 결과 |

이 분리가 필요한 가장 큰 이유는 **Config와 실제 실행 배치가 동일한 개념이 아니기 때문**이다.

예를 들어

```text
distribution = N-major
K-split = 2
```

라는 Config가 있더라도 실제 `P_N`, 마지막 scheduling round에서의 active device 수, tile assignment는 문제의 $M,N,K$와 device 수에 따라 달라진다.

따라서 AutoTuner는 **정책을 탐색하고**, Planner는 그 정책으로부터 **구체적인 실행 계획을 생성**한다.

---

# 3. Stage 0 — PIM Tuning Context 생성

## 목적

현재 GEMM과 target PIM hardware를 AutoTuner가 판단할 수 있는 정규화된 형태로 만든다.

### 입력

```text
aten::_int_mm

A : [M, K]
B : [K, N]

dtype
layout
stride
```

그리고 하드웨어 정보:

```text
target
num_units

local_memory_per_unit
device_memory
alignment

execution-unit constraints
target-specific constraints
```

이다.

예를 들어 다음과 같은 구조가 가능하다.

```text
PIMProblem
 ├─ M
 ├─ N
 ├─ K
 ├─ input_dtype
 ├─ output_dtype
 ├─ layout
 └─ residency information


PIMHardwareProfile
 ├─ target = UPMEM / AiMX / HBM-PIM
 ├─ num_units
 ├─ local_memory
 ├─ device_memory
 ├─ alignment
 ├─ execution_unit_count
 └─ target constraints
```

## 왜 필요한가

동일한 configuration이라도 shape와 hardware가 달라지면 전혀 다른 성능을 보인다.

예를 들어

```text
M = 16
```

에서 큰 `T_M`은 대부분 padding이 될 수 있지만,

```text
M = 4096
```

에서는 큰 `T_M`이 데이터 재사용을 높일 수 있다.

또한

```text
256 devices
```

와

```text
2048 devices
```

에서는 최적 $P_M, P_N, P_K$가 달라질 가능성이 높다.

따라서 AutoTuner의 기본 specialization key는 최소한

$$
(M, N, K, \mathrm{dtype}, \mathrm{target}, \mathrm{hardware})
$$

를 포함해야 한다.

---

# 4. Stage 1 — PIM Config Space

## 목적

AutoTuner가 변경할 수 있는 모든 성능 결정 변수를 명시한다.

Config는 다음과 같이 계층화한다.

```text
PIMConfig
│
├── Block Config
│     ├─ BM
│     ├─ BN
│     └─ BK
│
├── Block Group Config
│     ├─ GM
│     ├─ GN
│     └─ GK
│
├── Execution Tile Config
│     ├─ TM
│     ├─ TN
│     └─ TK
│
├── Distribution Config
│     ├─ PM
│     ├─ PN
│     ├─ PK
│     └─ distribution_policy
│
└── Reduction Config
      └─ reduction_policy
```

예:

```text
BLOCK
    BM = 32
    BN = 32
    BK = 16

GROUP
    GM = 4
    GN = 8
    GK = 4

EXECUTION TILE
    TM = 128
    TN = 2048
    TK = 1024

DISTRIBUTION
    PM = 1
    PN = 128
    PK = 2

REDUCTION
    device_tree
```

---

## 4.1 왜 계층적 Search Space가 필요한가

이 parameter들은 서로 독립적이지 않다.

```text
Block
   │
   ▼
Group
   │
   ▼
Execution Tile
   │
   ▼
Device Distribution
   │
   ▼
Reduction
```

의 dependency가 존재한다.

예를 들어 `BLOCK_K`가 증가하면:

```text
local buffer 증가
       ↓
가능한 GROUP_K 감소
       ↓
execution tile 구성 변화
       ↓
K tile 개수 변화
       ↓
split-K 필요성 변화
       ↓
reduction 비용 변화
```

가 발생할 수 있다.

따라서 단순 Cartesian product를 모두 brute force하는 것은 매우 비효율적이다.

---

# 5. Stage 1-A — Derived Configuration

가능하다면 모든 크기를 독립적인 tuning knob로 두지 않는 것이 좋다.

예를 들어 execution tile이 block group의 정수배라는 구조라면:

$$
T_M = B_M G_M E_M
$$

$$
T_N = B_N G_N E_N
$$

$$
T_K = B_K G_K E_K
$$

로 정의할 수 있다.

그러면 tuner는

```text
BM BN BK
GM GN GK
EM EN EK
```

만 탐색하고,

```text
TM TN TK
```

는 자동으로 계산한다.

### 이유

예를 들어 다음 configuration은 구조적으로 이상할 수 있다.

```text
BM = 64
GM = 8

TM = 96
```

`TM`이 block/group hierarchy와 맞지 않는다면 이런 configuration은 애초에 생성해서는 안 된다.

따라서 가능한 값 사이의 구조적 관계는 **search 결과로 발견하도록 두기보다 ConfigSpec 자체에 넣는 것이 좋다.**

---

# 6. Stage 1-B — Semantic Config Fragment

특히 강하게 연관된 parameter는 하나의 semantic tuning knob로 묶는다.

## 예: KSplitFragment

```text
KSplit Strategy 0

PK = 1
reduction = none


KSplit Strategy 1

PK = 2
reduction = device


KSplit Strategy 2

PK = 4
reduction = device


KSplit Strategy 3

PK = 4
reduction = host


KSplit Strategy 4

PK = 8
reduction = tree
```

왜 이렇게 해야 하는가?

다음 configuration은 의미가 없다.

```text
PK = 8
reduction = none
```

따라서 `PK`와 reduction strategy를 무관한 parameter로 탐색하면 invalid configuration이 과도하게 만들어진다.

AutoTuner 입장에서는:

```text
k_split_strategy = 0
k_split_strategy = 1
...
```

와 같은 하나의 categorical search dimension으로 볼 수 있다.

---

## 6.1 DistributionFragment

Device distribution도 다음과 같은 의미 단위로 구성할 수 있다.

```text
N-major

M-major

MN-balanced

NK-balanced

MK-balanced

MNK-balanced
```

그리고 Planner가 실제 tile grid를 보고:

$$
P_M, P_N, P_K
$$

를 계산한다.

이 방식의 장점은 AutoTuner가 단순 숫자가 아니라 **배치 전략 자체**를 탐색한다는 것이다.

---

# 7. Stage 2 — Initial Candidate Generation

LFBO는 아무런 데이터가 없는 상태에서는 어떤 영역이 좋은지 알 수 없다.

따라서 먼저 initial population이 필요하다.

PIM에서는 완전 random만 사용하기보다 **PIM 의미를 갖는 seed**를 같이 사용하는 것이 좋다.

### Seed A — Reduction 최소화

```text
PK = 1
N-major
```

목표:

```text
partial output 최소
reduction 최소
```

### Seed B — Parallelism 최대화

```text
PM / PN을 최대한 사용
PK = 1
```

목표:

```text
output parallelism 최대화
```

### Seed C — Mild Split-K

```text
PK = 2
```

목표:

```text
utilization 증가와
reduction 증가 사이의 절충
```

### Seed D — Aggressive Split-K

```text
PK = 4 or 8
```

특히 output tile 수가 적은 GEMV-like 문제에서 필요할 수 있다.

### Seed E — Large execution tile

목표:

```text
reuse ↑
round 수 ↓
```

### Seed F — Small execution tile

목표:

```text
parallelism ↑
memory footprint ↓
```

이후 나머지 initial population은 random valid configuration으로 채운다.

---

# 8. Stage 3 — Constraint Engine

## 목적

실행해 볼 가치가 없는 configuration을 compile 이전에 제거한다.

```text
Candidate
    │
    ▼
Constraint Engine
    │
 ┌──┴───┐
 │      │
Fail    Pass
 │      │
drop    ▼
       Planner
```

이 과정이 매우 중요한 이유는 PIM search space에는 **명백하게 불가능한 configuration이 매우 많기 때문**이다.

---

# 9. Constraint 1 — Block / Group 구조

예를 들어 group이 block의 집합이라면:

$$
M_G = B_M G_M
$$

$$
N_G = B_N G_N
$$

$$
K_G = B_K G_K
$$

의 관계를 만족해야 한다.

또 execution tile 역시 group boundary를 유지해야 한다면 해당 배수 조건을 검사한다.

---

# 10. Constraint 2 — Local Memory

block group을 실행하기 위해 필요한 local memory를:

$$
S_{\mathrm{local}}
=
S_A+
S_B+
S_C+
S_{\mathrm{runtime}}
$$

라고 하면:

$$
S_{\mathrm{local}}
\le
C_{\mathrm{local}}
$$

이어야 한다.

AutoTuner가 local memory overflow configuration을 실제 target compiler까지 보내서 실패하도록 두지 않는다.

### 이유

```text
candidate
   ↓
target compile
   ↓
memory overflow
   ↓
compile fail
```

까지 진행하면 tuning budget이 낭비된다.

따라서 **수학적으로 확실히 invalid인 configuration은 surrogate model이 배우게 할 필요 없이 제거**한다.

---

# 11. Constraint 3 — PIM Device Memory

execution tile이 PIM device에 들어가야 한다.

일반적으로:

$$
S_{\mathrm{tile}}
=
S_A(T_M,T_K)
+
S_B(T_K,T_N)
+
S_C(T_M,T_N)
+
S_{\mathrm{metadata}}
$$

이다.

resident operand가 있다면:

$$
S_{\mathrm{tile}}
+
S_{\mathrm{resident}}
\le
C_{\mathrm{device}}
$$

이어야 한다.

이 제약이 필요한 이유는 execution tile 크기를 단순 compute efficiency만 보고 키울 수 없기 때문이다.

```text
Tile ↑

reuse ↑
launch/round ↓

하지만

device memory usage ↑
```

라는 trade-off가 존재한다.

---

# 12. Constraint 4 — Active Device Grid

distribution은 반드시:

$$
P_M P_N P_K
\le
N_{\mathrm{units}}
$$

를 만족해야 한다.

또한 tile grid보다 많은 partition을 생성하는 것도 의미가 없다.

예를 들어:

$$
P_N > G_N^{\mathrm{tile}}
$$

이면 실제로는 사용되지 않는 partition이 만들어진다.

따라서:

$$
P_N \le G_N^{\mathrm{tile}}
$$

등의 조건도 고려한다.

---

# 13. Constraint 5 — Target-specific Constraints

공통 Config를 사용하더라도 최종 target은 서로 다른 물리적 제약을 갖는다.

```text
Common Constraints

block hierarchy
memory capacity
device count
reduction legality
      │
      ▼
Target Constraint
      │
 ┌────┼─────────┐
 ▼    ▼         ▼
UPMEM AiMX   HBM-PIM
```

예를 들어 각 target에서는:

```text
alignment
execution width
local execution unit 수
memory granularity
target-specific tile requirement
```

등을 검사할 수 있다.

이렇게 하면 common search space를 유지하면서도 hardware별로 불가능한 candidate를 제거할 수 있다.

---

# 14. Stage 4 — PIM Planner

Constraint를 통과한 Config를 **구체적인 실행 계획으로 변환**한다.

```text
Problem
   +
Hardware
   +
Config
   │
   ▼
PIM Planner
   │
   ├── execution tile grid
   ├── active device grid
   ├── tile assignment
   ├── execution rounds
   ├── transfer placement
   └── reduction topology
```

Planner는 AutoTuner가 아니다.

즉 Planner는:

> "이 Config가 주어졌을 때 실제로 어떻게 실행해야 하는가?"

를 결정한다.

---

# 15. Stage 4-A — Execution Tile Grid

execution tile이:

$$
T_M, T_N, T_K
$$

이면 전체 logical tile grid는:

$$
E_M=
\left\lceil
\frac{M}{T_M}
\right\rceil
$$

$$
E_N=
\left\lceil
\frac{N}{T_N}
\right\rceil
$$

$$
E_K=
\left\lceil
\frac{K}{T_K}
\right\rceil
$$

이다.

따라서 logical execution space는:

```text
E_M × E_N × E_K
```

가 된다.

---

# 16. Stage 4-B — Active Device Grid

Config와 tile grid를 이용하여:

$$
P_M\times P_N\times P_K
$$

active grid를 결정한다.

예를 들어 256 unit에서:

```text
1 × 256 × 1
```

```text
2 × 64 × 2
```

```text
4 × 32 × 2
```

는 모두 256 units를 사용할 수 있다.

하지만 이들은 의미가 완전히 다르다.

### `P_N` 증가

주로 서로 다른 output column을 병렬 처리한다.

```text
Reduction 필요성 낮음
```

### `P_M` 증가

서로 다른 output row를 병렬 처리한다.

```text
Reduction 필요성 낮음
```

### `P_K` 증가

동일한 output tile을 여러 device가 partial sum으로 계산한다.

```text
Compute parallelism ↑

하지만

Reduction ↑
```

이 trade-off가 distribution tuning의 핵심이다.

---

# 17. Stage 4-C — Device Scheduling

logical tile 수가 active device 수보다 작으면:

```text
Tile count <= Device count

        ↓

single round
```

이 가능하다.

반대로:

```text
Tile count > Device count
```

이면:

```text
Round 0
Device set

Round 1
same Device set reuse

Round 2
same Device set reuse
...
```

가 된다.

Planner는 각 round에서:

```text
device 0 → tile ?
device 1 → tile ?
...
```

를 명확하게 결정해야 한다.

결과는 `pim.device_plan`에 기록한다.

---

# 18. Stage 4-D — Reduction Plan

특히:

$$
P_K > 1
$$

인 경우 partial result가 만들어진다.

```text
             K

        ┌────┼────┐
        ▼    ▼    ▼
       K0   K1   K2

        │    │    │
        ▼    ▼    ▼
       D0   D1   D2

        │    │    │
       C0   C1   C2
         \   │   /
          \  │  /
          reduction
              │
              ▼
              C
```

Planner는 최소한 다음을 결정해야 한다.

```text
partial output ownership

reduction location

reduction ordering

reduction rounds

reduction data movement
```

결과는:

```text
pim.reduction_plan
```

으로 표현한다.

---

# 19. 왜 Device Plan과 Reduction Plan을 별도로 만드는가

distribution과 reduction은 연관되어 있지만 동일한 문제는 아니다.

예를 들어:

```text
PM=1
PN=64
PK=4
```

라는 device distribution은 동일해도,

```text
host reduction
```

```text
device-local reduction
```

```text
tree reduction
```

은 서로 다른 실행 방법이다.

따라서:

```text
pim.device_plan
```

은 **누가 어떤 계산을 하는지**,

```text
pim.reduction_plan
```

은 **partial result를 어떻게 최종 결과로 합치는지**

를 표현하게 한다.

이 분리는 이후 target lowering에서도 유용하다.

---

# 20. Stage 5 — Plan Feature Extraction

이 단계가 일반적인 GPU parameter autotuning과 PIM autotuning을 구분하는 중요한 부분이다.

AutoTuner는 단순히:

```text
BM=?
BN=?
PK=?
```

만 Random Forest에 넣지 않는다.

Planner가 생성한 실제 execution plan으로부터 **PIM-specific feature**를 계산한다.

```text
PIM Config
     │
     ▼
PIM Plan
     │
     ▼
Plan Feature Extractor
     │
     ├─ utilization
     ├─ rounds
     ├─ transfer bytes
     ├─ replication
     ├─ reduction bytes
     ├─ padding
     ├─ imbalance
     └─ memory pressure
```

---

# 21. Feature 1 — Effective Device Utilization

단순히:

$$
\frac{P_M P_N P_K}{N_{\mathrm{units}}}
$$

만 사용하는 것은 부족하다.

예를 들어 256 devices를 사용하며:

```text
Round 0 : 256 active
Round 1 : 256 active
Round 2 : 32 active
```

라면 실제 평균 utilization은 낮다.

따라서:

$$
U_{\mathrm{eff}}
=
\frac{
\sum_{r=1}^{R} A_r
}{
R N_{\mathrm{units}}
}
$$

를 사용한다.

- $R$: scheduling round 수
- $A_r$: round $r$에서 실제 active unit 수

위 예에서는:

$$
U_{\mathrm{eff}}
=
\frac{256+256+32}{3\times256}
\approx 0.708
$$

이다.

즉 70.8%의 effective utilization이다.

---

# 22. Feature 2 — Load Imbalance

모든 device가 active하더라도 workload가 균등하다는 보장은 없다.

각 device workload를 $W_d$라고 하면:

$$
I=
\frac{\max_{d} W_d}
{\frac{1}{D}\sum_{d} W_d}
$$

를 사용할 수 있다.

균등한 경우:

$$
I \approx 1
$$

이다.

또는 coefficient of variation:

$$
CV=
\frac{\sigma(W_d)}
{\mu(W_d)}
$$

를 사용할 수 있다.

### 왜 필요한가

```text
256 devices active
```

만 보면 utilization은 높아 보이지만,

```text
128 devices → 10 tiles
128 devices → 1 tile
```

이라면 실행 시간은 heavy device에 의해 결정된다.

따라서 utilization과 imbalance는 반드시 분리해야 한다.

---

# 23. Feature 3 — Transfer Bytes

PIM에서는 단순 tensor 원본 크기가 아니라 **실제로 device 전체에 전달되는 총 데이터**가 중요하다.

$$
B_{\mathrm{H2D}}
=
\sum_d
\left(
B_A^{(d)}
+
B_B^{(d)}
+
B_{\mathrm{metadata}}^{(d)}
\right)
$$

이다.

특히 operand replication을 별도로 기록한다.

```text
A original bytes
A replicated bytes

B original bytes
B replicated bytes

descriptor bytes

output bytes
```

예를 들어 N축 partition으로 A를 여러 device가 공유하지 못하고 각 device에 복제해야 한다면:

```text
PN ↑
   │
   ▼
parallelism ↑

하지만

A replication ↑
```

가 될 수 있다.

따라서 distribution의 효과를 compute만으로 판단해서는 안 된다.

---

# 24. Feature 4 — Reduction Cost

`P_K`가 증가하면 partial output이 증가한다.

따라서 다음 값을 Plan에서 계산한다.

```text
partial output count

reduction bytes

reduction operations

reduction rounds
```

예를 들어:

$$
B_{\mathrm{reduction}}
$$

을 명시적인 feature로 사용한다.

이렇게 하면 Random Forest가:

```text
PK 증가
      ↓
utilization 증가

BUT

reduction_bytes 증가
```

라는 관계를 실제 결과를 통해 학습할 수 있다.

---

# 25. Feature 5 — Padding

tile shape가 problem shape를 정확하게 나누지 못하면 padding이 생긴다.

예:

$$
M_{\mathrm{pad}}
=
\left\lceil
\frac{M}{T_M}
\right\rceil T_M
$$

따라서:

$$
B_{\mathrm{padding}}
=
B_{\mathrm{executed}}
-
B_{\mathrm{useful}}
$$

또는:

$$
R_{\mathrm{padding}}
=
\frac{
\text{padded work}
}{
\text{executed work}
}
$$

를 사용할 수 있다.

Padding은 단순 memory waste가 아니라:

```text
additional transfer
additional compute
additional storage
```

를 동시에 유발할 수 있기 때문에 중요한 feature이다.

---

# 26. Feature 6 — Scheduling Rounds

$$
R
=
\text{number of device-set reuse rounds}
$$

를 기록한다.

같은 총 compute량이라도:

```text
1 round
```

과

```text
64 rounds
```

은 launch/descriptor/runtime overhead가 크게 다를 수 있다.

따라서 round 수 자체도 performance predictor가 된다.

---

# 27. Feature Vector

결과적으로 LFBO에 전달할 feature vector는 다음과 같이 구성한다.

```text
Config Features
────────────────────────

log2(BM)
log2(BN)
log2(BK)

log2(GM)
log2(GN)
log2(GK)

log2(TM)
log2(TN)
log2(TK)

PM
PN
PK

distribution strategy
reduction strategy


Plan Features
────────────────────────

active units

effective utilization

execution tile count

scheduling rounds

H2D bytes
D2H bytes

A replication bytes
B replication bytes

reduction bytes
reduction rounds

padding ratio

load imbalance

local memory usage ratio
device memory usage ratio
```

Power-of-two size는 `log2` 값으로 표현하면 tree가 크기 관계를 학습하기 쉽다.

반면:

```text
host reduction
device reduction
tree reduction
```

처럼 순서가 없는 categorical 값은 임의의 숫자 크기에 의미를 부여하지 않도록 categorical encoding을 사용하는 것이 좋다.

---

# 28. Stage 6 — Candidate Lowering & Compilation

유망한 candidate만 실제 lowering한다.

```text
PIMConfig
    │
    ▼
PIMPlan
    │
    ▼
PIMTemplate
    │
    ▼
TTIR
    │
    ▼
Triton Shared IR
    │
    ▼
PIM IR
    │
    ▼
Target IR
    │
    ▼
Binary / Runtime Plan
```

모든 candidate를 lowering하면 tuning 비용이 지나치게 커지므로 LFBO가 선택한 일부 candidate만 실제 compile한다.

---

# 29. Compile-time Config와 Runtime-plan Config 분리

가능하다면 config를 두 부류로 나눈다.

## Compile-time

```text
BM BN BK

local block/group implementation

kernel reduction structure
```

처럼 generated kernel 자체를 변경하는 값이다.

## Runtime-plan

```text
device assignment

PM PN PK

execution round ordering

device-set reuse
```

중 binary를 바꾸지 않고 runtime metadata로 변경 가능한 값이다.

이 경우:

```text
          Compiled Kernel A
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
      Plan 0    Plan 1     Plan 2
```

와 같이 같은 binary를 여러 Plan에서 재사용할 수 있다.

### 이유

AutoTuning의 실제 비용은 단순 실행 시간만이 아니다.

```text
config generation
+
lowering
+
target compile
+
execution
```

비용이 존재한다.

PIM에서는 target compilation이 비쌀 수 있으므로 binary reuse의 가치가 크다.

---

# 30. Stage 7 — Real Hardware Benchmark

최종 objective는 실제 hardware에서 측정한 end-to-end latency이다.

$$
J(c) = T_{\mathrm{total}}(c)
$$

초기 sequential 구현에서는 breakdown을 다음처럼 생각할 수 있다.

$$
T_{\mathrm{total}}
\approx
T_{\mathrm{stage}}
+
T_{\mathrm{H2D}}
+
T_{\mathrm{compute}}
+
T_{\mathrm{reduction}}
+
T_{\mathrm{D2H}}
+
T_{\mathrm{launch}}
$$

그러나 향후 transfer/compute overlap이나 double buffering을 지원한다면 위 항목의 단순 합이 실제 latency와 같지 않을 수 있다.

따라서 **항상 실제 wall-clock `T_total`을 최종 objective로 사용**한다.

각 breakdown은 objective가 아니라 설명 및 학습을 위한 auxiliary information이다.

---

# 31. 왜 Utilization을 Objective로 사용하지 않는가

다음 두 configuration을 생각할 수 있다.

```text
Config A

utilization = 100%

replication = 매우 큼
reduction = 매우 큼


Config B

utilization = 75%

replication = 작음
reduction = 없음
```

Config B가 전체 latency에서 더 빠를 수 있다.

따라서:

$$
\max\; \mathrm{Utilization}
$$

이 목표가 아니다.

목표는:

$$
\min\, T_{\mathrm{total}}
$$

이며,

```text
utilization
transfer
reduction
padding
imbalance
```

는 성능을 설명하는 feature이다.

---

# 32. Benchmark Result

각 실행 결과는 다음과 같이 보존한다.

```text
PIMBenchmarkResult
│
├─ config
├─ plan
│
├─ total_ms
│
├─ stage_ms
├─ h2d_ms
├─ compute_ms
├─ reduction_ms
├─ d2h_ms
├─ launch_ms
│
├─ effective_utilization
├─ active_units
├─ rounds
│
├─ h2d_bytes
├─ d2h_bytes
├─ replicated_bytes
├─ reduction_bytes
│
├─ padding_ratio
└─ load_imbalance
```

동일 configuration을 여러 번 측정하고 median 등의 robust statistic을 사용하는 것이 좋다.

PIM 실행에서는 host runtime, DMA, scheduling 등으로 measurement noise가 존재할 수 있기 때문에 single measurement만 사용하면 search가 noise에 끌려갈 수 있다.

---

# 33. Stage 8 — LFBO Dataset 구성

이제 지금까지 실제 benchmark한 configuration으로 학습 dataset을 만든다.

$$
X_i =
\left[
\mathrm{Config}_i,\ \mathrm{PlanFeature}_i
\right]
$$

$$
y_i =
\mathrm{PerformanceLabel}_i
$$

이다.

LFBO의 핵심은 latency 자체를 regression하는 것이 아니라 **“좋은 configuration인가?”를 classification하는 것**이다.

예를 들어 상위 약 10%의 configuration을 `good`으로 분류할 수 있다.

```text
fastest top 10%

        ↓

GOOD = 1


remaining configs

        ↓

BAD = 0
```

---

# 34. 왜 Regression이 아니라 Classification인가

목표는 모든 configuration의 latency를 정확하게 예측하는 것이 아니다.

AutoTuner가 알아야 하는 것은:

> "다음에 어떤 configuration을 실제로 실행해 보는 것이 가치가 있는가?"

이다.

예를 들어 매우 느린 configuration이:

```text
300 ms
```

인지

```text
500 ms
```

인지는 큰 의미가 없다.

둘 다 현재 best가 20 ms라면 관심 대상이 아니다.

반면:

```text
18 ms
20 ms
22 ms
```

근처의 configuration을 정확하게 구분하는 것이 중요하다.

Classification은 좋은 영역에 model capacity를 집중하기에 적합하다.

---

# 35. Invalid와 Failed Configuration은 구분한다

### Static Invalid

예:

```text
PM × PN × PK > num_units

local memory overflow

invalid block hierarchy
```

이런 candidate는 Constraint Engine에서 제거한다.

LFBO training에도 넣지 않는 것이 기본이다.

왜냐하면 이미 deterministic rule로 제거할 수 있기 때문이다.

### Compile / Runtime Failure

반대로 static constraint로 예상하지 못한:

```text
target compiler failure

runtime failure

timeout
```

등은 BAD sample로 기록할 수 있다.

이런 정보는 Random Forest가 위험한 영역을 피하도록 하는 데 도움이 된다.

---

# 36. Stage 9 — Random Forest Surrogate

수집한 데이터를 이용해:

```text
Config + Plan Features
          │
          ▼
    Random Forest
          │
          ▼
     P(GOOD)
```

를 예측한다.

예를 들어 한 decision tree가 다음과 같은 구조를 학습할 수 있다.

```text
          effective_util > 0.70?
              /          \
            no            yes
           BAD             │
                           ▼
             reduction_bytes < 8 MB?
                    /             \
                  yes              no
                   │               BAD
                   ▼
                  PK <= 4?
                   /   \
                 yes    no
                  │     BAD
                  ▼
                GOOD
```

또 다른 tree는:

```text
                 TN >= 1024?
                   /    \
                 no      yes
                BAD       │
                          ▼
                replication < threshold?
                     /          \
                   yes           no
                    │            BAD
                    ▼
                  GOOD
```

와 같은 패턴을 찾을 수 있다.

즉 AutoTuner가 명시적으로:

```text
PK는 항상 2가 좋다.
```

와 같은 rule을 작성하지 않는다.

실제 hardware measurement를 통해:

```text
이 shape와 이 hardware에서는
PK=2가 좋은 영역을 만든다.
```

를 학습한다.

---

# 37. Stage 10 — Search Copies

현재까지 가장 좋은 configuration 여러 개를 search starting point로 유지한다.

예:

```text
Best Config 1
Best Config 2
Best Config 3
Best Config 4
Best Config 5
```

### 이유

PIM search space에는 여러 종류의 local optimum이 존재할 가능성이 높다.

예를 들어:

```text
Family A

PK=1
large tile
low transfer
```

와

```text
Family B

PK=4
small tile
high parallelism
```

모두 각자 좋은 영역을 형성할 수 있다.

Best 하나만 따라가면 다른 전략 family를 너무 빨리 버릴 수 있다.

---

# 38. Stage 11 — Tree-guided Neighbor Generation

Random Forest를 단순 candidate scorer로만 사용하는 것이 아니라, decision tree가 현재 좋은 configuration을 분류할 때 사용한 parameter를 이용해 새로운 neighbor 탐색을 guide한다.

PIM에서는 이를 다음처럼 적용한다.

```text
Current Good Config
        │
        ▼
Random Forest의 tree 하나 선택
        │
        ▼
해당 Config의 decision path 분석
        │
        ▼
중요하게 사용된 feature 확인
        │
        ▼
관련 PIM Config Fragment 찾기
        │
        ▼
Semantic Neighbor 생성
```

---

# 39. PIM에서는 Feature → Config Influence Mapping이 필요하다

예를 들어 decision tree가:

```text
reduction_bytes
```

를 중요하게 사용했다고 하자.

하지만 `reduction_bytes`는 직접 tuning parameter가 아니다.

다음 parameter의 결과이다.

```text
PK

TK

reduction strategy

output tile size
```

따라서 다음 mapping을 유지한다.

```text
reduction_bytes
      │
      ├─ KSplitFragment
      ├─ ExecutionTile-K
      └─ ReductionFragment


effective_utilization
      │
      ├─ DistributionFragment
      ├─ ExecutionTile
      └─ SchedulingFragment


replication_bytes
      │
      ├─ DistributionFragment
      ├─ Tile shape
      └─ Residency policy
```

즉 tree가 **어떤 결과 feature가 중요한지** 알려주면 AutoTuner는 그 feature에 영향을 미치는 semantic Config를 변경한다.

---

# 40. Semantic Neighbor 예시

현재 best가:

```text
TM = 128
TN = 1024
TK = 1024

PM = 1
PN = 64
PK = 2

reduction = device
```

라고 하자.

Tree path가:

```text
effective_utilization
       ↓
reduction_bytes
       ↓
TN
```

을 중요하게 봤다면 다음 neighbor를 만들 수 있다.

```text
Candidate A
TN : 1024 → 2048


Candidate B
PK : 2 → 1
reduction : device → none


Candidate C
PK : 2 → 4
reduction : device → tree


Candidate D
Distribution:
N-major → MN-balanced
```

즉 모든 knob를 random하게 흔드는 것이 아니라 **현재 model이 중요하다고 본 성능 요인을 바꿀 수 있는 configuration을 우선적으로 생성**한다.

---

# 41. Stage 12 — Candidate Scoring

새로 만들어진 수백 개 candidate를 모두 실제 실행하지 않는다.

먼저:

```text
Candidate
   │
   ▼
Constraint
   │
   ▼
Planner
   │
   ▼
Plan Features
   │
   ▼
Random Forest
   │
   ▼
P(GOOD)
```

를 계산한다.

그리고 높은 확률을 가진 일부만 실제 compile한다.

예:

```text
300 generated candidates

        ↓

Random Forest

        ↓

Top 10%

        ↓

30 candidates

        ↓

compile + benchmark
```

이것이 LFBO를 사용하는 핵심 이유이다.

---

# 42. Stage 13 — Diversity Penalty

단순히 `P(GOOD)`만 높은 candidate를 선택하면 거의 동일한 configuration이 반복 선택될 수 있다.

예:

```text
C0: TN=1024 PK=2
C1: TN=1024 PK=2
C2: TN=1024 PK=2
C3: TN=2048 PK=2
...
```

이렇게 한 영역만 측정하면 같은 benchmark budget으로 얻는 정보가 적다.

따라서:

```text
high P(GOOD)
        +
different from already selected candidates
```

인 candidate를 선호한다.

좋은 batch는 예를 들어:

```text
Candidate 0
large tile / PK=1

Candidate 1
medium tile / PK=2

Candidate 2
small tile / PK=4

Candidate 3
MN-balanced

Candidate 4
N-major
```

처럼 서로 다른 performance region을 탐색해야 한다.

---

# 43. Stage 14 — Iterative Search

전체 LFBO loop는 다음과 같다.

```text
Initial Seeds
      │
      ▼
Benchmark
      │
      ▼
Training Dataset
      │
      ▼
Random Forest
      │
      ▼
Tree-guided Candidate Generation
      │
      ▼
Constraint
      │
      ▼
Planner
      │
      ▼
Plan Features
      │
      ▼
RF Scoring
      │
      ▼
Diversity Selection
      │
      ▼
Compile + Benchmark
      │
      ▼
New Measurements
      │
      └────────────────────┐
                           │
                 Random Forest Retrain
                           │
                           └────── ↺
```

---

# 44. Stage 15 — Search Termination

다음 조건 중 하나 이상을 사용할 수 있다.

### Generation budget

```text
max_generations
```

### Benchmark budget

```text
max_benchmarks
```

### Wall-clock budget

```text
max_autotune_seconds
```

### Improvement threshold

현재 best가 여러 generation 동안 거의 개선되지 않으면 종료한다.

$$
\frac{
T_{\text{prev-best}}-
T_{\text{new-best}}
}{
T_{\text{prev-best}}
}
<\epsilon
$$

### Patience

```text
N generations 동안
meaningful improvement 없음

        ↓

stop
```

---

# 45. Stage 16 — Config Cache

최종 best configuration은 저장한다.

```text
Tuning Key
     │
     ▼
Best PIM Config
     +
Plan metadata
```

cache key에는 최소한 다음을 포함하는 것이 좋다.

```text
M
N
K

dtype
layout

target type
target architecture/version

num_units

local-memory capacity
device-memory capacity

PIM template version/hash

compiler version

residency state
```

특히 hardware 또는 template code가 바뀌었는데 이전 configuration을 잘못 재사용하는 것을 막기 위해 version/hash가 필요하다.

---

# 46. Cache Warm Start

정확히 동일한 shape가 아니더라도 유사한 과거 configuration을 initial seed로 사용할 수 있다.

예:

```text
Old

M=512
N=8192
K=8192

best config A


New

M=1024
N=8192
K=8192

        ↓

config A를 initial seed로 사용
```

이 방식은 search를 처음부터 random하게 시작하는 것보다 빠르게 좋은 영역으로 접근할 수 있다.

---

# 47. Stage 17 — Best PIM Choice를 Inductor로 반환

Inner PIM search가 끝나면:

```text
Best PIM Config
       +
Best PIM Plan
```

으로 하나의 `PIMTemplateChoice`를 만든다.

그 결과:

```text
CPU Choices
     │
GPU Choices
     │
Best PIM Choice
     │
     ▼
autotune_select_algorithm
     │
     ▼
CPU / GPU / PIM Winner
```

가 된다.

---

# 48. 왜 PIM의 수백 개 Config를 직접 Inductor에 넣지 않는가

다음 구조는 피하는 것이 좋다.

```text
CPU 1

GPU 1
GPU 2

PIM cfg0
PIM cfg1
PIM cfg2
...
PIM cfg500

       ↓

autotune_select_algorithm
```

이유는 PIM configuration 간에는 다음과 같은 강한 계층적 dependency가 존재하기 때문이다.

```text
block
  ↓
group
  ↓
tile
  ↓
memory feasibility
  ↓
device distribution
  ↓
reduction
```

이를 단순 finite choice list로 만들면 invalid configuration 생성과 compile 비용이 매우 커진다.

따라서:

```text
PIM specialized tuner
       ↓
Best PIM

       +

CPU / GPU

       ↓
Inductor selector
```

의 두 단계 구조가 적합하다.

---

# 49. PIM에서 LFBO가 실제로 학습해야 하는 핵심 Trade-off

## Case 1 — 충분한 Output Parallelism

예:

```text
M large
N large
```

이면:

```text
PM / PN만 사용

        ↓

충분한 device parallelism

        ↓

PK=1 가능

        ↓

reduction 제거
```

가 최적일 가능성이 있다.

## Case 2 — GEMV-like Shape

예:

```text
M small
```

이면 output tile 수가 적다.

```text
PM × PN
    │
    ▼
사용 가능한 parallelism 부족
    │
    ▼
device utilization 낮음
```

이 경우:

```text
PK ↑
```

로 추가 병렬성을 만들 수 있다.

하지만:

```text
PK ↑
 │
 ├─ compute parallelism ↑
 ├─ utilization ↑
 │
 ├─ partial output ↑
 ├─ reduction ↑
 └─ transfer ↑
```

가 된다.

AutoTuner가 찾아야 하는 것은:

$$
\text{parallelism gain}
>
\text{reduction + transfer overhead}
$$

가 되는 지점이다.

이를 static heuristic 하나로 결정하지 않고 hardware measurement로 판단한다.

---

# 50. 핵심 Cost Relationship

PIM AutoTuning의 본질적인 관계는 다음과 같다.

```text
                   Execution Tile
                         │
             ┌───────────┼────────────┐
             ▼           ▼            ▼
         Reuse ↑     Tile Count ↓   Memory ↑
                                      │
                                      ▼
                               capacity constraint


                 Device Distribution
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Utilization      Replication       Reduction
        ↑                ↑                ↑
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                    Total Latency
```

즉:

$$
\text{Tile}
\leftrightarrow
\text{Memory}
\leftrightarrow
\text{Parallelism}
\leftrightarrow
\text{Replication}
\leftrightarrow
\text{Transfer}
\leftrightarrow
\text{Reduction}
$$

이라는 상호 의존적인 search problem이다.

이것이 단순한 Triton `BLOCK_M/N/K` tuning과 가장 크게 다른 부분이다.

---

# 51. AutoTuner Interface 제안

## PIMProblem

```text
PIMProblem {
    M
    N
    K

    input_dtype
    output_dtype

    A_layout
    B_layout

    A_resident
    B_resident
}
```

## PIMHardwareProfile

```text
PIMHardwareProfile {
    target

    num_units

    local_memory_bytes
    device_memory_bytes

    alignment

    execution_units_per_device

    target_constraints
}
```

## PIMConfig

```text
PIMConfig {
    block_m
    block_n
    block_k

    group_m
    group_n
    group_k

    execution_m
    execution_n
    execution_k

    distribution_policy

    p_m
    p_n
    p_k

    reduction_policy
}
```

## PIMPlan

```text
PIMPlan {
    execution_tile_shape

    execution_grid

    active_device_grid

    num_rounds

    device_assignments

    transfer_plan

    reduction_plan
}
```

## PIMPlanFeatures

```text
PIMPlanFeatures {
    effective_utilization

    load_imbalance

    h2d_bytes
    d2h_bytes

    a_replication_bytes
    b_replication_bytes

    reduction_bytes
    reduction_rounds

    padding_ratio

    scheduling_rounds

    local_memory_pressure
    device_memory_pressure
}
```

## PIMBenchmarkResult

```text
PIMBenchmarkResult {
    config
    plan

    total_ms

    stage_ms
    h2d_ms
    compute_ms
    reduction_ms
    d2h_ms
    launch_ms

    plan_features

    compile_status
    correctness
}
```

---

# 52. AutoTuner Pseudocode

```text
function tune(problem, hardware):

    spec = build_config_space(problem, hardware)

    cached = lookup_similar_configs(
        problem,
        hardware
    )

    population =
        heuristic_seeds
        + cached
        + random_valid_configs

    results = []

    ------------------------------------------------

    for config in population:

        if not constraints.valid(config):
            continue

        plan = planner.build(
            problem,
            hardware,
            config
        )

        features =
            extract_plan_features(plan)

        result =
            compile_and_benchmark(
                config,
                plan
            )

        results.append(
            config,
            features,
            result
        )

    ------------------------------------------------

    while not stop_condition():

        labels =
            make_quantile_labels(results)

        forest =
            train_random_forest(
                configs,
                plan_features,
                labels
            )

        search_copies =
            select_best_configs(results)

        candidates = []

        for config in search_copies:

            tree =
                forest.sample_tree()

            important_features =
                tree.trace(config)

            fragments =
                influence_map(
                    important_features
                )

            neighbors =
                semantic_mutate(
                    config,
                    fragments
                )

            candidates += neighbors

        ------------------------------------------------

        valid_candidates = []

        for config in candidates:

            if not constraints.valid(config):
                continue

            plan =
                planner.build(
                    problem,
                    hardware,
                    config
                )

            features =
                extract_plan_features(plan)

            probability =
                forest.predict_good(
                    config,
                    features
                )

            valid_candidates.append(
                config,
                plan,
                features,
                probability
            )

        ------------------------------------------------

        selected =
            rank_with_diversity_penalty(
                valid_candidates
            )

        ------------------------------------------------

        for candidate in selected:

            result =
                compile_and_benchmark(
                    candidate.config,
                    candidate.plan
                )

            results.append(result)

    ------------------------------------------------

    best =
        argmin(results.total_ms)

    cache(best)

    return best
```

---

# 53. 단계별 구현 우선순위

처음부터 모든 optimization을 search space에 넣는 것은 권장하지 않는다.

## Phase 1 — Core Distribution AutoTuning

우선 다음만 넣는다.

```text
BM BN BK

GM GN GK

TM TN TK

PM PN PK

distribution policy

reduction policy
```

Plan feature:

```text
effective utilization

rounds

transfer bytes

reduction bytes

padding

imbalance
```

목표:

```text
device distribution autotuning이
실제로 fixed heuristic보다 좋은가?
```

를 검증한다.

## Phase 2 — Residency / Replication

그다음:

```text
A resident

B resident

replication policy

streaming policy
```

를 추가한다.

이 단계부터:

```text
transfer minimization
```

과 PIM memory 특성이 더 강하게 search space에 반영된다.

## Phase 3 — Persistent Execution

추가 knob:

```text
descriptor reuse

device-set reuse

persistent execution

command reuse
```

를 넣는다.

Plan feature:

```text
descriptor_build_count

descriptor_transfer_bytes

descriptor_reuse_factor
```

등을 추가한다.

## Phase 4 — Transfer / Compute Pipeline

마지막으로:

```text
stage size

double buffering

prefetch depth

transfer/compute overlap
```

등을 탐색한다.

이 단계에서는 component latency의 합 대신 반드시 실제 pipeline makespan:

$$
T_{\mathrm{wall-clock}}
$$

을 objective로 사용한다.

---

# 54. 최종 설계의 핵심

본 설계에서 Helion으로부터 가져오는 것은 Helion IR이나 Helion frontend가 아니다.

가져오는 핵심은 다음 세 가지이다.

### 1. Classification-based LFBO

```text
모든 latency를 정확히 예측

             X

좋은 config 영역을 식별

             O
```

### 2. Tree-guided Search

```text
모든 parameter random mutation

             X

현재 surrogate가 중요하게 본
parameter 중심의 neighbor search

             O
```

### 3. Diversity-aware Batch Selection

```text
비슷한 후보만 반복 benchmark

             X

서로 다른 promising region을
동시에 탐색

             O
```

---

# 55. 본 PIM AutoTuner의 차별화된 핵심

Helion의 search mechanism을 그대로 가져오는 것만으로는 PIM AutoTuner가 완성되지 않는다.

본 설계의 핵심은 **LFBO 앞에 PIM-specific planning layer가 존재한다는 것**이다.

```text
Config
   │
   ▼
Hardware Constraint
   │
   ▼
PIM Plan
   │
   ▼
Plan-derived Features
   │
   ▼
LFBO
```

즉 AutoTuner가 직접 학습하는 것은 단순히:

```text
BLOCK_M=32가 빠르다.
```

가 아니다.

궁극적으로는:

```text
이 GEMM shape에서는

이 execution tile이
device memory에 적합하고,

이 distribution이
device를 충분히 활용하면서,

operand replication은 과도하지 않고,

split-K로 얻는 compute parallelism이
추가 reduction 비용보다 크므로,

이 Config가 가장 빠르다.
```

라는 관계이다.

따라서 본 설계를 한 식으로 정리하면:

$$
\begin{aligned}
\text{Config}
&\rightarrow \text{Constraint}
\\
&\rightarrow \text{PIM Plan}
\\
&\rightarrow \{\text{Utilization},\ \text{Transfer},\ \text{Reduction},\ \text{Padding},\ \text{Imbalance}\}
\\
&\rightarrow \text{Real Execution}
\\
&\rightarrow \text{LFBO Feedback}
\\
&\rightarrow \text{Better Config}
\end{aligned}
$$

이다.

최종적으로 AutoTuner가 최적화하는 대상은:

$$
\text{Block}
+
\text{Group}
+
\text{Execution Tile}
+
\text{Device Distribution}
+
\text{Reduction}
$$

이며,

그 결과로:

$$
\min\, T_{\mathrm{end-to-end}}
$$

를 만족하는 `pim.device_plan`과 `pim.reduction_plan`을 생성하는 것이 본 AutoTuner의 최종 목표이다.

---

# 56. 참고 설계 원칙 요약

본 AutoTuner는 Helion의 `LFBOTreeSearch` 설계에서 다음 아이디어를 참고한다.

- Random Forest 기반 good/bad configuration classification
- 상위 성능 configuration을 중심으로 한 search copies
- decision-tree path를 이용한 중요 knob 중심 neighbor 생성
- surrogate prediction을 이용한 expensive benchmark 후보 축소
- similarity/diversity를 고려한 batch candidate 선택
- 이전 tuning 결과를 활용한 warm start

그러나 PIM-specific search space, memory constraints, device distribution, reduction plan, operand replication, transfer cost, effective utilization 및 imbalance는 별도의 PIM Planner와 Plan Feature Extractor에서 모델링한다.

즉 전체 설계의 핵심은 다음과 같다.

```text
Helion-inspired Search Mechanism
                +
PIM-specific Config Space
                +
PIM-specific Constraint Engine
                +
PIM-specific Planner
                +
PIM-specific Plan Features
                +
Real Hardware Measurement
```

이를 통해 단순 block-size autotuning이 아니라 **PIM의 memory-centric execution과 multi-device distribution을 동시에 고려하는 hierarchical autotuning**을 수행한다.
