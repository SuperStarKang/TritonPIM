# PIM Common — 연산(compute) lowering 설계

> TritonPIM의 PIM Common 중간층 설계 중 **연산** 부분 정리.
> 입력은 `--triton-to-linalg-experimental` 출력(memref+linalg), 목표는 target 무관한 PIM Common IR로
> 연산을 표현하고, target에서 지원 안 하는 연산은 기본 연산으로 분해(legalization)하는 것.
>
> 관련 문서:
> - 메모리 접근 lowering 설계 → `pim-common-memory-lowering-design.md` (이하 "메모리 설계 문서")
> - 연산 패턴 전수 → `triton-shared-compute-patterns.md` (이하 "연산 패턴 문서")
> - 메모리 접근 패턴 전수 → `triton-shared-memory-access-patterns.md`

---

## 0. 연산은 메모리와 근본적으로 다르다

| | 메모리 | 연산 |
|---|---|---|
| 입력 형태 | `reinterpret_cast`+`alloc`+`copy` 삼중구조 | 이미 `linalg.matmul`/`reduce`/`generic` (깔끔) |
| 새 dialect | 필요 (`pim.access` — strided/wrap/gather 표현할 게 없음) | **최소화** (linalg가 이미 완벽) |
| 핵심 과제 | 불필요한 버퍼 제거 | (1) 무엇을 named op 유지, (2) 지원 안 하는 연산 분해 |

**핵심 원칙**: 연산은 새 op을 최소화하고 **기존 linalg를 최대한 재사용**한다. `pim.matmul` 같은 걸
만들지 않는다 — `linalg.matmul`로 충분하다.

---

## 1. 연산 표현: 두 종류로 나눈다

연산에는 성격이 다른 두 종류가 있고, 표현 방식이 갈린다.

### (1) 구조화 연산 (structured) — named op 유지

- `linalg.matmul`, `linalg.reduce` — 시그니처 명확, 두 target이 **하드웨어 원시 연산으로 직접 지원**
- 근거: matmul.c(`OPCODE_MATMUL`), elewise.c(`OPCODE_ADD/SUB/MUL`)가 실증 — AiMX가 descriptor 하나로 처리
- **풀지 않고 named op 그대로** 둔다. target이 descriptor/명령으로 직접 매핑

### (2) 자유형 연산 (freeform) — body 분석 필요

- `linalg.generic` — body 안에 임의의 arith/math 조합
- 두 target의 지원 여부가 **body 내용에 따라 갈림** (add는 지원, 임의 함수는 아닐 수 있음)
- body를 분석해서 target 원시 연산에 매핑되는지 판단

### 표현 전략 (연산 패턴 문서 범주 → PIM Common)

| 연산 패턴 범주 | PIM Common 표현 | 근거 |
|---|---|---|
| B. matmul | `linalg.matmul` 유지 | 두 target 원시 지원 |
| C. reduce | `linalg.reduce` 유지 | (AiMX 지원 범위 확인 필요) |
| A. pointwise | `linalg.generic` + body 분류 태그 | body가 원시 연산에 매핑되나? |
| D. scan | `linalg.generic` 유지 | " |
| E–H. 레이아웃/생성 | `tensor.*` / `linalg` 유지 | 데이터 이동 없음, 대부분 no-op화 |

즉 **연산은 대부분 linalg를 그대로 두되, `linalg.generic`만 body 분류가 필요**하다.

---

## 2. 연산-메모리 결합: 분리해서 두고, 결합은 target이 한다

matmul.c에서 봤듯 AiMX는 **A descriptor(메모리) → B descriptor(메모리) → matmul(연산)**이 하나의
descriptor chain에 섞인다. PIM Common에서는 이를 **값 의미론으로 명확히 분리**한다.

```mlir
%a = pim.load %dA : !pim.access<...> -> tensor<...>          // 메모리: A 가져오기
%b = pim.load %dB : !pim.access<...> -> tensor<...>          // 메모리: B 가져오기
%c = linalg.matmul ins(%a, %b) outs(%init) -> tensor<...>    // 연산: matmul
     pim.store %c, %dC                                        // 메모리: C 내보내기
```

- **PIM Common 레벨**: load/matmul/store가 **데이터 의존성**으로 연결(%a → matmul → %c). "무엇이 무엇에
  의존하는가"만 표현.
- **target 레벨**: 이 의존성 그래프를 보고 결합. AiMX는 하나의 **descriptor chain**으로 엮음(A/B/C
  descriptor + OPCODE_MATMUL). UPMEM은 "load 명령 + MAC 루프 + store 명령"으로.

**왜 분리가 맞나**: 결합 방식이 target마다 다름(AiMX=chain, UPMEM=순차 명령). PIM Common에서 미리
결합하면 한 방식에 묶임. **의존성만 남기고 결합은 유예**하는 것이 메모리 설계의 "버퍼 암묵화"와 같은
철학. 값 의미론(load=Read, matmul=pure, store=Write)이 이를 받쳐줌 — 의존성이 SSA로 명확하니 target이
어떤 순서/형태로든 엮을 수 있음.

---

## 3. 지원 안 하는 연산 분해 (legalization)

"target 기본 연산으로 분해" = 컴파일러 용어 **legalization**. "이 target이 이해하는 연산 집합(legal
set)으로 바꾼다". MLIR `ConversionTarget`이 이 메커니즘.

### 구조

```
각 target이 "내가 지원하는 연산" 선언 (legal set)
  UPMEM legal:  matmul, reduce, 임의 arith/math pointwise (범용 프로세서)
  AiMX legal:   matmul, add/sub/mul, bias(broadcast) — descriptor로 표현되는 것만

연산이 legal set에 있으면  → 그대로 매핑
연산이 legal set에 없으면  → 분해 규칙 적용 → legal 연산들로
```

**핵심: target마다 legal set이 다르므로, 같은 연산도 분해 여부가 다르다.**

예 — `exp` (softmax의 일부):
- **UPMEM**: `math.exp` 지원(범용, libm 유사) → 분해 불필요, 그대로
- **AiMX**: descriptor에 exp 없음 → **분해** (다항식 근사 `exp(x) ≈ 1 + x + x²/2 + ...`) 또는 host

예 — `division`:
- AiMX가 mul만 지원 → `a/b` = `a * (1/b)`, `1/b`는 Newton-Raphson 반복(mul/sub)으로 분해

### 분해는 target lowering에서 (PIM Common 아님)

- PIM Common은 `linalg.generic { body: math.exp }`를 **그대로 둠** (target 무관하게 "exp를 한다"만 표현)
- UPMEM target: exp가 legal → 유지
- AiMX target: exp가 illegal → 분해 규칙 발동 → add/sub/mul 조합

**이유**: 무엇이 legal인지가 target마다 다르므로 분해도 target 지식이 필요. PIM Common에서 미리 분해하면
UPMEM에서 불필요하게 풀린 형태가 됨. (메모리 설계에서 "물리 재작성은 target"과 같은 논리.)

### illegal 연산의 세 가지 결말

| 결말 | 언제 | 예 |
|---|---|---|
| **분해 (decompose)** | legal 연산 조합으로 표현 가능 | `div` → `mul`+Newton, `exp` → 다항식 근사 |
| **host fallback** | 분해가 비현실적이거나 부정확 | 복잡한 초월함수, 임의 gather |
| **컴파일 에러** | 분해도 fallback도 불가/무의미 | 지원 불가능한 근본적 연산 |

목표는 **분해를 최대한 활용** — 사용자는 `exp`를 쓰고, AiMX에서도 컴파일러가 근사로 돌아가게 하는 것이
"컴파일러가 알아서 하드웨어를 잘 쓰게" 하는 핵심.

### 분해 규칙 = target별 rewrite pattern 라이브러리

target별로 분해 rewrite pattern을 등록("exp → 다항식", "div → Newton"). illegal 연산을 만나면 매칭.
재사용 가능한 라이브러리처럼 축적 — 새 연산 지원이 필요하면 분해 패턴 하나 추가.

### 근사 정확도 자동 결정

분해 시 정확도/연산량 trade-off(예: exp 다항식 근사 차수)는 **컴파일러가 자동으로** 정한다. 근사 차수를
사용자가 아니라 컴파일러가 결정하여, 사용자는 논리 연산만 쓰고 컴파일러가 target에 맞는 근사 수준을
선택. (구체적 결정 방법 — 고정 차수 vs autotuning 탐색 vs 정확도 목표 기반 — 은 추후.)

---

## 4. PIM Common에서의 연산 최적화: body "안"이 아니라 body "사이"

**결론: body 내부(산술 자체)는 최적화 여지가 적고, op 사이(연산 간 구조)가 PIM Common의 최적화 영역.**

### body 내부는 왜 손댈 게 적나

- **이미 깔끔**: `--triton-arith-to-linalg`가 만든 `linalg.generic` body는 arith/math 최소 조합. 구조적
  낭비 거의 없음.
- **산술 최적화는 upstream이 함**: `x*1`→`x`, `x+0`→`x`, 상수 폴딩 등은 표준 canonicalization이 처리.
  PIM Common이 손대면 바퀴 재발명.
- **body 변형의 상당수가 target 의존**: 분해(`div`→Newton), 근사(`exp`→다항식)는 body 변형이지만 legal
  set(target)에 의존 → target lowering 몫.

### op 사이 구조 최적화 (PIM Common 영역, target 무관)

**1. 연산 융합 (fusion) — 중간 텐서 제거 (제일 중요)**

메모리의 "불필요한 버퍼 제거"의 연산 버전.

```mlir
// BEFORE — 중간 텐서 %t 발생
%t = linalg.generic { %x + %y }        // add → 중간 텐서
%r = linalg.generic { %t * %z }        // mul (%t 소비)

// AFTER — 하나로 융합, 중간 텐서 소멸
%r = linalg.generic { (%x + %y) * %z } // add+mul 한 body에
```

- **target 무관** — 두 pointwise 융합은 어느 하드웨어에서도 유효
- **메모리 절감과 직결** — 중간 텐서가 없어지면 그걸 담을 staging 버퍼도 없어짐 (메모리 설계 §6과 연결)
- **linalg가 이미 지원** — MLIR 표준 fusion
- 두 target 다 이득: AiMX는 descriptor 수 감소, UPMEM은 중간 버퍼 write/read 왕복 소멸

**2. load-compute 융합 — 메모리 설계 §6.5와 연결**

```mlir
%a = pim.load %dA               // A를 버퍼로
%r = linalg.generic { f(%a) }   // 그 버퍼를 소비
// → "load → generic" 패턴을 융합 단위로 태깅 → target이 chain으로 엮기 쉬움
```

**3. reduction 결합** — 같은 축의 여러 reduce(예: mean의 sum과 count)를 한 순회로. target 무관.

**4. 죽은 연산 제거 (DCE)** — 결과 미사용 `linalg.generic` 제거. 값 의미론(연산 pure) 위 표준.

### 최적화 위치 정리

| 최적화 | body "안" vs "사이" | PIM Common에서? | 이유 |
|---|---|---|---|
| 산술 단순화 (`x*1`→`x`) | 안 | ❌ (upstream) | 표준 canonicalization |
| 상수 폴딩 | 안 | ❌ (upstream) | 표준 |
| **elementwise 융합** | 사이 | ✅ | target 무관, 중간 텐서 제거 |
| **load-compute 융합** | 사이 | ✅ (태깅) | 메모리 §6.5와 연결 |
| **reduction 결합** | 사이 | ✅ | target 무관 |
| **DCE** | 사이 | ✅ | 값 의미론 위 표준 |
| 분해 (`div`→Newton) | 안 | ❌ (target) | legal set이 target 의존 |
| 근사 (`exp`→다항식) | 안 | ❌ (target) | 정확도/연산량이 target 의존 |

### 메모리 설계와의 대칭

이 구분은 우연이 아니다. PIM Common의 역할은 **"target 무관한 구조 최적화"**이고, 구조는 항상 op
**사이**의 관계:

- 메모리: transfer들 사이의 중복/왕복 제거 (passthrough/CSE/LICM)
- 연산: compute들 사이의 융합/중간값 제거 (fusion/결합/DCE)

op **내부**(메모리 필드값, 연산 산술)는 (a) 이미 최소이거나 (b) target 물리 의존 → 앞단(canonicalization)
또는 뒷단(target)이 담당. PIM Common은 그 사이 **연결 관계만** 최적화.

특히 **융합**은 메모리와 연산을 잇는 최적화라 가치가 큼 — elementwise를 합치면 중간 텐서가 없어지고, 중간
텐서가 없어지면 메모리 staging 버퍼도 없어짐. **연산 융합 → 메모리 절감**. 이것이 "PIM에 맞게 최적화"의
실질.

---

## 5. lowering 파이프라인 (연산 부분)

메모리 설계 문서 §6의 Stage C를 구체화.

```
PIM Common raising 중 연산 처리:
  ├─ 구조화 연산(matmul/reduce) → named op 유지 (scalar 루프로 풀지 않음)
  ├─ 자유형 연산(generic)       → body 분류 태그 (pointwise/broadcast/scan)
  └─ op 사이 최적화             → fusion / load-compute 융합 / reduction 결합 / DCE
  │
PIM Common IR: named linalg 유지 + generic 분류 + 융합됨, 연산-메모리는 의존성으로 분리

target lowering (연산 분해):
  ├─ legal 연산      → 직접 매핑 (matmul→chain/MAC, add→opcode)
  ├─ illegal 연산    → 분해 (target rewrite pattern) 또는 host fallback
  └─ 근사 차수       → 컴파일러 자동 결정
```

> named op을 PIM Common에서 유지하는 이유(메모리 설계 §6 재확인): AiMX가 matmul을 descriptor chain으로,
> elementwise를 opcode로 매핑하려면 named/분류된 형태여야 함. scalar 루프로 풀면 AiMX 매핑이 깨짐. 푸는
> 것은 UPMEM target에서만.

---

## 6. 정리 표 — 연산 표현 & lowering

| 연산 범주 | PIM Common 표현 | UPMEM lowering | AiMX lowering |
|---|---|---|---|
| **matmul** | `linalg.matmul` 유지 | MAC 루프 | matmul descriptor chain |
| **reduce** | `linalg.reduce` 유지 | reduction 루프 | 지원 확인 필요, 아니면 분해 |
| **pointwise (legal body)** | `linalg.generic` + 태그 | arith/math 루프 | opcode descriptor (add/sub/mul) |
| **pointwise (illegal body)** | `linalg.generic` 유지 | (대부분 legal) | **분해** (다항식 근사 등) 또는 host |
| **broadcast** | `linalg.generic`(broadcast map) | 루프 | bias descriptor (elewise.c) |
| **scan / gather 등** | linalg 유지 | 루프 | 분해 또는 host |
| **레이아웃 (reshape 등)** | `tensor.*` 유지 | 주소 재매핑 (no-op) | 주소 재매핑 (no-op) |

---

## 7. 핵심 설계 결정 요약

| 항목 | 결정 |
|---|---|
| 새 dialect | **최소화** — linalg 최대 재사용, `pim.matmul` 안 만듦 |
| 구조화 연산(matmul/reduce) | named op 유지, scalar 루프로 안 풂 |
| 자유형 연산(generic) | body 분류 태그 (pointwise/broadcast/scan) |
| 연산-메모리 결합 | 의존성으로 분리, chain 결합은 target 유예 |
| 분해(legalization) | target에서 (legal set이 target 의존), rewrite pattern 라이브러리 |
| 근사 정확도 | 컴파일러 자동 결정 |
| PIM Common 연산 최적화 | op 사이 구조만 (fusion/load-compute 융합/reduction 결합/DCE) |
| body 내부 | 손대지 않음 (산술=upstream, 분해/근사=target) |
| 메모리와의 대칭 | 둘 다 op "사이" 최적화. 융합이 연산·메모리를 연결 |

---

## 8. 추후 결정할 열린 설계점

- **AiMX reduce legality** — matmul.c/elewise.c엔 matmul·elementwise만. AiMX가 reduction(합/최대)을
  descriptor로 지원하는지 확인. 지원 안 하면 reduce도 분해 대상(합 → 반복 add).
- **근사 차수 자동 결정 방법** — 고정 차수 vs autotuning 탐색 vs 정확도 목표 기반
- **host fallback 경계** — 어떤 연산까지 분해하고 어디서 host로 뺄지 판단 기준
- **fusion 범위** — 어디까지 융합할지 (elementwise 체인 / load-compute / matmul-epilogue 등)
- **`linalg.generic` body 분류 태그의 정확한 형태** — 어떤 메타데이터로 pointwise/broadcast/scan 구분
- **분해 pattern 라이브러리 구조** — target별 rewrite pattern 등록/관리 방식
