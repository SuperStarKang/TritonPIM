# PIM Common — 메모리 접근 lowering 설계

> TritonPIM의 PIM Common 중간층 설계 중 **메모리 접근** 부분 정리.
> 입력은 `--triton-to-linalg-experimental` 출력(memref+linalg), 목표는 target 무관한 PIM Common IR로
> raising하면서 불필요한 버퍼를 제거하는 것.
>
> 관련 문서:
> - 메모리 접근 패턴 전수 → `triton-shared-memory-access-patterns.md` (이하 "메모리 패턴 문서")
> - 연산 패턴 전수 → `triton-shared-compute-patterns.md`

---

## 0. 전체 목표와 위치

- **목표**: PIM Common은 target(UPMEM / AiMX) 무관한 중간층. **"PIM Common만 있으면 어떤 target으로든
  내려갈 수 있다."**
- **2단계 최적화 구조**: PIM Common에서 target 무관 최적화(버퍼 제거) → target IR에서 물리 최적화
  (인터리빙, HW 타일).
- **핵심 원칙**: PIM Common은 **논리적 접근 명세만** 담고, 물리적인 것(하드웨어 배치)은 담지 않는다.

### 전체 파이프라인 (메모리 부분 위치)

```
autotuner (target별 config 탐색, 바깥 루프)
   │ 각 config로 반복
   ▼
커널 로직(target 무관) + config(target별: block 크기 등)
   ▼
ttshared IR ──────────────── Triton 구조화 (memref+linalg, program_id=func arg)
   ▼  [Stage A/B/C  ← 이 문서]
PIM Common IR ────────────── 버퍼 제거, 논리 접근, 유닛별 논리 범위 분석 (target 무관)
   ▼  [target lowering]
target IR ────────────────── 물리 배치(인터리빙/padding), HW 타일, 유닛 매핑
   ├→ host codegen ────────── 유닛 분배 + 실행 호출 루프
   └→ device codegen ──────── UPMEM: DPU 프로그램 / AiMX: descriptor 생성기
   ▼
execute
```

---

## 1. 핵심 문제: 삼중구조와 불필요한 버퍼

메모리 패턴 문서에서 확인한 사실: 모든 structured 접근은 **descriptor + 암묵버퍼 + transfer**
삼중구조다.

```mlir
%view = memref.reinterpret_cast %arg0 to offset: [%o], sizes: [4, 256], strides: [1, 5] : ...
%buf  = memref.alloc() : memref<4x256xbf16>          // ← 낭비 원흉
memref.copy %view, %buf                              // opaque, 최적화 불가
%val  = bufferization.to_tensor %buf restrict writable
```

문제점 (메모리 패턴 문서 §6):
- **load마다 private 버퍼를 eager하게 생성**(value semantics 위해). read-only fast path 없음.
- 이 버퍼들이 **downstream에서 정리되지 않음** — `one-shot-bufferize`는 이미 memref인 `copy`를 못 지우고,
  `--eliminate-empty-tensors`는 비활성화되어 있음.
- 대부분은 필요한 staging DMA지만, 진짜 낭비 5케이스 존재: passthrough 이중복사, 반복 load 비dedup,
  masked over-alloc, 버퍼 미재사용, wrap full materialization.

---

## 2. PIM Common 메모리 op set

`reinterpret_cast → copy` 삼중구조를 **descriptor + transfer** 둘로 접고 버퍼를 없앤다.

```mlir
// addressing (reinterpret_cast 대응, 무비용 논리 descriptor)
%d = pim.access %base offset [%o] sizes [4, 256] strides [1, 5]
       : memref<*xbf16, #pim.host> -> !pim.access<4x256xbf16>

// transfer (alloc+copy+to_tensor를 접음, 버퍼 암묵)
%v = pim.load  %d           : !pim.access<4x256xbf16> -> tensor<4x256xbf16>   // Read effect
     pim.store %v2, %d      : tensor<4x256xbf16>, !pim.access<4x256xbf16>     // Write effect
     pim.copy  %ds, %dd     : !pim.access<...>, !pim.access<...>              // access→access 직접
%v = pim.gather  %base index %idx pad %p : ..., tensor<Kxi32> -> tensor<KxT>  // 완전 random
     pim.scatter %v, %base index %idx    : tensor<KxT>, ..., tensor<Kxi32>
```

| PIM Common op | 대응 memref | 역할 |
|---|---|---|
| `pim.access` | `memref.reinterpret_cast` | 논리 주소 descriptor (무비용) |
| `pim.load` | `alloc`+`copy`+`to_tensor` **접음** | host→local transfer, 값 생성(버퍼 암묵), Read |
| `pim.store` | `materialize_in_destination` | local→host transfer, Write |
| `pim.copy` | (passthrough 최적화 산출물) | descriptor→descriptor 직접 전송 |
| `pim.gather` / `pim.scatter` | UnstructuredToMemref | 원소별 random 접근 |

**핵심 효과 3가지** (raising 시 동시 발생):
1. **`alloc` 소멸** — `pim.load`가 값(tensor)을 직접 생성 → staging 버퍼가 IR에서 사라짐. 물리 할당은 target 유예.
2. **effect 명시화** — load=Read, store=Write. opaque `copy`가 막았던 alias 분석이 열림 → 표준 최적화 작동.
3. **aliasing 구성상 해결** — load 결과는 메모리와 분리된 immutable 값.

---

## 3. `pim.access` 필드 (논리적 접근 명세)

> 상세 타입/verifier 구현은 추후. 여기서는 설계 확정 사항만.

### 담는 것 / 안 담는 것

**담는다 (논리적, target 무관)**: base 텐서, 축별 offset/size/stride, wrap 경계(④), valid 길이(⑤),
index 텐서(⑥⑦), program id 의존성(offset이 pid의 함수).

**안 담는다 (물리적, target에서)**: 8채널 인터리빙, padding, HW 타일 폭(8×16×16), 로컬 버퍼 물리 주소,
batch 분해, 유닛 매핑.

### 필드 정의

```mlir
%d = pim.access %base
       offset  [%o0, %o1, ...]        // 축별 시작 위치 (static 또는 SSA/pid 함수)
       sizes   [S0, S1, ...]          // 타일 shape (항상 static)
       strides [%st0, %st1, ...]      // 축별 간격 (static 또는 SSA, broadcast=0)
       { 선택적 clause }
       : memref<*xT, #pim.host> -> !pim.access<S0xS1x...xT>
```

| 필드 | 필수? | 타입 | 활성 범주 | 의미 |
|---|---|---|---|---|
| `base` | 필수 | `memref<*xT, #pim.host>` | 전부 | 원본 host 버퍼 |
| `offset` | 필수 | `Variadic<Index>` | 전부 | 축별 시작 (pid/iter_arg SSA 허용) |
| `sizes` | 필수 | `DenseI64ArrayAttr` (static) | 전부 | 타일 shape |
| `strides` | 필수 | mixed static/`Index` | 전부 | 축별 간격 (broadcast=0) |
| `order` | 선택 | `DenseI32ArrayAttr` | ③ block ptr | 축 우선순위 (논리적) |
| `wrap` | 선택 | `Variadic<Index>` (0=없음) | ④ modulo | 순환 경계 |
| `valid` + `pad` | 선택 | `Variadic<Index>` + value | ⑤ masked | 축별 유효 길이 + 채움값 |
| `gather` | 선택 | `dim: I64` + `index: tensor` | ⑥ row-gather | 인덱스 축 + 오프셋 텐서 |

### 설계 결정 근거

- **결정 1 — descriptor(`!pim.access`)를 값으로 분리** (inline 안 함): load/store가 같은 region을 가리킬 때
  하나의 `%d`를 공유 → store→load forwarding, CSE 용이. (§5 최적화가 "동일 descriptor 판별"에 의존.)
- **결정 2 — `sizes`만 static, offset/stride는 dynamic 허용**: 타일 shape는 컴파일타임 상수(block config),
  offset은 pid 함수라 런타임 SSA 필요.
- **결정 3 — offset이 pid를 SSA로 참조** (특수 필드 없음): pid 의존성을 별도 필드로 두지 않고 offset을
  `%pid_m * BM * K` 같은 affine 식으로. → 유닛별 범위 분석이 **표준 affine 분석**으로 됨.
- **결정 4 — 물리적인 것 배제**: `order`조차 논리적 "축 우선순위"만 담고 실제 인터리빙은 target 결정.
  padding/HW 타일 폭은 아예 없음. (matmul.c의 `padded_q_size`, 8채널 인터리빙은 target lowering이 도입.)

### verifier 규칙 (조합 유효성)

- **R1 rank 일치**: `offset`/`sizes`/`strides`/(`wrap`/`valid`) 길이 = base 논리 rank.
- **R2 gather 배타성**: `gather`와 `wrap` 동시 사용 불가 (순환 vs 임의 인덱스 충돌). `gather`+`valid`는 허용.
- **R3 gather dim 범위**: `gather.dim < rank`, `index` shape가 그 축 size와 정합.
- **R4 valid ≤ sizes**: 각 축 `valid[i] ≤ sizes[i]`. `valid` 있으면 `pad` 필수.
- **R5 order 전용성**: `order`는 `wrap`/`gather`와 함께 안 씀. 있으면 rank 일치.
- **R6 stride 0 허용**: broadcast 축은 `stride=0` 가능. 단 `gather.dim`에는 금지.
- **R7 copy 타입 일치**: 두 access의 element type/shape 일치 (dtype 변환은 별도).

---

## 4. 패턴별 raising 매핑 (7범주)

각 범주가 단일 descriptor-carrying op로 접히면서 explicit 버퍼·루프·split·fill이 사라지고, 실현 방식은
target 유예.

| 범주 | BEFORE (memref 클러스터) | LOWERING | AFTER (PIM Common) | 제거되는 버퍼/연산 |
|---|---|---|---|---|
| ① strided | `reinterpret_cast`+`alloc`+`copy`+`to_tensor` | descriptor/transfer 분리, 버퍼 암묵화 | `pim.access` + `pim.load` | **alloc, opaque copy** |
| ② loop-carried | scf.for + (`reinterpret_cast`+`alloc`+`copy`)×iter | 루프 보존, 버퍼만 암묵화 | scf.for + `pim.load`(offset=iter_arg) | **iter별 alloc** (루프 유지) |
| ③ block ptr | `reinterpret_cast`(order)+`alloc`+`copy` | order를 clause로 유예 | `pim.access {order}` + `pim.load` | **alloc** |
| ④ wrap | **2×**(`reinterpret_cast`+`copy`)+`subview`×2+`dim`+`alloc` | split을 `wrap` clause로 유예 | `pim.access {wrap}` + `pim.load` **1개** | **copy×2, subview×2, dim, alloc** |
| ⑤ masked | `minsi`+`subview`+`fill`+`copy`+`scf.if`+`alloc` | 경계를 `valid`+`pad`로 유예 | `pim.access {valid,pad}` + `pim.load` | **fill, subview, scf.if, full-alloc** |
| ⑥ row-gather | scf.for+`extract`+행별`reinterpret_cast`+`copy` | 1축 index를 `gather` clause로 | `pim.access {gather}` + `pim.load` | **loop, 행별 alloc** |
| ⑦ element | scf.for+`linalg.generic`+원소별`extract`/`store` | 별도 gather/scatter op | `pim.gather` / `pim.scatter` | **loop, generic, per-elem extract** |
| store | `reinterpret_cast`+`materialize_in_destination` | descriptor/transfer 분리 | `pim.access` + `pim.store` | (원래 중간버퍼 없음) |

**⑥/⑦ 분리 근거**: ⑥은 "한 축만 인덱스, 나머지 contiguous"라 descriptor(`pim.access`+gather clause)로 표현되고
`pim.load`로 읽음. ⑦은 "모든 원소 임의 위치"라 descriptor 표현 불가 → 별도 op. 원본에서도 ⑥은
StructuredToMemref, ⑦은 UnstructuredToMemref로 갈렸던 것과 일치.

### 대표 예시 — GEMM의 A 로드 (① + ②)

```mlir
scf.for %k = %c0 to %K step %BK iter_args(%acc = %init, %ko = %c0) -> (...) {
  %dA = pim.access %A offset [%pid_m * %BM * %K, %ko] sizes [%BM, %BK] strides [%K, 1]
          : memref<*xf32, #pim.host> -> !pim.access<?x?xf32>
  %a  = pim.load %dA : !pim.access<?x?xf32> -> tensor<?x?xf32>
  // ... linalg.matmul(%a, %b) ...
  %nko = arith.addi %ko, %BK : index
  scf.yield %nacc, %nko : ...
}
```

`offset [%pid_m * %BM * %K, %ko]` — 0축은 pid 함수(유닛 분배 분석 대상), 1축은 iter_arg(스트리밍).
하나의 형태로 유닛 분배 정보와 루프 스트리밍이 동시 표현.

---

## 5. 버퍼 제거 최적화 (값 의미론 위 rewrite)

effect 모델링(load=Read, store=Write) 덕분에 값 의미론 위에서 표준 rewrite가 작동. 이것이 메모리 패턴
문서 §6.4의 낭비를 제거한다.

| 최적화 | rewrite | gate (안전조건) | 없애는 낭비 |
|---|---|---|---|
| passthrough elision | `pim.store(%dd, pim.load(%ds))` → `pim.copy(%ds, %dd)` | 값 store외 미사용 + 사이 region write 없음 | #1 이중복사 |
| load CSE/dedup | 동일 descriptor `pim.load` → 값 공유 | 사이 region write 없음 | #2 반복 load |
| LICM | loop-invariant `pim.load` → 루프 밖 hoist | descriptor invariant + 루프 내 write 없음 | #2 루프 내 중복 |
| store→load forwarding | `pim.store(%d,%v); … pim.load(%d)` → `%v` 전달 | 사이 동일 region write 없음 | RAW 왕복 |
| dead-transfer elim | 결과 미사용 `pim.load` → DCE | (Read-only) | 죽은 전송 |
| masked over-alloc 제거 | `valid`가 descriptor clause → full-alloc+fill 불필요 | (구성상) | #3 over-alloc |
| buffer 재사용/풀링 | *(Stage 2 = target으로 유예)* | scratchpad 용량 aware | #4 미재사용 (target 관심사) |

**핵심**: raising(§4)이 **버퍼를 구조적으로 제거**하고, rewrite(§5)가 **값 의미론 위에서 중복 전송 제거**.
#4(버퍼 재사용/풀링)만 물리 용량 의존이라 target으로 유예.

---

## 6. lowering 파이프라인 (3 stage)

```
--triton-to-linalg-experimental 출력 (memref + linalg + scf, program_id=func arg)
  │
  ├─ Stage A: 메모리 raising
  │    memref.reinterpret_cast+alloc+copy+to_tensor  →  pim.access + pim.load
  │    materialize_in_destination                     →  pim.store
  │    (7범주 → pim.access clause, §4)
  │
  ├─ Stage B: 버퍼 제거 rewrite (§5)
  │    passthrough elision / load CSE / LICM / store→load fwd / dead-transfer DCE
  │
  └─ Stage C: 연산 정규화 (연산 문서 참조)
       linalg.matmul / linalg.reduce → named op 유지 (scalar 루프로 풀지 않음)
       linalg.generic → body/indexing_map 분류
  │
PIM Common IR: SPMD(program_id symbolic) + scf.for(작성자 타일) + pim.access/load/store + named linalg, 버퍼 clean
```

> Stage C에서 named op을 유지하는 이유: AiMX가 matmul을 descriptor chain으로, elementwise를 opcode로
> 매핑하려면 named/분류된 형태여야 함. scalar 루프로 풀면 AiMX 매핑이 깨짐. 푸는 것은 UPMEM target에서만.

---

## 7. 실행 모델 & 분배 (메모리 관점)

### SPMD 구조 (두 target 공통)

두 target 모두 **분배 레벨에서 SPMD** (program_id → 물리 유닛). matmul.c / elewise.c의 `fpga_id`가
이를 확증.

| 레벨 | UPMEM | AiMX |
|---|---|---|
| inter-unit 분배 | grid → DPU (program_id → DPU) | grid → FPGA (program_id → fpga_id) |
| intra-unit 구조 | tasklet (나눠서 병렬화 가능) | 8ch×16bank×16alu (**나눌 수 없음, 통째 실행**) |
| 한 program 의미 | DPU가 **실행**하는 명령 시퀀스 | 호스트가 **생성**하는 descriptor chain (imperative 루프) |

> AiMX는 채널 개별 제어 불가 → FPGA 하나가 **쪼갤 수 없는 최소 실행 단위**. 유닛을 검은 상자로 보고
> "데이터 넣고 → 통째 실행 → 결과 받기". UPMEM만 유닛 내부(tasklet)를 추가로 다룸.

### 3-level tiling 분리

| tiling 레벨 | 무엇 | 어디서 결정 |
|---|---|---|
| ① Grid (inter-unit) | program_id → 물리 유닛 | 분배 (host+target) |
| ② Author block (intra-unit) | 작성자 타일/루프 (BLOCK_M/N/K, K-loop) | **PIM Common** (Triton에서 보존) |
| ③ Hardware (intra-unit) | HW 구조 세분 (8×16×16, batch, padding) | **target lowering** (도입) |

**③은 PIM Common에 없다.** matmul.c의 p/q/r 루프·batch 분해, elewise.c의 `r += ALU*BANK*CH`는 전부
하드웨어 타일링이라 target이 도입. PIM Common은 ②(작성자 타일, program_id symbolic)까지만.

### block → grid 인과 & 타일링 결정

**인과 방향: block이 원인, grid가 결과** (grid → block 아님).

```
block 크기 결정 (사용자/autotuning)
  → grid 개수 = ceil(문제크기 / block)        ← grid는 여기서 파생
  → 각 program id의 offset = f(program_id, block)   예: offset = pid * BLOCK_M
```

예: M=1000, BLOCK_M=64 → grid = ceil(1000/64) = 16 (program id 0~15). program id 3 → offset = 192
(192~255행 담당). 분배는 grid를 "쪼개는" 게 아니라, **이미 나뉜 grid 조각을 물리 유닛에 나눠주는** 것.

**타일링에는 성격이 다른 두 결정이 섞여 있다:**

| 결정 | 무엇 | 성격 | 결정하는 것 |
|---|---|---|---|
| **1. 용량** | block이 유닛 로컬 메모리에 들어가는가 | **하드 제약** (안 맞으면 실행 불가) | block 크기 **상한** → grid 개수 하한 |
| **2. 분배** | 유닛이 몇 block을 가져가 처리하는가 | **소프트 최적화** (어떻게든 돌지만 균등해야 빠름) | grid → 유닛 매핑 |

관계: 결정 1이 "block을 얼마나 크게 할 수 있나"(→ grid 최소 개수)를, 결정 2가 "그 grid를 유닛들이 어떻게
나눠갖나"를 정한다.

**PIM Common은 두 결정 모두 고려하지 않는다** (이유가 다름):

- **결정 1(용량)**: 유닛 로컬 메모리 크기가 target마다 다름(WRAM 64KB vs AiMX 상이). PIM Common은 target을
  모르니 판단 불가. block 크기는 autotuning이 target 보고 정해서 **이미 상수로** `sizes`·loop step에 박힌 채
  도착 → 용량 검증은 **PIM Common보다 앞(autotuning)** 에서 끝난 상태.
- **결정 2(분배)**: 유닛 개수가 target마다 다름(2048 DPU vs 8 FPGA). grid→유닛 매핑은 target 관심사.
  PIM Common은 program_id를 symbolic하게만 두고 관여 안 함.

**PIM Common이 받는 것 / 안 받는 것:**

| | 내용 |
|---|---|
| **받는 것** | block 크기(`sizes`·step에 상수), program_id의 offset 사용(`%pid*BM`), num_programs(함수 인자) |
| **안 받는 것** | 유닛 로컬 메모리 크기(용량), 유닛 개수, grid→유닛 매핑 |

즉 PIM Common에서 block은 **"왜 이 크기인지 모르지만 주어진 타일 크기"**. 유닛 메모리에 맞아서인지
autotuning이 골라서인지 신경 쓰지 않고, "한 program_id가 이 크기 타일을 이 offset에서 처리한다"만 표현.

> 이 분리가 "PIM Common = target 무관"의 핵심. 용량을 고려하면 "WRAM 64KB 가정"이, 유닛 개수를 고려하면
> "2048 유닛 가정"이 IR에 새어들어 다른 target에서 무효가 된다. 그래서 용량은 앞단(autotuning), 유닛 개수는
> 뒷단(target IR)으로 밀어내고, PIM Common은 그 사이 target 무관 논리만 담는다.

### 유닛 사이 분배 (3단계)

Triton 커널이 주는 정보: program id 범위(`num_programs`, 함수 인자) + program id→데이터 범위
(`pim.access` offset 식).

1. **각 유닛 논리 데이터 범위**: offset 식에 pid 값 대입 → 유닛별 범위. **target 무관** (affine 관계).
   → PIM Common에서 공용 분석.
2. **id → 유닛 매핑**: grid ≤ 유닛이면 1:1, grid > 유닛이면 유닛당 반복. 정책은 target 결정.
3. **host 데이터 전송**: 1+2 합쳐 "u번 유닛에 어느 슬라이스". 입력 A/B는 유닛 간 겹치면 복제, 출력 C는
   유닛마다 disjoint라 충돌 없음.

### 논리 범위(공용) vs 물리 재작성(target)

offset 처리는 두 부분으로 갈림:

| offset으로 하는 일 | target 무관? | 어디서 |
|---|---|---|
| **범위 읽기** ("유닛당 몇 행 필요?") | 무관 (수학적 관계) | PIM Common (공용) |
| **주소 재작성** (물리 주소: 인터리빙/padding/HW 타일) | **target마다 완전히 다름** | target |

matmul.c의 "Layout input matrix" 단계(8채널 인터리빙 재배치, `padded_q_size`)가 물리 재작성의 예.
사용자는 논리적 형태만 작성하고, layout 물리 변환은 **컴파일러가 target에서 자동으로** (사용자 명시 X).
→ PIM Common엔 인터리빙/padding이 없어야 UPMEM과 AiMX가 공유 가능.

---

## 8. Portability 수준 (정직한 범위)

- **커널 로직(program id, offset, loop 구조)**: target 무관하게 재사용.
- **config 값(block 크기 등)**: target에 묶임 → 직접 지정 또는 autotuning으로 target별 탐색.
- 즉 **"커널 로직 하나 + target별 튜닝 값"**. 완전 자동("커널 하나로 아무 하드웨어나")은 아니며,
  이는 Triton의 autotune과 동일한 현실적 portability 수준.

---

## 9. 확정 사항 요약

| 항목 | 결정 |
|---|---|
| 실행 모델 | SPMD (program_id → 유닛), 두 target 공통. imperative 루프 보존 |
| PIM Common op | `pim.access`(descriptor) + `pim.load`/`store`/`copy`/`gather`/`scatter`(transfer) |
| 버퍼 | `pim.load`가 값 생성 → staging 버퍼 암묵화, 물리 할당은 target 유예 |
| `pim.access` | 논리 필드만 (offset/sizes/strides + wrap/valid/gather/order clause), 물리 배제 |
| 버퍼 최적화 | 값 의미론 + effect 위 표준 rewrite (§5) |
| tiling | ②(작성자)까지 PIM Common, ①③은 target |
| 유닛 분배 | 논리 범위 분석은 PIM Common 공용, 물리 재작성은 target |
| layout 변환 | target 자동 (사용자 명시 X, PIM Common엔 없음) |

## 10. 추후 결정할 열린 설계점

- `#pim.host` vs 로컬 메모리(`!pim.local`) 표현 — 버퍼를 얼마나 일찍 명시할지
- `pim.copy`를 사용자 표현으로도 허용할지, 최적화 산출물로만 둘지
- element type 변환(load 중 dtype 캐스팅)을 access에 담을지 별도 연산으로 뺄지
- 유닛별 논리 범위 분석 방법 1(PIM Common에서 태그) vs 방법 2(target에서, 물리 재작성에 흡수)의 최종 선택
  — 현재는 "논리 범위=공용 + 물리 재작성=target" 혼합으로 기움
- host 층 생성 방식 (컴파일러 완전 생성 vs template + 정보 주입)
- `pim.access` 정확한 타입 시그니처 & verifier 구현
