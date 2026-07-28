# PIM IR 구현

Triton 커널을 PIM 하드웨어에서 실행하기 위한 중간 계층의 현재 구현을 기술한다.
`PIMCommon_Design.md`가 무엇을 만들지에 대한 설계라면, 이 문서는 무엇이 만들어져
있고 각 부분이 왜 그 모양인지를 기술한다. 둘이 어긋나는 곳은 §11에 모아 두었다.

용어는 이렇게 쓴다.

| 용어 | 뜻 |
| --- | --- |
| 반복 공간(iteration space) | grid 전체가 도는 인덱스 범위 |
| 인스턴스(instance) | grid 좌표 하나에 해당하는 프로그램 실행 하나 |
| 유닛(unit) | 물리 연산기 하나. UPMEM에서는 DPU |
| 타일(tile) | 인스턴스 하나가 한 번에 다루는 데이터 조각 |
| 부분 행렬(submatrix) | 유닛 하나가 받는 데이터 영역 |
| 복제(replication) | 같은 데이터가 여러 유닛에 들어가는 것 |
| 결합(combine) | 여러 유닛의 부분 결과를 합치는 것 |

---

## 1. 이 계층이 존재하는 이유

Triton은 커널 하나를 컴파일하고, 그것을 grid 전체에 뿌려 실행한다. GPU에서는
하드웨어 스케줄러가 thread block을 SM에 배정한다. PIM에서는 그럴 수 없다.
**데이터를 미리 유닛 메모리에 넣어야 하기 때문에**, 어느 인스턴스가 어느 유닛에서
도는지를 컴파일러와 host 코드가 정해야 한다.

그 결정에 필요한 것은 세 가지다.

1. 각 인스턴스가 원본 버퍼의 **어디를** 읽고 쓰는가
2. 인스턴스들 사이에 **같은 데이터를 읽는** 관계가 있는가
3. 여러 인스턴스가 **같은 자리에 쓰면** 결과를 합쳐야 하는가

셋 다 target과 무관하다. UPMEM이든 다른 PIM이든 똑같이 필요하다. 그래서 target
IR보다 앞에 한 계층을 두고, 여기서 답을 낸 뒤 target이 그 답을 자기 메모리 계층과
명령 집합에 맞게 실현한다.

경계선은 이렇게 긋는다.

> **PIM IR은 "어느 데이터가 어느 유닛에 몇 번 가는가"까지, Target IR은 "그 데이터가
> 유닛 안에서 어떤 계층을 어떤 단위로 오가는가"부터.**

---

## 2. 전체 흐름

### 2.1 컴파일

각 단계가 **무엇을 결정하는지**를 함께 적는다. 그게 층을 나눈 이유다.

```
@triton.jit 커널  +  BLOCK_*  +  PIM 옵션
   │
   ├ [1] Triton frontend                        결정: 없음 (언어 → IR)
   ▼
TTIR                            "한 인스턴스가 하는 일" (포인터 연산)
   │
   ├ [2] --triton-to-linalg-experimental        결정: 포인터 → memref, tl.load → memref 접근
   ▼
TTShared IR                     "한 인스턴스가 하는 일" (memref + linalg)
   │
   ├ [3] --convert-ttshared-to-pim              결정: 접근 패턴을 descriptor로 명시
   ├     --pim-build-grid                       결정: program_id를 반복 공간 loop로
   ├     -cse / -loop-invariant-code-motion
   ├     --pim-remove-redundant-transfers       결정: 중복 전송 제거
   ├     -canonicalize
   ├     --pim-normalize-accumulator            결정: 누산 형태 정규화
   ├     --pim-analyze-reuse                    결정: 축별 의존, 쪼갤 수 있는 축
   ├     --pim-analyze-transfer                 결정: 전송 크기, 유닛당 상주량
   ▼
PIM IR (인스턴스 형태)           "모든 인스턴스가 하는 일" — 유닛 개념 없음
   │
   ├ [4] --pim-partition                        결정: ★ 인스턴스 → 유닛 배정
   │                                                  각 유닛의 부분 행렬
   │                                                  어느 부분 행렬이 복제되나
   ├     -cse / -canonicalize
   ├     --pim-classify-compute                 결정: 없음 (태그)
   ▼
PIM IR (유닛 형태)               "유닛 하나가 하는 일 + 유닛 경계의 데이터 이동"
   │
   ├ [5] --convert-pim-to-<target>              결정: 메모리 계층, DMA 단위,
   │                                                  유닛 내부 병렬, 레이아웃 변환
   ▼
Target IR                        "실제로 실행될 코드의 구조"
   │
   ├ [6] translate                              결정: 없음 (IR → 소스, 1:1)
   ▼
host 소스 + device 소스
```

[5]와 [6]은 아직 없다. [6]에서 결정이 없어야 한다는 것이 "템플릿을 쓰지 않는다"의
정확한 의미다. Target IR을 보면 나올 코드가 정해져 있어야 한다.

### 2.2 런타임

```
Python:  matmul_kernel[grid](A, B, C, M, N, K, BLOCK_M=32, ...,
                             target="upmem", split_m=4, split_n=32, split_k=16)
   │
   ├ 캐시 키 계산 — 시그니처 + constexpr + PIM 옵션 (grid는 안 들어감)
   ├ 캐시 미스면 위 컴파일 흐름 전체 실행
   ├ grid 정규화 → (g_x, g_y, g_z)
   ▼
Launcher:  host_fn(A_ptr, B_ptr, C_ptr, M, N, K, g_x, g_y, g_z)
   ▼
생성된 host 코드:
   ├ 유닛 확보 + device 프로그램 적재
   ├ 유닛 좌표 계산: u → (pm, pn, kj)
   ├ 각 유닛의 부분 행렬 계산 (grid와 SPLIT로부터)
   ├ 복제 처리: 서로 다른 조각만 준비하고, 전송은 유닛 수만큼
   ├ 스칼라 브로드캐스트
   ├ 전체 launch
   ├ 부분 결과 회수
   └ 결합 축 방향으로 합치기
```

**grid 값은 컴파일 캐시 키에 들어가지 않는다.** `triton/python/triton/runtime/jit.py`
에서 캐시 조회가 grid 정규화보다 먼저 일어난다. 그러므로 생성되는 host 코드는 grid
값이 아니라 **grid로부터 배정을 계산하는 식**을 담아야 한다.

```c
/* 이렇게 (grid를 런타임 인자로 받음) */
uint32_t tiles_x = (g_x + SPLIT_M - 1) / SPLIT_M;
uint32_t rows_per_unit = tiles_x * BLOCK_M;

/* 이렇게 하면 안 됨 (컴파일 시점 상수) */
const uint32_t Mu = 256;
```

`BLOCK_M`과 `SPLIT_M`은 컴파일 시점 상수가 맞다(둘 다 캐시 키에 들어간다).
`g_x`는 런타임 값이다.

---

## 3. 핵심 결정 — grid는 반복 공간, 유닛 수는 별개

이 결정이 나머지 전부를 정한다.

> **grid는 "작업을 몇 개의 타일로 나눌지"이고, 유닛 수는 "기계를 몇 개로 나눌지"다.
> 서로 다른 수이고, 서로 독립이다.**

실제 설정으로 보자. `M=1024, N=8192, K=8192, BLOCK=(8,32,16), splits=(4,32,16)`.

| | 값 | 무엇이 정하나 |
| --- | --- | --- |
| grid | `(⌈1024/8⌉, ⌈8192/32⌉)` = **(128, 256)** | 문제 크기 ÷ BLOCK. 사용자가 Python에서 쓴다 |
| 인스턴스 수 | 128 × 256 = **32768** | 위에서 자동 |
| 유닛 수 | 4 × 32 × 16 = **2048** | 하드웨어에 있는 DPU 개수 |

32768 ≠ 2048이므로 유닛 하나가 인스턴스 16개를 맡는다. 버리는 인스턴스는 없다.
GPU에서 thread block이 SM보다 많을 때와 같은 일이고, 다른 점은 배정을 하드웨어가
아니라 컴파일러가 정한다는 것뿐이다.

### 3.1 대응 규칙

유닛에 좌표 `(pm, pn, kj)`를 붙인다. 각각 `[0,4)`, `[0,32)`, `[0,16)`.

```
유닛 (pm, pn, kj)가 맡는 인스턴스:
    gx ∈ [pm·32, pm·32 + 32)        128개 타일을 4 유닛이 32개씩
    gy ∈ [pn·8,  pn·8 + 8)          256개 타일을 32 유닛이 8개씩
그 인스턴스들 안에서 맡는 K 범위:
    k  ∈ [kj·512, kj·512 + 512)     K=8192를 16 유닛이 512씩
```

### 3.2 병렬 축과 축약 축은 성격이 다르다

| | 무엇을 나누나 | 결합 필요? |
| --- | --- | --- |
| `split_m`, `split_n` | **반복 공간**(grid)을 나눔 | 없음 — 유닛마다 출력의 다른 부분을 쓴다 |
| `split_k` | **인스턴스 안의 loop**를 나눔 | **있음** — 여러 유닛이 같은 출력에 부분합을 쓴다 |

IR이 이 구분을 갖고 있다.

```mlir
pim.split_axes = [{axis = "x", source = "grid"},
                  {axis = "y", source = "grid"},
                  {axis = "k", source = "reduction", combine = "add"}]
```

`source`가 `"grid"`면 결합 없음, `"reduction"`이면 `combine`으로 결합. 이 속성이
host 코드의 마지막 단계를 결정한다.

### 3.3 왜 M, N, K에서 직접 나누지 않는가

"grid를 안 보고 `Mu = M / split_m`으로 나누면 되지 않나"가 자연스러운 대안인데,
두 가지가 깨진다.

**첫째, 나누어떨어짐 조건이 두 겹이 된다.**

```
M에서 나눌 때:    M % split_m == 0  이고  (M / split_m) % BLOCK_M == 0
grid에서 나눌 때:  없음 (⌈ ⌉로 처리)
```

`M=1000, BLOCK_M=32, split_m=4`이면

- M에서: `1000/4 = 250`, `250 % 32 ≠ 0` → 실패
- grid에서: `g_x = ⌈1000/32⌉ = 32`, 유닛당 8타일 → 성공, 마지막 타일만 마스킹

**둘째, 마스킹 정보를 못 쓴다.** PIM IR에 경계 처리가 이미 계산되어 있다.

```mlir
%64 = pim.access %arg0 offset %arg14 sizes [32, 16] strides[%8, %c1]
      valid[%37, %63] pad %cst : f32
```

`%37`은 `min(M - m_off, 32)`, `%63`은 `min(K - k·16, 16)`이다. grid loop를 좁히면
이 계산이 그대로 따라오고, M/N/K에서 다시 나누면 이 계산과 무관해진다.

---

## 4. Dialect

`include/triton-shared/Dialect/PIM/IR/PIMDialect.td`. target 용어(DPU, MRAM,
tasklet 등)는 여기 등장하지 않는다.

### 4.1 `!pim.access` 와 `pim.access`

논리 주소 명세다. 데이터를 옮기지 않으므로 `Pure`다.

```mlir
%20 = pim.access %arg0 offset %19 sizes [8, 16] strides[%4, %c1] origin[%41, %50]
      : memref<*xi8> -> !pim.access
```

| 필드 | 뜻 |
| --- | --- |
| `base` | 원본 host 버퍼 (`memref<*xT>`) |
| `offset` | 선형 시작 위치 (원소 단위) |
| `sizes` | 타일 shape. Triton의 BLOCK이라 컴파일 시점 상수 |
| `strides` | 축별 간격. broadcast 축은 0 |
| `origin` | **축별** 타일 시작 위치 |
| `valid` (선택) | 축별 유효 범위. 경계 타일에서 실제로 읽는 크기 |
| `pad` (선택) | 유효 범위 밖을 채울 값 |
| `gather_dim` + `index` (선택) | 한 축을 index 텐서가 정하는 행 gather |

**`origin`이 왜 필요한가.** `offset`은 축을 합쳐 놓은 값 하나다.

```mlir
%18 = arith.index_cast %arg11 : i32 to index     // k
%19 = arith.addi %42, %18 : index                // %42 = (gx·8)·K
%20 = pim.access %arg0 offset %19 ...
```

"A 타일의 행 시작은 `gx·8`, 열 시작은 `k`"를 알려면 `addi → muli → index_cast →
grid loop`를 거슬러 올라가야 한다. 그리고 그 되짚기를 target legalize마다 다시
구현하게 된다. 축별로 두면 유닛의 loop 범위를 대입하는 것만으로 부분 행렬이 나온다.

항상 `offset = Σ_d origin[d] · strides[d]` 가 성립한다. 분해가 안 되는 경우
(§10.1)에는 `origin`을 통째로 비운다. 부분적으로 채우면 읽는 쪽이 나머지를 0으로
오해한다.

verifier가 확인하는 것:

| 규칙 | 내용 |
| --- | --- |
| R1 | `strides` 개수 == `sizes` rank |
| R2 | `valid`를 쓰면 개수 == rank |
| R2' | `origin`을 쓰면 개수 == rank |
| R3 | `pad`는 `valid`와 함께만 (저장은 `valid`만도 가능) |
| R4 | `gather_dim`과 `index`는 쌍이고, 축은 범위 안 |

### 4.2 전송 op

| op | 효과 | 뜻 |
| --- | --- | --- |
| `pim.load` | `MemRead` | access가 가리키는 영역을 tensor로 |
| `pim.store` | `MemWrite` | tensor를 access가 가리키는 영역으로 |
| `pim.copy` | `MemRead`+`MemWrite` | access → access 직접 전송 (로컬 버퍼 없이) |
| `pim.gather` | `MemRead` | 원소별 임의 위치 읽기 |
| `pim.scatter` | `MemWrite` | 원소별 임의 위치 쓰기 |

**descriptor와 전송을 나눈 이유**가 중요하다. faithful codegen에서 한 타일은
여러 메모리 계층을 거친다. UPMEM이면 host→MRAM, MRAM→WRAM 두 번이다. descriptor
하나에 전송 여러 개가 붙는 형태가 정확히 그것을 표현한다.

### 4.3 `pim.units`

유닛 하나가 하는 일을 담는 region이다. **밖이 host, 안이 device로 갈린다.**

```mlir
pim.units axes ["x", "y", "k"] counts [4, 32, 16] {
^bb0(%pm: index, %pn: index, %kj: index):
  ...
}
```

block argument가 이 유닛의 좌표다. 속성이 아니라 region인 이유가 있다.

| | 속성 (attribute) | 구조 (region) |
| --- | --- | --- |
| target이 무시하면 | 조용히 넘어감 | 컴파일 실패 |
| IR이 바뀌면 | 낡아도 모름 | 같이 바뀜 |
| 검증 | 없음 | verifier |

유닛 배정은 무시되면 안 되는 정보라 구조로 둔다.

### 4.4 `pim.unit_data`

유닛 하나가 자기 메모리에 갖고 있어야 하는 원본 버퍼의 영역이다.
**host의 전송 코드가 전부 여기서 나온다.**

```mlir
pim.unit_data %arg0 origin[%31, %33] sizes[%32, %34] strides[%2, %c1]
    {direction = "in", access = 0 : i64, replicated_over = ["y"]} : memref<*xi8>
```

`pim.access`와 대비하면 이렇다.

| | `pim.access` | `pim.unit_data` |
| --- | --- | --- |
| 단위 | 인스턴스 하나가 한 번에 다루는 타일 | 유닛 하나가 전체 실행 동안 쓰는 영역 |
| 예 (A) | 8 × 16 = 128 B | 256 × 512 = 128 KB |
| 만드는 곳 | `--convert-ttshared-to-pim` | `--pim-partition` |
| 읽는 쪽 | device 코드 | **host 코드** |

| 필드 | 뜻 | 어디서 나오나 |
| --- | --- | --- |
| `base` | 어느 커널 인자인가 | access의 base |
| `origin` / `sizes` / `strides` | 영역 | access의 origin + 유닛 loop 범위 |
| `direction` | `"in"` \| `"out"` | `pim.role` |
| `access` | 유도한 access의 `pim.access_id` | — |
| `replicated_over` | 이 축만 다른 유닛은 같은 데이터를 받는다 | `pim.invariant_grid_dims` |
| `combine` / `combine_over` | 이 축만 다른 유닛이 부분 결과를 쓴다 | `pim.split_axes` |

`access` 필드가 있어야 유닛 영역 안에서 타일이 어디인지를 되짚을 수 있다.
`pim.access.origin − pim.unit_data.origin`이 유닛 로컬 좌표다. 같은 버퍼를 여러
access가 쓰면 `base`만으로는 어느 것인지 정할 수 없으므로 id로 가리킨다.

**복제와 결합은 같은 사실의 양면이다.**

```
어떤 access가 쪼갠 축 a에 의존하지 않는다
  ├─ 입력이면 → a만 다른 유닛들이 같은 데이터를 받는다   → replicated_over
  └─ 출력이면 → a만 다른 유닛들이 같은 자리에 쓴다       → combine_over
```

---

## 5. Pass 상세

### 5.1 `--convert-ttshared-to-pim`

`--triton-to-linalg-experimental`이 낸 메모리 접근 3중 구조를 `pim.access` +
`pim.load`/`store`로 끌어올려 staging 버퍼를 없앤다.

**입력 (기본 로드)**

```mlir
%rc = memref.reinterpret_cast %A to offset: [%13], sizes: [8, 16], strides: [%6, 1]
%buf = memref.alloc() : memref<8x16xi8>
memref.copy %rc, %buf
%t = bufferization.to_tensor %buf restrict writable
```

**출력**

```mlir
%d = pim.access %A offset %13 sizes [8, 16] strides[%6, %c1] origin[%3, %12]
     : memref<*xi8> -> !pim.access
%t = pim.load %d : !pim.access -> tensor<8x16xi8>
```

지원하는 관용구는 8가지다.

| | 형태 | 결과 |
| --- | --- | --- |
| ① | 기본 로드 (`copy` + `to_tensor`) | `access` + `load` |
| ② | strided | 같음 (`strides`에 반영) |
| ③ | block | 같음 (`sizes` rank 2) |
| ⑤ | masked (`subview` × 2 + `copy`) | `access{valid, pad?}` + `load` |
| ⑥ | row gather (`scf.for{extract + rc + copy}`) | `access{gather_dim, index}` + `load` |
| ⑦ | element gather / scatter (masked 포함) | `gather` / `scatter` |
| — | tile 저장 (`materialize_in_destination`) | `store` |
| — | 스칼라 저장 (`affine.store`, 크기 1) | `from_elements` + `store` |

wrap-around 로드(`% M` 인덱스로 두 영역이 생기는 형태)는 하나의 `pim.access`로
표현할 수 없다. 조용히 두면 target에서 깨지므로 여기서 에러를 낸다.

**`origin` 분해.** `offset`을 축으로 되돌린다. `addi` 항을 펼친 뒤 항마다:

```
muli(x, strides[d])  → 축 d의 origin이 x
어느 축과도 안 맞음   → strides가 상수 1인 축(마지막 축)의 origin
```

한 축에 두 항이 오거나 축을 못 찾으면 빈 결과를 낸다.

실제 i8 GEMM에서:

```mlir
%13 = arith.addi %7, %12 : index          // %7 = %3 · %6,  %12 = index_cast(k)
                    strides = [%6, 1]
→ origin[0] = %3   (muli의 다른 쪽)
  origin[1] = %12  (stride 1 축)
```

이 분해는 **conversion 시점에 해야 한다.** `-cse`/`-loop-invariant-code-motion`이
지나간 뒤에는 주소 계산이 loop 밖으로 흩어져 훨씬 어렵다.

### 5.2 `--pim-build-grid`

runtime이 돌던 grid 반복을 IR 안의 `scf.for`로 옮긴다. 이것이 없으면 `pid_m`이
그냥 스칼라라, 유닛 분배가 IR 변환이 아니라 코드 생성기 안의 산수가 된다.

triton-shared ABI: 커널 인자 끝 6개가 `num_programs(x,y,z)` 다음
`program_id(x,y,z)`다.

**입력**

```mlir
func.func @k(%A: memref<*xi8>, ..., %arg6: i32, %arg7: i32, %arg8: i32,
                                   %arg9: i32, %arg10: i32, %arg11: i32) {
  %2 = arith.muli %arg9, %c8_i32 : i32    // pid_x · BLOCK_M
```

**출력**

```mlir
func.func @k(%A: memref<*xi8>, ..., %arg6: i32, %arg7: i32, %arg8: i32) {
  %0 = arith.index_cast %arg6 : i32 to index
  %1 = arith.index_cast %arg7 : i32 to index
  scf.for %gx = %c0 to %0 step %c1 {
    %7 = arith.index_cast %gx : index to i32
    %8 = arith.muli %7, %c8_i32 : i32
    scf.for %gy = %c0 to %1 step %c1 { ... } {pim.grid_dim = "y"}
  } {pim.grid_dim = "x"}
```

세부 결정 세 가지:

- **`program_id` 인자를 지운다.** 남겨 두면 인자 끝 3개를 program ID로 보는 뒤
  패스가 `num_programs`를 program ID로 오인한다. `num_programs`는 loop 상한으로
  쓰이므로 남긴다.
- **실제로 쓰이는 축만 loop로 만든다.** 안 쓰는 축은 범위가 1이라 loop가 필요 없다.
- **상한을 먼저 전부 만든다.** loop를 쌓으면서 만들면 안쪽 loop 상한의 cast가
  바깥 loop 안에 들어가 loop-invariant인 채로 남는다.

`pim.grid_dim` 속성이 grid loop와 인스턴스 내부 loop(K 타일 등)를 구분하며,
이후 분석·분할이 전부 이 구분 위에서 돈다.

### 5.3 `--pim-remove-redundant-transfers`

`load`(Read) / `store`(Write) / `access`(Pure) 효과 위에서 중복 전송을 없앤다.

**(a) hoisting.** loop 안에서 매번 같은 영역을 읽는 `pim.load`를 loop 앞으로
옮긴다. 조건 세 가지.

```
- access가 loop 밖에서 정의된다 (= 유도변수에 의존하지 않는다)
- loop 안에 그 버퍼를 쓰는 op이 없다
- loop 몸체 블록의 직속 자식이다
```

세 번째가 필요한 이유: `scf.if` 안의 load를 빼면 실행되지 않았을 읽기가 실행된다.

**grid loop에서는 빼지 않는다.** 인스턴스 사이에 공유되는 전송인지는 분할 구조가
정해진 뒤에야 알 수 있어서, 여기서 빼면 유닛마다 사본이 필요하다는 사실이 사라진다.

**(b) passthrough.** `store(load(%src), %dst)` → `copy(%src, %dst)`.

**(c) load 재사용.** 같은 access의 뒤 load를 앞 load 결과나 앞 store 값으로 대체한다.

순서가 중요하다. hoisting을 먼저 해야 loop 밖으로 나온 load들이 같은 블록에 모여
재사용 판정을 받는다.

**같은 access의 판별은 SSA 동일성**이다. 그래서 이 pass 앞에 `-cse`를 돌려 `Pure`한
`pim.access`를 합쳐 두는 것을, hoisting은 그 앞의 `-loop-invariant-code-motion`이
불변 access를 loop 밖으로 빼 둔 것을 전제한다.

`mayAlias`는 base memref가 다르면 겹치지 않는다고 본다. **이것은 가정이다** —
Triton은 커널 포인터 인자가 서로 다른 버퍼임을 보장하지 않는다.

### 5.4 `--pim-normalize-accumulator`

**입력**

```mlir
%mm = linalg.matmul ins(%a, %b) outs(%zero) -> tensor<8x32xi32>
%r  = linalg.generic ins(%acc, %mm) outs(%acc) { arith.addi } -> tensor<8x32xi32>
scf.yield %r
```

**출력**

```mlir
%mm = linalg.matmul ins(%a, %b) outs(%acc) -> tensor<8x32xi32>
scf.yield %mm
```

`acc + (0 + A·B)` 와 `acc + A·B` 가 같다는 것을 쓴다. 접는 조건 다섯 개를 전부
확인하고 하나라도 어긋나면 그대로 둔다.

| 조건 | 왜 |
| --- | --- |
| generic이 두 입력을 그대로 더하기만 함 (축 전부 parallel, map 전부 항등) | 다른 계산이 섞이면 누산기에 접을 수 없다 |
| generic의 `outs`가 그 누산기 | 아니면 다른 곳에 쓰는 것 |
| 계산 결과가 그 generic에서만 쓰임 | `outs`를 바꾸면 다른 사용처가 영향받는다 |
| 계산의 `outs`가 **0으로 채워짐** | `Z ≠ 0`이면 등식이 성립하지 않는다 |
| 누산기가 tensor 타입 iter_arg | — |

**효과.** i8 GEMM 기준으로 임시 타일 1024 B와 K 반복마다 256회 덧셈이 사라진다.
그리고 접기 전에는 그 임시 타일이 `pim.load` 결과가 아니라 `pim.co_resident`
목록에서 빠져, 상주량이 실제(2688 B)보다 작게(1664 B) 나왔다. 접으면 1664가 맞는
값이 된다.

**이어서 누산기 초기값을 그 loop 바로 앞으로 내린다.**

```mlir
// 내리기 전 (LICM이 grid loop 밖까지 빼 둔 상태)
%2 = tensor.empty() : tensor<8x32xi32>
%3 = linalg.fill ins(%c0_i32) outs(%2)
scf.for %gx { scf.for %gy {
  %r = scf.for %k iter_args(%acc = %3) { ... }

// 내린 뒤
scf.for %gx { scf.for %gy {
  %X = tensor.empty() : tensor<8x32xi32>
  %Y = linalg.fill ins(%c0_i32) outs(%X)
  %r = scf.for %k iter_args(%acc = %Y) { ... }
```

tensor는 값 의미라 LICM이 빼는 것이 정당하다. 하지만 **버퍼로 내리면 초기화가 한 번만
돌아 두 번째 인스턴스부터 앞 결과가 남는다.** 유닛 하나가 64개 인스턴스를 도는데 첫
출력 타일만 맞고 나머지 63개가 틀린다.

접기가 선행 조건이다. 접기 전에는 초기값이 `iter_args`와 `matmul`의 `outs` 두 곳에서
쓰여 함부로 옮길 수 없다.

### 5.5 `--pim-analyze-reuse`

grid 축과 각 access의 의존 관계를 분석해 속성으로 단다. 치환은 하지 않는다.

| 대상 | 속성 | 내용 |
| --- | --- | --- |
| func | `pim.execution_model` | `"spmd"` |
| func | `pim.grid_dims` | grid loop 중첩 순서 (`["x", "y"]`) |
| func | `pim.split_axes` | 쪼갤 수 있는 축 목록 |
| access | `pim.access_id` | 뒤 패스가 가리키는 이름 |
| access | `pim.role` | `"input"` \| `"output"` |
| access | `pim.invariant_grid_dims` | 의존하지 **않는** grid 축 |
| scf.for | `pim.reduction_combine` | 누산 loop의 결합 연산자 |

**의존성은 backward slice로 구한다.** arith 체인, `scf.for`의 iter_arg와 결과,
`scf.if`의 결과를 따라간다.

`scf.for`의 **결과**를 따라가는 것이 중요하다. 피연산자(`lb`/`ub`/`step`/`init`)만
보면 몸체가 `yield`하는 값을 놓친다. 저장 값이 loop 결과인 경우(split-K) 의존이
통째로 누락된다.

추적 불가한 leaf를 만나면 모든 축에 의존한다고 본다. 재사용 가능성을 과대주장하지
않는다.

**영역을 정하는 것은 offset 하나가 아니다.** `strides`는 축 간격을, `valid`는 경계
타일의 유효 범위를, `index`는 row gather가 실제로 읽는 행을 정한다. `pad`는 주소는
아니지만 유효 범위 밖 원소의 값이 되어 load 결과를 바꾸므로 함께 본다.

**`pim.access_id`를 여기서 붙이는 이유.** `pim.access`는 `Pure`라 `-cse`가 같은
것을 합친다. id를 더 일찍(conversion에서) 붙이면 전부 달라져 합쳐지지 않고,
`--pim-remove-redundant-transfers`의 SSA 동일성 판별이 무너진다. 그래서 cse 뒤인
여기서 붙인다.

**grid 축 결합 판정.** 이 부분이 미묘하다.

```
출력 위치가 축 d에 불변  →  d만 다른 인스턴스들이 같은 자리에 쓴다
```

여기까지는 맞지만, "같은 자리에 쓴다"가 "합쳐야 한다"를 함의하지 않는다. 저장되는
**값**도 d에 불변이면 모든 인스턴스가 같은 것을 쓰는 중복 실행이고, 합치면 결과가
유닛 수만큼 커진다.

| 저장 위치 | 저장 값 | 뜻 | 판정 |
| --- | --- | --- | --- |
| d에 불변 | d에 **의존** | 부분 결과 | `combine` |
| d에 불변 | d에 **불변** | 중복 실행 | 결합 금지 |
| d에 불변 | 추적 실패 | 모름 | 그 축을 아예 쪼개지 않음 |

그래서 `pim.store`의 **값 피연산자**를 따로 역추적한다. `pim.copy`가 소비하는
출력이면 src access의 의존을 본다.

**쪼갤 수 없는 축.** 다음은 `pim.split_axes`에 올리지 않는다.

- `pim.scatter`가 있으면 grid 축 전부 (서로 다른 유닛이 같은 원소에 쓰지 않는다고
  보장할 수 없다)
- reduction loop 몸체에 `pim.store`/`pim.scatter`가 있으면 그 축 (반복마다 부수
  효과가 있다)
- 결합 연산자를 못 정하는 축 (모르는 채 쪼개면 target이 덮어쓰는 코드를 만든다)

### 5.6 `--pim-analyze-transfer`

| 대상 | 속성 | 내용 |
| --- | --- | --- |
| access | `pim.tile_bytes` | 한 번에 옮기는 바이트 |
| func | `pim.co_resident` | 한 순간에 함께 있어야 하는 타일 목록 |

`tile_bytes`는 `sizes` 곱 × element bit width를 올림한 값이다. `valid`는 동적
값이라 못 쓰므로 `sizes` 전체를 세는 **상한**이다. 8비트 미만 타입에서 0이 되지
않도록 올림 나눗셈을 쓴다 — i1 타일 100개면 13 바이트다.

`co_resident`는 가장 깊이 있는 전송의 블록을 잡고 두 부류를 모은다.

```
streaming  : 그 블록에서 매 반복 새로 읽는 타일 (버퍼 하나를 돌려쓴다)
persistent : 그 블록을 감싸는 loop를 관통하는 것
             - tensor 타입 iter_arg (누산기)
             - 밖에서 한 번 읽어 안에서 쓰는 타일
```

뒤엣것이 `--pim-remove-redundant-transfers`가 loop 밖으로 뺀 load다. **빠뜨리면
용량을 실제보다 작게 보고 들어가지 않는 분할을 고른다.**

```
실제 필요:  A 2048 + B 4096 + 누산기 8192 = 14336 B
빠뜨리면:        B 4096 + 누산기 8192 = 12288 B   (14% 적음)
```

합계가 아니라 목록으로 두는 이유는 target의 메모리 계층이 하나가 아니어서다. 어느
타일을 빠른 쪽에 두고 어느 것을 스트리밍할지는 개별 크기를 알아야 정한다.

`pim.gather`는 `!pim.access`를 거치지 않으므로 `tile_bytes`가 붙지 않는다. 원소별
임의 접근이라 전송량이 index 텐서의 값에 달려 컴파일 시점에 셀 수 없다. 반면
상주량은 결과 tensor 타입에서 나오므로 목록에는 든다.

### 5.7 `--pim-partition`

반복 공간을 유닛 수만큼 나눈다. **여기까지가 target 무관이고, 이 결과를 어느 메모리에
어떤 단위로 올릴지부터가 target 몫이다.**

옵션:

| 옵션 | 뜻 |
| --- | --- |
| `splits` | 축별 유닛 수. `pim.split_axes` 순서대로 하나씩 |
| `num-units` | 기계의 유닛 개수. 0이면 확인하지 않음 |
| `local-memory-bytes` | 유닛당 빠른 메모리. 0이면 확인하지 않음 |

**(a) loop 범위 좁히기.**

```
trips  = ⌈(ub − lb) / step⌉        전체 반복 횟수
per    = ⌈trips / n⌉               유닛당 배정량
chunk  = per · step
newLb  = lb + u · chunk
newUb  = min(newLb + chunk, ub)    원래 상한으로 자른다
mine   = ⌈max(newUb − newLb, 0) / step⌉    이 유닛이 실제로 도는 횟수
```

마지막 줄이 중요하다. `unit_data`의 크기가 여기서 나오는데, **배정량(`per`)을 쓰면
마지막 유닛이 원본 버퍼 밖을 가리킨다.**

```
g_x = 130, split_m = 4  →  per = ⌈130/4⌉ = 33
유닛 3:  newLb = 99,  newUb = min(132, 130) = 130   → 실제 타일 31개
         per를 쓰면 sizes[0] = 33 × 8 = 264
         origin[0] = 792,  792 + 264 = 1056 > 1040 = 실제 행 수  ← 버퍼 밖
```

유닛이 반복보다 많으면 빈 범위가 되고 크기도 0이 된다.

**(b) `pim.units` region 생성.** 축 loop nest 전체를 region 안으로 옮긴다.

**(c) `pim.unit_data` 유도.** access마다:

```
origin[d] = access.origin[d] 에 그 loop의 새 하한을 대입 (backward slice를 복제)
sizes[d]  = 그 축을 움직이는 loop의 실제 반복 횟수 × access.sizes[d]
            (움직이는 loop가 없으면 access.sizes[d] 그대로)
```

축을 움직이는 loop가 **둘 이상이면** 만들지 않는다. 유닛 하나가 맡는 영역이
직사각형이 아니어서 `origin`/`sizes`로 적을 수 없다. "없음"과 "둘 이상"을 구별해야
한다 — 뭉뚱그리면 크기를 실제보다 작게 적어 host가 데이터를 덜 보낸다.

`origin`이 비어 있거나(§10.1) 유닛 좌표로 되돌릴 수 없는 형태면 건너뛴다.
**없는 것이 틀린 것보다 낫다.**

### 5.8 `--pim-classify-compute`

각 계산 op에 target 무관한 종류를 `pim.op`로 단다.

```
matmul → gemm,  matvec → gemv,  transpose → transpose,
reduce → reduce,  fill → fill,
generic → reduction iterator 있으면 reduce
        / linalg.index 있으면 index_generation
        / 입력 맵이 차원을 drop하면 broadcast
        / 아니면 elementwise
```

합법성·opcode·분해는 태그하지 않는다(target 몫). 분류 태그는 스냅샷이라 op을 바꾸는
패스 뒤에 낡으므로 **맨 마지막**에 돌린다.

---

## 6. 전체 예시 — i8 GEMM

`pim_backend/examples/test_pim_matmul_tiled.py`의 `matmul_kernel_i8`.

```python
@triton.jit
def matmul_kernel_i8(A, B, C, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)
```

`BLOCK=(8, 32, 16)`, `splits=(4, 32, 16)`.

### 6.1 TTShared IR (입력)

```mlir
%8 = scf.for %arg12 = %c0_i32 to %arg5 step %c16_i32 iter_args(%arg13 = %1)
     -> (tensor<8x32xi32>) : i32 {
  %12 = arith.index_cast %arg12 : i32 to index
  %13 = arith.addi %7, %12 : index
  %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%13],
                        sizes: [8, 16], strides: [%6, 1] ...
```

K loop 상한이 `%arg5`(= K)로 **런타임 값**이다. 이것이 비용 모델을 낼 수 없는 이유다
(§11).

### 6.2 PIM IR (유닛 형태, 최종)

```mlir
func.func @matmul_kernel_i8(%arg0: memref<*xi8>, %arg1: memref<*xi8>, %arg2: memref<*xi32>,
                            %arg3: i32, %arg4: i32, %arg5: i32,
                            %arg6: i32, %arg7: i32, %arg8: i32)
    attributes {pim.co_resident = [{access = 0 : i64, bytes = 128 : i64, lifetime = "streaming"},
                                   {access = 1 : i64, bytes = 512 : i64, lifetime = "streaming"},
                                   {bytes = 1024 : i64, lifetime = "persistent"}],
                pim.execution_model = "spmd",
                pim.grid_dims = ["x", "y"],
                pim.split_axes = [{axis = "x", source = "grid"},
                                  {axis = "y", source = "grid"},
                                  {axis = "k", combine = "add", source = "reduction"}]} {
  %0 = arith.index_cast %arg6 : i32 to index    // g_x
  %1 = arith.index_cast %arg7 : i32 to index    // g_y
  %2 = arith.index_cast %arg5 : i32 to index    // K  (= lda)
  %3 = arith.index_cast %arg4 : i32 to index    // N  (= ldb = ldc)

  pim.units axes ["x", "y", "k"] counts [4, 32, 16] {
  ^bb0(%pm: index, %pn: index, %kj: index):
    // ── 유닛 범위 ───────────────────────────────────────────────
    %6  = arith.ceildivsi %0, %c4 : index          // 유닛당 x 타일 배정량
    %7  = arith.muli %pm, %6 : index               // newLb_x
    %9  = arith.minsi %8, %0 : index               // newUb_x
    %11 = arith.maxsi %10, %c0 : index             // 실제 x 반복 횟수
    ... (y, k 도 같은 모양) ...

    // ── 유닛이 받는 데이터 ──────────────────────────────────────
    pim.unit_data %arg0 origin[%31, %33] sizes[%32, %34] strides[%2, %c1]
        {direction = "in", access = 0 : i64, replicated_over = ["y"]} : memref<*xi8>
    pim.unit_data %arg1 origin[%33, %37] sizes[%34, %38] strides[%3, %c1]
        {direction = "in", access = 1 : i64, replicated_over = ["x"]} : memref<*xi8>
    pim.unit_data %arg2 origin[%31, %37] sizes[%32, %38] strides[%3, %c1]
        {direction = "out", access = 2 : i64,
         combine = "add", combine_over = ["k"]} : memref<*xi32>

    // ── 유닛 하나가 실행하는 부분 ───────────────────────────────
    scf.for %gx = %7 to %9 step %c1 {
      %41 = ... // m_off = gx · 8
      scf.for %gy = %13 to %15 step %c1 {
        %46 = ... // n_off = gy · 32
        %X = tensor.empty() : tensor<8x32xi32>
        %Y = linalg.fill {pim.op = "fill"} ins(%c0_i32) outs(%X)
        %47 = scf.for %k = %22 to %24 step %c16_i32 iter_args(%acc = %Y)
              -> (tensor<8x32xi32>) : i32 {
          %50 = arith.index_cast %k : i32 to index
          %52 = pim.access %arg0 offset %51 sizes [8, 16] strides[%2, %c1] origin[%41, %50]
                {pim.access_id = 0 : i64, pim.invariant_grid_dims = array<i64: 1>,
                 pim.role = "input", pim.tile_bytes = 128 : i64} : memref<*xi8> -> !pim.access
          %53 = pim.load %52 : !pim.access -> tensor<8x16xi8>
          %56 = pim.access %arg1 offset %55 sizes [16, 32] strides[%3, %c1] origin[%50, %46]
                {pim.access_id = 1 : i64, pim.invariant_grid_dims = array<i64: 0>,
                 pim.role = "input", pim.tile_bytes = 512 : i64} : memref<*xi8> -> !pim.access
          %57 = pim.load %56 : !pim.access -> tensor<16x32xi8>
          %58 = linalg.matmul {pim.op = "gemm"} ins(%53, %57) outs(%acc) -> tensor<8x32xi32>
          scf.yield %58 : tensor<8x32xi32>
        } {pim.reduction_combine = "add"}
        %63 = pim.access %arg2 offset %62 sizes [8, 32] strides[%3, %c1] origin[%41, %46]
              {pim.access_id = 2 : i64, pim.invariant_grid_dims = array<i64>,
               pim.role = "output", pim.tile_bytes = 1024 : i64} : memref<*xi32> -> !pim.access
        pim.store %47, %63 : tensor<8x32xi32>, !pim.access
      } {pim.grid_dim = "y"}
    } {pim.grid_dim = "x"}
  }
  return
}
```

### 6.3 숫자 검산

`M=1024, N=8192, K=8192`일 때:

```
g_x = ⌈1024/8⌉  = 128    →  유닛당 x 타일 = ⌈128/4⌉  = 32
g_y = ⌈8192/32⌉ = 256    →  유닛당 y 타일 = ⌈256/32⌉ = 8
K 반복 = 8192/16 = 512   →  유닛당 K 반복 = ⌈512/16⌉ = 32
```

| | origin | sizes | 바이트 |
| --- | --- | --- | --- |
| A | `[pm·256, kj·512]` | `[32·8, 32·16] = [256, 512]` | 128 KB |
| B | `[kj·512, pn·256]` | `[32·16, 8·32] = [512, 256]` | 128 KB |
| C | `[pm·256, pn·256]` | `[32·8, 8·32] = [256, 256]` | 256 KB (i32) |

유닛 하나가 `32 × 8 = 256`개 인스턴스를 돌고, 각 인스턴스가 `8 × 32` 출력 타일을
만든다. `256 × 1024 = 262144 = 256 × 256` 원소로 C 영역과 맞는다.

### 6.4 복제와 전송량

```
A는 pn과 무관  →  서로 다른 A 조각 4 × 16 = 64개,   각각 32개 유닛이 공유
B는 pm과 무관  →  서로 다른 B 조각 16 × 32 = 512개, 각각  4개 유닛이 공유
C는 전부 다름  →  2048개, kj 방향 16개를 host가 결합
```

| | 원본 크기 | 서로 다른 조각 합 | 버스 위 총량 |
| --- | --- | --- | --- |
| A | 8 MB | 64 × 128 KB = 8 MB | 2048 × 128 KB = **256 MB** |
| B | 64 MB | 512 × 128 KB = 64 MB | 2048 × 128 KB = **256 MB** |
| C | 32 MB | — | 2048 × 256 KB = **512 MB** |

`replicated_over`를 읽으면 서로 다른 조각만 준비하고 전송만 유닛 수만큼 하면 된다.

### 6.5 유닛 메모리 소요

```
pim.co_resident = A 타일 128 + B 타일 512 + 누산기 1024 = 1664 B
```

UPMEM WRAM이 64 KB이므로 `1664 ≤ 65536` 판정이 바로 된다. `--pim-partition`이
`local-memory-bytes` 옵션을 받으면 이 확인을 컴파일 시점에 한다.

---

## 7. 속성 목록

### 7.1 func

| 속성 | 붙이는 곳 | 읽는 곳 | 내용 |
| --- | --- | --- | --- |
| `pim.execution_model` | analyze-reuse | Target IR | `"spmd"` |
| `pim.grid_dims` | analyze-reuse | partition | grid loop 중첩 순서 |
| `pim.split_axes` | analyze-reuse | partition | 쪼갤 수 있는 축 + 결합 연산자 |
| `pim.co_resident` | analyze-transfer | partition, Target IR | 동시 상주 타일 목록 |

### 7.2 `pim.access`

| 속성 | 붙이는 곳 | 읽는 곳 | 내용 |
| --- | --- | --- | --- |
| `pim.access_id` | analyze-reuse | analyze-transfer, partition | 이 access를 가리키는 이름 |
| `pim.role` | analyze-reuse | partition | `"input"` \| `"output"` |
| `pim.invariant_grid_dims` | analyze-reuse | partition | 의존하지 않는 grid 축 |
| `pim.tile_bytes` | analyze-transfer | Target IR | 한 번에 옮기는 바이트 |

### 7.3 `scf.for`

| 속성 | 붙이는 곳 | 읽는 곳 | 내용 |
| --- | --- | --- | --- |
| `pim.grid_dim` | build-grid | 모든 뒤 패스 | `"x"` \| `"y"` \| `"z"` |
| `pim.reduction_combine` | analyze-reuse | partition | 누산 loop의 결합 연산자 |

### 7.4 계산 op

| 속성 | 붙이는 곳 | 읽는 곳 | 내용 |
| --- | --- | --- | --- |
| `pim.op` | classify-compute | Target IR | 연산 종류 |

---

## 8. Target IR이 읽어야 하는 것

유닛 형태 PIM IR과 Target IR의 대응이 거의 이름 바꾸기가 되도록 설계했다.

| 유닛 형태 PIM IR | host 코드 | device 코드 |
| --- | --- | --- |
| `pim.units` region **밖** | 전부 | — |
| `pim.units` region **안** | — | 전부 |
| `counts` | 유닛 수, 좌표 분해 | — |
| block argument | `DPU_FOREACH` 인덱스 → 좌표 | 자기 좌표 (브로드캐스트) |
| `pim.unit_data` `direction="in"` | 원본에서 모아 전송 | — |
| `pim.unit_data` `direction="out"` | 회수 | — |
| `replicated_over` | 중복 준비 생략 | — |
| `combine` / `combine_over` | 결합 트리 | — |
| region 안의 `pim.access` `sizes` | — | 내부 메모리 버퍼 크기 |
| region 안의 `pim.load` / `store` | — | 큰 메모리 ↔ 빠른 메모리 이동 |
| `pim.access` `valid` / `pad` | — | 안쪽 loop 상한, 채움 |
| `pim.op` | — | 계산 커널 선택 |
| `pim.co_resident` | — | 빠른 메모리 배치 |

유닛 로컬 좌표는 `pim.access.origin − pim.unit_data.origin`이며, 짝은 `access`
필드로 찾는다.

**Target IR이 정하는 것**(PIM IR에 없는 것):

- 메모리 계층 배치와 계층 사이 전송 단위
- 레이아웃 변환 (예: 연속 접근을 위한 전치)
- 유닛 내부 병렬 분배와 동기화
- 정렬 제약에 따른 padding
- 결합을 어디서 하는지 (host / 유닛 / 가속기)
- 계산과 전송의 중첩 (double buffering)

---

## 9. 백엔드 옵션

`pim_backend/backend/compiler.py`의 `PIMOptions`.

```python
target: str = None                   # "upmem" | "aimx" — 없으면 즉시 에러
device_id: int = 0

# 기계 — 이름을 target 중립으로 둔다 (UPMEM에서는 유닛=DPU, 빠른 메모리=WRAM)
num_units: int = 2048
local_memory_bytes: int = 64 * 1024

# 분할 — 지금은 사용자가, 나중에는 autotune이
split_m: int = 1
split_n: int = 1
split_k: int = 1
local_lanes: int = 1                 # 유닛 내부 병렬 수
```

전부 `PIMOptions.hash()`가 `__dict__` 전체를 쓰므로 컴파일 캐시 키에 들어간다.
기계 값을 바꾸면 재컴파일된다.

`splits`는 `pim.split_axes` 순서대로 전달된다. 지금 분석이 내는 순서는 grid 축(중첩
순서) 다음 reduction 축이므로 `split_m, split_n, split_k`가 그대로 맞는다.

**축 이름 기반으로 바꾸는 것을 권한다.** `split_m/n/k`는 GEMM 전용 이름이고, IR이
내는 축 이름(`x`/`y`/`z` + `k`/`k0`/`k1`)과 다르다. 1차원 grid 커널이나 축약 loop가
둘인 커널을 쓰면 위치 대응이 무너진다.

```python
split: dict = None   # {"x": 4, "y": 32, "k": 16}
```

---

## 10. 알려진 한계

### 10.1 `origin` 분해가 안 되는 형태

주소를 loop `iter_args`로 나르는 커널이 있다. 상류 CPU 예제
(`triton_shared/python/examples/test_matmul.py`)가 이 형태다.

```mlir
%43:3 = scf.for %k = ... iter_args(%aoff = %32, %boff = %c0, %acc = %2) {
  %64 = pim.access %arg0 offset %aoff ...        // 주소가 loop를 타고 흐른다
  %70 = arith.addi %aoff, %c16 : index
```

`offset`이 `addi`/`muli` 체인이 아니라 block argument라 축으로 분해할 수 없다.
`origin`이 비고, 그 access는 `pim.unit_data`가 나오지 않는다.

PIM backend 예제(`test_pim_matmul_tiled.py`)는 유도변수에서 매번 계산하는 형태라
해당하지 않는다. 필요해지면 이 형태를 정규화하는 것이 해법이다 — Target IR이 두
경우를 각자 다루는 것보다 낫다.

### 10.2 1차원 grid + grouped ordering

```mlir
%20 = arith.divsi %19, %7 : i32      // pid 하나에서 m, n을 둘 다 뽑는다
%27 = arith.divsi %26, %23 : i32
```

grid 축이 하나뿐이라 A도 B도 그 축에 의존한다. GEMM인데 `invariant_grid_dims`가
전부 비고 재사용이 0으로 나온다. 분석이 조용히 아무것도 못 찾는다.

이 형태를 감지해 경고를 내는 것이 좋겠다.

### 10.3 `pim.co_resident`의 형제 블록

가장 깊은 블록을 하나만 고른다. 같은 깊이의 형제 블록이 둘이면(연속된 reduction
loop 두 개) 먼저 만난 쪽만 목록에 든다. 역시 용량을 실제보다 작게 본다.

### 10.4 `pim.copy` verifier 없음

`PIMCommon_Design.md` §6.2가 두 access의 base element type이 같은지 확인하라고
한다. 지금은 `--pim-remove-redundant-transfers`가 타입이 이미 맞는 load+store
쌍에서만 만들어 위반이 불가능하다. Target IR이 직접 만들기 시작하면 필요하다.

### 10.5 `mayAlias` 가정

`--pim-remove-redundant-transfers`가 base memref가 다르면 겹치지 않는다고 본다.
Triton은 커널 포인터 인자가 서로 다른 버퍼임을 보장하지 않는다.

---

## 11. `PIMCommon_Design.md`와 어긋나는 곳

| 설계 | 구현 | 이유 |
| --- | --- | --- |
| §16 비용 모델 (`transfer_count`, `compute_ops`, `grid_extents`) | **없음** | 셋 다 감싸는 loop 반복 횟수를 컴파일 시점 상수로 요구하는데, 실제 커널에서 K loop 상한은 런타임 인자이고 grid 상한은 `num_programs` 인자라 상수인 적이 없다. 값이 필요해지면(autotune) 반복 횟수와 반복 1회분을 나눠, 곱하는 쪽은 그 수를 아는 launch 계층에 두어야 한다 |
| §16.5 "K 상한과 base 주소만 달라진다" | grid 축은 맞고 축약 축은 형태에 따라 다름 | §10.1 참고 |
| §5.5 grid 축 결합 판정 | 저장 **값**의 축 의존으로 바뀜 | 위치만 보면 중복 실행을 부분 결과로 오인해 결과가 유닛 수만큼 커진다 |
| §20.2 "IR을 바꾸는 패스를 전부 돌린 뒤 분석" | `--pim-partition`이 예외 | 분석이 낸 `pim.split_axes`를 입력으로 쓰므로 분석 뒤에 올 수밖에 없다 |
| `pim.access`가 `offset` 하나 | `origin`이 추가됨 | §4.1 참고 |
| 유닛 형태 없음 | `pim.units` / `pim.unit_data` 추가 | §4.3, §4.4 참고 |

---

## 12. 파일 위치

```
include/triton-shared/Dialect/PIM/IR/PIMDialect.td        dialect 정의
lib/Dialect/PIM/IR/PIMDialect.cpp                         verifier

include/triton-shared/Conversion/TTSharedToPIM/Passes.td  pass 정의와 문서
lib/Conversion/TTSharedToPIM/
  TTSharedToPIMPass.cpp              --convert-ttshared-to-pim
  PIMBuildGridPass.cpp               --pim-build-grid
  PIMRemoveRedundantTransfersPass.cpp --pim-remove-redundant-transfers
  PIMNormalizeAccumulatorPass.cpp    --pim-normalize-accumulator
  PIMAnalyzeReusePass.cpp            --pim-analyze-reuse
  PIMAnalyzeTransferPass.cpp         --pim-analyze-transfer
  PIMPartitionPass.cpp               --pim-partition
  PIMClassifyComputePass.cpp         --pim-classify-compute

test/Conversion/TTSharedToPIM/       pass별 lit 테스트
pim_backend/backend/compiler.py      파이프라인과 옵션
```

파이프라인은 `compiler.py`의 `PIM_COMMON_PASSES`에 한 줄씩 있고, 실험할 때 필요 없는
단계는 그 줄을 주석 처리해 껐다 켤 수 있다.

---

## 13. 직접 확인하는 방법

전체 파이프라인을 실제 커널에 돌린다.

```bash
cd /home/dlrkdals/TritonPIM
<build-dir>/bin/triton-shared-opt \
  --convert-ttshared-to-pim --pim-build-grid -cse -loop-invariant-code-motion \
  --pim-remove-redundant-transfers -canonicalize \
  --pim-normalize-accumulator \
  --pim-analyze-reuse --pim-analyze-transfer \
  --pim-partition="splits=4,32,16 num-units=2048 local-memory-bytes=65536" \
  -cse -canonicalize --pim-classify-compute \
  dump/<hash>/matmul_kernel_i8.ttsharedir
```

IR 덤프를 단계별로 보려면:

```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/home/dlrkdals/TritonPIM/dump \
TRITON_ALWAYS_COMPILE=1 python pim_backend/examples/test_pim_matmul_tiled.py
```

`dump/<hash>/`에 `ttsharedir`, `pim_common`이 단계별로 쌓인다.

테스트:

```bash
cd /home/dlrkdals/TritonPIM/triton_shared
cmake --build <build-dir> --target check-triton-shared-lit-tests
```
