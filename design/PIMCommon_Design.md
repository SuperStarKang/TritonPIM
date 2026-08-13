# TTShared IR 이후 PIM Common IR 설계

## 1. 설계 목표

PIM Common IR은 TTShared의 구조화된 IR을 PIM에서 실행 가능한 형태로 표현하되, 특정 PIM 하드웨어에 종속되지 않는 중간 계층입니다.

대상 하드웨어는 UPMEM, AiMX, HBM-PIM입니다. 세 구조의 공통 성질은 다음과 같습니다.

* 메모리가 물리적으로 나뉘고, 연산 유닛은 자기 몫에만 직접 접근한다
* 유닛끼리 직접 통신할 수 없고, 데이터 교환은 호스트를 거친다
* 호스트가 전송을 명시적으로 수행하며 캐시가 없다
* 같은 값을 여러 유닛에 보내는 경로가 따로 있다
* 유닛 수는 많고 유닛당 메모리는 작다

여기서 나오는 최적화 문제는 세 target에서 같습니다. 어느 축을 유닛에 분할할 것인가, 어느 데이터가 여러 유닛에서 같은가, 유닛 간 리덕션이 필요한가. PIM Common IR은 이 공통 문제를 한 번만 풀어 Target IR에 넘깁니다.

PIM Common IR이 담당하는 범위는 다음과 같습니다.

* PIM에서 실행되는 논리적 작업 단위 보존
* grid를 명시적 loop로 표현해 program instance 사이의 관계를 드러냄
* 입력과 출력 데이터의 논리적 접근 영역 표현
* 외부 메모리와 PIM 계산 영역 사이의 데이터 이동 표현
* 반복되는 데이터 접근과 중복 전송 제거
* 축을 쪼갤 때 나누는 데이터와 복제하는 데이터의 구분
* 쪼갤 수 있는 축과, 쪼개면 유닛 밖 결합이 필요한 축의 구분
* 전송량, 유닛당 연산량, 유닛당 상주량 계산에 필요한 값 제공
* GEMM, reduction, elementwise와 같은 연산 종류 표현

다음 정보는 PIM Common IR에 포함하지 않습니다.

* DPU 개수, AiMX Card 개수, AiM chip과 bank 개수
* MRAM, WRAM, global buffer와 같은 물리 메모리 계층과 그 용량
* 분할 후보 중 무엇을 고를지에 대한 결정
* Split-K 분할 개수의 결정
* alignment와 padding 방식
* DMA descriptor 및 명령어 형식
* 실제 데이터 residency
* Target별 연산 fusion과 decomposition

이러한 정보는 UPMEM IR이나 AiMX IR과 같은 Target IR에서 결정합니다.

### 분석, 결정, 실현의 분리

PIM Common과 Target IR의 경계를 정하는 원칙은 다음과 같습니다.

> 사실은 PIM Common이 기록하고, 수치는 Target IR이 넣는다.

분할을 예로 들면 세 단계로 나뉩니다.

| 단계 | 하는 일 | 누가 | 근거 |
| --- | --- | --- | --- |
| 분석 | 무엇이 공유되는가, 무엇이 같은 유닛 메모리에 있어야 하는가, 어느 축을 쪼갤 수 있는가 | PIM Common | 전부 주소 식과 loop 구조에서 나옵니다 |
| 결정 | 몇 개 유닛에 어떤 비율로 쪼갤 것인가 | Target IR | 유닛 개수와 유닛 메모리 용량을 알아야 합니다 |
| 실현 | 유닛 인덱스에서 데이터 영역을 계산하고 전송·결합 코드를 낸다 | Target codegen | 전송 API와 메모리 계층에 종속됩니다 |

PIM Common은 숫자를 만들지 않고 사실만 기록합니다. 전송량과 상주량은 대역폭·용량과 곱해야 시간이 되므로, PIM Common은 곱하기 전의 항만 내보내고 Target IR이 자기 값을 넣습니다.

이 경계 덕분에 분할 실현은 IR loop를 재구조화하지 않아도 됩니다. 유닛에 줄 인자와 base 주소를 계산하는 일이므로 Target codegen에서 끝납니다.

---

# 2. 전체 변환 구조

## 2.1 입력 IR

`--triton-to-linalg-experimental`은 단일 변환이 아니라 내부 파이프라인입니다.

```text
triton-to-structured      → tts.make_tptr, tts.load, tts.store
triton-to-unstructured    → tts.gather, tts.scatter
triton-arith-to-linalg    → linalg.matmul / reduce / generic
                             program ID 인자 6개가 여기서 붙는다
structured-to-memref      → memref.reinterpret_cast, alloc, copy, subview
unstructured-to-memref    → linalg.generic + tensor.extract / memref.store
ptr 정리, dead value 제거, cse, canonicalize
```

PIM Common IR은 이 파이프라인이 **모두 끝난 결과물**을 입력으로 받습니다. Triton 컴파일 산출물 중 `.ttsharedir`이 그 결과물입니다.

이 레벨을 고르는 이유는 세 가지입니다.

* grid 크기가 항상 SSA 값으로 존재한다 (§5.2). `tts` 레벨이나 TTIR에서는 커널이 `tl.num_programs()`를 호출했을 때만 얻을 수 있다.
* 주소가 `memref.reinterpret_cast`의 offset / sizes / strides로 명시되어 있어 전송량 계산에 바로 쓸 수 있다.
* `--triton-to-linalg-experimental`은 외부 코드다. 그 안에 개입하지 않으면 상류 변경에 덜 흔들린다.

대가도 있습니다. `tts` 레벨이 명시적으로 들고 있던 정보(마스크 여부, block pointer의 `order`, gather 축)를 memref 관용구에서 다시 알아내야 합니다. 랩어라운드 접근(④)을 지원하지 못하는 것도 이 때문입니다 — 필요한 경계값이 memref 레벨에는 남지 않습니다.

## 2.2 입력에 나타나는 op

`--convert-ttshared-to-pim`이 다루어야 하는 op은 다음과 같습니다. §1의 dialect 목록보다 넓습니다.

| dialect | 자주 나타나는 op |
| --- | --- |
| `memref` | `reinterpret_cast`, `alloc`, `copy`, `subview`, `cast`, `store` |
| `bufferization` | `to_tensor`, `materialize_in_destination`, `alloc_tensor` |
| `tensor` | `empty`, `extract`, `insert`, `extract_slice`, `expand_shape`, `collapse_shape` |
| `linalg` | `matmul`, `reduce`, `transpose`, `fill`, `generic`, `index` |
| `scf` | `for`, `if` |
| `arith`, `math`, `affine` | 주소 계산, 계산 본문, `affine.store` |

특히 다음 두 가지가 중요합니다.

* **store는 `memref.copy`가 아니라 `bufferization.materialize_in_destination`입니다.** 마스크가 있으면 `tensor.extract_slice` + `memref.subview`가 앞에 붙습니다.
* **마스크 로드의 `linalg.fill`은 `scf.if`로 감싸여 있고, Triton 쪽에 `other` 값이 있을 때만 나타납니다.** `other` 없이 마스크만 있으면 `fill`이 없습니다.

## 2.3 변환

메모리 접근은 `pim.*` 연산으로 끌어올리고, 계산은 기존 `linalg` 연산을 유지합니다.

```text
memref.reinterpret_cast
+ memref.alloc
+ memref.copy
+ bufferization.to_tensor
        ↓
pim.access
+ pim.load
```

```text
linalg.matmul
linalg.reduce
linalg.generic
        ↓
동일한 linalg 연산 유지
```

연산은 이미 `linalg.matmul`, `linalg.reduce`, `linalg.generic`과 같은 구조화된 형태로 표현되므로, PIM Common에서 동일한 의미를 갖는 새로운 연산을 다시 정의하지 않습니다.

변환 후 IR의 구성은 다음과 같습니다.

```text
PIM Common IR
  ├─ pim.access / load / store / copy / gather / scatter
  ├─ scf.for  (grid loop + instance 내부 loop)
  ├─ linalg
  ├─ tensor
  ├─ arith
  └─ math
```

---

# 3. PIM Common IR의 구성

PIM Common IR은 크게 세 부분으로 구성됩니다.

```text
PIM Common IR
 ├─ 실행 구조
 │    ├─ func.func
 │    ├─ grid loop (scf.for + pim.grid_dim)
 │    └─ instance 내부 loop (scf.for)
 │
 ├─ 메모리 접근과 데이터 이동
 │    ├─ pim.access
 │    ├─ pim.load
 │    ├─ pim.store
 │    ├─ pim.copy
 │    ├─ pim.gather
 │    └─ pim.scatter
 │
 └─ 계산
      ├─ linalg.matmul
      ├─ linalg.matvec
      ├─ linalg.reduce
      ├─ linalg.transpose
      ├─ linalg.fill
      ├─ linalg.generic
      ├─ tensor
      ├─ arith
      └─ math
```

PIM Common IR의 기본 원칙은 다음과 같습니다.

> SPMD 실행 구조를 grid loop로 명시하고, 복잡한 memref 접근 패턴은 논리적 주소와 전송 연산으로 단순화하며, 계산 의미는 linalg에 보존한다.

---

# 4. 최소 PIM 연산 구성

초기 PIM Common dialect는 다음 연산으로 구성합니다.

| 연산            | 역할                              |
| ------------- | ------------------------------- |
| `pim.access`  | 데이터의 논리적 주소와 접근 영역 표현           |
| `pim.load`    | access가 가리키는 영역을 읽어 tensor 값 생성 |
| `pim.store`   | tensor 값을 access가 가리키는 영역에 기록   |
| `pim.copy`    | 두 access 사이의 직접 데이터 이동          |
| `pim.gather`  | 원소 단위의 비정형 데이터 읽기               |
| `pim.scatter` | 원소 단위의 비정형 데이터 쓰기               |

계산은 다음 기존 연산을 사용합니다.

| 연산 종류       | 표현                                |
| ----------- | --------------------------------- |
| GEMM        | `linalg.matmul`                   |
| GEMV        | `linalg.matvec`                   |
| Reduction   | `linalg.reduce`                   |
| Transpose   | `linalg.transpose`                |
| Fill        | `linalg.fill`                     |
| Elementwise | `linalg.generic`                  |
| Broadcast   | `linalg.generic`                  |
| Index generation | `linalg.generic` + `linalg.index` |

이 구조에서는 메모리 접근 의미만 PIM dialect에서 정의하고, 계산 의미는 표준 MLIR dialect가 담당합니다.

---

# 5. 실행 구조: grid와 program instance

## 5.1 grid는 어느 IR에도 없다

Triton에서 커널은 grid로 실행됩니다.

```python
grid = (32, 16)
matmul_kernel[grid](a, b, c, M, N, K, ...)
```

이 한 줄이 커널 몸체를 32 × 16 = 512번 실행합니다. 각 실행이 program instance 하나이고 서로 독립적입니다.

중요한 점은 **grid가 어느 IR 레벨에도 없다**는 것입니다. TTIR도 TTShared도 program instance 하나만 표현하고, 512번 반복하는 일은 runtime이 합니다. 그래서 instance 사이에 어떤 데이터가 겹치는지를 IR에서 볼 수 없습니다.

지금 UPMEM 경로에서는 그 loop가 `UPMEMToHost.cpp`가 생성하는 C 코드 안에 `DPU_FOREACH`로 존재합니다. A 전송의 중복 제거도 그 안에 손으로 작성되어 있습니다. 즉 최적화가 컴파일러 밖에 있고, GEMM 하나에만 적용됩니다.

`--pim-build-grid`는 그 loop를 IR로 옮깁니다. 잃어버린 정보를 복원하는 것이 아니라 **책임을 runtime에서 컴파일러로 옮기는 것**입니다.

## 5.2 program ID의 실제 표현

`triton-arith-to-linalg`는 모든 커널 인자 끝에 i32 6개를 붙입니다.

```text
... 사용자 인자 ..., num_programs_x, num_programs_y, num_programs_z,
                     program_id_x,   program_id_y,   program_id_z
```

실제 `matmul_kernel`의 시그니처입니다.

```mlir
func.func @matmul_kernel(
    %arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>,
    %arg3: i32, %arg4: i32, %arg5: i32,       // M, N, K
    %arg6: i32, %arg7: i32, %arg8: i32,       // stride_am, stride_bk, stride_cm
    %arg9: i32, %arg10: i32, %arg11: i32,     // num_programs x, y, z
    %arg12: i32, %arg13: i32, %arg14: i32)    // program_id x, y, z
```

`num_programs` 인자는 커널이 쓰지 않아도 항상 붙습니다. 이것이 중요합니다. **TTShared는 grid 크기가 항상 SSA 값으로 존재하는 첫 레벨입니다.**

## 5.3 grid loop 세우기

`--pim-build-grid`는 program ID 인자를 loop의 induction variable로 바꾸고, `num_programs` 인자를 loop의 상한으로 씁니다.

**변환 전**

```mlir
func.func @matmul_kernel(%A, %B, %C, %M, %N, %K, %sam, %sbk, %scm,
                         %np_x: i32, %np_y: i32, %np_z: i32,
                         %pid_x: i32, %pid_y: i32, %pid_z: i32) {
  ...          // %pid_x 로 tile 좌표를 계산
  return
}
```

**변환 후**

```mlir
func.func @matmul_kernel(%A, %B, %C, %M, %N, %K, %sam, %sbk, %scm,
                         %np_x: i32, %np_y: i32, %np_z: i32) {
  %ub_x = arith.index_cast %np_x : i32 to index
  scf.for %gx = %c0 to %ub_x step %c1 {
    %pid_x = arith.index_cast %gx : index to i32
    ...        // 몸체는 그대로
  } {pim.grid_dim = "x"}
  return
}
```

`pim.execution_model`은 이 패스가 달지 않습니다. §5.5의 분석이 답니다.

바뀌는 것은 세 가지뿐입니다.

| 항목 | 변환 전 | 변환 후 |
| --- | --- | --- |
| program ID | 함수 인자 | grid loop의 induction variable |
| num_programs | 쓰이지 않음 | grid loop의 상한 |
| 몸체 | 함수 최상위 | grid loop 안 |

실제로 쓰이는 program ID 축만 loop로 만듭니다. 쓰이지 않는 축은 크기가 1이므로 loop가 필요 없습니다.

변환 후 program ID 인자는 지웁니다. 남겨 두면 뒤 패스가 인자 끝 3개를 계속 program ID로 인식합니다. `num_programs` 인자는 loop 상한으로 살아 있고, 이것이 호스트가 받는 grid 크기가 됩니다.

### loop가 두 종류가 된다

이 구분이 이후 모든 분석의 기준입니다.

* **grid loop** — program instance 사이의 반복. `pim.grid_dim` 속성이 붙는다.
* **instance 내부 loop** — K tile loop 등. 속성이 없다.

## 5.4 grouped ordering

Triton matmul 튜토리얼은 1차원 grid를 쓰고 커널 안에서 `pid_m`, `pid_n`을 유도합니다.

```python
num_pid_in_group = GROUP_M * num_pid_n
group_id     = pid // num_pid_in_group
first_pid_m  = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

이것을 grouped ordering이라 하고, TTShared IR에서는 `divsi` / `remsi` / `minsi` 연쇄로 나타납니다.

목적은 **어떤 instance가 동시에 실행되는지를 조절해 GPU L2 cache 적중률을 높이는 것**입니다. C가 9 × 9 tile일 때 처음 9개 출력 tile을 만들며 읽는 양은 다음과 같습니다.

| ordering | 읽는 A tile | 읽는 B tile | 합 |
| --- | ---: | ---: | ---: |
| row-major | 9 | 81 | 90 |
| grouped (GROUP_M = 3) | 27 | 27 | 54 |

PIM에서는 이 최적화를 쓰지 않습니다. 이유는 두 가지입니다.

첫째, 세 target 모두 L2 cache가 없어 목적하는 대상이 없습니다.

둘째, grouped ordering이 PIM에 필요한 정보를 뭉갭니다. "n이 다른 instance들이 같은 A tile을 쓴다"는 관계가 나눗셈과 나머지 연산 뒤에 숨습니다. 2차원 grid에서는 `pid_m`, `pid_n`이 곧바로 그 관계를 나타냅니다.

따라서 **PIM용 커널은 2차원 이상의 grid로 작성합니다.**

```python
pid_m = tl.program_id(axis=0)
pid_n = tl.program_id(axis=1)
```

`--pim-build-grid`는 grouped ordering 연산을 알아보지 않습니다. `%pid_x`를 induction variable로 바꿀 뿐이고 그 아래 연산은 그대로 둡니다. 그래서 커널이 grouped ordering을 쓰든 안 쓰든, `GROUP_M`이 얼마든, 어떤 변형으로 쓰였든 같은 결과를 냅니다. 의존하는 것은 §5.2의 인자 규약 하나뿐입니다.

1차원 grid로 작성된 커널은 grid 축이 하나뿐이라 분할 후보가 없습니다. PyTorch Inductor가 생성한 커널처럼 1차원 grid가 불가피한 경우에는, grouped ordering을 알아보고 다차원으로 되돌리는 정규화 패스를 따로 추가합니다. 초기 구현 범위에는 넣지 않습니다.

---

## 5.5 재사용 분석

`--pim-analyze-reuse`는 각 `pim.access`가 어느 grid 축에 의존하는지 분석합니다.

판별은 backward slice로 합니다. access의 주소를 결정하는 값들을 거슬러 올라가다 `pim.grid_dim`이 붙은 loop의 induction variable을 만나면 그 축에 의존한다고 봅니다. 속성이 없는 loop의 induction variable은 instance 내부의 일이므로 무시합니다. 추적이 불가능한 값을 만나면 보수적으로 모든 축에 의존한다고 봅니다.

### 추적 대상

접근 영역을 결정하는 것은 offset 하나가 아닙니다. 다음을 모두 추적합니다.

| 피연산자 | 결정하는 것 |
| --- | --- |
| `offset` | 시작 위치 |
| `strides` | 축 간격 |
| `valid` | 경계 tile의 유효 범위 |
| `index` | row gather가 읽는 행 |
| `pad` | 유효 범위 밖 원소의 값 |

`pim.invariant_grid_dims = [d]`의 뜻은 "d가 바뀌어도 같은 데이터를 읽는다"입니다. 위 중 하나라도 d에 의존하면 읽는 결과가 달라지므로 불변이 아닙니다.

`pad`는 주소가 아니라 값이지만, `pim.load` 결과의 일부가 되므로 함께 봅니다. 재사용이란 결국 읽어 둔 tile을 다시 쓰는 것이라, 패딩 값이 달라지면 같은 tile이 아닙니다.

`index`를 빠뜨리면 정확도가 아니라 정확성 문제가 됩니다. row gather에서 index tensor가 d에 의존하는데 offset은 아닌 경우, 불변이라고 표시하면 Target IR이 그것을 믿고 tile을 유닛에 상주시켜 잘못된 행을 읽습니다.

### 결과

```mlir
func.func @matmul_kernel(...) attributes {
  pim.execution_model = "spmd",
  pim.grid_dims = ["m", "n"]
}
```

```mlir
%dA = pim.access %A ... {
  pim.role = "input",
  pim.invariant_grid_dims = array<i64: 1>,
  pim.reuse_realizable = false
}
```

`pim.grid_dims`는 grid loop의 `pim.grid_dim` 속성을 바깥부터 모은 것입니다. `pim.invariant_grid_dims`의 인덱스는 이 순서를 가리킵니다.

`pim.role`은 access를 소비하는 방식으로 정합니다. `pim.store`나 `pim.copy`의 dst면 `output`, 그 외면 `input`입니다.

이 속성들은 Target lowering과 디버깅을 위한 정보이며, 프로그램의 계산 의미를 바꾸지 않습니다. 분석이 보수적으로 나와도 성능만 손해이고 결과는 옳습니다.

### 재사용 실현 가능성

`pim.invariant_grid_dims`는 재사용의 **가능성**을 말합니다. 그것이 실현되려면 그 데이터를 한 번만 받아 유닛 메모리에 들고 있어야 하고, 그러려면 전송을 반복 밖으로 뺄 수 있어야 합니다.

주소가 grid 아닌 loop의 induction variable에 의존하면 `pim.access`가 그 loop 안에서 만들어지므로 전송을 밖으로 뺄 수 없습니다 (§15.6의 첫 조건). GEMM의 A와 B가 이 경우입니다.

```mlir
%aoff = arith.addi %m_off, %k_off : index    // %k_off 는 K loop 유도변수에서 옴
%dA = pim.access %A offset %aoff ...
```

같은 backward slice에서 판별되므로 `pim.reuse_realizable`로 함께 답니다. 거짓이면 그 access의 재사용 할인은 loop 순서를 바꾸지 않는 한 얻을 수 없다는 뜻이고, 그 대가가 얼마인지는 `pim.tile_bytes`(§16.2)로 target이 계산합니다.

### 쪼갤 수 있는 축

어느 축을 쪼갤 수 있는지, 쪼개면 유닛 밖 결합이 필요한지를 함수 속성으로 모읍니다.

```mlir
func.func @matmul_kernel(...) attributes {
  pim.grid_dims  = ["m", "n"],
  pim.split_axes = [{axis = "m", source = "grid"},
                    {axis = "n", source = "grid"},
                    {axis = "k", combine = "add", source = "reduction"}]
}
```

| 필드 | 뜻 |
| --- | --- |
| `axis` | 축 이름 |
| `source` | `grid`면 grid 축, `reduction`이면 instance 내부 reduction loop (§5.6) |
| `combine` | 있으면 쪼갤 때 부분 출력이 생기며 이 연산자로 합쳐야 합니다 |

`pim.grid_dims`에 있는데 `pim.split_axes`에 없는 축은 쪼갤 수 없다는 뜻입니다. `pim.scatter`가 있는 커널이 그 경우입니다 — 인덱스 텐서의 값을 컴파일 시점에 모르므로 서로 다른 유닛이 같은 원소에 쓰지 않는다고 보장할 수 없습니다.

`combine`이 붙는 판정은 두 가지입니다. 출력 access가 그 grid 축에 불변이거나, 그 축이 reduction loop이거나입니다. 둘 다 "그 축만 다른 실행들이 같은 출력을 만든다"는 같은 사실입니다.

---

## 5.6 reduction 구조 분석

PIM Common은 reduction이 어디에서 일어나고, 여러 장치에 나누어 실행할 경우 무엇을 합쳐야 하는지를 속성으로 표시합니다. 물리 용량(하나의 장치에 담기는지)이나 결합 방식(Host 결합, 하드웨어 누적기 등)은 결정하지 않고, 논리 구조만 표현합니다. 실제 분할과 결합은 Target IR이 정합니다.

### 내부 reduction loop

하나의 program instance 안에서 accumulator에 누적하는 loop에는 결합 연산자를 함께 표시합니다.

판별 기준은 다음과 같습니다.

* tensor 타입 iter_arg를 계산 op의 outs(accumulator)로 받아 누적하고
* 그 결과를 다시 같은 iter_arg로 yield하는 `scf.for`

```mlir
%r = scf.for %tk = %c0 to %grid_k step %c1
    iter_args(%acc = %initial) -> tensor<32x64xf32>
    { pim.reduction_combine = "add" } {
  ...
  %next = linalg.matmul ins(%a, %b) outs(%acc) -> tensor<32x64xf32>
  scf.yield %next : tensor<32x64xf32>
}
```

결합 연산자는 누적 계산 op에서 읽습니다.

| 계산 op                          | `pim.reduction_combine`       |
| ------------------------------ | ----------------------------- |
| `linalg.matmul` / `linalg.matvec` | `add` (곱을 더해 누적)          |
| `linalg.reduce`                | body의 결합 arith               |
| `linalg.generic` (reduction)   | body의 결합 arith               |

body의 결합 arith는 `addf`/`addi`→`add`, `maxf`/`maxsi`→`max`, `mulf`/`muli`→`mul`과 같이 대응시킵니다.

이 표시는 Target IR이 loop의 reduction 축을 여러 유닛에 나눌 때, 각 유닛의 부분합을 어떤 연산으로 합칠지 알려 줍니다.

### 실제 IR에서의 모양

위 예제는 `linalg.matmul`이 직접 누산기를 `outs`로 받는 형태인데, triton-shared가 만드는 GEMM은 그렇지 않습니다.

```mlir
%68 = linalg.matmul  ins(%a, %b) outs(%zero_init)                // %zero_init 은 루프 불변
%69 = linalg.generic ins(%acc, %68) outs(%acc) { arith.addf }    // 별도 누적
scf.yield %69
```

matmul은 루프 밖에서 만든 zero tensor에 쓰고, 누산기에 더하는 일은 별도의 `linalg.generic`이 합니다. 따라서 **누적하는 op은 `linalg.generic`이고 결합 연산자는 그 body의 `arith.addf`에서 읽습니다.** 판별 기준(tensor iter_arg를 `outs`로 받아 그 결과를 yield)은 그대로 성립하므로 분석은 두 형태 모두에서 동작합니다.

이 모양은 K 반복마다 tile 크기의 elementwise add가 한 번씩 더 붙는다는 뜻이기도 합니다. `matmul(outs=zero) + add(acc)`를 `matmul(outs=acc)`로 합치는 정규화는 완전히 target 독립적이지만, 초기 구현 범위에는 넣지 않습니다.

### 유닛 밖 결합이 필요한 축

reduction loop는 그 자체로 쪼갤 수 있는 축입니다. 서로 다른 유닛이 K 구간을 나눠 맡으면 각자 완전한 크기의 부분합을 만들고, 그것들을 합쳐야 최종 출력이 됩니다.

grid 축에서도 같은 일이 생깁니다. 출력 access가 축 d에 불변이면 d만 다른 instance들이 같은 출력 영역에 쓰므로 합쳐야 합니다.

두 경우 모두 `pim.split_axes`의 `combine` 필드로 표시합니다 (§5.5).

```mlir
func.func @matmul_kernel(...) attributes {
  pim.grid_dims  = ["m", "n"],
  pim.split_axes = [{axis = "m", source = "grid"},
                    {axis = "n", source = "grid"},
                    {axis = "k", combine = "add", source = "reduction"}]
}
```

`combine` 값은 그 reduction loop의 `pim.reduction_combine`에서 가져옵니다. 이 표시는 "덮어쓰지 말고 합쳐야 한다"는 논리 사실이며, 그 축을 실제로 쪼갤지와 합치는 위치·방식은 Target IR이 결정합니다.

---

# 6. `pim.access`

## 6.1 기본 역할

`pim.access`는 원본 memref의 특정 영역을 가리키는 논리적 주소 descriptor입니다.

```mlir
%dA = pim.access %A
  offset %a_offset
  sizes [32, 256]
  strides [%a_stride_m, %a_stride_k]
  : memref<*xbf16> -> !pim.access
```

`pim.access`는 다음 정보를 표현합니다.

* 원본 base memref
* 시작 offset
* 각 차원의 tile 크기
* 각 차원의 stride
* 경계 tile의 유효 영역
* masked 영역에 사용할 padding 값
* 선택적인 gather 축과 index tensor
* access의 input / output 역할
* program ID에 대한 접근식 의존 관계

`pim.access` 자체는 데이터를 이동하지 않는 Pure 연산입니다.

실제 물리 주소 계산과 메모리 배치는 Target IR에서 결정합니다.

---

## 6.2 단위와 element type

`pim.access`의 수치 피연산자는 모두 **원소 단위**입니다. 바이트가 아닙니다. `memref.reinterpret_cast`의 의미를 그대로 물려받았기 때문입니다.

| 피연산자 | 단위 |
| --- | --- |
| `offset` | 원소 |
| `strides` | 원소 |
| `sizes` | 원소 개수 |
| `valid` | 원소 개수 |

element type은 `!pim.access`에 없고 **base memref가 들고 있습니다.**

```mlir
%dA = pim.access %A offset %o sizes [32, 16] strides [%s, %c1]
      : memref<*xf32> -> !pim.access                    // base 에 f32
%a  = pim.load %dA : !pim.access -> tensor<32x16xf32>   // load 결과에도 f32
```

바이트가 필요한 계산(§16.2의 전송량, §16.6의 상주량, Target IR의 DMA 크기)은 base memref에서 element type을 읽어 변환합니다.

```text
바이트 = 원소 개수 × element bit width / 8
```

`pim.load`의 결과 tensor에서 읽어도 같은 값이지만, base memref 쪽이 낫습니다. load나 store가 없는 access에서도 동작하기 때문입니다.

8비트 미만인 타입(`pim.gather`의 mask `tensor<64xi1>` 등)에서는 정수 나눗셈이 0이 되므로 올림 나눗셈을 씁니다.

혼합 정밀도는 load 결과와 누산기 tensor의 타입이 다른 것으로 표현됩니다.

```mlir
%a = pim.load %dA : !pim.access -> tensor<32x16xi8>              // 입력
%c = linalg.matmul ins(%a, %b) outs(%acc : tensor<32x64xi32>)    // 누적
pim.store %c, %dC : tensor<32x64xi32>, !pim.access
```

`pim.copy`는 tensor를 거치지 않으므로 두 access의 base element type이 같은지 verifier가 확인합니다.

---

## 6.3 Offset 표현

현재 구조에서는 하나의 선형 offset과 축별 stride를 사용합니다.

```mlir
%d = pim.access %base
  offset %linear_offset
  sizes [32, 256]
  strides [%stride0, %stride1]
  : memref<*xbf16> -> !pim.access
```

2차원 좌표가 다음과 같다고 가정합니다.

```text
row = pid_m × BLOCK_M
column = k_offset
```

선형 offset은 다음과 같이 계산할 수 있습니다.

```text
linear_offset
  = row × row_stride
  + column × column_stride
```

PIM Common에서는 이 논리적인 offset 식만 보존합니다.

다음 정보는 Target IR에서 추가합니다.

* channel interleaving
* bank mapping
* alignment 보정
* hardware padding
* descriptor address 형식

---

## 6.4 Masked access

경계 tile에서는 tile 전체 크기와 실제 유효한 크기가 다를 수 있습니다.

```mlir
%d = pim.access %A
  offset %offset
  sizes [128]
  strides [%c1]
  valid [%valid]
  pad %zero
  : memref<*xf32> -> !pim.access
```

각 필드의 의미는 다음과 같습니다.

| 필드      | 의미                |
| ------- | ----------------- |
| `sizes` | 논리적 tile 크기       |
| `valid` | 실제로 접근 가능한 유효 크기  |
| `pad`   | 유효 영역 밖의 load 결과값 |

예를 들어 `sizes = 128`, `valid = 20`이면 앞의 20개 원소만 메모리에서 읽고, 나머지 108개는 `pad` 값으로 채웁니다.

Store에서는 유효 영역만 기록하므로 padding 값이 필요하지 않습니다.

```mlir
%d = pim.access %C
  offset %offset
  sizes [128]
  strides [%c1]
  valid [%valid]
  : memref<*xf32> -> !pim.access
```

---

## 6.5 Row gather

하나의 축은 index tensor로 선택하지만, 각 선택 영역 내부가 연속적이거나 strided한 경우 `pim.access`의 gather clause를 사용합니다.

```mlir
%d = pim.access %A
  offset %c0
  sizes [8, 64]
  strides [%stride0, %stride1]
  gather 0 index %indices
  : memref<*xf32> -> !pim.access
```

이 예제는 `%indices`로 8개의 행을 선택하고, 각 행에서 64개 원소를 읽는 접근을 표현합니다.

```text
indices[0]이 가리키는 행의 64개 원소
indices[1]이 가리키는 행의 64개 원소
...
indices[7]이 가리키는 행의 64개 원소
```

행 내부 접근은 정형적이므로 하나의 access descriptor로 표현할 수 있습니다.

---

# 7. 비정형 Gather와 Scatter

각 원소가 서로 독립적인 주소를 사용하는 접근은 `pim.gather`와 `pim.scatter`로 표현합니다.

## 7.1 Gather

```mlir
%values = pim.gather %A index %indices
  : memref<*xf32>, tensor<64xi32>
    -> tensor<64xf32>
```

의미는 다음과 같습니다.

```text
values[i] = A[indices[i]]
```

Mask가 있는 경우 다음과 같이 표현합니다.

```mlir
%values = pim.gather %A
  index %indices
  mask %mask : tensor<64xi1>
  other %zero : f32
  : memref<*xf32>, tensor<64xi32>
    -> tensor<64xf32>
```

```text
mask[i] == true  → A[indices[i]] 읽기
mask[i] == false → zero 사용
```

## 7.2 Scatter

```mlir
pim.scatter %values, %C index %indices
  : tensor<64xf32>,
    memref<*xf32>,
    tensor<64xi32>
```

의미는 다음과 같습니다.

```text
C[indices[i]] = values[i]
```

Mask가 있으면 활성화된 원소만 기록합니다.

```mlir
pim.scatter %values, %C
  index %indices
  mask %mask : tensor<64xi1>
  : tensor<64xf32>,
    memref<*xf32>,
    tensor<64xi32>
```

---

# 8. 메모리 접근 패턴 변환

TTShared structured IR에서 나타나는 메모리 접근 패턴은 다음과 같이 변환합니다.

| 접근 패턴 | TTShared structured IR | PIM Common IR |
| --- | --- | --- |
| Contiguous/strided load | `reinterpret_cast` + `alloc` + `copy` + `to_tensor` | `pim.access` + `pim.load` |
| Loop-carried | 위 구조가 `scf.for` 내부에 존재 | loop 유지, 내부 접근만 변환 |
| Block pointer | memref 단계에서 strided 접근으로 표현 | `pim.access` |
| Masked load (`other` 있음) | `alloc` + `scf.if { linalg.fill }` + `subview` × 2 + `copy` | `pim.access {valid, pad}` |
| Masked load (`other` 없음) | `alloc` + `subview` × 2 + `copy` | `pim.access {valid}` |
| Store | `materialize_in_destination` | `pim.access` + `pim.store` |
| Masked store | `extract_slice` + `subview` + `materialize_in_destination` | `pim.access {valid}` + `pim.store` |
| Row gather | 행별 `scf.for` + `reinterpret_cast` + `subview` × 2 + `copy` | `pim.access {gather}` |
| Element gather | `memref.cast` + `to_tensor` + `linalg.generic` + `tensor.extract` | `pim.gather` |
| Element scatter | `linalg.generic` + `memref.store` | `pim.scatter` |
| Wrap-around | 두 view와 두 번의 copy | 미지원 (§2.1) |

주의할 점 세 가지입니다.

* store는 `memref.copy`가 아니라 `bufferization.materialize_in_destination`입니다.
* 마스크 로드의 `linalg.fill`은 `scf.if`로 감싸여 있고, Triton 쪽 `other` 값이 있을 때만 나타납니다. 마스크 유무와 `other` 유무는 별개입니다.
* 유효 범위는 정적 속성이 아니라 `minsi` / `maxsi` / `subi` 연쇄로 계산된 동적 `index` 값입니다.

모든 structured 접근은 `memref.reinterpret_cast`를 중심으로 주소, 크기, stride를 복원할 수 있습니다.

---

# 9. `pim.load`

`pim.load`는 `pim.access`가 가리키는 데이터를 읽고 tensor 값을 생성합니다.

```mlir
%a = pim.load %dA
  : !pim.access -> tensor<32x256xbf16>
```

`pim.load`의 의미는 다음과 같습니다.

* access가 가리키는 영역을 읽음
* 계산에서 사용할 immutable tensor 값을 생성
* Read memory effect를 가짐
* staging buffer의 존재는 IR에 나타내지 않음
* 실제 DMA와 local buffer allocation은 Target IR에서 결정

다음 구조가 하나의 `pim.load`로 변환됩니다.

```mlir
%view = memref.reinterpret_cast %A ...
%buffer = memref.alloc()
memref.copy %view, %buffer
%tensor = bufferization.to_tensor %buffer
```

```text
        ↓
```

```mlir
%dA = pim.access %A ...
%tensor = pim.load %dA
```

이 변환을 통해 매 접근마다 생성되는 명시적인 staging buffer를 제거할 수 있습니다.

---

# 10. `pim.store`

`pim.store`는 tensor 값을 `pim.access`가 가리키는 영역에 기록합니다.

```mlir
pim.store %c, %dC
  : tensor<32x64xf32>, !pim.access
```

`pim.store`는 Write memory effect를 가집니다.

경계 tile에서는 `valid` 범위만 기록합니다.

```mlir
%dC = pim.access %C
  offset %c_offset
  sizes [32, 64]
  strides [%stride_cm, %stride_cn]
  valid [%valid_m, %valid_n]
  : memref<*xf32> -> !pim.access

pim.store %c, %dC
  : tensor<32x64xf32>, !pim.access
```

Target lowering에서는 이 연산을 다음과 같이 구체화할 수 있습니다.

```text
UPMEM:
local buffer → MRAM → Host

AiMX:
accumulator 또는 bank 결과
→ descriptor 기반 output transfer
→ Host memory
```

---

# 11. `pim.copy`

`pim.copy`는 두 논리적 access 사이의 직접 데이터 이동을 표현합니다.

```mlir
pim.copy %src, %dst
  : !pim.access, !pim.access
```

다음과 같은 passthrough 구조를 단순화할 때 사용합니다.

```mlir
%value = pim.load %src
pim.store %value, %dst
```

```text
        ↓
```

```mlir
pim.copy %src, %dst
```

중간 tensor가 다른 계산에서 사용되지 않는다면 load와 store를 따로 유지할 필요가 없습니다.

Target IR은 `pim.copy`를 다음 방식 중 하나로 구현할 수 있습니다.

* Host-side memory copy
* Host-to-device transfer
* Device-to-host transfer
* PIM 내부 DMA
* descriptor 기반 copy
* load/store loop

---

# 12. 계산 연산 표현

## 12.1 Named operation

구조가 명확한 계산은 named linalg operation으로 유지합니다.

```text
linalg.matmul
linalg.matvec
linalg.reduce
linalg.transpose
linalg.fill
```

GEMM은 다음과 같이 표현합니다.

```mlir
%zero = arith.constant 0.0 : f32
%empty = tensor.empty() : tensor<32x64xf32>

%init = linalg.fill
  ins(%zero : f32)
  outs(%empty : tensor<32x64xf32>)
  -> tensor<32x64xf32>

%c = linalg.matmul
  ins(%a, %b
    : tensor<32x256xbf16>,
      tensor<256x64xbf16>)
  outs(%init : tensor<32x64xf32>)
  -> tensor<32x64xf32>
```

`linalg.matmul`을 유지하면 Target IR에서 연산 구조를 다시 분석할 필요 없이 하드웨어 matmul primitive에 직접 연결할 수 있습니다.

```text
AiMX:
linalg.matmul → OPCODE_MATMUL descriptor

UPMEM:
linalg.matmul → DPU MAC loop
```

연산 패턴은 pointwise, contraction, reduction, transpose, broadcast 등의 형태로 이미 구조화되어 있습니다.

---

## 12.2 `linalg.generic`

Elementwise, broadcast, generation과 같은 연산은 `linalg.generic`으로 표현합니다.

### Elementwise add

```mlir
%result = linalg.generic
    {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel"]
    }
    ins(%a, %b
      : tensor<128xf32>,
        tensor<128xf32>)
    outs(%empty : tensor<128xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %sum = arith.addf %x, %y : f32
    linalg.yield %sum : f32
}
```

### Broadcast

```mlir
%result = linalg.generic
    {
      indexing_maps = [#broadcast_map, #identity],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%bias : tensor<64xf32>)
    outs(%empty : tensor<32x64xf32>) {
  ^bb0(%b: f32, %out: f32):
    linalg.yield %b : f32
}
```

### Generation

```mlir
%result = linalg.generic
    {
      indexing_maps = [#identity],
      iterator_types = ["parallel"]
    }
    outs(%empty : tensor<128xi32>) {
  ^bb0(%out: i32):
    %i = linalg.index 0 : index
    %value = arith.index_cast %i : index to i32
    linalg.yield %value : i32
}
```

`linalg.generic`의 body에는 `arith` 또는 `math` 연산이 포함될 수 있습니다.

```mlir
%exp = math.exp %x : f32
%sum = arith.addf %x, %y : f32
```

Target IR은 body를 분석하여 직접 지원, 분해 또는 Host fallback 여부를 결정합니다.

---

# 13. Compute 분류

Target lowering이 연산의 전체 구조를 쉽게 식별할 수 있도록 linalg 연산에 `pim.op` 속성을 붙입니다.

```text
--pim-classify-compute
```

분류 종류는 다음과 같습니다.

| `pim.op`            | 판별 기준                                    |
| ------------------- | ---------------------------------------- |
| `gemm`              | `linalg.matmul`                          |
| `gemv`              | `linalg.matvec`                          |
| `transpose`         | `linalg.transpose`                       |
| `reduce`            | `linalg.reduce` 또는 reduction iterator    |
| `fill`              | `linalg.fill`                            |
| `index_generation`  | body에 `linalg.index` 존재                  |
| `broadcast`         | 입력 indexing map이 일부 차원을 제거               |
| `elementwise`       | 모든 iterator가 parallel이고 identity mapping |
| `scan`              | 이전 원소 또는 index-carried 값에 의존             |

예시는 다음과 같습니다.

```mlir
%c = linalg.matmul
  ins(%a, %b)
  outs(%init)
  {pim.op = "gemm"}
```

```mlir
%result = linalg.generic
  {
    indexing_maps = [...],
    iterator_types = ["parallel"],
    pim.op = "elementwise"
  }
  ...
```

`linalg.generic`은 이름만으로 종류를 알 수 없습니다. reduction iterator 유무, indexing map, body를 함께 봐야 하고, 그 판별은 target과 무관합니다. 한 벌만 두어 target들이 같은 연산을 서로 다르게 분류하는 일을 막습니다.

`index_generation`은 전송이 필요 없는 연산입니다. iota를 호스트에서 만들어 보내는 대신 유닛이 자기 인덱스로 만들 수 있으므로, 이 태그가 있으면 target이 그 결정을 할 수 있습니다.

분류 패스는 연산 구조가 모두 정리된 뒤 실행합니다.

```text
canonicalization
LICM
기타 구조 변환
  ↓
pim-classify-compute
```

분류 후 구조가 바뀌면 속성이 실제 연산 구조와 달라질 수 있기 때문에 late pass로 배치합니다.

---

# 14. 데이터 재사용 분석

데이터 재사용 가능성은 `pim.access`가 어떤 grid 축에 의존하는지로 찾습니다. 분석 방법과 추적 대상은 §5.5에 있습니다.

2차원 grid로 작성한 GEMM의 접근 관계는 다음과 같습니다.

```text
A: (pid_m, k)
B: (k, pid_n)
C: (pid_m, pid_n)
```

각 access의 grid 축 의존성은 다음과 같습니다.

| 데이터 | `x`(m) 의존 | `y`(n) 의존 | 재사용 가능성 |
| --- | ---: | ---: | --- |
| A | O | X | n이 다른 instance 사이에서 재사용 가능 |
| B | X | O | m이 다른 instance 사이에서 재사용 가능 |
| C | O | O | 각 instance가 서로 다른 출력 생성 |

분석 결과는 속성으로 표현합니다. 인덱스는 `pim.grid_dims`의 순서를 가리킵니다.

```mlir
%dA = pim.access %A ... {
  pim.role = "input",
  pim.invariant_grid_dims = array<i64: 1>     // n 에 불변
}
```

```mlir
%dB = pim.access %B ... {
  pim.role = "input",
  pim.invariant_grid_dims = array<i64: 0>     // m 에 불변
}
```

GEMV는 같은 분석에서 다른 모양이 나옵니다. `y[M] = A[M,K] · x[K]`를 m 한 축의 grid로 쓰면 x만 m에 불변입니다.

```text
A: (pid_m, k)   → 불변 축 없음
x: (k)          → m 에 불변
y: (pid_m)      → 불변 축 없음
```

| 데이터 | `m` 의존 | 재사용 가능성 |
| --- | ---: | --- |
| A | O | 없음. 유닛마다 서로 다른 행 밴드 |
| x | X | 모든 instance가 같은 x를 읽음 |
| y | O | 없음. 각 instance가 서로 다른 출력 |

### 전제 조건

이 분석은 **grid 축이 서로 독립된 induction variable로 존재할 때만** 의미 있는 결과를 냅니다.

1차원 grid에 grouped ordering을 쓴 커널에서는 A, B, C의 주소가 모두 같은 `pid` 하나로 역추적됩니다. 그러면 `grid_dims`가 축 하나뿐이고 모든 access의 `invariant_grid_dims`가 빈 배열이 되어, 위 표의 정보가 나오지 않습니다. 분석이 틀리는 것이 아니라 **찾을 것이 없는 것**입니다.

따라서 §5.4의 규칙 — PIM용 커널은 다차원 grid로 작성한다 — 이 이 분석의 전제입니다.

### 이 속성이 확정하지 않는 것

실제 데이터 residency는 확정하지 않습니다. Target IR은 다음을 검사한 뒤 재사용 여부를 결정합니다.

* 같은 데이터를 쓰는 instance들이 같은 유닛에 배치되는가 (§16.1의 분할이 결정)
* 유닛 메모리에 데이터를 유지할 공간이 있는가 (§16.6)
* 데이터를 유지하는 것이 재전송보다 유리한가
* 여러 유닛에 같은 값을 보내는 경로를 쓰는 것이 더 나은가

---

# 15. PIM Common의 데이터 이동 최적화

PIM Common에서는 물리 하드웨어 정보가 없어도 안전하게 수행할 수 있는 최적화를 적용합니다.

## 15.1 동일 access CSE

```mlir
%d0 = pim.access %A offset %o sizes ... strides ...
%d1 = pim.access %A offset %o sizes ... strides ...
```

두 access가 동일하면 하나로 합칩니다.

```mlir
%d = pim.access %A offset %o sizes ... strides ...
```

`pim.access`가 Pure 연산이므로 표준 CSE를 사용할 수 있습니다.

---

## 15.2 반복 load 제거

```mlir
%v0 = pim.load %d
%v1 = pim.load %d
```

두 load 사이에 동일 영역에 대한 write가 없으면 하나의 값을 공유할 수 있습니다.

```mlir
%v = pim.load %d
```

---

## 15.3 Passthrough 제거

```mlir
%v = pim.load %src
pim.store %v, %dst
```

중간 값이 다른 계산에 사용되지 않으면 다음과 같이 바꿉니다.

```mlir
pim.copy %src, %dst
```

---

## 15.4 Store-to-load forwarding

```mlir
pim.store %v, %d
%r = pim.load %d
```

두 연산 사이에 같은 영역을 수정하는 연산이 없다면 load 결과는 `%v`와 같습니다.

```mlir
%r = %v
```

---

## 15.5 Dead load 제거

```mlir
%v = pim.load %d
```

`%v`가 어디에도 사용되지 않으면 load를 제거할 수 있습니다.

단, load가 fault나 synchronization과 같은 관찰 가능한 의미를 갖지 않는다는 PIM Common의 의미 정의가 필요합니다.

---

## 15.6 Loop-invariant load hoisting

```mlir
scf.for %i = ... {
  %v = pim.load %d
  ...
}
```

다음처럼 옮길 수 있습니다.

```mlir
%v = pim.load %d

scf.for %i = ... {
  ...
}
```

### 조건

세 가지를 모두 만족해야 옮깁니다.

| 조건 | 이유 |
| --- | --- |
| `%d`가 loop 밖에서 정의된다 | loop 안에서 만들어지면 반복마다 읽는 영역이 달라질 수 있습니다. |
| loop 안에 그 버퍼를 쓰는 op이 없다 | 반복 중간에 값이 바뀌면 한 번만 읽은 결과가 틀립니다. |
| load가 loop 몸체 블록의 직속 자식이다 | `scf.if` 안의 load를 빼면 조건이 거짓일 때도 읽게 됩니다. |

첫 조건은 `pim.access`가 이미 loop 밖에 있어야 한다는 뜻입니다. `pim.access`는 `Pure`이므로 표준 `-loop-invariant-code-motion`이 빼 줍니다. 그래서 파이프라인에서 그 패스를 `--pim-remove-redundant-transfers` **앞**에 둡니다 (§20.1).

주소가 loop 유도변수에 의존하면 이 조건이 영원히 성립하지 않습니다. 그 사실을 `pim.reuse_realizable`로 남겨 target이 재사용 할인을 기대하지 않도록 합니다 (§5.5).

여러 겹의 loop는 안쪽부터 처리하면 저절로 올라갑니다. 안쪽 loop에서 빠져나온 load는 바깥 loop 몸체의 직속 자식이 되므로 다음 차례에 다시 후보가 됩니다.

### grid loop에서는 빼지 않는다

instance 내부 loop에서 빼면 그 instance의 반복 전송이 줄어듭니다. 이것이 지금 하는 일입니다.

grid loop에서 빼면 "grid 전체가 이 tile을 한 번만 읽는다"는 뜻이 되는데, 실제 하드웨어는 그렇게 동작하지 않습니다. 유닛마다 자기 사본이 필요하고, 몇 개의 사본이 필요한지는 §16.1의 분할 구조가 정해져야 알 수 있습니다. 여기서 빼면 그 정보가 사라집니다. 따라서 `pim.grid_dim`이 붙은 loop는 건너뜁니다.

유닛마다 몇 개의 사본이 필요한지는 §16.2의 전송량 식이 표현하고, 실제 사본 수는 분할이 정해진 뒤 Target codegen이 계산합니다.

---

## 15.7 메모리 효과와 표준 패스

`pim.load`는 `MemRead` 효과를 선언합니다. 이것이 표준 패스에 어떻게 걸리는지는 최적화마다 다릅니다.

| 최적화 | 표준 패스로 되는가 | 이유 |
| --- | --- | --- |
| §15.1 동일 access CSE | `-cse`로 됩니다 | `pim.access`는 `Pure`입니다. |
| §15.5 dead load 제거 | `-canonicalize`로 됩니다 | MLIR의 `wouldOpBeTriviallyDead`는 효과가 전부 `Read`인 op을 지울 수 있다고 판정합니다. `MemRead`만 있는 `pim.load`가 여기 해당합니다. |
| §15.6 hoisting | **되지 않습니다** | `-loop-invariant-code-motion`은 `isMemoryEffectFree(op) && isSpeculatable(op)`인 op만 옮깁니다. `MemRead`는 첫 조건에서 걸립니다. |

따라서 §15.6만 `--pim-remove-redundant-transfers`에 직접 구현합니다. 그 패스에는 이미 §15.2와 §15.4를 위한 쓰기 장벽 판정(`mayWrite`)이 있으므로 같은 안전성 논리를 씁니다.

hoisting은 표준 LICM이 요구하는 `isSpeculatable`을 `pim.load`에 대해 가정하는 셈입니다. loop가 0번 도는 경우 원래 하지 않았을 읽기를 하게 되기 때문입니다. §15.5가 이미 "load는 fault나 synchronization 같은 관찰 가능한 의미를 갖지 않는다"를 전제하므로 같은 가정입니다.

### alias 가정

hoisting이 안전한지 판단하려면 loop 안의 쓰기가 읽는 영역과 겹치는지 알아야 합니다. 현재는 `pim.access`의 base memref가 다르면 겹치지 않는다고 봅니다. 영역 단위가 아니라 버퍼 단위로 보므로, 같은 버퍼의 겹치지 않는 두 영역도 겹친다고 판정합니다. 안전한 쪽입니다.

**이것은 가정입니다.** Triton은 커널 포인터 인자가 서로 다른 버퍼임을 보장하지 않습니다. 제자리 연산에서는 같은 버퍼가 두 인자로 들어올 수 있습니다.

나중에 PyTorch 그래프 레벨을 붙이면 두 인자가 같은 텐서인지 실제로 확인할 수 있으므로, 이 가정을 검증된 사실로 바꿀 수 있습니다. §21의 `pim.buffer_id`가 그 자리입니다.

---

# 16. 작업 분할

데이터 접근, 분배, 분할은 서로 다른 세 분석이 아닙니다. 전부 **주소 식이 어느 grid 축에 의존하는가** 하나에서 나옵니다.

| 단계 | 무엇을 정하는가 | 근거 속성 | 절 |
| --- | --- | --- | --- |
| 접근 | 어떤 영역을 어떤 모양으로 읽고 쓰는가 | `pim.access`의 offset · sizes · strides · valid · index | §6, §7 |
| 분배 | 유닛마다 다른 조각인가, 같은 데이터인가 | `pim.invariant_grid_dims` | §16.1 |
| 분할 | 어느 축을 쪼갤 수 있고 어디서 결합이 필요한가 | `pim.split_axes` | §16.5 |

`pim.access`가 영역 전체를 하나의 식으로 들고 있기 때문에 연쇄가 성립합니다. 그 식의 backward slice를 보면 축 의존성이 나오고 (§5.5), 의존성이 정해지면 분배가 정해지고 (§16.1), 분배가 정해지면 전송량이 계산되며 (§16.2), 여기에 부수 효과와 누적 구조를 함께 보면 쪼갤 수 있는 축이 정해집니다 (§16.5).

연쇄의 어느 단계에도 하드웨어 수치가 들어가지 않습니다. 유닛 개수, 대역폭, 용량은 마지막에 항을 곱할 때만 필요하고 그것은 Target IR의 몫입니다.

세 target에서 같은 이유는 §1의 공통 성질입니다. 유닛이 자기 몫에만 접근하고 유닛끼리 통신할 수 없으면, 복제는 반드시 실제 전송이 되고 부분합은 반드시 유닛 밖에서 합쳐집니다. 이 두 가지가 분배와 분할의 비용을 만드는 전부입니다.

## 16.1 나누는 데이터와 복제하는 데이터

축 `d`를 유닛 `s_d`개에 쪼갤 때 각 데이터가 어떻게 되는지는 한 줄로 정해집니다.

> access가 축 `d`에 **의존**하면 → 유닛마다 서로 다른 조각을 받는다 (**나눈다**)
> access가 축 `d`에 **불변**이면 → 유닛마다 같은 데이터가 필요하다 (**복제한다**)

유닛끼리 통신할 수 없으므로 복제는 실제 전송으로 나타납니다. 이것이 §1이 말한 공통 성질에서 곧바로 따라 나오는 유일한 결과이고, 세 target에서 같습니다.

판정에 필요한 것은 `pim.invariant_grid_dims` 하나뿐입니다 (§5.5).

**GEMM** `C[M,N] = A[M,K] · B[K,N]`, grid는 m, n 두 축입니다.

| 데이터 | `I_a` | m축 분할 | n축 분할 |
| --- | --- | --- | --- |
| A | {n} | 나눈다 | **복제한다** |
| B | {m} | **복제한다** | 나눈다 |
| C | {} | 나눈다 | 나눈다 |

**GEMV** `y[M] = A[M,K] · x[K]`, grid는 m 한 축입니다.

| 데이터 | `I_a` | m축 분할 |
| --- | --- | --- |
| A | {} | 나눈다 |
| x | {m} | **복제한다** |
| y | {} | 나눈다 |

두 연산의 차이가 여기서 갈립니다. GEMM은 복제 대상이 A와 B 둘이고 둘 다 큽니다. GEMV는 복제 대상이 x 하나이고 A에 비해 아주 작습니다. 같은 규칙에서 나온 결과이며 연산별 특수 처리가 필요 없습니다.

Target codegen에서 이 구분은 전송 API 선택이 됩니다.

| | 호스트가 하는 일 |
| --- | --- |
| 나눈다 | 유닛마다 다른 시작 주소로 전송 |
| 복제한다 | 여러 유닛에 같은 데이터를 전송 |

서로 다른 조각의 개수는 `Π{s_j : j ∉ I_a}`, 복제 배수는 `Π{s_j : j ∈ I_a}`입니다.

---

## 16.2 전송량 계산

축 `d`의 grid 크기를 `g_d`, 유닛 수를 `s_d`라 하면 grid 축 분할의 전송량은 다음과 같습니다.

```text
전송량 = Σ_a  tile_bytes_a × transfer_count_a × Π{ s_j : j ∈ I_a } × Π{ g_j : j ∉ I_a }
```

* `tile_bytes_a` — 접근 `a`가 한 번에 옮기는 바이트
* `transfer_count_a` — instance 하나 안에서 `a`가 전송되는 횟수
* `I_a` — `a`의 불변 축 집합 (§5.5)

읽는 법은 §16.1 그대로입니다. 접근 `a`가 축 `j`에 불변이면 복제되므로 `s_j`배, 불변이 아니면 나뉘어 서로 다른 조각 `g_j`개가 되므로 `g_j`배입니다.

`s`가 전부 1인 경우가 전송량의 하한이고, 그 값이 **실제로 필요한 데이터양**입니다.

### `tile_bytes`와 `transfer_count`를 나누어 둔다

```text
tile_bytes     = sizes 곱 × element 바이트
transfer_count = 감싸는 grid 아닌 loop 들의 반복 횟수 곱
```

`sizes`는 원소 개수이므로 바이트로 바꿔야 합니다. element type은 base memref에서 읽습니다 (§6.2).

두 값을 곱해 두지 않는 이유는 세 가지입니다.

* **전송 건수를 복원할 수 있어야 합니다.** 캐시가 없고 호스트가 명시적으로 전송하는 구조에서 건당 고정비는 실질 비용입니다. `BLOCK_K`만 바꾸면 총 바이트는 그대로인데 건수는 그 비율만큼 달라집니다.
* **유닛당 상주량 계산에 `tile_bytes`가 필요합니다** (§16.6).
* **`transfer_count`는 셀 수 없을 때가 있습니다.** K loop 상한이 런타임 값이면 세지 못합니다. 나누어 두면 항상 상수인 `tile_bytes`는 남고, `transfer_count`만 빠집니다.

`transfer_count`를 세는 기준은 `pim.access`가 아니라 **전송 op(`pim.load` / `pim.store` / `pim.copy`)의 위치**입니다. hoisting이 `pim.access`는 두고 `pim.load`만 loop 밖으로 옮길 수 있어서 둘의 위치가 다를 수 있습니다. 한 access를 여러 곳에서 전송하면 각각을 더합니다.

정확한 값을 얻으려면 중복 전송 제거가 끝난 뒤에 세어야 합니다. K loop에서 불변인 전송이 아직 안에 남아 있으면 실제보다 크게 세게 됩니다. 그래서 분석 패스는 변환 패스 뒤에 놓습니다 (§20.2).

`valid`는 SSA 값이라 컴파일 시점에 알 수 없으므로 `sizes` 전체를 셉니다. 따라서 경계 tile에서는 실제보다 크게 잡힌 상한입니다.

`pim.gather`와 `pim.scatter`는 `!pim.access`를 거치지 않으므로 이 계산에 들어가지 않습니다. 원소별 임의 접근이라 전송량이 index 텐서의 값에 달려 있어 컴파일 시점에 셀 수 없습니다.

```mlir
%dA = pim.access %A offset %o sizes [32, 16] strides [%c1, %c1] {
        pim.tile_bytes          = 2048 : i64,
        pim.transfer_count      = 32 : i64,
        pim.invariant_grid_dims = array<i64: 1>,
        pim.reuse_realizable    = false,
        pim.role                = "input"
      } : memref<*xf32> -> !pim.access
```

### 재사용 할인은 실현될 때만 얻는다

`Π{s_j : j ∈ I_a}`가 `Π{g_j}`보다 작다는 것은 "그 데이터를 유닛에 올려 두고 여러 instance가 재사용한다"는 뜻입니다. `pim.reuse_realizable`이 거짓이면 그 재사용이 일어나지 않으므로, 그 access는 불변 축에서도 `g_j`배로 세야 합니다 (§5.5).

```text
불변 축 j의 배수 = s_j   (pim.reuse_realizable = true)
                  g_j   (pim.reuse_realizable = false)
```

### 예시: GEMM

`C[4096,4096] = A[4096,4096] × B[4096,4096]`, f32, `BLOCK_M = BLOCK_N = BLOCK_K = 64`

grid는 m = 64, n = 64입니다. K loop는 64번 돕니다. 각 행렬은 64 MiB입니다.

| access | `tile_bytes` | `transfer_count` | instance당 | `I_a` |
| --- | ---: | ---: | ---: | --- |
| A | 16,384 | 64 | 1,048,576 | {n} |
| B | 16,384 | 64 | 1,048,576 | {m} |
| C | 16,384 | 1 | 16,384 | {} |

```text
A = 1,048,576 × g_m(64) × s_n = 67,108,864 × s_n
B = 1,048,576 × s_m × g_n(64) = 67,108,864 × s_m
C =    16,384 × g_m × g_n     = 67,108,864
```

| `s_m` | `s_n` | 유닛 수 | 전송량 | 하한 대비 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 192 MiB | 1.0배 |
| 8 | 8 | 64 | 1.06 GiB | 5.7배 |
| 64 | 32 | 2048 | 6.06 GiB | **32.3배** |

### 예시: GEMV

`y[4096] = A[4096,4096] · x[4096]`, f32, `BLOCK_M = 64, BLOCK_K = 256`

grid는 m = 64입니다. K loop는 16번 돕니다.

| access | `tile_bytes` | `transfer_count` | instance당 | `I_a` |
| --- | ---: | ---: | ---: | --- |
| A | 65,536 | 16 | 1,048,576 | {} |
| x | 1,024 | 16 | 16,384 | {m} |
| y | 256 | 1 | 256 | {} |

```text
A = 1,048,576 × g_m(64) = 64 MiB      s와 무관
x =    16,384 × s_m
y =       256 × g_m(64) = 16 KiB
```

| `s_m` | 유닛 수 | 전송량 | 하한 대비 |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 64.03 MiB | 1.000배 |
| 8 | 8 | 64.14 MiB | 1.002배 |
| 64 | 64 | 65.02 MiB | **1.015배** |

**유닛을 64배로 늘려도 전송이 1.5%만 늡니다.** 복제되는 것이 x 하나뿐이고 그 크기가 A의 1/4096이기 때문입니다. 같은 식이 GEMM에서는 32배, GEMV에서는 1.015배를 내놓습니다. 갈리는 지점은 복제 대상의 `tile_bytes`입니다.

---

## 16.3 결정은 Target IR이 한다

PIM Common은 후보 목록을 만들지 않습니다. 후보를 열거하려면 유닛 개수를 알아야 하고, 그것은 §1이 배제한 정보입니다. 대신 **곱하기 전의 항**을 내보내고 Target IR이 자기 수치를 넣습니다.

```mlir
func.func @matmul_kernel(...) attributes {
  pim.grid_dims    = ["m", "n"],
  pim.grid_extents = array<i64: 64, 64>,
  pim.split_axes   = [{axis = "m", source = "grid"},
                      {axis = "n", source = "grid"},
                      {axis = "k", combine = "add", source = "reduction"}],
  pim.co_resident  = [{access = 0 : i64, bytes = 16384 : i64, lifetime = "streaming"},
                      {access = 1 : i64, bytes = 16384 : i64, lifetime = "streaming"},
                      {bytes = 16384 : i64, lifetime = "persistent"}],
  pim.compute_ops  = 33554432 : i64
}
```

Target IR이 하는 일은 다음과 같습니다.

1. `pim.split_axes`로 쪼갤 수 있는 축을 추린다
2. 자기 유닛 개수 `U`를 만족하는 `s` 조합을 만든다
3. 유닛 메모리 용량을 넘는 조합을 버린다 (§16.6)
4. 남은 조합을 시간으로 정렬한다
5. 고른 `s`로 유닛별 데이터 영역과 전송·결합 코드를 만든다

4번의 시간 식은 다음과 같습니다. `BW`, `throughput`, `capacity`는 target 값입니다.

```text
T(s) =  전송량(s) / BW
      + pim.compute_ops × (Π g_d) / (U × throughput_unit)
      + 결합 연산 수(s) / throughput_combine

제약:  유닛당 상주량(s) ≤ capacity
```

`pim.split_axes`의 `combine`은 결합이 필요하다는 사실과 연산자만 말하고 결합을 어디서 하는지는 말하지 않습니다. 부분합을 호스트로 올려 더할 수도 있고, chip 안의 buffer로 모을 수도 있습니다. `throughput_combine`은 Target IR이 자기 경로에 맞는 값을 넣는 자리입니다.

**전송량 하나만으로는 후보를 고를 수 없습니다.** 전송량은 모든 `s_d`에 대해 단조 비감소이므로 그것만 정렬하면 항상 유닛 하나를 고르게 되는데, 그 설정은 연산 시간이 가장 길고 대개 유닛 메모리에도 들어가지 않습니다. 유닛을 늘리는 이득은 두 번째 항에 있습니다.

`M = N = K = 4096` f32 GEMM에서 유닛 하나는 A 64 MiB + B 64 MiB + C 64 MiB = 192 MiB를 담아야 합니다. **용량 제약이 유닛 수의 하한을 정하고, 전송량이 상한을 정하며, 최적점은 그 사이에서 연산 시간이 결정합니다.**

### grid 크기를 모를 때

`pim.grid_extents`는 grid loop 상한이 상수일 때만 답니다. `--pim-build-grid`가 세우는 상한은 triton-shared ABI의 `num_programs` 인자이고 (§5.3), 그 값은 launcher가 실행 시점에 넘깁니다. 커널 하나를 여러 shape에 재사용하기 때문입니다.

grid 크기를 몰라도 나머지 항은 전부 남습니다.

| 속성 | grid 크기가 필요한가 |
| --- | --- |
| `pim.tile_bytes` (access마다) | 아니요 |
| `pim.transfer_count` (access마다) | 아니요 (감싸는 loop 반복이 상수면) |
| `pim.invariant_grid_dims`, `pim.reuse_realizable` (access마다) | 아니요 |
| `pim.split_axes`, `pim.co_resident`, `pim.compute_ops` (func) | 아니요 |
| `pim.grid_extents` (func) | 예 |

`g_d`는 실행 시점에 `num_programs`로 알 수 있으므로, 전송량 식을 호스트에서 평가하거나 launch grid를 컴파일 시점에 특수화해 IR에 상수로 박을 수 있습니다. 어느 쪽을 쓸지는 Target IR이 정합니다.

---

## 16.4 일반적인 `BLOCK_K` loop

하나의 program instance가 하나의 C tile을 담당하고 K 방향을 반복하면서 누산기에 누적합니다.

```mlir
%result = scf.for %tk = %c0 to %grid_k step %c1
    iter_args(%acc = %initial) -> tensor<32x64xf32> {

  %dA = pim.access %A ...
  %dB = pim.access %B ...
  %a = pim.load %dA
  %b = pim.load %dB

  %prod = linalg.matmul ins(%a, %b) outs(%zero_init)
  %next = linalg.generic ins(%acc, %prod) outs(%acc) { arith.addf }

  scf.yield %next
} {pim.reduction_combine = "add"}
```

이 경우 reduction은 instance 하나 안에서 끝납니다.

```text
C_tile = 0
for k_tile:
    C_tile += A_tile × B_tile
```

`BLOCK_K`는 grid 차원을 만들지 않고 이 loop의 반복 횟수만 정합니다. 이 loop를 유닛에 쪼개지 않으면 reduction은 instance 하나 안에서 끝나고 유닛 밖 결합이 필요 없습니다.

쪼갤지 여부는 Target IR이 정하므로, PIM Common은 이 loop를 `pim.split_axes`에 `source = "reduction"`으로 올려 두기만 합니다 (§16.5).

---

## 16.5 reduction 축 분할 (Split-K)

### 언제 필요한가

grid가 작으면 유닛이 남습니다.

```text
M = 256, N = 256, K = 4096
BLOCK_M = 32, BLOCK_N = 64, BLOCK_K = 64
→ grid_m = 8, grid_n = 4  →  instance 32개
```

유닛이 2048개면 32개만 씁니다. 반면 K loop는 64번 돌고, 여기에 병렬성이 남아 있습니다. 그래서 grid 축뿐 아니라 **reduction loop도 분할 축으로 둡니다.** `pim.split_axes`의 `source = "reduction"` 항목이 그것입니다 (§5.5).

### 유닛에 나누는 방법

reduction 축을 `s_k`개 유닛에 쪼갠다는 것은 각 유닛이 K 구간의 `1/s_k`만 맡는다는 뜻입니다. 유닛 하나가 보는 것은 축소된 문제입니다.

```text
유닛 (·, ·, kj) 가 받는 것:  A[·, kj·Ku : (kj+1)·Ku],  B[kj·Ku : (kj+1)·Ku, ·]
유닛이 만드는 것:            C tile 전체 크기의 부분합
```

`Ku = K / s_k`입니다. 커널 몸체는 그대로이고 K 상한과 base 주소만 달라집니다. IR의 loop 구조를 바꾸지 않아도 되는 이유가 이것입니다.

### 합법성

두 조건이며 모두 IR에서 확인됩니다. 판단은 PIM Common이 합니다.

1. loop에 `pim.reduction_combine`이 붙어 있다 (결합 법칙이 성립하는 누적이다)
2. loop 몸체에 `pim.store` / `pim.scatter`가 없다 (반복마다 부수 효과가 없다)

만족하면 `pim.split_axes`에 `source = "reduction"` 항목으로 올라가며 `combine` 필드가 함께 붙습니다.

부동소수점 덧셈은 엄밀히는 결합 법칙이 성립하지 않으므로 이 분할은 결과를 바꿉니다. 알려진 trade-off이며 `s_k`가 클수록 커집니다.

### 비용

grid 축 분할과 규칙이 다릅니다. reduction 축은 `transfer_count` 안에 들어 있으므로 §16.2의 곱에 넣지 않고 따로 셉니다.

| | `s_k`에 대한 의존 |
| --- | --- |
| 입력 access (k에 의존) | **무관.** 유닛당 `transfer_count / s_k`이고 유닛이 `s_k`배라 상쇄됩니다 |
| 출력 access (k에 불변) | **정비례.** 유닛마다 완전한 크기의 부분합을 내보냅니다 |
| 결합 연산 수 | `(s_k − 1) × 출력 원소 수` |

`M = N = 256, K = 4096`, f32 GEMM입니다.

| | `s_k` = 1 (유닛 32개) | `s_k` = 64 (유닛 2048개) |
| --- | ---: | ---: |
| A 전송 | 16 MB | 16 MB |
| B 전송 | 32 MB | 32 MB |
| C 전송 | 0.25 MB | **16 MB** |
| 합계 | 48.25 MB | 64 MB |
| 유닛당 연산 | 16.8 MFLOP | 262 KFLOP |
| 유닛 밖 결합 | 없음 | 4.13 M 덧셈 |

```text
연산 시간 ∝ 1/s_k
출력 비용 ∝ s_k
```

**둘의 합이 최소가 되는 `s_k`가 존재하므로 최대로 키우는 것이 답이 아닙니다.**

### 출력 크기가 결론을 가른다

같은 규칙이 GEMV에서는 반대 결론을 냅니다. 출력이 작기 때문입니다.

`s_k = 32` 기준입니다.

| | 출력 크기 | 부분합 전송 | 결합 연산 수 | 전체 전송 대비 |
| --- | ---: | ---: | ---: | ---: |
| GEMM `M = N = 4096` | 64 MiB | 2 GiB | 520 M | 하한의 10배 |
| GEMV `M = 4096` | 16 KiB | 512 KiB | 127 K | 0.8% |

GEMV는 유닛 2048개를 `s_m = 64`, `s_k = 32`로 채워도 전송이 하한의 1.02배입니다. GEMM은 같은 유닛 수에서 32배입니다. `pim.split_axes`의 `combine` 표시는 두 경우에 똑같이 붙고, 갈리는 것은 출력 access의 `pim.tile_bytes`입니다. 그래서 `combine` 표시만으로는 부족하고 출력 크기가 함께 있어야 합니다.

---

## 16.6 유닛당 상주량과 연산량

### 상주량

한 순간에 유닛 메모리에 함께 있어야 하는 tile들을 목록으로 냅니다. 가장 깊이 있는 `pim.load`의 블록을 잡아, 그 블록의 `pim.load` 결과 tensor들과 그 블록이 `scf.for` 몸체라면 tensor 타입 iter_arg를 모읍니다. GEMM이면 A tile, B tile, 누산기가 됩니다.

```mlir
func.func @matmul_kernel(...) attributes {
  pim.co_resident = [{access = 0 : i64, bytes = 2048 : i64, lifetime = "streaming"},
                     {access = 1 : i64, bytes = 4096 : i64, lifetime = "streaming"},
                     {bytes = 8192 : i64, lifetime = "persistent"}]
}
```

`access`는 함수 안 `pim.access`의 순서 인덱스입니다. 누산기는 access를 거치지 않으므로 이 필드가 없습니다.

`pim.gather`의 결과도 목록에 듭니다. 전송량은 index 텐서의 값에 달려 컴파일 시점에 셀 수 없지만 (§16.2), 상주량은 결과 tensor의 타입에서 그대로 나옵니다. gather도 access를 거치지 않으므로 `access` 필드가 없습니다. 목록에서 하나라도 빠지면 Target IR이 용량을 실제보다 작게 보고 들어가지 않는 분할을 고르므로, 셀 수 있는 것은 전부 넣습니다.

`lifetime`은 loop 구조에서 나옵니다. reduction loop 안에서 매 반복 새로 읽히면 `streaming`, loop를 관통하면 `persistent`입니다. 버퍼 하나를 돌려쓸 수 있는지가 이것으로 갈립니다.

합계가 아니라 목록으로 두는 이유는 target의 메모리 계층이 하나가 아니기 때문입니다. 어느 tile을 빠른 쪽에 두고 어느 것을 스트리밍할지는 개별 크기를 알아야 정할 수 있습니다.

GEMV `BLOCK_M = 64, BLOCK_K = 256`, f32의 목록입니다.

| tile | bytes | lifetime |
| --- | ---: | --- |
| A | 65,536 | streaming |
| x | 1,024 | streaming |
| y (누산기) | 256 | persistent |

A tile 하나가 98%를 차지합니다. 합계만 보면 "용량 초과"로 끝나지만, 목록을 보면 `BLOCK_K`를 줄여 A tile만 작게 하고 x와 y는 그대로 두면 된다는 것이 보입니다.

`BLOCK_K`는 전송 총량에 영향을 주지 않습니다. instance 하나가 결국 읽는 A의 총량은 `BLOCK_M × K`로 고정이고, `BLOCK_K`는 그것을 몇 번에 나눠 읽느냐만 정합니다. `BLOCK_K`가 정하는 것은 상주량과 전송 건수입니다.

f32 GEMM 기준입니다.

| BLOCK M / N / K | 상주량 합 |
| --- | ---: |
| 32 / 64 / 16 | 14 KB |
| 64 / 64 / 64 | 48 KB |
| 128 / 128 / 64 | 128 KB |

용량 판단은 §1이 배제한 항목이므로 Target IR이 합니다. 다만 **계산은 tile 크기와 element 크기만 쓰므로 target과 무관합니다.**

### 연산량

instance 하나가 하는 산술 연산 횟수를 함수 속성으로 냅니다.

```mlir
func.func @matmul_kernel(...) attributes {
  pim.compute_ops = 2097152 : i64
}
```

`linalg.matmul`의 tile이 `[BM, BK] × [BK, BN]`이고 reduction loop가 `t`번 돌면 `2 · BM · BN · BK · t`입니다. tile shape과 loop 횟수만 쓰므로 target과 무관합니다.

이 값이 §16.3 시간 식의 두 번째 항입니다. 유닛 하나가 맡는 instance 수는 `(Π g_d) / U`이므로 유닛당 연산량은 `pim.compute_ops × (Π g_d) / U`이고, **유닛 수에 반비례합니다.** 전송량이 유닛 수에 비례해 느는 것과 반대 방향이라, 이 항이 없으면 분할의 이득이 식에 나타나지 않습니다.

`BLOCK`은 Python autotune 설정에서 온 값이라 PIM Common에 도착할 때 이미 IR에 상수로 박혀 있습니다. Target IR은 그것을 바꿀 수 없고 맞는지 확인만 할 수 있습니다. target에 맞춰 `BLOCK`을 고르려면 autotuning 쪽으로 정보를 돌려보내야 합니다 (§16.7).

---

## 16.7 BLOCK과 분할은 함께 탐색한다

`BLOCK`과 분할은 같은 목적(유닛을 채우는 것)을 서로 다른 방식으로 달성하므로 결합된 탐색 공간을 이룹니다.

`M = 256, N = 256, K = 4096`, 유닛 2048개인 경우입니다.

| BLOCK_M | BLOCK_N | grid | Split-K 없이 쓰는 유닛 | 2048개를 채우려면 |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 8 × 4 | 32 | S = 64 |
| 16 | 32 | 16 × 8 | 128 | S = 16 |
| 16 | 16 | 16 × 16 | 256 | S = 8 |

세 줄 다 2048개를 쓰지만 비용이 다릅니다. tile을 작게 하면 유닛당 연산 대비 전송 비율이 나빠지고, `s_k`를 키우면 출력 전송과 결합이 늘어납니다. **`BLOCK`을 먼저 고정하고 분할을 정하면 최적을 놓칩니다.**

§16.2의 전송량, §16.6의 상주량과 연산량은 실행하지 않고 계산되므로 컴파일 전에 후보를 걸러낼 수 있습니다.

1. `BLOCK` × 분할 조합을 나열한다
2. 상주량이 유닛 메모리를 넘는 조합을 버린다
3. §16.3의 `T(s)`로 순위를 매긴다
4. 상위 몇 개만 컴파일해 측정한다

2번과 3번을 나누는 것이 중요합니다. 2번은 **실행 가능 여부**이므로 넘으면 버립니다. 3번은 **추정**이므로 순위만 정하고 최종 판단은 4번의 측정에 맡깁니다. 전송량만으로 3번을 하면 항상 유닛 하나를 1등으로 고르는데, 그 설정은 2번에서 이미 걸러졌거나 연산 시간이 가장 깁니다 (§16.3).

`BLOCK`은 커널의 `tl.constexpr`이고 분할은 유닛 배치 파라미터이므로 둘 다 커널 호출 시점에 정해집니다. 두 축을 한 탐색 공간에 두면 2번과 3번을 컴파일 전에 수행하고 살아남은 조합만 컴파일할 수 있습니다.

---

# 17. Target별 연산 Legalization

PIM Common은 연산의 논리적 의미를 유지하고, 특정 하드웨어에서 지원되는지는 결정하지 않습니다.

예를 들어 다음 elementwise 연산이 있다고 가정합니다.

```mlir
%result = linalg.generic ... {
  ^bb0(%x: f32, %out: f32):
    %e = math.exp %x : f32
    linalg.yield %e : f32
}
```

UPMEM에서는 다음과 같이 처리할 수 있습니다.

```text
math.exp
→ DPU software implementation
→ 근사 함수
→ 범용 연산 루프
```

AiMX에서는 native descriptor가 없다면 다음 중 하나를 선택할 수 있습니다.

```text
math.exp
  ├─ add/mul 기반 다항식 근사
  ├─ Host fallback
  └─ 지원 불가능 오류
```

Target lowering에서 가능한 결과는 다음과 같습니다.

| 처리 방식          | 의미                        |
| -------------- | ------------------------- |
| Direct mapping | hardware primitive에 직접 매핑 |
| Decomposition  | 지원되는 기본 연산 조합으로 분해        |
| Approximation  | 근사식을 사용하여 구현              |
| Host fallback  | Host에서 연산 수행              |
| Compile error  | 지원 가능한 구현이 없음             |

예를 들어 AiMX가 add, sub, mul만 지원한다면 다음과 같은 변환을 고려할 수 있습니다.

```text
exp(x)
≈ 1 + x + x²/2 + x³/6
```

```text
a / b
= a × reciprocal(b)
```

reciprocal은 Newton–Raphson 반복으로 분해할 수 있습니다.

연산의 지원 집합과 정확도·성능 trade-off는 Target마다 다르므로 legalization은 Target lowering에서 수행합니다.

---

# 18. Fusion의 위치

연산 fusion은 Target IR에서 수행합니다.

예를 들어 다음 연산이 있다고 가정합니다.

```text
matmul
→ bias add
→ activation
```

UPMEM에서는 하나의 tasklet loop 안에서 세 연산을 연속 수행하는 것이 적절할 수 있습니다.

```text
load A/B
→ MAC loop
→ bias add
→ activation
→ store
```

AiMX에서는 각 연산이 별도의 descriptor나 opcode에 대응할 수 있습니다.

```text
matmul descriptor
→ bias descriptor
→ activation descriptor
```

또는 bias를 accumulator 초기값에 반영하는 Target 고유 최적화가 가능할 수 있습니다.

따라서 PIM Common에서는 다음 데이터 의존성만 유지합니다.

```text
%matmul_result
  → %bias_result
  → %activation_result
```

실제 descriptor 결합이나 loop fusion은 Target IR에서 결정합니다.

---

# 19. GEMM PIM Common IR 예시

`M = N = 1024, K = 512`, `BLOCK_M = 32, BLOCK_N = 64, BLOCK_K = 16`, 2차원 grid로 작성한 커널입니다. 파이프라인을 모두 통과한 뒤의 모습입니다.

```mlir
func.func @matmul_kernel(
    %A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
    %M: i32, %N: i32, %K: i32,
    %stride_am: i32, %stride_bk: i32, %stride_cm: i32,
    %np_x: i32, %np_y: i32, %np_z: i32)
    attributes {
      pim.execution_model = "spmd",
      pim.grid_dims       = ["m", "n"],
      pim.split_axes      = [{axis = "m", source = "grid"},
                             {axis = "n", source = "grid"},
                             {axis = "k", combine = "add", source = "reduction"}],
      pim.co_resident     = [{access = 0 : i64, bytes = 2048 : i64, lifetime = "streaming"},
                             {access = 1 : i64, bytes = 4096 : i64, lifetime = "streaming"},
                             {bytes = 8192 : i64, lifetime = "persistent"}],
      pim.compute_ops     = 2097152 : i64
    } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32

  %sam  = arith.index_cast %stride_am : i32 to index
  %sbk  = arith.index_cast %stride_bk : i32 to index
  %scm  = arith.index_cast %stride_cm : i32 to index
  %ub_m = arith.index_cast %np_x : i32 to index
  %ub_n = arith.index_cast %np_y : i32 to index

  scf.for %gm = %c0 to %ub_m step %c1 {
    scf.for %gn = %c0 to %ub_n step %c1 {

      // m_off = gm × BLOCK_M,  n_off = gn × BLOCK_N
      // valid_m, valid_n 은 minsi/maxsi/subi 연쇄로 계산 (경계 tile)

      %empty = tensor.empty() : tensor<32x64xf32>
      %init  = linalg.fill {pim.op = "fill"}
               ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>

      %result = scf.for %tk = %c0 to %grid_k step %c1
          iter_args(%acc = %init) -> tensor<32x64xf32> {

        // a_off = m_off × stride_am + k_off
        // b_off = k_off × stride_bk + n_off

        %dA = pim.access %A offset %a_off sizes [32, 16] strides [%sam, %c1]
              valid [%valid_m, %valid_k] pad %zero
              {pim.role = "input", pim.invariant_grid_dims = array<i64: 1>,
               pim.tile_bytes = 2048 : i64, pim.transfer_count = 32 : i64,
               pim.reuse_realizable = false}
              : memref<*xf32> -> !pim.access

        %dB = pim.access %B offset %b_off sizes [16, 64] strides [%sbk, %c1]
              valid [%valid_k, %valid_n] pad %zero
              {pim.role = "input", pim.invariant_grid_dims = array<i64: 0>,
               pim.tile_bytes = 4096 : i64, pim.transfer_count = 32 : i64,
               pim.reuse_realizable = false}
              : memref<*xf32> -> !pim.access

        %a = pim.load %dA : !pim.access -> tensor<32x16xf32>
        %b = pim.load %dB : !pim.access -> tensor<16x64xf32>

        %prod = linalg.matmul {pim.op = "gemm"}
                ins(%a, %b : tensor<32x16xf32>, tensor<16x64xf32>)
                outs(%init : tensor<32x64xf32>) -> tensor<32x64xf32>

        %next = linalg.generic {
                  indexing_maps = [#identity, #identity, #identity],
                  iterator_types = ["parallel", "parallel"],
                  pim.op = "elementwise"
                }
                ins(%acc, %prod : tensor<32x64xf32>, tensor<32x64xf32>)
                outs(%acc : tensor<32x64xf32>) {
          ^bb0(%x: f32, %y: f32, %out: f32):
            %s = arith.addf %x, %y : f32
            linalg.yield %s : f32
        } -> tensor<32x64xf32>

        scf.yield %next : tensor<32x64xf32>
      } {pim.reduction_combine = "add"}

      // c_off = m_off × stride_cm + n_off
      %dC = pim.access %C offset %c_off sizes [32, 64] strides [%scm, %c1]
            valid [%valid_m, %valid_n]
            {pim.role = "output", pim.invariant_grid_dims = array<i64>,
             pim.tile_bytes = 8192 : i64, pim.transfer_count = 1 : i64,
             pim.reuse_realizable = true}
            : memref<*xf32> -> !pim.access

      pim.store %result, %dC : tensor<32x64xf32>, !pim.access

    } {pim.grid_dim = "n"}
  } {pim.grid_dim = "m"}

  return
}
```

읽을 때 주의할 점 두 가지입니다.

* 주소 계산은 읽기 쉽게 주석으로 줄였습니다. 실제 IR에서 A와 B의 offset은 K loop의 `iter_args`로 옮겨 다니며 매 반복 증가합니다. 매번 `k_off`에서 새로 계산하지 않습니다.
* 누산은 `linalg.matmul`이 아니라 뒤따르는 `linalg.generic`이 합니다 (§5.6).

이 IR이 표현하는 정보입니다.

* grid 두 축과, 쪼갤 수 있는 축 세 개 (m, n, 그리고 결합이 필요한 k)
* A, B, C의 논리적 접근 범위와 경계 tile 유효 크기
* K 방향 반복과 누산, 그 결합 연산자
* A는 n축에, B는 m축에 불변 — 쪼갤 때 복제 대상
* A와 B의 주소가 K loop에 의존해 그 재사용이 실현되지 않는다는 사실
* 전송량 식의 항 (`tile_bytes`, `transfer_count`)
* 유닛 하나가 동시에 담아야 하는 tile 목록과 각각의 수명
* instance 하나의 연산량
* load–compute–store 데이터 의존성과 연산 종류

Target IR이 결정하는 정보입니다.

* 사용할 유닛 개수와 축별 분할 비율
* 유닛 인덱스에서 데이터 영역을 계산하는 방법
* MRAM, WRAM, bank 메모리 배치와 실제 buffer 크기
* B의 실제 residency
* DMA alignment, padding 형식, descriptor chain
* 부분합을 합치는 위치와 순서
* 연산 fusion과 legalization
* Host와 device의 역할 분담


# 20. Pass Pipeline

## 20.1 전체 흐름

```text
Triton Python
   ↓
TTIR
   ↓ --triton-to-linalg-experimental
TTShared IR
   ↓
--convert-ttshared-to-pim         memref 접근 패턴 raising, staging buffer 제거
   ↓
--pim-build-grid                  program ID → grid loop (§5.3)
   ↓
-cse                              동일한 pim.access 병합 (§15.1)
   ↓
-loop-invariant-code-motion       불변 주소 계산과 Pure한 pim.access를 loop 밖으로 (§15.6 선행)
   ↓
--pim-remove-redundant-transfers  instance 내부 hoisting, load-store를 copy로,
   │                              반복 load 제거, store-to-load forwarding (§15)
   ↓
-canonicalize                     죽은 pim.load 정리 (§15.5)
   ↓
--pim-analyze-reuse               role, invariant_grid_dims, reuse_realizable,
   │                              reduction_combine, split_axes (§5.5, §5.6)
   ↓
--pim-analyze-transfer            tile_bytes, transfer_count, co_resident,
   │                              compute_ops, grid_extents (§16.2, §16.6)
   ↓
--pim-classify-compute            gemm / reduce / elementwise 등 분류 (§13)
   ↓
PIM Common IR
   ├─ --legalize-pim-to-upmem
   └─ --legalize-pim-to-aimx
```

`compiler.py`의 `PIM_COMMON_PASSES`가 이 순서 그대로입니다.

## 20.2 순서의 근거

파이프라인은 **변환 패스를 전부 돌린 뒤 분석 패스를 전부 돌리는** 두 구간으로 나뉩니다. 분석 결과는 IR의 최종 모습을 기술해야 하고, IR을 바꾸는 패스가 뒤에 오면 그 기술이 낡기 때문입니다. 이 규칙에는 예외를 두지 않습니다.

**`--convert-ttshared-to-pim`이 `--pim-build-grid`보다 앞에 온다.** 둘 다 가능하지만, raise 패스의 matcher들이 지금 보는 IR 모양을 유지하는 편이 낫습니다. 이미 동작하는 패스를 흔들지 않습니다.

**`--pim-build-grid`가 나머지보다 앞에 온다.** 뒤의 패스들이 모두 "grid 축"이라는 개념 위에서 동작합니다. grid loop가 없으면 분석할 대상이 없습니다.

**`-loop-invariant-code-motion`이 `--pim-remove-redundant-transfers`보다 앞에 온다.** §15.6의 첫 조건이 "`pim.access`가 loop 밖에 있다"입니다. `pim.access`는 `Pure`라 표준 LICM이 빼 주므로, 그 뒤에 돌아야 load가 따라 나옵니다.

**분석 패스가 `--pim-remove-redundant-transfers`와 `-canonicalize` 뒤에 온다.** `pim.transfer_count`는 전송 op의 위치를 세므로 hoisting이 끝난 뒤라야 실제 전송 횟수와 맞습니다 (§16.2). `-canonicalize`가 죽은 access를 지운 뒤에 분석하면 사라질 op에 속성을 다는 일도 없습니다.

**`--pim-analyze-reuse`가 `--pim-analyze-transfer`보다 앞에 온다.** 전송량 식이 `pim.invariant_grid_dims`를 입력으로 씁니다.

**`--pim-classify-compute`가 마지막이다.** 분류 태그는 찍는 순간의 스냅샷이라 이후 구조가 바뀌면 낡습니다 (§13).

## 20.3 IR을 바꾸는 패스는 하나뿐이다

`--pim-remove-redundant-transfers`만 IR을 바꿉니다. 나머지는 전부 속성만 붙입니다.

분할도 IR 변환이 아닙니다. 유닛에 나누는 일은 유닛 인덱스에서 데이터 영역과 축소된 문제 크기를 계산하는 것이고, 커널 몸체는 그대로입니다 (§16.5). 그래서 분할을 반영하려고 grid loop를 쪼개거나 재배치할 필요가 없고, Target codegen이 전송·결합 코드를 만들 때 처리합니다.

---

# 21. 구현 현황과 범위

## 21.1 구현되어 있는 것

| 항목 | 위치 |
| --- | --- |
| `pim` dialect 6개 op + `!pim.access` + verifier | `include/triton-shared/Dialect/PIM/IR/PIMDialect.td` |
| `--convert-ttshared-to-pim` | `lib/Conversion/TTSharedToPIM/TTSharedToPIMPass.cpp` |
| `--pim-build-grid` | 같은 디렉터리 |
| `--pim-remove-redundant-transfers` (hoisting 포함) | 같은 디렉터리 |
| `--pim-analyze-reuse` | 같은 디렉터리 |
| `--pim-analyze-transfer` | 같은 디렉터리 |
| `--pim-classify-compute` | 같은 디렉터리 |
| 파이프라인 배선 | `pim_backend/backend/compiler.py` |
| 테스트 | `test/Conversion/TTSharedToPIM/` |

분석 패스가 다는 속성은 다음과 같습니다. 어느 것도 IR을 바꾸지 않고, Target IR이 읽을 근거만 남깁니다.

| 속성 | 붙는 곳 | 다는 패스 | 절 |
| --- | --- | --- | --- |
| `pim.grid_dim` | grid loop | `--pim-build-grid` | §5.3 |
| `pim.execution_model`, `pim.grid_dims` | func | `--pim-analyze-reuse` | §5.5 |
| `pim.role`, `pim.invariant_grid_dims` | `pim.access` | `--pim-analyze-reuse` | §5.5, §16.1 |
| `pim.reuse_realizable` | `pim.access` | `--pim-analyze-reuse` | §5.5, §15.6 |
| `pim.reduction_combine` | reduction loop | `--pim-analyze-reuse` | §5.6 |
| `pim.split_axes` | func | `--pim-analyze-reuse` | §5.5, §16.5 |
| `pim.tile_bytes`, `pim.transfer_count` | `pim.access` | `--pim-analyze-transfer` | §16.2 |
| `pim.co_resident` | func | `--pim-analyze-transfer` | §16.6 |
| `pim.compute_ops` | func | `--pim-analyze-transfer` | §16.6 |
| `pim.grid_extents` | func | `--pim-analyze-transfer` | §16.2 |
| `pim.op` | 계산 op | `--pim-classify-compute` | §13 |

커널이 다차원 grid로 작성되어 있어야 분할 후보가 둘 이상 나옵니다 (§5.4).

## 21.2 다음 단계로 미루는 것

**Host/device 분리.** grid loop가 곧 호스트 loop이므로 §5.3이 절반을 해 둔 셈입니다. 어디서 자를지는 전송이 어디로 옮겨지는지 본 뒤에 정합니다.

**`--legalize-pim-to-upmem` 재작성.** `compiler.py`가 호출하지만 구현이 없습니다. §16의 출력을 입력으로 받도록 새로 씁니다. `UPMEMToHost.cpp`의 `split_m` / `split_n` / `split_k` 모델은 §16.1과 같은 구조이므로 대부분 그대로 씁니다. 그 파일의 `A_DIST = SPLIT_M × SPLIT_K`가 §16.1의 "서로 다른 조각 수" `Π{s_j : j ∉ I_A}`이고, 같은 버퍼를 `SPLIT_N`개 유닛이 공유하는 것이 복제 배수 `Π{s_j : j ∈ I_A}`입니다. 지금은 그 값들을 사람이 정하고 있고, 새로 쓸 때는 §16의 속성에서 유도합니다.

맞춰야 할 것은 두 가지입니다.

* `extractGemmBlocks`가 memref 위의 `linalg.matmul`을 기대합니다. PIM Common은 tensor 위에 있으므로 사이에 bufferization이 필요합니다.
* GEMM 전용입니다. 그 외 커널을 어디로 보낼지 정해야 합니다. `pim.op`가 그 분기의 근거입니다 (§13).

**AiMX 경로.** `--legalize-pim-to-aimx` 등이 모두 stub입니다.

**PyTorch 그래프 레벨.** §16까지 안정된 뒤에 붙입니다.

커널 하나만 보면 놓치는 것이 두 가지 있습니다.

```python
y = torch.relu(x @ W1)
z = y @ W2
```

Inductor는 커널 둘을 만듭니다. 지금 구조라면 `y`가 유닛에서 호스트로 나갔다가 다시 들어옵니다. `y`가 1024×1024 f32면 왕복 8 MB입니다. 두 커널이 이어져 있다는 것을 알면 `y`는 유닛에 그대로 둘 수 있습니다. `W1`과 `W2`는 호출마다 값이 같으므로 한 번 상주시키면 이후 전송이 0입니다. 512×1024 f32면 호출당 2 MB입니다.

둘 다 단일 커널 IR에서는 보이지 않습니다. 지금 커널이 받는 것은 `memref<*xf32>`뿐이라, 커널 1의 `%A`와 커널 2의 `%B`가 같은 텐서인지 알 방법이 없습니다.

지금 넣어 둘 것은 자리뿐입니다. 함수 인자에 그래프 레벨 식별자를 붙여 두면 됩니다.

```mlir
func.func @matmul_kernel(
    %A: memref<*xf32> {pim.buffer_id = "W1", pim.buffer_kind = "weight"},
    %B: memref<*xf32> {pim.buffer_id = "t7", pim.buffer_kind = "temp"}, ...)
```

`weight`는 호출 사이에 값이 바뀌지 않는 버퍼, `temp`는 그래프 안에서만 사는 중간 결과입니다.

이 식별자는 §15.7의 alias 가정과도 이어집니다. 지금은 "커널 인자 memref들은 서로 겹치지 않는다"를 가정으로 두고 있는데, 그래프 레벨에서는 두 인자가 같은 텐서인지 실제로 확인할 수 있습니다.

**`matmul(outs=zero) + add(acc)` 정규화.** target 독립적이지만 전송 최적화보다 효과가 작습니다 (§5.6).

**1차원 grid 정규화.** grouped ordering을 알아보고 다차원으로 되돌리는 패스. Inductor 커널을 받을 때 필요합니다 (§5.4).

## 21.3 지원하지 않는 것

* 랩어라운드 접근 — 필요한 경계값이 memref 레벨에 남지 않습니다 (§2.1)
* 여러 kernel 사이의 데이터 residency
* 비동기 전송, 전송과 연산의 overlap
* 여러 PIM target의 동시 실행
* 유닛 간 직접 통신
* PIM Common 단계의 유닛 개수 결정과 분할 후보 선택
* 물리 메모리 layout, alignment, descriptor 생성
* Target별 fusion과 legalization
* 복잡한 collective communication

---

# 22. 최종 정의

PIM Common IR은 다음과 같이 정의합니다.

> PIM Common IR은 TTShared의 구조화된 IR을 입력으로 받아, 메모리 접근 관용구를 논리적 주소와 데이터 이동 연산으로 끌어올리고, runtime에 있던 grid를 명시적 loop로 옮겨 program instance 사이의 데이터 관계를 드러내며, 축을 유닛에 쪼갤 때 무엇이 나뉘고 무엇이 복제되며 어디서 결합이 필요한지를 기록해 Target IR에 넘기는 PIM-neutral 중간 계층이다.

역할 구분은 다음과 같습니다.

```text
PIM Common IR
  ├─ 논리적 데이터 접근과 load/store/copy
  ├─ grid loop와 instance 내부 loop
  ├─ linalg 계산 의미와 연산 종류
  ├─ grid 축별 불변성 — 나누는 데이터와 복제하는 데이터
  ├─ 재사용 할인이 실현되는지
  ├─ 쪼갤 수 있는 축과 결합 연산자
  ├─ 전송량·상주량·연산량 식의 항
  └─ target-neutral 중복 전송 제거

Target IR
  ├─ 유닛 개수와 유닛 메모리 용량
  ├─ 축별 분할 비율 선택
  ├─ 물리 메모리 배치와 용량 판정
  ├─ alignment와 padding
  ├─ 실제 residency
  ├─ descriptor와 명령어 생성
  ├─ Target별 legalization과 fusion
  └─ Host/device code generation
```

핵심 설계 원칙은 두 가지입니다.

> PIM Common IR은 어떤 데이터를 어떤 방식으로 접근하고 어떤 계산을 수행하는지를 표현하며, 실제 하드웨어에서 어디에 배치하고 어떻게 실행할지는 Target IR에 위임한다.

> 사실은 PIM Common이 기록하고, 수치는 Target IR이 넣는다. IR을 바꾸는 것은 중복 전송 제거 하나뿐이다.
