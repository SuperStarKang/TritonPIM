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
* 반복되는 데이터 접근과 중복 전송 분석
* 분할 후보별 전송량 계산
* reduction loop를 grid 축으로 승격하는 변환 (Split-K)
* 유닛 하나가 동시에 담아야 하는 working set 크기 계산
* GEMM, reduction, elementwise와 같은 연산 종류 표현
* Target lowering에서 활용할 데이터 의존성과 재사용 가능성 제공

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

### 구조와 파라미터의 분리

PIM Common과 Target IR의 경계를 정하는 원칙은 다음과 같습니다.

> 구조는 PIM Common이 만들고, 파라미터는 Target IR이 정한다.

유닛 분할을 예로 들면, "각 grid 축을 유닛 부분과 순차 부분으로 쪼갠다"는 구조와 "각 후보의 전송량이 얼마인가"라는 계산은 하드웨어 정보 없이 형태만으로 결정되므로 PIM Common이 합니다. "유닛을 몇 개 쓸 것인가"는 유닛 개수를 알아야 하므로 Target IR이 정합니다.

변환 코드는 PIM Common에 한 벌만 두고, Target IR은 인자를 주어 호출합니다. 같은 loop 변환을 target마다 다시 구현하면 공통 계층을 둔 의미가 없습니다.

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
| Generation  | `linalg.generic` + `linalg.index` |

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
                         %np_x: i32, %np_y: i32, %np_z: i32)
    attributes {pim.execution_model = "spmd"} {
  %ub_x = arith.index_cast %np_x : i32 to index
  scf.for %gx = %c0 to %ub_x step %c1 {
    %pid_x = arith.index_cast %gx : index to i32
    ...        // 몸체는 그대로
  } {pim.grid_dim = "x"}
  return
}
```

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

## 5.5 논리적 작업 분석

`--pim-analyze-logical-work`는 각 `pim.access`가 어느 grid 축에 의존하는지 분석합니다.

판별은 backward slice로 합니다. access의 주소를 결정하는 값들을 거슬러 올라가다 `pim.grid_dim`이 붙은 loop의 induction variable을 만나면 그 축에 의존한다고 봅니다. 속성이 없는 loop의 induction variable은 instance 내부의 일이므로 무시합니다. 추적이 불가능한 값을 만나면 보수적으로 모든 축에 의존한다고 봅니다.

### 추적 대상

접근 영역을 결정하는 것은 offset 하나가 아닙니다. 다음 네 가지를 모두 추적합니다.

| 피연산자 | 결정하는 것 |
| --- | --- |
| `offset` | 시작 위치 |
| `strides` | 축 간격 |
| `valid` | 경계 tile의 유효 범위 |
| `index` | row gather가 읽는 행 |

`pim.invariant_program_dims = [d]`의 뜻은 "d가 바뀌어도 같은 데이터 영역을 접근한다"입니다. 네 가지 중 하나라도 d에 의존하면 영역이 달라지므로 불변이 아닙니다.

`index`를 빠뜨리면 정확도가 아니라 정확성 문제가 됩니다. row gather에서 index tensor가 d에 의존하는데 offset은 아닌 경우, 불변이라고 표시하면 Target IR이 그것을 믿고 tile을 유닛에 상주시켜 잘못된 행을 읽습니다.

### 결과

```mlir
func.func @matmul_kernel(...) attributes {
  pim.execution_model = "spmd",
  pim.program_dims = ["x", "y"]
}
```

```mlir
%dA = pim.access %A ... {
  pim.role = "input",
  pim.invariant_program_dims = array<i64: 1>
}
```

`pim.program_dims`는 grid loop의 `pim.grid_dim` 속성을 바깥부터 모은 것입니다.

`pim.role`은 access를 소비하는 방식으로 정합니다. `pim.store`나 `pim.copy`의 dst면 `output`, 그 외면 `input`입니다.

이 속성들은 Target lowering과 디버깅을 위한 정보이며, 프로그램의 계산 의미를 바꾸지 않습니다. 분석이 보수적으로 나와도 성능만 손해이고 결과는 옳습니다.

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

### program instance 간 reduction

서로 다른 program instance가 같은 출력 영역에 부분합을 쓰는 경우는, 출력 access가 어떤 grid 축에 불변인지로 판별합니다.

출력 access가 축 d에 불변이면, d만 다른 instance들이 같은 출력을 접근하므로 그 결과를 합쳐야 합니다. 그 축들을 함수 속성으로 표시합니다.

```mlir
func.func @matmul_kernel(...) attributes {
  pim.execution_model = "spmd",
  pim.program_dims = ["x", "y", "k"],
  pim.reduce_across_programs = ["k"]   // k 축 instance 사이 부분합 결합 필요
}
```

이 상황은 저절로 생기지 않습니다. 일반적인 GEMM에서는 각 instance가 서로 다른 출력 tile을 만들므로 출력이 모든 축에 의존하고, 이 속성은 나타나지 않습니다. **§16.5의 Split-K 변환이 reduction loop를 grid 축으로 올렸을 때 나타납니다.**

이 표시는 "덮어쓰지 말고 합쳐야 한다"는 논리 사실이며, 합치는 위치와 방식은 Target IR이 결정합니다.

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

바이트가 필요한 계산(§16.2의 전송량, §16.6의 working set, Target IR의 DMA 크기)은 base memref에서 element type을 읽어 변환합니다.

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

Target lowering이 연산의 전체 구조를 쉽게 식별할 수 있도록 linalg 연산에 `pim.class` 속성을 붙입니다.

```text
--pim-classify-compute
```

분류 종류는 다음과 같습니다.

| `pim.class`   | 판별 기준                                    |
| ------------- | ---------------------------------------- |
| `gemm`        | `linalg.matmul`                          |
| `gemv`        | `linalg.matvec`                          |
| `transpose`   | `linalg.transpose`                       |
| `reduce`      | `linalg.reduce` 또는 reduction iterator    |
| `splat`       | `linalg.fill`                            |
| `generation`  | body에 `linalg.index` 존재                  |
| `broadcast`   | 입력 indexing map이 일부 차원을 제거               |
| `elementwise` | 모든 iterator가 parallel이고 identity mapping |
| `scan`        | 이전 원소 또는 index-carried 값에 의존             |

예시는 다음과 같습니다.

```mlir
%c = linalg.matmul
  ins(%a, %b)
  outs(%init)
  {pim.class = "gemm"}
```

```mlir
%result = linalg.generic
  {
    indexing_maps = [...],
    iterator_types = ["parallel"],
    pim.class = "elementwise"
  }
  ...
```

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

분석 결과는 속성으로 표현합니다. 인덱스는 `pim.program_dims`의 순서를 가리킵니다.

```mlir
%dA = pim.access %A ... {
  pim.role = "input",
  pim.invariant_program_dims = array<i64: 1>     // y 에 불변
}
```

```mlir
%dB = pim.access %B ... {
  pim.role = "input",
  pim.invariant_program_dims = array<i64: 0>     // x 에 불변
}
```

### 전제 조건

이 분석은 **grid 축이 서로 독립된 induction variable로 존재할 때만** 의미 있는 결과를 냅니다.

1차원 grid에 grouped ordering을 쓴 커널에서는 A, B, C의 주소가 모두 같은 `pid_x` 하나로 역추적됩니다. 그러면 `program_dims`가 축 하나뿐이고 모든 access의 `invariant_program_dims`가 빈 배열이 되어, 위 표의 정보가 나오지 않습니다. 분석이 틀리는 것이 아니라 **찾을 것이 없는 것**입니다.

따라서 §5.4의 규칙 — PIM용 커널은 다차원 grid로 작성한다 — 이 이 분석의 전제입니다.

### 이 속성이 확정하지 않는 것

실제 데이터 residency는 확정하지 않습니다. Target IR은 다음을 검사한 뒤 재사용 여부를 결정합니다.

* 같은 데이터를 쓰는 instance들이 같은 유닛에 배치되는가 (§16.1의 분할이 결정)
* 유닛 메모리에 데이터를 유지할 공간이 있는가 (§16.6의 working set)
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

`%d`가 `%i`에 의존하지 않고 loop 내부에 같은 영역을 쓰는 op이 없다면 다음처럼 옮길 수 있습니다.

```mlir
%v = pim.load %d

scf.for %i = ... {
  ...
}
```

instance 내부 loop에서 빼면 그 instance의 반복 전송이 줄고, grid loop에서 빼면 여러 instance가 공유하는 전송이 줄어듭니다. 후자는 §16.1의 분할 구조가 정해진 뒤에 가능합니다.

---

## 15.7 메모리 효과와 표준 패스

`pim.load`는 `MemRead` 효과를 선언합니다. 이 때문에 **§15.5와 §15.6은 표준 패스로 얻어지지 않습니다.**

* `-canonicalize`의 DCE는 `isMemoryEffectFree`인 op만 지웁니다. `MemRead`는 여기서 걸립니다.
* `-loop-invariant-code-motion`은 `isMemoryEffectFree(op) && isSpeculatable(op)`인 op만 옮깁니다. 역시 걸립니다.

따라서 두 최적화는 `--pim-remove-buffers`에 직접 구현합니다. 그 패스에는 이미 §15.2와 §15.4를 위한 쓰기 장벽 판정(`mayWrite`)이 있으므로 같은 안전성 논리를 씁니다.

반면 §15.1은 표준 `-cse`로 얻어집니다. `pim.access`는 `Pure`이므로 걸리지 않습니다.

### alias 가정

hoisting이 안전한지 판단하려면 loop 안의 쓰기가 읽는 영역과 겹치는지 알아야 합니다. 현재는 `pim.access`의 base memref가 다르면 겹치지 않는다고 봅니다.

**이것은 가정입니다.** Triton은 커널 포인터 인자가 서로 다른 버퍼임을 보장하지 않습니다. 제자리 연산에서는 같은 버퍼가 두 인자로 들어올 수 있습니다.

나중에 PyTorch 그래프 레벨을 붙이면 두 인자가 같은 텐서인지 실제로 확인할 수 있으므로, 이 가정을 검증된 사실로 바꿀 수 있습니다. §21의 `pim.buffer_id`가 그 자리입니다.

---

# 16. 작업 분할, K loop, Split-K

## 16.1 유닛 축과 순차 축

grid loop가 IR에 있으면 분할을 표현할 수 있습니다. 각 grid 축을 **유닛 부분과 순차 부분으로 쪼갭니다.**

```mlir
scf.for %um = 0 to SPLIT_M {                 // 유닛 사이. 동시 실행
  scf.for %un = 0 to SPLIT_N {               // 유닛 사이
    scf.for %sm = 0 to grid_m/SPLIT_M {      // 유닛 안. 순차 실행
      scf.for %sn = 0 to grid_n/SPLIT_N {    // 유닛 안
        // gm = %um × (grid_m/SPLIT_M) + %sm
        // gn = %un × (grid_n/SPLIT_N) + %sn
        body
      }
    }
  }
}
```

바깥 loop의 각 반복이 서로 다른 유닛이고, 안쪽 loop는 유닛 하나가 시간을 들여 도는 것입니다.

전송이 줄어드는 이유가 여기서 나옵니다. 바깥 변수에만 의존하는 값은 안쪽 loop가 도는 내내 같은 값이므로 안쪽 loop 밖으로 뺄 수 있습니다.

```mlir
scf.for %un = 0 to SPLIT_N {
  %dB = pim.access %B offset f(%un) ...
  %b  = pim.load %dB                    // 여기로 뺀다
  scf.for %sm = 0 to grid_m {
    %dA = pim.access %A offset g(%sm) ...
    %a  = pim.load %dA
    ...
  }
}
```

"B tile을 유닛에 한 번 올려놓고 재사용한다"는 문장이 IR에서는 "`pim.load`를 안쪽 loop 밖으로 뺀다"가 됩니다. 그래서 어느 축을 바깥에 두느냐가 어느 데이터를 아끼느냐를 결정합니다.

이 구조는 `UPMEMToHost.cpp`의 `split_m` / `split_n` / `split_k` 모델과 같습니다. 바깥 loop 개수가 `SPLIT_*`이고, 안쪽 loop와 몸체가 device 몸체가 됩니다.

---

## 16.2 전송량 계산

축 `d`의 grid 크기를 `g_d`, 유닛 수를 `s_d`, 유닛 안 순차 횟수를 `q_d = g_d / s_d`라 하고, 총 유닛 수를 `U = Π s_d`라 하면 전송량은 다음과 같습니다.

```text
전송량 = U × Σ_a  E_a × Π{ q_j : j ∉ I_a }
```

* `E_a` — instance 하나가 접근 `a`로 읽거나 쓰는 총 바이트
* `I_a` — `a`의 불변 축 집합 (§5.5)

곱 부분은 유닛 하나가 도는 동안 접근 `a`가 서로 다른 tile을 건드리는 횟수입니다. `a`가 축 `j`에 불변이면 `j`가 달라져도 같은 tile이므로 곱에서 빠집니다.

### `E_a`는 tile 크기가 아니다

`E_a`는 tile 바이트에 **instance 안에서 그 access가 실행되는 횟수**를 곱한 값입니다.

```text
E_a = (sizes 곱 × element 바이트) × (감싸는 grid 아닌 loop 들의 반복 횟수 곱)
```

`sizes`는 원소 개수이므로 바이트로 바꿔야 합니다. element type은 base memref에서 읽습니다 (§6.2).

K loop 안에 있는 access는 K loop 반복 횟수만큼 실행되므로 그 횟수를 곱해야 합니다. 이 곱을 빠뜨리면 값이 크게 어긋납니다.

정확한 값을 얻으려면 instance 내부의 hoisting이 끝난 뒤에 세어야 합니다. K loop에서 불변인 전송이 아직 안에 남아 있으면 실제보다 크게 세게 됩니다. 그래서 `--pim-partition`은 `--pim-remove-buffers` 뒤에 놓습니다.

### 예시

`C[1024,1024] = A[1024,512] × B[512,1024]`, f32, `BLOCK_M = 32, BLOCK_N = 64, BLOCK_K = 16`

grid는 m = 32, n = 16입니다. K loop는 32번 돕니다.

| access | `sizes` | tile | 반복 | `E_a` | `I_a` |
| --- | --- | ---: | ---: | ---: | --- |
| A | `[32, 16]` | 2 KB | 32 | 64 KB | {n} |
| B | `[16, 64]` | 4 KB | 32 | 128 KB | {m} |
| C | `[32, 64]` | 8 KB | 1 | 8 KB | {} |

| `s_m` | `s_n` | 유닛 수 | A | B | C | 합 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 16 | 32 MB | 2 MB | 4 MB | **38 MB** |
| 32 | 1 | 32 | 2 MB | 64 MB | 4 MB | **70 MB** |
| 32 | 16 | 512 | 32 MB | 64 MB | 4 MB | **100 MB** |

실제로 필요한 데이터는 A 2 MB + B 2 MB + C 4 MB = 8 MB입니다. 마지막 줄은 필요량의 12배를 전송합니다.

전송만 보면 첫 줄이 최선이지만 유닛을 16개만 씁니다. 마지막 줄은 512개를 쓰지만 전송이 100 MB입니다. **선택은 유닛 개수에 달려 있습니다.**

---

## 16.3 결정은 Target IR이 한다

`--pim-partition`은 후보별 전송량 표를 함수 속성으로 답니다. 어느 것을 고를지는 정하지 않습니다.

```mlir
func.func @matmul_kernel(...) attributes {
  pim.grid_extents    = array<i64: 32, 16>,
  pim.partition_cost  = [
    {splits = array<i64: 1, 16>,  bytes = 39845888 : i64},
    {splits = array<i64: 32, 1>,  bytes = 73400320 : i64},
    {splits = array<i64: 32, 16>, bytes = 104857600 : i64}
  ],
  pim.grid_permutable = true
}
```

Target IR이 하는 일은 다음과 같습니다.

1. 표와 자기 유닛 개수를 보고 후보를 고른다
2. `pim.grid_permutable`을 확인한다
3. grid loop를 쪼개고 재배치해 유닛 축을 바깥으로 보낸다
4. hoisting을 다시 돌린다
5. 바깥 loop를 유닛 dispatch로 바꾸고 전송 명령을 만든다

3번과 4번은 PIM Common의 패스를 인자만 주어 호출합니다. target마다 다시 구현하지 않습니다. target마다 다른 것은 1번과 5번뿐입니다.

`pim.grid_permutable`은 grid 반복의 순서를 바꿔도 되는지를 나타냅니다. 반복들이 서로 독립일 때만 합법이고, 독립이 깨지는 경우가 §16.5의 `pim.reduce_across_programs`입니다. 그 속성이 비어 있으면 참입니다.

### 표가 만들어지는 조건

표를 만들려면 grid 크기가 상수여야 합니다. `num_programs`가 runtime 값이면 접히지 않으므로 이 경우 표를 만들지 않고 넘어갑니다. 근거 없는 수를 IR에 넣는 것보다 낫습니다.

M, N, K가 컴파일 시점 상수로 특수화되면 `grid_m = ceildiv(M, BLOCK_M)`이 접혀 상수가 됩니다. autotuning은 config마다 다시 컴파일하고 PyTorch Inductor는 shape을 특수화하므로, 실제로는 상수인 경우가 기본이 됩니다.

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

instance 사이의 reduction은 필요하지 않습니다. `BLOCK_K`는 grid 차원을 만들지 않고 이 loop의 반복 횟수만 정합니다.

---

## 16.5 Split-K: reduction loop를 grid 축으로 승격

### 언제 필요한가

grid가 작으면 유닛이 남습니다.

```text
M = 256, N = 256, K = 4096
BLOCK_M = 32, BLOCK_N = 64, BLOCK_K = 64
→ grid_m = 8, grid_n = 4  →  instance 32개
```

유닛이 2048개면 32개만 씁니다. 반면 K loop는 64번 돌고, 여기에 병렬성이 남아 있습니다.

K는 grid 차원이 아니라 instance 안의 순차 loop입니다. 그래서 **순차 reduction loop를 잘라 새 grid 축으로 올려야** 합니다.

### 변환

`--pim-split-reduction=factor=S`가 K loop를 `S × (trip/S)`로 자르고 바깥 조각을 grid 축으로 올립니다.

**변환 전**

```mlir
scf.for %gm = 0 to 8 {
  scf.for %gn = 0 to 4 {
    %init = linalg.fill ins(%zero) outs(%empty)
    %acc = scf.for %k = 0 to 64 iter_args(%a = %init) {
      ...
      scf.yield %next
    } {pim.reduction_combine = "add"}
    pim.store %acc, %dC
  } {pim.grid_dim = "y"}
} {pim.grid_dim = "x"}
```

**변환 후 (S = 64)**

```mlir
scf.for %gm = 0 to 8 {
  scf.for %gn = 0 to 4 {
    scf.for %gk = 0 to 64 {                  // 새 grid 축
      %init = linalg.fill ins(%zero) outs(%empty)     // 안으로 들어옴
      %acc = scf.for %k = 0 to 1 iter_args(%a = %init) {
        %kk = arith.addi (arith.muli %gk, %c1), %k
        ...
        scf.yield %next
      } {pim.reduction_combine = "add"}
      pim.store %acc, %dC_partial            // %gk 에 무관한 주소
    } {pim.grid_dim = "k"}
  } {pim.grid_dim = "y"}
} {pim.grid_dim = "x"}
```

누산기 초기화가 새 loop **안**으로 들어가야 합니다. 각 `%gk`가 자기 부분합을 0에서 시작해야 하기 때문입니다.

### 기존 분석이 그대로 동작한다

변환 후 `%dC_partial`의 주소는 `%gm`, `%gn`에만 의존하고 `%gk`에는 의존하지 않습니다. §5.5의 분석이 자동으로 이렇게 답니다.

```mlir
%dC_partial = pim.access %C ... {
  pim.role = "output",
  pim.invariant_program_dims = array<i64: 2>      // k 에 불변
}

func.func @matmul_kernel(...) attributes {
  pim.reduce_across_programs = ["k"]
}
```

출력이 어떤 grid 축에 불변이면 그 축만 다른 instance들이 같은 출력에 쓰므로 합쳐야 한다는 뜻이고, 합치는 연산자는 K loop의 `pim.reduction_combine`이 알려 줍니다. **부분합 처리에 필요한 정보가 추가 작업 없이 나옵니다.**

### 합법성

세 조건이며 모두 IR에서 확인됩니다. 판단은 PIM Common이 합니다.

1. loop에 `pim.reduction_combine`이 붙어 있다 (결합 법칙이 성립하는 누적이다)
2. loop 몸체에 `pim.store` / `pim.scatter`가 없다 (반복마다 부수 효과가 없다)
3. 누산기 초기값이 `linalg.fill` 같은 상수 채움이다 (`%gk`마다 다시 만들 수 있다)

부동소수점 덧셈은 엄밀히는 결합 법칙이 성립하지 않으므로 Split-K는 결과를 바꿉니다. 알려진 trade-off이고, 현재 `UPMEMToHost.cpp`의 tree reduce도 같은 성질을 갖습니다.

### 비용

위 예제(`M = N = 256, K = 4096`, f32)의 값입니다.

| | S = 1 (유닛 32개) | S = 64 (유닛 2048개) |
| --- | ---: | ---: |
| A 전송 | 16 MB | 16 MB |
| B 전송 | 32 MB | 32 MB |
| C 전송 | 0.25 MB | **16 MB** |
| 합계 | 48.25 MB | 64 MB |
| 유닛당 연산 | 16.8 MFLOP | 262 KFLOP |
| 호스트 리덕션 | 없음 | 4.13 M 덧셈 |

**A와 B 전송은 S와 무관합니다.** 유닛 하나가 읽는 K 구간이 `1/S`로 줄고 유닛 수가 `S`배 늘어 정확히 상쇄됩니다.

**C 전송은 S에 정비례합니다.** 유닛마다 완전한 `BLOCK_M × BLOCK_N` 부분합을 만들어 내보내기 때문입니다. 여기에 `(S-1) × M × N`번의 리덕션이 붙습니다.

```text
연산 시간 ∝ 1/S
출력 비용 ∝ S
```

**둘의 합이 최소가 되는 S가 존재하므로 S를 최대로 키우는 것이 답이 아닙니다.** S를 고르는 것은 유닛 개수와 대역폭을 알아야 하므로 Target IR의 몫입니다. PIM Common은 합법성과 위 비용의 형태만 제공합니다.

---

## 16.6 working set

`BLOCK_K`는 전송 총량에 영향을 주지 않습니다. instance 하나가 결국 읽는 A의 총량은 `BLOCK_M × K`로 고정이고, `BLOCK_K`는 그것을 몇 번에 나눠 읽느냐만 정합니다.

`BLOCK_K`가 결정하는 것은 **working set**, 즉 한 순간에 유닛 메모리에 있어야 하는 양입니다.

```text
working set = (BLOCK_M × BLOCK_K)      A tile
            + (BLOCK_K × BLOCK_N)      B tile
            + (BLOCK_M × BLOCK_N)      누산기
```

f32 기준입니다.

| BLOCK M / N / K | working set |
| --- | ---: |
| 32 / 64 / 16 | 14 KB |
| 64 / 64 / 64 | 48 KB |
| 128 / 128 / 64 | 128 KB |

용량 판단은 §1이 배제한 항목이므로 Target IR이 합니다. 다만 **계산은 tile 크기와 element 크기만 쓰므로 target과 무관합니다.** PIM Common이 계산해 값으로 남기고, 들어가는지는 target이 판단합니다.

```mlir
func.func @matmul_kernel(...) attributes {
  pim.working_set_bytes = 14336 : i64
}
```

가장 안쪽 loop 몸체의 `pim.load` 결과 tensor들과 tensor 타입 iter_arg의 바이트 합입니다.

`BLOCK`을 autotuning으로 탐색하면 용량을 넘는 config를 반드시 만나게 됩니다. 이 값이 있으면 target이 그 자리에서 실패시킬 수 있고, autotuner가 해당 config를 건너뛸 근거가 됩니다.

`BLOCK`은 Python autotune 설정에서 온 값이라 PIM Common에 도착할 때 이미 IR에 상수로 박혀 있습니다. Target IR은 그것을 바꿀 수 없고 맞는지 확인만 할 수 있습니다. target에 맞춰 `BLOCK`을 고르려면 autotuning 쪽으로 정보를 돌려보내야 합니다.

---

## 16.7 BLOCK과 분할은 함께 탐색한다

`BLOCK`과 분할은 같은 목적(유닛을 채우는 것)을 서로 다른 방식으로 달성하므로 결합된 탐색 공간을 이룹니다.

`M = 256, N = 256, K = 4096`, 유닛 2048개인 경우입니다.

| BLOCK_M | BLOCK_N | grid | Split-K 없이 쓰는 유닛 | 2048개를 채우려면 |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 8 × 4 | 32 | S = 64 |
| 16 | 32 | 16 × 8 | 128 | S = 16 |
| 16 | 16 | 16 × 16 | 256 | S = 8 |

세 줄 다 2048개를 쓰지만 비용이 다릅니다. tile을 작게 하면 유닛당 연산 대비 전송 비율이 나빠지고, S를 키우면 출력 전송과 리덕션이 늘어납니다. **`BLOCK`을 먼저 고정하고 분할을 정하면 최적을 놓칩니다.**

§16.2의 전송량과 §16.6의 working set은 실행하지 않고 계산되므로, 실제로 컴파일하기 전에 후보를 걸러낼 수 있습니다.

1. `BLOCK` × 분할 조합을 나열한다
2. working set으로 거른다
3. 전송량으로 순위를 매긴다
4. 상위 몇 개만 컴파일해 측정한다

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
      pim.execution_model   = "spmd",
      pim.program_dims      = ["x", "y"],
      pim.grid_extents      = array<i64: 32, 16>,
      pim.grid_permutable   = true,
      pim.working_set_bytes = 14336 : i64,
      pim.partition_cost    = [
        {splits = array<i64: 1, 16>,  bytes = 39845888 : i64},
        {splits = array<i64: 32, 1>,  bytes = 73400320 : i64},
        {splits = array<i64: 32, 16>, bytes = 104857600 : i64}
      ]
    } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32

  %sam  = arith.index_cast %stride_am : i32 to index
  %sbk  = arith.index_cast %stride_bk : i32 to index
  %scm  = arith.index_cast %stride_cm : i32 to index
  %ub_x = arith.index_cast %np_x : i32 to index
  %ub_y = arith.index_cast %np_y : i32 to index

  scf.for %gx = %c0 to %ub_x step %c1 {
    scf.for %gy = %c0 to %ub_y step %c1 {

      // m_off = gx × BLOCK_M,  n_off = gy × BLOCK_N
      // valid_m, valid_n 은 minsi/maxsi/subi 연쇄로 계산 (경계 tile)

      %empty = tensor.empty() : tensor<32x64xf32>
      %init  = linalg.fill {pim.class = "splat"}
               ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>

      %result = scf.for %tk = %c0 to %grid_k step %c1
          iter_args(%acc = %init) -> tensor<32x64xf32> {

        // a_off = m_off × stride_am + k_off
        // b_off = k_off × stride_bk + n_off

        %dA = pim.access %A offset %a_off sizes [32, 16] strides [%sam, %c1]
              valid [%valid_m, %valid_k] pad %zero
              {pim.role = "input", pim.invariant_program_dims = array<i64: 1>}
              : memref<*xf32> -> !pim.access

        %dB = pim.access %B offset %b_off sizes [16, 64] strides [%sbk, %c1]
              valid [%valid_k, %valid_n] pad %zero
              {pim.role = "input", pim.invariant_program_dims = array<i64: 0>}
              : memref<*xf32> -> !pim.access

        %a = pim.load %dA : !pim.access -> tensor<32x16xf32>
        %b = pim.load %dB : !pim.access -> tensor<16x64xf32>

        %prod = linalg.matmul {pim.class = "gemm"}
                ins(%a, %b : tensor<32x16xf32>, tensor<16x64xf32>)
                outs(%init : tensor<32x64xf32>) -> tensor<32x64xf32>

        %next = linalg.generic {
                  indexing_maps = [#identity, #identity, #identity],
                  iterator_types = ["parallel", "parallel"],
                  pim.class = "elementwise"
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
            {pim.role = "output", pim.invariant_program_dims = array<i64>}
            : memref<*xf32> -> !pim.access

      pim.store %result, %dC : tensor<32x64xf32>, !pim.access

    } {pim.grid_dim = "y"}
  } {pim.grid_dim = "x"}

  return
}
```

읽을 때 주의할 점 두 가지입니다.

* 주소 계산은 읽기 쉽게 주석으로 줄였습니다. 실제 IR에서 A와 B의 offset은 K loop의 `iter_args`로 옮겨 다니며 매 반복 증가합니다. 매번 `k_off`에서 새로 계산하지 않습니다.
* 누산은 `linalg.matmul`이 아니라 뒤따르는 `linalg.generic`이 합니다 (§5.6).

이 IR이 표현하는 정보입니다.

* grid 두 축과 각 축의 크기
* A, B, C의 논리적 접근 범위와 경계 tile 유효 크기
* K 방향 반복과 누산, 그 결합 연산자
* A는 y축에, B는 x축에 불변 — 재사용 가능성
* 분할 후보 세 개와 각각의 전송량
* 유닛 하나가 담아야 하는 working set 크기
* load–compute–store 데이터 의존성과 연산 종류

Target IR이 결정하는 정보입니다.

* 사용할 유닛 개수와 분할 후보 선택
* Split-K 분할 개수
* grid loop 재배치와 유닛 dispatch
* MRAM, WRAM, bank 메모리 배치와 실제 buffer 크기
* B의 실제 residency
* DMA alignment, padding 형식, descriptor chain
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
--convert-ttshared-to-pim      memref 접근 패턴 raising, staging buffer 제거
   ↓
--pim-build-grid               program ID → grid loop (§5.3)
   ↓
-cse                           동일한 pim.access 병합 (§15.1)
   ↓
-canonicalize
   ↓
--pim-analyze-logical-work     role, invariant_program_dims,
   │                           reduction_combine, reduce_across_programs (§5.5, §5.6)
   ↓
--pim-remove-buffers           반복 load 제거, passthrough → pim.copy,
   │                           store-to-load forwarding, instance 내부 hoisting (§15)
   ↓
--pim-partition                분할 후보별 전송량, working set (§16.2, §16.6)
   ↓
--pim-classify-compute         gemm / reduce / elementwise 등 분류 (§13)
   ↓
PIM Common IR
   ├─ --legalize-pim-to-upmem
   └─ --legalize-pim-to-aimx
```

## 20.2 순서의 근거

**`--pim-build-grid`가 앞에 온다.** 뒤의 세 패스가 모두 "grid 축"이라는 개념 위에서 동작합니다. grid loop가 없으면 분석할 대상이 없습니다.

**`--convert-ttshared-to-pim`이 `--pim-build-grid`보다 앞에 온다.** 둘 다 가능하지만, raise 패스의 matcher들이 지금 보는 IR 모양을 유지하는 편이 낫습니다. 이미 동작하는 패스를 흔들지 않습니다.

**`--pim-analyze-logical-work`가 `--pim-partition`보다 앞에 온다.** 전송량 계산이 `pim.invariant_program_dims`를 입력으로 씁니다.

**`--pim-remove-buffers`가 `--pim-partition`보다 앞에 온다.** §16.2의 `E_a`는 instance 안에서 access가 실행되는 횟수를 곱한 값입니다. instance 내부 hoisting이 끝난 뒤라야 그 횟수가 실제 전송 횟수와 맞습니다.

**`--pim-classify-compute`가 마지막이다.** 분류 태그는 찍는 순간의 스냅샷이라 이후 구조가 바뀌면 낡습니다 (§13).

## 20.3 Target 쪽에서 호출하는 PIM Common 패스

다음 두 패스는 PIM Common에 두지만 파이프라인에 기본으로 넣지 않습니다. Target IR이 파라미터를 정한 뒤 호출합니다 (§16.3).

| 패스 | 파라미터 | 하는 일 |
| --- | --- | --- |
| `--pim-split-reduction` | `factor` | reduction loop를 grid 축으로 승격 (§16.5) |
| `--pim-apply-partition` | `splits` | grid 축을 유닛/순차로 쪼개고 재배치 |

재배치 후에는 `--pim-remove-buffers`를 다시 돌려 grid 수준의 전송을 뺍니다. 같은 패스를 재사용하므로 target마다 구현하지 않습니다.

---

# 21. 구현 현황과 범위

## 21.1 구현되어 있는 것

| 항목 | 위치 |
| --- | --- |
| `pim` dialect 6개 op + `!pim.access` + verifier | `include/triton-shared/Dialect/PIM/IR/PIMDialect.td` |
| `--convert-ttshared-to-pim` | `lib/Conversion/TTSharedToPIM/TTSharedToPIMPass.cpp` |
| `--pim-remove-buffers` | 같은 디렉터리 |
| `--pim-analyze-logical-work` | 같은 디렉터리 |
| `--pim-classify-compute` | 같은 디렉터리 |
| 파이프라인 배선 | `pim_backend/backend/compiler.py` |

## 21.2 이 문서에서 새로 정하는 것

| 항목 | 절 |
| --- | --- |
| `--pim-build-grid` | §5.3 |
| `pim.invariant_program_dims`의 추적 대상을 네 피연산자로 확장 | §5.5 |
| `--pim-partition` (전송량 표, working set) | §16.2, §16.6 |
| `--pim-split-reduction` | §16.5 |
| `--pim-remove-buffers`에 hoisting 추가 | §15.6, §15.7 |
| PIM용 커널은 다차원 grid로 작성 | §5.4 |

## 21.3 다음 단계로 미루는 것

**Host/device 분리.** grid loop가 곧 호스트 loop이므로 §5.3이 절반을 해 둔 셈입니다. 어디서 자를지는 전송이 어디로 옮겨지는지 본 뒤에 정합니다.

**`--legalize-pim-to-upmem` 재작성.** `compiler.py`가 호출하지만 구현이 없습니다. §16의 출력을 입력으로 받도록 새로 씁니다. `UPMEMToHost.cpp`의 `split_m` / `split_n` / `split_k` 모델은 §16.1과 같은 구조이므로 대부분 그대로 씁니다. 다만 두 가지를 맞춰야 합니다.

* `extractGemmBlocks`가 memref 위의 `linalg.matmul`을 기대합니다. PIM Common은 tensor 위에 있으므로 사이에 bufferization이 필요합니다.
* GEMM 전용입니다. 그 외 커널을 어디로 보낼지 정해야 합니다.

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

## 21.4 지원하지 않는 것

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

> PIM Common IR은 TTShared의 구조화된 IR을 입력으로 받아, 메모리 접근 관용구를 논리적 주소와 데이터 이동 연산으로 끌어올리고, runtime에 있던 grid를 명시적 loop로 옮겨 program instance 사이의 데이터 관계를 드러내며, 분할 후보별 전송량과 working set을 계산해 Target IR에 넘기는 PIM-neutral 중간 계층이다.

역할 구분은 다음과 같습니다.

```text
PIM Common IR
  ├─ 논리적 데이터 접근과 load/store/copy
  ├─ grid loop와 instance 내부 loop
  ├─ linalg 계산 의미
  ├─ grid 축별 불변성 (재사용 가능성)
  ├─ reduction 구조와 결합 연산자
  ├─ 분할 후보별 전송량
  ├─ working set 크기
  ├─ loop 변환 코드 (분할 적용, reduction 승격)
  └─ target-neutral 중복 제거

Target IR
  ├─ 유닛 개수
  ├─ 분할 후보 선택과 Split-K 분할 개수
  ├─ 물리 메모리 배치와 용량 판정
  ├─ alignment와 padding
  ├─ 실제 residency
  ├─ descriptor와 명령어 생성
  ├─ Target별 legalization과 fusion
  └─ Host/device code generation
```

핵심 설계 원칙은 두 가지입니다.

> PIM Common IR은 어떤 데이터를 어떤 방식으로 접근하고 어떤 계산을 수행하는지를 표현하며, 실제 하드웨어에서 어디에 배치하고 어떻게 실행할지는 Target IR에 위임한다.

> 구조는 PIM Common이 만들고, 파라미터는 Target IR이 정한다. 변환 코드는 한 벌만 둔다.
