# TTShared IR 이후 PIM Common IR 설계

## 1. 설계 목표

PIM Common IR은 TTShared의 구조화된 IR을 PIM에서 실행 가능한 형태로 표현하되, 특정 PIM 하드웨어에 종속되지 않는 중간 계층입니다.

PIM Common IR이 담당하는 범위는 다음과 같습니다.

* PIM에서 실행되는 논리적 작업 단위 보존
* 입력과 출력 데이터의 논리적 접근 영역 표현
* 외부 메모리와 PIM 계산 영역 사이의 데이터 이동 표현
* 반복되는 데이터 접근과 중복 전송 분석
* GEMM, reduction, elementwise와 같은 연산 종류 표현
* Target lowering에서 활용할 데이터 의존성과 재사용 가능성 제공

다음 정보는 PIM Common IR에 포함하지 않습니다.

* DPU 개수
* AiMX Card 개수
* AiM chip과 bank 개수
* MRAM, WRAM, global buffer와 같은 물리 메모리 계층
* 실제 tile-to-device 매핑
* alignment와 padding 방식
* DMA descriptor 및 명령어 형식
* 실제 데이터 residency
* Target별 연산 fusion과 decomposition

이러한 정보는 UPMEM IR이나 AiMX IR과 같은 Target IR에서 결정합니다.

---

# 2. 전체 변환 구조

PIM Common IR은 `--triton-to-linalg-experimental`을 통과한 구조화된 IR을 입력으로 사용합니다.

```text
Triton IR
  ↓
--triton-to-linalg-experimental
  ↓
TTShared structured IR
  ├─ memref
  ├─ tensor
  ├─ linalg
  ├─ scf
  ├─ arith
  └─ math
  ↓
--convert-ttshared-to-pim
  ↓
PIM Common IR
  ├─ pim.access
  ├─ pim.load
  ├─ pim.store
  ├─ pim.copy
  ├─ pim.gather
  ├─ pim.scatter
  ├─ linalg
  ├─ scf
  ├─ tensor
  ├─ arith
  └─ math
```

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

---

# 3. PIM Common IR의 구성

PIM Common IR은 크게 세 부분으로 구성됩니다.

```text
PIM Common IR
 ├─ 실행 구조
 │    ├─ func.func
 │    ├─ program ID를 나타내는 함수 인자
 │    └─ scf.for
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

> SPMD 실행 구조와 tile loop는 유지하고, 복잡한 memref 접근 패턴은 논리적 주소와 전송 연산으로 단순화하며, 계산 의미는 linalg에 보존한다.

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

# 5. 논리적 tile 작업 표현

## 5.1 Program ID 기반 SPMD 구조

Triton에서 하나의 program instance는 하나의 논리적 tile 작업을 수행합니다.

PIM Common IR에서는 이 구조를 함수 인자와 offset 계산식으로 유지합니다.

```mlir
func.func @matmul(
    %A: memref<*xbf16>,
    %B: memref<*xbf16>,
    %C: memref<*xf32>,
    %pid_m: index,
    %pid_n: index,
    %M: index,
    %N: index,
    %K: index) {
  ...
}
```

각 program ID의 의미는 다음과 같습니다.

```text
%pid_m: 출력 C의 M 방향 tile 좌표
%pid_n: 출력 C의 N 방향 tile 좌표
```

실제 데이터 위치는 program ID와 block size를 이용하여 계산합니다.

```mlir
%m_offset = arith.muli %pid_m, %BLOCK_M
%n_offset = arith.muli %pid_n, %BLOCK_N
```

이 구조에는 특정 DPU나 AiMX Card에 대한 정보가 포함되지 않습니다.

Target IR은 각 program instance를 실제 하드웨어 실행 단위에 배치합니다.

```text
UPMEM:
program instance → DPU

AiMX:
program instance → Card → chip → bank
```

---

## 5.2 논리적 작업 분석

PIM Common 단계에서 다음 정보를 분석할 수 있습니다.

* program ID 차원 수
* program ID별 출력 영역
* 서로 다른 program instance의 출력 영역 중첩 여부
* 입력 access가 어떤 program ID에 의존하는지
* 내부 reduction loop가 존재하는지
* program instance 간 reduction이 필요한지
* 특정 program 차원에서 동일한 데이터가 반복 사용되는지

이를 위한 분석 패스를 다음과 같이 구성할 수 있습니다.

```text
PIMLogicalWorkAnalysis
```

분석 결과는 필요에 따라 속성으로 표현할 수 있습니다.

```mlir
func.func @matmul(...) attributes {
  pim.execution_model = "spmd",
  pim.program_dims = ["m", "n"]
}
```

이 속성은 Target lowering과 디버깅을 지원하는 정보이며, 프로그램의 계산 의미를 변경하지 않습니다.

---

## 5.3 reduction 구조 분석

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

이 표시는 Target IR이 loop의 reduction 축을 여러 장치에 나눌 때, 각 장치의 부분합을 어떤 연산으로 합칠지 알려 줍니다.

### program instance 간 reduction

서로 다른 program instance가 같은 출력 영역에 부분합을 쓰는 경우(Split-K 등)는, 출력 access가 어떤 program 차원에 불변인지로 판별합니다.

출력 access가 program 차원 d에 불변이면, d만 다른 program instance들이 같은 출력을 접근하므로 그 결과를 합쳐야 합니다. 이때 그 차원들을 함수 속성으로 표시합니다.

```mlir
func.func @matmul(...) attributes {
  pim.execution_model = "spmd",
  pim.program_dims = ["m", "n", "k"],
  pim.reduce_across_programs = ["k"]   // k 차원 program 사이 부분합 결합 필요
}
```

일반적인 GEMM에서는 각 program이 서로 다른 출력 tile을 생성하므로(출력이 모든 program 차원에 의존) 이 속성은 나타나지 않습니다. 출력이 겹치는 경우에만 표시됩니다.

이 표시는 "덮어쓰지 말고 합쳐야 한다"는 논리 사실이며, 합치는 위치와 방식은 Target IR이 결정합니다.

---

# 6. `pim.access`

## 6.1 기본 역할

`pim.access`는 원본 memref의 특정 영역을 가리키는 논리적 주소 descriptor입니다.

```mlir
%dA = pim.access %A
  offset %a_offset
  sizes array<i64: 32, 256>
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
* access의 입력, weight, output 역할
* program ID에 대한 접근식 의존 관계

`pim.access` 자체는 데이터를 이동하지 않는 Pure 연산입니다.

실제 물리 주소 계산과 메모리 배치는 Target IR에서 결정합니다.

---

## 6.2 Offset 표현

현재 구조에서는 하나의 선형 offset과 축별 stride를 사용합니다.

```mlir
%d = pim.access %base
  offset %linear_offset
  sizes array<i64: 32, 256>
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

## 6.3 Masked access

경계 tile에서는 tile 전체 크기와 실제 유효한 크기가 다를 수 있습니다.

```mlir
%d = pim.access %A
  offset %offset
  sizes array<i64: 128>
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
  sizes array<i64: 128>
  strides [%c1]
  valid [%valid]
  : memref<*xf32> -> !pim.access
```

---

## 6.4 Row gather

하나의 축은 index tensor로 선택하지만, 각 선택 영역 내부가 연속적이거나 strided한 경우 `pim.access`의 gather clause를 사용합니다.

```mlir
%d = pim.access %A
  offset %c0
  sizes array<i64: 8, 64>
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

| 접근 패턴              | TTShared structured IR            | PIM Common IR             |
| ------------------ | --------------------------------- | ------------------------- |
| Contiguous/strided | `reinterpret_cast + alloc + copy` | `pim.access + pim.load`   |
| Loop-carried       | 위 구조가 `scf.for` 내부에 존재            | loop 유지, 내부 접근만 변환        |
| Block pointer      | memref 단계에서 strided 접근으로 표현       | `pim.access`              |
| Masked tile        | `fill + subview + copy`           | `pim.access {valid, pad}` |
| Row gather         | 행별 loop와 reinterpret cast         | `pim.access {gather}`     |
| Element gather     | `linalg.generic + tensor.extract` | `pim.gather`              |
| Element scatter    | `linalg.generic + memref.store`   | `pim.scatter`             |
| Wrap-around        | 두 view와 두 번의 copy                 | 추후 지원                     |

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
  sizes array<i64: 32, 64>
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

데이터 재사용 가능성은 `pim.access`의 offset 식이 어떤 program ID에 의존하는지 분석하여 찾습니다.

GEMM의 접근 관계는 다음과 같습니다.

```text
A: (pid_m, k)
B: (k, pid_n)
C: (pid_m, pid_n)
```

보다 구체적으로는 다음과 같습니다.

```text
A offset
  = pid_m × BLOCK_M × stride_am
  + k_offset × stride_ak

B offset
  = k_offset × stride_bk
  + pid_n × BLOCK_N × stride_bn

C offset
  = pid_m × BLOCK_M × stride_cm
  + pid_n × BLOCK_N × stride_cn
```

각 access의 program ID 의존성은 다음과 같습니다.

| 데이터 | `pid_m` 의존 | `pid_n` 의존 | 재사용 가능성                  |
| --- | ---------: | ---------: | ------------------------ |
| A   |          O |          X | N 방향 program 사이에서 재사용 가능 |
| B   |          X |          O | M 방향 program 사이에서 재사용 가능 |
| C   |          O |          O | 각 program이 서로 다른 출력 생성   |

분석 결과는 선택적인 속성으로 표현할 수 있습니다.

```mlir
%dA = pim.access %A ...
  {
    pim.role = "input",
    pim.invariant_program_dims = [1]
  }
```

```mlir
%dB = pim.access %B ...
  {
    pim.role = "weight",
    pim.invariant_program_dims = [0]
  }
```

각 속성의 의미는 다음과 같습니다.

```text
invariant_program_dims = [0]
→ program dimension 0이 바뀌어도 같은 데이터 영역을 접근함

invariant_program_dims = [1]
→ program dimension 1이 바뀌어도 같은 데이터 영역을 접근함
```

이 속성은 실제 데이터 residency를 확정하지 않습니다.

Target IR은 다음 조건을 검사한 뒤 재사용 여부를 결정합니다.

* 같은 데이터를 사용하는 program들이 같은 물리 장치에 배치되는가
* 장치 메모리에 데이터를 유지할 공간이 있는가
* 데이터를 유지하는 것이 재전송보다 유리한가
* 실행 순서를 재사용에 맞게 조정할 수 있는가
* 여러 장치에 broadcast하는 것이 더 효율적인가

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

`%d`가 `%i`에 의존하지 않고 loop 내부에 aliasing write가 없다면 다음처럼 이동할 수 있습니다.

```mlir
%v = pim.load %d

scf.for %i = ... {
  ...
}
```

이는 하나의 program 내부에서 반복되는 전송을 줄이는 최적화입니다.

서로 다른 program instance 사이에서의 공유는 실제 장치 배치가 필요하므로 Target IR에서 결정합니다.

---

# 16. 일반 K loop와 Split-K

## 16.1 일반적인 `BLOCK_K` loop

일반적인 Triton GEMM에서는 하나의 program instance가 하나의 C tile을 담당하고, K 방향을 반복하면서 동일한 accumulator에 누적합니다.

```mlir
%result = scf.for %tk = %c0 to %grid_k step %c1
    iter_args(%acc = %initial)
    -> tensor<32x64xf32> {

  %dA = pim.access %A ...
  %dB = pim.access %B ...

  %a = pim.load %dA
  %b = pim.load %dB

  %next = linalg.matmul
    ins(%a, %b)
    outs(%acc)
    {pim.class = "gemm"}

  scf.yield %next
}
```

이 경우 reduction은 하나의 program 내부에서 이루어집니다.

```text
C_tile = 0

for k_tile:
    C_tile += A_tile × B_tile
```

별도의 program 간 reduction은 필요하지 않습니다.

---

## 16.2 Split-K

Split-K에서는 여러 program instance가 같은 C tile에 대한 부분합을 생성합니다.

```text
program 0:
C_partial_0 = A[:, K0] × B[K0, :]

program 1:
C_partial_1 = A[:, K1] × B[K1, :]

program 2:
C_partial_2 = A[:, K2] × B[K2, :]
```

최종 결과는 다음과 같습니다.

```text
C = C_partial_0
  + C_partial_1
  + C_partial_2
```

PIM Common에서는 다음 두 정보를 구분해야 합니다.

```text
내부 K loop
→ 하나의 program 내부 누적

Split-K
→ 서로 다른 program 사이의 partial result 결합
```

이 두 경우는 §5.3의 분석으로 구분합니다. 내부 K loop는 reduction loop에 `pim.reduction_combine`으로, Split-K는 출력이 특정 program 차원에 불변인지로 판별해 `pim.reduce_across_programs`로 표시합니다.

초기 구현에서는 Split-K를 다음과 같이 표현할 수 있습니다.

* 각 program이 별도의 partial result 영역에 store
* 별도의 `linalg.reduce` 또는 reduction kernel 생성
* reduction 위치는 Target IR에서 결정

향후 Split-K를 자주 활용하고 Target별 reduction 구현을 명시적으로 제어해야 한다면 별도의 `pim.reduce`를 추가할 수 있습니다.

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

```mlir
func.func @matmul(
    %A: memref<*xbf16>,
    %B: memref<*xbf16>,
    %C: memref<*xf32>,
    %pid_m: index,
    %pid_n: index,
    %M: index,
    %N: index,
    %K: index) attributes {
  pim.execution_model = "spmd",
  pim.program_dims = ["m", "n"]
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %zero_bf16 = arith.constant 0.0 : bf16

  %m_offset = arith.muli %pid_m, %BLOCK_M
  %n_offset = arith.muli %pid_n, %BLOCK_N

  %m_remaining = arith.subi %M, %m_offset
  %n_remaining = arith.subi %N, %n_offset

  %valid_m = arith.minsi %BLOCK_M, %m_remaining
  %valid_n = arith.minsi %BLOCK_N, %n_remaining

  %empty = tensor.empty() : tensor<32x64xf32>

  %initial = linalg.fill
    ins(%zero : f32)
    outs(%empty : tensor<32x64xf32>)
    -> tensor<32x64xf32>

  %grid_k = arith.ceildivsi %K, %BLOCK_K

  %result = scf.for %tk = %c0 to %grid_k step %c1
      iter_args(%acc = %initial)
      -> tensor<32x64xf32> {

    %k_offset = arith.muli %tk, %BLOCK_K
    %k_remaining = arith.subi %K, %k_offset
    %valid_k = arith.minsi %BLOCK_K, %k_remaining

    %a_row_offset =
      arith.muli %m_offset, %stride_am

    %a_k_offset =
      arith.muli %k_offset, %stride_ak

    %a_offset =
      arith.addi %a_row_offset, %a_k_offset

    %b_k_offset =
      arith.muli %k_offset, %stride_bk

    %b_col_offset =
      arith.muli %n_offset, %stride_bn

    %b_offset =
      arith.addi %b_k_offset, %b_col_offset

    %dA = pim.access %A
      offset %a_offset
      sizes array<i64: 32, 256>
      strides [%stride_am, %stride_ak]
      valid [%valid_m, %valid_k]
      pad %zero_bf16
      {
        pim.role = "input",
        pim.invariant_program_dims = [1]
      }
      : memref<*xbf16> -> !pim.access

    %dB = pim.access %B
      offset %b_offset
      sizes array<i64: 256, 64>
      strides [%stride_bk, %stride_bn]
      valid [%valid_k, %valid_n]
      pad %zero_bf16
      {
        pim.role = "weight",
        pim.invariant_program_dims = [0]
      }
      : memref<*xbf16> -> !pim.access

    %a = pim.load %dA
      : !pim.access -> tensor<32x256xbf16>

    %b = pim.load %dB
      : !pim.access -> tensor<256x64xbf16>

    %next = linalg.matmul
      ins(%a, %b
        : tensor<32x256xbf16>,
          tensor<256x64xbf16>)
      outs(%acc : tensor<32x64xf32>)
      {pim.class = "gemm"}
      -> tensor<32x64xf32>

    scf.yield %next : tensor<32x64xf32>
  }

  %c_row_offset =
    arith.muli %m_offset, %stride_cm

  %c_col_offset =
    arith.muli %n_offset, %stride_cn

  %c_offset =
    arith.addi %c_row_offset, %c_col_offset

  %dC = pim.access %C
    offset %c_offset
    sizes array<i64: 32, 64>
    strides [%stride_cm, %stride_cn]
    valid [%valid_m, %valid_n]
    {
      pim.role = "output"
    }
    : memref<*xf32> -> !pim.access

  pim.store %result, %dC
    : tensor<32x64xf32>, !pim.access

  return
}
```

이 IR은 다음 정보를 표현합니다.

* `%pid_m`, `%pid_n` 기반의 논리적 output tile
* `BLOCK_M`, `BLOCK_N`, `BLOCK_K`
* A, B, C의 논리적 접근 범위
* 경계 tile의 유효 크기
* K 방향 반복과 accumulator
* A의 N 방향 재사용 가능성
* B의 M 방향 재사용 가능성
* load–compute–store 데이터 의존성
* GEMM 연산 종류

다음 정보는 Target IR에서 결정합니다.

* 사용할 PIM 장치 수
* program instance의 실제 장치 매핑
* MRAM, WRAM, bank 메모리 배치
* 실제 local buffer 크기
* B의 실제 residency
* DMA alignment
* padding 형식
* descriptor chain
* 연산 fusion
* Host와 device의 역할 분담

---

# 20. 권장 Pass Pipeline

```text
Triton IR
   ↓
--triton-to-linalg-experimental
   ↓
TTShared structured IR
   │
   │ memref + tensor + linalg + scf
   ↓
--convert-ttshared-to-pim
   │
   │ memref 접근 패턴 raising
   │ pim.access/load/store/gather/scatter 생성
   │ staging buffer 제거
   │ linalg 연산 유지
   ↓
-cse
   │
   │ 동일한 pim.access 병합
   ↓
--pim-remove-buffers
   │
   │ 반복 load 제거
   │ passthrough → pim.copy
   │ store-to-load forwarding
   ↓
-canonicalize
   │
   │ 사용되지 않는 연산과 값 제거
   ↓
-loop-invariant-code-motion
   │
   │ 안전한 loop-invariant 접근과 연산 이동
   ↓
--pim-analyze-logical-work
   │
   │ program ID와 access offset 관계 분석
   │ 독립 작업과 중첩 작업 구분
   │ 재사용 가능한 program dimension 분석
   │ program 간 reduction 필요 여부 분석
   ↓
--pim-classify-compute
   │
   │ gemm / gemv / reduce / broadcast
   │ elementwise / generation 등 분류
   ↓
Optimized PIM Common IR
   ├─ ConvertPIMCommonToUPMEM
   └─ ConvertPIMCommonToAiMX
```

현재 기본 변환과 최적화 파이프라인은 다음과 같이 구성할 수 있습니다.

```text
--convert-ttshared-to-pim
-cse
--pim-remove-buffers
-canonicalize
-loop-invariant-code-motion
--pim-classify-compute
```

여기에 논리적 작업과 program ID 의존성을 분석하는 `--pim-analyze-logical-work`를 추가하면 데이터 재사용과 Target mapping에 필요한 정보를 체계적으로 제공할 수 있습니다.

---

# 21. 첫 번째 구현 범위

초기 구현에서는 다음 기능을 지원합니다.

## 지원 범위

* 단일 PIM kernel
* program ID 기반 SPMD 실행
* `scf.for` 기반 K tile loop
* contiguous 및 strided access
* loop-carried access
* masked boundary tile
* row gather
* 원소별 gather와 scatter
* `linalg.matmul`
* `linalg.reduce`
* `linalg.generic`
* compute 종류 분류
* 중복 access와 load 제거
* passthrough copy 최적화
* loop-invariant 접근 분석

## 제외 범위

* wrap-around access
* 여러 kernel 사이의 데이터 residency
* 비동기 전송
* 전송과 연산의 overlap
* 여러 PIM target의 동시 실행
* 장치 간 직접 통신
* PIM Common 단계의 실제 장치 개수 결정
* 물리 메모리 layout
* Target별 descriptor 생성
* Target별 fusion
* Split-K 전용 PIM operation
* 복잡한 collective communication

---

# 22. 최종 정의

PIM Common IR은 다음과 같이 정의할 수 있습니다.

> PIM Common IR은 TTShared의 구조화된 memref와 linalg IR을 입력으로 받아, 메모리 접근 관용구를 논리적 주소와 데이터 이동 연산으로 끌어올리고 불필요한 staging buffer를 제거하며, program ID와 반복문으로 표현된 SPMD tile 구조와 linalg 계산 의미를 유지하는 PIM-neutral 중간 계층이다.

PIM Common과 Target IR의 역할은 다음과 같이 구분됩니다.

```text
PIM Common IR
  ├─ 논리적 데이터 접근
  ├─ 논리적 load/store/copy
  ├─ SPMD program 구조
  ├─ tile 내부 반복
  ├─ linalg 계산 의미
  ├─ 데이터 의존성
  ├─ program 차원별 재사용 가능성
  └─ target-neutral 중복 제거

Target IR
  ├─ 실제 장치 개수
  ├─ program-to-device mapping
  ├─ 물리 메모리 배치
  ├─ alignment와 padding
  ├─ 실제 residency
  ├─ descriptor와 명령어 생성
  ├─ Target별 legalization
  ├─ Target별 fusion
  └─ Host/device code generation
```

핵심 설계 원칙은 다음과 같습니다.

> PIM Common IR은 어떤 데이터를 어떤 방식으로 접근하고 어떤 계산을 수행하는지를 표현하며, 실제 하드웨어에서 어디에 배치하고 어떻게 실행할지는 Target IR에 위임한다.
