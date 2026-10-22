# PIM Common IR 설계

이 문서는 TTShared IR을 PIM Common IR로 바꾸는 pass들과, Target이 소비하는 Common
계약을 설명한다. Target별 lowering은 [UPMEM Target 설계](UPMEMTarget_Implementation.md)와
[AiMX Target 설계](AiMXTarget_Design.md)가 다룬다. 아래 `text` 예시는 타입·상수·access
선언을 생략한 의사 코드이며 그대로 파싱되는 MLIR이 아니다.

## 1. Common이 결정하는 것

**내용**

Common은 계산 의미, 원본 메모리 접근, 4단계 tiling 계층, 논리 Device 분할, Device별
상주 영역, 입력 공유 축, K 부분합 결합을 확정한다. Target은 이 사실을 자기 메모리
계층, 실행 단위, 전송 API로 옮기며 분할 수나 K 범위를 다시 정하지 않는다.

**근거**

- 덧셈 reduction 여부는 주소가 아니라 Linalg scalar region과 SCF 누산기에서만 판정할
  수 있다.
- 같은 입력을 받는 논리 Device가 물리적으로 인접한지는 Target topology의 문제다.
  Common은 공유 관계(`shared_over`)만 기록한다.
- 실행 순서와 전송 범위가 IR에 명시되어야 Host와 Device 코드가 같은 의미를 공유한다.

**예시**

```text
Common                               Target(UPMEM)
계산: linalg + scf                 →  WRAM 계산, tasklet 분할
주소: pim.access                  →  로컬 MRAM tile 주소
상주 범위: pim.place               →  고정 MRAM slot, Host pool
공유 축: shared_over               →  공유 pool 주소, 물리 DPU 묶음
부분합: pim.combine                →  Host add 결합
broadcast/scatter → launch → gather →  prepare/push → 실행 → 회수
```

연산마다 `pim.gemm`, `pim.gemv`를 두지 않는다. 계산은 Linalg가 표현하고 PIM Op는
배치·이동·결합처럼 기존 계산 IR에 없는 의미만 표현한다.

## 2. Op와 각각의 이유

정의: [PIMDialect.td](../include/triton-shared/Dialect/PIM/IR/PIMDialect.td).

**내용**

PIM dialect는 주소 명세, 입출력, 계획 형성 중의 Device 영역, 최종 실행 계획을 표현하는
Op만 가진다. `pim.device`는 계획 형성 중간 표현이고 `pim-form-host-plan`이
`pim.partition`으로 교체한다.

**근거**

- 최종 IR에 `pim.device`가 남으면 Target이 진단한다
  ([PIMToUPMEMPass.cpp](../lib/Conversion/PIMToUPMEM/PIMToUPMEMPass.cpp) 진입부).
- `PartitionOp::verify`가 입력별 전송 1회, 출력별 회수 1회, launch 1개, 전송 순서를
  검사한다([PIMDialect.cpp](../lib/Dialect/PIM/IR/PIMDialect.cpp)).

**예시**

| Op | 표현하는 사실 | 소비자 |
|---|---|---|
| `pim.access` | buffer, offset, origin, sizes, strides, valid, pad, id | Group 형성, placement, Target DMA 주소 |
| `pim.load` | access에서 계산용 tensor를 읽음 | Target의 입력 DMA 위치 |
| `pim.store` | partial 결합이 없는 최종 결과 저장 | Target 출력 DMA, Host 대입 |
| `pim.combine` | K partial과 최종 destination, combiner(`add`) | Target 출력 DMA, Host 누산 |
| `pim.device` | 계획 형성 중의 `[m,n,k]` 논리 좌표와 계산 영역 | placement·재사용 pass |
| `pim.partition` | 논리 분할 `[Pm,Pn,Pk]`와 완성된 실행 계획 | Target lowering 진입점 |
| `pim.place` | access 하나의 Execution Tile 영역과 고정 할당량 | Target 고정 slot, Host pool |
| `pim.broadcast` | `shared_over`가 있는 입력의 배포 | Target 공유 주소 전송 |
| `pim.scatter` | 공유 없는 입력의 배포 | Target 전송 |
| `pim.launch` | Device에서 실행할 계산 본문 | Target device 함수 |
| `pim.gather` | 출력 placement의 회수 | Target D2H |

전송 Op는 정수 access id로 같은 partition의 `pim.place`를 참조하며 새 buffer handle을
만들지 않는다. `broadcast`와 `scatter`는 수신 관계만 다르고 특정 SDK API를 강제하지
않는다.

## 3. 속성

### 3.1 4단계 tiling 계층과 논리 좌표

**내용**

모든 계층 벡터는 고정 순서 `[m,n,k]`이고 없는 축은 1이다. 각 계층은 바로 아래 계층의
개수로 지정한다. GEMV는 n이 없으므로 `Bn=Gn=Tn=Pn=1`이다.

**근거**

- `BlockAttr`, `BlockGroupAttr`, `ExecutionTileAttr`, `DeviceMappingAttr`의 verifier가
  rank 3과 양수를 요구한다([PIMDialect.cpp](../lib/Dialect/PIM/IR/PIMDialect.cpp)).
- `pim-verify`는 존재하지 않는 축의 Group·Execution Tile 크기가 1인지 검사한다.

**예시**

| 표현 | 단위 | 의미 |
|---|---|---|
| `#pim.block<shape=[Bm,Bn,Bk]>` | 원소 | Triton `BLOCK_M/N/K` |
| `#pim.block_group<blocks=[Gm,Gn,Gk]>` | Block 개수 | Device가 한 번의 local 계산에 쓰는 tile |
| `#pim.execution_tile<groups=[Tm,Tn,Tk]>` | Group 개수 | Device 한 번 실행에 상주시키는 범위 |
| `#pim.device_mapping<partitions=[Pm,Pn,Pk]>` | 논리 Device 수 | 한 round에서 축별 병렬 분할 수 |

```text
Group 원소 크기          = Block × [Gm,Gn,Gk]
Execution Tile 원소 크기 = Group 원소 크기 × [Tm,Tn,Tk]
논리 Device 수           = Pm × Pn × Pk  (≤ profile.num_units)
canonical 논리 번호      = ((pm*Pn)+pn)*Pk+pk
```

Common pass는 전달받은 Group·Tile·partition 값을 그대로 IR에 기록한다. 용량을 채우거나
비용식으로 크기를 고르지 않는다. 값 선정은 [AutoTuner 설계](AutoTuner_Design.md)의 몫이다.

### 3.2 Target profile

**내용**

`#pim.target_profile`은 `target`, `num_units`, `banks_per_unit`, `cores_per_bank`,
`lanes_per_core`, `bank_capacity_limit_bytes`, `local_capacity_limit_bytes`를 가진다.
Common verifier는 partition 곱, Device당 고정 할당 바이트, local tile 바이트를 이 예산과
비교한다.

**근거**

- `pim-map-devices`: `Pm×Pn×Pk > num_units`이면 진단한다.
- `pim-verify`: Device의 `allocation_shape` 바이트 합이 unit 용량을 넘거나, local tile
  바이트(`pim::localTileBytes`)가 `local_capacity_limit_bytes`를 넘으면 진단한다.

**예시**

[target.py](../pim_backend/backend/target.py)의 UPMEM 기본값:

| 필드 | 값 |
|---|---|
| `num_units` | 2048 |
| `banks_per_unit`, `cores_per_bank` | 1, 1 |
| `lanes_per_core` | 16 |
| `bank_capacity_limit_bytes` | 60 MiB |
| `local_capacity_limit_bytes` | 46680 |

### 3.3 공유 축 `shared_over`

**내용**

`shared_over`는 입력 placement가 같은 round에서 어떤 논리 좌표에만 달라질 때 같은 영역을
가리킨다는 사실이다. 입력 이름이나 A/B 순서로 판정하지 않고 origin·stride·valid·pad의
loop 의존성으로 판정한다.

**근거**

- [PIMPlaceDataPass.cpp](../lib/Conversion/TTSharedToPIM/PIMPlaceDataPass.cpp)
  `placeAccess`는 origin, strides, valid, pad를 움직이는 loop를 모으고, 계층 축 중
  사용되지 않은 축을 `shared_over`에 넣는다. 출력에는 붙이지 않는다.

**예시**

```text
GEMM A[m,k]: n에 불변 → shared_over=[n]
GEMM B[k,n]: m에 불변 → shared_over=[m]
GEMV x[k]  : m에 불변 → shared_over=[m]
GEMV A[m,k]: 공유 없음
```

모든 scheduling round에서 입력을 다시 준비하고 전송한다. `shared_over`는 같은 round
안의 Device 간 공유이며 round 사이의 H2D 생략을 뜻하지 않는다.

### 3.4 Loop 표식

**내용**

loop의 역할은 속성으로 명시하고 뒤 pass는 이 속성만 읽는다. 주소를 움직이는 loop를
보고 reduction 여부를 추측하지 않는다.

**근거**

- `pim.reduction`은 `pim-form-compute`만 붙이고 `pim-form-block-groups`,
  `pim-form-execution-tiles`, `pim-map-devices`는 이 속성으로 K loop를 찾는다.

**예시**

| 표식 | 붙이는 pass | 의미 |
|---|---|---|
| `pim.axis="m"/"n"` | build-logical-grid | source grid에서 복원한 출력 병렬 축 |
| `pim.reduction="add"` | form-compute | Linalg가 SCF 누산기를 add로 갱신 |
| `pim.scheduling_axis` | map-devices | Host round loop(`n`, `m`, `reduction_k`) |
| `pim.temporal_reduction` | map-devices | Device가 현재 round에 도는 local K loop |
| `pim.device_origin` | map-devices | local IV → 전역 Group 좌표 복원 산술 |
| `pim.min_trip_count` | map-devices | partition=1 축의 진입한 round는 비어 있지 않음 |
| `pim.reused_input` | reuse-execution-tile-inputs | loop 밖으로 이동한 load. Target이 로컬화 |

## 4. 변환 순서

**내용**

[compiler.py](../pim_backend/backend/compiler.py)의 `PIM_COMMON_PASSES`가 순서를 정한다.
각 pass는 다음 pass가 소비할 사실을 확정하고, 패턴에 맞지 않는 계산은 진단한다.

**근거**

- AiMX target에서는 `--pim-reuse-execution-tile-inputs`를 제거하고 마지막
  `-canonicalize`를 붙이지 않는다. AiMX `pim.load`는 상주 panel의 view이고, SCF
  canonicalize가 단일 반복 loop와 축 속성을 지울 수 있기 때문이다(`compiler.py` 주석).

**예시**

```text
TTShared IR
  pim-normalize-address → cse → LICM → canonicalize
  pim-form-compute
  convert-ttshared-to-pim
  pim-build-logical-grid
  pim-recover-access-origin
  pim-set-target-profile
  pim-form-block-groups       (group-m/n/k)
  pim-form-execution-tiles    (tile-m/n/k)
  pim-map-devices             (p-m/p-n/p-k)
  pim-place-data → pim-verify
  pim-reuse-execution-tile-inputs → pim-verify     (UPMEM만)
  pim-form-host-plan → pim-verify → cse → canonicalize(UPMEM만)
Target lowering
```

## 5. 계산 패턴 인식 (`pim-form-compute`)

구현: [PIMFormComputePass.cpp](../lib/Conversion/TTSharedToPIM/PIMFormComputePass.cpp).

### 5.1 누산기 정규화

**내용**

`compute(outs=zero)` 결과를 identity-map `linalg.generic(add)`로 누산기에 더하고 그
결과를 같은 iter_arg로 yield하는 형태를 `compute(outs=acc)`로 바꾼다. 그 다음 누산기
loop를 `pim.reduction="add"`로 표시하고, 누산기 초기화 chain을 loop 바로 앞으로 옮긴다.

**근거**

- `classifyAddCompute`: scalar body의 최종 add가 누산기와 독립 contribution을 더해야
  한다. contribution이 이전 누산기에 의존하면 K를 나눌 수 없으므로 진단한다.
- `markReduction`: 초기값이 0 fill이 아니거나 iter_arg가 둘 이상이면 진단한다.
- `sinkInit`: bufferization 뒤에도 출력 instance마다 초기화가 실행되게 한다.

**예시**

```text
변환 전: partial = matmul(A, B, zero);  next = acc + partial
변환 후: next    = matmul(A, B, acc)          // scf.for {pim.reduction="add"}
```

### 5.2 GEMV 정규화

**내용**

TTShared가 내는 두 GEMV 배치를 `linalg.matvec(A, x, init)`로 바꾼다. 곱셈 앞의 sign
extension(INT8 → INT32)은 벗겨 내고 좁은 원래 값을 matvec 입력으로 쓴다.

**근거**

- `formMatvec`: reduce는 input + accumulator의 sum, 곱셈은 두 직접 입력의 identity/
  all-parallel 곱, broadcast map은 정확히 `(m,k)->(0,k)`여야 한다.
- `stripSignExtension`: `arith.extsi`이거나 body가 `extsi` 하나인 identity generic만
  벗긴다. `linalg.matvec`는 입력을 출력 타입으로 sign extension해 곱하므로 의미가 같다.
- 곱셈 tensor와 reduce 결과의 원소 타입이 같아야 한다. 곱셈 **후** 확장(INT8 곱 →
  INT32)은 곱셈 결과 타입이 달라 이 패턴에 해당하지 않는다.
- 다른 사용자가 없는 broadcast/expand_shape/extension/mul/transpose/reduce만 지운다.

**예시**

```text
x_i8[K] → expand_shape[1,K] → broadcast[M,K] → extsi → x_i32[M,K] ┐
A_i8[M,K] ─────────────────────────────────── extsi → A_i32[M,K] ┴→ mul_i32
    → reduce(dim=1)                       또는 transpose[1,0] → reduce(dim=0)
⇒ linalg.matvec(A_i8, x_i8, acc_i32)
```

Triton 소스에서는 곱하기 전에 넓혀야 이 형태가 된다.

```python
acc += tl.sum(a.to(tl.int32) * x.to(tl.int32)[None, :], axis=1)
```

### 5.3 출력 float 변환 흡수 (`convert-ttshared-to-pim`)

**내용**

저장 값이 원소 타입만 바꾸는 identity generic(`truncf` 또는 `extf` 하나)이면 그 generic을
지우고 변환 전 값을 `pim.store`에 연결한다.

**근거**

- [TTSharedToPIMPass.cpp](../lib/Conversion/TTSharedToPIM/TTSharedToPIMPass.cpp)
  `foldFloatConversion`: `pim.store`/`pim.combine`은 목적지 buffer 원소 타입으로 쓴다.
  변환을 흡수하면 K partial 결합 앞에 epilogue가 끼지 않는다.

**예시**

```text
acc_f32 → generic{truncf} → bf16 → store C_bf16
⇒ store(acc_f32 → C_bf16)       // 타입 변환은 저장 시점에 수행
```

## 6. 메모리 접근과 논리 grid

### 6.1 메모리 접근 끌어올리기 (`convert-ttshared-to-pim`)

**내용**

TTShared의 load/store 패턴을 `pim.access` + `pim.load/store`로 바꾸고 staging buffer를
없앤다. 변환되지 않은 memref 접근이 남으면 진단한다.

**근거**

- 같은 alloc을 두 copy가 채우는 wrap-around load는 하나의 access로 표현할 수 없어
  진단한다.
- 같은 reinterpret_cast를 load와 store가 함께 쓰면 하나의 access를 공유한다.

**예시**

| TTShared 표현 | 변환 결과 |
|---|---|
| reinterpret_cast → copy to alloc → to_tensor | access + load |
| reinterpret_cast/subview → (fill) padded alloc/subview로 copy | valid/pad를 가진 access + load |
| materialize_in_destination(dest=reinterpret_cast) | access + store |
| materialize_in_destination(dest=subview) | valid를 가진 access + store |

### 6.2 주소 정규화와 origin 복원

**내용**

`pim-normalize-address`는 주소를 나르는 iter_arg를 `initial + i*advance`로 다시 쓴다.
`pim-recover-access-origin`은 선형 offset을 stride와 대조해 축별 origin을 복원한다.

**근거**

- [PIMNormalizeAddressPass.cpp](../lib/Conversion/TTSharedToPIM/PIMNormalizeAddressPass.cpp):
  닫힌 형태가 아니면 뒤의 origin 복원이 실패한다.
- [PIMRecoverAccessOriginPass.cpp](../lib/Conversion/TTSharedToPIM/PIMRecoverAccessOriginPass.cpp):
  `valid`가 알려 주는 축 시작점을 먼저 대조한다. stride 배수만으로는 `N ≤ BLOCK_N`인 열
  시작을 행 축과 구별할 수 없기 때문이다. tile에 없는 원본 축은 크기 1로 앞에 붙인다.

**예시**

```text
offset = Σ origin[d] × stride[d]
A: offset = (pid_m*BM)*stride_am + (k*BK)*stride_ak
   → origin = [pid_m*BM, k*BK]
```

### 6.3 논리 grid 복원 (`pim-build-logical-grid`)

**내용**

triton-shared ABI의 마지막 6개 인자(num_programs×3, program_id×3)에서 program_id 사용을
찾아 m/n `scf.for`로 바꾸고 program_id 인자를 시그니처에서 지운다.

**근거**

- 직접 pid, `pid / d`와 `pid % d` 쌍, GROUP_M 순회 순서를 인식한다.
- source에 별도 split-K 축은 허용하지 않는다. K는 5절의 semantic reduction이다.

**예시**

```text
pid_m = program_id(0); pid_n = program_id(1)
⇒ scf.for %m {pim.axis="m"} { scf.for %n {pim.axis="n"} { body } }
```

## 7. Group, Execution Tile, scheduling

### 7.1 Block Group 형성 (`pim-form-block-groups`)

**내용**

`pim.axis` loop와 `pim.reduction` loop에 Group factor를 적용한다. loop 상한을
`ceildiv(tiles, g)`로 바꾸고, 해당 loop가 origin을 움직이는 access 차원의 크기를 g배로
키우며, Linalg/tensor shape도 맞춘다.

**근거**

- [PIMFormBlockGroupsPass.cpp](../lib/Conversion/TTSharedToPIM/PIMFormBlockGroupsPass.cpp):
  상한이 g로 나누어떨어지지 않거나 runtime 값이면, 그 축을 쓰는 access는 원본 extent를
  복원할 dynamic valid를 가져야 한다. 없으면 존재하지 않는 Block을 읽으므로 진단한다.
- Block shape는 regroup 뒤 Group 원소 shape를 Block 개수로 나눠 기록한다.

**예시**

```text
Group valid = min(max(original_extent - group_origin, 0), group_size)
```

### 7.2 Execution Tile 기록 (`pim-form-execution-tiles`)

**내용**

주어진 `tile-m/n/k`(Group 개수)를 `#pim.execution_tile<groups=[Tm,Tn,Tk]>`로 기록한다.
0은 Group 하나다.

**근거**

- [PIMFormExecutionTilesPass.cpp](../lib/Conversion/TTSharedToPIM/PIMFormExecutionTilesPass.cpp):
  해당 loop가 없는 축에 2 이상을 주면 진단한다. semantic reduction loop가 둘 이상이면
  진단한다.

**예시**

```text
tile-m=4 tile-n=4 tile-k=14 → #pim.execution_tile<groups=[4,4,14]>
```

### 7.3 Device 분할과 round (`pim-map-devices`)

**내용**

축마다 Group 개수를 Execution Tile로 묶고, Tile을 `(round, device 좌표)`에 round-major로
배정한다. Host round loop는 존재하는 축에 대해 `n → m → reduction_k` 순서로 고정된다.
K는 `Pk`(공간)와 `reduction_k` round(시간)로 나뉘며 둘은 같은 reduction domain의 부분
범위다. local loop는 `0..trips`로 바뀌고 몸체 첫머리에 `pim.device_origin` 산술이 생긴다.

**근거**

- [PIMMapDevicesPass.cpp](../lib/Conversion/TTSharedToPIM/PIMMapDevicesPass.cpp)
  `scheduleLoop`, `mapFunc`.
- source `pim.axis="k"`, 존재하지 않는 축의 partition > 1, `Pm×Pn×Pk > num_units`는
  진단한다.
- M/N이 없는 K-only reduction은 누산기 초기화 chain과 직접 store를 함께 `pim.device`
  안으로 옮긴다. 그렇지 않으면 초기화가 round 밖에서 한 번만 실행된다.

**예시**

```text
axis 하나 (G=전체 Group 수, T=Tile당 Group 수, P=partition 수)
exec_tiles = ceildiv(G, T)
rounds     = ceildiv(exec_tiles, P)
tile 번호  = round*P + p
first      = tile 번호 * T
trips      = min(T, max(G - first, 0))

G=10, T=2, P=2:
           p=0            p=1
round 0:   tile0 (g0,g1)  tile1 (g2,g3)
round 1:   tile2 (g4,g5)  tile3 (g6,g7)
round 2:   tile4 (g8,g9)  tile5 (trips=0)
```

```text
for n_round { for m_round { for k_round {
  pim.device(pm, pn, pk) {
    scf.for m_local in [0, trips_m) { m = start_m + m_local   // pim.device_origin
      scf.for n_local ... { scf.for k_local {pim.temporal_reduction} ... }
} } } }
```

### 7.4 부분합 명시

**내용**

K를 나누면 `(reduction_k round, pk)`마다 0에서 시작한 partial이 생긴다. 원래 reduction
결과를 저장하던 `pim.store`를 `pim.combine ... kind "add"`로 바꾼다. `Pk=1`이어도 K
round가 여러 개일 수 있으므로 reduction이 있으면 항상 combine을 쓴다.

**근거**

- `mapFunc`: reduction 결과를 직접 저장하지 않고 다른 연산을 거친 store(epilogue)는
  진단한다. partial마다 비선형 epilogue를 적용하면 최종 결과와 다르기 때문이다.
- reduction 결과 하나에 최종 store가 정확히 하나여야 한다.

**예시**

```text
C[m,n] = Σ_{k_round} Σ_{pk} partial(m, n, pk, k_round)
```

## 8. Placement (`pim-place-data`)

**내용**

access마다 현재 Device의 첫 Group 좌표에서 `pim.place`를 만든다. `size`는 이번 round의
유효 원소 수이고 `allocation_shape`는 모든 round에 공통인 고정 공간이다.

**근거**

- [PIMPlaceDataPass.cpp](../lib/Conversion/TTSharedToPIM/PIMPlaceDataPass.cpp):
  `capacityTiles`는 계층 Group 개수(`Tm/Tn/Tk`)이고 runtime trip 수가 아니다.
- valid에서 원본 extent를 복원해 `size = min(max(extent - origin, 0), span)`으로 자른다.
- 계층 밖 loop가 buffer 축을 움직이거나, 분할된 축이 다른 loop와 함께 한 buffer 축을
  움직이면 연속 영역 하나로 표현할 수 없어 진단한다.

**예시**

```text
start[d]            = 첫 Group 좌표에서 다시 만든 origin[d]
size[d]             = min(max(extent[d] - start[d], 0), trips × access.sizes[d])
allocation_shape[d] = (그 축을 모는 계층 loop의 Group 개수) × access.sizes[d]
direction           = "in" | "out"
shared_over         = 입력 주소가 의존하지 않는 계층 축 (3.3절)

A[1024,28672], Group [64,128], Tile [4,·,14]
→ allocation_shape = [256, 1792], shared_over=[n]
```

## 9. Device 내부 load 이동과 Host 계획

### 9.1 입력 load 이동 (`pim-reuse-execution-tile-inputs`)

**내용**

M/N Group loop에 불변인 읽기 전용 입력의 access/load를 그 loop 앞으로 옮긴다. 같은
Device가 같은 상주 입력을 Group마다 local memory로 다시 읽지 않게 하기 위함이다.

**근거**

- [PIMReuseExecutionTileInputsPass.cpp](../lib/Conversion/TTSharedToPIM/PIMReuseExecutionTileInputsPass.cpp):
  load와 대상 loop 사이의 loop는 모두 singleton(Tile Group 수 1)이어야 한다. load 결과는
  Linalg DPS 입력으로만 쓰여야 하고, Device 안에 그 access id로 쓰는 store/combine이나
  알 수 없는 memory effect가 없어야 한다.
- 대상 loop가 비어 있을 수 있으면 0/1회 도는 보호 `scf.for`로 감싼다.
- 이동한 access의 origin은 전역 값으로 남기고 `pim.reused_input`을 붙인다. Target이
  `origin - place.start`로 로컬화한다.

**예시**

```text
변환 전: for m { for k(singleton) { x = load(X[k]); y = matvec(A[m,k], x, y) } }
변환 후: x = load(X[k0]) {pim.reused_input};  for m { for k { y = matvec(..., x, y) } }
```

이 pass는 Host round, H2D 양, Group 크기, loop 순서를 바꾸지 않는다.

### 9.2 실행 계획 생성 (`pim-form-host-plan`)

**내용**

`pim.device` 본문을 placement, Host scalar 설정, 입력 전송, launch, 출력 회수로 나누고
`pim.partition`으로 교체한다.

**근거**

- [PIMFormHostPlanPass.cpp](../lib/Conversion/TTSharedToPIM/PIMFormHostPlanPass.cpp):
  부수 효과 없는 int/index/float 산술 중 계산 결과에 의존하지 않는 것만 Host에 남긴다.
  placement가 계산 결과에 의존하면 진단한다.
- `shared_over`가 비어 있지 않은 입력은 `broadcast`, 나머지는 `scatter`가 된다.

**예시**

```text
pim.device {                      pim.partition {
  scalar setup                      scalar setup
  place A, B, C          →          place A, B, C
  compute                           broadcast A; broadcast B
}                                   launch { compute }
                                    gather C
                                  }
```

## 10. 검증과 지원 범위

**내용**

`pim-verify`는 Target 진입 전에 Common 계약을 검사한다. verifier 통과는 Common 계약
충족을 뜻하며, Target이 모든 계산 leaf를 지원한다는 뜻은 아니다.

**근거**

- [PIMVerifyPass.cpp](../lib/Conversion/TTSharedToPIM/PIMVerifyPass.cpp) 검사 항목:
  profile·계층 속성 존재, 비활성 축 크기 1, local tile 원소 수와 Group 크기 일치,
  local tile 바이트 ≤ local 용량, K는 하나의 temporal reduction과 하나의 `reduction_k`
  round, `p_m/p_n/p_k > 1`에 대응 축 존재, zero padding만 허용, access id 중복 금지,
  access–placement 일대일, `allocation_shape[d]` = access 크기 × 그 축을 모는 계층의
  Execution Tile Group 수, 정적 `size[d]`는 `[0, allocation_shape[d]]` 범위, Device 할당
  바이트 ≤ unit 용량, 최종 출력의 K 독립성, K partial은 `pim.combine`만 사용.
- scheduling 산술 전체가 7.3절 식과 같은지 증명하지는 않는다. 그 식은 생성 pass가
  보장한다.

**예시**

| 경우 | 상태 |
|---|---|
| named `linalg.matmul`, `linalg.matvec` | 지원 |
| 5.2절 GEMV 패턴(곱셈 전 sign extension 포함) | `linalg.matvec`로 정규화 |
| 곱셈 후 INT32 확장하는 GEMV | 정규화 대상 아님 |
| identity/all-parallel elementwise | Target scalar emitter 범위에서 지원 |
| 임의 broadcast/transpose generic | 미지원 |
| K-only `linalg.reduce` | Common 지원, UPMEM은 tasklet 1개 요구 |
| reduction 결과 앞의 epilogue | 진단 (float 변환 하나는 5.3절로 흡수) |
| wrap-around load, 불규칙 gather/scatter | 진단 |
