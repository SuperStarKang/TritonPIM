# PIM Common IR 설계와 변환 근거

이 문서는 현재 PIM Common → UPMEM 경로를 설명한다. 먼저 표현을 둔 이유를 정리하고,
그 표현을 만드는 패턴과 Target이 소비하는 계약을 설명한다. 지원되는 IR과 앞으로
지원할 수 있는 연산을 구분한다. 아래 `text` 예시는 타입·상수·access 선언을 생략한
의사 코드이며 그대로 파싱하는 MLIR이 아니다.

## 1. 무엇을 Common에서 결정하는가

**내용**

Common은 계산 의미, 원본 메모리 접근, 논리 분할, 전송할 영역과 부분합 결합을 정한다.
UPMEM은 이 정보를 MRAM/WRAM, tasklet, SDK 전송과 실제 DPU 배정으로 옮긴다.
Common을 입력으로 받은 Target이 원본 Triton 코드의 GEMM/GEMV 이름을 다시 찾을
필요가 없도록 한다.

**근거**

- 같은 주소를 사용한다는 사실만으로 덧셈 reduction인지 판단할 수 없다. 계산 의미는
  Linalg의 scalar region과 SCF 누산기에서 읽어야 한다.
- 같은 입력을 받는 논리 Device가 실제 SDK 전송 묶음에 인접하는지는 별도 문제다.
  Common은 공유 관계를 기록하고 물리 배정은 UPMEM이 결정한다.
- 실행 순서와 전송 범위가 IR에 있으면 Host와 Device가 같은 의미를 공유한다.
  Target에서 새로운 K window나 별도의 출력 plane을 추측해 만들지 않는다.

**예시**

```text
Common                                  UPMEM
계산: linalg + scf                    →  WRAM 계산과 tasklet 분할
주소: pim.access                     →  로컬 MRAM 타일 주소
상주 범위: pim.place                  →  고정 MRAM 공간과 Host pool
공유 축: shared_over                  →  공유 pool 주소와 물리 DPU 묶음
부분합: pim.combine                   →  MRAM 부분합 저장 + Host 합산
입력 전송 → launch → gather           →  prepare/push → 실행 → 회수
```

연산마다 `pim.gemm`, `pim.gemv`, `pim.elementwise`를 추가하지 않는다. 기존 Linalg가
계산을 표현하고 PIM Op는 배치·이동·결합 중 기존 계산 IR에 없는 의미를 표현한다.

## 2. 남기는 Op와 각각의 이유

정의: [PIMDialect.td](../include/triton-shared/Dialect/PIM/IR/PIMDialect.td).

| Op | 표현하는 사실 | 필요한 이유와 소비자 |
|---|---|---|
| `pim.access` | 원본 buffer, offset, origin, sizes, strides, valid, pad, id | 주소와 데이터 이동을 분리한다. Group 형성·placement·DMA가 같은 주소 명세를 사용한다. |
| `pim.load` | access에서 계산용 tensor를 읽음 | 실제 입력 소비 위치다. UPMEM은 이 위치에 MRAM→WRAM DMA를 만든다. |
| `pim.store` | 최종 결과를 access에 저장 | elementwise처럼 partial 결합이 없는 출력을 나타낸다. |
| `pim.combine` | partial과 최종 destination, combiner | split-K/여러 K round의 결과를 덮어쓰지 않도록 결합 의미를 보존한다. 현재 UPMEM combiner는 add다. |
| `pim.device` | 계획 형성 중의 논리 좌표와 계산 영역 | 기존 분석 pass들이 배치와 계산을 함께 다루는 중간 표현이다. 최종 Target 입력에는 남지 않는다. |
| `pim.partition` | 논리 `[pm,pn,pk]`와 완성된 실행 계획 | 논리 분할을 실제 DPU 열거 순서와 분리한다. |
| `pim.place` | access 하나의 Execution Tile 범위와 고정 할당량 | Group 하나의 access만으로 전체 H2D 영역을 알 수 없으므로 필요하다. |
| `pim.broadcast` | 고유 입력별 공유 수신자들에게 배포 | `shared_over`의 동치 그룹 전체에 입력이 필요함을 실행 계획에 나타낸다. |
| `pim.scatter` | 입력 placement를 Device별로 배포 | 공유 없는 입력의 전송 시점을 명시한다. |
| `pim.launch` | Device에서 실행할 계산 본문 | Host 주소·전송 계산과 Device 계산을 구조적으로 분리한다. |
| `pim.gather` | 출력 placement의 회수 | Host가 결과를 사용할 수 있는 시점을 명시한다. partial 결합 의미는 combine에서 온다. |

`broadcast`/`scatter`는 수신 관계가 다르지만 UPMEM에서는 둘 다 `stage + to_mram`으로
내린다. Common 이름이 특정 SDK API를 강제하지 않는다. `broadcast`의 입력은 여러
고유 타일일 수 있다. 전송 Op의 정수 access id는 같은 partition의 place를 참조하며
새 buffer handle을 추가하지 않는다.

`pim.device`를 `pim.partition`으로 바꾸는 단계는 중간 표현을 끝내는 경계다. 최종 IR에
두 Op를 겹쳐 두지 않는다. Target은 미변환 `pim.device`를 진단한다.

## 3. 속성에 저장하는 정보와 저장하지 않는 정보

### 3.1 세 계층과 논리 좌표

| 표현 | 단위 | 사용 이유 |
|---|---|---|
| `#pim.block<shape=[Bm,Bn,Bk]>` | 원소 | source의 기본 계산 눈금 |
| `#pim.block_group<blocks=[Gm,Gn,Gk]>` | Block 개수 | 한 번의 local 계산에서 사용할 tile 크기 |
| `#pim.execution_tile<groups=[Tm,Tn,Tk]>` | Group 개수 | Device 한 번 실행에 상주시킬 범위 |
| `#pim.device_mapping<partitions=[Pm,Pn,Pk]>` | 논리 Device 수 | 동일 round에서 각 축을 병렬 분할하는 수 |

```text
Group 원소 크기          = Block × Group의 Block 개수
Execution Tile 원소 크기 = Group 원소 크기 × Tile의 Group 개수
Device 수               = Pm × Pn × Pk
```

벡터 순서는 `[m,n,k]`이며 없는 축은 1이다. GEMV는 일반적으로 n이 없으므로 `Pn=1`,
`Bn=Gn=Tn=1`이다. Device 좌표 `pm,pn,pk`는 각 축의 위치이고 `Pm,Pn,Pk`는 개수다.
Target에서도 같은 좌표 의미를 이어받는다.

현재 Common pass들은 전달된 Group/Tile/partition 값을 materialize한다. Host cache
크기로 Triton Block을 자동 선정하거나 비용식으로 scheduling loop 순서를 고르지 않는다.
Group을 확대하는 것과 Device 한 번 실행에 여러 Group을 넣는 것은 다른 변환이다.

### 3.2 Profile

`pim.target_profile`은 `target`, `num_units`, `banks_per_unit`, `cores_per_bank`,
`lanes_per_core`, `bank_capacity_limit_bytes`, `local_capacity_limit_bytes`를 가진다.
Common은 분할 수와 resident/local tile 바이트를 이 예산과 비교한다. UPMEM은 다시
자신의 topology와 DMA 제약을 검사한다. Profile은 Host L1/L2/L3 용량 모델이 아니다.

### 3.3 공유 사실과 수명

`shared_over`는 같은 round에서 어떤 논리 좌표만 달라져도 동일 입력 영역인지 나타낸다.
GEMM의 `A[m,k]`에는 n, `B[k,n]`에는 m, GEMV의 `x[k]`에는 m이 공유 축이다.

이 정보는 UPMEM의 고유 Host pool slot과 물리 DPU 묶음을 정하는 직접적인 근거다.
입력 이름이나 A/B 순서로 공유를 결정하지 않는다.

모든 scheduling round에서 입력을 준비하고 전송한다. Host pool의 **공간**은 여러
round가 재사용하지만 이전 round의 **내용**을 믿고 전송을 생략하는 속성이나 pass는 없다.
따라서 Host 입출력 간 alias 증명을 실행 계획에 추가하지 않는다.

### 3.4 Loop 표식

| 표식 | 생성 근거와 소비 |
|---|---|
| `pim.axis="m"/"n"` | source grid에서 복원한 출력 parallel loop. Group·배치·Target 로컬화에 사용 |
| `pim.reduction="add"` | Linalg가 SCF 누산기를 add로 갱신하는 관계. K 분할과 combine의 근거 |
| `pim.scheduling_axis` | Host의 m/n/reduction_k round. 실행 영역과 자원 수명의 경계 |
| `pim.temporal_reduction` | 한 Device가 현재 round에 맡는 local K loop |
| `pim.device_origin` | local IV를 전역 Group 좌표로 복원하는 산술. UPMEM이 MRAM 로컬 좌표로 바꾸는 지점 |
| `pim.min_trip_count` | partition=1인 축의 진입한 round는 비어 있지 않다는 근거. local load 이동에서 사용 |
| `pim.reused_input` | local loop 밖으로 이동한 load의 주소를 `origin-place.start`로 로컬화하기 위한 표식 |

마지막 두 표식은 아래의 Device 내부 load 이동에 쓰인다. Host H2D 생략을 뜻하지 않는다.

## 4. 전체 변환 순서

실제 pipeline: [compiler.py](../pim_backend/backend/compiler.py)의 `PIM_COMMON_PASSES`.

```text
TTShared IR
  pim-normalize-address → cse / LICM / canonicalize
  pim-form-compute
  convert-ttshared-to-pim
  pim-build-logical-grid
  pim-recover-access-origin
  pim-set-target-profile
  pim-form-block-groups
  pim-form-execution-tiles
  pim-map-devices
  pim-place-data → pim-verify
  pim-reuse-execution-tile-inputs → pim-verify
  pim-form-host-plan → pim-verify → cse / canonicalize
UPMEM lowering
```

각 단계는 다음 단계가 소비할 사실을 확정한다. 패턴에 맞지 않는 계산을 임의의 다른
연산으로 치환하거나 지원하지 않는 메모리 접근을 숨기는 경로는 두지 않는다.

## 5. 계산 패턴을 어떻게 찾는가

구현: [PIMFormComputePass.cpp](../lib/Conversion/TTSharedToPIM/PIMFormComputePass.cpp).

### 5.1 누산기 정규화

**찾는 패턴**: `compute(outs=zero)`의 결과를 기존 누산기에 더하는 identity-map
`linalg.generic(add)`이며, 결과가 같은 SCF iter_arg로 yield된다.

**변환**: 계산의 outs를 기존 누산기로 연결한다. 그 후 Linalg scalar body의 최종 add가
누산기와 독립 contribution을 더하는지 확인한다. 하나의 tensor 누산기/result와 zero
초기값이어야 `pim.reduction="add"`로 표시한다.

**이유**: 임시 결과와 별도 덧셈을 없애면서 K partial의 의미를 확정한다. contribution이
이전 누산기에 의존하면 K를 독립적으로 나눌 수 없으므로 진단한다. 초기화 chain은 해당
loop 바로 앞으로 이동해 bufferization 후에도 출력 instance마다 초기화되게 한다.

```text
기존: partial = compute(A, B, zero)
      next    = acc + partial
변환: next    = compute(A, B, acc)
```

### 5.2 GEMV 정규화의 현재 범위

**찾는 패턴**:

```text
x[K] → expand_shape[1,K] → broadcast[M,K]
                                   × A[M,K]
                                   ↓
                 reduce(dim=1), 또는 transpose[1,0] → reduce(dim=0)
```

- broadcast 입력 map은 `(m,k)->(0,k)`, 출력 map은 identity여야 한다.
- broadcast region은 입력을 그대로 yield해야 한다.
- 곱셈은 두 직접 입력을 곱하는 identity/all-parallel generic이어야 한다.
- reduction은 input과 accumulator를 직접 더하는 sum이어야 한다.
- 곱셈 tensor와 reduction 결과의 원소 타입이 같아야 한다.

**변환**: `linalg.matvec(A,x,init)`를 만들고 더 이상 사용하지 않는 broadcast/mul/
transpose/reduce chain만 지운다. 다른 사용자가 있는 중간 결과는 보존한다.

**이유**: 벡터를 M행으로 확장한 임시 계산을 없애며, Target이 indexing map만으로
A/y의 행 분할과 x 공유를 읽을 수 있다. 이미 named matvec이면 이 인식은 필요 없다.

### 5.3 현재 INT8 gemv.py가 실패하는 이유

[gemv.py](../pim_backend/examples/gemv.py)는 다음 식을 사용한다.

```python
acc += tl.sum(a * x[None, :], axis=1)
```

설치된 Triton [semantic.py](../../triton/python/triton/language/semantic.py)의
`integer_promote_impl`/`mul`은 두 INT8 tensor의 곱을 INT8로 만든다.
[standard.py](../../triton/python/triton/language/standard.py)의
`_pick_sum_dtype`/`sum`은 이 곱셈 결과를 INT32로 넓힌 뒤
합산한다. 따라서 예상 dataflow에는 다음 폭 변환이 있다.

```text
A_i8 × broadcast(x_i8) → product_i8 → signed extension_i32 → sum_i32
```

현재 `formMatvec`는 reduction 입력에서 바로 곱셈 generic을 찾는다. 중간 cast를
통과하지 않으며, 곱셈 결과 타입과 reduction 결과 타입도 같아야 한다. 이 표현은 그
패턴의 범위 밖이다. 보고된 오류는 INT8 broadcast가 UPMEM 변환까지 남았고, UPMEM의
identity-only generic 검사에서 거부되었음을 보여준다. 오류의 임시 전체 IR 파일은
점검 시점에 남아 있지 않아 cast/transpose의 정확한 Op 배치는 재현 확인하지 않았다.

타입 검사를 지우거나 cast를 무조건 건너뛰면 안 된다.

```text
INT8(100) × INT8(2), 곱셈 후 확장:  INT8(200) = -56 → INT32(-56)
INT32(100) × INT32(2):                              INT32(200)
```

표준적인 INT8 입력·INT32 곱셈/누산 GEMV를 의도한다면 source에서 곱셈 전에 확장하는
의미가 명시되어야 한다. 향후 matcher는 이 선행 확장의 signedness·폭·shape를 증명한
경우에만 mixed-type matvec로 바꿔야 한다. 곱셈 후 확장은 같은 변환 대상이 아니다.
현재 예제와 matcher에 이 수정은 적용하지 않았다.

기존 [form_compute_matvec.mlir](../test/Conversion/TTSharedToPIM/form_compute_matvec.mlir)는
f32 패턴을 다룬다. 그 테스트가 존재한다는 사실로 INT8 예제 지원까지 주장할 수 없다.

## 6. 메모리 접근과 논리 grid 패턴

### 6.1 메모리 접근 올리기

구현: [TTSharedToPIMPass.cpp](../lib/Conversion/TTSharedToPIM/TTSharedToPIMPass.cpp).

| 찾는 TTShared 표현 | 변환 | 근거 |
|---|---|---|
| reinterpret_cast → copy to alloc → to_tensor | access + load | 임시 Host형 tile buffer가 아니라 명시적 읽기 의미로 유지 |
| reinterpret_cast/subview → padded alloc/subview로 copy | valid/pad를 가진 access + load | mask의 유효 영역과 채움값 보존 |
| materialize_in_destination | access + store | 실제 출력 목적지를 보존 |
| extract_slice → subview destination | valid를 가진 access + store | tail 밖에는 쓰지 않음 |

주소 정규화는 loop-carried pointer 식을 후속 origin 분석이 읽을 수 있는 식으로
정리한다. wrap-around처럼 두 개의 떨어진 영역을 하나의 access로 표현할 수 없는
load와 변환되지 않은 memref/bufferization 접근은 진단한다.

### 6.2 논리 축 복원과 origin

[PIMBuildLogicalGridPass.cpp](../lib/Conversion/TTSharedToPIM/PIMBuildLogicalGridPass.cpp)는
TTShared의 grid ABI에서 program_id 사용을 찾는다. 직접 pid, div/rem으로 평탄화된
좌표, 인식 가능한 grouped order를 m/n SCF loop로 복원한다. 별도의 source split-K
축은 허용하지 않는다. K는 5절에서 확정한 semantic reduction이다.

[PIMRecoverAccessOriginPass.cpp](../lib/Conversion/TTSharedToPIM/PIMRecoverAccessOriginPass.cpp)는
선형 offset을 stride와 대조해 다음 식의 origin을 복원한다.

```text
offset = Σ origin[d] × stride[d]
```

이유는 offset만으로는 어느 logical 축이 buffer의 어느 차원을 움직이는지 알기 어렵기
때문이다. origin은 Group 크기 확대, placement, 공유 축 분석에 같은 근거를 제공한다.

## 7. Group, Execution Tile, scheduling 생성

### 7.1 Group 형성

[PIMFormBlockGroupsPass.cpp](../lib/Conversion/TTSharedToPIM/PIMFormBlockGroupsPass.cpp)는
`pim.axis`와 semantic reduction loop를 찾아 요청된 Group factor를 적용한다.
`collectDrivers`로 각 access 차원을 움직이는 축을 찾고 그 차원의 tile 크기를 키운다.
Linalg/tensor shape도 이에 맞춰 갱신한다.

```text
기존 Block valid를 그대로 두지 않고:
Group valid = min(max(original_extent - group_origin, 0), group_size)
```

이유는 mask를 Block 크기에 남겨 두면 Group으로 확대된 뒷부분을 읽지 못하기 때문이다.

### 7.2 Execution Tile과 round

[PIMFormExecutionTilesPass.cpp](../lib/Conversion/TTSharedToPIM/PIMFormExecutionTilesPass.cpp)는
주어진 Group 개수를 기록한다. 용량을 채우거나 비용으로 크기를 탐색하는 pass가 아니다.
[PIMMapDevicesPass.cpp](../lib/Conversion/TTSharedToPIM/PIMMapDevicesPass.cpp)는 각 축에
다음 식을 적용한다.

```text
rounds = ceildiv(ceildiv(total_groups, groups_per_tile), partitions)
first  = (round * partitions + device_coordinate) * groups_per_tile
trips  = min(groups_per_tile, max(total_groups - first, 0))
```

M/N은 출력 Group 영역, K는 같은 reduction domain의 부분 범위다. `Pk>1`과 K round는
각각 공간과 시간 분할이며 별도의 중복 K domain이 아니다.

현재 Host round 생성 순서는 존재하는 축에 대해 n → m → reduction_k로 고정되어 있다.
비용 기반 loop interchange는 구현하지 않았다. canonical 논리 Device 번호
`((pm*Pn)+pn)*Pk+pk`, Host round 순서, 실제 물리 DPU 배정은 서로 다른 개념이다.

### 7.3 부분합을 명시하는 이유

K를 나누면 매 `(round,pk)`가 zero에서 시작한 partial을 만든다. 원래 reduction 결과를
저장하던 store를 `pim.combine ... kind "add"`로 바꾼다. `Pk=1`이어도 K round가 여러
개일 수 있으므로 같은 표현을 쓴다. partial에 비선형 epilogue를 먼저 적용하면 최종
결과에 적용하는 것과 달라지므로 이 단계는 그러한 epilogue를 거부한다.

## 8. Placement와 shared_over 분석

구현: [PIMPlaceDataPass.cpp](../lib/Conversion/TTSharedToPIM/PIMPlaceDataPass.cpp).

1. access의 origin/stride를 현재 Device의 첫 Group 좌표로 다시 만든다.
2. 해당 buffer 차원을 움직이는 local loop들의 범위에서 `size`를 구한다.
3. 실제 trip 수 대신 계층의 고정 Group 개수로 `allocation_shape`를 구한다.
4. 원본 extent를 사용해 tail의 유효 크기를 자른다.
5. 입력 origin/stride/valid/pad의 loop 의존성을 모아 사용하지 않는 축을 `shared_over`로 둔다.

```text
GEMM A: origin[m,k], stride/valid/pad가 n에 불변 → shared_over=[n]
GEMM B: origin[k,n], stride/valid/pad가 m에 불변 → shared_over=[m]
GEMV x: origin[k],   stride/valid/pad가 m에 불변 → shared_over=[m]
```

`size`는 이번 round에서 의미 있는 원소 수이고 `allocation_shape`는 모든 round에서
유지할 고정 공간이다. tail 때문에 두 값이 달라질 수 있다. UPMEM은 고정 공간으로
전송하고 `size`로 유효한 패킹/출력 복원을 제한한다.

계층에 없는 runtime loop가 resident extent를 결정하거나 partition된 여러 축이 한
buffer 차원을 함께 움직여 단일 영역으로 표현할 수 없으면 진단한다. 현재 access/
placement 구현을 임의의 불규칙 gather/scatter 지원으로 해석하면 안 된다.

## 9. Device 내부 load 이동과 Host 계획

### 9.1 기존 local load 재사용

[PIMReuseExecutionTileInputsPass.cpp](../lib/Conversion/TTSharedToPIM/PIMReuseExecutionTileInputsPass.cpp)는
입력 access의 모든 operand가 M/N Group loop에 불변이고, 결과가 읽기 전용으로만
사용되는 경우 access/load를 그 loop 앞으로 옮긴다. 알 수 없는 memory effect나 입력
slot 쓰기가 있으면 이동하지 않는다. 넘는 중간 loop는 singleton이어야 한다.

이유는 같은 Device의 동일 resident 입력을 매 Group마다 WRAM에 다시 읽지 않기
위함이다. 실행 여부가 불명확한 loop는 SCF 보호 loop로 빈 실행을 보존한다. 이동된
access의 전역 origin은 유지하며 Target이 `origin-place.start`로 로컬화한다.

이 pass는 Host round, H2D payload, 물리 배정, Group 크기나 계산 loop 순서를 바꾸지
않는다. Host pool 공유, MRAM→WRAM 읽기 재사용, 이전 round의 H2D 생략을 구분한다.

### 9.2 명시적인 실행 계획 생성

[PIMFormHostPlanPass.cpp](../lib/Conversion/TTSharedToPIM/PIMFormHostPlanPass.cpp)는
placement와 부수 효과 없는 scalar 설정을 Host 쪽에 둔다. 나머지 계산 및 그 계산에
의존하는 Op를 launch로 옮긴다. 계산 결과에 placement가 의존하면 진단한다.

```text
pim.device                         pim.partition
  scalar setup                       scalar setup + place
  place                    →         입력마다 broadcast(shared) 또는 scatter
  compute                            launch { compute }
                                     출력마다 gather
```

Host buffer alloc/dealloc은 UPMEM이 scheduling loop 밖에 만들며 입력 패킹/전송은
round 안에 둔다. 최종 계획에는 round를 넘는 전송 생략 지시를 두지 않는다.

## 10. 검증 범위와 현재 지원 한계

[PIMVerifyPass.cpp](../lib/Conversion/TTSharedToPIM/PIMVerifyPass.cpp)는 계층 크기,
access/place id·rank 대응, 고정 allocation과 메모리 예산, zero padding,
semantic/local K와 scheduling marker의 대응, final output의 K 독립성,
partial 결과의 combine 연결을 검사한다.

`PartitionOp::verify`와 전송 verifier는 입력별 전송/출력별 회수의 누락·중복,
launch 개수와 전송 순서, broadcast의 공유 축 존재를 검사한다.

현재 verifier는 scheduling 산술 전체가 위 수식과 동치인지 증명하는 분석기가 아니다.
정확한 first/trips 생성은 생성 pass와 그 테스트가 담당한다. Common verifier 통과도
Target의 모든 계산 연산 지원을 보장하지 않는다.

| 경우 | 현재 UPMEM 연결 상태 |
|---|---|
| named `linalg.matmul`, `linalg.matvec` | 지원되는 shape/type/계층 계약에서 lowering 경로 존재 |
| 위 5.2절 GEMV 패턴 | Common이 matvec로 정규화 |
| INT8 곱셈 뒤 INT32 sum인 현재 gemv.py | cast가 끼는 패턴 미지원; broadcast가 남아 실패 |
| identity/all-parallel elementwise | scalar C emitter 지원 범위에서 가능 |
| arbitrary broadcast/transpose generic | 일반 계산 경로에서 미지원 |
| K-only reduce | 현재 UPMEM은 tasklet 1개 요구 |
| 임의 convolution/불규칙 gather/fused epilogue | Common 표식 일부를 분석할 수 있어도 전체 지원을 뜻하지 않음 |

다음 [UPMEM Target 설계](UPMEMTarget_Implementation.md)는 이 Common 계약이 실제
Host/Device 코드와 물리 배정으로 변환되는 근거를 설명한다.
