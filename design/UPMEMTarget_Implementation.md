# PIM Common에서 UPMEM으로의 Lowering 설계

범위는 [PIM Common IR](PIMCommonIR_Design.md), UPMEM IR, Host/Device 코드 생성이다.
현재 구현을 설명하며 성능 개선을 이미 실측한 것처럼 기술하지 않는다. 아래 의사
코드는 SSA 상수·타입·심볼 선언을 일부 생략한다.

## 1. Target IR을 따로 두는 이유

**내용**

Common이 정한 논리 계산과 데이터 영역을 UPMEM의 메모리 계층, tasklet, 물리 DPU,
SDK 전송 단위로 구체화한다. Target lowering은 Common의 partition 수, Group/Tile
경계, K partial의 의미를 바꾸지 않는다.

**근거**

- Host 원본은 strided layout일 수 있지만 DPU DMA는 연속된 MRAM 타일을 소비한다.
- 같은 입력을 받는 DPU의 논리 관계만으로 실제 rank/member/CI 위치를 알 수 없다.
- Host가 보낼 metadata와 DPU가 읽는 metadata는 같은 ABI를 사용해야 한다.
- WRAM 공유, tasklet 행 분할, DMA 완료 동기화는 Target의 책임이다.

**예시**

```text
Common 최종 IR: partition + place + broadcast/scatter + launch + gather
    │ convert-pim-to-upmem
    ▼
UPMEM IR: upmem.* + memref/scf/linalg/arith
    ├─ upmem-to-host   → Host C++: pool, 준비, SDK 전송, 실행, 회수/합산
    └─ upmem-to-device → DPU C: MRAM/metadata 선언, WRAM, DMA, 계산, barrier
```

연산 의미를 나타내는 `linalg.matvec` 등은 Target IR에도 남고 Device 번역기가 C로
내린다. UPMEM IR 생성 성공과 최종 C 생성 성공은 별도의 단계다.

## 2. Op를 둔 근거

정의: [UPMEMDialect.td](../include/triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.td).

| Op | 필요한 사실/동작 | 구체적인 소비 |
|---|---|---|
| `upmem.mram` | DPU의 고정 MRAM slot과 공유 심볼 | Device 전역 배열, Host 전송 대상 이름 |
| `upmem.host_var` | Host가 Device에 제공할 scalar | 공통 metadata 구조체의 필드 |
| `upmem.acquire` | 프로그램이 적재된 DPU set | runtime의 확보/적재 및 set 재사용 |
| `upmem.bind_devices` | 논리 묶음과 실제 DPU의 대응 | SDK 열거 번호 → 논리 번호 표 |
| `upmem.foreach_unit` | 논리 Device별 Host 준비 | 좌표 loop, 전송 포인터 배열과 metadata 준비 |
| `upmem.stage` | 원본 영역을 타일 순서로 패킹 | 공유 pool slot 채우기와 zero padding |
| `upmem.to_mram` | 해당 DPU 입력 slot 전송 | 포인터 등록 + 입력별 집합 push |
| `upmem.to_host_var` | DPU별 scalar 값 기록 | metadata 필드 대입 후 통합 전송 |
| `upmem.launch` | 현재 입력으로 Device 실행 | 동기식 SDK launch |
| `upmem.gather` | MRAM 출력을 논리 위치에 회수 | assignment를 사용한 D2H |
| `upmem.foreach_group` | 회수된 각 논리 DPU 결과 처리 | canonical 좌표로 출력 pool 순회 |
| `upmem.unstage` | 원본 layout 복원, 필요 시 누산 | 첫 partial 대입, 이후 partial 덧셈 |
| `upmem.device_func` | DPU에서 실행할 본문 | Device main |
| `upmem.tasklet_id` | 현재 tasklet의 작업 번호 | 행과 DMA chunk 분담 |
| `upmem.get_host_var` | 전달된 scalar 읽기 | metadata 구조체 필드 참조 |
| `upmem.barrier` | WRAM 적재/소비 완료 경계 | tasklet 동기화 |
| `upmem.mram_read/write` | MRAM↔WRAM 연속 DMA | 정렬·크기 제약이 있는 Device 전송 |

Host pool 수명은 기존 `memref.alloc/dealloc`, 반복은 `scf.for`, 계산은 Linalg를
사용한다. 같은 의미의 별도 PIM allocation/compute/loop Op를 추가하지 않는다.

`!upmem.assignment`는 배정표이며 `!upmem.set`은 확보한 집합, `!upmem.unit`은
foreach 본문의 DPU 손잡이다. 실제 Device 메모리는 `#upmem<space mram>` 또는
`#upmem<space wram>`으로 구별한다.

현재 정의에는 `upmem.broadcast`나 `upmem.num_tasklets` Op가 없다. Common broadcast는
아래의 공유 포인터 전송으로 구현하고, tasklet 수는 `upmem.lanes`로 전달한다.

## 3. Lowering 진입에서 읽는 패턴

구현: [PIMToUPMEMPass.cpp](../lib/Conversion/PIMToUPMEM/PIMToUPMEMPass.cpp).

| Common 입력 | 찾는 방법 | 생성하는 표현/이유 |
|---|---|---|
| `pim.partition` | body의 place/action/launch/gather | Host와 Device를 이미 구분한 계획을 소비 |
| 입력 전송 | broadcast/scatter의 access id로 place 조회 | 입력의 개수·이름에 의존하지 않는 stage/to_mram |
| 출력 회수 | gather의 access id로 place 조회 | 출력마다 gather/unstage |
| Group tile 크기 | launch 안의 `pim.access.sizes` | WRAM 계산과 tile-major MRAM의 눈금 |
| local M/N loop | `pim.axis`와 `pim.device_origin` | 실행 횟수 전달과 로컬 좌표화 |
| local K와 Host K round | `pim.temporal_reduction`, `pim.scheduling_axis="reduction_k"` | Common이 이미 분할한 K 범위 유지 |
| partial 결과 | `pim.combine`의 destination/kind | 출력별 Host add 결합 여부 |

미변환 `pim.device`, 누락된 compute loop, 잘못된 profile/topology, 지원하지 않는
combiner는 진단한다. Common 실행 계획을 Target에서 다시 만드는 경로는 없다.

## 4. 고정 MRAM과 Host pool

### 4.1 MRAM 크기는 allocation_shape로 결정

**찾는 정보**: 각 place의 정적 `allocation_shape`와 원소 타입.

```text
MRAM bytes(input/output i) = product(allocation_shape_i) × element_bytes_i
필요 MRAM 합계            = Σ MRAM bytes(i)
```

**변환**: place마다 정확한 크기의 `upmem.mram`을 만들고 profile 예산과 비교한다.

**이유**: Execution Tile slot은 모든 round에서 같은 크기로 사용한다. 원본 전체 행렬
크기나 이번 tail의 size를 MRAM 할당 크기로 사용하지 않는다. 메모리 예산을 A/B/C
개수로 균등 분할하지도 않는다.

```text
allocation_shape = [256,2048], dtype=i8 → MRAM slot 512 KiB
마지막 round size = [100,300]이어도 slot은 512 KiB
```

MRAM 심볼은 적재된 Device 프로그램의 고정 공간이다. 입력 내용은 각 round에서 다시
전송한다. 이전 호출/round의 입력이 유효하다고 간주해 H2D를 생략하지 않는다.

### 4.2 Host pool의 고유 slot

```text
입력 pool slot 수 = product(P_axis for axis not in shared_over)
출력 pool slot 수 = Pm × Pn × Pk
```

| 예시 | shared_over | slot 수 |
|---|---|---:|
| GEMM A | n | Pm × Pk |
| GEMM B | m | Pn × Pk |
| GEMV x | m, Pn=1 | Pk |
| DPU별 출력 partial | 공유하지 않음 | Pm × Pn × Pk |

**변환**: acquire, bind_devices, pool alloc은 scheduling loop 밖에 만들고 dealloc은
그 뒤에 둔다. stage와 gather는 그 pool의 slot을 채운다.

**이유**: 같은 입력의 DPU별 복사본을 Host에 여러 개 만들 필요가 없다. 공유 주소는
물리 전송 묶음에서 같은 cache line을 재사용할 수 있는 조건이기도 하다. pool 공간
재사용은 round 간 전송 생략과 다른 기능이다.

```text
acquire → bind_devices → pool 할당
for scheduling round:
    고유 입력 slot마다 초기화 + 패킹
    입력별 전송 + metadata 전송
    launch
    gather + unstage/combine
pool 해제
```

### 4.3 stage가 찾고 만드는 주소

place의 `start/size/strides/allocation_shape`는 stage의
`origin/sizes/strides/pad_to`로, access의 `sizes`는 `tile`로 전달한다.

```text
pool_key = shared_over 축을 제외한 논리 좌표의 평탄화
buffer   = pool + pool_key × slot_elements
```

공유 축 좌표가 모두 0인 대표 Device에서 slot을 한 번 채우고 나머지는 같은 주소를
등록한다. 각 round마다 slot 전체를 0으로 초기화한 뒤 유효 영역만 패킹한다. tail이
줄어들어도 이전 round 내용이 padding에 남지 않게 하기 위함이다.

MRAM 내부 배치는 tile-major다.

```text
tile_number[d] = local_origin[d] / tile[d]
tile_count[d]  = allocation_shape[d] / tile[d]
MRAM 위치      = flatten(tile_number, tile_count) × product(tile)
                 + 타일 내부 row-major 위치
```

GEMM B의 타일 내부 순서는 `[Ktile,Ntile]`이며 계산은 `A[m,k]*B[k,n]`이다.
Host stage는 SDK의 byte interleaving을 미리 수행하지 않는다.

## 5. 공유 정보를 물리 DPU 배정에 적용

### 5.1 컴파일 시 논리 묶음 선정

**찾는 정보**: partition `[Pm,Pn,Pk]`, 입력별 `shared_over`, fixed-slot bytes.
입력이 A인지 B인지 검사하지 않는다.

후보 `G=[gm,gn,gk]`는 각 원소가 1/2/4/8이고 해당 partition의 약수이며,
`gm*gn*gk <= 8`이다. 같은 member의 8개 CI 자리를 정수 개의 묶음으로 채울 수 있다.

```text
N = Pm × Pn × Pk
공유 배수_i(G) = product(g_axis for axis in shared_over_i)
비용(G)        = Σ [ bytes_i × N / 공유 배수_i(G) ]
```

최소 비용을 선택하고 비용이 같으면 더 큰 묶음을 선택한다. 그것도 같으면 현재 후보
열거 순서의 첫 항목을 유지한다.

**이유**: 같은 SDK 전송 묶음에서 반복 참조할 Host 원본 주소를 늘려 고유 입력 바이트를
줄이기 위한 모델이다. cache miss 수나 transfer latency를 직접 예측하는 식이 아니다.
Host cache 용량, associativity, NUMA, SDK worker 수를 비용식에 넣지 않는다.

**예시**: `P=[4,128,4]`, A/DPU=512 KiB, B/DPU=448 KiB.

| 묶음 G | 8 DPU에서 고유한 입력 | 묶음당 모델 비용 |
|---|---|---:|
| `[1,8,1]` | A 1개 + B 8개 | 4096 KiB |
| `[2,4,1]` | A 2개 + B 4개 | 2816 KiB |
| `[4,2,1]` | A 4개 + B 2개 | 2944 KiB |

이 경우 `[2,4,1]`을 고른다. 실제 H2D payload는 배정 전후 동일하다. 비용 감소를
전송 바이트 감소나 동일 비율의 시간 개선으로 해석하지 않는다.

### 5.2 실행 시 실제 topology 연결

[upmem_runtime.c](../pim_backend/backend/upmem_runtime.c)의 `upmem_bind_devices`는
SDK의 `dpu_get_member_id`, `dpu_get_slice_id`로 실제 위치를 읽는다.

```text
한 전송 묶음 = 같은 rank, 같은 member의 서로 다른 CI
              CI 하나 안의 여러 member를 묶는 것이 아님

assignment[SDK enumeration index] = canonical logical index
canonical logical index          = ((pm*Pn)+pn)*Pk+pk
```

8개 CI가 모두 활성인 member를 먼저 순회하고 불완전한 member를 나중에 배정한다.
이유는 중간에 비활성 DPU가 있어도 뒤의 완전한 묶음 정렬이 밀리지 않게 하기 위함이다.
불완전한 묶음에서도 각 논리 Device는 한 번씩 배정하지만 기대 공유도를 모두 달성하지
못할 수 있다. 할당 수/topology가 잘못되면 오류를 낸다.

### 5.3 입력·metadata·gather에 같은 표를 사용

```text
논리 순서로 준비:
    input_ptr[logical] = shared pool slot 주소
    metadata[logical] = 그 좌표의 실행 정보

물리 SDK 순서로 전송:
    prepare(dpu, input_ptr[assignment[physical]])
    prepare(dpu, &metadata[assignment[physical]])

물리 SDK 순서로 회수:
    prepare(dpu, output_pool[assignment[physical]])
```

입력만 순서를 바꾸면 다른 DPU의 metadata를 쓰거나 출력이 잘못 합쳐질 수 있다.
`bind_devices`를 foreach와 gather가 모두 참조하게 한 이유다.

이 변환은 Host의 canonical 준비 loop와 Device의 계산 loop를 교환하지 않는다.
물리 위치와 논리 일을 연결하는 표만 바꾼다. 순열 자체는 추가 push를 만들지 않는다.

## 6. Host 전송과 metadata

구현: [UPMEMToHost.cpp](../lib/Target/UPMEM/UPMEMToHost.cpp).

### 6.1 준비와 push를 분리하는 이유

`foreach_unit`은 먼저 논리 좌표별 scalar/패킹/포인터를 준비한다. 그 뒤 각 입력에
대해 SDK의 DPU_FOREACH로 포인터를 등록하고 set 전체에 한 번 push한다.

```text
입력이 A와 B이면 매 round:
    A: 모든 DPU prepare → A push 1회
    B: 모든 DPU prepare → B push 1회
    metadata: 모든 DPU prepare → 통합 push 1회
    launch
```

따라서 A/B/metadata를 통째로 한 번 push하는 구현은 아니다. B0/B1/... 타일마다
별도로 push하는 구현도 아니다. 한 입력 slot의 서로 다른 DPU 포인터들이 한 집합
전송에 들어간다. Common broadcast도 같은 방식이며 공유 DPU들은 같은 주소를 쓴다.

이 구조는 공유 입력 pool 주소를 유지한다. 모든 DPU에 A+B+metadata를 합친 사본을
만들면 현재의 공유 주소 관계가 없어지므로 단순한 push 수 축소와 같은 선택이 아니다.
현재 구현에는 전송 방식 선택 옵션이나 별도 합본 전송 경로를 추가하지 않았다.

### 6.2 metadata ABI

[MetadataLayout.h](../lib/Target/UPMEM/MetadataLayout.h)는 `upmem.host_var` 선언 순서로
공통 필드 배치와 padding을 정한다. Host/Device가 같은 구조체 정의를 사용한다.
`index`는 전송 시 int32이며 전체 구조체 크기는 8바이트 배수다. 생성 코드에는 크기와
필드 offset을 확인하는 컴파일 시점 검사가 있다.

일반 Common lowering은 모든 필드를 채우므로 metadata를 한 번 보낸다. 손으로 작성한
Target IR이 일부 필드만 갱신하면 기록된 연속 구간만 전송한다. 이는 전달하지 않은
필드를 덮지 않기 위한 기존 ABI 동작이며 round 입력 상주 분석과 관계없다.

### 6.3 Host thread와 시간 해석

`upmem.host_threads`는 stage/unstage의 타일 복사 loop에 적용되는 OpenMP 병렬도다.
SDK 전송 worker 수나 CPU affinity를 뜻하지 않는다. `emitTileLoops`는 복사 크기가
일정 기준을 넘을 때만 이 병렬 loop를 사용한다.

계측은 prepare, h2d, kernel, d2h, reduction, buffer_alloc/free, acquire, buffer_init,
layout_a/b/other/c를 구분한다. h2d에는 SDK prepare+push가 포함된다. `operand_role`은
matmul의 입력 패킹 시간 분류에만 쓰며 물리 배정 비용에는 쓰지 않는다. 현재 matvec
입력 패킹은 layout_other로 집계된다. 계측 구간은 SDK의 cache miss 자체를 측정하지 않는다.

## 7. Device 계산과 DMA 패턴

### 7.1 tensor를 WRAM으로 바꾸기

`collectLocalTiles`가 load 결과와 Linalg outs의 buffer 소유 관계를 구한다. tensor
iter_arg는 초기 accumulator와 같은 WRAM 공간을 사용한다. Linalg를 memref 형태로
다시 만들고 값 전달용 tensor iter_arg를 제거한다.

이유는 매 K 반복마다 새 accumulator를 할당하지 않고 같은 WRAM을 갱신하기 위함이다.
local loop 상한/pitch 등 Device 밖에서 오는 정수는 host_var 통로로 전달한다. 현재
외부 runtime float scalar를 이 통로로 자동 전달하는 일반 경로는 없다.

### 7.2 전역 좌표를 로컬 MRAM 좌표로 바꾸기

보통 `pim.device_origin`의 전역 시작을 0으로 바꾸면 DPU 안의 tile 번호가 된다.
Common의 local load 이동으로 이 표식이 사라질 수 있는 reused input은
`origin-place.start`로 명시적으로 로컬화한다. 그렇지 않으면 다음 round의 전역
시작이 작은 MRAM slot 안의 offset으로 사용될 수 있다.

### 7.3 tasklet 분할은 indexing map으로 결정

지원되는 Linalg operand의 indexing map에서 출력 m이 leading dimension이면 행을
나누고, m을 사용하지 않으면 WRAM tile 전체를 공유한다. m이 다른 위치에 있거나 map이
지원되는 projected permutation이 아니면 진단한다. 출력 행 수는 tasklet 수로
나누어떨어져야 한다.

| 계산 | 행을 나누는 operand | 공유하는 operand |
|---|---|---|
| matmul | A, C | B |
| matvec | A, y | x |
| identity pointwise | 모든 동일 shape operand | 없음 |

예를 들어 matvec의 A[8,16], x[16], y[8]을 tasklet 4개가 처리하면 A/y는 각자 2행,
x는 같은 16원소다. 이는 Device 내부 WRAM 공유이며 DPU 간 Host x pool 공유와 다른
단계다.

### 7.4 DMA와 barrier

`buildDMA`는 타일을 연속 MRAM 구간으로 읽고 쓴다. 현재 구현은 2048바이트 이내의
행 묶음을 고르며, tasklet t는 chunk t, t+lanes, ...를 담당한다. 타일이 작아 chunk가
하나이면 실제 DMA는 tasklet 하나만 수행할 수 있다.

DMA verifier는 크기·정렬 제약을 확인한다. 한 행 자체가 지원 전송 크기를 넘거나
정렬 조건을 만족하지 못하는 shape를 임의로 다른 layout으로 바꾸지 않는다.

```text
협력 input DMA
    ↓ barrier: 모든 입력 chunk 도착
tasklet별 담당 행 계산
    ↓ barrier: 반복 계산의 입력 소비 완료
다음 K 반복의 DMA
```

뒤 barrier는 다음 반복이 공유 WRAM 입력을 덮기 전에 소비가 끝나도록 한다.

### 7.5 계산 leaf의 지원 범위

[UPMEMToDevice.cpp](../lib/Target/UPMEM/UPMEMToDevice.cpp)는 fill/matmul/matvec/reduce와
identity-map/all-parallel generic을 처리한다. generic은 단일 출력·동일 정적 shape와
[CEmitter.h](../lib/Target/UPMEM/CEmitter.h)가 지원하는 scalar arith를 요구한다.
`math.exp`, 일반 broadcast generic, 임의 transpose 등을 자동 지원하지 않는다.

K-only reduce는 tasklet-local partial을 합치는 표현이 없으므로 tasklet 1개를 요구한다.
단순히 Common에 reduction 표식이 있다는 이유로 임의 병렬 reduction이 되는 것은 아니다.

## 8. Gather와 partial 결합

Common의 출력 `allocation_shape`만큼 DPU마다 회수하되 assignment로 논리 pool 위치를
정한다. `foreach_group`은 그 pool을 canonical 논리 순서로 순회하고 unstage가 원본
strided 영역에 복원한다.

```text
pim.store 출력:
    C[destination] = received

pim.combine(kind=add) 출력:
    if reduction_round == 0 and pk == 0:
        C[destination] = received
    else:
        C[destination] += received
```

물리 배정이 바뀌어도 논리 partial 결합 순서는 유지된다. destination은 Common의
place/access가 정하며 Target이 output rank에서 split-K plane을 추측하지 않는다.

## 9. GEMV 전체 예시

f32 A[1024,1024], x[1024], y[1024], P=[8,1,8], Block=[8,1,16],
Group=[1,1,1], Execution Tile groups=[16,1,8]을 가정한다.

```text
DPU(pm,pk)의 입력: A[128*pm : +128, 128*pk : +128], x[128*pk : +128]
DPU(pm,pk)의 출력: y_partial[128]
Common shared_over: A=[], x=[m]
```

배정 비용은 A에는 공유가 없고 x의 m 공유만 이득이므로 G=[8,1,1]을 고른다. 완전한
rank에서 같은 member의 CI0..7은 같은 pk, 서로 다른 pm을 받는다.

```text
member 0: (m0,k0) (m1,k0) ... (m7,k0) → 모두 x_pool[0]
member 1: (m0,k1) (m1,k1) ... (m7,k1) → 모두 x_pool[1]
...
member 7: (m0,k7) (m1,k7) ... (m7,k7) → 모두 x_pool[7]
```

Host는 A pool 64 slot, x pool 8 slot, y pool 64 slot을 사용한다. Device는 16개의
M Group과 8개의 K Group을 순회하며 WRAM matvec를 계산한다. gather 후 같은 pm의
pk=0..7 partial을 더한다.

A payload는 4 MiB이고 x payload는 복제분 포함 32 KiB다. 입력 공유 모델이 개선되어도
A가 대부분이므로 GEMM에서 큰 입력을 공유할 때와 같은 시간 감소를 보장하지 않는다.

현재 `gemv.py` 예제(저장소에 포함되지 않는 로컬 예제)의 INT8 broadcast 오류는 이 물리 배정이 실행되기 전의 계산
정규화 문제다. 곱셈 후 INT32 확장이 끼는 패턴과 의미 차이는
[Common 설계 5절](PIMCommonIR_Design.md#5-계산-패턴을-어떻게-찾는가)에 설명했다.

## 10. 검증 자료와 남은 확인

아래는 저장소에 존재하는 테스트의 목적이며 이번 문서 정리에서 실행한 결과가 아니다.

| 테스트 | 확인 목적 |
|---|---|
| `test/Conversion/PIMToUPMEM/convert.mlir` | 기본 Common→Target/Host/Device 연결 |
| `test/Conversion/PIMToUPMEM/matvec.mlir` | f32 named matvec, DPU 1개/tasklet 4개, x WRAM 공유 |
| `test/Conversion/TTSharedToPIM/form_compute_matvec.mlir` | f32 frontend 패턴 정규화 |
| `test/Conversion/PIMToUPMEM/elementwise_pipeline.mlir` | 동일 shape add+mul 전체 변환 경로 |
| `test/Conversion/PIMToUPMEM/device_mapping.mlir` | 공유 축·입력 바이트에 따른 GEMM 묶음 선택 |
| `test/Target/UPMEM/device_mapping_runtime.mlir` | mock topology에서 배정의 일대일 대응과 비활성 DPU 처리 |
| `test/Target/UPMEM/host_pool_sharing.mlir` | 고유 입력 slot 패킹과 주소 공유 |
| `test/Target/UPMEM/metadata_bundle.mlir` | Host/Device metadata ABI와 통합 전송 |

GEMV의 다중 DPU·split-K·물리 배정을 합친 수치 검증, INT8 선행 확장 패턴 지원,
실제 SDK에서 배정 변경 전후 H2D 비교는 별도 확인이 필요하다. 컴파일 비용식은 cache
재사용의 가설을 제공하며 성능 측정이나 cache miss 측정을 대신하지 않는다.

사용자가 수정한 도구를 빌드한 후 기존 matvec 경로만 살펴보는 명령 예시는 다음과 같다.
도구가 PATH에 등록되어 있고 저장소 루트에서 실행하는 경우다.

```sh
triton-shared-opt --pim-form-host-plan --convert-pim-to-upmem \
  test/Conversion/PIMToUPMEM/matvec.mlir \
  | triton-shared-translate --upmem-to-device
```

예상 생성 코드는 tasklet당 2행, K 16원소의 곱셈·누산 loop다. 이 f32 예제의 성공과
현재 INT8 `gemv.py`의 성공을 같은 것으로 취급하지 않는다.
