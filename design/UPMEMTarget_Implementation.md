# PIM Common에서 UPMEM으로의 Lowering 설계

범위는 [PIM Common IR](PIMCommonIR_Design.md)의 `pim.partition`을 UPMEM IR로 바꾸는
`convert-pim-to-upmem`, UPMEM IR을 Host C++/DPU C로 바꾸는 번역기, 그리고 생성 코드가
호출하는 [upmem_runtime.c](../pim_backend/backend/upmem_runtime.c)다. 모델 단위 상주
실행은 [ModelRuntime 설계](ModelRuntime_Design.md)가 다룬다. 아래 의사 코드는 SSA 상수·
타입·심볼 선언을 일부 생략한다.

## 1. Target lowering의 역할

**내용**

Common이 정한 논리 분할, Group/Execution Tile 경계, K partial 의미를 UPMEM의 MRAM/WRAM,
tasklet, 물리 DPU 위치, SDK 전송 단위로 구체화한다. Target은 partition 수와 K 범위를
바꾸지 않는다.

**근거**

- Host 원본은 strided layout이지만 DPU DMA는 연속 MRAM 구간만 읽는다.
- 논리 공유 관계만으로는 실제 rank/member/CI 위치를 알 수 없다.
- Host가 쓰는 metadata와 DPU가 읽는 metadata는 같은 ABI여야 한다.
- WRAM 공유, tasklet 행 분할, DMA 완료 동기화는 Target의 책임이다.
- [compiler.py](../pim_backend/backend/compiler.py) `_pim_common_to_upmem`이
  `--convert-pim-to-upmem=host-threads=<host_threads> nr-tasklets=<nr_tasklets>`(기본 8, 0)를
  실행하고, `_upmem_to_artifact`가 `--upmem-to-host`/`--upmem-to-device` 결과와 device 소스의
  `// pim.lanes = N`에서 읽은 tasklet 수를 `{host, device, target_ir, nr_tasklets}` 번들로 묶는다.

**예시**

```text
Common 최종 IR: partition { place, broadcast/scatter, launch, gather }
    │ convert-pim-to-upmem (host-threads, nr-tasklets)
    ▼
UPMEM IR: upmem.* + memref/scf/linalg/arith
    ├─ upmem-to-host        → Host C++: pool, 패킹, 전송, launch, 회수/합산
    ├─ upmem-to-model-host  → 모델 세션용 Host C++ (7.4절)
    └─ upmem-to-device      → DPU C: MRAM/metadata 선언, WRAM, DMA, 계산, barrier
```

## 2. UPMEM Op

정의: [UPMEMDialect.td](../include/triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.td).

**내용**

UPMEM dialect는 DPU 집합, Host 준비·전송, Device 메모리·동기화만 표현한다. 반복은
`scf.for`, Host pool 수명은 `memref.alloc/dealloc`, 계산은 Linalg를 그대로 쓴다.

**근거**

- 같은 의미의 PIM 전용 allocation/loop/compute Op를 두지 않는다.
- Device 메모리는 `#upmem<space mram>`, `#upmem<space wram>`으로 구별한다.

**예시**

| Op | 역할 | 번역 결과 |
|---|---|---|
| `upmem.mram` | DPU 고정 MRAM slot 심볼 | `__mram_noinit T name[n]`, Host 전송 대상 이름 |
| `upmem.host_var` | Host가 쓰는 DPU scalar | 공통 metadata 구조체 필드 |
| `upmem.acquire` | 프로그램이 적재된 DPU 집합 | `upmem_acquire(n, DPU_BINARY)` |
| `upmem.bind_devices` | 물리 순번 → 논리 번호 표 | `upmem_bind_devices(...)` |
| `upmem.foreach_unit` | 논리 DPU별 Host 준비 | 좌표 loop + 포인터·metadata 배열 |
| `upmem.stage` | 원본 영역을 tile-major slot에 패킹 | `memset` + tile loop 복사 |
| `upmem.to_mram` | 입력 slot을 DPU MRAM에 전송 | 입력별 `prepare` + `push` |
| `upmem.to_host_var` | DPU별 scalar 기록 | metadata 필드 대입 |
| `upmem.launch` | 동기 실행 | `dpu_launch(_set, DPU_SYNCHRONOUS)` |
| `upmem.gather` | MRAM 출력 회수 | assignment 순서의 `prepare` + `push FROM_DPU` |
| `upmem.foreach_group` | 회수된 slot을 논리 순서로 순회 | 좌표 loop |
| `upmem.unstage` | 원본 layout 복원, 필요 시 누산 | `=` 또는 `+=` tile loop |
| `upmem.device_func` | DPU 본문 | `main()` |
| `upmem.tasklet_id` | 현재 tasklet 번호 | `me()` |
| `upmem.get_host_var` | metadata 필드 읽기 | 구조체 필드 참조 |
| `upmem.barrier` | tasklet 동기화 | `barrier_wait(&pim_barrier)` |
| `upmem.mram_read/write` | MRAM↔WRAM 연속 DMA | `mram_read`/`mram_write` |

`!upmem.set`은 확보한 집합, `!upmem.unit`은 foreach 본문의 DPU 손잡이,
`!upmem.assignment`는 물리 SDK 순번 → canonical 논리 번호 표다.

## 3. Lowering 진입 조건과 읽는 정보

구현: [PIMToUPMEMPass.cpp](../lib/Conversion/PIMToUPMEM/PIMToUPMEMPass.cpp).

**내용**

진입 시 profile의 합법성을 확인한 뒤, `pim.partition` 본문의 place/전송/launch/gather와
loop 표식을 읽는다. Common 실행 계획을 Target에서 다시 만드는 경로는 없다.

**근거**

- `checkLegality`: `target="upmem"`, `banks_per_unit=1`, `cores_per_bank=1`,
  `bank_capacity_limit_bytes > 0`, `1 ≤ lanes ≤ lanes_per_core`를 요구한다. `lanes`는
  `nr-tasklets` 옵션이고 0이면 `lanes_per_core`다.
- 미변환 `pim.device`, source `pim.axis="k"`, temporal K loop와 `reduction_k` round 중
  하나만 있는 경우, `add` 외 combiner는 진단한다.
- M축이 없는 K-only `linalg.reduce`는 tasklet-local partial 결합이 IR에 없으므로
  `lanes=1`을 요구한다.

**예시**

| Common 입력 | 찾는 방법 | 생성 결과 |
|---|---|---|
| 입력 placement | broadcast/scatter의 access id | `upmem.stage` + `upmem.to_mram` |
| 출력 placement | gather의 access id | `upmem.gather` + `foreach_group` + `unstage` |
| Group tile 크기 | launch 안 `pim.access.sizes` | WRAM buffer shape, MRAM tile 눈금 |
| local M/N loop | `pim.axis` + `pim.device_origin` | 상한은 metadata, origin은 0 기준 |
| local K loop | `pim.temporal_reduction` + `reduction_k` round | origin 0 기준, 상한은 Host 값 전달 |
| partial 결과 | `pim.combine` destination/kind | 출력별 Host add 결합 |
| 계측 역할 | `linalg.matmul` 입력 0/1의 access | `operand_role` = `lhs`/`rhs` |

## 4. Tiling 계층과 UPMEM 자원의 대응

**내용**

Common의 4단계 계층은 UPMEM에서 다음 자원에 대응한다. 한 DPU는 한 round에 Execution
Tile 하나를 MRAM에 받고, 그 안의 Group을 하나씩 WRAM에 올려 모든 tasklet이 함께
계산한다.

**근거**

- Execution Tile은 `pim.place.allocation_shape`로 MRAM slot 크기가 된다(5절).
- Group은 `pim.access.sizes`로 WRAM buffer와 DMA 단위가 된다(8절).
- partition은 `upmem.acquire`의 DPU 수가 된다. `numUnits = Pm×Pn×Pk`.

**예시**

```text
Block  [Bm,Bn,Bk]            Triton BLOCK_*        (Device 코드에는 별도 단위 없음)
Group  Block×[Gm,Gn,Gk]       WRAM tile 1벌         tasklet들이 행으로 나눠 계산
ExecTile Group×[Tm,Tn,Tk]     MRAM 고정 slot        DPU local loop m/n/k
Partition [Pm,Pn,Pk]          DPU Pm·Pn·Pk개        Host round마다 전체 launch 1회

Host:  for n_round { for m_round { for k_round {
         stage+push(입력) → launch → gather → unstage(=, +=) } } }
DPU :  for m_local < tiles_m { for n_local < tiles_n {
         fill C_wram; for k_local < trips_k { DMA A,B → matmul } ; DMA C } }
```

## 5. MRAM 고정 slot과 Host pool

### 5.1 MRAM 크기

**내용**

place마다 `product(allocation_shape) × elem_bytes` 크기의 `upmem.mram`을 만든다. 모든
place의 합이 `bank_capacity_limit_bytes`를 넘으면 진단한다.

**근거**

- Execution Tile slot은 모든 round에서 같은 크기로 쓴다. 원본 행렬 크기나 마지막 tail의
  `size`를 할당 크기로 쓰지 않는다.
- 진단 문구: `the execution tile needs N byte of MRAM but the budget is B byte — lower
  tile_m, tile_n, or tile_k`.

**예시**

```text
allocation_shape=[256,1792], i8 → MRAM slot 448 KiB
마지막 round size=[100,300]이어도 slot은 448 KiB
```

### 5.2 Host pool slot

**내용**

입출력마다 Host pool을 하나 할당한다. 입력 pool은 `shared_over` 축을 뺀 좌표 조합마다
slot 하나, 출력 pool은 논리 DPU마다 slot 하나다. acquire·bind·pool alloc은 scheduling
loop 밖에, dealloc은 그 뒤에 둔다.

**근거**

- 같은 입력의 DPU별 복사본을 Host에 만들지 않는다. 공유 DPU들은 같은 Host 주소를
  전송한다.
- pool 공간은 round 사이에 재사용하지만 내용은 round마다 다시 채운다.

**예시**

| 버퍼 | shared_over | slot 수 |
|---|---|---|
| GEMM A | n | Pm × Pk |
| GEMM B | m | Pn × Pk |
| GEMV x | m | Pk |
| GEMV A | 없음 | Pm × Pk |
| 출력 partial | 출력은 공유 없음 | Pm × Pn × Pk |

### 5.3 stage: tile-major 패킹

**내용**

`upmem.stage`는 원본 strided 영역을 Group tile 단위로 연속된 slot에 복사한다. 공유 축
좌표가 모두 0인 대표 DPU에서만 slot을 채우고, 매 round 채우기 전에 slot 전체를 0으로
초기화한다.

**근거**

- [UPMEMToHost.cpp](../lib/Target/UPMEM/UPMEMToHost.cpp) `emitStage`, `tileIndex`,
  `emitTileLoops`.
- 0 초기화로 tail round의 padding에 이전 round 값이 남지 않는다.
- 마지막 tile이 부분 tile이면 tile loop를 올림으로 돌고 tile 안에서 `size` 밖을
  건너뛴다.
- `host_threads > 1`이면 바깥 tile loop에 `#pragma omp parallel for collapse(rank)`를 붙이고,
  (올림 tile 수의 곱 × tile 원소 수) > `host_threads × 512`일 때만 병렬로 돈다.
  `host_threads`는 이 복사 loop의 병렬도이며 SDK 전송 worker 수가 아니다.

**예시**

```text
pool_key = shared_over를 뺀 축 좌표를 [m,n,k] 순서로 평탄화
slot     = pool + pool_key × slot_elements

tile_count[d] = allocation_shape[d] / tile[d]
MRAM 위치     = flatten(tile_number, tile_count) × product(tile) + tile 내부 row-major

A slot [256,1792] i8, Group tile [64,128] → tile_count [4,14], tile 8192 B
 ┌────────┬────────┬─────┬─────────┬────────┬─────┐
 │ t(0,0) │ t(0,1) │ ... │ t(0,13) │ t(1,0) │ ... │   t(i,j) 시작 = (i*14+j)*8192 B
 └────────┴────────┴─────┴─────────┴────────┴─────┘
```

GEMM B의 tile 내부 순서는 `[Ktile, Ntile]`이며 계산은 `A[m,k]*B[k,n]`이다. Host stage는
SDK의 byte interleaving을 미리 수행하지 않는다.

## 6. 논리 DPU와 물리 DPU 배정

### 6.1 컴파일 시 묶음 선택

**내용**

같은 rank·같은 member의 CI 8개가 한 SDK 전송 묶음이다. 이 8개 DPU가 같은 Host 입력
주소를 최대한 공유하도록 묶음 모양 `G=[gm,gn,gk]`를 고른다.

**근거**

- `PIMToUPMEMPass.cpp`: 각 원소는 1/2/4/8이고 해당 partition의 약수, `gm·gn·gk ≤ 8`.
- 비용이 최소인 G를 고른다. 같으면 곱이 큰 G, 그것도 같으면 열거 순서(gm, gn, gk
  오름차순)에서 먼저 나온 G다.
- `BindDevicesOp::verify`: G의 곱이 8의 약수이고 partition 곱이 acquire 수와 같아야 한다.
- 비용은 묶음 안에서 반복 참조되는 Host 주소를 늘리기 위한 모델이다. H2D payload 바이트는
  G와 무관하게 같다. cache miss나 전송 시간을 예측하는 식이 아니다.

**예시**

```text
N            = Pm × Pn × Pk
공유배수_i(G) = product(g_axis for axis in shared_over_i)
비용(G)       = Σ_i bytes_i × N / 공유배수_i(G)
```

`P=[4,32,16]`, A/DPU = B/DPU = 448 KiB (A는 n 공유, B는 m 공유). `Pm=4`라 `gm ≤ 4`다.

| G | 비용 ∝ 1/gn + 1/gm | 결과 |
|---|---|---|
| [1,8,1] | 1.125 | |
| [2,2,2] | 1.0 | |
| [2,4,1] | 0.75 | 선택 (곱 8로 동률, 먼저 열거됨) |
| [4,2,1] | 0.75 | |

### 6.2 실행 시 topology 연결

**내용**

`upmem_bind_devices`는 SDK의 `dpu_get_member_id`, `dpu_get_slice_id`로 각 DPU의
`(member, CI)`를 읽고, 물리 순번(ordinal)을 논리 번호로 바꾼 표를 만든다. CI 8개가 모두
활성인 member를 먼저 배정하고 불완전한 member를 나중에 배정한다.

**근거**

- [upmem_runtime.c](../pim_backend/backend/upmem_runtime.c) `upmem_bind_instances`.
- 불완전한 member를 먼저 배정하면 뒤의 완전한 묶음 정렬이 밀린다. 불완전한 member에서도
  각 논리 DPU는 한 번씩 배정되지만 기대 공유도를 모두 달성하지 못할 수 있다.
- 할당 수가 partition 곱과 다르거나 member/CI id가 범위를 벗어나거나 중복되면 abort한다.

**예시**

```text
물리 순회: rank 순서 → member 0..7 → CI 0..7 (pass 0: 완전 member, pass 1: 나머지)

outer = ordinal / |G|      (몇 번째 묶음)
inner = ordinal % |G|      (묶음 안 위치)
for d in k, n, m:          (마지막 축부터)
    coord[d] = (outer % (P[d]/G[d])) * G[d] + inner % G[d]
    outer   /= P[d]/G[d];   inner /= G[d]
logical = ((coord_m*Pn)+coord_n)*Pk+coord_k
assignment[물리 SDK 순번] = logical
```

`P=[4,32,16]`, `G=[2,4,1]`:

```text
member0 (outer=0): CI0..3=(m0, n0..3, k0)  CI4..7=(m1, n0..3, k0)  → A 2 slot, B 4 slot
member1 (outer=1): 같은 m/n, k1
...
member7          : k7          → rank 0 = (m0..1, n0..3, k0..7)
다음 rank        : k8..15, 이어서 n4..7, ... , m2..3
```

### 6.3 같은 표를 전송·metadata·회수에 사용

**내용**

Host는 논리 순서로 포인터와 metadata를 준비하고, SDK 물리 순서로 전송·회수할 때
assignment로 논리 위치를 찾는다.

**근거**

- 입력만 순서를 바꾸면 다른 DPU의 metadata를 쓰거나 출력이 잘못 합쳐진다. 그래서
  `foreach_unit`과 `gather`가 같은 `bind_devices` 결과를 참조한다.
- 순열 자체는 추가 push를 만들지 않는다.

**예시**

```c
DPU_FOREACH(_set, _dpu, _idx) dpu_prepare_xfer(_dpu, _pA[assign[_idx]]);
dpu_push_xfer(_set, DPU_XFER_TO_DPU, "k_mram_0", 0, bytesA, DPU_XFER_DEFAULT);
DPU_FOREACH(_set, _dpu, _idx) dpu_prepare_xfer(_dpu, C_pool + assign[_idx] * elemsC);
dpu_push_xfer(_set, DPU_XFER_FROM_DPU, "k_mram_2", 0, bytesC, DPU_XFER_DEFAULT);
```

## 7. Host 전송과 metadata

### 7.1 준비와 push 분리

**내용**

`foreach_unit`은 먼저 논리 DPU마다 좌표를 풀고 stage·포인터·metadata를 준비한다. 그
뒤 입력 slot마다 `DPU_FOREACH` prepare와 집합 push를 한 번씩 수행하고, metadata를 push한다.

**근거**

- `dpu_prepare_xfer`는 DPU 단위이고 `dpu_push_xfer`는 집합 단위다(`UPMEMToHost.cpp`
  상단 주석).
- 전송 바이트가 대상 MRAM 심볼보다 크면 번역 단계에서 진단한다.

**예시**

```text
매 round:
  준비 loop: for _u in 논리 DPU { 좌표, stage, _pA[_u], _pB[_u], _args[_u] }
  A: prepare×N → push 1회
  B: prepare×N → push 1회
  metadata: prepare×N → push 1회
  launch
  C: prepare×N → push(FROM_DPU) 1회 → foreach_group unstage
```

### 7.2 metadata ABI

**내용**

[MetadataLayout.h](../lib/Target/UPMEM/MetadataLayout.h)가 `upmem.host_var` 선언 순서로
구조체 필드 배치와 padding을 정하고, Host와 DPU가 같은 정의를 쓴다. `index`는 int32로
전송하고 구조체 크기는 8바이트 배수다.

**근거**

- 생성 코드에 구조체 크기와 필드 offset의 컴파일 시점 검사가 있다.
- 필드 내용: 축별 local loop 상한(`*_tiles_m/n`), MRAM slot의 축 간격(`*_pitch_<id>_<d>`),
  Device에서 다시 만들 수 없는 Host 정수(`*_host_<i>`, 예: local K 상한).
- 모든 필드를 쓰면 push 1회다. 일부 필드만 쓰는 IR이면 연속으로 쓴 구간만 전송해 쓰지
  않은 필드를 덮지 않는다.

**예시**

```c
struct __attribute__((packed, aligned(8))) _pim_metadata_t {
  int32_t f0; // <kernel>_tiles_m     m local Group 수
  int32_t f1; // <kernel>_tiles_n
  int32_t f2; // <kernel>_pitch_0_1   A slot의 축 1 길이
  ...
};
__host struct _pim_metadata_t _pim_metadata;   // DPU 쪽 선언
```

필드는 `f<순번>`이고 주석에 `upmem.host_var` 심볼 이름이 붙는다. 정렬은 각 필드의 자연
정렬이며 부족한 자리는 `_pad<i>` 배열로 채운다.

### 7.3 DPU 집합 확보와 계측

**내용**

`upmem_acquire`는 같은 DPU 수로 이미 확보한 집합을 재사용하고, binary 경로가 바뀐 경우에만
다시 적재한다. Host 코드는 `PIM_TIMING` 구간을 단계별로 잰다.

**근거**

- `upmem_runtime.c` `upmem_runtime_reserve`, `upmem_runtime_load`. 새 집합은
  `dpu_alloc(n, "sgXferEnable=true,sgXferMaxBlocksPerDpu=1024")`로 잡는다.
- 상주 블록이 집합을 소유하면 `upmem_resident_acquire`로 넘기고, session이 열린 상태에서
  상주 블록 없이 호출하면 abort한다(`upmem_acquire`).
- 계측 슬롯: prepare, h2d, kernel, d2h, reduction, buffer_alloc, buffer_free, acquire,
  buffer_init, layout_a, layout_b, layout_other, layout_c. h2d는 prepare+push를 포함한다.
- `operand_role`이 `lhs`/`rhs`인 stage는 layout_a/layout_b, 그 밖의 입력(matvec 포함)은
  layout_other로 집계한다. 첫 partial 대입은 layout_c, 이후 누산은 reduction이다.

**예시**

```text
_pim_start(H2D); push A; push B; push metadata; _pim_stop(H2D);
_pim_start(KERNEL); dpu_launch(...); _pim_stop(KERNEL);
```

같은 이름 순서가 [driver.py](../pim_backend/backend/driver.py) `_STAGE_NAMES["upmem"]`이고,
launcher는 `pim_timing_ms`를 그 순서로 읽는다(`pim_timing_version = 3`).

### 7.4 모델 세션용 Host 번역

**내용**

`upmem-to-model-host`는 같은 UPMEM IR을 `struct upmem_model_kernel *_model`을 첫 인자로 받는
진입 함수로 낸다. 집합·배정·pool·전송은 세션 객체가 소유하고, 생성 코드는 batch 안의 job마다
stage와 metadata만 채운다. 세션 쪽 동작은 [ModelRuntime 설계](ModelRuntime_Design.md)가 다룬다.

**근거**

- `UPMEMToHost.cpp`의 `model` 분기, `emitModelForeachUnit`, `emitStage`, `emitGather`,
  `emitForeachGroup`.
- `upmem_model_wants`는 `upload_weights`가 `UPMEM_MODEL_KV_PACK`이면 KV 역할 binding 중
  선택된 job만, 그 밖에는 weight 또는 input 역할 binding만 참으로 돌려준다
  ([upmem_model.c](../pim_backend/backend/upmem_model.c)).

**예시**

| UPMEM IR | `upmem-to-host` | `upmem-to-model-host` |
|---|---|---|
| 모듈 선언 | 없음 | `pim_model_pool_count`, `pim_model_pool_bytes[]`, `pim_model_metadata_bytes`, `pim_model_device_mapping[]` |
| 함수 본문 | 1회 실행 | `for (batch_base = 0; batch_base < instances; batch_base += batch_size)` |
| `memref.alloc/dealloc` | `malloc`/`free` | `_model->pools[i]`, 해제 없음 |
| `upmem.acquire` / `bind_devices` | `upmem_acquire` / `upmem_bind_devices` | `_model->set` / `_model->mapping` |
| `upmem.stage` | 대표 DPU에서 패킹 | `upmem_model_wants(_model, binding, _job)`이 참인 job만 패킹 |
| `upmem.to_mram` | prepare + push | `_model->pointers[binding][_job]` 기록 후 `upmem_model_transfer(_model, binding, 0)` |
| metadata | 연속 구간별 push | `_model->metadata`에 기록 후 `upmem_model_metadata` |
| `upmem.launch` | `dpu_launch` | `upmem_model_launch(_model)` |
| `upmem.gather` | prepare + push FROM_DPU | `upmem_model_transfer(_model, binding, 1)` |
| `foreach_group` + `unstage` | 항상 | `_model->upload_weights == 0`일 때만 |

`_job = _batch_slot × 논리 DPU 수 + _u`이고, stage·unstage의 host 원본은
`_model->arguments[_instance × argument_count + i]`다.

## 8. Device 계산과 DMA

구현: `PIMToUPMEMPass.cpp` `convertDevice`, `buildDMA`,
[UPMEMToDevice.cpp](../lib/Target/UPMEM/UPMEMToDevice.cpp).

### 8.1 tensor를 WRAM buffer로

**내용**

`pim::collectLocalTiles`로 load 결과와 Linalg outs의 buffer 소유 관계를 구하고, 소유자마다
WRAM buffer 한 벌을 전역 `__dma_aligned` 배열로 만든다. Linalg를 memref 형태로 다시 만들고
값 전달용 tensor iter_arg를 없앤다.

**근거**

- K 반복마다 새 accumulator를 할당하지 않고 같은 WRAM을 갱신한다.
- WRAM은 작아서 스택에 둘 수 없다(`UPMEMToDevice.cpp` 주석).
- Device 밖의 정수는 `materializeExternals`가 metadata 통로로 넘긴다. int/index가 아닌
  Host 값은 진단한다.

**예시**

```c
__dma_aligned int8_t  w0[8192];   // A Group tile 64x128
__dma_aligned int8_t  w1[8192];   // B Group tile 128x64
__dma_aligned int32_t w2[4096];   // C Group tile 64x64
```

### 8.2 전역 좌표를 로컬 MRAM 좌표로

**내용**

`pim.device_origin`의 전역 시작값을 0으로 바꾸면 local IV가 DPU 안의 Group 번호가 된다.
`pim.reused_input` access는 `origin - place.start`로 명시적으로 로컬화한다.

**근거**

- singleton IV가 상수로 확정되면 `pim.device_origin` 산술이 사라질 수 있다. 로컬화하지
  않으면 다음 round의 전역 시작이 작은 MRAM slot 안의 offset으로 쓰인다.
- K-dependent access가 남아 있는데 temporal K origin이 없으면 진단한다.

**예시**

```text
Common:  k = k_round_start + k_local      → 전역 K Group 좌표
UPMEM :  k = 0 + k_local                  → slot 안의 K Group 번호
```

### 8.3 tasklet 분할

**내용**

Group 하나는 모든 tasklet이 함께 계산한다. Linalg indexing map에서 출력 m(`d0`)이
leading dimension인 operand는 행을 `rows/lanes`씩 나누고, m을 쓰지 않는 operand는 WRAM
tile 전체를 공유한다.

**근거**

- `rowSliceOf`: 행 수가 `lanes`로 나누어떨어지지 않으면 진단한다.
- indexing map이 projected permutation이 아니거나 m이 leading이 아닌 위치에 있으면
  진단한다.
- generic은 identity-map, all-parallel만 받는다.

**예시**

| 계산 | 행을 나누는 operand | 공유 operand |
|---|---|---|
| matmul | A, C | B |
| matvec | A, y | x |
| identity pointwise | 모든 operand | 없음 |

```text
C tile 64행, lanes 16 → tasklet t: C[4t : 4t+4], A[4t : 4t+4], B 전체
```

### 8.4 DMA chunk와 barrier

**내용**

`buildDMA`는 Group tile 하나를 연속 MRAM 구간으로 읽고 쓴다. 2048바이트 이하이면서 행 수를
나누어떨어지게 하는 가장 큰 행 묶음을 chunk로 하고, tasklet t가 chunk `t, t+lanes, …`를
옮긴다.

**근거**

- `UPMEMDialect.cpp` `verifyDMASize`: 한 DMA는 8바이트 배수, 2048바이트 이하여야 한다.
  한 행이 이를 어기는 shape를 다른 layout으로 바꾸지 않는다.
- chunk가 하나이면 tasklet 0만 DMA를 수행한다.
- barrier 위치: 입력이 있는 계산 앞(모든 chunk 도착), K 반복 안 계산 뒤(다음 DMA가 공유
  WRAM을 덮기 전 소비 완료), 출력 DMA 앞(모든 행 계산 완료)과 뒤(다음 fill이 누산기를 덮기
  전 저장 완료).

**예시**

```text
MRAM offset = flatten(origin[d] / tile[d], pitch[d] / tile[d]) × tile_elems
            + chunk × rows_per_chunk × row_elems

for chunk = tid; chunk < nchunks; chunk += lanes:  mram_read(A + off, w0 + chunk*..., bytes)
barrier
matmul(tasklet 행 slice)
barrier                                   // K 반복 안일 때
```

### 8.5 계산 leaf의 C 코드

**내용**

`upmem-to-device`는 fill, matmul, matvec, reduce, identity/all-parallel generic을 C loop로
낸다. 반복 범위는 WRAM buffer의 정적 shape다.

**근거**

- matmul/matvec의 곱셈은 C `*`를 그대로 쓴다. `__builtin_mul_sl_sl_rrr`로 바꾸면 inline
  assembly 때문에 loop 펼치기와 register 재사용이 막혀 느려진다(`emitMatmul` 주석).
- K가 64 이하이면 `#pragma unroll`을 붙인다. clang은 16을 넘는 K loop를 스스로 펼치지
  않는다.
- 피연산자는 누산기 타입으로 cast해 곱하므로 i8 입력·i32 누산이 그대로 표현된다.
- generic은 [CEmitter.h](../lib/Target/UPMEM/CEmitter.h)가 지원하는 scalar arith만 받는다.
- `main()`은 Host 변수 `_pim_model_active`가 0이면 바로 끝난다. 모델 세션이 일부 DPU를
  비활성화할 때 쓴다.

**예시**

```c
// matmul: tasklet 행 slice 위에서
for (int32_t _m = 0; _m < M; ++_m)
  for (int32_t _n = 0; _n < N; ++_n) {
    int32_t _s = C[_m * N + _n];
#pragma unroll
    for (int32_t _k = 0; _k < K; ++_k)
      _s += (int32_t)A[_m * K + _k] * (int32_t)B[_k * N + _n];
    C[_m * N + _n] = _s;
  }

// matvec
for (int32_t _m = 0; _m < M; ++_m) {
  int32_t _s = y[_m];
  for (int32_t _k = 0; _k < K; ++_k) _s += (int32_t)A[_m * K + _k] * (int32_t)x[_k];
  y[_m] = _s;
}
```

## 9. Gather와 partial 결합

**내용**

출력 `allocation_shape`만큼 DPU마다 회수하되 assignment로 pool 위치를 정한다.
`foreach_group`은 pool을 canonical 논리 순서로 돌고 `unstage`가 원본 strided 영역에
복원한다. `pim.combine` 출력은 첫 contribution만 대입하고 나머지는 더한다.

**근거**

- `buildOutput`: 판정값 = `reduction_k round IV + pk`. 0일 때만 대입이다.
- 출력 capacity가 DPU 좌표에 의존하면 진단한다. 모든 DPU가 같은 크기를 돌려줘야 한다.
- 물리 배정이 바뀌어도 논리 결합 순서는 유지된다.

**예시**

```text
pim.store 출력:        C[region(pm,pn)]  = slot(pm,pn,0)
pim.combine(add) 출력: if (k_round + pk == 0) C[region] = slot
                       else                   C[region] += slot
```

## 10. 전체 예시

### 10.1 GEMM

**내용**

INT8 `A[1024,28672] × B[28672,8192] → C` INT32, Block=[64,64,128], Group=[1,1,1],
Tile=[4,4,14], P=[4,32,16](DPU 2048개)을 가정한다. 축별 Execution Tile 수가 P와 같아 round는
1회이고 DPU 하나가 Tile 하나를 맡는다.

**근거**

- 7.3절(Common) round 식, 5절 MRAM 식, 6절 배정 식, 8절 DMA 규칙에 대입한 값이다.

**예시**

```text
              m              n              k
Group 수      1024/64 = 16   8192/64 = 128  28672/128 = 224
Exec Tile 수  16/4 = 4       128/4 = 32     224/14 = 16
round 수      1              1              1
Tile 원소     256            256            1792

DPU(pm,pn,pk): A[256pm:+256, 1792pk:+1792]   448 KiB  shared_over=[n]
               B[1792pk:+1792, 256pn:+256]   448 KiB  shared_over=[m]
               C_partial[256,256] i32        256 KiB  (MRAM 합 1152 KiB)
Host pool: A 4×16=64 slot, B 32×16=512 slot, C 2048 slot
G = [2,4,1] (6.1절 표)
DPU local loop: m 4 × n 4 × k 14 Group
WRAM: A 8 KiB + B 8 KiB + C 16 KiB = 32 KiB  (≤ local_capacity_limit_bytes 기본 46680, pim-verify 검사)
tasklet 16: C·A 4행씩, B 공유
DMA chunk: A 16행×4, B 32행×4, C 8행×8 (각 2048 B)
Host: 같은 (pm,pn)의 pk=0..15 partial 합산
```

### 10.2 GEMV

**내용**

INT8 `A[4096,4096] · x[4096] → y` INT32, Block=[32,1,64], Group=[1,1,1], Tile=[8,1,8],
P=[16,1,8](DPU 128개)을 가정한다.

**근거**

- 커널이 A와 x를 INT32로 넓힌 뒤 곱해도 `PIMFormComputePass.cpp` `formMatvec`가 sign
  extension 앞의 INT8 값으로 `linalg.matvec`를 만든다. 그래서 WRAM의 A·x는 i8이고 누산만 i32다.
- x만 m을 공유하므로 `gm`이 클수록 비용이 줄고, `gm·gn·gk ≤ 8`에서 `G=[8,1,1]`이다.
- 6.2절 식에 `G=[8,1,1]`, `P=[16,1,8]`을 대입하면 묶음 `j`는 `k = j % 8`,
  `m = (j / 8) × 8 + CI`다.

**예시**

```text
              m              k
Group 수      4096/32 = 128  4096/64 = 64
Exec Tile 수  128/8 = 16     64/8 = 8
round 수      1              1
Tile 원소     256            512

DPU(pm,pk): A[256pm:+256, 512pk:+512]  128 KiB
            x[512pk:+512]              512 B   shared_over=[m]
            y_partial[256] i32         1 KiB
Host pool: A 128 slot, x 8 slot, y 128 slot

member j (rank 0): (m0..7,  k j)   → 8 DPU가 x_pool[j] 공유
member j (rank 1): (m8..15, k j)

DPU local loop: m 8 × k 8 Group
WRAM: A 32×64 = 2 KiB, x 64 B, y 128 B
tasklet 16: A·y 2행씩, x 공유
DMA: A 2048 B chunk 1개(tasklet 0), x 64 B, y 128 B
Host: 같은 pm의 pk=0..7 partial 합산
```

A payload가 x보다 훨씬 크므로 x 공유는 H2D 시간에 미치는 영향이 GEMM의 A/B 공유보다 작다.

## 11. 지원 범위와 제약

**내용**

아래 조건을 만족하지 않으면 lowering 또는 번역 단계에서 진단한다.

**근거**

- `PIMToUPMEMPass.cpp`, `UPMEMDialect.cpp`, `UPMEMToHost.cpp`, `UPMEMToDevice.cpp`의
  진단 문구.

**예시**

| 조건 | 위반 시 |
|---|---|
| `banks_per_unit = cores_per_bank = 1` | 진입 진단 |
| `1 ≤ nr-tasklets ≤ lanes_per_core` | 진입 진단 |
| `allocation_shape` 정적, MRAM 합 ≤ bank 예산 | lowering 진단 |
| Group tile 행 수 % lanes = 0 | lowering 진단 |
| DMA 한 번: 8 B 배수, ≤ 2048 B | UPMEM verifier 진단 |
| 계산: fill/matmul/matvec/reduce/identity generic | lowering·번역 진단 |
| K-only reduce는 lanes = 1 | lowering 진단 |
| combiner는 `add` | lowering 진단 |
| WRAM buffer는 정적 shape, scalar 원소 | 번역 진단 |
| Host round loop는 값을 나르지 않음 | 번역 진단 |
