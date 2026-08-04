# PIM IR 재설계 — target-parametric 설계

경로 기준: 상대 경로는 `/home/dlrkdals/TritonPIM/triton_shared/`, 그 밖은 절대 경로.

**§13이 구현에서 확정된 계약이고, 본문과 어긋나는 자리에서는 §13이 기준이다.**
이 문서는 `MemoryLowering_Design.md`의 target-independent PIM Common 범위를 대체한다 —
`MemoryLowering_Design.md`는 이제 UPMEM·AiMX의 physical address와 descriptor 계층만 기술한다.

---

# 0. 범위와 계층 선택의 근거

## 0.1 공통 execution hierarchy와 target-specific memory topology를 분리한다

> This PIM IR defines a common compiler abstraction for **digitally programmable PIM targets**.
> The common hierarchy describes **execution and placement roles**, not physical memory
> organization.

```
공통 execution hierarchy          PIM Device → PIM Group → PIM Core
target-specific memory topology   Channel → Pseudo-channel → DRAM Bank Group → Bank
```

## 0.2 왜 Bank Group을 공통 최하위 계층으로 쓰면 안 되는가

`Bank Group`은 두 의미가 충돌한다. 하나는 추상 배치 단위이고, 다른 하나는 High Bandwidth
Memory의 실제 physical hierarchy다. AttAcc가 직접적인 반례다 — 공식 simulator가 세 variant를
제공한다.

| AttAcc variant | processing unit 위치 | 추상 "Bank Group"이 가리키게 되는 것 |
|---|---|---|
| `AttAcc_bank` | bank마다 | 하나의 physical bank |
| `AttAcc_BG` | physical DRAM Bank Group마다 | **하나의 physical DRAM Bank Group** |
| `AttAcc_buffer` | buffer die의 pseudo-channel마다 | 하나의 pseudo-channel |

이름 하나가 서로 다른 세 물리 구조를 뜻하게 되고, `AttAcc_BG`에서는 둘을 **구별할 수 없다.**
Samsung High Bandwidth Memory Processing-In-Memory도 processing unit이 bank I/O boundary에
있고 여러 bank의 processing unit이 같은 command를 lockstep으로 수행하므로 같은 혼동이 생긴다.

**따라서 최하위 공통 계층은 `PIM Core`다.** 물리적으로 그것이 bank인지, DRAM Bank Group인지,
pseudo-channel의 buffer die인지는 **target 확장**이 말한다(§2.4).

## 0.3 지원 범위 — topology 표현과 실행 가능한 lowering을 구분한다

| target | 3계층 topology 표현 | 지금의 GEMM subset lowering |
|---|---|---|
| UPMEM | 가능 | **가능** — 이 문서의 구현 대상 |
| AiMX (Silent-PIM 계열) | 가능 | 설계 완료, 코드 0줄 |
| Samsung High Bandwidth Memory Processing-In-Memory | 가능 | 부분적으로 가능 (GEMV·vector arithmetic) |
| AttAcc의 GEMV 부분 | 가능 | operation semantics 추가 필요 |
| AttAcc full attention | 가능 | **현재 op set으로 불가능** |

```
Phase 1   UPMEM GEMM (구현 대상), AiMX GEMM
          Samsung: GEMV, vector arithmetic, 제한된 batched GEMV
          AttAcc: GEMV 부분만
          masked load 는 pad = 0 만 지원

Phase 2   reduction, group accumulator, softmax, barrier, full attention dataflow
          nonzero pad, 여러 PIM Group, 일반 cyclic placement
```

구현 대상 커널은 `gemm.py`, `split_k.py`, `grouped_gemm.py` 셋뿐이다. 그 밖의 TTShared 패턴은
지원하지 않고, **지원하지 않는다는 사실을 진단으로 알린다.**

---

# 1. 전체 그림 — Triton Python에서 하드웨어까지

## 1.1 Triton BLOCK이 tile을 정의하고, Grid는 그것을 열거한다

**Grid가 tile을 새로 자르는 것이 아니다.** tile의 모양은 이미 `BLOCK_M`/`BLOCK_N`/`BLOCK_K`가
정했고, Grid는 그렇게 만들어진 tile들을 열거할 뿐이다.

```
Triton Python
  BLOCK_M / BLOCK_N / BLOCK_K
        │
        ▼
Logical tile creation           output-row tile × output-column tile
        │
        ▼
Triton Grid                     모든 logical tile을 열거
        │
        ▼
Execution wave 구성              한 PIM Group이 lockstep으로 함께 도는 tile 묶음     §4.2
        │
        ▼
PIM Device mapping              wave를 PIM Device에 균형 분배                     §4.3
        │
        ▼
PIM Core mapping                logical tile 또는 그 fragment를 PIM Core에         §5.2
        │
        ▼
Physical memory placement       PIM Core 안의 buffer offset과 resident region      §6
        │
        ▼
Target lowering                 실제 address, descriptor, memory request
```

용어를 처음 등장할 때 정의한다.

- `M` — number of output rows
- `N` — number of output columns
- `K` — reduction dimension

| 커널 안의 요소 | 확정하는 것 | 컴파일 시점에 상수인가 |
|---|---|---|
| `BLOCK_M` | 한 logical tile의 output row 수 | **예** — `pim.access.sizes`에 정적 값으로 남는다 |
| `BLOCK_N` | 한 logical tile의 output column 수 | **예** |
| `BLOCK_K` | 한 번의 contraction이 처리하는 reduction element 수 | **예** |
| `grid = (...)` | 병렬 인스턴스의 축 **개수** | 축 개수는 예, **extent는 아니오** |
| K loop | PIM Core 안에서 순차 실행되는 reduction | 반복 횟수는 아니오 (`K`가 런타임) |
| `acc += tl.dot(a,b)` | 누산기의 타입과 위치 | 예 |
| `SPLIT_K` + grid 축 하나 | K를 몇 조각으로 나눠 별도 출력면에 쓸지 | **예** |
| `GROUP_M` | tile 방문 순서(스케줄) | 예 |
| `M`, `N`, `K`, stride 인자 | 문제 크기 | **아니오** — launch 인자 |

## 1.2 하강 경로

```
gemm.py / split_k.py / grouped_gemm.py
   │  @triton.jit,  grid=(...)     ← PIMOptions가 확정되고 컴파일 캐시 키에 들어간다
   ▼                                  grid extent는 캐시 키에 없다 (driver.py:164-166)
TTIR       →  --triton-to-linalg-experimental  →  TTShared IR
   │
   │ ①  --pim-normalize-address      주소 재귀를 닫힌 형태로   ← raise보다 반드시 앞
   │ ②  --convert-ttshared-to-pim    pim.access / pim.load / pim.store
   │ ③  --pim-build-logical-grid     source grid → logical tile loop nest
   │ ④  --pim-recover-access-origin  access의 축별 origin을 logical 좌표로 채운다
   │ ⑤  --pim-form-compute           누산기 정규화, 계산 형태 확정
   ▼
Logical PIM IR
   │ ⑥  --pim-form-execution-waves   core footprint로 wave 구성 + padding
   │ ⑦  --pim-map-devices            wave → PIM Device 균형 분배
   │ ⑧  --pim-map-cores              logical tile fragment → PIM Core 좌표, lane 수 선택
   │ ⑨  --pim-place-data             최종 region·replication·capacity materialize
   ▼
Placed PIM IR      ← target-parametric 형태. 이 문서의 목표 산출물
   │ ⑩  --pim-verify                 공통 fail-fast 검증
   │ ⑪  --pim-legacy-unit-form       전이용 호환 하강 [한시적]
   ▼
pim.units / pim.unit_data
   │ ⑫  --convert-pim-to-upmem       진입 즉시 UPMEM legality gate
   ▼
UPMEM IR → host.cpp / device.c → .pimobj
```

profile을 처음 읽는 pass는 ⑥이고, 그 앞의 어떤 pass도 읽지 않는다.

**`--pim-select-group-mapping`을 두지 않는다.** Phase 1에서 두 target 모두 device당 PIM Group이
하나라 `pim_group_index = 0`이 profile에서 바로 유도된다. lockstep legality는 ⑩에서 검사한다.
PIM Group이 여럿인 target을 실제로 붙일 때 pass를 추가하는 편이 단순하다.

## 1.3 source grid와 logical grid를 나눠야 하는 이유

| 커널 | source grid | logical tile 좌표 |
|---|---|---|
| `gemm.py` | `(x, y)` | `(output_row_tile, output_column_tile)` |
| `split_k.py` | `(x, y)` | `(output_row_tile, output_column_tile, explicit_k_partition)` |
| `grouped_gemm.py` | `(x)` | `(output_row_tile, output_column_tile)` |

source grid 축은 사용자가 launch 편의로 고른 것이고, **데이터가 어떻게 쪼개지는지를 말하지
않는다.** `grouped_gemm.py`는 축이 하나뿐이지만 실제로는 2차원 tile 공간을 순회한다.

**이 구분을 하지 않으면 지금 조용히 틀린다.** `grouped_gemm`처럼 축이 하나면 C origin의 축 0과
축 1이 **같은 loop**를 가리키는데, `driverLoop`(`PIMPartitionPass.cpp:50-74`)는 한 값이 **두**
loop에 의존할 때만 `ambiguous`를 세우므로 진단이 나가지 않는다. 그 결과
`buildUnitData`(`:306-309`)가 두 축 모두에 `sizes[d] = trips × tile`을 적어, device가 흩어진
tile `trips`개를 갖는데 `trips·32 × trips·32` 정사각형을 소유한다고 기록한다. device 2048개에
tile 16384개면 `trips=8` → 실제 32 KiB 대신 256 KiB를 주장하고, `upmem.unstage`가 그 정사각형
64 tile을 전부 되돌려 쓴다(`UPMEMToHost.cpp:481-494`). 56개가 쓰레기 값이다.

## 1.4 컴파일 시점과 실행 시점의 경계

| 항목 | 컴파일 시점 | 근거 |
|---|---|---|
| logical 축의 개수와 순서 | **예** | `pim.units.counts`가 `DenseI64ArrayAttr` (`PIMDialect.td:144`) |
| device partition 인자 | **예** | 곱이 `upmem.acquire`의 상수 유닛 수, `malloc` 개수, C loop 상한 (`UPMEMToHost.cpp:174-176, 209-212`) |
| PIM Core shape, tile footprint | **예** | profile 값 |
| lane 수 | **예** | device 빌드 플래그 `-DNR_TASKLETS` (`driver.py:95`)이자 `.so` 캐시 키 (`driver.py:80`) |
| MRAM 버퍼 바이트 수 | **예** | `__mram_noinit T name[N]`이 상수를 요구 (`PIMToUPMEMPass.cpp:164-167`) |
| tile 개수 | **아니오** | grid loop 상한이 `num_programs` 인자 (`PIMBuildGridPass.cpp:84-86`) |
| device당 tile 수, start/size | **아니오** | `narrowLoop`가 이미 런타임 산술로 낸다 (`PIMPartitionPass.cpp:96-139`) |

따라서 **"tile을 device 수로 나눈다"는 계산은 컴파일 시점에 수로 확정되지 않는다.** partition
인자만 상수이고, 나눗셈은 생성된 host 코드가 launch 시점에 수행한다.

---

# 2. 기계 모델

## 2.1 3계층 정의

```
System
├── Host
└── PIM Devices
     └── PIM Group          command / synchronization / shared functional unit의 범위
          └── PIM Core      하나 또는 소수의 bank에 연결된 processing unit
               ├── attached bank memory
               ├── local storage      (scratchpad 또는 register file)
               └── parallel execution lanes
```

**PIM Device** — 독립적인 PIM kernel 또는 descriptor sequence를 실행할 수 있는 hardware
device. UPMEM은 DPU 하나, AiMX는 FPGA PIM Device 하나, Samsung은 High Bandwidth Memory stack
하나다. DIMM·board·card는 **공통 계층이 아니다** — runtime resource organization이다.

**PIM Group** — command scope다. 단순한 물리 container가 아니다. Samsung의 Single-Bank mode와
All-Bank Processing-In-Memory mode가 서로 다른 command scope를 갖는 것이 이 계층의 근거다.
AttAcc에서는 group이 shared accumulator unit과 softmax unit도 소유한다. **Phase 1에서는 두
target 모두 device당 group이 하나라 tile partition 축으로 나타나지 않는다.**

**PIM Core** — 하나 또는 소수의 local memory bank에 연결된 processing unit. data placement와
bank-local computation의 단위다.

**parallel execution lanes** — 계층이 아니다. PIM Core 하나 안의 병렬 실행 자원이고, 성격이
target마다 다르다.

```
parallel_execution_kind = "multithread"   UPMEM tasklet — 독립 스레드, 서로 다른 tile
parallel_execution_kind = "simd"          AiMX arithmetic lane — 한 계산을 함께 수행
```

**두 성격을 `ways` 하나로 묶으면 안 된다.** AiMX의 16 lane은 logical tile을 16개의 독립
work item으로 나누는 것이 아니라 **하나의 bank-local computation을 함께 수행하는 SIMD
datapath**다. 예산식이 여기서 갈린다(§5.3).

`unit`이라는 단어는 새 표면에서 사라진다. `pim.units` / `pim.unit_data` / `pim.unit_origin`은
**전이 표현으로만** 남는다.

## 2.2 target 대응

| | UPMEM | AiMX (Silent-PIM 계열) |
|---|---|---|
| runtime resource container | DIMM / rank | AiMX Card |
| **PIM Device** | **DPU** — **2048개** (`compiler.py:228`) | **FPGA PIM Device** — **8개** (`MemoryLowering_Design.md:290`) |
| **PIM Group** / device | 1 | 1 — FPGA 전체의 lockstep scope |
| **PIM Core shape** | `[1]` | `[8, 16]` = channel × bank (`:245`) |
| attached bank / core | MRAM. 물리 64 MB, usable 32 MiB | bank 1개. 용량 미측정 |
| local storage | scratchpad — WRAM 물리 64 KB, usable 48 KiB | register file — vecA 128 B, vecB 32 B, vACC 미측정 |
| parallel lanes / core | 16 tasklet | 16 arithmetic lane |
| `parallel_execution_kind` | `multithread` | `simd` |
| `requires_uniform_instruction_sequence` | `false` | **`true`** |
| 데이터 타입 | 범용 (커널이 정한다) | bf16 입력 / fp32 누산 |
| 권장 BLOCK (M, N, K) | (32, 32, 16) | (16, 16, 16) — §2.5 |
| **logical tile의 core footprint** | **`[1]`** | **`[1, 16]`** — 한 channel의 bank 전부 |

마지막 행이 이 판의 핵심이다.

> **UPMEM에서는 logical tile 하나를 PIM Core 하나가 소유한다. AiMX에서는 logical tile 하나가
> 한 channel의 16개 bank-local PIM Core에 걸쳐 협력 실행된다.**

따라서 PIM Core mapping의 정의는 이렇게 된다.

```
logical tile 또는 logical tile fragment를 하나 이상의 PIM Core에 배치하고,
각 logical dimension과 PIM Core topology의 대응을 결정한다.
```

**두 target의 병렬성이 서로 다른 계층에 있다.** UPMEM은 device 2048개에 device당 core 1개라
병렬성이 전부 inter-device다. AiMX는 device 8개에 device당 core 128개라 대부분 intra-device다.

**AiMX Card와 UPMEM rank는 공통 계층에 넣지 않는다.** Card는 runtime resource container이고,
공통 profile에는 선택된 FPGA PIM Device 수만 전달한다. rank는 UPMEM 확장의 transfer
파라미터다(§2.4).

### 참고 — 같은 계층에 다른 target을 올리면

```
Samsung High Bandwidth Memory Processing-In-Memory
  PIM Device = one High Bandwidth Memory stack
  PIM Group  = one controller-visible command broadcast scope
               (Single-Bank / All-Bank / All-Bank PIM mode가 서로 다른 scope)
  PIM Core   = one Programmable Computing Unit + attached banks + 16-lane SIMD

AttAcc
  PIM Device = one High Bandwidth Memory stack
  PIM Group  = one or multiple pseudo-channels + shared accumulator + shared softmax
  PIM Core   = one GEMV unit.  attached memory는 variant가 정한다
```

## 2.3 `#pim.target_profile` — 공통 8필드

**공통 profile에는 공통 pass가 실제로 읽는 것만 둔다.** 나머지는 전부 target 확장으로 뺀다.

```mlir
// UPMEM
#pim.target_profile<
  target                                = "upmem",
  number_of_pim_devices                 = 2048,      // compiler.py:228
  pim_core_shape                        = [1],
  number_of_parallel_lanes_per_pim_core = 16,
  parallel_execution_kind               = "multithread",
  requires_uniform_instruction_sequence = false,
  usable_bank_capacity_bytes            = 33554432,  // core당. 물리 MRAM 64 MB 중 커널 몫
  usable_local_capacity_bytes           = 49152      // core당. 물리 WRAM 64 KB 중 커널 몫
>

// AiMX
#pim.target_profile<
  target                                = "aimx",
  number_of_pim_devices                 = 8,         // MemoryLowering_Design.md:290
  pim_core_shape                        = [8, 16],   // :245
  number_of_parallel_lanes_per_pim_core = 16,        // :245
  parallel_execution_kind               = "simd",
  requires_uniform_instruction_sequence = true,
  usable_bank_capacity_bytes            = <미기록>,   // 필드 자체가 없다 — §2.6
  usable_local_capacity_bytes           = 0          // scratchpad 없음
>
```

각 필드의 소비자가 정확히 하나씩 있다.

| 필드 | 읽는 pass |
|---|---|
| `number_of_pim_devices` | `--pim-map-devices` |
| `pim_core_shape` | `--pim-form-execution-waves`, `--pim-map-cores` |
| `number_of_parallel_lanes_per_pim_core` | `--pim-map-cores` (상한) |
| `parallel_execution_kind` | `--pim-map-cores` (예산식), `--pim-verify` |
| `requires_uniform_instruction_sequence` | `--pim-form-execution-waves` (padding), `--pim-verify` |
| `usable_bank_capacity_bytes` | `--pim-verify` |
| `usable_local_capacity_bytes` | `--pim-map-cores` (lane 상한), `--pim-verify` |
| `target` | `--pim-legacy-unit-form` 이후 분기 |

`number_of_pim_cores_per_group`은 `pim_core_shape`의 곱이라 필드가 아니다.
`number_of_pim_groups_per_device`도 Phase 1에서 항상 1이라 두지 않는다 — 값이 하나뿐인 필드는
정보를 나르지 않는다.

**`requires_uniform_memory_address`는 공통 profile에서 뺀다.** 이유는 §2.7이다.

## 2.4 target 확장

공통 pass가 읽지 않는 것은 전부 여기로 간다. target lowering과 target legality gate만 읽는다.

```mlir
#pim.upmem_ext<
  number_of_devices_per_transfer_group = 64      // rank. push_xfer 묶음 최적화용
>

#pim.aimx_ext<
  pim_core_axis_names                 = ["channel", "bank"],
  pim_core_placement                  = "bank",
  vector_register_a_capacity_bytes    = 128,
  vector_register_b_capacity_bytes    = 32,
  accumulator_register_capacity_bytes = <미기록>,   // §6.6의 legality가 여기 걸린다
  input_element_type                  = "bfloat16",
  accumulator_element_type            = "float32"
>
```

`pim_core_placement`는 `bank | dram_bank_group | pseudo_channel_buffer | device_private_dram`
이다. Samsung·AttAcc를 붙일 때는 physical memory topology도 같은 확장에 들어간다 —
`number_of_channels`, `number_of_pseudo_channels_per_channel`,
`number_of_dram_bank_groups_per_pseudo_channel`, `number_of_banks_per_dram_bank_group`,
`supported_execution_modes`, `command_scope_for_each_execution_mode`.

**공통 profile은 8필드로 고정하고, target이 늘어도 늘지 않는다.** 새 target이 필요로 하는
것은 자기 확장에 넣는다. 공통 pass가 그것을 읽어야 하는 상황이 오면, 그때가 공통 필드를
하나 늘려야 한다는 신호다.

## 2.5 target별 BLOCK

BLOCK은 컴파일러가 정하지 않는다. Python 커널의 `tl.constexpr`이고 ttshared 단계에서 이미
`pim.access.sizes`의 정적 값으로 도착한다. profile이 하는 일은 (i) BLOCK이 capacity와 core
topology에 맞는지 검사하고, (ii) `min_dot_size` hook(`compiler.py:270`, 지금 `(1,1,1)`)에 최소
granularity를 알리는 것이다.

| target | 권장 BLOCK (M, N, K) |
|---|---|
| **UPMEM** | **(32, 32, 16)** — WRAM 예산. lane당 `32·16 + 16·32 + 32·32·4 = 5120 byte`. 실측 기준 구성이기도 하다(`UPMEMTarget_Implementation.md:411-412`) |
| **AiMX** | **(16, 16, 16)** — 아래 유도 |

### AiMX BLOCK의 유도 — 각 축의 근거가 서로 다르다

`vector_register_a_capacity_bytes = 128` (= 4 × 256-bit word),
`vector_register_b_capacity_bytes = 32` (= 1 × 256-bit word). bf16이 16 bit이므로 256-bit
word 하나에 **bf16 16개**가 들어간다.

| 축 | 값 | 근거 |
|---|---|---|
| `BLOCK_M` | **16** | **16 arithmetic lane**이 output row 원소를 하나씩 맡는다. 그리고 vACC가 `BLOCK_M × fp32`를 담아야 한다 — §6.6의 미확정 legality |
| `BLOCK_N` | **16** | **channel당 bank 수.** bank(= PIM Core) 하나가 output column 하나를 맡는다 |
| `BLOCK_K` | **16** | **vecB 32 byte = bf16 16개.** bank-private B vector 하나가 vecB에 정확히 들어간다. K reduction 자체는 순차다 |

**lane 수가 `BLOCK_K`를 정하는 것이 아니다.** lane은 output row에 대응하므로 `BLOCK_M`을
설명한다. `BLOCK_K`의 직접적인 근거는 vecB 용량이다. 앞 판이 이 둘을 잘못 이었다.

tile 바이트: A `16 × 16 × bf16 = 512`, B `16 × 16 × bf16 = 512`, 누산기
`16 × 16 × fp32 = 1024`. 합 **2048 byte**.

`min_dot_size`에는 `BLOCK_K = 16`만 싣는다 — `(1, 1, 16)`. `BLOCK_M`은 vACC에, `BLOCK_N`은
core topology에 걸려 있어 dot-shape 제약이 아니다.

**세 참조 커널은 지금 형태로는 AiMX에 매핑되지 않는다.** AiMX는 bf16 입력 / fp32 누산인데 세
커널 전부 int8 × int8이다(`gemm.py:96-97`, `split_k.py:76-77`, `grouped_gemm.py:123-124`).

## 2.6 capacity 값의 세 가지 상태

**`0`을 "검사 생략, 항상 안전"으로 읽으면 안 된다.** 용량을 모르는 채 검사를 건너뛰는 것은
conservative가 아니라 **permissive**다 — oversized tile을 통과시킨다. 세 상태를 구분한다.

| 상태 | 뜻 | 동작 |
|---|---|---|
| 값이 있음 | 알려진 용량 | 정적 legality 검사 |
| `0` | **그 저장소가 존재하지 않는다** | 그 검사가 성립하지 않으므로 생략. AiMX의 `usable_local_capacity_bytes = 0`이 이 경우 — scratchpad가 없다 |
| **필드 없음** | 미측정 | **그 값을 필요로 하는 검사에 도달하는 순간 진단** |

AiMX의 `usable_bank_capacity_bytes`와 `accumulator_register_capacity_bytes`는 세 번째다.
`pim_target="aimx"`로 실제 배치를 시도하면 `--pim-verify`가 그 자리에서 실패한다. **앞 판이
§2.3과 §2.4에서 서로 다르게 적었던 것을 이 규칙 하나로 통일한다.**

AiMX의 register 용량 검사는 `usable_local_capacity_bytes`가 아니라 `#pim.aimx_ext`의
`vector_register_*` / `accumulator_register_*` 필드로 수행한다 — AiMX legality gate 소관이다.

## 2.7 instruction lockstep과 memory address equality를 분리한다

**앞 판의 AiMX profile은 자기 operand mapping과 모순됐다.** `requires_uniform_memory_address
= true`로 적었는데, §6.6의 mapping은 bank마다 다른 B column과 다른 output column을 쓴다.
같은 global address일 수가 없다. 그 verifier 규칙대로면 **문서가 정의한 AiMX mapping 자체가
거부된다.**

둘은 다른 조건이다.

| 조건 | 뜻 | AiMX |
|---|---|---|
| `requires_uniform_instruction_sequence` | 같은 command sequence, 같은 반복 횟수, 같은 reduction 길이 | **`true`** |
| uniform memory address | 여러 processing unit이 **같은 global address**를 접근 | **`false`** |

주소 동일성은 Samsung의 All-Bank Processing-In-Memory mode처럼 command semantics가 그것을
요구할 때만 성립한다. 그것은 **mode-specific legality**이고 target lowering이 검사한다.
공통 profile에서 뺀다.

AiMX에서 bank-local address pattern이 서로 같아야 하는 제약이 있다면, 그것도 주소 동일성이
아니라 **pattern 동일성**이고 target이 본다.

## 2.8 lockstep의 범위는 PIM Group이다 — device 사이가 아니다

**앞 판은 lockstep이면 모든 device의 tile 수가 같아야 한다고 적었는데, 틀렸다.** lockstep
scope는 PIM Group이고 AiMX에서는 FPGA PIM Device 하나가 독립된 PIM Group이다. Card 수준의
global lockstep은 어디에도 기록돼 있지 않다.

| 범위 | 조건 |
|---|---|
| 같은 PIM Group 안의 PIM Core | **같은 command sequence, 같은 반복 횟수, 같은 reduction 길이.** 마지막 wave의 빈 자리는 group 안에서 padding |
| 서로 다른 PIM Device | **wave 수가 하나 정도 달라도 된다.** balanced block이 이미 그렇게 낸다 |

device 사이까지 같게 하려면 별도의 Card-level execution constraint가 있어야 하고, 지금은
없다. **device 간 균형 분배와 group 내부 lockstep을 분리한다.**

`tile_capacity_per_partition`이 device 불변이어야 한다는 조건은 살아 있지만 **이유가 다르다** —
`dpu_push_xfer`가 모든 DPU에 같은 바이트 수를 요구하기 때문이고(`UPMEMToHost.cpp:213,
263-264`), lockstep과 무관한 UPMEM transfer 제약이다.

## 2.9 `combine`은 profile 필드가 아니다

combine은 hardware capacity가 아니라 program semantics이자 lowering policy다. **커널 수준
IR에는 어떤 경우에도 combine을 만들지 않는다.** `split_k.py`는 `torch.zeros((SPLIT_K, M, N))`을
할당하고(`:57`) `c.sum(dim=0)`을 Python에서 수행한다(`:69`). `explicit_k_partition`은 **실제
버퍼 차원**이므로 그냥 배치되는 축이다. AttAcc의 group-level accumulator처럼 하드웨어가 결합
자원을 제공하는 target은 Phase 2다.

## 2.10 lane 수는 능력값이고, 실제 선택값은 예산이 정한다 (UPMEM)

**확인해야 할 모순이 하나 있다.** 저장소는 `BLOCK 32/32/16`, 256 유닛, `local_lanes=16`이
8보다 빠르다고 적는다 — 9.09 대 11.26 ms (`UPMEMTarget_Implementation.md:415-417`). 그런데
예산 검사식으로 계산하면 통과하지 못한다.

```
lane당 = A 32×16 i8 (512) + B 16×32 i8 (512) + 누산기 32×32 i32 (4096) = 5120
5120 × 16 = 81 920  >  65 536 (WRAM 물리 전체)     (PIMToUPMEMPass.cpp:433-437)
```

덤프도 같은 값을 적는다(`gemm_kernel.pim_common:3`). 게다가 `allocFor`는 버퍼가 없는 linalg
init까지 잡아 실제로는 **11 264**다(`PIMToUPMEMPass.cpp:367-393`).

| 기준 | lane당 바이트 | usable 48 KiB에서 최대 | 물리 64 KB에서 최대 |
|---|---:|---:|---:|
| `pim.co_resident` | 5 120 | **9** | **12** |
| `allocFor` 실제 | 11 264 | **4** | **5** |

```bash
$OPT --convert-pim-to-upmem="unit-memory-bytes=33554432 local-lanes=16 local-memory-bytes=49152" \
     $DUMP/TNQUJN5*/gemm_kernel.pim_common 2>&1 | grep -i "fast memory\|budget"
```

**어느 쪽이든 조용히 틀리지 않도록 표현을 나눈다.** profile의
`number_of_parallel_lanes_per_pim_core = 16`은 **능력값(상한)** 이고, 실제 사용할 lane 수는
`--pim-map-cores`가 예산 안에서 고른다(§5.3). **AiMX에는 이 문제가 없다** —
`parallel_execution_kind = "simd"`라 버퍼가 lane마다 복제되지 않는다.

---

# 3. 논리 좌표 복원

## 3.1 pointer recurrence normalization — 모든 것의 전제

세 커널 전부 `a_ptrs += BLOCK_K * stride_ak` 형태로 주소를 loop iter_arg에 나른다
(`gemm.py:60-61`, `split_k.py:45-46`, `grouped_gemm.py:56-57`).

**이것을 풀지 않으면 입력 배치가 통째로 사라지고, 그 손실이 조용하다.** 실측이 있다.
`workspace/Attention/results-gemm/ir/*/*.pim_common`에서 `pim.unit_data` 개수를 세면 11개 변종
중 10개가 3~4개인데 `G03_pointer_advance`만 **1개**다(C만 배치된다).
`dump/…/gemm_kernel.pim_common`도 똑같이 1개다(`:58`) — A는 `origin[%c0, %arg18]`(`:91`,
`%arg18`은 loop의 index block 인자), B는 **`origin` 절이 아예 없다**(`:93`).

기구: `buildUnitData`가 origin 성분마다 `rebuild`를 부르는데, `rebuild`는 nest 안에서 정의된
block 인자에 대해 null을 돌려주고(`PIMPartitionPass.cpp:29-31`) 호출자가 아무것도 내지 않고
반환한다(`:290-291`). 이어서 `buildDMA`가 `byAccess.at(id)`를 null 검사 없이 호출해 그대로
죽는다 — 실제 로그가 있다(`results-gemm/logs/G03_g03_pointer_advance.stderr.log`, 스택 프레임
#10 = `PIMToUPMEMPass.cpp:614:37`).

### 정합 조건과 일반형

`offset = initial + iteration × advance`는 항상 성립하지 않는다. 일반형은

```
iteration = (induction_variable - lower_bound) / loop_step
offset    = initial + iteration × advance
```

**실측: 세 커널의 모든 `scf.for`가 `lower_bound == 0`, `loop_step == 1`이다.** 커널당
`scf.for`는 정확히 하나이고 `scf.while`·`affine.for`는 없다(`gemm .ttsharedir:52`,
`split_k:49`, `grouped:51`). 그럼에도 pass는 일반형을 내고 조건을 **명시적으로 검사한다** —
입력이 임의의 TTShared IR이고, Triton `range(start, end, step)`이 `lower_bound ≠ 0`,
`loop_step ≠ 1`을 만들 수 있기 때문이다. `lower_bound=0, loop_step=1`이면 잉여 산술이
`-canonicalize`에서 상수로 확정되어 사라지므로 세 커널에서는 op이 늘지 않는다.

재작성 조건 (전부 성립할 때만):

1. iter_arg가 `index` 타입이거나 유도변수와 같은 정수 타입이다.
2. yield 피연산자의 정의 op이 `arith.addi`다.
3. 그 `addi`의 피연산자 하나가 **정확히** 그 block 인자다 (다단 형태는 실패로 본다).
4. 나머지 피연산자 `advance`가 loop region 밖에서 정의된다.
5. `loop_step`이 상수 0이 아니다.

**재작성은 RAUW로 끝난다.** body 첫머리에 `%off = initial + iteration·advance`를 만들고 block
인자의 모든 사용처를 `%off`로 치환한다. `addi`는 건드리지 않는다 — RAUW 후 yield가
`addi(%off, advance) = initial + (iteration+1)·advance`를 내므로 loop 의미가 보존되고,
**loop 결과가 쓰이든 말든 무관하게 적법하다.** iter_arg 제거는 별개의 선택적 정리이고, 세
커널은 index 결과가 전부 loop 뒤에서 죽어 있어 안전하다.

### 진단

무조건 에러를 내면 주소와 무관한 index iter_arg까지 잡는다. **"주소 운반자"로 범위를 한정하고,
그 안에서는 예외 없이 에러**를 낸다. 주소 운반자 판정: block 인자에서
`addi/muli/subi/index_cast`만 거쳐 `memref.reinterpret_cast`의 `offset` 피연산자에 도달한다.

```
error: pim-normalize-address: address recurrence carried by iter_arg #0 cannot be put in
       closed form: the yielded value is not addi(<iter_arg>, advance) with advance defined
       outside the loop
note:  PIM placement needs offset(i) = initial + i*advance to recover the per-core region of
       this buffer; without it no data placement can be computed
note:  index with an affine expression of the loop variable (a_ptr + (k*BLOCK_K)*stride_ak)
       instead of accumulating the pointer (a_ptrs += BLOCK_K*stride_ak)
```

## 3.2 `origin` 복원은 stride SSA 동일성으로 되지 않는다

앞 문서에서 "정규화 뒤에는 축별 origin 인식이 곱셈 항 매칭만으로 끝난다"고 적었다. **틀렸다.**
B측이 여전히 실패한다.

`decomposeOffset`(`TTSharedToPIMPass.cpp:49-88`)은 항의 `muli` 피연산자를 stride 목록과
**SSA 동일성**으로 대조한다(`mul.getRhs() == s`). gemm의 정규화된 B offset은
`addi(muli(%iv, %12), %5)`이고 `%12 = index_cast(muli(%arg7, %c16_i32))`인데
(`gemm_kernel.ttsharedir:50-51`), stride 목록에 있는 것은 `%8 = index_cast(%arg7)`이다(`:47`).
둘은 다른 SSA 값이라 매칭이 실패하고, 두 항이 모두 "stride == 1" 축으로 떨어져 충돌한 뒤
`return {}`이 된다(`:79-80`). **B의 origin은 비어 있는 채로 남는다.**

**해법: `--pim-recover-access-origin`을 별도 pass로 두고, stride SSA 동일성이 아니라 "이 항이
어느 logical 좌표를 포함하는가"로 축을 배정한다.** `--pim-build-logical-grid`가 방금
`output_row_tile`, `output_column_tile`, `explicit_k_partition`의 SSA 값을 만들었으므로, 각
항의 backward slice에 어느 좌표가 나타나는지만 보면 된다.

이것이 `--convert-ttshared-to-pim`의 출력 불변식에서 **`origin`이 비어 있어도 된다**고 적는
이유다.

## 3.3 세 커널의 복원식

### `gemm.py` — 항등

grid nest의 두 loop 유도변수는 `arith.muli %iv, <BLOCK>` 로만 쓰인다
(`gemm_kernel.ttsharedir:41,43`). `divsi`/`remsi`/`minsi`가 없다. extent는 grid loop 상한
그대로다. **`ceildiv(M, BLOCK_M)`으로 다시 유도하면 안 된다** — `M`과 launch grid는 독립된
런타임 값이다.

### `split_k.py` — 런타임 divisor 위의 div/rem 쌍

```
:32  %3 = divsi(addi(%arg4,127), 128) ; divisor = ceildiv(N, BLOCK_N)      ← 런타임
:33  %4 = divsi(%arg13, %3)           ; output_row_tile    = pid / divisor
:34  %5 = remsi(%arg13, %3)           ; output_column_tile = pid % divisor
:39  %10 = muli(%arg14, 64)           ; explicit_k_partition = program_id(1)
```

매처가 증명해야 하는 것:

1. `divsi`와 `remsi`의 **RHS가 동일한 SSA 값**이다. 이것이 division-with-remainder 쌍 판정이다.
2. divisor가 grid nest 전체를 지배한다. divisor의 backward slice의 모든 잎이 `func.func`
   block 인자이거나 상수임을 구조적으로 확인한다. 이 커널의 잎 집합은 `{%arg4}`다.
3. `divisor == ceildiv(N, BLOCK_N)`은 **선택적**이다. 1+2만으로 좌표 복원은 이미 건전하다.
4. `number_of_output_row_tiles`가 이 IR에 없다. `ceildiv(gridX, divisor)`로 유도한다 —
   launcher가 `gridX == cdiv(M,BM)·cdiv(N,BN)`를 보장한다(`split_k.py:59`).
5. **`SPLIT_K`는 constexpr에서 못 읽는다.** ttshared 단계에서 constexpr는 이미 사라졌다.
   K loop 전진량 리터럴 `1024`(`:103`)와 k tile 폭 `64`(`:51`)의 비 `1024/64 = 16`으로
   복원한다. 이것이 `SPLIT_K`가 보이는 유일한 자리다.

### `grouped_gemm.py` — 13-op 사슬

```
:28  %3  = divsi(addi(%arg3,31), 32)  ; number_of_output_row_tiles      런타임
:30  %5  = divsi(addi(%arg4,31), 32)  ; number_of_output_column_tiles   런타임
:31  %6  = muli(%5, 2)                ; tiles_per_group
:32  %7  = divsi(%arg12, %6)          ; group_index
:33  %8  = muli(%7, 2)                ; first_row_tile_of_group
:35  %10 = minsi(subi(%3,%8), 2)      ; rows_in_group                   런타임
:36  %11 = remsi(%arg12, %6)          ; index_within_group
:38  %13 = addi(%8, remsi(%11,%10))   ; output_row_tile
:39  %14 = divsi(%11, %10)            ; output_column_tile
```

**전단사 증명** (배치를 logical tile 좌표 위에서 해도 된다는 근거): `GROUP_M = G`,
row tile 수 `R`, column tile 수 `C`라 하자. `group_index < ⌊R/G⌋`이면 `rows_in_group = G`이고
`index_within_group`이 `[0, G·C)` 전체를 훑으므로 `r ↦ (r mod G, r div G)`가 `[0,G)×[0,C)`로
전단사. 꼬리 그룹에서는 `rows_in_group = R - first ∈ [1,G)`이고 `index_within_group`이
`[0, rows_in_group·C)`를 훑으므로 전단사. 두 상이 `[0,R)×[0,C)`를 분할한다. **`minsi`가 바로
꼬리를 전단사로 만드는 장치다.**

**단 grid를 과다 launch하면 전단사가 깨지고 mask로 막히지도 않는다.** `R=C=4, G=2, pid=16`이면
`rows_in_group=min(0,2)=0`이 되어 divisor가 0이다 — 정수 나눗셈 예외이지 masked store가
아니다. `grouped_gemm.py:74`가 정확히 `R·C`만 launch하므로 오늘은 발화하지 않지만, 복원
pass가 masked-store fallback에 기대면 안 된다.

## 3.4 grouped 순서는 스케줄이지 배치가 아니다 — 조건부로 참

주장은 **복원 pass가 `pid` loop 하나를 2중 nest로 바꿀 때에만** 참이다. 지금 IR 그대로는
§1.3에서 본 대로 거짓이고 조용히 틀린다. 재구성 뒤에는 origin 축마다 정확히 하나의 loop가
대응하므로 `driverLoop`가 서로 다른 loop를 돌려주고 `sizes`가 옳은 직사각형이 된다. 그리고
**13-op 사슬은 통째로 죽어 사라진다.**

순서 자체는 **구조가 아니라 metadata로** 보존한다:
`pim.schedule = #pim.schedule<order = "grouped", axis = "output_row", group = 2>`.

(i) device는 동시 실행되고 데이터를 공유하지 않으므로 device 간 순서는 의미가 없다.
(ii) device 안에서도 오늘은 순서가 아무것도 바꾸지 않는다 — `pim.access`가 K loop 안에 있어
tile마다 MRAM→WRAM DMA를 다시 낸다(`gemm_kernel.pim_common:91,93`).

---

# 4. Execution wave와 PIM Device mapping

## 4.1 이름

축약 대신 전부 full name을 쓴다. `M`, `N`, `K`만 GEMM 표준 표기라 유지한다.

| 이름 | 뜻 |
|---|---|
| `number_of_output_row_tiles` / `..._column_tiles` | `ceildiv(M, BLOCK_M)` / `ceildiv(N, BLOCK_N)`. **런타임** |
| `number_of_explicit_k_partitions` | 커널이 선언한 `SPLIT_K`. 없으면 1. 컴파일 시점 |
| `logical_tile_core_footprint` | logical tile 하나가 차지하는 PIM Core subset의 shape |
| `number_of_logical_tiles_per_wave` | core shape 위의 non-overlapping footprint 배치 수 |
| `number_of_available_pim_devices` | profile의 `number_of_pim_devices` |
| `number_of_output_row_device_partitions` 등 | 컴파일 시점 |
| `output_row_partition_index` 등 | 이 device의 좌표 |
| `minimum_tiles_per_partition` / `number_of_partitions_with_one_additional_tile` | 몫 / 나머지 |
| `first_tile_index` / `number_of_tiles_assigned` | 이 device가 맡는 첫 tile 번호 / 개수 |
| `tile_capacity_per_partition` | 버퍼 크기용. **device 불변이어야 한다** |

```
device_index =
    (
      output_row_partition_index
      × number_of_output_column_device_partitions
      + output_column_partition_index
    )
    × number_of_explicit_k_device_partitions
    + explicit_k_partition_index
```

이 자릿수 배열은 **바꿀 수 없다** — host emitter의 좌표 해체가 row-major 마지막 축 우선으로
고정돼 있고(`UPMEMToHost.cpp:158-170`), reduce 축이 가장 안쪽이라고 가정한다(`:443-446`).

## 4.2 wave 크기는 scalar 나눗셈이 아니라 topology 위의 배치 수다

**앞 판의 `cores_per_group / cores_per_tile`은 AiMX에서 우연히 맞을 뿐이다.** PIM Core 16개가
한 channel의 16 bank인지, 여러 channel에 흩어진 16개인지, 물리적으로 인접하지 않은 16개인지를
구별하지 못한다.

**필요한 것은 shape다.**

```
PIM Core shape                [8 channels, 16 banks]
logical tile core footprint   [1 channel, 16 banks]
non-overlapping 배치 수        8
```

`--pim-form-execution-waves`가 검사할 것:

1. footprint의 각 축이 core shape의 대응 축을 나눈다.
2. 하나의 logical tile이 **연속된** core subset을 쓴다.
3. 서로 다른 logical tile의 core subset이 겹치지 않는다.

| target | core shape | footprint | tiles / wave |
|---|---|---|---:|
| UPMEM | `[1]` | `[1]` | **1** |
| AiMX | `[8, 16]` | `[1, 16]` | **8** |

UPMEM에서는 wave가 tile 하나라 이 pass가 항등이다. AiMX의 기본 wave 구성은 **같은 output-row
tile과 같은 reduction range를 공유하는 연속된 output-column tile 8개**다.

```
one PIM Group execution wave
├── channel 0 → output-column tile  c
├── channel 1 → output-column tile  c+1
...
└── channel 7 → output-column tile  c+7
```

한 wave가 내는 output 범위는 `16 × (8 × 16)` = `16 × 128` element다. **이것은 `BLOCK_N = 128`을
뜻하지 않는다** — logical tiling과 hardware execution bundle은 다른 층이다.

`number_of_output_column_tiles`가 8의 배수가 아니면 마지막 wave에 빈 channel이 생긴다.
`requires_uniform_instruction_sequence = true`이면 **group 안에서 padding**한다(§6.3).

## 4.3 device mapping 선택 알고리즘

입력은 profile 값과 `number_of_explicit_k_partitions`뿐이다. tile 개수는 **런타임이라 쓸 수
없다.**

```
1. explicit split-K에서만 K를 나눈다. reduction loop는 절대 나누지 않는다.
   number_of_explicit_k_device_partitions =
       (number_of_explicit_k_partitions > 1) ? <그 값의 약수 중 선택> : 1
   remaining = number_of_available_pim_devices / number_of_explicit_k_device_partitions

2. remaining 을 output_row × output_column 으로 인수분해한다.
   M, N을 컴파일 시점에 모르므로 중립 선택은 정사각 인수분해다.
   |row_partitions - column_partitions| 를 최소화하고, 동률이면 row >= column.
   (전송량 점수식을 쓰지 않는다 — M, N이 없으면 그 식은 M=N을 암묵 가정하고
    직사각 GEMM을 잘못 나눈다.)

3. #pim.upmem_ext 의 number_of_devices_per_transfer_group 이 있으면, 동률 후보 중
   column partition 수가 그 값의 배수인 쪽을 고른다. push_xfer 묶음이 좋아진다.
```

**놀고 있는 device는 숨기지 않고 remark로 보고한다.** 남는 것을 reduction loop 분할로 메우지
않는다 — 그러면 합성 combine이 필요해지고, 그것은 금지 사항이다.

**AiMX에서는 이 pass가 하는 일이 작다.** device가 8개뿐이라 인수분해 후보가 몇 개 없다.
AiMX의 실제 병렬성은 §5.2의 core mapping에 있다.

## 4.4 balanced block — `mod`가 아니라

`mod`(cyclic) 분배의 **동기는 옳다.** 단순 blocked 분할이 꼬리에서 device를 놀린다는 것은
저장소가 이미 결함으로 기록한 사실이다(`UPMEMTarget_Implementation.md:444-446`).

**그런데 `mod`는 데이터 영역을 block-cyclic으로 만든다.** device가 맡는 tile이
`{i, i+P, i+2P, …}`이므로 행으로 옮기면 등간격 집합이다. 현행 전송 경로는 **축마다 연속 구간
하나**만 표현한다 — `origin[d]/tile[d]`을 MRAM tile 번호로 쓰고
(`PIMToUPMEMPass.cpp:644-656`) host는 `origin[d] + _t·tile[d] + _e`만 훑는다
(`UPMEMToHost.cpp:313-321, 335-344`). **경계 검사가 없어 조용히 틀린다.**

전송량은 두 방식이 같다. device당 tile 수가 양쪽 다 `⌈G/P⌉`이기 때문이다.

**해법: balanced block.** `mod`의 균형을 얻으면서 축마다 연속 구간 하나를 유지한다.

```
minimum_tiles_per_partition =
    number_of_output_row_tiles / number_of_output_row_device_partitions

number_of_partitions_with_one_additional_tile =
    number_of_output_row_tiles % number_of_output_row_device_partitions

first_tile_index =
    output_row_partition_index × minimum_tiles_per_partition
  + min(output_row_partition_index, number_of_partitions_with_one_additional_tile)

number_of_tiles_assigned =
    minimum_tiles_per_partition
  + (output_row_partition_index < number_of_partitions_with_one_additional_tile ? 1 : 0)

tile_capacity_per_partition =
    minimum_tiles_per_partition
  + (number_of_partitions_with_one_additional_tile != 0 ? 1 : 0)
```

row tile 129개, partition 64개: 한 device가 3 tile, 63개가 2 tile, **놀지 않는다.** 단순
blocked면 `⌈129/64⌉ = 3`이라 43개만 일하고 21개가 논다. `arith` 7개로 끝나고,
`tile_capacity_per_partition`이 오늘의 `per = ceildiv(G,P)`와 수치가 같아 버퍼 크기 경로가
그대로 유지된다.

**`first_tile_index`가 `device_index × block`이 아니라는 점이 중요하다.** balanced block의
시작점은 나머지 `r` 때문에 `min(index, r)` 항을 갖는다. 그래서 `pim.place`는 이 값을 **직접
싣는다** — 유도식으로 두면 target이 그 식을 다시 해석해야 한다(§6.1).

## 4.5 split-K — 커널이 선언한 것만 보존, 단계 1은 `1`

`split_k.py`의 K 분배는 소스가 정한 것이라 협상 대상이 아니다: `offs_k = pid_k·BLOCK_K`로
시작해 `BLOCK_K·SPLIT_K`씩 전진한다(`:29, 45-46`). IR에도 리터럴 `1024`로 나온다(`:103`).
device가 갖는 K 원소 집합은 블록 64, 주기 1024의 block-cyclic이다.

**밀도 판정식.** 블록 `b`, 주기 `p`, 이 device가 덮는 잉여류 개수 `r`에 대해 `r·b == p`이면
합집합이 연속 구간 하나로 축약된다.

| k device partition | device당 k partition 수 | 판정 | A/B의 K 영역 |
|---|---|---|---|
| `1` | 16 (전부) | `16 × 64 == 1024` → **조밀** | 연속. 현행 경로 그대로 |
| `16` | 1 | `1 × 64 ≠ 1024` → **block-cyclic** | Phase 2 |

> **Phase 1은 k device partition을 1로 고정한다.** `explicit_k_partition`은 device 축이 아니라
> device **안의 실행 wave**가 된다. A·B의 K 영역이 연속이고, 부분합 C는 16개 면이 전부 그
> device에 있어 조밀하다. 수치는 동일하고 combine은 여전히 없다. **일반 cyclic placement 없이
> `split_k.py`가 컴파일된다.**

**별도의 벽 하나.** `split_k.py`의 선언 기본값 `BLOCK 128/128/64`는 어떤 매핑으로도 UPMEM에서
실행되지 않는다. 동시 상주량이 `128·64 + 64·128 + 128·128·4 = 81 920 byte`인데 WRAM 물리
전체가 `65 536 byte`다(`PIMPartitionPass.cpp:224-228`). **더 작은 BLOCK을 요구하는 진단**을
낸다 — tile을 임의로 자르면 reduction 방향에서 combine이 필요해지기 때문이다. 아래 배치표는
권장 BLOCK인 `32/32/16` 변종을 쓴다.

---

# 5. PIM Core mapping

## 5.1 PIM Group은 tile 축이 아니라 command scope다

**앞 판은 output column 축이 device → channel → bank의 세 계층에 걸친다고 적었는데, 틀렸다.**
AiMX에서 channel은 PIM Group이 아니다. FPGA 전체가 하나의 PIM Group이고, channel과 bank는
`[8,16]`으로 구성된 **PIM Core coordinate**다.

```
device partition  → PIM Device
group index       → 항상 0                     (tile 축이 아니다)
channel index     → PIM Core topology의 첫 번째 좌표
bank index        → PIM Core topology의 두 번째 좌표
```

즉 output column placement에 쓰이는 것은 **device level 하나와 core level의 좌표 두 개**이지
세 계층이 아니다.

## 5.2 PIM Core mapping

### UPMEM — 항등

`pim_core_shape = [1]`이므로 device에 할당된 것이 전부 그 하나의 core에 들어간다.

```
device에 할당된 A/B/C shard
  → 그 DPU의 MRAM (attached bank)
  → 현재 tile만 WRAM (local storage)
  → tasklet들이 device에 할당된 tile들을 병렬 실행
```

**UPMEM 경로에서 이 pass가 항등이라는 점이 구현 순서를 정한다.** §10의 단계 1~8까지 이 pass를
비워 두고도 `gemm.py`가 끝까지 간다.

### AiMX — logical tile 하나가 한 channel의 16 core에 걸친다

```
one logical 16 × 16 output tile
  → footprint [1 channel, 16 banks]
  → 각 PIM Core(bank)가 output column 하나
  → 각 PIM Core의 16 arithmetic lane이 output row 원소
```

| logical dimension | AiMX hardware |
|---|---|
| output row element | arithmetic lane |
| output column element | **bank = PIM Core** |
| output column tile | channel (core topology 축 0) |
| reduction dimension | 순차 누적 |
| 여러 FPGA PIM Device | 독립된 wave 묶음 |

`#pim.core_mapping`이 이것을 적는다.

```mlir
#pim.core_mapping<
  core_axis_roles = ["output_column_tile", "output_column"],   // channel, bank
  tile_footprint  = [1, 16],
  lanes           = 16,
  lane_axis_role  = "output_row"
>
```

UPMEM은 `core_axis_roles = []`, `tile_footprint = [1]`, `lane_axis_role = "whole_tile"`
(tasklet마다 다른 tile)이다.

**channel과 bank 좌표를 target lowering까지 미루면 안 된다.** A-side broadcast, B-side
bank-private placement, output co-location이 전부 이 좌표로 결정된다. target lowering으로
미룰 것은 concrete bank-local address, row/column address, alignment와 padding, descriptor
encoding, standard memory request sequence, register load/write-back sequence다.

## 5.3 parallel execution lanes

**`multithread` (UPMEM tasklet).** lane `w`가 평탄화된 wave 번호 `w, w+lanes, w+2·lanes, …`를
실행하고, **lane마다 사적 누산기와 사적 tile 버퍼**를 가지며 **barrier가 필요 없다** — 서로
다른 wave는 겹치지 않는 output tile을 쓴다. `distributeTasklets`가 오늘 그렇게 한다
(`PIMToUPMEMPass.cpp:542-583`). `allocFor`가 tile 버퍼마다 lane 차원을 앞에 붙이는 것(`:370`)이
누산기를 사적으로 만든다. 예산식에 `× lanes`가 붙는다.

**`simd` (AiMX arithmetic lane).** 16 lane이 **한 PIM Core의 계산을 함께 수행하고 누산기를
공유한다**(vACC). 버퍼가 복제되지 않으므로 예산식에 `× lanes`가 붙지 않는다. lane은 IR에
work 분배로 나타나지 않는다 — 16 lane이 나누는 것은 `BLOCK_M = 16`개의 output row 원소이고
PIM Core 하나가 맡은 계산의 내부다. `MemoryLowering_Design.md:255-259`가 `8×16×16`을
③ Hardware intra-unit으로 분류해 target lowering에 배정한 것과 일치한다.

**lane 상한을 `pim.co_resident`로 계산하면 안 된다** (UPMEM). `coResident`는 가장 안쪽 블록의
`pim.load` tile과 loop iter_arg만 센다 — gemm에서 `5120`. 실제로 거부하는 검사는 `allocFor`가
누적한 `wramBytes`이고, 버퍼가 없는 linalg init까지 들어가 **11 264**다.

> **`allocFor`가 순회하는 집합과 같은 집합을 세는 헬퍼 하나를 만들어 이 pass와 PIMToUPMEM이
> 함께 호출한다.**

```
lanes = max(1, min(number_of_parallel_lanes_per_pim_core,
                   usable_local_capacity_bytes > 0
                     ? usable_local_capacity_bytes / bytes_per_lane
                     : number_of_parallel_lanes_per_pim_core))
```

`lanes = 0`을 낼 수 없게 클램프한다. scratchpad가 없는 target(`= 0`)에서는 그 나눗셈이
성립하지 않으므로 능력값을 그대로 쓴다 — §2.6의 세 상태 규칙과 같다.

공통 예산 규칙: **`bytes_per_lane × lanes ≤ usable_local_capacity_bytes`**
(`simd`이면 `× lanes` 없이). 지금 `--pim-partition`은 lane 인자를 빼먹고
(`PIMPartitionPass.cpp:219-229`) target만 포함한다(`PIMToUPMEMPass.cpp:428-438`).

---

# 6. 배치 — `pim.place`

## 6.1 최종 값을 직접 싣는다

**앞 판은 `levels`·`kind`·`period`에서 `start`·`size`·`capacity`를 유도하게 했는데, 세 가지가
문제였다.** (i) `start = Σ index·block`이 balanced block의 `first_tile_index`를 표현하지
못한다(§4.4). (ii) `kind`가 축 전체에 하나인데 level은 여럿이라 nested mapping에서 의미가
불명확해진다. (iii) Phase 1은 일반 cyclic placement가 필요 없다 — 연속 block, replication,
split-K용 `base_offset` 셋뿐이다.

**그래서 최종 placement 값을 직접 materialize한다.**

```mlir
pim.place %buffer
    start    [%s0, %s1]        // SSA index. balanced block 이 낸 값 그대로
    size     [%n0, %n1]        // SSA index. extent 로 클램프된 값
    capacity [%c0, %c1]        // SSA index. partition 불변
    extent   [%e0, %e1]        // SSA index. 원본 버퍼의 실제 크기
    stride   [%t0, %t1]        // SSA index
    base_offset %b             // SSA index. 축과 직교한 스칼라. §6.4
    {direction = in, access = 0 : i64,
     replicated_over = ["output_column_device_partition"],
     mapping = #pim.place_mapping<...>}     // 정적 metadata. 아래
  : memref<*xi8>
```

`start`/`size`/`capacity`/`extent`/`stride`는 오늘의 `pim.unit_data`와 같은 모양이라 전이
하강이 거의 항등이다. 늘어난 것은 `extent`(§6.3)와 `base_offset`(§6.4) 둘뿐이다.

`mapping`은 **이 값을 어떻게 만들었는지 설명하는 정적 metadata**다 — 어느 device partition,
어느 core 좌표가 어느 축을 골랐는지. 검증과 진단에만 쓰이고, **target lowering은 이것을
해석하지 않고 최종 region만 소비한다.** 이 분리가 앞 판보다 단순하고 지금 구현을 그대로
살린다.

`replicated_over`는 정적 문자열 목록이다. 덤프로 검증된다 —
`g09_epilogue_bias.pim_common:50` A → `["y"]`, `:56` B → `["x"]`, `:57` bias → `["x"]`,
`:58` C → 없음.

## 6.2 배치는 output tile ownership에서 시작한다

A와 B를 단순히 같은 크기로 나누면 GEMM의 reuse 관계를 잃는다. 순서가 정해져 있다.

```
1. logical output tile ownership 을 먼저 결정한다        (§4)
2. 그 output 을 계산하는 데 필요한 A/B region 을 유도한다
3. 같은 device 또는 PIM Core 에 co-locate 한다
4. 불변 operand 만 필요한 범위에 replication 한다
5. capacity 를 넘으면 여러 execution wave 로 처리한다
```

Plain GEMM의 device 간 분배:

| tensor | device 간 분배 | replication |
|---|---|---|
| A | output-row 방향으로 partition | output-column device partition에 대해 replication |
| B | output-column 방향으로 partition | output-row device partition에 대해 replication |
| C | output-row와 output-column 방향으로 partition | 없음 |

**복제 판정이 살아 있어야 전송량이 폭발하지 않는다.** `replicated_over`가 비면
`distinctCount`(`UPMEMToHost.cpp:283-292`)가 device 수를 그대로 곱한다. 단 이 dedup은 **host
측 staging 중복 제거일 뿐**이고, 같은 바이트가 여전히 모든 DPU로 간다(`:357-368`).

## 6.3 mask와 padding — target-independent하지 않다

### 지금 조용히 틀리고 있는 것

`pim.access`는 `valid` + `pad`를 나른다(`PIMDialect.td:66-74`). 그런데 이 값을 읽는 곳은
저장소 전체에서 의존 수집(`PIMAnalyzeReusePass.cpp:109,111`)과 verifier(`PIMDialect.cpp:45,51`)
뿐이다. `pim.unit_data`에는 `valid` 필드가 없고, `upmem.stage`/`unstage`에도 없고, host loop은
클램프되지 않는다(`UPMEMToHost.cpp:335-344, 313-321`).

저장소가 만든 산출물로 확인된다. `G05_masked_boundary`, `M=N=K=72`, `BLOCK 32/32/16`, f32,
device 1개. `report.csv`는 `compiled`로 기록한다 — 에러가 없다. PIM 단계 IR은 mask를 갖고 있고
(`g05_masked_boundary.pim_common:90` `valid[%48, %72] pad %cst`) target IR에서 사라진다
(`.target_ir:164`).

`⌈72/32⌉ = 3`, `⌈72/16⌉ = 5`이므로 A의 stage sizes는 `[96, 80]`인데 원본은 `72×72`,
stride 72다.

| | 증상 |
|---|---|
| (a) | `A_tile[0][k=72]`가 선형 `0·72+72 = A[1][0]`에서 읽힌다. Triton은 0을 요구한다 |
| (b) | 최대 선형 색인 `95·72+79 = 6919` — 5184 원소 할당의 **범위 밖 읽기** |
| (c) | unstage가 `[96,96)`을 쓴다. `r<72, c∈[72,96)`에서 `r·72+c == (r+1)·72+(c-72)` — `C[1][0]`이 두 번 쓰이고, 그 loop이 `#pragma omp parallel for collapse(2)`라(`:326-334`) **경합**이다 |
| (d) | 최대 선형 `95·72+95 = 6935` — **범위 밖 쓰기** |

### 공통 계층이 하는 일

`pim.place`가 축별 `extent`를 나르고, `--pim-place-data`가 `size`를 `extent`로 클램프한다.
**여기까지가 target-independent다.**

### target별 padding은 다르다

| | UPMEM | AiMX |
|---|---|---|
| 기구 | host staging loop에 `if (start + _t·tile + _e < extent)` 가드. `calloc`(`UPMEMToHost.cpp:377`)이 건너뛴 자리에 0을 공급 | **descriptor 수준** |
| 보장해야 하는 것 | 범위 밖 읽기·쓰기 없음 | ① invalid input load가 0을 만든다 ② descriptor가 invalid memory를 접근하지 않는다 ③ computation이 lockstep을 유지한다 ④ invalid output store가 억제된다 |

**AiMX의 빈 channel에 `extent = 0`만 두는 것으로는 부족하다.** store는 막히지만 input load의
correctness가 보장되지 않는다. target lowering이 predicated load 또는 명시적 zero operand를
만들어야 한다.

**Phase 1은 `pad = 0`만 지원한다.** nonzero pad는 진단을 낸다 — 세 커널이 전부 `other=0`이고
(`gemm.py:47`, `split_k.py:40`, `grouped_gemm.py:51`), 0이 아닌 pad는 `calloc` 기구로 공짜로
나오지 않아 별도 채우기가 필요하다.

## 6.4 `split_k.py`의 부분합 C — `base_offset`이 없으면 표현 불가

`split_k.py:57`이 `torch.zeros((SPLIT_K, M, N))`을 잡고 `:64-65`가
`c.stride(1), c.stride(2), c.stride(0)`을 넘기므로 `stride_ck = M·N`이다. 그런데 C의
`reinterpret_cast`는 **rank 2**이고 strides가 `[stride_cm, 1]`이다(`:113`) —
`stride_ck`(`%arg9`)는 그 목록에 **없다**.

`decomposeOffset`은 `:107-112`의 addi 사슬을 세 항으로 펼치는데 하나만 stride와 매칭되고
나머지 둘이 "stride == 1" 축으로 몰려 충돌한다 → **C의 origin이 비고 `unit_data`가 아예 생기지
않는다**(`PIMPartitionPass.cpp:279-282`).

축을 하나 더 만들어 rank 3으로 가는 우회는 깨진다. `buildDMA`가 rank를 access가 아니라 **WRAM
버퍼**에서 가져오기 때문이다(`PIMToUPMEMPass.cpp:614-618`) — 그 버퍼는 rank-2 누산기다.
offset loop(`:644-656`)이 `d=0..1`만 돌아 **`origin[2]`을 아예 읽지 않는다.** 게다가
`emitMatmul`이 rank-2가 아닌 WRAM 버퍼를 거부한다(`UPMEMToDevice.cpp:204-206`).

> **해법: `pim.place`의 스칼라 `base_offset`.** 선형 주소에 더해지고 tile 축과 직교한다 —
> `explicit_k_partition_index · stride_ck`이 정확히 그 형태다. `upmem.stage`/`unstage`가 `src`
> 대신 `src + base_offset`을 쓰면 host 쪽 덧셈 하나로 끝나고 `buildDMA`는 손대지 않는다.

`buildDMA`의 `rank`도 `access.getOrigin().size()`에서 가져오고 WRAM rank와 같은지 명시적으로
검사해, rank 불일치가 조용한 주소 절단이 아니라 진단이 되게 한다.

## 6.5 UPMEM 배치표

`M = N = K = 4096`, int8 × int8 → int32, `number_of_pim_devices = 2048`.

| | `gemm.py` | `split_k.py` (32/32/16 변종) | `grouped_gemm.py` |
|---|---|---|---|
| BLOCK M/N/K | 32 / 32 / 16 | 32 / 32 / 16, `SPLIT_K=16` | 32 / 32 / 32, `GROUP_M=2` |
| source grid | `(128, 128)` | `(16384, 16)` | `(16384)` |
| logical tile 수 | row 128, col 128 | row 128, col 128, k part 16 | row 128, col 128 |
| tiles / wave | 1 | 1 | 1 |
| row × col × k device partitions | `64 × 32 × 1` | `64 × 32 × 1` | `64 × 32 × 1` |
| 사용 device | 2048 / 2048 | 2048 / 2048 | 2048 / 2048 |
| device당 실행 wave | `2 × 4 = 8` | `2 × 4 × 16 = 128` | `2 × 4 = 8` |
| **A** | row block, K 전체 → 64행×4096 = **256 KiB**, column partition에 복제 | 동일. K **연속** | 동일 |
| **B** | K 전체, column block → 4096×128 = **512 KiB**, row partition에 복제 | 동일 | 동일 |
| **C** | 64×128 i32 = **32 KiB**, 복제 없음, combine 없음 | 부분합 `[16][64][128]` i32 = **512 KiB**, `base_offset = k_index·M·N`, **combine 없음** | 32 KiB |
| 최종 C | — | **커널 IR에 없다.** Python `c.sum(dim=0)` | — |
| core mapping | 항등 (core 1개) | 항등 | 항등 |

## 6.6 AiMX operand flow — Silent-PIM 계열

`BLOCK 16/16/16`, bf16 입력 / fp32 누산일 때 한 logical tile은

```
C tile = 16 output rows × 16 output columns
A tile = 16 output rows × 16 reduction elements
B tile = 16 reduction elements × 16 output columns
```

### A-side — streamed / broadcast operand

고정된 reduction position에서 `A[:, reduction_index]`는 bf16 16개, 정확히 256-bit word
하나다. vecA가 word 4개를 담으므로 네 reduction position을 한 번에 들 수 있다.

```
vecA word 0..3 = A[:, reduction_index + 0..3]
```

이 vector는 **같은 channel의 16개 PIM Core가 공유한다.** A는 channel 안에서 복제 또는
broadcast이고 `replicated_over`에 core 축 `bank`이 들어간다.

### B-side — bank-private operand

PIM Core 하나가 output column 하나를 맡으므로 그 core의 B는

```
B[16 reduction elements, one output column] = bf16 16개 = 32 byte
```

이고 **vecB 하나에 정확히 들어간다.** 이것이 `BLOCK_K = 16`의 근거다.

```
bank 0..15  vecB = B[:, output_column + 0..15]
```

**bank마다 주소가 다르다.** 그래서 `requires_uniform_memory_address = false`여야 한다(§2.7).

### 누산기 — 아직 증명되지 않은 legality

PIM Core 하나가 output column 하나에 대한 `BLOCK_M`개의 fp32 부분합을 들고 있어야 한다.

```
vACC ≥ BLOCK_M × 4 byte = 64 byte
```

`accumulator_register_capacity_bytes`가 미측정이라 **`BLOCK_M = 16`의 legality가 아직
증명되지 않았다.** vecA와 vecB 용량만으로는 vACC legality를 증명할 수 없다. §2.6의 세 번째
상태(필드 없음)라 AiMX legality gate가 그 자리에서 진단을 낸다.

### 전체 mapping

```
AiMX Card
→ 8 FPGA PIM Devices
→ one PIM Group per FPGA
→ [8 channels, 16 banks] PIM Cores per group
→ one bank per PIM Core
→ 16 arithmetic lanes per PIM Core
```

```
Triton Grid
→ logical 16 × 16 output tiles
→ 같은 output-row 좌표를 갖는 tile들을 묶는다
→ 연속된 output-column tile 최대 8개가 하나의 lockstep wave  (footprint [1,16] × 8)
→ wave들을 선택된 FPGA PIM Device에 균형 분배 (device 사이는 lockstep 아님)
→ 각 channel이 logical tile 하나
→ 각 bank가 output-column fragment 하나
→ 16 arithmetic lane이 output-row 원소
→ reduction 원소는 순차 누적
```

---

# 7. op set — 최소로

판정 기준을 하나로 고정한다. **어떤 PIM pass가 그것을 읽을 때에만 PIM op이다.**

## 7.1 최종 op — 5개 (오늘 8개)

```mlir
pim.access     논리 주소 명세 (base, offset, origin, strides, sizes, valid, pad)
pim.load       access → tensor
pim.store      tensor → access
pim.device     PIM Device 하나가 하는 일
               (region, NoTerminator, block 인자 = device partition 좌표)
pim.place      버퍼 하나의 최종 배치 (§6.1)
```

attribute 5개 + loop attribute 1개:

```
#pim.target_profile    module. 공통 8필드 (§2.3)
#pim.upmem_ext / #pim.aimx_ext   module. target 확장 (§2.4)
#pim.device_mapping    pim.device의 필수 파라미터
#pim.core_mapping      core_axis_roles / tile_footprint / lanes / lane_axis_role
#pim.place_mapping     pim.place의 정적 metadata. 검증·진단 전용
pim.axis               scf.for
```

`#pim.device_mapping`을 **필수 파라미터**로 두면 매핑 전에는 op이 존재하지 않고 매핑 후에는
null일 수 없어, `state = logical | placed` enum이 만들 "표현 가능하지만 불법인" 상태가 생기지
않는다. func 계층 attribute로 두면 안 되는 이유도 있다 — `PIMPartitionPass.cpp:264`가
`pim.grid_dims`를 무방비로 가져오고 `:318`이 다른 attribute의 가드 안에서 그것을 색인하는,
실재하는 null 역참조 부류가 그것이다.

## 7.2 불채택과 근거

| 후보 | 판정 | 근거 |
|---|---|---|
| PIM Group region op | **Phase 1 불채택** | 두 target 모두 device당 group이 1개다. `pim_group_index = 0`이 profile에서 유도되고 lockstep legality는 `--pim-verify`가 본다. **device당 group이 2개 이상인 target이 붙으면 중첩 region이 필요해진다** |
| PIM Core region op | **불채택** | core는 배치의 좌표이지 프로그램의 scope가 아니다 |
| `pim.grid` | **불채택 → `pim.axis` loop attribute** | target이 축↔loop를 이미 `scf.for`의 attribute로 묶는다(`PIMToUPMEMPass.cpp:291-298`) |
| `pim.co_locate` | **불채택 (유도됨)** | A·B·C의 `place`가 같은 partition·core 좌표를 참조하므로 정의상 같은 곳에 온다 |
| `pim.yield` | **불채택** | region이 SSA 값을 내지 않는다. `pim.units`가 오늘 `NoTerminator`다(`PIMDialect.td:142`) |
| `pim.copy` | **삭제** | 하강 소비자 0. 남으면 `PIMToUPMEMPass.cpp:529-534`에서 에러 |
| `pim.gather` / `pim.scatter` | **보류** | 세 커널이 만들지 않는다. §8.2의 진단 대상 |
| `pim.units` / `pim.unit_data` | **한시 유지** | 전이 표현 |

`pim.place`가 `pim.access`의 attribute가 될 수 없는 이유 넷 — 전부 덤프로 확인된다.
(1) **scope** — `unit_data`는 region body의 직계 자식으로 수집되고(`:121-123`) `pim.access`는
loop nest 깊숙이 있다. (2) **1:1이 아니다** — 하나의 access가 load와 store 양쪽에 쓰이고,
반대로 여러 access가 한 버퍼를 가리킨다. (3) **매개변수가 다르다** — 배치는 partition index의
함수이고 access는 wave 좌표의 함수다. (4) **수명이 다르다** — `PIMToUPMEM`이 모든
`pim.access`를 지우지만(`:497`) 배치에서 만든 staging은 그보다 오래 살아야 한다(`:239-247`).

## 7.3 `pim.contract`와 `pim.elementwise` — 결정이 필요하다

**정보량이 세 커널에서 0이다.** 셋 다 canonical rank-2 `(M,K)×(K,N)`이므로 축 역할이 영구히
항등이고, `linalg.matmul`은 그 역할을 **위치로 이미 고정**한다.

도입 비용은 실재한다 — `LinalgOp` 인터페이스로 분기하는 지점 세 곳을 다시 써야 한다:
WRAM 버퍼 배정(`PIMToUPMEMPass.cpp:410-426`), tensor→memref 재생성(`:441-458`), device
emitter(`UPMEMToDevice.cpp:179-185`).

**`pim.elementwise`도 같다.** 다만 앞 판단을 정정한다 — "emitter가 `linalg.generic`을 이미
처리한다"고 가정했는데 **처리하지 않는다.** `UPMEMToDevice.cpp:178-185`가 C 형태를 갖는 것은
`linalg::FillOp`와 `linalg::MatmulOp` 둘뿐이고 나머지는
`emitError("upmem-to-device: no C form for this op")`로 떨어진다. G10이 정확히 이 에러로
실패한다. 그러나 **emitter의 공백이지 IR 표현의 공백이 아니다.**

**권고: Phase 1에서는 도입하지 않는다.** Phase 2에서는 바뀐다 — AttAcc full attention이
GEMV·reduction·softmax·accumulator·barrier를 구분해야 하므로 op 이름이 의미를 나르는 편이
유리하다. 그때 함께 도입한다.

---

# 8. pass pipeline

| # | pass | 하는 일 하나 | 검증 커널 |
|---|---|---|---|
| 1 | `--pim-normalize-address` | 주소 재귀를 닫힌 형태로 | 3개 전부 |
| 2 | `-cse -loop-invariant-code-motion -canonicalize` | — | 3개 전부 |
| 3 | `--convert-ttshared-to-pim` | 관용구 → `pim.access`/`load`/`store`. **`origin`은 빈다** | 3개 전부 |
| 4 | `--pim-build-logical-grid` | source grid → logical tile loop nest | **grouped가 판별 케이스** |
| 5 | `--pim-recover-access-origin` | access의 축별 origin을 logical 좌표로 채운다 | 3개 전부 |
| 6 | `--pim-form-compute` | 누산기를 matmul의 `outs`로 통합 | 3개 전부 |
| 7 | `--pim-form-execution-waves` | core footprint로 wave 구성 + padding. **UPMEM 항등** | AiMX |
| 8 | `--pim-map-devices` | wave → PIM Device 균형 분배 | 3개 전부 |
| 9 | `--pim-map-cores` | tile fragment → core 좌표, lane 수 선택. **UPMEM은 lane만** | AiMX |
| 10 | `--pim-place-data` | 최종 region·replication·capacity materialize | 3개 전부 |
| 11 | `--pim-verify` | 공통 fail-fast 검증 (§8.1) | 3개 전부 |
| 12 | `--pim-legacy-unit-form` | 전이용 호환 하강 | gemm 먼저 |
| 13 | `-cse -canonicalize` | — | 3개 전부 |
| 14 | `--convert-pim-to-upmem` | 진입 즉시 UPMEM legality gate | gemm |

앞 판보다 두 가지가 바뀌었다. **`--pim-logical-grid`를 4·5로 쪼갰다** — 하나가 loop nest 생성,
좌표 복원, origin 복원, 의존 분석을 다 하고 있었다. 의존 분석은 유일한 소비자가 replication
판정이라 10번에 흡수했다. **`--pim-select-group-mapping`을 없앴다** — Phase 1에서 항상
`pim_group_index = 0`이라 mapping이 없고, legality만 남는데 그것은 11번의 몫이다.

**UPMEM 경로에서 7번이 항등이고 9번은 lane 선택만 한다.** 세 커널을 초록으로 만드는 데
필요한 새 pass는 1·4·5·6·8·10·11·12 여덟 개다.

AiMX target 변환은 `--convert-pim-to-aimx`가 descriptor와 standard memory request sequence를
생성한다.

## 8.1 `--pim-verify`가 검사하는 것

**정적 값과 런타임 값을 구분한다.** `start`, `size`, `extent`가 런타임 SSA면 컴파일 시점
verifier가 부등식을 증명할 수 없다.

| 값 | 검사 방식 |
|---|---|
| 셋 다 정적 | 컴파일 시점에 부등식을 직접 검사 |
| 하나라도 런타임 | **placement pass가 clamp 식을 만들었는지 구조적으로 검사** — `size`의 정의 op이 `min`/`max` 사슬인지 |
| 그래도 필요하면 | host 또는 target 코드에 런타임 guard 생성 |

부등식 자체도 고친다. **경계 partition에서 `size = 0`인데 `start > extent`일 수 있고, 그
partition은 합법이다.** `start + size ≤ extent`는 그때 실패한다.

```
size ≥ 0
size > 0 이면:  start < extent  이고  size ≤ extent - start
```

또는 placement pass가 빈 partition의 `start`를 `extent`로 클램프한다. 둘 중 하나를 규약으로
정하고 verifier가 그것을 검사한다.

나머지 항목:

1. 모든 `pim.access`에 대응하는 `pim.place`가 있고 `pim.access_id`가 **양방향으로** 유일하다.
   (`PIMToUPMEMPass.cpp:154-156, 613-614`가 null 검사 없이 역참조하고 `DenseMap::at`을 쓴다)
2. `pim.access.origin`이 비어 있지 않고 rank가 `sizes`와 같다. (`buildDMA:644,651`)
3. `valid`/`pad`를 가진 access의 `place`에 `extent`가 있고 지배한다. `pad`가 0이 아니면 진단.
4. device partition의 곱 ≤ `number_of_pim_devices`.
5. `tile_footprint`가 `pim_core_shape`를 나누고, tile들의 core subset이 연속이며 겹치지 않는다.
6. `bytes_per_lane × lanes ≤ usable_local_capacity_bytes`
   (`parallel_execution_kind = "simd"`면 `× lanes` 없이. `= 0`이면 검사 생략, **필드 없음이면
   진단**).
7. PIM Core당 **버퍼 하나의** 상주량 ≤ `usable_bank_capacity_bytes / 버퍼 개수`.
   **필드 없음이면 진단.**
8. `requires_uniform_instruction_sequence = true`이면 **한 PIM Group 안의** 모든 PIM Core가
   같은 반복 횟수와 같은 reduction 길이를 갖는다. **device 사이는 검사하지 않는다**(§2.8).
9. 공배치: 같은 logical 축을 분할하는 두 access가 같은 분할 수를 쓴다.

7번이 나눗셈을 포함하는 이유가 있다. UPMEM은 bank 예산을 **버퍼 개수로 균등 분할**한다 —
`bufferBytes = unitMemoryBytes / max(1, inputs+outputs)` (`PIMToUPMEMPass.cpp:147`). 총합만
검사하면 통과하지만 개별 버퍼가 넘치는 구성이 나온다.

DMA 정렬(8 byte)과 최대 크기(2048 byte)는 여기 두지 않는다 — UPMEM legality gate 소관이고
상수가 `UPMEMDialect.cpp:13-14`에만 살아야 한다. 그 gate가 잡아야 할 것 하나: **지금 생산자가
자기 verifier가 거부하는 IR을 만든다.** `PIMToUPMEMPass.cpp:624-629`의 `perDMA` 축소 loop는
(i) `% 8` 정렬 조건이 없고 (ii) 행 하나가 이미 2048 byte를 넘으면 `perDMA=1`로 빠져나와 초과
전송을 낸다.

## 8.2 미변환 패턴 진단

dialect conversion framework로 옮기지 않는다. 현행 raise는 pattern이 아니라 독립 walk 8개라
482줄 전면 재작성이 되고, `linalg`는 6번까지 합법이어야 해서 target 정의가 단계마다 달라진다.
대신 **각 pass가 자기 출력 불변식을 스스로 검사한다** — 3번 끝에서 메모리 잔여물, 6번 끝에서
계산 잔여물.

```cpp
// 3번 끝, erase loop 직후. 하나라도 남으면 뒤의 배치 pass가 그 버퍼를 보지 못한 채
// 조용히 부분 배치를 만든다.
func.walk([&](Operation *op) {
  StringRef d = op->getDialect()->getNamespace();
  if (d == "memref" || d == "bufferization") residual.push_back(op);
});
```

보조 규칙: `MemoryEffectOpInterface`를 갖고 memref 피연산자에 Read/Write 효과가 있는데
`pim` dialect가 아닌 op도 잔여물로 본다.

---

# 9. 삭제 / 유지 / 신규

## 9.1 삭제

| 대상 | 위치 | 줄 수 |
|---|---|---|
| `pim.copy` op + verifier | `PIMDialect.td:95-101` | 7 |
| raise의 죽은 walk 4종 (row-gather+plain load, scalar store, gather ×2, scatter ×2) | `TTSharedToPIMPass.cpp:233-457` | 190 |
| `--pim-remove-redundant-transfers` 전체 + ODS | 파일 전체 | **188** |
| `--pim-classify-compute` 전체 + ODS | 파일 전체 | **79** |
| `--pim-analyze-transfer` 전체 + ODS (약 70줄은 헬퍼로 이식) | 파일 전체 | **206** |
| combine 승격 기구 | `PIMAnalyzeReusePass.cpp` | ~120 |
| `--pim-partition`의 `splits` 경로 | `PIMPartitionPass.cpp:150-217` + ODS | ~60 |
| `PIMOptions.split_m/n/k` | `compiler.py:236-238, 142` | 4 |
| `pim.execution_model` 생산자 (소비자 0) | `PIMAnalyzeReusePass.cpp:207` | 1 |
| | **합계** | **~855** |

**후속 삭제** (세 커널 통과 확인 뒤): combine을 만들지 않으므로
`unit_data.combine/combine_over` 경로 전체가 죽는다 — 약 120줄. **단 Phase 2에서 AttAcc의
group-level accumulator가 이 자리를 다시 쓸 수 있으니 커밋을 분리해 둔다.**

`--pim-normalize-accumulator`(126줄)와 `--pim-build-grid`(131줄)는 삭제가 아니라 **이관**이다.

**순서 주의.** raise walk 삭제를 세 커널이 초록이 되기 전에 하면 안 된다. 지금 `.pimobj`까지
가는 유일한 커널들(G00/G01/G02/G05/G07)이 그 walk를 쓰는지 확인되지 않았다.

## 9.2 유지 — target 계약이라 끊으면 조용히 틀리는 것

| 대상 | 끊었을 때 |
|---|---|
| `scf.for`의 `pim.grid_dim` (문자열) | `collectAxes`가 `StringAttr` 비교로만 축을 묶는다(`PIMToUPMEMPass.cpp:294-296`). 실패하면 **위치 기반 fallback**(`:299-303`)이 발화한다 |
| `arith.addi`의 `pim.unit_origin` | `:206-212`가 `if (a.unitOrigin)` 가드 안에서만 LHS를 0으로 바꾼다. **없으면 진단 없이** device 주소가 전역 좌표로 남는다 |
| `pim.access_id` | `:154-156`, `:613-614`가 null 검사 없이 역참조 |
| `pim.invariant_grid_dims` → `replicated_over` | 끊기면 `distinctCount`가 device 수를 그대로 곱해 host staging이 폭증한다 |
| `narrowLoop`의 클램프 | §6.1의 `size` 계산이 이것을 그대로 쓴다 |

**`--pim-legacy-unit-form`이 반드시 내야 하는 것**: 위 다섯 개 전부. shim 끝에 세
가지(`grid_dim`, `unit_origin`, `access_id`)의 존재를 확인하는 사후 조건 검사를 둔다.

## 9.3 신규

| 대상 | 형태 | 분량 |
|---|---|---|
| `#pim.target_profile` 외 attribute 5종 | 신규 `PIMAttrs.td` | ~110 |
| `pim.device`, `pim.place` | op | ~90 |
| `--pim-normalize-address` | pass | ~90 |
| `--pim-build-logical-grid` | pass | ~180 |
| `--pim-recover-access-origin` | pass | ~110 |
| `--pim-form-compute` (`normalize-accumulator` 흡수) | pass | ~140 |
| `--pim-form-execution-waves` | pass | ~50 (UPMEM 항등) |
| `--pim-map-devices` | pass | ~110 |
| `--pim-map-cores` | pass | ~70 |
| `--pim-place-data` | pass | ~150 |
| `--pim-verify` | pass | ~170 |
| `--pim-legacy-unit-form` (한시) | pass | ~180 |
| mask/extent 가드 (`stage`/`unstage` + emitter 5곳) | 수정 | ~120 |

**가장 큰 덩어리는 `--pim-build-logical-grid`이고 줄일 수 있다.** `grouped_gemm`의 13-op
`GROUP_M` 매처가 그 절반이며 파이프라인 전체에서 가장 부서지기 쉬운 template이다. §3.4에서 본
대로 그 순열은 **배치에 아무 영향이 없다.** `grouped_gemm.py`를
`program_id(0)`/`program_id(1)` 2차원 grid로 다시 쓰면 의미가 동일하고 약 180줄이 70줄이 된다.

---

# 10. 구현 순서와 검증

빌드·실행은 직접 진행하신다.

| # | 작업 | 검증 |
|---|---|---|
| 0 | lane 예산 모순 확인 (§2.10) | `local-lanes=16`이 통과하는지 |
| 1 | `--pim-normalize-address` 신설 | 세 `.ttsharedir`에 이 pass만 적용 → offset이 block 인자가 아니다 |
| 2 | raise 죽은 walk 삭제 + 미변환 진단 | `.pim_common`이 이전과 동일 |
| 3 | `--pim-build-logical-grid` + `--pim-recover-access-origin` | 커널마다 `origin[` 개수가 3 (지금 gemm은 1) |
| 4 | `#pim.target_profile` + `--pim-map-devices` | 옵션 다섯 개가 profile 하나로. `splits` 옵션 제거 |
| 5 | `pim.place` + `--pim-place-data` + `--pim-verify` | `gemm.py` end-to-end, 수치 일치 |
| 6 | mask/extent 가드 | **`G05_masked_boundary`가 `M=N=K=72`에서 수치 일치** (지금 조용히 틀린다) |
| 7 | `--pim-legacy-unit-form` | `grouped_gemm.py`가 crash 대신 진단 또는 통과 |
| 8 | `base_offset` + k device partition 1 | `split_k.py` end-to-end, `c.sum(dim=0)` 일치 |
| 9 | `--pim-map-cores`의 lane 상한을 `allocFor` 집합으로 | `lanes>1`에서 `gemm.py` |
| 10 | `--pim-form-execution-waves` + core 좌표 | AiMX가 붙을 때. 그전까지는 항등 |

**1번이 없으면 3번 이후가 전부 검증 불가다.** `PIMBackend.hash()`가 `GPUTarget`만
돌려주므로(`compiler.py:308-310`) C++ pass를 고쳐도 Triton 컴파일 캐시가 무효화되지 않는다 —
시작 전에 캐시 키에 무언가를 하나 넣어야 한다. **9번까지 wave·core mapping이 항등이므로 UPMEM
경로만으로 검증된다.**

```bash
cd /home/dlrkdals/TritonPIM/triton_shared
OPT=<build-dir>/bin/triton-shared-opt
DUMP=/home/dlrkdals/TritonPIM/dump

# 단계 0 — lane 예산 모순
$OPT --convert-pim-to-upmem="unit-memory-bytes=33554432 local-lanes=16 local-memory-bytes=49152" \
     $DUMP/TNQUJN5*/gemm_kernel.pim_common 2>&1 | grep -i "fast memory\|budget"

# 단계 1 — 주소 정규화만
$OPT --pim-normalize-address $DUMP/TNQUJN5*/gemm_kernel.ttsharedir \
  | grep 'reinterpret_cast %arg0'
#   → offset: [%...] 이 iter_arg(%argNN)가 아니어야 한다

# 단계 3 — origin 3/3
for k in gemm_kernel matmul_split_k_kernel matmul_grouped_kernel; do
  $OPT --pim-normalize-address --convert-ttshared-to-pim \
       --pim-build-logical-grid --pim-recover-access-origin \
       $DUMP/*/$k.ttsharedir | grep -c 'origin\['
done
#   → 각각 3

# 단계 5 — end-to-end
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=$DUMP TRITON_ALWAYS_COMPILE=1 \
  python /home/dlrkdals/TritonPIM/workspace/GEMM/gemm.py
#   → .pimobj 까지, 그리고 수치 일치
#   → gemm_kernel.pim_common 에 pim.place 가 3개 (지금 pim.unit_data 는 1개)
```

---

# 11. 지금 만들지 않는 것

| 항목 | 왜 | 확장 지점 |
|---|---|---|
| 비용 모델 / 후보 탐색 | tile 개수가 런타임이라 컴파일 시점에 순위를 매길 수 없다 | 반복 횟수와 반복 1회분을 나눠 emit하고 곱하는 쪽을 host에 두면 열린다 |
| 일반 cyclic placement | Phase 1은 k device partition 1로 피한다 (§4.5) | `pim.place`에 축별 `period`를 더한다. **한 규약만 쓴다** — `unit_origin` LHS를 0으로 만들면서 동시에 partition index를 빼면 이중 차감이 되어 조용히 틀린다 |
| PIM Group region op / group mapping pass | 두 target 모두 device당 1개다 | device당 group이 2개 이상인 target이 붙을 때 (§7.2) |
| physical memory topology 필드 | UPMEM·AiMX가 쓰지 않는다 | Samsung·AttAcc를 붙일 때 target 확장에 넣는다 (§2.4) |
| nonzero pad | 세 커널이 전부 `other=0`이다 | `calloc` 기구로 안 나오므로 별도 채우기가 필요하다 (§6.3) |
| AttAcc full attention | reduction·softmax·group accumulator·barrier가 필요하다 | Phase 2. `pim.contract` 도입 시점과 맞물린다 |
| `pim.co_locate` | 세 커널에서 유도된다 (§7.2) | fused attention에서 K와 V가 서로 다른 contraction의 피연산자가 될 때 |
| PyTorch 그래프 계층 | 코드 0줄이고 런타임·emitter가 데이터 비상주로 설계돼 있다 | `driver.py:174`, `pim_runtime.c:6-14`, `UPMEMToHost.cpp:179-181` |

---

# 12. 채워야 하는 값과 결정이 필요한 항목

## 12.1 값이 비어 있는 칸

1. **AiMX의 `accumulator_register_capacity_bytes`** — **가장 먼저 필요하다.** `BLOCK_M = 16`은
   PIM Core 하나가 `16 × fp32 = 64 byte`의 부분합을 들 수 있어야 성립한다(§6.6). vecA와 vecB
   용량만으로는 증명할 수 없다.
2. **AiMX의 `usable_bank_capacity_bytes`** — 미측정.
3. **vecA / vecB 중 어느 쪽이 A(=`[M,K]`)이고 어느 쪽이 B(=`[K,N]`)인가** — §6.6은 vecA가
   A-side streamed operand, vecB가 B-side bank-private operand라고 놓았다. 반대면 `BLOCK_M`과
   `BLOCK_N`의 역할이 바뀐다. `BLOCK_K = 16`(vecB 용량)은 그래도 유지된다.
4. **UPMEM의 `number_of_devices_per_transfer_group = 64`** — 저장소 근거가 아니라 표준 rank
   크기다. target 확장의 최적화 파라미터라 틀려도 정확성에 영향이 없다.
5. **UPMEM의 런타임·스택 여유분** — `usable_*`와 물리값(64 MB / 64 KB)의 차이가 지금
   32 MiB / 48 KiB로 고정돼 있다. lane 수에 비례해 커진다.

## 12.2 결정

1. **lane 16과 `BLOCK 32/32/16`이 WRAM 안에 드는가** (§2.10). 실측 기록과 예산 검사식이
   어긋난다. 단계 0의 명령으로 먼저 확인해야 한다. 능력값/선택값 분리로 어느 쪽이든 안전하다.
2. **빈 partition의 규약** (§8.1). `start`를 `extent`로 클램프할지, verifier가 `size > 0`
   조건부로 검사할지. 둘 중 하나를 정해야 placement pass와 verifier가 같은 규약을 쓴다.
3. **`pim.contract` / `pim.elementwise`를 지금 도입하는가** (§7.3). 권고는 "Phase 2에서 함께".
4. **`grouped_gemm.py`를 2차원 grid로 다시 쓸 수 있는가** (§9.3). 가능하면 가장 취약한 매처가
   사라지고 `--pim-build-logical-grid`가 180줄에서 70줄로 준다. 배치 결과는 동일하다.
5. **`split_k.py`의 BLOCK을 32/32/16으로 낮출 수 있는가** (§4.5). 선언 기본값 `128/128/64`는
   동시 상주량 81 920 byte로 WRAM 물리 64 KB를 넘어 **어떤 매핑으로도** UPMEM에서 실행되지
   않는다.
6. **AiMX용 커널의 dtype** (§2.5). AiMX는 bf16 입력 / fp32 누산인데 세 참조 커널은 전부
   int8 × int8이다.

---

# 13. 구현에서 확정한 계약

본문과 어긋나는 자리에서는 이 절이 기준이다. 각 항목은 구현된 pass·op과 일대일로 대응한다.

## 13.1 pass 순서 — core mapping이 wave 구성보다 앞이다

execution wave를 만들려면 `logical_tile_core_footprint`가 있어야 하는데 그것을 만드는 것이
`--pim-map-cores`다. 그래서 순서가 바뀐다.

```
 1  --pim-normalize-address
 2  -cse -loop-invariant-code-motion -canonicalize
 3  --convert-ttshared-to-pim
 4  --pim-build-logical-grid
 5  --pim-recover-access-origin
 6  --pim-form-compute
 7  --pim-set-target-profile        ← profile을 module에 붙인다
 8  --pim-map-cores                 ← core_axis_roles / tile_footprint / lanes
 9  --pim-form-execution-waves      ← footprint로 wave 구성
10  --pim-map-devices
11  --pim-place-data
12  --pim-verify
13  -cse -canonicalize
14  --convert-pim-to-upmem
```

`#pim.target_profile`은 ⑦이 붙이고, ⑧부터가 그것을 읽는다. profile 값은 `PIMOptions`에서
pass 옵션으로 들어오며, target마다 고정된 값(core shape, lane 성격, uniform 여부)은
`compiler.py`의 `_TARGET_PROFILE`에 있다. **⑭가 legality gate에서 읽고 나면 profile과
`#pim.core_mapping`을 떼어 낸다** — target IR은 UPMEM만 담아야 하고, 남겨 두면 뒤 단계가
기계 값을 다시 읽을 수 있다는 오해를 만든다.

lane 수 선택은 `--pim-map-cores`가 함께 한다. 별도 pass로 나누지 않은 이유는 그것이
`#pim.core_mapping` 하나를 만드는 일의 일부이고, 나누면 attribute 하나에 producer가 둘이
되기 때문이다.

**실제로 쓰는 lane 수는 번들에 실려 launcher로 간다.** `--upmem-to-device`가 device 소스
첫머리에 `// pim.lanes = N`을 적고 `compiler.py`가 그것을 읽어 번들에 넣으며 `driver.py`가
`-DNR_TASKLETS`로 쓴다. 요청값과 선택값이 다를 수 있으므로 같은 값이 IR과 device 빌드 양쪽에
가야 한다.

## 13.2 wave·core 좌표는 단계 1에서 materialize하지 않는다

UPMEM은 `pim_core_shape = [1]`이라 wave 하나가 logical tile 하나이고 core 좌표가 없다.
그래서 `--pim-map-cores`와 `--pim-form-execution-waves`가 좌표를 만들지 않는다. 대신
표현할 수 없는 구성에서 **진단을 낸다** — PIM Core가 여럿인 topology, wave당 logical tile이
둘 이상인 footprint가 그것이다.

좌표를 만들게 되면 `pim.device` region 안의 `scf.for`에 `pim.axis`로 이름을 달고
`pim.place`가 그 유도변수를 직접 쓴다. `pim.axis`는 이미 그 자리를 그렇게 쓰고 있다 —
device partition 좌표는 `pim.device`의 block argument이고, 그 밖의 좌표는 전부 `pim.axis`
loop의 유도변수다. `#pim.place_mapping`은 검증·진단 전용이고 target lowering은 읽지 않는다.

## 13.3 `pim.place`에 `extent`와 `base_offset`을 두지 않는다

**`size`가 이미 클램프된 값이다.** `--pim-place-data`가

```
size[d] = mask 있음 ? clamp(extent[d] - start[d], 0, tile 수 × tile[d])
                    : tile 수 × tile[d]
```

로 낸다. `extent[d]`는 `pim.access.valid[d]`의 식에서 되짚는다 — triton-shared의 mask 생성이
내는 두 꼴 `min(max(min(start + tile, extent), start) - start, tile)`과
`min(max(extent - start, 0), tile)`에서, 좌표에 무관하고 상수도 아닌 쪽이 크기다. 되짚지
못하면 진단을 낸다.

**"마지막 tile의 valid를 더한다"로는 안 된다.** grid가 실제 크기보다 넓게 잡히면
(`split_k.py`에서 `BLOCK_K × SPLIT_K > K`인 경우) 마지막 tile이 통째로 무효라 그 식이
원본 버퍼 밖을 가리킨다. 크기를 잘라 두어야 host staging의 원소 가드가 실제 경계를 지킨다.

되짚은 크기는 `size`를 만들 때 소비되고 op에는 남지 않는다. `start + size`가 언제나 원본
버퍼 안이므로 target lowering이 크기를 다시 볼 일이 없다.

`base_offset`도 두지 않는다. split-K의 출력면은 **placement 축**으로 표현한다(§13.4).

## 13.4 split-K의 출력면은 버퍼 축이다

`--pim-recover-access-origin`은 offset의 각 항을 좌표 쪽과 간격 쪽으로 가르고, 간격 쪽을
`index_cast`와 상수 곱을 벗겨 낸 기준 값으로 access의 strides와 대조한다. 어느 축과도 맞지
않는 간격이면 **크기 1의 축을 앞에 새로 만든다.**

`split_k.py`의 C가 그 경우다. `reinterpret_cast`는 rank 2이고 strides에 `stride_ck`가 없는데
offset에는 `pid_k · stride_ck` 항이 있다. 그래서 access가 rank 3이 된다.

```
sizes   [1, BLOCK_M, BLOCK_N]
strides [stride_ck, stride_cm, 1]
origin  [explicit_k_partition, output_row, output_column]
```

`pim.place`도 같은 rank가 되어 16개 출력면의 상주량·transfer layout·면 사이 간격이 전부
표현된다. `upmem.stage`/`unstage`/`gather`가 rank에 무관하므로 host 경로는 그대로다.
`buildDMA`는 tile shape를 WRAM 버퍼가 아니라 `pim.access.sizes`에서 가져온다 — 앞쪽 축의
크기가 1이면 그 축은 MRAM 안의 자리 계산에만 나타나고 WRAM 버퍼는 rank 2로 남는다.

`SPLIT_K`를 컴파일 시점 상수로 복원할 필요도 없다. explicit K 분할 축의 tile 개수는 그 축의
logical grid loop 상한, 즉 `num_programs`이고 런타임 값이어도 배치가 성립한다.

## 13.5 경계 tile — 올림 나눗셈과 원소 가드

host staging loop는 두 가지를 함께 해야 한다.

```
tile loop 상한   ceildiv(size[d], tile[d])      ← 부분 tile도 돈다
원소 가드        _t·tile[d] + _e < size[d]      ← 부분 tile 안의 범위 밖 원소를 건너뛴다
```

`calloc`이 건너뛴 자리에 0을 공급하므로 `pad = 0`은 여기서 공짜로 나온다. 0이 아닌 pad는
`--pim-verify`가 진단한다.

## 13.6 용량 검사는 공통과 target으로 나눈다

| 자리 | 검사 |
|---|---|
| `--pim-verify` | rank·id 일대일·nonnegative size·`size ≤ capacity`(정적일 때)·clamp 식 존재·lane 예산·footprint 나눗셈 |
| `--convert-pim-to-upmem` 진입 | target·lane 성격·bank 용량 기록 여부·**버퍼 하나의 상주량 ≤ bank 예산 / 버퍼 개수**(정적일 때) |
| 생성된 host 코드 | 배정량이 런타임 값이면 `dpu_push_xfer` 직전에 MRAM 심볼 크기와 대조하고 넘치면 즉시 중단 |

`usable_bank_capacity_bytes / 버퍼 개수`는 UPMEM lowering policy라 공통 verifier에 두지 않는다.

## 13.7 combine을 만들지 않으므로 결합 경로를 지웠다

`pim.place`에 `combine`·`combine_over`가 없고, `upmem.foreach_group`의
`reduce_axis`/`reduce_count`/`combine`도 없앴다. 그룹은 device 하나의 출력 버퍼를 원본으로
되돌리는 단위일 뿐이다. Phase 2에서 AttAcc의 group-level accumulator가 필요해지면 그때
되살린다.

## 13.8 전이용 유닛 형태를 두지 않는다

`pim.units`/`pim.unit_data`와 `--pim-legacy-unit-form`을 만들지 않는다.
`pim.device`+`pim.place`가 `pim.units`+`pim.unit_data`와 같은 모양이라
`--convert-pim-to-upmem`이 직접 소비하는 편이 shim보다 짧고, shim에는 `extent`와 새 축을
실을 자리가 없다. 함께 정리한 것:

- 삭제한 op: `pim.copy`, `pim.gather`, `pim.scatter`, `pim.units`, `pim.unit_data`,
  그리고 `pim.access`의 `gather_dim`/`index` clause
- 삭제한 pass: `--pim-build-grid`, `--pim-remove-redundant-transfers`,
  `--pim-normalize-accumulator`(→ `--pim-form-compute`), `--pim-analyze-reuse`,
  `--pim-analyze-transfer`, `--pim-partition`, `--pim-classify-compute`
- 삭제한 raise walk: row-gather, 원소별 gather/scatter, 스칼라 저장
- 삭제한 옵션: `PIMOptions.split_m/n/k`, `--convert-pim-to-upmem`의
  `unit-memory-bytes`/`local-lanes`/`local-memory-bytes`(전부 profile에서 읽는다)

`pim.access_id`는 discardable attribute가 아니라 `pim.access`의 필수 `id` 필드가 되었고,
`pim.grid_dim`은 `pim.axis`로, `pim.unit_origin`은 `pim.device_origin`으로 이름이 바뀌었다.

`--convert-ttshared-to-pim`은 끝에서 남은 `memref`/`bufferization` op을 진단한다. 지원하지
않는 접근 패턴이 조용히 통과해 부분 배치가 만들어지는 것을 막는다.

## 13.9 device 분배는 컴파일 시점 static heuristic이다

`--pim-map-devices`가 하는 일은 정확히 두 가지다.

1. **compile-time static partition heuristic** — explicit K 분할 축은 1로 두고, 남은 device
   수를 정사각에 가깝게 인수분해한다. 동률이면 column partition 수가 transfer group 크기의
   배수인 쪽을 고른다. tile 개수가 런타임이라 컴파일 시점에는 이것 말고 고를 근거가 없다.
2. **balanced block** — 선택된 partition 안에서 런타임 tile을 균형 분배한다.

tile 모양을 보고 partition 수를 고르는 것은 지원하지 않는다. 그러려면 `M`·`N`을 컴파일
specialization 값으로 만들어 캐시 키에 넣거나, host runtime이 partition shape를 고르거나,
여러 static 후보를 컴파일해 런타임에 고르는 것 중 하나가 필요하다. 같은 이유로 **놀고 있는
device를 컴파일 시점 remark로 보고하지 않는다** — 그 수는 런타임에만 정해진다.

## 13.10 device가 필요로 하는 런타임 스칼라는 host 변수로 간다

device 몸체가 host func 안의 값을 가리키면 `--convert-pim-to-upmem`이 그 자리에 다시 만든다.
상수처럼 뿌리까지 값이 이어지면 복제하고, **커널 인자에서 나오는 수는 device 안에서 만들 수
없으므로 `upmem.host_var` 통로를 낸다.** reduction loop의 반복 횟수 `ceildiv(K, BLOCK_K)`가
그것이다 — 축 loop의 반복 횟수와 같은 통로를 쓰되, 축이 아니므로 이름이 `_host_N`이다.

`--pim-place-data`가 loop 상한을 device region 밖에 다시 만들고 `-cse`가 원본을 그것으로
합치므로, reduction loop의 상한은 언제나 host 쪽에 놓인다. 그래서 이 통로가 없으면
`materializeExternals`가 그 자리에서 실패한다.

전송은 `upmem.to_host_var`(DPU마다)로 한다. `upmem.broadcast`가 아닌 이유는 그 값이
`upmem.foreach_unit` 몸체 안에서 계산되어 집합 단위 전송이 지배 관계를 만족하지 못하기
때문이다.

## 13.11 단계별 지원 범위

```
Phase 1 구현       UPMEM GEMM (gemm.py / split_k.py / grouped_gemm.py)
표현 가능성 평가    AiMX GEMM, Samsung 실행 모드, AttAcc GEMV
Phase 2 구현       AiMX target lowering, Samsung 실행 모드, AttAcc GEMV와 attention
```

3계층으로 Samsung과 AttAcc를 **표현할 수 있다는 것**과 지금 **lowering할 수 있다는 것**은
다르다. Phase 1에 실제로 코드가 있는 target은 UPMEM 하나다.

AiMX의 vecA/vecB operand binding, `BLOCK_M`/`BLOCK_N`/`BLOCK_K`의 capacity 유도, A-side
broadcast 방향, B-side bank-private 배치는 **전부 미확정**이다. vecB의 16개 원소가 어느
logical 차원에 대응하는지는 instruction dataflow가 정하므로, vecB가 reduction vector를
들 때에만 32 byte 용량이 `BLOCK_K = 16`의 근거가 된다. 수정된 AiMX의 descriptor 또는
register-transfer-level 명세로 확인해야 한다.
