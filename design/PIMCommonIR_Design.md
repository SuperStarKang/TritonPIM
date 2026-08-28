# PIM Common IR 설계 및 구현 계약

이 문서는 Triton Shared IR과 target IR 사이에 위치하는 PIM Common IR의 현재
계약을 정의한다. UPMEM, AiMX 및 이후 추가되는 PIM target은 이 IR을 다시 계획하지
않고 그대로 해석하여 자신의 target IR을 만들어야 한다.

현재 계약은 GEMM, GEMV와 제한된 reduction-free elementwise lowering에 대해
구현되어 있으며, `test/Conversion/TTSharedToPIM`의 19개 테스트가 통과한다.

---

## 1. 컴파일 경계

```text
TTIR
  -> Triton Shared IR
  -> PIM Common IR
  -> Target IR (UPMEM IR, AiMX IR, ...)
  -> target code generation
```

각 계층의 책임은 다음과 같다.

| 계층 | 결정하는 것 |
| --- | --- |
| Triton Shared IR | 연산 의미, tensor/access 패턴, Triton Block |
| PIM Common IR | Block–Block Group–Execution Tile, device 배치, 데이터 영역 |
| Target IR | 물리 실행 눈금, padding, target 전송 명령, reduction 결합 위치 |

PIM Common IR에는 DPU, tasklet, MRAM, FPGA channel, descriptor, vACC 같은
target 용어가 나타나지 않는다. 반대로 target lowering은 Common IR이 결정한
Block Group 수, Execution Tile 수 또는 device partition을 조용히 변경하면 안 된다.
target 제약에 맞지 않으면 target 고유 진단을 내야 한다.

---

## 2. 3계층 구조

모든 shape는 고정 순서 `[m, n, k]`를 사용한다. 존재하지 않는 축의 원소 크기와
Execution Tile Group 수는 0이다.

### 2.1 Block

Triton program instance 하나가 처리하는 유효 원소 크기다.

```mlir
#pim.block<shape = [block_m, block_n, block_k]>
```

### 2.2 Block Group

인접한 Block을 각 축에서 몇 개 묶는지 나타낸다. 하나의 Block Group이 한 PIM
Device 내부에서 bank, core, lane이 협력하여 실행하는 compute 단위다.

```mlir
#pim.block_group<blocks = [group_m, group_n, group_k]>
```

Block Group 원소 크기는 별도 attribute에 중복 저장하지 않는다.

```text
block_group_shape = block.shape * block_group.blocks
```

### 2.3 Execution Tile

한 PIM Device가 한 번의 resident 실행에서 담당하는 Block Group grid다. compute뿐
아니라 transfer와 residency의 단위다.

```mlir
#pim.execution_tile<groups = [exec_m, exec_n, exec_k]>
```

Execution Tile 원소 크기도 파생값이다.

```text
execution_tile_shape =
    block.shape * block_group.blocks * execution_tile.groups
```

따라서 Common IR에서 독립적으로 선택하는 계층 변수는 Block 원소 크기, Group의
Block 개수, Execution Tile의 Group 개수뿐이다. `useful`, `capacity_limit` 같은
중복 shape attribute는 두지 않는다.

---

## 3. 축과 reduction

### 3.1 M/N

`#pim.axis<"m">`과 `#pim.axis<"n">`은 출력 parallel 축이다. 이 loop의
induction variable은 Block Group 좌표다.

### 3.2 temporal K

일반 GEMM/GEMV의 K loop는 `pim.axis`가 없는 device-local reduction loop다.
`execution_tile.groups[k]`는 하나의 spatial output plane 안에서 resident하게
처리할 temporal K Block Group 수를 뜻한다.

### 3.3 spatial split-K

명시적인 `#pim.axis<"k">`는 temporal reduction loop가 아니라 spatial split-K
Execution Tile 좌표다. 서로 다른 좌표는 같은 M/N 출력 영역에 대한 별도 partial을
만든다.

```text
pim.axis="k"             = spatial partial plane
execution_tile.groups[k] = plane 내부 temporal K residency
```

partial을 host에서 합칠지 device에서 합칠지는 target 특성이다. Common IR은 spatial
K 좌표와 temporal K residency를 제공하고, target lowering이 결합 방식을 선택한다.

---

## 4. Target profile과 intra-device plan

### 4.1 Target profile

module에는 target 공통 topology와 사용 가능한 용량만 기록한다.

```mlir
#pim.target_profile<
  target = "...",
  number_of_pim_devices = ...,
  number_of_banks_per_pim_device = ...,
  number_of_pim_cores_per_bank = ...,
  number_of_parallel_lanes_per_pim_core = ...,
  parallel_execution_kind = "multithread" | "simd",
  usable_bank_capacity_bytes = ...,
  usable_local_capacity_bytes = ...
>
```

homogeneous device 전체의 usable capacity는 다음과 같이 파생한다.

```text
usable_device_capacity_bytes =
    number_of_banks_per_pim_device * usable_bank_capacity_bytes
```

### 4.2 Intra-device plan

하나의 Block Group을 device 내부에서 나누는 방법이다.

```mlir
#pim.intra_device_plan<
  bank_partitions = [Pm, Pn, Pk],
  core_partitions = [Cm, Cn, Ck],
  lanes = L,
  lane_axis = "m" | "n" | "k"
>
```

```text
Block Group
  -> bank_partitions로 bank subtile 분할
     -> core_partitions로 core subtile 분할
        -> lane_axis에서 lanes개로 분할
```

현재 verifier는 topology 개수와 local storage를 검사한다. intra-device K 분할은
bank/core/lane partial 결합 계약이 아직 없으므로 허용하지 않는다.

---

## 5. PIM dialect 구성

### 5.1 계층과 배치 attribute

| attribute | 역할 |
| --- | --- |
| `#pim.target_profile` | 공통 topology와 usable capacity |
| `#pim.block` | Triton Block 유효 원소 shape |
| `#pim.block_group` | Block Group당 Block 개수 |
| `#pim.execution_tile` | Execution Tile당 Group 개수 |
| `#pim.intra_device_plan` | Group 내부 bank/core/lane 분할 |
| `#pim.device_mapping` | 동시에 사용하는 device grid |
| `#pim.place_mapping` | buffer 축과 PIM 축의 대응 |

### 5.2 연산

| op | 역할 |
| --- | --- |
| `pim.access` | host buffer의 strided tile 주소 명세 |
| `pim.load` | access를 통해 local tensor를 읽음 |
| `pim.store` | local tensor를 access로 저장 |
| `pim.device` | PIM Device 하나가 실행하는 region |
| `pim.place` | device에 상주하거나 전송할 최종 buffer 영역 |

compute는 target-neutral `linalg` op로 유지한다. 현재 주요 compute op는
`linalg.matmul`, `linalg.matvec`, identity-mapped `linalg.generic`이다.

---

## 6. Common lowering pipeline

현재 backend가 사용하는 pass 순서는 다음과 같다.

```text
pim-normalize-address
convert-ttshared-to-pim
pim-build-logical-grid
pim-recover-access-origin
pim-form-compute
pim-set-target-profile
pim-form-block-groups
pim-plan-intra-device
pim-form-execution-tiles
pim-map-devices
pim-place-data
pim-verify
```

### 6.1 Block과 Block Group 형성

`pim-form-block-groups`는 실제 `pim.access`와 local tensor shape에서 Triton Block
크기를 읽고, 요청된 Block 개수만큼 loop/access/tensor를 묶는다. 결과로 func에
`pim.block`, `pim.block_group`을 기록한다.

옵션 0은 자동 비용 탐색이 아니라 가장 작은 항등 Group, 즉 Block 하나를 뜻한다.

### 6.2 Execution Tile 형성

`pim-form-execution-tiles`는 존재하는 축에서 옵션으로 받은 Group 수를 기록한다.
옵션 0은 존재하는 축에서 Group 하나, 존재하지 않는 축에서 0이다. PIM DRAM을
무조건 채우거나 작은 문제를 DRAM 크기까지 padding하지 않는다.

### 6.3 Device mapping

`pim-map-devices`는 전체 Block Group grid를 Execution Tile grid로 자르고 여러 PIM
Device에 배치한다.

```text
execution_tile_index = round * active_devices + device_coordinate
first_group          = execution_tile_index * execution_tile.groups
local_groups         = min(execution_tile.groups, remaining_groups)
```

M/N 기본 scheduling granularity는 `execution_tile.groups[m/n]`를 직접 사용한다.
`pim.access`에서 같은 크기를 다시 추론하지 않는다. spatial `pim.axis="k"`는 loop
한 회전이 이미 Execution Tile 하나이므로 granularity가 1이다.

Execution Tile 수가 active device 수보다 많으면 동일한 physical device set을
scheduling round에서 반복 사용한다. 마지막 round에 Tile이 없는 device는 0회 loop를
실행한다.

### 6.4 Placement

`pim-place-data`는 access마다 `pim.place`를 만든다.

```text
start      = 이 device가 맡는 첫 원소
size       = tail을 반영한 실제 유효 영역
capacity   = Execution Tile의 고정 resident 영역
strides    = 원본 host buffer의 stride
direction  = in | out
replicated_over = 같은 데이터를 공유하는 device 축
```

M/N capacity도 `execution_tile.groups[m/n]`를 직접 사용한다. round가 바뀌어도 같은
physical address의 buffer를 재사용하고 실제 데이터만 바꿀 수 있다.

### 6.5 검증

`pim-verify`는 planning metadata와 실제 compute/access가 일치하는지 검사한다.

```text
Block.shape * BlockGroup.blocks == 실제 device-local Group shape
ExecutionTile.groups[axis] > 0  <=> 해당 compute 축이 존재
product(device partitions) <= available devices
product(bank/core partitions)와 lanes가 topology 범위 안
local Group buffer가 usable local capacity 안
access와 place가 id로 일대일 대응
size <= capacity 또는 runtime clamp가 존재
```

계층 크기의 최초 생성과 최종 검증만 access를 분석한다. scheduling과 placement는 이미
materialize된 계층 metadata를 단일 기준으로 사용한다.

---

## 7. 연산별 매핑

### 7.1 GEMM

```text
Block          = [Bm, Bn, Bk]
Block Group    = [Gm, Gn, Gk]
Execution Tile = [Em, En, Ek]
```

M/N은 output parallel 축이고 K는 device-local temporal reduction이다. explicit
split-K 커널만 별도 `pim.axis="k"`를 갖는다.

### 7.2 GEMV

```text
Block          = [Bm, 0, Bk]
Block Group    = [Gm, 1, Gk]
Execution Tile = [Em, 0, Ek]
```

Triton이 만드는 broadcast → multiply → transpose/reduce 패턴은
`pim-form-compute`에서 `linalg.matvec`로 정규화된다. M은 output parallel 축,
K는 temporal reduction이다.

### 7.3 Elementwise

```text
1-D: Block=[Bm,0,0], Group=[Gm,1,1], Execution=[Em,0,0]
2-D: Block=[Bm,Bn,0], Group=[Gm,Gn,1], Execution=[Em,En,0]
```

현재 지원 범위는 reduction이 없고 모든 iterator가 parallel인 identity-mapped
`linalg.generic`이다. 모든 operand는 같은 static shape를 사용한다. 하나의 scalar
region 내부에는 add/sub/mul 같은 여러 elementwise 연산을 fusion할 수 있다.

---

## 8. GEMM 예시

다음을 가정한다.

```text
C[1024, 2048] = A[1024, 4096] * B[4096, 2048]
Block          = [32, 32, 32]
Block Group    = [2, 4, 2]
Execution Tile = [2, 3, 4]
device mapping = [m:2, n:4]
```

파생 shape는 다음과 같다.

```text
Block Group shape    = [64, 128, 64]
Execution Tile shape = [128, 384, 256]
```

전체 문제의 Group 수와 M/N Execution Tile 수는 다음과 같다.

```text
Groups       = [1024/64, 2048/128, 4096/64] = [16,16,64]
M tiles      = ceil(16/2) = 8
N tiles      = ceil(16/3) = 6
M rounds     = ceil(8/2 devices) = 4
N rounds     = ceil(6/4 devices) = 2
```

개념적인 Common IR은 다음과 같다.

```mlir
func.func @gemm(...) attributes {
  pim.block = #pim.block<shape = [32, 32, 32]>,
  pim.block_group = #pim.block_group<blocks = [2, 4, 2]>,
  pim.intra_device_plan = #pim.intra_device_plan<...>
} {
  scf.for %round_n = ... {
    scf.for %round_m = ... {
      pim.device <axes = ["m", "n"], partitions = [2, 4]>
          extents[...] attributes {
            pim.execution_tile =
                #pim.execution_tile<groups = [2, 3, 4]>
          } {
      ^bb0(%device_m: index, %device_n: index):
        pim.place ...
        scf.for %group_m = 0 to min(2, %remaining_m) {
          scf.for %group_n = 0 to min(3, %remaining_n) {
            scf.for %reduction_group = ... {
              %a = pim.load ...
              %b = pim.load ...
              %c = linalg.matmul ...
              pim.store %c, ...
            }
          }
        }
      }
    }
  }
}
```

하나의 device coordinate는 한 round에서 완성된 Execution Tile 하나를 담당한다.
Block Group은 그 device 내부의 동일한 intra-device plan으로 반복 실행된다.

---

## 9. Target lowering 입력 계약

PIM Common IR만으로 target lowering은 다음 정보를 얻을 수 있다.

| 필요한 정보 | Common IR 위치 |
| --- | --- |
| 물리 device 수와 topology | `pim.target_profile` |
| Triton Block shape | `pim.block` |
| device 내부 compute 단위 | `pim.block_group` |
| resident transfer/compute 범위 | `pim.execution_tile` |
| bank/core/lane 분할 | `pim.intra_device_plan` |
| device 좌표와 동시 device 수 | `pim.device.mapping` |
| scheduling round | `pim.device` 바깥 `scf.for` |
| host buffer의 device별 영역 | `pim.place` |
| local compute | `linalg`와 device-local `scf.for` |

target lowering에 남는 결정은 target 물리 특성뿐이다.

- UPMEM: MRAM/WRAM 배치, DMA 분할, tasklet/barrier, host/device reduction 결합
- AiMX: batch/register 물리 눈금, padding, descriptor program, vACC 사용

---

## 10. 제거된 중복 인터페이스

다음 인터페이스는 실제 실행 IR과 중복되거나 소비되지 않아 제거했다.

- C++ `TargetSpec` registry와 UPMEM/AiMX TargetSpec
- `#pim.inter_device_plan`
- `#pim.reduction_plan`
- `execution_tile.order`
- Block Group의 중복 useful shape
- func와 `pim.device`에 동시에 존재하던 Execution Tile attribute

Device schedule은 `scf.for + pim.device.mapping`, reduction 사실은 K axis와 reduction
loop, shape는 3계층 곱에서 각각 파생한다.

---

## 11. 완료 범위

### 11.1 완료된 Common IR 코어

- 3계층 attribute와 파생 shape
- Block Group IR 재구성
- Execution Tile 형성
- multi-device scheduling round
- M/N device partition과 explicit spatial split-K
- placement, replication, dynamic tail
- intra-device bank/core/lane plan
- hierarchy/access 정합성 검증
- GEMM, GEMV, 제한된 elementwise Common lowering

이 범위에서는 PIM Common IR 구현이 완료됐으며, target lowering 작업을 진행할 수 있다.

### 11.2 아직 지원하지 않는 연산군

다음은 3계층 계약의 변경이 아니라 새로운 연산 인식과 lowering 구현이 필요한 후속
범위다.

- standalone general reduction
- convolution
- 여러 `linalg` op를 하나의 residency로 묶는 일반 multi-op fusion
- irregular gather/scatter

따라서 “모든 가능한 PIM 연산의 lowering이 완료됐다”는 의미는 아니다. 현재 목표인
GEMM, GEMV와 초기 elementwise에 필요한 Common IR 코어가 완료됐다는 의미다.

---

## 12. 테스트

Common lowering 회귀 테스트는 다음 명령으로 실행한다.

```bash
$LIT -sv $SUITE/Conversion/TTSharedToPIM
```

현재 결과:

```text
Total Discovered Tests: 19
  Passed: 19 (100.00%)
```

주요 테스트:

| 파일 | 검사 내용 |
| --- | --- |
| `form_block_groups.mlir` | GEMM Block/Group 형성과 K Group |
| `form_compute_matvec.mlir` | Triton GEMV 패턴의 `linalg.matvec` 정규화 |
| `form_execution_tiles_elementwise.mlir` | reduction-free Execution Tile |
| `map_devices.mlir` | M/N scheduling과 spatial split-K |
| `map_devices_partitions.mlir` | access 재추론 없는 metadata-only scheduling |
| `place_data_split_k*.mlir` | split-K placement와 blocked/cyclic 판정 |
| `verify_hierarchy.mlir` | metadata와 실제 compute tile 불일치 진단 |
