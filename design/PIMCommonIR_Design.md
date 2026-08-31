# PIM Common IR 설계

## 1. 목표와 경계

PIM Common IR은 Triton 연산을 target과 무관한 세 계층 실행 구조로 표현한다.

1. Block: Triton kernel 한 instance가 계산하는 원소 눈금
2. Block Group: 한 PIM unit 안의 compute 단위
3. Execution Tile: 한 PIM unit의 compute + transfer + residency 단위

Execution Tile 하나는 항상 PIM unit 하나가 담당한다. 완성된 per-unit Execution
Tile들을 `[p_m,p_n,p_k]` device grid에 배치하고, Tile 수가 active unit 수보다 많으면
같은 unit 집합을 scheduling round마다 재사용한다.

Common IR은 연산별 target 구현을 선택하지 않는다. `pim.access/load/store`, linalg
compute, 세 hierarchy attribute, placement만 만들고 UPMEM/AiMX lowering이 이 의미를
각 target IR로 옮긴다. Reduction 축·K device partition·temporal round·partial combiner는 Common IR이
확정한다. 그 combiner를 host accumulation, vACC 등 어느 target primitive로 구현할지와
물리 padding, descriptor program, tasklet 협력 방식만 target lowering의 책임이다.

Common lowering의 판단 순서는 다음과 같이 고정한다.

1. **어떤 연산인가:** linalg compute와 loop-carried accumulator/yield 관계를 읽는다.
2. **어떤 데이터를 쓰는가:** `pim.access`의 buffer, origin, sizes, strides를 만든다.
3. **얼마나 묶는가:** Block과 Block Group을 materialize한다.
4. **시간/공간 실행:** Execution Tile, scheduling round, device mapping을 만든다.
5. **무엇을 resident시키는가:** Execution Tile 하나의 `pim.place`를 만든다.
6. **실제 어디에 놓는가:** Common IR을 변경하지 않고 target memory와 primitive로 옮긴다.

앞 단계의 의미를 뒤 단계가 access 모양이나 target 특성에서 다시 추론하지 않는다.

## 2. 공통 hardware profile

```mlir
#pim.target_profile<
  target = "upmem",
  num_units = 2048,
  banks_per_unit = 1,
  cores_per_bank = 1,
  lanes_per_core = 16,
  bank_capacity_limit_bytes = 62914560,
  local_capacity_limit_bytes = 49152>
```

필드는 다음 일곱 개뿐이다.

| 필드 | 의미 |
|---|---|
| `target` | 뒤에서 선택할 target lowering |
| `num_units` | 동시에 사용할 수 있는 PIM 실행 unit 수 |
| `banks_per_unit` | unit 하나의 bank 수 |
| `cores_per_bank` | bank 하나의 PIM core 수 |
| `lanes_per_core` | core 하나가 제공하는 최대 lane 수 |
| `bank_capacity_limit_bytes` | bank 하나의 큰 메모리 byte 상한 |
| `local_capacity_limit_bytes` | core 하나의 local memory byte 상한. 없으면 0 |

`parallel_execution_kind`, target constraint map, reserved byte, descriptor byte는 기록하지
않는다. 실제 lane 실행 방식과 runtime/toolchain 값은 target config에 둔다.

Python 쪽 표현은 다음 구조다.

```text
PIMTargetSpec
  common_profile: PIMTargetProfile
  runtime_config: PIMTargetRuntimeConfig
```

UPMEM runtime config는 `nr_tasklets`, `host_threads`를 가진다. `nr_tasklets=0`이면
`lanes_per_core`를 사용하며, 양수이면 `1..lanes_per_core`에서 실제 활성 tasklet 수를
고른다. AiMX는 `lanes_per_core` 전체를 사용하고 runtime config에 `batch`,
`base_fpga`를 둔다.

## 3. 세 hierarchy attribute

세 attribute의 축 순서는 항상 `[m,n,k]`다. 실제 compute loop가 없는 축도 모든
계층에서 곱의 항등원인 1로 기록한다. parallel 축은 `pim.axis`, reduction 축은
`pim.reduction`, 데이터 영역은 `pim.access`에서 각각 독립적으로 읽는다.

```mlir
pim.block = #pim.block<shape = [32, 32, 16]>
pim.block_group = #pim.block_group<blocks = [2, 2, 2]>
pim.execution_tile = #pim.execution_tile<groups = [4, 8, 56]>
```

- Block 원소 shape: `[32,32,16]`
- Group 원소 shape: Block × blocks = `[64,64,32]`
- Execution Tile 원소 shape: Group × groups = `[256,512,1792]`

`capacity_limit_m/n/k`처럼 같은 값을 다시 저장하지 않는다. 원소 shape는 위 세
attribute에서 파생한다. 작은 문제에서는 필요한 Group 수만 Tile에 넣고, 큰 문제에서는
capacity 상한 안의 Tile을 scheduling round로 반복한다. 문제 전체를 PIM DRAM 크기로
padding하지 않는다.

Block을 Group으로 묶어 `pim.access.sizes`가 커지면 기존 Block mask도 그대로 둘 수
없다. Common lowering은 valid 식에서 원본 extent를 복원하고 다음 식으로 Group valid를
다시 materialize한다.

```text
group_valid = min(max(original_extent - group_origin, 0), group_size)
```

따라서 `sizes`, `valid`, load/store tensor shape는 항상 동일한 Block Group 범위를
나타낸다. 기존 `pad` 값은 확대된 Group의 invalid tail에도 그대로 사용한다.

## 4. Device mapping

Device mapping은 고정 순서의 정수 세 개만 가진다.

```mlir
pim.device #pim.device_mapping<partitions = [4, 16, 2]>
    attributes {pim.execution_tile = #pim.execution_tile<groups = [4, 8, 56]>} {
^bb0(%device_m: index, %device_n: index, %device_k: index):
  ...
}
```

- `p_m=4`: m 방향 Execution Tile을 네 unit에 배치
- `p_n=16`: n 방향 Execution Tile을 열여섯 unit에 배치
- `p_k=2`: 두 unit이 같은 semantic K domain의 서로 다른 partial Execution Tile을 담당

`p_k`는 수학적으로 split-K factor와 같은 역할을 한다. 차이는 결정 시점과 IR
소유권이다. source split-K는 kernel ABI와 `program_id(k)`가 K partial domain을 먼저
만들지만, `p_k`는 linalg에서 확정된 하나의 reduction domain을 Common scheduling
단계에서 나눈다. 따라서 `p_k`나 `tile_k`를 바꿔도 source kernel ABI와 output rank는
바뀌지 않는다.

partition 곱은 `num_units` 이하여야 한다. m/n compute axis가 없거나 semantic K
reduction이 없는 축의 partition은 1이다.
Common pass는 device 수나 축을 자동 선택하지 않는다. 전달받은 세 정수를 그대로
materialize한다.

각 logical 축에서 Common IR이 만드는 좌표는 다음과 같다.

```text
execution_tile_index = round * p_axis + device_coordinate
first_group          = execution_tile_index * groups_per_tile
```

따라서 Execution Tile 수가 device 수보다 커도 같은 device 집합의 반복 실행으로 전체
문제를 처리한다. `axis_extents`는 scheduling loop에 이미 표현되어 있으므로
`pim.device`에 중복 저장하지 않는다.

M/N에서는 `pim.axis` domain에, K에서는 `pim.reduction` domain에 같은 scheduling
식을 적용한다.

```text
execution_tiles = ceildiv(total_groups, groups_per_tile)
rounds          = ceildiv(execution_tiles, p_axis)
first           = (round * p_axis + device_coordinate) * groups_per_tile
trips           = min(groups_per_tile, max(total_groups - first, 0))
```

M/N의 `groups_per_tile`은 `execution_tile.groups[m/n]`이고 K의
`groups_per_tile`은 `execution_tile.groups[k]`, 즉 `tile_k`다. K에서
`device_coordinate`는 `device_k`, `round`는 `pim.scheduling_axis="reduction_k"`의
유도변수다. 따라서 spatial K와 temporal K는 별도 domain의 Cartesian product가
아니라 하나의 semantic reduction domain을 각각 device와 round 차원으로 나눈 것이다.

## 5. Access와 placement

`pim.access`는 주소 명세이며 데이터 이동이 아니다.

```mlir
%a = pim.access %A offset %offset sizes [64, 32]
     strides[%lda, %c1] origin[%m, %k] {id = 0 : i64}
     : memref<*xi8> -> !pim.access
%tile = pim.load %a : !pim.access -> tensor<64x32xi8>
```

`pim.place`는 Execution Tile 하나가 이 buffer에서 필요로 하는 영역을 표현한다.

```mlir
pim.place %A
  start[%m_begin, %k_begin]
  size[%valid_m, %valid_k]
  allocation_shape[%tile_m, %tile_k]
  strides[%lda, %c1]
  {direction = "in", access = 0 : i64, replicated_over = ["n"]}
  : memref<*xi8>
```

- `start`: 현재 device/round가 맡은 첫 원소
- `size`: tail clamp를 반영한 실제 유효 원소 수
- `allocation_shape`: Execution Tile 재사용을 위한 고정 buffer 모양
- `strides`: 원본 buffer stride
- `replicated_over`: 접근이 불변이라 같은 데이터를 받는 partition 축

`start`와 tail을 반영한 `size`는 runtime 값일 수 있지만 `allocation_shape`는 항상
Block×Block Group×Execution Tile에서 파생되는 정적 fixed slot이다. 계층으로
분류되지 않은 loop의 runtime trip count를 allocation 크기로 사용하는 fallback은
없다. 그런 loop가 resident buffer 축을 움직이면 Common lowering이 진단한다.

이 계약은 reduction K에도 예외가 없다. 전체 semantic K domain이 Execution Tile의
K residency보다 크면 Common IR이 `pim.scheduling_axis = "reduction_k"` loop를 만들고
device-local reduction loop를 `execution_tile.groups[k]` Group 이하로 제한한다. 따라서
GEMM의 A/B `allocation_shape[K]`는 항상 `block_k × group_k × tile_k`이며 전체 K가
아니다. 각 `(round, device_k)`의 `start[K]`와 tail `size[K]`도 Common
IR에 이미 계산되어 있다.

여러 K window가 만드는 partial output은 `pim.combine partial into destination`에
원래 reduction combiner를 기록한다. 현재 정규화되는 GEMM/GEMV는 `kind "add"`다. UPMEM은
이를 host accumulation으로, AiMX는 vACC로 구현할 수 있지만 window 경계나 combine
종류를 다시 추론하거나 바꿀 수는 없다.

K scheduling에는 새 source plane이나 scope 변수를 추가하지 않는다. Common IR에서
K의 유일한 compute domain은 `pim.reduction` loop다. `pim.device`의 `partitions[k]`는
한 round에 동시에 실행할 K device 수이고 `pim.scheduling_axis="reduction_k"` loop는
그 device 집합을 재사용하는 temporal round다.

```text
k_tiles        = ceildiv(total_k_groups, tile_k)
k_rounds       = ceildiv(k_tiles, p_k)
first_k_group  = (round * p_k + device_k) * tile_k
local_k_groups = min(tile_k, max(total_k_groups - first_k_group, 0))
```

`pim.combine`은 모든 `(round, device_k)` partial에 적용된다. `p_k=1`이어도 K tile이
여러 round라면 combine 대상이고, `p_k>1`이면 같은 round의 서로 다른 device partial도
같은 최종 M/N access에 결합된다. Target은 host combine이나 device/vACC 중 구현 위치만
고르며 축이나 combiner를 access 주소에서 재해석하지 않는다.

Reduction 의미는 `PIMFormCompute`가 linalg와 SSA dataflow에서 한 번만 정한다.

```mlir
%result = scf.for ... iter_args(%acc = %zero) -> tensor<...> {
  %next = linalg.matmul ... outs(%acc : tensor<...>) -> tensor<...>
  scf.yield %next : tensor<...>
} {pim.reduction = "add"}
```

즉, compute의 DPS init이 loop의 accumulator이고, linalg scalar body가 그 accumulator를
add에 사용하며, compute 결과가 같은 위치로 직접 yield되고 loop init이 add identity인
0일 때 add reduction이다. 연산 이름을 열거하지 않으므로 같은 accumulator 계약의
matmul, matvec, reduce, convolution, generic에 동일하게 적용된다. 단지 어떤 이름 없는
loop가 `pim.access` origin을 변화시킨다는 이유로 reduction으로 판정하지 않는다.
Temporal K로 나뉜 뒤에도 이 attribute는 loop에 남고, partial store의
`pim.combine`은 이 값을 그대로 받는다. Partial 결과에 epilogue를 먼저 적용한 뒤
combine하는 순서는 의미가 다르므로 Common lowering이 진단한다.

M/N 출력 축이 없는 scalar/vector reduction도 같은 계약을 사용한다. 이 경우
zero-identity를 만드는 tensor chain, temporal reduction loop, direct partial store를
하나의 `pim.device` region에 함께 넣는다. zero-init은 각 Execution Tile마다 다시
실행되고 store에는 원래 combiner가 기록된다. reduction 결과에 epilogue가 먼저
적용되거나 direct `pim.access`가 없으면 의미를 임의로 바꾸지 않고 진단한다.

`replicated_over`는 `pim.access`의 축별 `origin` dependency에서 파생한다. `offset`은
origin과 stride를 선형화한 중복 값이므로 축 dependency를 다시 판정하는 데 쓰지 않는다.
별도의 `axis_sources`는 없다.
예를 들어 GEMM의 A는 `(m,k)`에 의존하므로 n partition에서 복제되고, B는 `(k,n)`에
의존하므로 m partition에서 복제된다.

## 6. Lowering 순서

```text
TTShared IR
  -> address normalization
  -> compute 정규화 및 pim.reduction 의미 확정
  -> pim.access/load/store 형성
  -> source m/n logical grid 복원(K는 semantic reduction으로 유지)
  -> PIMTargetProfile 부착
  -> Block Group 형성
  -> Execution Tile 형성
  -> [p_m,p_n,p_k] device mapping 및 scheduling round 형성
  -> pim.place 형성
  -> PIM Common IR 검증
  -> UPMEM IR 또는 AiMX IR
```

Common verifier는 hierarchy 정합성, semantic K의 `(round, device_k)` 분할과 local
window의 대응,
access/place 일대일 대응, hierarchy에서 파생한 정확한 resident allocation,
partial store의 explicit combine, local working set 상한과 Execution Tile allocation의
unit capacity 상한을 검사한다. target 물리 padding을 포함한 최종 용량은 target
lowering이 검사한다.

K scheduling의 대응은 두 loop가 함께 존재한다는 뜻에 그치지 않는다. verifier는 host
round가 `0 .. ceildiv(ceildiv(total_groups, tile_k), p_k)`의 unit-step loop인지,
local window가 같은 `total_groups`에서
`first = (round * p_k + device_k) * tile_k`와
`min(tile_k, max(total_groups - first, 0))`를 사용하는지 확인한다. marked global K
origin과 K를 따라 움직이는 모든 `pim.place.start`도 같은 `first`에 의존해야 한다.
따라서 매 round 첫 K window만 다시 처리하는 잘못된 Common IR은 target lowering 전에
거부된다.

## 7. GEMM 예시

Triton Block이 `[32,32,16]`, Group이 `[2,2,2]` Block, Execution Tile이
`[4,8,56]` Group이고 mapping이 `[4,16,2]`라고 하자.

1. 한 Block은 `C[32x32] += A[32x16] * B[16x32]`를 계산한다.
2. 한 Group은 Block을 묶어 `C[64x64]`와 K strip 32를 계산한다.
3. 한 Execution Tile은 `[256,512,1792]` 원소 범위를 한 unit에 상주시킨다.
4. device `(dm,dn,dk)`는 해당 m/n Execution Tile과 K partial 하나를 맡는다.
5. 전체 m/n Tile 수가 4×16보다 크면 바깥 scheduling round가 같은 device를 다음
   Execution Tile에 다시 사용한다.
6. 한 device가 round마다 K Group 56개(원소 1792개)를 처리한다. round `r`와 K device
   `dk`의 시작 Group은 `(r * 2 + dk) * 56`이며, K tile이 더 많으면 다음
   `reduction_k` round에서 같은 두 K device를 재사용한다. A/B placement의 K
   allocation은 항상 1792다.
7. 각 `(round, device_k)`의 C partial은 `pim.combine ... kind "add"`가 된다.
8. A placement에는 `replicated_over=["n"]`, B에는 `replicated_over=["m"]`가 붙는다.

UPMEM lowering은 이 구조를 DPU별 MRAM/WRAM과 tasklet 협력 실행으로 옮긴다. UPMEM에서
reduction window가 여러 개면 host가 partial을 합친다. AiMX lowering은 같은 Common
IR에서 물리 Group을 `[batch,4096/batch,256/batch]` 눈금으로 padding하고 descriptor
chain 및 vACC reduction을 만든다. Common IR의 연산 의미나 device mapping은 다시
추론하지 않는다.

## 8. 연산 표현

- elementwise: reduction 축 없이 Block–Group–Tile
- matvec/GEMM: 출력 parallel 축과 reduction 축
- reduction: Group/Tile 경계의 partial 결과
- convolution: spatial, output-channel, reduction 축을 m/n/k에 대응
- fused operator: 동일 Execution Tile residency 안의 여러 linalg compute op
- gather/scatter: irregular `pim.access`를 사용하는 Block 실행

Target lowering은 Common IR의 access와 compute 의미를 인식해 target primitive로 옮긴다.
연산마다 별도의 Common IR policy나 target 선택 변수를 추가하지 않는다.

Autotuner의 `PIMProblem`은 이 의미를 Python에서 정규화한 분석 결과다. PIM compute
op에서 `compute_kind`를 읽고, unique `pim.place`마다 direction, dtype, layout,
access-dependent axis를 갖는 `PIMAccess` 하나를 만든다. 입력 access가 사용하지 않는
축은 `replicated_over`가 되고, input에는 있지만 최종 output에는 없는 `k`는 reduction
축이 된다. 이 분석 값은 Config나 새 IR attribute가 아니며, 동일한 12개 hierarchy와
device mapping 값을 탐색하기 위한 footprint/transfer 계산에만 사용한다.
