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
각 target IR로 옮긴다. Reduction 위치, 물리 padding, descriptor program, tasklet 협력
방식은 target lowering의 책임이다.

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

세 attribute의 축 순서는 항상 `[m,n,k]`다. 없는 축은 Block과 Execution Tile에서 0,
Block Group에서는 곱의 항등원인 1이다.

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
- `p_k=2`: 두 unit이 서로 다른 K partial Execution Tile을 담당

partition 곱은 `num_units` 이하여야 한다. 커널에 없는 축의 partition은 1이다.
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

`replicated_over`는 access dependency에서 파생한다. 별도의 `axis_sources`는 없다.
예를 들어 GEMM의 A는 `(m,k)`에 의존하므로 n partition에서 복제되고, B는 `(k,n)`에
의존하므로 m partition에서 복제된다.

## 6. Lowering 순서

```text
TTShared IR
  -> address normalization
  -> pim.access/load/store 형성
  -> m/n/k logical grid 복원
  -> compute 정규화
  -> PIMTargetProfile 부착
  -> Block Group 형성
  -> Execution Tile 형성
  -> [p_m,p_n,p_k] device mapping 및 scheduling round 형성
  -> pim.place 형성
  -> PIM Common IR 검증
  -> UPMEM IR 또는 AiMX IR
```

Common verifier는 hierarchy 정합성, access/place 일대일 대응, local working set 상한,
정적으로 계산 가능한 Execution Tile allocation의 unit capacity 상한을 검사한다. 동적
allocation과 target 물리 padding을 포함한 최종 용량은 target lowering이 검사한다.

## 7. GEMM 예시

Triton Block이 `[32,32,16]`, Group이 `[2,2,2]` Block, Execution Tile이
`[4,8,56]` Group이고 mapping이 `[4,16,1]`이라고 하자.

1. 한 Block은 `C[32x32] += A[32x16] * B[16x32]`를 계산한다.
2. 한 Group은 Block을 묶어 `C[64x64]`와 K strip 32를 계산한다.
3. 한 Execution Tile은 `[256,512,1792]` 원소 범위를 한 unit에 상주시킨다.
4. device `(dm,dn,0)`는 해당 m/n Execution Tile 하나를 맡는다.
5. 전체 m/n Tile 수가 4×16보다 크면 바깥 scheduling round가 같은 device를 다음
   Execution Tile에 다시 사용한다.
6. A placement에는 `replicated_over=["n"]`, B에는 `replicated_over=["m"]`가 붙는다.

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
