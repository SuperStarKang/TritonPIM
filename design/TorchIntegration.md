# Torch → PIM integration

## 흐름

```text
torch.compile
  -> aten._int_mm lowering
  -> PIMProblem(M,N,K, compute op, pim.access dependencies)
  -> PIMConfig 선택
       직접 지정 | Torch offline autotune cache | heuristic seed
  -> semantic GEMM TritonTemplate
       source grid = M/N
       source reduction = complete K
  -> TTShared IR
  -> PIM Common IR
       Block -> Block Group -> Execution Tile
       temporal/spatial scheduling
       pim.place -> pim.combine
  -> UPMEM IR | AiMX IR
  -> target runtime
  -> C[M,N]
```

Torch source에는 `SPLIT_K`, `pid_k`, `partial[p_k,M,N]`가 없다. `p_k`는
`#pim.device_mapping`에 들어가는 Common scheduling 값이며, `tile_k`와 함께 하나의
semantic reduction Block Group domain을 분할한다. Target은 Common `pim.combine`이
명시한 contribution과 최종 destination을 그대로 lowering한다.

## 공유 값

| 값 | 소유자 |
|---|---|
| M/N/K, compute op, dtype/layout별 access dependency | `PIMProblem` |
| 일곱 common hardware 필드 | `PIMTargetSpec.common_profile` |
| `nr_tasklets`, `host_threads`, `batch`, `base_fpga` | target runtime config |
| Block–Group–Execution Tile, `[p_m,p_n,p_k]` | `PIMConfig` |
| total groups, Execution Tiles, scheduling rounds | `pim_hierarchy()` |

`PIMConfig`는 보정 없이 backend option으로 이동한다. `block_*`는 template constexpr,
나머지 계층과 mapping 값은 PIM compiler option이다.

## Config 선택과 cache

```text
PIMTorchConfig(config=...)                 직접 지정
PIMTorchConfig(autotune_cache_path=...)    측정된 best
PIMTorchConfig()                           Common IR heuristic seed
```

Torch offline tuner와 production lookup은 같은
`torch-int-mm-semantic-k-v1` template version을 사용한다. 직접 Triton 예제와 cache
파일은 공유할 수 있지만 context entry는 분리한다. 같은 `PIMConfig`라도 Torch
wrapper/layout을 포함한 end-to-end objective와 직접 launch objective가 다르기 때문이다.

## Backend option

공통 option:

```text
pim_target
num_units
banks_per_unit
cores_per_bank
lanes_per_core
bank_capacity_limit_bytes
local_capacity_limit_bytes
group_m/n/k
tile_m/n/k
p_m/n/k
```

Target runtime option:

```text
UPMEM: nr_tasklets, host_threads
AiMX:  batch, base_fpga
```

## 현재 template 조건

- 2-D CPU `int8` 입력 두 개와 `int32` 출력
- static M/N/K
- A/B row-major contiguous; 아니면 lowering이 copy를 삽입
- M/N/K tail은 masked access와 Common local trip count로 처리
- template 주변 fusion은 아직 지원하지 않음

`K % (p_k * BLOCK_K) == 0` 조건은 없다. K partition과 round는 source가 아니라
Common scheduling이 정한다.
