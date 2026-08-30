# Torch → PIM integration

## 흐름

```text
torch.compile
  -> aten._int_mm lowering
  -> PIMProblem(M,N,K,compute op,pim.place accesses)
  -> PIMConfig 선택
       직접 지정 | offline autotune cache | heuristic seed
  -> Triton split-K template
  -> TTShared IR
  -> PIM Common IR
  -> UPMEM IR | AiMX IR
  -> target runtime
  -> p_k partial host sum
```

Inductor lowering은 fake tensor를 다루므로 실제 측정은 lowering 밖의
`torch_backend/examples/autotune_torch.py`가 수행한다. 후보마다 `PIMConfig`를
주입해 새로 compile하고, PIM launch와 partial 합을 포함한 end-to-end 시간을
autotuner에 돌려준다.

## 공유 값

Torch integration은 별도 실행 plan을 만들지 않는다.

| 값 | 소유자 |
|---|---|
| M/N/K, compute op, dtype/layout별 access dependency | `PIMProblem` |
| 일곱 common hardware 필드 | `PIMTargetSpec.common_profile` |
| `nr_tasklets`, `host_threads`, `batch`, `base_fpga` | target runtime config |
| Block–Group–Execution Tile, `[p_m,p_n,p_k]` | `PIMConfig` |

`PIMConfig`는 그대로 backend option으로 이동한다. Execution Tile 원소 extent와
scheduling round는 `Block × Group × Tile`과 problem shape에서 파생하며 별도 Plan이나
compiler option으로 저장하지 않는다.

## Config 선택 우선순위

```text
PIMTorchConfig(spec=PIMTargetSpec(...))    공유 target profile/runtime
PIMTorchConfig(config=...)                 직접 지정
PIMTorchConfig(autotune_cache_path=...)    측정된 best
PIMTorchConfig()                           기본 Common IR seed
```

세 경로 모두 동일한 `ConstraintEngine`과 template support 검사를 통과해야 한다.
직접 지정하거나 cache에서 읽은 config를 lowering이 보정하지 않는다.

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

`local_lanes`, `device_axis_partitions`, `capacity_limit_*`, policy 문자열은 사용하지
않는다.

## 현재 template 조건

- 2-D CPU `int8` 입력 두 개와 `int32` 출력
- static M/N/K
- `K % (p_k * BLOCK_K) == 0`
- masked tile을 사용하지 않으면 M/N이 BLOCK_M/N으로 나누어떨어짐
- A/B는 row-major 연속. 아니면 lowering이 contiguous copy를 삽입

각 `p_k` plane 안의 reduction은 target lowering이 수행한다. 서로 다른 PIM Device의
partial은 torch 그래프가 host에서 합친다.
