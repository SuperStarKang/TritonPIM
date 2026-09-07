# torch_backend — torch.compile → PIM

`torch.compile` 그래프의 `aten._int_mm`을 하나의 semantic GEMM으로 만들고,
나머지 연산은 CPU Inductor C++ 경로로 실행한다.

```text
CPU int8 A[M,K], B[K,N]
        │
        ▼
aten._int_mm lowering
        │
        ├─ PIMProblem(M,N,K, A(m,k), B(k,n), C(m,n))
        └─ PIMConfig 선택
             직접 지정 | Torch autotune cache | heuristic seed
        │
        ▼
PIM semantic GEMM TritonTemplate
  source grid: M/N only
  source K: one complete semantic reduction
        │
        ▼
TTIR → TTShared IR → PIM Common IR
        │
        ├─ Block Group
        ├─ Execution Tile
        ├─ [p_m,p_n,p_k] scheduling
        ├─ pim.place
        └─ pim.combine
        │
        ▼
UPMEM | AiMX target lowering
        │
        ▼
CPU int32 C[M,N]
```

Tensor device는 계속 CPU다. PIM은 tensor device가 아니라 이 template의 compile 및
execution backend다.

## Common IR configuration

Torch integration은 별도 실행 plan을 만들지 않는다.

| 값 | 소유자 |
|---|---|
| M/N/K, compute와 access dependency | `PIMProblem` |
| Common target profile과 runtime config | `PIMTargetSpec` |
| Block/Group/Execution Tile와 `[p_m,p_n,p_k]` | `PIMConfig` |
| 계층별 group/tile/round 계산 | `pim_hierarchy()` |

`block_*`만 Triton template constexpr에 들어간다. `group_*`, `tile_*`, `p_*`는
backend option으로 Common IR에 전달된다. 특히 `p_k`는 source grid, constexpr,
output shape에 나타나지 않는다.

```text
block_group_extent[d] = block[d] * group[d]
total_groups[d]       = ceildiv(problem[d], block_group_extent[d])
execution_tiles[d]    = ceildiv(total_groups[d], tile[d])
rounds[d]             = ceildiv(execution_tiles[d], p[d])
```

K 역시 같은 semantic domain을 `tile_k`와 `p_k`가 temporal/spatial로 나눈다.
서로 다른 `(round_k, device_k)` contribution은 Common `pim.combine`과 target
lowering이 최종 `C[M,N]`에 결합한다. Torch graph에는 `partial[p_k,M,N]`이나 CPU
partial sum이 없다.

## Config 선택

```text
PIMTorchConfig(config=...)                 직접 지정
PIMTorchConfig(autotune_cache_path=...)    Torch 경로에서 측정한 best
PIMTorchConfig()                           heuristic seed
```

세 경로 모두 `ConstraintEngine`을 통과한다. 직접 지정하거나 cache에서 읽은 값을
Torch lowering이 보정하지 않는다.

Torch cache context는 `torch-int-mm-semantic-k-v1` template version을 사용한다.
직접 Triton 예제와 같은 JSON 파일을 쓸 수 있지만 측정 objective가 다르므로 entry는
분리된다. 기본 파일은 `$TRITON_CACHE_DIR/pim_autotune_cache.json`이다.

## Offline autotuning

Inductor lowering은 fake tensor를 다루므로 실제 장비 측정은 lowering 밖의
`examples/autotune_torch.py`가 수행한다.

```text
candidate PIMConfig
        │
        ▼
PIMTorchConfig(config=candidate)
        │
        ▼
torch.compile
        │
        ├─ warm-up 1회
        ├─ end-to-end median
        ├─ correctness sample
        └─ stage/h2d/kernel/d2h/combine median
        │
        ▼
AutoTuner record/cache
```

실제 실행에서는 같은 template version과 target spec으로 cache를 조회한다.

## 역할별 파일

| 파일 | 역할 |
|---|---|
| `config.py` | target/config 선택과 Inductor 안전장치 |
| `_autotuner.py` | Common AutoTuner 객체의 단일 import 지점 |
| `execution.py` | config 선택, Common hierarchy 로그, backend option 전달 |
| `template.py` | M/N source grid와 하나의 semantic K reduction |
| `lowering.py` | `aten._int_mm` → PIM template output `C[M,N]` |
| `scheduling.py` | PIM template는 Triton, 나머지는 CPU C++ scheduling |
| `runtime_compat.py` | CPU tensor와 PIM compile target을 분리하는 runtime 연결 |
| `backend.py` | `torch.compile` backend entry |

## 현재 지원 범위

- 입력은 2-D CPU `torch.int8` 두 개
- 출력은 CPU `torch.int32`
- static M/N/K (`dynamic=False`)
- A/B는 row-major contiguous; 아니면 lowering이 contiguous copy를 삽입
- template 주변 prologue/epilogue fusion은 아직 열지 않음
- CPU fallback 없이 지원 범위를 벗어나면 진단

M/N/K tail은 template의 masked access와 Common scheduling의 local trip count로
표현한다. source split-K divisibility 조건은 없다.
