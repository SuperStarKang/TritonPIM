# torch_backend — torch.compile → PIM

torch.compile된 그래프에서 `aten._int_mm`만 PIM split-K template로 실행하고,
나머지 연산은 기존 CPU Inductor(C++/OpenMP)로 실행한다.

```
CPU int8 tensor A, B
        │
        ▼
torch._int_mm(A, B)                     TorchDynamo FX graph
        │
        ▼
Inductor lowering
        ├─ 다른 CPU 연산 ─────────────► CppScheduling
        └─ aten._int_mm
               │
               ▼
        zeros[SPLIT_K,M,N] (CPU Cpp)
               │ mutated input
        PIM split-K TritonTemplate ────► TritonScheduling
               │                          → triton.compile(target="pim")
               │                          → TTIR → TTShared IR → PIM IR → artifact
               │                          → PIMLauncher
               ▼
        partial[SPLIT_K,M,N]
               │
        sum(dim=0)  (CPU Cpp)
               ▼
        C[M,N], CPU, int32
```

tensor device는 계속 CPU다. PIM은 tensor device가 아니라 **template의
compile / execution backend**다.

## 쓰는 법

`pim_backend` 디렉토리를 `sys.path`에 얹고 `torch_backend`를 import한다
(`examples/`가 `autotuner`를 쓰는 방식과 같다).

```python
import sys
from pathlib import Path
sys.path.insert(0, "/home/dlrkdals/TritonPIM/triton_shared/pim_backend")

import torch
from torch_backend import PIMInductorBackend, PIMTorchConfig

config = PIMTorchConfig(target="upmem", split_k=4)

compiled = torch.compile(
    fn,
    backend=PIMInductorBackend(config),
    fullgraph=True,
    dynamic=False,
)
```

환경변수는 하나도 쓰지 않는다.

- 검증 스크립트: `examples/torch_int_mm.py` (단일 MM), `examples/torch_attention.py`

## 파일

| 파일 | 역할 |
|---|---|
| `config.py` | `PIMTorchConfig` — "무엇을 고를지"와 Inductor 안전장치만. 실행 계획은 여기 없다. |
| `_autotuner.py` | autotuner 패키지를 쓰기 위한 단일 진입점(sys.path + re-export). |
| `plan.py` | problem/hardware/config/plan을 묶고 template metadata로 옮긴다. seed 선택과 autotune cache 조회도 여기. |
| `template.py` | `pim_split_k_template` — `examples/split_k.py`의 커널을 Inductor TritonTemplate으로 옮긴 것. |
| `lowering.py` | `aten._int_mm` → (zeros → PIM template → CPU sum). 그래프 모양 검사와 전역 lowering 교체 범위 관리. |
| `scheduling.py` | `PIMCombinedScheduling` — PIM template는 TritonScheduling, 나머지는 CppScheduling. |
| `runtime_compat.py` | torch runtime monkeypatch 두 개 (backend 명시 선택, template 기준 compile 분기). |
| `backend.py` | `PIMInductorBackend` — torch.compile backend entry. |

## shape/stride를 런타임 인자로 넘기는 이유

Inductor의 `{{size}}` / `{{stride}}` hook은 static shape에서 정수 리터럴을 낸다.
그러면 `PIMRecoverAccessOriginPass`가 split-K 출력면을 별도 축으로 승격하지 못한다.
그 pass의 `canonicalStride`는 stride 식을 `(기준 SSA 값, 상수 배수)`로 가른다.

```
stride_ck 가 상수 M*N   → base = null, coeff = M*N
tile 축 0 stride 도 상수 N → base = null, coeff = N
→ base 가 둘 다 null 로 일치하고 (M*N) % N == 0 이라 pid_k 항이 행 축에 흡수된다
```

출력 access가 rank-2가 되고, 버퍼 축 0을 `pid_k`와 `pid_m`이 함께 몰게 되어
`PIMPlaceDataPass`의 `checkBlockedComposite`가 진단한다.

stride가 런타임 인자면 `base`가 SSA 값이라 어느 tile 축의 `base`(null)와도 일치하지
않는다. 그때 pass는 `promote()`로 크기 1짜리 축을 앞에 새로 만든다 — 주석 그대로
"split-K의 출력면처럼 원본 버퍼에는 있는데 tile에는 없는 축"이다. 축마다 loop가
하나씩만 남으므로 composite 판정 자체가 일어나지 않는다.
`examples/split_k.py`가 통과하는 이유가 이것이다.

그래서 `template.py`는 `{{rt_size}}` / `{{rt_stride}}` hook으로 M/N/K와 stride
여섯 개를 `args.sizevars`에 실어 런타임 스칼라 인자로 만든다. Inductor의 기존
통로라 `python_argdefs`가 `SizeArg`를 만들고 `call_kernel`이 값을 실어 준다.
값이 같은 것끼리는 한 인자로 모이고, 값이 1인 인자는 Triton이 constexpr로
특수화한다 — 둘 다 직접 경로와 같다.

## 실행 계획은 autotuner가 갖는다

block/group/execution tile, 축별 device 분할 수, 하드웨어 자원은 이 패키지가
들고 있지 않다. `autotuner/`의 객체를 그대로 쓴다 (autotuner는 읽기만 하고
고치지 않는다).

| 무엇 | 어디서 |
|---|---|
| `PIMProblem` (M,N,K,dtype,layout) | `lowering.validate_pim_int_mm`이 `mm_args` 결과에서 만든다 |
| `PIMHardwareProfile` (num_units, lanes, memory 예산) | `autotuner/targets.py`의 등록표 |
| `PIMConfig` (block/group/execution, p_m/p_n/p_k, policy) | heuristic seed 또는 autotune cache |
| `PIMPlan` (active_device_grid, rounds, transfer/reduction plan) | `PIMPlanner(problem, hardware).build(config)` |
| config가 이 shape에서 성립하는가 | `ConstraintEngine(problem, hardware).check(config)` |
| `backend_options` dict | `autotuner.decorator.pim_launch_options` |

`plan.py`가 하는 일은 둘뿐이다 — 어느 config를 고를까, 고른 것을 template
metadata로 어떻게 옮길까.

```
PIMTorchConfig(split_k=4)          ← 원하는 P_K의 heuristic seed를 고른다
PIMTorchConfig(autotune_cache_path=...)  ← offline autotune cache의 best (단계 B)
```

실제로 고른 config는 lowering이 로그로 찍는다.

```
python examples/torch_int_mm.py -v
  PIM _int_mm (M,N,K)=(128,256,1024): block=(16,16,16) group=(4,4,8)
  execution=(2,1,2) tile=(128,64,256) devices=(1,4,4)=16/2048 rounds=1
  distribution=N-major reduction=host
```

### torch 쪽에서 손대는 두 곳

1. **`pin_row_partitions=True` (기본)** — `resolve_distribution("N-major")`는
   P_N을 먼저 최대로 잡고 남은 budget을 P_M에 주므로 `num_units`가 크면 P_M>1이
   나온다. 그 형태는 아직 검증 전이라 1로 고정한다. 출력 access가 rank-3
   (plane,row,col)로 승격되므로 구조상 문제는 없어 보인다.
2. **`_fit_k_axis`** — seed의 execution 유도는 device 활용 단위에서 나오므로
   K가 작은 shape에서는 `tile_k`가 K 몫(spatial K plane)을 넘긴다. 예: K=128,
   P_K=4면 plane이 32인데 seed는 tile_k=256을 낸다. 줄이는 방향으로만 맞춘다.

## 지원 조건 (strict 모드)## 지원 조건 (strict 모드)

하나라도 어기면 `UnsupportedPIMIntMM`으로 즉시 멈춘다. CPU fallback은 하지 않는다 —
fallback이 있으면 PIM template compile이 실패했는데도 테스트가 통과한 것처럼 보인다.

- 입력 2개, 각각 2-D
- A, B dtype `torch.int8`, 출력 `torch.int32`
- tensor device CPU
- shape static (`dynamic=False`)
- `K_A == K_B`

template 고유 조건 (`plan.check_template_support`):

- `K % (P_K * BLOCK_K) == 0` — 커널이 K를 P_K개의 **연속** 구간으로 가른다.
  `ConstraintEngine`의 Constraint 5도 같은 것을 보지만 그쪽은
  `target_constraints["k_split_block_multiple"]`가 있는 target에서만 본다.
  이 template은 target과 무관하게 blocked 분할을 전제하므로 여기서도 본다.
- `M % BLOCK_M == 0`, `N % BLOCK_N == 0` — 나누어떨어지면 mask를 만들지 않는다.
  mask의 extent가 상수면 `PIMPlaceDataPass`의 `isExtentCandidate`가 extent
  복원에 실패한다. 확인하려면 `allow_masked_tiles=True`.
- `reduction_policy in ("host", "none")` — 이 lowering은 항상 CPU 합을 붙인다.
  `TARGET_REDUCTION_POLICY`가 `"device"`인 target(aimx)에서는 device가 이미
  합친 결과에 또 합치게 되므로 거부한다.

나머지(memory 예산, 축별 분할 수, alignment, tile grid)는 `ConstraintEngine`이
본다 — 우리가 검사를 복제하지 않는다.
- A, B row-major 연속 — 아니면 lowering이 복사를 하나 끼운다
  (`ir.ExternKernel.require_contiguous`). 연산 자체는 여전히 PIM에서 돈다.

attention의 QK처럼 transpose가 필요한 자리는 `k.transpose(0, 1).contiguous()`로
쓴다.

## 아직 하지 않은 것

- CPU–PIM partition, TTShared IR fan-out, 비동기 실행
- PIM template 주변 fusion (epilogue / prologue 모두 막아 두었다)
- compile / runtime autotuning (단계 C). 지금 붙일 수 있는 것은 offline cache
  조회(단계 B)까지다 — lowering은 fake tensor를 다루므로 실측할 입력이 없다.
- 학습(backward) 그래프. `_int_mm`은 미분 대상이 아니라 지금 경로에서는 문제가
  없지만, backward compile은 `PIMInductorBackend.__call__`이 끝난 뒤에 일어나므로
  lowering 교체 범위 밖이다.
