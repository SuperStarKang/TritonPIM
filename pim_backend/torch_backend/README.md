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

config = PIMTorchConfig(
    target="upmem",
    block_m=32, block_n=32, block_k=16, split_k=4,
    num_units=256, local_lanes=8, host_threads=8,
    group_m=1, group_n=1, group_k=1,
    tile_m=32, tile_n=32, tile_k=256,
    column_partitions=8,          # p_n — 아래 "device 분할" 참고
)

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
| `config.py` | `PIMTorchConfig` — 모든 설정. `backend_options()`가 PIMOptions로 갈 dict를 만든다. offline autotune cache 조회(단계 B)도 여기 있다. |
| `template.py` | `pim_split_k_template` — `examples/split_k.py`의 커널을 Inductor TritonTemplate으로 옮긴 것. |
| `lowering.py` | `aten._int_mm` → (zeros → PIM template → CPU sum). 지원 조건 검사와 전역 lowering 교체 범위 관리. |
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

## device 분할

자동 인수분해(`device_axis_partitions=""`)를 쓸 수 없다. `PIMMapDevicesPass`는
`explicit_k_partition`을 device 축에서 제외하고 `output_row_tile` /
`output_column_tile` 에만 `factorize(num_units)`를 나눠 주므로, split-K가 device
병렬로 전혀 쓰이지 않는다.

```
output_row_tile:<p_m>, output_column_tile:<p_n>, explicit_k_partition:<p_k>
```

`column_partitions` / `row_partitions`로 주면 위 문자열을 만들어 주고
(`p_k = split_k`), `device_axis_partitions`에 직접 써도 된다.

```
p_n <= ceil(N / BLOCK_N)          그보다 크면 device가 논다
p_m <= ceil(M / BLOCK_M)
p_m * p_n * p_k <= num_units
split_k % p_k == 0
```

`row_partitions`의 기본값 1은 직접 경로에서 확인된 형태다. 출력 access가 rank-3
(plane, row, col)로 승격되면 축마다 loop가 하나씩이라 1보다 커도 구조상 문제가
없지만, 아직 검증하지 않았다.

## 지원 조건 (strict 모드)

하나라도 어기면 `UnsupportedPIMIntMM`으로 즉시 멈춘다. CPU fallback은 하지 않는다 —
fallback이 있으면 PIM template compile이 실패했는데도 테스트가 통과한 것처럼 보인다.

- 입력 2개, 각각 2-D
- A, B dtype `torch.int8`, 출력 `torch.int32`
- tensor device CPU
- shape static (`dynamic=False`)
- `K_A == K_B`
- `K % (SPLIT_K * BLOCK_K) == 0`
- `M % BLOCK_M == 0`, `N % BLOCK_N == 0`
  — 나누어떨어지면 template이 mask를 만들지 않는다. Inductor는 static shape를
  상수로 확정하는데, `PIMPlaceDataPass`의 `isExtentCandidate`가 상수를 extent
  후보에서 배제하므로 mask가 있으면 extent 복원이 실패한다. 직접
  경로(`examples/split_k.py`)는 M/N/K가 런타임 인자라 extent가 SSA 값이어서 이
  문제가 없다. 제약이 풀렸는지 확인하려면 `allow_masked_tiles=True`.
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
