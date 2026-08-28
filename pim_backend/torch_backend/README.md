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
from torch_backend import (
    PIMInductorBackend, PIMTorchConfig, target_spec,
)

config = PIMTorchConfig(spec=target_spec("upmem"))

compiled = torch.compile(
    fn,
    backend=PIMInductorBackend(config),
    fullgraph=True,
    dynamic=False,
)
```

환경변수는 하나도 쓰지 않는다.

- 검증 스크립트: `examples/torch_int_mm.py` (단일 MM), `examples/torch_attention.py`
- 튜닝 스크립트: `examples/autotune_torch.py`
- PIM-LLM 비교: `examples/attention_compare.py`
- 설계 문서: `design/TorchIntegration.md`

## 파일

| 파일 | 역할 |
|---|---|
| `config.py` | `PIMTorchConfig` — "무엇을 고를지"와 Inductor 안전장치만. 실행 계획은 여기 없다. |
| `_autotuner.py` | autotuner 패키지를 쓰기 위한 단일 진입점(sys.path + re-export). |
| `execution.py` | problem/target/config를 묶고 template metadata로 옮긴다. seed 선택과 autotune cache 조회도 여기. |
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

## Common IR configuration은 autotuner가 고른다

block/group/execution tile, 축별 device 분할 수, 하드웨어 자원은 이 패키지가
들고 있지 않다. `autotuner/`의 객체를 그대로 쓴다 (autotuner는 읽기만 하고
고치지 않는다).

| 무엇 | 어디서 |
|---|---|
| `PIMProblem` (M,N,K,dtype,layout) | `lowering.validate_pim_int_mm`이 `mm_args` 결과에서 만든다 |
| `PIMTargetSpec` | compiler와 autotuner가 공유하는 `backend/target.py` |
| `PIMConfig` (block/group/tile, p_m/p_n/p_k) | heuristic seed 또는 autotune cache |
| config가 이 shape에서 성립하는가 | `ConstraintEngine(problem, target).check(config)` |
| `backend_options` dict | `autotuner.decorator.pim_launch_options` |

`execution.py`가 하는 일은 둘뿐이다 — 어느 config를 고를까, 고른 것을 template
metadata로 어떻게 옮길까. 고르는 방법은 셋이고 우선순위가 있다.

```
PIMTorchConfig(spec=PIMTargetSpec(...))  공유 target profile/runtime
PIMTorchConfig(config=PIMConfig(...))    ① 그대로 쓴다. 보정도 P_M 고정도 안 한다
PIMTorchConfig(autotune_cache_path=...)  ② 실측된 best (단계 B). miss는 fail-fast
PIMTorchConfig()                         ③ 기본 Common IR seed. 정답 확인용
```

①은 autotuner가 "이 config를 재라"고 요구할 때의 통로다. ②③과 달리 어떤
보정도 하지 않는다 — 값을 바꾸면 잰 것과 기록한 것이 달라진다. 유효성 판정
(`ConstraintEngine`)은 셋 다 똑같이 거친다.

③은 성능 숫자를 내는 용도가 아니다. 실제 최적점은 Block, Group, Execution Tile,
`p_m/p_n/p_k`를 함께 측정하는 autotuner가 고른다.

### 실측해서 cache를 채우기

```bash
python examples/autotune_torch.py --m 8192 --n 8192 --k 128   # attention QK
python examples/autotune_torch.py --m 8192 --n 128  --k 8192  # attention P.V
```

`examples/autotune_gemm.py`와 다른 점: 그쪽은 `examples/split_k.py`의 커널을
직접 launch한 것을 재고, 이쪽은 **torch 경로 자체**를 잰다(측정에 PIM launch와
CPU partial 합이 모두 들어간다). config마다 `torch.compile`을 새로 만들어 재고
그 결과를 `autotuner.tune`의 `benchmark_fn`으로 넘긴다 — 튜닝이 lowering 안에서
일어날 이유가 없다. 밖에는 실제 텐서가 있다.

두 스크립트가 같은 cache 파일을 쓰고 template을 cache key에 넣지 않으므로, 같은
`(M,N,K)`를 두 경로에서 튜닝하면 나중 것이 이긴다.

실제로 고른 config는 lowering이 로그로 찍는다.

```
python examples/torch_int_mm.py -v
  PIM _int_mm (M,N,K)=(128,256,1024): block=(16,16,16) group=(4,4,8)
  tile=(2,1,2) extent=(128,64,256) devices=(1,4,4)=16/2048 rounds=1
```

## 지원 조건

하나라도 어기면 `UnsupportedPIMIntMM`으로 즉시 멈춘다. CPU fallback은 하지 않는다.

- 입력 2개, 각각 2-D
- A, B dtype `torch.int8`, 출력 `torch.int32`
- tensor device CPU
- shape static (`dynamic=False`)
- `K_A == K_B`

template 고유 조건 (`execution.check_template_support`):

- `K % (P_K * BLOCK_K) == 0` — 커널이 K를 P_K개의 **연속** 구간으로 가른다.
  이 조건은 Common IR 제약이 아니라 현재 split-K Triton template의 조건이다.
- `M % BLOCK_M == 0`, `N % BLOCK_N == 0` — 나누어떨어지면 mask를 만들지 않는다.
  mask의 extent가 상수면 `PIMPlaceDataPass`의 `isExtentCandidate`가 extent
  복원에 실패한다. 확인하려면 `allow_masked_tiles=True`.
- 각 `p_k` device 내부 reduction은 target lowering이 수행하고, 서로 다른
  `p_k` partial은 torch 그래프의 host sum이 합친다.

나머지(memory 예산, 축별 분할 수, tile grid)는 `ConstraintEngine`이
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
