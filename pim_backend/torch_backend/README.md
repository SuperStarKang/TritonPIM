# torch_backend — torch.compile → PIM

`torch.compile` 그래프의 FP32 `aten.mm`과 INT8 `aten._int_mm`을 semantic GEMM으로 만들고,
나머지 연산은 CPU Inductor C++ 경로로 실행한다.

```text
CPU float32 또는 int8 A[M,K], B[K,N]
        │
        ▼
aten.mm / aten._int_mm lowering
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
CPU float32 또는 int32 C[M,N]
```

Tensor device는 계속 CPU다. PIM은 tensor device가 아니라 이 template의 compile 및
execution backend다.

## 모델 단위 Session 실행

`PIMSession`에 진입하면서 DPU를 먼저 확보하고 실제 자원 수를 조회한다.
`prepare(layers, example_inputs, configure=...)`는 Torch block들을 FX로 분석하고
LayerPlan을 생성한 뒤 config, batch 배정, 공통 주소와 컴파일을 확정한다.
`run_model()`은 Layer 진입 시 weight를 올리고 준비한 CPU/PIM 구간을 순서대로 실행한다.
동일 형태의 독립 activation GEMM과 FP32 `aten.bmm`은 하나의 native batch 실행으로 묶는다.

```python
def configure(gemms, spec):
    return {
        gemm.node: PIMTorchConfig(spec=spec, config=gemm_configs[gemm.mnk])
        for gemm in gemms
    }


with PIMSession(target_spec("upmem", num_units=512, nr_tasklets=16)) as session:
    plans = session.prepare(blocks, example_inputs, configure=configure)
    output = session.run_model(*inputs)
```

`blocks`는 실행 순서대로 나열한 Torch module이며 `example_inputs`는 첫 block의 positional 입력 tuple이다.
`configure(gemms, spec)`에는 각 block의 GEMM 분석 결과와 실제 확보한 자원 spec이 전달된다.
이 함수는 원본 FX GEMM 이름을 `PIMTorchConfig`에 연결한다. 위 예제의 `gemm_configs`는
`(M, N, K)`별 명시적 config 사전이며, 대신 `autotune_cache_path`로 기존 측정 후보를 제공할 수 있다.
LayerPlan이 해당 후보에서 Execution Tile, partition, `p_b`, round를 결정한다.
CPU 컴파일 준비에서 계산한 실제 출력으로 다음 block을 분석하며, tuple 출력은 positional 입력으로 전달한다.
모든 block의 메모리 요구량을 모은 뒤 adapter가 공통 주소와 native artifact를 확정한다.
`build_layer_plan()`은 이 과정에서 쓰는 기존 FX 분석 함수로 유지한다.
Activation/ready table은 만들지 않는다. Weight는 현재 Layer만 device에 상주하며,
KV API는 우선 CPU 저장소를 제공한다. 정적 shape, UPMEM, 동기 실행을 지원한다.
UPMEM 모델 stage의 p_k > 1은 host가 K 부분합을 누적하며, 모델 경로에서 정답 검증은 아직 하지 않았다.
Resident weight GEMM은 단일 instance·단일 M/N/K round로 제한한다.

설계·상세 범위·검증 명령은 [모델 runtime 설계](../../design/ModelRuntime_Design.md),
실행 예제는 [torch_model_session.py](examples/upmem/torch_model_session.py)를 참고한다.

## Model compile

`compile_model()`은 첫 layer를 분석해(GEMM packing 포함) stage class마다 config를
autotuning하고, 장치 할당 없이 모든 layer를 컴파일한 뒤 메모리 계획(slot)을 세운다.
반환된 session은 open할 때 장치를 할당하고 slot마다 stage를 결합한다. 같은 프로세스에서 실행한다.

```python
from torch_backend import PIMPerformanceProfile, compile_model
from torch_backend.template import TORCH_GEMM_TEMPLATE_VERSION

performance = PIMPerformanceProfile.load("model-performance.json",
                                         kernel_version=TORCH_GEMM_TEMPLATE_VERSION)
compiled = compile_model(layers, example_inputs, target="aimx", units=1,
                         performance=performance,
                         runtime_options={"base_fpga": 0})
with compiled.session as session:
    output = session.run_model(*inputs)
```

```text
compile time  첫 layer 분석 (GEMM packing → stage class) → class별 제약 tune() (weight 상주 후 E 측정)
              → class별 가장 빠른 config → 모든 layer PIMSession.compile (할당 없음)
              → plan_memory: 모두 들어가면 layer마다 slot, 아니면 고정 slot N−1개 + 공유 slot 1개
runtime       open (할당, unit 수 확인, slot 결합) → run_model → close
```

모든 layer는 분석한 layer와 stage 구성이 같아야 한다. 실행 예제는
[e2e/benchmark_e2e.py](examples/aimx/e2e/benchmark_e2e.py)를 참고한다.

새 seed에 필요한 단일 unit 자료는 같은 모델의 첫 layer에서 생성할 수 있다.
`calibrate_model()`은 packing된 stage와 분리 실행할 member stage의 Block/Group
조합을 모두 수집한다. 실제 튜닝과 같은 runtime 옵션과 unit 수를 전달한다.
측정은 단일 unit에서 수행하며, 포화점을 확인하지 못하면 오류를 낸다.

```python
from torch_backend import calibrate_model

performance = calibrate_model(layers[0], example_inputs, target="aimx", units=1,
                              runtime_options={"base_fpga": 0},
                              sweep_extents=(8192, 8192, 8192), repeat=3)
performance.save("model-performance.json")
```

FX packing의 비용 비교는 정적인 연산량·메모리 계산이다. 실행 seed를 고르지 않으므로
실측 자료를 요구하지 않는다. 실행 config를 선택할 때부터 포화점 규칙을 적용한다.

## Common IR configuration

일반 GEMM lowering의 3계층 정보는 기존 Common IR config를 사용한다.
모델 경로의 `LayerPlan`은 이 정보에 CPU/PIM 순서와 instance 배정을 연결한다.

| 값 | 소유자 |
|---|---|
| M/N/K, compute와 access dependency | `PIMProblem` |
| Common target profile과 runtime config | `PIMTargetSpec` |
| Block/Group/Execution Tile와 `[p_m,p_n,p_k]` | `PIMConfig` |
| 계층별 group/tile/round 계산 | `pim_hierarchy()` |

`block_*`와 누산 dtype `ACC_TYPE`이 Triton template constexpr에 들어간다. `group_*`, `tile_*`, `p_*`는
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
PIMTorchConfig(performance=performance)    실측 포화점에 따른 heuristic seed
```

세 경로 모두 `ConstraintEngine`을 통과한다. 직접 지정하거나 cache에서 읽은 값을
Torch lowering이 보정하지 않는다.
명시 config와 cache가 모두 없을 때 `performance`도 없으면 오류를 낸다.
직접 Triton GEMM의 자료와 Torch template의 자료는 커널 버전이 달라 공유하지 않는다.
측정·seed 규칙 및 실행 명령은 [autotuner 문서](../autotuner/README.md#단일-unit-자료-생성)를 참고한다.

Torch cache context는 `torch-gemm-semantic-k-v2` template version을 사용한다.
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
        └─ TIMING_STAGES의 행동별 median (UPMEM timing v3)
        │
        ▼
AutoTuner record/cache
```

실제 실행에서는 같은 template version과 target spec으로 cache를 조회한다.

## UPMEM GEMM 계측: PIM-LLM과 같은 compute/transfer 범위

UPMEM timing version 3는 PIM-LLM의 기존 PGEMM `timer[1/2/3]`에 맞춰
`h2d/kernel/d2h`를 독립적인 시작·종료 타이머로 측정한다. PIM-LLM 코드는
수정하지 않는다. 두 구현 모두 호스트의 monotonic clock으로 재는 경과시간이다.

| 항목 | TritonPIM에서 측정하는 작업 |
|---|---|
| `acquire` | `upmem_acquire`: DPU set 조회, 필요시 확보·바이너리 적재 |
| `prepare` | 입력/출력 좌표·크기·공유 slot 키 계산, 포인터·메타데이터 대입 |
| `buffer_alloc` | round 반복문 밖에서 입력/출력 Host pool을 한 번씩 할당 |
| `buffer_init` | stage가 실행되는 고유 입력 slot과 매 round 메타데이터의 0 초기화 |
| `layout_a` | GEMM lhs를 `[Mtile,Ktile]`로 패킹하는 복사 루프 |
| `layout_b` | GEMM rhs를 원본 stride에서 row-major `[Ktile,Ntile]` 타일로 패킹하는 복사 루프 |
| `layout_other` | GEMM lhs/rhs 역할이 없는 입력의 패킹 루프 |
| `h2d` | 모든 입력 행렬·메타데이터의 `dpu_prepare_xfer` 루프 + 동기식 `dpu_push_xfer` |
| `kernel` | `dpu_launch(set, DPU_SYNCHRONOUS)` 바로 앞부터 반환 직후까지 |
| `d2h` | 결과의 `dpu_prepare_xfer` 루프 + 동기식 `dpu_push_xfer` |
| `layout_c` | 최초 부분합을 출력의 strided 주소로 복사 (`=`) |
| `reduction` | 이후 K 부분합을 출력의 strided 주소에 누산 (`+=`) |
| `buffer_free` | 모든 round와 결과 반영이 끝난 뒤 입력/출력 Host pool 해제 |

Host pool의 `malloc/free`는 round 반복문 밖에서 별도로 측정하며 H2D/D2H에 포함하지 않는다.
독립된 타이머를 사용하므로 할당·해제 비용이 `kernel`이나 transfer에 섞이지 않는다.
`prepare`는 할당·초기화·layout·reduction 구간에서 중지하고 다시 시작하므로 중복 집계가 없다.
패킹 타이머는 OpenMP 루프 전체를 감싼 wall time이며 원소마다 타이머를 호출하지 않는다.
`layout_b`는 원본의 stride 주소 계산과 타일 버퍼 복사 시간이다. `reduction`도 layout 주소 계산과 누산이
융합되어 있어 순수 덧셈 시간은 아니다. `layout_c`만으로 전체 출력 layout 비용을 뜻하지 않는다.
시간 배열은 host GEMM 진입 시 한 번만 초기화하고, 모든 scheduling round의
시간을 합산한다. 할당·해제와 계측 비용은 GEMM 전체 호출 시간에 계속 포함된다.
위 항목에 속하지 않는 런처·loop 제어 등의 비용은 전체 호출 시간과 항목 합계의
차이로 남는다.

`h2d/d2h`에는 SDK의 포인터 등록·전송·동기화 비용이 포함되고, `kernel`에는
DPU 내부 DMA와 동기화도 포함된다. B 타일은 `[Ktile,Ntile]`로 저장하며 DPU는
`A[m,k] * B[k,n]`를 계산한다. Hybrid Transpose는 제거했고 행동별 계측은 유지한다.
논리 타일 순서, padding, 공유 입력의 중복 패킹 제거와 MRAM slot 크기는 유지한다.
입력 pool 공간은 round 사이에 재사용하지만, 입력 패킹·H2D·metadata·D2H는
매 round 실행한다. 같은 round에서 `shared_over` 축만 다른 DPU들은 같은
입력 pool slot을 공유하므로 그 slot은 한 번만 패킹한다.
호스트 행동별 시간은 PGEMM의 preprocess/postprocess와 직접 대응하지 않는다.

반영할 때는 사용자가 lowering과 translator를 다시 빌드한다:

```bash
cmake --build /home/dlrkdals/TritonPIM/triton/build/cmake.linux-x86_64-cpython-3.12 \
  --target triton-shared-opt triton-shared-translate
```

Python backend compile cache version도 변경했다. 재빌드 후 새 Python 프로세스에서 실행한다.
driver는 네이티브 host의 timing version을 실행 전에 확인하고 버전이 다르면 중단한다.
일반 UPMEM 실행은 `TIMING_STAGES` 순서의 13개 항목이며 AiMX의 5개 항목은 유지한다.
생성 host의 ABI 검사에 쓰는 `UPMEM_TIMING_VERSION`은 3으로 유지한다.
기존 `combine` 시간은 v3의 `layout_c + reduction`과 출력 쪽 `prepare`로 나뉜다.
이전 JSON의 `stage/combine/host_alloc/host_free`는 당시 범위로 해석해야 한다.

## 역할별 파일

| 파일 | 역할 |
|---|---|
| `config.py` | target/config 선택과 Inductor 안전장치 |
| `_autotuner.py` | Common AutoTuner 객체의 단일 import 지점 |
| `execution.py` | config 선택, Common hierarchy 로그, backend option 전달 |
| `template.py` | M/N source grid와 하나의 semantic K reduction |
| `lowering.py` | FP32 `aten.mm`, INT8 `aten._int_mm` → PIM template output `C[M,N]` |
| `scheduling.py` | PIM template는 Triton, 나머지는 CPU C++ scheduling |
| `runtime_compat.py` | CPU tensor와 PIM compile target을 분리하는 runtime 연결 |
| `backend.py` | `torch.compile` backend entry |
| `fx_analysis.py` | GEMM shape/dtype, weight 출처 분석 |
| `fx_normalization.py` | `linear`/`addmm` 정규화와 원본 GEMM 대응 |
| `fx_packing.py` | GEMM packing: n 잇기(공유 A), m 쌓기(공유 B), block-diagonal(target model 판정) |
| `model.py` | weight freeze, packing, `LayerPlan`, `build_layer_plan`: FX graph를 CPU/PIM stage partition으로 분할 |
| `model_policy.py` | 첫 layer의 stage class 분석과 model stage 제약 |
| `model_compile.py` | 제약 autotuning, stage 측정, class별 가장 빠른 config, `compile_model` |
| `../backend/residency.py` | 메모리 계획: slot 수, 고정 slot, layer → slot 대응표 |
| `../backend/runtime.py` | 공통 `RuntimeBackend` 계약과 요청 단위 `PIMSession` |
| `../backend/upmem_adapter.py` | UPMEM 모델 경로: stage 배치·program 링크, DPU 할당/해제와 stage 결합 |
| `../backend/upmem_artifact.py` | UPMEM stage의 MRAM binding 해석과 주소 고정 DPU program 링크 |
| `../backend/aimx_adapter.py` | AiMX 모델 경로: FPGA 자원, stage 배치와 context 결합 |

예제는 target별로 나뉜다: [examples/upmem](examples/upmem), [examples/aimx](examples/aimx).
목록은 [examples/README.md](examples/README.md)에 있다.


## 현재 지원 범위

- 입력은 2-D CPU `torch.float32` 두 개 또는 `torch.int8` 두 개
- 출력은 각각 CPU `torch.float32` 또는 `torch.int32`
- static M/N/K (`dynamic=False`)
- A/B는 row-major contiguous; 아니면 lowering이 contiguous copy를 삽입
- template 주변 prologue/epilogue fusion은 아직 열지 않음
- CPU fallback 없이 지원 범위를 벗어나면 진단

M/N/K tail은 template의 masked access와 Common scheduling의 local trip count로
표현한다. source split-K divisibility 조건은 없다.
