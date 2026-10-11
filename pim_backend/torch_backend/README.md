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
KV API는 우선 CPU 저장소를 제공한다. 정적 shape, UPMEM, `p_k=1`, 동기 실행을 지원한다.
Resident weight GEMM은 단일 instance·단일 M/N/K round로 제한한다.

설계·상세 범위·검증 명령은 [모델 runtime 설계](../../design/ModelRuntime_Design.md),
실행 예제는 [torch_model_session.py](examples/torch_model_session.py)를 참고한다.

FX graph 전체의 weight를 UPMEM에 유지하는 별도 실행 경로는
[ResidentBlock 사용법](RESIDENT.md)을 참고한다. 기존 lowering을 유지하며,
모든 weight가 같은 전체 DPU 집합과 공용 입력·출력 scratch를 사용한다.
`prepare_block()`은 `linear`/`addmm`를 먼저 정규화하고 각 `mm`을 PIM 구간으로 분리한다.
Bias, scaling, reshape 등은 CPU 구간에서 처리한다.
`examples/attention/torch_resident_attention.py`는 1024/4096 single-head FP32 attention의
weight residency, 정확도와 준비/반복 실행 시간을 측정한다. 설정과 JSON 항목은
[ResidentBlock 사용법](RESIDENT.md)의 attention 실험 항목에 설명되어 있다.
`examples/transformer/torch_resident_transformer.py --layers 1`은 여기에 causal mask, RoPE,
output projection, RMSNorm, residual, SwiGLU FFN을 포함한다. `--layers 2` 또는
`--layers 4`로 서로 다른 weight를 가진 여러 block을 하나의 resident graph에서 실행한다.
Normal orientation, FP32, 2048 DPU·16 tasklet을 사용하며, 준비 전 MRAM과 host weight
capture 크기를 출력한다. 자세한 설정은 같은 문서의 Transformer stack 실험 항목에 있다.

INT8 성능 측정은 [examples/README.md](examples/README.md)의 세 benchmark를 사용한다.
Attention kernel, projection을 포함한 Attention layer, FFN을 포함한 Transformer block을
각각 실행하며, session alloc/free와 weight 준비, 반복 실행, partition profile을 분리한다.
모델 정확도 비교나 checkpoint 로딩은 하지 않는다.

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
PIMTorchConfig()                           heuristic seed
```

세 경로 모두 `ConstraintEngine`을 통과한다. 직접 지정하거나 cache에서 읽은 값을
Torch lowering이 보정하지 않는다.

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
Resident 실행은 runtime이 SDK H2D를 직접 측정하고 `program_load`, `resident_host`를
분리한 15개 항목을 보고한다. `TIMING_VERSION`은 일반 UPMEM 실행에서 3,
resident 실행에서 4이며 JSON에는 이 실행별 버전을 기록한다.
생성 host의 ABI 검사에 쓰는 `UPMEM_TIMING_VERSION`은 3으로 유지한다.
기존 `combine` 시간은 v3의 `layout_c + reduction`과 출력 쪽 `prepare`로 나뉜다.
이전 JSON의 `stage/combine/host_alloc/host_free`는 당시 범위로 해석해야 한다.
Attention profiler는 JSON에 `pim_timing_version`도 기록한다.

`pim_backend/examples/pim_transpose_gemm.py`는 같은 `C=AB`를 계산하면서 Host B의
저장 배치만 비교한다. `row_major`는 원래 B(stride `(N,1)`), `k_contiguous`는
`B.T.contiguous().T`(같은 `[K,N]` 값, stride `(1,K)`)를 전달한다. A와 출력,
GEMM 설정과 backend의 row-major `[Ktile,Ntile]` 타일 패킹은 두 경우에 동일하다.
직접 Triton을 호출해 실제 stride를 전달하므로 Torch lowering의 contiguous 복사가 없다.

각 layout에서 warm-up 3회 이후 10회 측정한 중앙값을 나란히 출력하고,
모든 실행 결과를 동일한 CPU int32 reference와 정확히 비교한다.
`input_transform`은 매 호출의 B 준비 시간(전치 버퍼 할당 포함), `layout_b`는
native B 패킹 시간이다. `input_transform+layout_b`도 매 호출에서 합산한 뒤
중앙값을 낸다. `launch_wall`은 입력 준비를 제외하고, `e2e`는 입력 준비와 launch를
포함한다. native stage는 `launch_wall` 안에 포함되므로 이 값들을 다시 합산하지 않는다.
출력 할당, 입력 생성, reference 계산과 정확도 검사는 측정에서 제외한다.

block `(32,32,64)`, group `(1,1,1)`, partition `(4,16,1)`의 64 DPU를 사용한다.
Host threads는 8, tasklet은 16, staging reuse는 켠다.

| `(M,N,K)` | `(tile_m,tile_n,tile_k)` |
|---|---|
| `(512,512,512)` | `(4,1,8)` |
| `(1024,4096,1024)` | `(8,8,16)` |
| `(1024,8192,2048)` | `(8,16,32)` |

각 shape는 padding 없이 한 round에 맞으며 split-K는 사용하지 않는다.
backend가 준비된 환경에서 사용자가 실행한다:

```bash
python /home/dlrkdals/TritonPIM/triton_shared/pim_backend/examples/pim_transpose_gemm.py
```

## Attention 한 번의 실행을 표로 보기

`examples/attention/profile_attention.py`는 Attention 전체를 `torch.compile(fullgraph=True)`로
컴파일하고, 준비 호출 뒤 **한 번의 compiled 호출**에서 실행된 커널들을 표로 낸다.
기본 범위는 head 하나이며 `--group G`로 KV head 하나를 공유하는 query head G개를
함께 측정한다. 입력 Q/K/V는 int8, 최종 출력은 복원된 float32다. 출력 복원도 측정에
포함된다. QKV/output projection, 입력 양자화, 전체 layer의 KV group loop는 포함하지 않는다.

환경과 backend 빌드가 준비된 뒤 사용자가 직접 실행하는 명령:

```bash
conda activate pim_triton
cd /home/dlrkdals/TritonPIM
source set_env.sh
UPMEM_PROFILE=backend=hw TRITON_ALWAYS_COMPILE=0 \
  python triton_shared/pim_backend/torch_backend/examples/attention/profile_attention.py \
  --seq 256 --head-dim 128 --causal --num-units 2048
```

`--num-units`는 장비의 사용 가능한 DPU 수에 맞춘다. 실제 활성 DPU 수는 QK/PV별로
표에 나온다. 기본 `--warmup 1`은 컴파일과 초기 실행을 포함한 준비 호출이며 본 측정에서
제외된다. 따라서 명령 한 번에 기본 준비 1회 + 측정 1회가 실행된다. 정확도 확인이나
별도의 참조 실행은 자동 수행하지 않는다.

출력 열은 `operator | shape | where | time (ms) | % total | dtype`이다.

- 실제 생성된 `cpp_*` 함수와 PIM `.run` 호출을 감싼다. CPU 연산이 fusion되어 있으면
  `fused` 한 행으로 표시한다. softmax 등의 개별 ATen 시간을 임의로 분배하지 않는다.
- PIM shape는 `A @ B -> C`, CPU shape는 출력과 임시 버퍼를 포함한 tensor 인자들의
  서로 다른 shape 목록이다. 정확한 인자 순서, dtype, stride는 JSON에 남는다.
- QK/PV의 `CPU+UPMEM` 시간은 host launcher와 전송을 포함한다. 아래 별도 표의
  `TIMING_STAGES`의 13개 항목은 각 PIM 행에 이미 포함된 세부 시간이다.
  각 launch 직후 driver timing을 복사하므로 두 GEMM 모두 기록된다.
- `kernel`에는 DPU 내부 DMA가 포함되고, `stage`에는 runtime의 DPU 확보/바이너리
  재적재 비용도 들어갈 수 있다. QK/PV 교대 호출의 실제 비용을 그대로 측정한다.
- `wrapper/alloc/tracing/other`는 총시간에서 계측한 커널 호출 시간을 뺀 값이다.
  Python wrapper, 할당, 계측 기록 비용과 계측되지 않은 호출이 여기에 포함된다.
  총시간은 계측이 추가된 실행의 wall time이며, 입력 생성과 출력/파일 저장은 제외한다.

`--cache PATH`로 두 GEMM의 Torch autotune 결과를 사용한다. 없으면 heuristic seed이고,
캐시를 지정했는데 해당 shape/target 항목이 없으면 오류다. `--cpu-threads`와
`--host-threads`로 CPU 및 UPMEM host 작업의 스레드 수를 정할 수 있다.
`--json attention.json`은 개별 커널, 선택 설정, 환경, 준비/전체 시간을 저장하고,
`-v`는 원래 생성된 커널 이름과 PIM 설정도 출력한다.

`--save-inputs /tmp/attention_s1024.npz`는 입력 생성 뒤, 준비/측정 전에 2-D
`q/k/v/mask`를 저장한다. PIMLLM artifact의 `scripts/profile_attention_tvm.py`에
`--inputs /tmp/attention_s1024.npz`와 동일한 seq/head-dim/group을 전달하면 같은
데이터로 TVM Attention을 측정할 수 있다. TVM profiler는 내부 워밍업이 3회이므로
비교할 때 Torch도 `--warmup 3`으로 맞춘다. TVM의 TOTAL은 내부 그래프 계측 구간이며,
Torch의 TOTAL에는 Python 호출 경계 비용도 포함된다는 차이를 함께 기록한다.

계측기는 로컬 Torch의 `PyCodeCache.modules`와 생성된 Python wrapper를 사용한다.
warm-up 뒤 해당 모듈의 커널 참조만 임시 교체하고 호출 후 복원한다. Torch 내부 API가
바뀌거나 측정 중 재컴파일되면 계측을 다시 맞춰야 한다. 기대한 PIM 2회가 기록되지
않으면 결과를 정상 표로 출력하지 않고 진단한다.

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
| `resident.py` | 정규화된 FX graph 분할, 고정 weight 준비와 반복 실행 |
| `layer.py` | 하드웨어 할당 없는 `compile_layer`, `LayerPlan`, runtime FX load/run 노드 |
| `../backend/runtime.py` | 공통 `RuntimeBackend` 계약과 요청 단위 `PIMSession` |
| `../backend/upmem_adapter.py` | UPMEM artifact 준비, layer MRAM 교체, 명시적 자원 할당/해제 |

## 요청 단위 session과 Llama TTFT

`examples/e2e/torch_llama_ttft.py`는 로컬 Llama-2-7b-hf를 FP32로 불러온 뒤
각 layer의 7개 projection weight를 tensor별 absmax/128 scale로 INT8화한다.
Batch 1, sequence 1024, 32개 layer의 prefill과 첫 greedy token ID 생성을 수행하며
2048개 unit과 16개 UPMEM tasklet을 사용한다. Layer의 7개 projection과 head별
QK/PV 64개는 PIM에서 INT8×INT8→INT32로 실행한다. Activation scale은 1/32,
probability scale은 행별 max/127이다. 비선형 연산, embedding, 최종 RMSNorm,
마지막 token의 lm_head와 KV cache는 CPU FP32다. QKV/epilogue fusion과
후속 decode, calibration, HF CPU reference 정확도 검증은 포함하지 않는다.

```text
요청 전: weight INT8 양자화 + export + FX 분석/정규화 + PIM/CPU 컴파일 + ELF 링크
         compile_layer(...) -> LayerPlan × 32 (DPU 할당/H2D 없음)

요청:    tokenize -> PIMSession.__enter__ -> embedding
                       │ alloc 1회
                       v
         LoadLayerWeights(0) -> ready_0 -> RunLayer(0) -> hidden_0, KV_0, done_0
              ^                                                       │
              │            LoadLayerWeights(1) <──────────────────────┘
              │                    -> ready_1 -> RunLayer(1) -> ...
              │                         ...
         같은 전체 unit 집합     LoadLayerWeights(31) -> RunLayer(31)
                                                      │
                                    CPU norm -> lm_head -> argmax
                                                      │ 첫 token ID: TTFT 종료
                                           PIMSession.close -> free 1회
```

`LoadLayerWeights`와 `RunLayer`는 실제 실행되는 바깥 FX GraphModule의
`call_module` 노드다. 바깥 그래프는 Python으로 실행하고, 안쪽 layer의 GEMM과
CPU partition만 컴파일한다. load/run 모듈에는 `_is_impure=True`를 표시하고,
이전 layer의 completion token을 다음 weight 교체의 입력으로 연결한다.
session은 오래된 token, 다른 plan, 다른 thread의 실행을 거부한다.
같은 plan의 완료 token은 다음 `run_layer`에도 사용할 수 있다. 이 경우 weight를
다시 보내지 않는다. adapter의 `begin_run`은 실행 가능한 layer 상태를 확인한다.
Activation은 한 호출 안의 Q/K/V와 gate/up을 포함해 모든 GEMM에서 전송한다.
내용 비교와 검사용 host 복사본은 제거했으며 `activation_hits`는 항상 0이다.

공통 코드에는 DPU handle이나 MRAM 주소가 없다. `RuntimeBackend`는
컴파일 artifact 준비, 자원 open/close, layer 교체, kernel invocation, 누적
통계를 정의한다. `PIMTargetSpec.common_profile.target`이 adapter를 선택한다.
UPMEM adapter만 구현되어 있으며, 다른 target의 resident 실행 요청은
`NotImplementedError`로 종료한다. compiler의 3계층 lowering은 변경하지 않는다.

UPMEM에서는 session open/close가 `upmem_session_open/close`를 거쳐
`dpu_alloc/free`를 호출한다. layer 교체는 이전 native resident block의
host 상태를 해제한 뒤 같은 DPU 집합에 새 block을 연결한다. layer의 모든
weight와 공용 input/output scratch를 배치하고, 컴파일 때 링크한 MRAM 주소와
capture 결과를 대조한 뒤 weight를 전송한다. 현재는 `p_k=1`, 한 scheduling
round와 weight당 한 window를 요구한다. 실행 도중의 GEMM은 weight H2D를
생략하지만, 기존 generated host 코드의 weight packing까지 생략하지는 않는다.

모든 layer의 packed weight를 host에 보관하지 않는다. 각 load 노드에서 현재
layer만 capture하고, upload 직후 native staging buffer를 해제한다. 원본 모델과
각 plan의 contiguous/padded weight는 host에 계속 존재한다. 현재 배치의
INT8 weight 복제량은 layer당 6.09375 GiB, 전체 요청 weight H2D는 195 GiB다.
MRAM weight는 DPU당 layer 하나의 3.046875 MiB만 상주하며 scratch가 추가된다. 이는 최초 기준
구현의 비용이며, layer 교체가 weight 전송 자체를 없애는 것은 아니다.

기존 `prepare_block` 예제는 기존 lifecycle을 유지한다. 새 session 실험은
독립 프로세스에서 실행한다. 이전 API가 남긴 implicit DPU 할당이 있는
프로세스에서는 새 session이 소유권 충돌을 진단한다.

backend 환경과 하드웨어 설정을 준비한 뒤 사용자가 직접 실행한다:

```bash
cd /home/dlrkdals/TritonPIM/triton_shared/pim_backend/torch_backend/examples
python e2e/torch_llama_ttft.py \
  --checkpoint /path/to/models--meta-llama--Llama-2-7b-hf/snapshots/SNAPSHOT \
  --prompt-file e2e/prompt.txt
```

`prompt.txt`는 tokenizer의 특수 token을 포함해 1024 token 이상인 일반 텍스트다.
처음 1024 token을 사용한다. 실행 결과는 `examples/e2e/llama_int8_ttft_report/`에 저장된다.

| 결과 | 의미 |
|---|---|
| `request_fx.txt`, `.dot` | 실제 load/run 노드와 layer 사이 token/activation 의존성 |
| `layer_N_export.txt`, `.dot` | GEMM shape와 weight 출처를 표시한 원본 FX |
| `layer_N_partitions.txt`, `.dot` | PIM GEMM/CPU partition 실행 그래프 |
| `report.json` | 준비 시간, TTFT, layer별 load/run 시간, 전송량, dtype과 양자화 설정 |

`ttft_ms`는 tokenization 시작부터 첫 token ID의 `.item()`까지다. 자원 할당,
layer별 capture/packing/upload, activation 전송, 계산, CPU 후처리를 포함한다.
파일 읽기, 모델 로딩, weight 양자화, export/컴파일, 첫 token 이후 free는
제외한다. `request_ms_including_close`에는 free까지 포함된다.
`weight_load_ms` 안에 `capture_ms`와 `upload_ms`가 포함되므로 중복 합산하지 않는다.
capture에는 native packing 등이, upload에는 최초 program load와 weight H2D가
포함된다. `upload_breakdown_ms`는 upload 시간을 실제 SDK `h2d`, `program_load`,
나머지 `resident_host`로 나눈다. run 시간은 layer의 CPU 연산과 host 처리, 전송,
PIM 연산 전체다. TTFT 보고서의 `activation_transfer_policy`는 `always`이며
각 layer의 `run_transfers.activation_hits`는 0이다. 보고서는
`gemm_dtype="int8 x int8 -> int32"`, `cpu_dtype="float32"`,
`lm_head_scope="last_token_only"`, `accuracy_checked=false`를 기록한다.
Resident H2D v4의 상세 정의는 [실험 안내](examples/README.md)에 있다.

측정 중 Dynamo 재컴파일은 오류로 처리한다. 측정 후 allocation/free 1회씩,
2272번의 실제 PIM launch, 모든 layer의 run 구간 weight H2D=0을 검사한다.
CPU KV cache는 layer당 `[1,32,1024,128]` K/V이며 총 1 GiB다. 실행 통계를
저장하고 자원 수명과 전송 카운터를 확인한 뒤
`TTFT execution complete: resources=1/1 launches=2272 run_weight_h2d=0 activation_hits=0 gemm_dtype=int8->int32`을 출력한다.

PIMLLM 원본은 KL calibration과 warm-up 후 `gm.run()` 시간을 사용하며 전체
sequence의 lm_head를 계산한다. 현재 경로는 같은 GEMM dtype의 성능 실험이며,
PIMLLM과 전체 시간을 비교하려면 모델·lm_head 범위와 측정 경계도 맞춰야 한다.

## 현재 지원 범위

- 입력은 2-D CPU `torch.float32` 두 개 또는 `torch.int8` 두 개
- 출력은 각각 CPU `torch.float32` 또는 `torch.int32`
- static M/N/K (`dynamic=False`)
- A/B는 row-major contiguous; 아니면 lowering이 contiguous copy를 삽입
- template 주변 prologue/epilogue fusion은 아직 열지 않음
- CPU fallback 없이 지원 범위를 벗어나면 진단

M/N/K tail은 template의 masked access와 Common scheduling의 local trip count로
표현한다. source split-K divisibility 조건은 없다.
