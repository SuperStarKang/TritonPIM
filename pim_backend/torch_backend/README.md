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
        └─ stage/h2d/kernel/d2h/combine/host_alloc/host_free median (UPMEM)
        │
        ▼
AutoTuner record/cache
```

실제 실행에서는 같은 template version과 target spec으로 cache를 조회한다.

## UPMEM GEMM 계측: PIM-LLM과 같은 compute/transfer 범위

UPMEM timing version 2는 PIM-LLM의 기존 PGEMM `timer[1/2/3]`에 맞춰
`h2d/kernel/d2h`를 독립적인 시작·종료 타이머로 측정한다. PIM-LLM 코드는
수정하지 않는다. 두 구현 모두 호스트의 monotonic clock으로 재는 경과시간이다.

| 항목 | TritonPIM에서 측정하는 작업 |
|---|---|
| `stage` | DPU 확보/필요시 바이너리 적재, 입력 staging 버퍼 할당·패킹·패딩과 인자 준비 |
| `h2d` | 모든 입력 행렬·메타데이터의 `dpu_prepare_xfer` 루프 + 동기식 `dpu_push_xfer` |
| `kernel` | `dpu_launch(set, DPU_SYNCHRONOUS)` 바로 앞부터 반환 직후까지 |
| `d2h` | 결과의 `dpu_prepare_xfer` 루프 + 동기식 `dpu_push_xfer` |
| `combine` | 결과 배치 복원과 부분합 결합 |
| `host_alloc` | D2H 수신 버퍼의 크기 계산과 할당. 입력 staging 할당은 `stage`에 포함 |
| `host_free` | H2D 이후 입력 버퍼 해제와 결과 결합 이후 수신 버퍼 해제 |

H2D에서 `free()`를, D2H에서 수신 버퍼 `malloc()`을 제외한다. 독립된 타이머를
사용하므로 제외한 작업이 `kernel`이나 다음 round의 transfer에 섞이지 않는다.
시간 배열은 host GEMM 진입 시 한 번만 초기화하고, 모든 scheduling round의
시간을 합산한다. 할당·해제와 계측 비용은 GEMM 전체 호출 시간에 계속 포함된다.
위 항목에 속하지 않는 런처·loop 제어 등의 비용은 전체 호출 시간과 항목 합계의
차이로 남는다.

`h2d/d2h`에는 SDK의 포인터 등록·전송·동기화 비용이 포함되고, `kernel`에는
DPU 내부 DMA와 동기화도 포함된다. 데이터 전송량, 메타데이터 전송 횟수, 타일과
DPU 실행 코드는 이 계측 변경으로 바뀌지 않는다. `stage/combine/host_alloc/host_free`는
TritonPIM의 호스트 작업 구분이며 PGEMM의 preprocess/postprocess와 직접 대응하지 않는다.

반영할 때는 사용자가 translator를 다시 빌드한다:

```bash
cmake --build /home/dlrkdals/TritonPIM/triton/build/cmake.linux-x86_64-cpython-3.12 \
  --target triton-shared-translate
```

Python backend의 compile cache version도 변경했다. 이전 실행과 캐시를 분리하려면
`source set_env.sh` **뒤에** 다음 설정을 적용한 새 프로세스에서 GEMM을 실행한다:

```bash
export TRITON_CACHE_DIR=/home/dlrkdals/TritonPIM/.triton_cache_timing_v2
export TORCHINDUCTOR_CACHE_DIR=/home/dlrkdals/TritonPIM/.inductor_cache_timing_v2
export TRITON_ALWAYS_COMPILE=0
```

driver는 네이티브 host의 timing version을 실행 전에 확인한다. 이전 5개 항목을
내는 host가 발견되면 새 범위로 해석하지 않고 재빌드/캐시 갱신 안내와 함께 중단한다.
새 UPMEM 배열은 기존 5개 항목 뒤에 `host_alloc/host_free`를 추가한 7개 항목이며,
AiMX의 5개 항목은 유지한다. 이전 JSON에 저장된 H2D/D2H/combine은 이전 범위의 값이다.
Attention profiler는 JSON에 `pim_timing_version`도 기록한다.

## Attention 한 번의 실행을 표로 보기

`examples/profile_attention.py`는 Attention 전체를 `torch.compile(fullgraph=True)`로
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
  python triton_shared/pim_backend/torch_backend/examples/profile_attention.py \
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
  `stage/h2d/kernel/d2h/combine/host_alloc/host_free`는 각 PIM 행에 이미 포함된 세부 시간이다.
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
