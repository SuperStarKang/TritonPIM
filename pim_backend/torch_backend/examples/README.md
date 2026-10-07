# TritonPIM workload experiments

새 `benchmark_*.py`는 checkpoint 없이 합성 INT8 weight와 activation으로 연산 성능을
측정한다. batch=1, sequence/KV length=1024, 2048 DPUs, 16 tasklets이며,
Attention layer와 Transformer block은 Llama-2-7B의 hidden/head/FFN 크기를 사용한다.
정확도 비교, calibration, CPU reference 검증은 실행하지 않는다.

| 범위 | 연산 | Head 구성 | PIM GEMM 수 | 상주 projection weight 수 |
|---|---|---|---:|---:|
| Attention kernel | QKᵀ → scale/mask/softmax → PV | 1 × 128 | 2 | 0 |
| Attention layer | Q/K/V projection → RoPE → Attention → O projection | 32 × 128 | 68 | 4 |
| Transformer block | RMSNorm → Attention layer → residual → RMSNorm → SwiGLU FFN → residual | 32 × 128 | 71 | 7 |

Attention kernel의 입력 Q/K/V는 매 호출 전송하는 activation이다. Attention layer와
Transformer block에서는 projection weight만 준비 단계에 전송하며, norm weight,
RoPE 상수와 mask는 CPU에서 사용한다. QKV fusion은 적용하지 않았다.

## 디렉터리

```text
examples/
├── benchmark_common.py              # FX 분석, compile_layer, session, timing/JSON
├── fx_utils.py                      # FX text/DOT 출력
├── attention/
│   ├── benchmark_attention_kernel.py
│   ├── benchmark_attention_layer.py
│   ├── benchmark_model_attention_layer.py # Q/K/V/O projection + batched QK/PV
│   ├── int8_workloads.py
│   ├── torch_attention.py           # 기존 INT8 예제
│   ├── torch_attention_layer.py     # 기존 KV-group 예제
│   ├── torch_resident_attention.py  # 기존 single-head FP32 예제
│   ├── profile_attention.py
│   └── *.json                      # 기존 측정 결과
├── transformer/
│   ├── benchmark_transformer_block.py
│   ├── int8_workloads.py
│   ├── torch_resident_transformer.py
│   ├── torch_resident_llama_layer.py
│   ├── llama_two_layers_report/     # 이동된 스크립트의 이후 실행 결과
│   └── resident_transformer_*.json
├── e2e/
│   ├── torch_llama_ttft.py          # checkpoint 기반 INT8 layer GEMM TTFT
│   ├── benchmark_model_llama_ttft.py # Session-first checkpoint TTFT, baseline/batched 비교
│   ├── prompt.txt
│   └── llama_int8_ttft_report/      # INT8 TTFT 보고서와 FX graph
├── llama_ttft_report/              # 기존 결과: 폴더 이동 권한 오류로 유지
├── llama_two_layers_report/        # 기존 결과: 폴더 이동 권한 오류로 유지
└── torch_mm.py, torch_int_mm.py, ... # GEMM/FX 단위 예제와 실행 wrapper
```

기존 JSON 파일과 prompt 내용은 이동만 했다. 위 두 기존 리포트 폴더는 소유자가
`nobody:nogroup`이며 이동이 `Permission denied`로 거부되어 원래 위치에 남겼다.
기존 Transformer 예제의 로컬 진단 수정도
유지했다. 기존 `torch_attention_layer.py`와 `profile_attention.py`의 독립 실행은
저장소에 없는 `torch_attention_gqa`에 의존한다. 새 세 benchmark는 이 모듈을 사용하지 않는다.

## 실행

기존 PIM 실행 환경에서 `examples/`를 작업 디렉터리로 사용한다.

```bash
cd /home/dlrkdals/TritonPIM/triton_shared/pim_backend/torch_backend/examples
./cmd.sh attention/benchmark_attention_kernel.py
./cmd.sh attention/benchmark_attention_layer.py
./cmd.sh transformer/benchmark_transformer_block.py
```

각각 별도 프로세스로 실행한다. 기본값은 warm-up 1회, 측정 3회, 별도 profile 1회다.
`--warmup`, `--repeat`, `--report-dir`로 반복 횟수와 결과 위치를 지정할 수 있다.
기본 결과 디렉터리는 각 스크립트 옆 `attention_kernel_report/`,
`attention_layer_report/`, `transformer_block_report/`다.

```text
export / FX 분석 / compile_layer      측정 제외, DPU 할당 없음
  ↓
PIMSession.open                     alloc 1회, 별도 측정
  ↓
load_weights                        capture/packing/upload, 별도 측정
  ↓
warm-up                             weight 유지, 측정 제외
  ↓
run_layer × repeat                  매 호출 activation 재전송, weight H2D=0
  ↓
run_layer + partition profile       median과 별개로 1회 측정
  ↓
PIMSession.close                    즉시 free 1회, 별도 측정
```

기존 `compile_layer`는 CPU partition 컴파일을 준비하며 CPU GEMM으로 중간 tensor를
만든다. 이 준비 계산은 `compile_ms`에 포함되며 성능 sample이나 정확도 비교가 아니다.
모든 GEMM의 activation을 매번 전송한다. Q/K/V가 같은 입력을 읽거나 gate/up이
같은 입력을 읽어도 각각 전송하며, activation 내용 비교와 검사용 host 복사본은 만들지 않는다.
MRAM input/output scratch 공간은 계속 공유한다. `activation_hits`는 항상 0이다.
모든 GEMM은 동일한 전체 2048 DPU 집합을 사용한다.
session 실행 중 Dynamo 재컴파일이 필요해지면 오류를 내어 측정에 컴파일 비용이 섞이지 않게 한다.

예상되는 실행 카운터는 다음과 같다. 실제 시간은 실행 결과로 확인한다.

```text
sample=0 wall_ms=... weight_h2d=0 activation_h2d=54525952 activation_hits=0 launches=2
sample=0 wall_ms=... weight_h2d=0 activation_h2d=2818572288 activation_hits=0 launches=68
sample=0 wall_ms=... weight_h2d=0 activation_h2d=4076863488 activation_hits=0 launches=71
resource_opens=1 resource_closes=1
```

## 결과 해석

| 파일/필드 | 내용 |
|---|---|
| `exported_fx.txt/.dot` | 원본 FX와 GEMM별 operation, shape, weight, 배치 설정 |
| `partitioned_fx.txt/.dot` | 실제 실행되는 CPU/PIM partition 연결 |
| `report.json` | 설정, 준비/실행 시간, profile, 전송량, 자원 수명 |
| `activation_transfer_policy` | `always`: 같은 입력도 매 GEMM 전송 |
| `preparation.export_ms`, `compile_ms` | FX 추출·분석 및 PIM/CPU 컴파일 시간 |
| `preparation.session_open_ms` | session 자원 할당 시간 |
| `preparation.weight_load` | weight capture/upload 시간과 준비 단계 전송량 |
| `preparation.weight_load.upload_breakdown_ms` | 최초 upload의 실제 H2D, 프로그램 load, 나머지 host 시간 |
| `samples[].wall_ms`, `median_wall_ms` | weight가 이미 상주한 전체 호출 시간 |
| `samples[].transfers` | 호출별 weight/activation/output/metadata bytes, launch 수 |
| `profile.gemm_totals` | QKᵀ/PV는 head 합계, projection은 연산별 wall/native 시간 |
| `profile.partitions` | 각 CPU/PIM partition의 연산 목록과 시간 |
| `profile.partitions[].h2d_ms` | 실제 weight/activation/metadata 전송 시간; `native_ms.h2d`의 부분합 |
| `session_close_ms`, `session_totals` | free 시간, 준비·warm-up·sample·profile 전체 누적 카운터 |

CPU quantize/dequantize, RoPE, softmax, RMSNorm, SwiGLU와 residual은 호출 시간에
포함된다. CPU fusion이 일어난 partition은 연산 목록과 합산 시간을 기록하며
각 ATen 연산의 시간으로 나누지 않는다. PIM native 시간의 `kernel`은 동기
`dpu_launch` 시간이며 내부 DMA를 포함한다. `h2d/d2h`, `layout_a/b/c`,
`prepare`, `reduction`, buffer 관련 시간은 해당 PIM partition wall time에 이미 포함된다.
별도 profile의 instrumentation 비용을 반복 측정 median에 섞지 않는다.
weight H2D를 생략해도 기존 generated host의 weight packing은 실행 시간에 남는다.

Resident 보고서는 `profile.timing_version=4`를 사용한다. `native_ms.h2d`는
실제 SDK `dpu_prepare_xfer`와 동기식 `dpu_push_xfer[_symbol]` 구간만 측정한다.
`program_load`는 프로그램 교체 시간이며, 동일 ELF를 재사용하면 0이다.
`resident_host`는 이전의 넓은 H2D 구간에서 실제 SDK 전송과 프로그램 교체를
뺀 나머지다. 현재 실행에서는 wrapper 등록·분기 등의 비용이 남으며, activation
전송 생략 검사와 검사용 host scratch 할당·복사는 수행하지 않는다.
세 항목은 서로 겹치지 않는다. CPU layout A/B는 기존 별도 항목을 유지한다.

```text
native_ms.h2d = h2d_ms.weight + h2d_ms.activation + h2d_ms.metadata
기존 H2D 구간 = native_ms.h2d + native_ms.program_load + native_ms.resident_host
```

`h2d_ms`는 `native_ms.h2d`를 나눈 상세 내역이므로 native 단계 합에 다시 더하지 않는다.
`profile.gemm_totals`에도 같은 H2D 부분합을 연산별로 기록한다.
Weight capture 단계는 실제 SDK 전송이 없어 H2D를 누적하지 않는다. 최초 weight
upload는 `upload_breakdown_ms.h2d`에 기록하고, 실행 구간의 weight H2D는 0이다.
K/V/up을 포함한 모든 GEMM에서 activation과 metadata를 전송한다.
Weight capture에 사용하는 hash·비교·복사는 준비 단계에 남아 있다.

기존 version 3 결과의 H2D에는 프로그램 교체와 resident host 작업이 섞여 있다.
기존 JSON은 변경하지 않으며 version 4 breakdown을 얻으려면 새 프로세스에서 재측정한다.
H2D 계측과 activation 전송 정책은 runtime에서 처리하며 compiler lowering과 생성 host의 v3 ABI를 유지한다.
사용자가 다시 실행하면 소스 내용에 따른 runtime/host 라이브러리 cache key가 바뀌어
새 native 라이브러리를 생성한다. 이번 변경을 위해 MLIR compiler를 재빌드할 필요는 없다.

사용자가 확인할 항목은 `activation_transfer_policy="always"`, `activation_hits=0`,
`timing_version=4`, 위 H2D 부분합 관계, K/V/up의 activation 전송량이 양수인지,
`weight_h2d=0`, 기존 launch 수와 `resource_opens=resource_closes=1`이다.
이전 activation 재사용 정책의 v4 JSON도 있으므로 전송 정책을 함께 확인한다.
시간 값 자체는 재실행 결과로 확인한다.

`logical_gemm_gops_per_s`는 논리적 GEMM의 `Σ2MNK / 전체 호출 시간`이다.
padding 추가 계산과 CPU 연산은 분자에서 제외하고 시간에는 포함한다.
gate/up projection의 N=11008은 물리적으로 11264로 padding한다.
기존 lowering과 기본 배치를 그대로 사용하므로 INT8에 최적화된 tuning 결과를 뜻하지 않는다.

고정 activation scale은 1/32, synthetic weight scale은 1/(8√K)이며,
softmax 결과는 행별 최대값/127로 INT8화한다. GEMM은 INT8×INT8→INT32,
그 밖의 연산과 최종 출력은 FP32다. 이 코드는 학습된 Llama의 양자화 모델이나
PIMLLM의 calibration을 재현하지 않으며, PIMLLM 성능 수치는 아직 생성하지 않는다.
향후 비교에서는 같은 연산 범위의 `median_wall_ms`와 profile의 GEMM 시간을 사용한다.
세 benchmark의 시간은 TTFT가 아니다. E2E INT8 경로는 아래와 같이 별도로 실행한다.


## INT8 checkpoint TTFT

`e2e/torch_llama_ttft.py`는 실제 Llama-2-7b-hf checkpoint를 FP32로 읽은 뒤
각 layer의 7개 projection weight를 tensor별 `absmax/128` scale로 INT8화한다.
Weight 양자화와 32개 layer의 FX export/컴파일은 요청 전에 수행한다.
Q/K/V/O, Gate/Up/Down 및 head별 QK/PV는 모두 INT8×INT8→INT32 GEMM이다.
Activation scale은 1/32, probability scale은 행별 max/127로 기존 INT8
attention/block 경로와 같다. RMSNorm, RoPE, softmax, SwiGLU, residual,
embedding과 마지막 token의 lm_head는 CPU FP32다. KV cache도 CPU FP32다.
Calibration과 HF CPU reference 정확도 검증은 수행하지 않는다.

```bash
cd /home/dlrkdals/TritonPIM/triton_shared/pim_backend/torch_backend/examples
./cmd.sh e2e/torch_llama_ttft.py \
  --checkpoint /path/to/models--meta-llama--Llama-2-7b-hf/snapshots/SNAPSHOT \
  --prompt-file e2e/prompt.txt
```

Prompt는 special token을 포함해 최소 1024 tokens가 필요하다. 더 긴 입력은
처음 1024 tokens를 사용한다. 설정은 batch 1, 32 layers, 2048 DPUs, 16 tasklets다.

```text
요청 전: checkpoint 로드 → weight 양자화 → layer FX export / compile
요청: tokenize → session open → CPU embedding
      → [layer weight capture/upload → layer 실행] × 32
      → CPU last-token norm / lm_head / argmax → 첫 token ID
      → session close
```

`ttft_ms`는 tokenization 시작부터 첫 token ID 준비까지다. 요청의 DPU 할당,
layer별 capture/packing/weight upload와 실행을 포함한다. Weight 양자화와
컴파일은 `preparation_ms`에, 첫 token 이후 free는 `resource_close_ms`에 기록한다.
`request_ms_including_close`는 free까지 포함한다. 같은 전체 DPU 집합을 유지하면서
layer마다 weight를 교체하고, 모든 GEMM의 activation을 다시 전송한다.

결과는 `e2e/llama_int8_ttft_report/report.json`과 같은 디렉터리의 FX text/DOT다.
`gemm_dtype="int8 x int8 -> int32"`, `cpu_dtype="float32"`,
`lm_head_scope="last_token_only"`, `accuracy_checked=false`를 기록한다.
Layer별 `weight_load_ms`, `capture_ms`, `upload_breakdown_ms`, `run_ms`와
load/run 전송량을 함께 확인한다. 전체 launch는 2272회이며 각 layer의
`run_transfers.weight_bytes`와 `run_transfers.activation_hits`는 0이어야 한다.
Resource open/close는 각각 한 번이고 FP32 KV cache는 총 1 GiB다.

```text
TTFT_ms=... first_token_id=...
TTFT execution complete: resources=1/1 launches=2272 run_weight_h2d=0 activation_hits=0 gemm_dtype=int8->int32
```

이 경로는 PIMLLM과 GEMM dtype을 맞춘 성능 실험이다. PIMLLM 원본의 KL
calibration을 재현하지 않는다. PIMLLM의 warm-up 후 `gm.run()` 측정 및
전체 sequence의 lm_head 계산과 현재 요청 TTFT / last-token lm_head는 범위가
다르므로 전체 시간 비교에서는 이 경계도 맞춰야 한다.

## 새 LayerPlan: checkpoint E2E 비교

**내용**
[e2e/benchmark_model_llama_ttft.py](e2e/benchmark_model_llama_ttft.py)는 실제 Llama-2-7b-hf
checkpoint와 기존 E2E의 양자화 연산을 사용한다. Batch 1, prompt 1024 tokens, 32 layers,
32 heads, 2048 DPUs, 16 tasklets로 첫 greedy token까지 요청 한 번을 측정한다.
같은 model runtime에서 `baseline`과 `batched` GEMM config를 비교한다.

**근거**
- `PIMSession`이 먼저 DPU를 확보하고 `prepare()`가 32개 LayerPlan을 만든다.
- `model_configs()`와 Transformer benchmark의 Gate/Up zero padding을 재사용한다.
  Gate/Up은 물리 N=11264로 계산한 뒤 논리 크기 11008열을 반환한다.
- 각 block은 `(hidden, 누적 KV)`를 받아 갱신된 hidden과 KV tuple을 반환한다.
  기존 KV tensor는 참조로 전달하며, 모델 실행 후 Session table에 등록한다.
- `run_model()`은 layer 진입마다 weight를 전송한다. 전체 모델의 weight가 동시에
  device에 상주하지 않으며 요청의 `weight_bytes`는 0이 아니다.
- 준비 중 CPU 계산은 partition 컴파일을 위한 것이고 요청 GEMM은 PIM에서 실행한다.
  Mock, calibration, CPU reference 정확도 비교는 사용하지 않는다.

**예시**

기존 PIM 실행 환경에서 checkpoint 경로를 지정한다. Prompt는 special token을 포함해
최소 1024 tokens여야 한다. 두 명령은 별도 프로세스로 실행하며 결과 경로를 구분한다.

```bash
cd /home/dlrkdals/TritonPIM/triton_shared/pim_backend/torch_backend/examples/e2e

../cmd.sh benchmark_model_llama_ttft.py \
  --checkpoint /home/dlrkdals/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
  --prompt-file prompt.txt \
  --gemm-config baseline \
  --report-dir model_e2e_baseline

../cmd.sh benchmark_model_llama_ttft.py \
  --checkpoint /home/dlrkdals/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
  --prompt-file prompt.txt \
  --gemm-config batched \
  --report-dir model_e2e_batched
```

```text
요청 전
  checkpoint 로드 / projection weight INT8 양자화
    → PIMSession.open: DPU 할당 1회
    → prepare: FX 분석 / config·batch 배정 / CPU·PIM 컴파일 / 공통 메모리 배치

요청: TTFT 측정
  tokenize → CPU embedding
    → [layer weight upload → 준비된 CPU/PIM partition 실행 → hidden/KV 반환] × 32
    → CPU KV를 Session table에 등록
    → CPU last-token norm / lm_head / argmax → 첫 token ID

요청 후
  PIMSession.close: DPU 해제 1회 → JSON / layer별 FX text·DOT 저장
```

| 설정 | Layer stage 수 | QK/PV의 p_b | Layer당 launch | 전체 launch |
|---|---:|---:|---:|---:|
| model + baseline config | 9 | 1 | 71 | 2272 |
| model + batched config | 9 | 32 | 9 | 288 |

```text
gemm_config=batched prepare_ms=... expected_launches=288
TTFT_ms=... first_token_id=... launches=288 weight_bytes=... resources=1/1
```

`request_transfers.device_compiles=0`, `activation_hits=0`, alloc/free 각각 1회,
KV 32쌍과 총 `1073741824` bytes, 계획한 launch 수 및 layer weight 전송량을 확인한다.
시간과 token ID는 실측값이며 `accuracy_checked=false`다. Warm-up/repeat/median은 없고
프로세스마다 요청 한 번이다. 코드 작성 시 빌드나 하드웨어 실행은 수행하지 않았다.

| JSON 항목 | 측정 범위 |
|---|---|
| `prepare_ms` | 자원 확보 후 Session의 FX 분석·CPU/PIM 컴파일·배치 준비 |
| `preparation_ms` | checkpoint/tokenizer 로드 시작부터 준비 완료까지, 자원 할당 포함 |
| `ttft_ms` | 준비된 Session에서 tokenize부터 첫 token까지, 32개 layer weight upload 포함 |
| `model_to_token_ms` | tokenization 완료부터 첫 token까지 |
| `setup_to_token_ms` | checkpoint/tokenizer 로드 시작부터 첫 token까지 |
| `resource_open_ms`, `resource_close_ms` | 자원 할당·해제, `ttft_ms`에서 제외 |
| `request_transfers` | 해당 요청의 weight/activation/output bytes, compile/load/launch 수 |
| `preparation_transfers`, `session_totals` | 준비 완료 시점 및 Session 종료 시점 누적 counter |
| `layer_plans` | layer별 config, instance/p_b/round, Execution Tile, 메모리, 예상 launch |

두 새 실험의 `ttft_ms`를 먼저 비교한다. 기존 `torch_llama_ttft.py`의 TTFT에는
자원 할당과 요청 중 weight capture가 포함된다. 새 경로는 자원 할당과 실행 준비를
요청 전에 완료하므로 기존 TTFT와 직접 나눈 값을 batching만의 효과로 해석하지 않는다.
기존 Transformer 단독 benchmark의 `median_wall_ms`는 weight가 이미 상주한 한 Layer의
실행 시간이며, 이번 모델 전체 TTFT와 측정 범위가 다르다.

## 새 LayerPlan: Attention과 Transformer 비교

**내용**
새 실험은 `PIMSession 진입 → prepare(Torch blocks) → LayerPlan 생성` 순서로 준비하고 실제 UPMEM에서 실행한다.
기존 benchmark와 같은 seed, INT8 workload, 2048 DPUs, 16 tasklets를 사용한다.
Weight를 한 번 적재한 뒤 Layer 실행만 반복 측정하며, 준비와 weight 적재 비용은 별도 기록한다.

**근거**
- 기존 baseline의 `median_wall_ms`도 이미 weight가 상주한 Layer 실행 범위다.
- `session.run_model()`은 매 호출 weight를 다시 적재하므로, 이 실험에서는 Session의 weight loader를 한 번 호출하고 준비된 `plan.run_model()`을 반복한다.
- Transformer의 Gate/Up은 기존과 동일하게 N=11264로 계산하고 11008열만 반환한다. 추가한 padding 열의 weight는 0이며 논리 FFN 크기는 변하지 않는다.
- Mock 실행과 자동 CPU 대체 경로는 추가하지 않는다. 기존과 마찬가지로 수치 reference 검증은 수행하지 않고 `accuracy_checked=false`로 기록한다.

**예시**

| 파일 | 역할 |
|---|---|
| [model_benchmark_common.py](model_benchmark_common.py) | 새 Session 준비, 실행 측정, 실제 partition profile, JSON/FX 출력 |
| [attention/benchmark_model_attention_kernel.py](attention/benchmark_model_attention_kernel.py) | 1/32-head Attention; 동일 workload의 legacy/model 선택 |
| [attention/benchmark_model_attention_layer.py](attention/benchmark_model_attention_layer.py) | Q/K/V/O projection과 32-head batched QK/PV를 포함한 Attention Layer |
| [transformer/benchmark_model_transformer_block.py](transformer/benchmark_model_transformer_block.py) | 7개 resident projection과 batched QK/PV를 포함한 Transformer |

기존 baseline을 실행한 동일한 장비 환경에서, `examples/` 기준으로 실행한다.
기본 반복은 warm-up 3회와 sample 10회다. 각 명령은 별도 프로세스로 실행한다.

```bash
# 이미 측정한 baseline_kernel_1head와 비교: 같은 head 수와 GEMM config
python attention/benchmark_model_attention_kernel.py \
  --heads 1 --gemm-config baseline --warmup 3 --repeat 10 \
  --report-dir attention/model_kernel_1head

# 이미 측정한 baseline_block과 비교: 32 heads를 묶는 config
python transformer/benchmark_model_transformer_block.py \
  --gemm-config batched --warmup 3 --repeat 10 \
  --report-dir transformer/model_block_batched

# Config 변경 효과를 분리하는 대조군: 새 runtime + 기존 GEMM config
python transformer/benchmark_model_transformer_block.py \
  --gemm-config baseline --warmup 3 --repeat 10 \
  --report-dir transformer/model_block_same_config
```

`--gemm-config baseline`은 새 runtime에서도 head 하나에 2048 DPUs를 쓴다.
`batched`는 head 하나에 64 DPUs를 배정하고 32 heads를 한 batch round에 처리한다.
Projection config와 물리 padding은 두 설정에서 같다.

## 32-head Attention batching 실험

**내용**
Head batching 효과는 32-head legacy와 32-head model을 비교한다.
이미 측정한 1-head latency를 32-head latency와 직접 비교하지 않는다.
`--runtime legacy`는 기존 실제 UPMEM benchmark 함수를 동일한 32-head 입력으로 호출한다.

**근거**
- 두 경로 모두 `AttentionKernel(32)`와 seed=0의 같은 입력을 사용한다.
- 새 config는 QK와 PV 각각 하나의 batch round를 구성한다. 아래 후보는 최적값으로 측정된 설정이 아니다.

**예시**

```bash
python attention/benchmark_model_attention_kernel.py \
  --runtime legacy --heads 32 --gemm-config baseline \
  --report-dir attention/baseline_kernel_32heads

python attention/benchmark_model_attention_kernel.py \
  --runtime model --heads 32 --gemm-config batched \
  --report-dir attention/model_kernel_32heads
```

Block=(16,16,32), Group=(1,1,1), p=(8,8,1), p_b=32일 때:

| Stage | Tile | Execution Tile | Launch 수 |
|---|---|---|---:|
| QK | (8,8,4) | (128,128,128) | 1 |
| PV | (8,1,32) | (128,16,1024) | 1 |

## Projection을 포함한 32-head Attention Layer

**내용**
`benchmark_model_attention_layer.py`는 Q/K/V projection, RoPE, QK/softmax/PV,
O projection을 포함한다. 기존 `benchmark_attention_layer.py`와 동일한
`AttentionLayer`, seed=0, 합성 INT8 weight 및 FP32 입력을 사용하고 실제 UPMEM에서 실행한다.
Batch=1, sequence=1024, hidden=4096, heads=32, head_dim=128로 고정한다.

**근거**
- `benchmark_model_attention_kernel.py`는 이미 계산된 Q/K/V를 입력받으므로 projection을 포함하지 않는다.
- 새 실험은 기존 Attention Layer 범위에 맞춰 O projection도 포함한다. Norm, FFN, residual은 이 Layer 범위에 없다.
- Q/K/V/O weight 네 개는 준비 후 한 번 적재하고 반복 실행에서 유지한다.
  네 projection의 GEMM 시간은 sample에 포함되며 최초 weight upload는 별도로 기록한다.
- Q/K/V projection은 각각 실행한다. QK 32개와 PV 32개만 독립 instance로 묶는다.
- 각 projection은 전체 2048 DPUs를 사용한다. Batched QK/PV는 head당 64 DPUs로 32 heads를 동시에 처리한다.
- Checkpoint/tokenizer는 사용하지 않는다. 수치 정확도 비교와 mock 실행은 수행하지 않는다.

**예시**

```text
FP32 hidden [1,1024,4096]
  → CPU quantize
  → PIM Q / K / V projection                 3 launches
  → CPU dequantize / reshape / RoPE / quantize
  → PIM QK: 32 heads, p_b=32                 1 launch
  → CPU scale / causal mask / softmax / quantize
  → PIM PV: 32 heads, p_b=32                 1 launch
  → CPU dequantize / head 결합 / quantize
  → PIM O projection                        1 launch
  → CPU dequantize → FP32 [1,1024,4096]
```

`attention/`에서 각 명령을 별도 프로세스로 실행한다. Head 수는 32로 고정되어
`--heads` 인자는 사용하지 않는다. 세 실험의 warm-up/sample 횟수를 동일하게 지정한다.

```bash
cd /home/dlrkdals/TritonPIM/triton_shared/pim_backend/torch_backend/examples/attention

# 새 runtime + batched QK/PV + Q/K/V/O projection
../cmd.sh benchmark_model_attention_layer.py \
  --gemm-config batched --warmup 3 --repeat 10 \
  --report-dir model_attention_layer_batched

# 같은 새 runtime에서 config 변경 효과를 비교
../cmd.sh benchmark_model_attention_layer.py \
  --gemm-config baseline --warmup 3 --repeat 10 \
  --report-dir model_attention_layer_baseline

# 기존 runtime의 같은 Attention Layer 범위
../cmd.sh benchmark_attention_layer.py \
  --warmup 3 --repeat 10 \
  --report-dir legacy_attention_layer_32heads
```

| 측정 범위 | 논리 GEMM 수 | 새 runtime stage 수 | 호출당 launch 수 |
|---|---:|---:|---:|
| Attention kernel, 32 heads, batched | 64 | 2 | 2 |
| Attention Layer, baseline config | 68 | 6 | 68 |
| Attention Layer, batched config | 68 | 6 | 6 |
| 기존 Attention Layer | 68 | 해당 없음 | 68 |

```text
scope=attention_layer heads=32 gemm_config=batched logical_gemms=68 planned_stages=6
sample=0 wall_ms=... weight_h2d=0 activation_h2d=... launches=6 program_loads=...
attention_layer: median_wall_ms=... launches=6 logical_GEMM_GOPS=...
  q_proj: calls=1 instances=1 wall_ms=...
  k_proj: calls=1 instances=1 wall_ms=...
  v_proj: calls=1 instances=1 wall_ms=...
  QK^T: calls=1 instances=32 wall_ms=...
  PV: calls=1 instances=32 wall_ms=...
  o_proj: calls=1 instances=1 wall_ms=...
resource_opens=1 resource_closes=1
```

결과는 각 디렉터리의 `report.json`, `exported_fx.*`, `partitioned_fx.*`에 저장한다.
`median_wall_ms`로 같은 Attention Layer 범위를 비교하고,
`profile.gemm_totals`에서 projection과 QK/PV 시간을 구분한다.
Profile 시간은 sample median과 별도로 측정한 값이다. 기존 kernel-only 결과와
전체 Layer 시간을 직접 나눈 값을 speedup으로 해석하지 않는다.
실행 시 `resident_weights=4`, `hidden_dim=4096`, `ffn_dim=null`, sample의
`weight_bytes=0`, `device_compiles=0`, 계획한 launch 수와 alloc/free 1/1을 확인한다.
코드 작성 시 빌드나 하드웨어 실행은 수행하지 않았다.

## 결과 비교와 검증 상태

**내용**
각 결과 디렉터리의 `report.json`에서 `median_wall_ms`를 비교한다.
Speedup은 `baseline median_wall_ms / model median_wall_ms`로 계산한다.
이 실험 코드 작성 과정에서는 빌드·실행을 하지 않았다. 아래 값은 실행 시 확인할 기대값이다.

**근거**
- Warm-up과 별도 profile 실행은 sample median에서 제외한다.
- Session 준비 시간에 FX export와 plan 분석을 포함하고, weight 적재와 alloc/free 시간은 별도로 기록한다.
- 반복 실행에서 weight H2D=0, 추가 DPU ELF compile=0, 계획한 launch 수를 확인한다.
- 저장된 baseline은 CPU thread 수가 48이다. 새 JSON의 `cpu_threads`, 실제 target, dtype, head 수를 함께 비교한다.

**예시**

| 실험 | 실행당 기대 launch 수 | 비교할 baseline |
|---|---:|---|
| 새 runtime, 1-head, baseline config | 2 | `attention/baseline_kernel_1head/report.json` |
| 새 runtime, Transformer, baseline config | 71 | `transformer/baseline_block/report.json` |
| 새 runtime, Transformer, batched config | 9 | 같은 Transformer baseline |
| Legacy 32-head Attention | 64 | 새로 측정 |
| Model 32-head Attention, batched config | 2 | Legacy 32-head Attention |

```text
sample=0 wall_ms=... weight_h2d=0 activation_h2d=... launches=9 program_loads=...
transformer_block: median_wall_ms=... launches=9 logical_GEMM_GOPS=...
resource_opens=1 resource_closes=1 report=.../report.json
```

| JSON 항목 | 해석 |
|---|---|
| `stages` | 선택한 config, instance 수, p_b, round, 실제 참여 unit 수 |
| `preparation.prepare_ms` | Session 자원 확보 후 FX export/분석, config/주소 확정과 PIM/CPU 컴파일 전체 |
| `preparation.prepare_scope` | 준비 시간에 포함된 단계 |
| `preparation.weight_load` | 최초 weight packing/upload 시간과 전송량 |
| `samples[].transfers` | 호출별 전송 bytes와 compile/load/launch 수 |
| `profile.gemm_totals` | QK/PV는 instance 전체를 처리한 stage 시간 |
| `profile.partitions` | 실제 CPU/PIM partition별 wall time과 native timing |
| `unique_device_artifacts` | 준비된 stage들이 사용하는 고유 ELF 경로 수 |
| `session_totals.device_compiles` | Cache hit을 제외하고 이번 Session에서 새로 만든 ELF 수 |

새 profile의 native 값은 각 stage library에서 직접 읽는다. 기존 driver의 마지막 실행 timing을 재사용하지 않는다.
Session-first report에서는 기존 `preparation.export_ms`, `plan_ms`를 별도로 기록하지 않는다.
이전 model report와 준비 비용을 비교할 때는 이전의 `export_ms + plan_ms + prepare_ms`와 새 `prepare_ms`를 비교한다.
반복 실행의 `median_wall_ms` 측정 범위는 동일하게 weight 적재 이후의 Layer 실행이다.
Generated model host의 `native_ms.kernel`에는 DPU 활성 metadata 전송이 포함되고,
프로그램 load는 native timing 바깥의 partition wall time에 포함된다.
따라서 기존 resident v4의 `kernel/program_load/resident_host` 필드와 동일한 경계로 해석하지 않는다.
전체 효과는 우선 `median_wall_ms`, stage wall time, 전송 bytes로 비교한다.
