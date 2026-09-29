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
