# Silent-PIM (AiMX) 실험과 실행 방법

AiMX FPGA에서 돌리는 실험의 종류, 각 실험의 입력과 측정 범위, 실행 명령과 결과 위치를 정리한다.
target 구현은 [Silent-PIM 설계](Silent-PIM_Design.md), 모델 session과 상주 정책은
[Model Runtime 설계](ModelRuntime_Design.md)가 다룬다.

| 근거 | 경로 |
| --- | --- |
| 환경, 실험 스크립트 | `scripts/env.sh`, `scripts/README.md`, `scripts/experiments/*.sh` |
| Llama layer (prefill) | `pim_backend/torch_backend/examples/aimx/llama_workloads.py` (`LlamaAimxLayer`, `LlamaAimxAttention`, `PrefillLinear`) |
| Llama layer (decode) | `examples/aimx/llama_decode.py` (`LlamaAimxDecodeLayer`, `CheckpointLinear`) |
| compile, breakdown 공통 | `examples/aimx/experiment_common.py` |
| prefill / decode 측정 | `examples/aimx/e2e/benchmark_e2e.py`, `examples/aimx/e2e/benchmark_decode.py` |
| GEMM, attention, layer | `examples/aimx/gemm/benchmark_gemm.py`, `attention/benchmark_attention.py`, `layer/benchmark_layer.py` |
| kernel 단위 | `pim_backend/examples/silent_pim/` |

---

## 1. 공통 조건

| 항목 | 값 | 근거 |
| --- | --- | --- |
| 장치 | FPGA `--fpgas N`장, `--base-fpga 0`부터 | `experiment_common.add_device_arguments`. `[base, base+N)`이 감지된 장치 수 안이어야 한다 (`backend/compiler.py`) |
| GEMM | BF16 × BF16 → BF16 | AiMX 저장·연산 형식 (`llama_workloads.py` docstring) |
| CPU 부분 | FP32 (RMSNorm, RoPE, softmax, SwiGLU, residual) | `llama_workloads.py` |
| head packing | 컴파일러가 결정 (`--qk-heads 1 --pv-heads 1` 기본) | `torch_backend/fx_packing.py` 규칙 (1)(3), decode는 (4)(5) |
| KV 공유 | `--kv share` (query head h가 KV head h // group을 읽음) | `llama_workloads.py` |
| autotune | stage class마다 최대 `--max-benchmarks`(50)회, 실행마다 `--cache` | `experiment_common.compile_session`, `scripts/experiments/*.sh` |
| CPU thread | 8 | `torch.set_num_threads(8)` |
| lm_head | 마지막 행만 CPU FP32 | `benchmark_e2e.py`, `benchmark_decode.py` |
| prompt | `examples/upmem/e2e/prompt.txt` 앞에서 N token | 실험 스크립트 |

실행 환경은 `source triton_shared/scripts/env.sh`가 만든다 (`conda activate pim_triton`, `MODELS_DIR`,
AiMX SW 경로). `scripts/experiments/*.sh`는 이 파일을 직접 불러오며, 결과를 `$ROOT/results/` 아래에 둔다
(`ROOT` = `triton_shared`의 상위 디렉터리).

---

## 2. 실험 목록

| # | 실험 | 스크립트 | 입력 | 결과 (`$ROOT/results/…`) |
| --- | --- | --- | --- | --- |
| 1, 3 | GEMM 성능, autotune 수렴 | `experiments/run_gemm.sh <model> <fpgas…>` | prefill GEMM (M = 1024) | `gemm/<model>/f<N>/report.json` |
| 4 | Attention | `experiments/run_attention.sh <model> <fpgas…>` | layer 0의 QKV → QKᵀ → softmax → PV, 1024 token | `attention/<model>/f<N>/report.json` |
| 5 | e2e prefill (1024 token) | `experiments/run_e2e_prefill.sh <model> <fpgas…>` | 32 layer | `e2e_prefill/<model>/f<N>/report.json` |
| 6 | prefill (128/256/512/1024) | `experiments/run_prefill.sh <fpgas…>` | llama2/3 × 4 길이, 32 layer | `prefill/<model>/f<N>/s<seq>/` |
| 7 | decode (128/256/512/1024 뒤) | `experiments/run_decode.sh <fpgas…>` | llama2/3 × 4 길이, 128 token 생성 | `decode/<model>/f<N>/s<seq>/` |

모든 실험이 결과 폴더에 `autotune_cache.json`과 `log.txt`를 함께 둔다.

예제 단위의 실험과 검증 스크립트 (`examples/aimx/`):

| 스크립트 | 내용 |
| --- | --- |
| `layer/benchmark_layer.py --residency upload\|resident` | layer 하나: 요청마다 weight upload vs 상주 |
| `verify_llama_layer.py` | layer 하나: AiMX vs HF FP32 |
| `verify_kv_attention.py` | decode attention: 장치 상주 KV를 읽는 QK/PV vs torch BF16 |
| `e2e/verify_decode_model.py` | 모델 전체 decode, KV 상주 |
| `verify_bf16_gemm.py`, `verify_k_alignment.py` | BF16 host GEMM, A의 K pitch 정렬 |
| `benchmark_llama_ttft.py` | 32 layer, 1024 token TTFT와 첫 token |
| `run_aimx_prefill.sh`, `run_aimx_decode.sh` | 실험 6, 7과 같은 실행을 FPGA 2장으로 (`examples/aimx/e2e/compare/`) |

---

## 3. 실험 6: prefill (`run_prefill.sh` → `e2e/benchmark_e2e.py`)

### 3.1 흐름

```text
run_prefill.sh 4 8   →  f4, f8 × llama2, llama3 × 128/256/512/1024 = 16회 (실행마다 python 프로세스 하나)

benchmark_e2e.py --fpgas N --layers 32 --seq S
 1. HF checkpoint 로드 (BF16), HF FP32로 32 layer 기준 출력 계산             [측정 밖]
 2. layer마다 LlamaAimxLayer 생성 (PrefillLinear: [in, out] 연속 BF16 사본 하나), HF layer 해제
 3. compile_model({"prefill": …}, units=N)                                   [측정 밖]
      stage class별 autotune → packing/config 선택 → 32 layer compile → 메모리 배치
 4. session open: FPGA region 할당, slot마다 descriptor chain 생성·업로드        [측정 밖]
 5. 요청 1 + --repeat(3)회
      tokenize → embed → 32 layer (group마다 weight pack/push, PIM GEMM, CPU) → norm/lm_head → argmax
 6. report.json, forward_breakdown.csv, gemm.csv, attention.csv
```

### 3.2 측정 기준: group 0이 미리 올라가 있다고 본 TTFT

weight가 FPGA에 다 들어가지 않으면(groups > 1) 요청마다 모든 group을 다시 올린다. 실험은 group 0의
pack과 push를 뺀 값을 TTFT로 본다. group 1의 pack은 group 0이 실행되는 동안 background로 진행된다
(`runtime.GroupLoader`). 이 값은 첫 요청과 이후 요청에서 같은 방식으로 계산되며, 이후 요청 3개의 중앙값을 쓴다.

```text
요청:  [pack g0 | push g0 | run g0 (‖ pack g1) | push g1 | run g1 ...]
TTFT = ttft_ms − group0_load_ms   (= ttft_without_group0_load_ms)
```

### 3.3 결과 파일

| 파일 | 행 | 주요 열 |
| --- | --- | --- |
| `forward_breakdown.csv` | 요청별 (iter 0 = 첫 요청), `warm_mean`, `warm_median` | `ttft_ms`, `group0_load_ms`, `ttft_without_group0_load_ms`, `first_token_ms`, `layers_ms`, `h2d_ms`/`pim_ms`/`d2h_ms`/`cpu_ms`와 비율, `detail_*` |
| `gemm.csv` | weight GEMM stage별 (q/k/v/o, gate/up, down. packing 후 이름) | `calls`, `wall_ms`, native 8칸, `other_ms` (32 layer 합, 이후 요청 평균) |
| `attention.csv` | activation GEMM stage별 (QKᵀ, PV) | 같음 |
| `report.json` | — | `memory` (slots, groups), `config_report`, `stages`, `calls`, `summary` |
| `log.txt` | — | 선택된 config, `matches=`, `logits_cosine=` |

PIM-LLM 항목: `h2d` = weight pack + push + `desc_h2d` + `h2d`, `pim` = `invoke`, `d2h` = `d2h`,
`cpu` = `first_token_ms`에서 나머지. `detail_*` 합은 각 요청 행의 `ttft_ms`다. `warm_median`은 열마다 따로
고른 중앙값이라 합이 맞지 않을 수 있다. breakdown을 group 0 기준으로 보려면 `h2d_ms`와
`detail_weight_upload_ms`에서 `group0_load_ms`를 뺀다.

native 단계 (`backend/driver.py` `_STAGE_NAMES["aimx"]`): `alloc`, `build`, `desc_h2d`, `layout_in`, `h2d`,
`invoke`, `d2h`, `layout_out`.

---

## 4. 실험 7: decode (`run_decode.sh` → `e2e/benchmark_decode.py`)

### 4.1 흐름

AiMX session은 phase를 하나만 배치하므로 (`aimx_model.place`) decode는 decode phase만 compile한다.
prompt는 HF 모델이 CPU에서 prefill해 KV를 채운다 (측정 밖).

```text
benchmark_decode.py --fpgas N --prompt-tokens S --new-tokens 128
 1. HF 로드, prompt S token을 CPU prefill → KV cache (BF16, capacity S + 128)       [측정 밖]
 2. CPU 기준 생성 (같은 decode layer를 eager로)                                      [측정 밖]
 3. compile_model({"decode": …}, repeats={"decode": 127}, kv_inputs=…)              [측정 밖]
 4. session open, prompt KV를 장치에 write_kv
 5. free run 127 step   ← step_breakdown.csv의 대상
      step = embed → 32 decode layer (GEMV, KV 상주 QK/PV는 PIM, 최근 16 위치는 CPU)
             → 새 위치를 host cache에 기록 (16 step마다 장치로 전송) → norm/lm_head → argmax
      step 0 = decode weight upload + prompt KV 전송 + layout 대조 → 평균에서 제외
 6. teacher-forced run 127 step (정확도 비교용, 측정 안 함)
 7. step_breakdown.csv, gemm.csv, attention.csv
```

### 4.2 결과 파일

| 파일 | 행 | 주요 열 |
| --- | --- | --- |
| `step_breakdown.csv` | step별 + `mean_after_first` | `step_ms`, `layers_ms`, `h2d_ms`/`pim_ms`/`d2h_ms`/`cpu_ms`와 비율, `detail_cpu_outside_layers_ms`, `detail_weight_upload_ms`, `detail_kv_push_ms`, `detail_run_*`, `detail_cpu_layers_ms` |
| `gemm.csv` | weight GEMV stage별 | 첫 step을 뺀 step 평균 |
| `attention.csv` | KV 상주 QK, PV stage별 | 같음 |
| `log.txt` | — | `free: match=`, `forced: argmax match=`, `rel_l2` |

PIM-LLM 항목은 UPMEM과 같은 규칙이다: `h2d` = weight upload + `desc_h2d` + `h2d`, KV 전송(`kv_push`)은
`cpu`에 들어가고 `detail_kv_push_ms`로 따로 본다. `detail_*` 합은 각 step의 `step_ms`다.

---

## 5. 실행 순서 (FPGA 4/8장 서버)

```bash
# 1) 코드
cd <ROOT>/triton_shared && git checkout experiment && git pull origin experiment
# 2) Triton + triton_shared + pim_backend 다시 설치 (시스템 gcc, pim_triton env)
cd <ROOT> && ./triton_shared/scripts/3_build_triton.sh
# 3) 이전 결과와 autotune cache 삭제 (남아 있으면 탐색 없이 재사용)
rm -rf results/prefill results/decode
# 4) 실행 (같은 FPGA를 쓰므로 순서대로)
./triton_shared/scripts/experiments/run_prefill.sh 4 8
./triton_shared/scripts/experiments/run_decode.sh 4 8
```

`3_build_triton.sh`는 `CC=gcc CXX=g++`로 빌드한다. `llvm_toolchain` env는 LLVM 빌드(`2_build_llvm.sh`)에만
쓰고, Triton 빌드에서는 zstd 경로(`CMAKE_PREFIX_PATH`)만 쓴다.

---

## 6. 제약과 확인할 것

| 항목 | 내용 | 근거 |
| --- | --- | --- |
| FPGA 수 | 인자로 준 수보다 감지된 장치가 적으면 실패 | `backend/compiler.py` `base_fpga` 주석 |
| 호스트 RAM | groups > 1이면 group 하나의 weight 전체를 host에서 pack한다. checkpoint(약 13 GB)에 group image가 더해짐 | `runtime.GroupLoader`, `aimx_model._pack_group` |
| decode의 groups | KV를 읽는 stage는 layer끼리 slot을 공유할 수 없다. 32 layer weight가 다 들어가야 실행된다 (f1에서는 불가) | `aimx_model._create_stage` |
| 장치 할당 한도 | FPGA당 runtime heap `0x1FEE00000` (8 GB − 18 MB) | [Silent-PIM 설계](Silent-PIM_Design.md) §1 |
| prefill config 선택 | 요청 하나 기준(D = 1)이라 resident와 reloaded의 비용이 같고, E가 가장 작은 조합이 선택된다 | `model_compile.select_configs` |
| 정확도 | prefill `logits_cosine`, decode `rel_l2`. 짧은 seq(128)는 처음 실행할 때 확인 | `log.txt` |
| 재실행 | 같은 결과 폴더면 cache 재사용. 새로 튜닝하려면 폴더 삭제 | `--cache $out/autotune_cache.json` |

---

## 7. UPMEM과 비교할 때

| 항목 | Silent-PIM | UPMEM ([UPMEM 실험](UPMEM_Experiments.md)) |
| --- | --- | --- |
| GEMM dtype | BF16 | INT8 → INT32 |
| prefill 기준 값 | `warm_median` `ttft_without_group0_load_ms` | `warm_mean` (대개 groups = 1, warm upload 0) |
| decode 기준 값 | `mean_after_first` | `mean_after_first` |
| decode 실행 | HF CPU prefill 뒤 decode만 compile | prefill과 decode를 한 session에서 함께 compile |
| prefill KV | 출력 안 함 | layer마다 CPU에 보관 |
| lm_head | 마지막 행, CPU FP32 | 같음 |
