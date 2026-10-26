# UPMEM 실험과 실행 방법

UPMEM에서 돌리는 실험의 종류, 각 실험의 입력과 측정 범위, 실행 명령과 결과 위치를 정리한다.
target lowering은 [UPMEM Target 구현](UPMEMTarget_Implementation.md), 모델 session과 상주
정책은 [Model Runtime 설계](ModelRuntime_Design.md)가 다룬다.

| 근거 | 경로 |
| --- | --- |
| 실행 wrapper | `pim_backend/torch_backend/examples/upmem/cmd.sh` |
| prefill breakdown | `examples/upmem/run_upmem.py`, `examples/upmem/upmem_workloads.py` |
| prefill + decode | `examples/upmem/e2e/benchmark_model_llama_generate.py`, `e2e/llama_generation.py` |
| Llama layer | `examples/upmem/e2e/llama_workloads.py` (`LlamaPrefillLayer`), `e2e/benchmark_model_llama_ttft.py` (`ModelPrefillLayer`) |
| 길이 sweep | `examples/upmem/run_upmem_compare.sh`, `examples/upmem/summarize_compare.py` |
| block 단위 | `examples/upmem/attention/`, `examples/upmem/transformer/`, `examples/upmem/model_benchmark_common.py` |
| kernel 단위 | `pim_backend/examples/upmem/` |

---

## 1. 공통 조건

| 항목 | 값 | 근거 |
| --- | --- | --- |
| 장치 | DPU 2048개, tasklet 16 | `run_upmem.py` `RUNTIME_OPTIONS = {"nr_tasklets": 16}`, `--units 2048` |
| GEMM | INT8 × INT8 → INT32 (W8A8) | weight는 출력 채널별 absmax/127, activation은 token별 absmax/127 (`e2e/llama_workloads.py`) |
| CPU 부분 | FP32, Inductor C++ (`cpu_backend="cpp"`) | `run_upmem.py` `INDUCTOR_OVERRIDES` |
| WRAM | `--local-capacity-limit-bytes 46680` (Block Group buffer, 나머지는 tasklet stack) | `run_upmem_compare.sh` |
| CPU thread | `OMP_NUM_THREADS=8` | `run_upmem_compare.sh` |
| lm_head | 마지막 행만 CPU FP32 | `upmem_workloads.run_request` |
| 실행 | `./cmd.sh <script> …` = `sudo -E … UPMEM_BACKEND=hw LD_LIBRARY_PATH=<override> python …` | `cmd.sh` (LD_LIBRARY_PATH의 override 경로는 서버마다 맞춘다) |

`conda activate pim_triton` 후 `examples/upmem/`에서 실행한다.

---

## 2. 실험 목록

| # | 실험 | 스크립트 | 입력 | 결과 |
| --- | --- | --- | --- | --- |
| U1 | GEMM / GEMV kernel | `pim_backend/examples/upmem/{gemm,gemv,autotune_gemm,autotune_gemv,benchmark_gemv}.py` | shape 인자 | stdout, autotune cache |
| U2 | Attention kernel | `attention/benchmark_model_attention_kernel.py --heads 1\|32` | 합성 INT8, seq 1024 | `attention/model_kernel_*/report.json` |
| U3 | Attention layer | `attention/benchmark_model_attention_layer.py` | 합성, Q/K/V/O projection 포함 | `attention/model_attention_layer_*/` |
| U4 | Transformer block | `transformer/benchmark_model_transformer_block.py` | 합성, projection 7개 상주 | `transformer/model_block_*/` |
| U5 | Llama prefill TTFT breakdown | `run_upmem.py llama --seq S` | checkpoint, prompt S token | `report.json`, `forward_breakdown.csv`, `gemm.csv` |
| U6 | Llama prefill + decode | `e2e/benchmark_model_llama_generate.py --prompt-tokens S` | checkpoint, S token + 127 step | `report.json`, `step_breakdown.csv`, `gemm.csv` |
| U7 | 길이 sweep (U5 + U6) | `run_upmem_compare.sh` | llama2/3 × 128/256/512/1024 | `e2e/compare/<model>_s<seq>_{prefill,decode}/`, `summary.md` |
| U8 | 정확도 | `e2e/verify_llama_layer.py`, `e2e/check_llama_accuracy.py`, `e2e/check_llama_generate_cpu.py`, `e2e/verify_decode_gemv.py` | checkpoint 또는 U6 report | stdout, report |

U2~U4의 `--gemm-config`는 `baseline`(head 하나에 DPU 2048개)과 `batched`(head당 64 DPU,
32 head를 한 batch round)를 고른다 (`model_benchmark_common.add_benchmark_arguments`).
`--warmup`, `--repeat`, `--report-dir`도 같은 곳에서 받는다.

---

## 3. U5: prefill TTFT breakdown (`run_upmem.py`)

### 3.1 흐름

```text
checkpoint 로드 (FP32) → 32 ModelPrefillLayer (INT8 weight)
→ compile_model({"prefill": …}, target="upmem", units=2048)      [측정 밖]
→ session open                                                  [측정 밖]
→ 요청 --repeat 회 (0번 cold, 1번부터 warm)
     tokenize → embed → 32 layer (layer마다 KV를 CPU에 보관) → norm/lm_head(마지막 행) → argmax
→ report.json, forward_breakdown.csv, gemm.csv
```

`--repeat`는 2 이상이어야 한다 (cold 1 + warm 1 이상). 측정 중 재compile이 일어나면 오류로
멈춘다 (`torch._dynamo.config.patch(error_on_recompile=True)`).

### 3.2 breakdown

요청마다 `PartitionTimer`가 top-level partition의 wall time과 PIM 호출의 native 단계를 잰다.

| 항목 | 내용 |
| --- | --- |
| `cpu_ms` | tokenize + embedding + lm_head + CPU partition |
| `weight_upload_ms` | group weight pack + push (cold 요청, 또는 groups > 1) |
| `program_load_ms`, `kv_push_ms` | DPU program load, KV 전송 |
| `h2d_ms`, `kernel_ms`, `d2h_ms`, `reduction_ms` | native 단계 |
| `host_layout_ms` | 전송·launch·합산을 뺀 native 단계 (host pool 준비, layout 변환) |
| `invoke_outside_native_ms`, `other_ms` | invoke 중 native 밖, 나머지 |

`forward_breakdown.csv`는 요청마다 한 행과 `warm_mean` 행이다. `first_token_ms`(embedding부터
첫 token까지, tokenize 제외)를 PIM-LLM 항목 h2d(weight upload 포함) / d2h / pim(kernel) /
cpu(나머지)로 나누고, 그 뒤에 `detail_*`로 모든 breakdown 항목을 둔다 (합 = `ttft_ms`).
`gemm.csv`는 PIM partition마다 warm 요청의 평균이다.

### 3.3 실행

```bash
cd examples/upmem
export OMP_NUM_THREADS=8
./cmd.sh run_upmem.py llama --checkpoint <Llama-2-7b> --seq 1024 --repeat 3 \
    --local-capacity-limit-bytes 46680 --report-dir e2e/llama2_s1024_prefill
```

---

## 4. U6: prefill + decode (`benchmark_model_llama_generate.py`)

### 4.1 흐름

prefill과 decode를 **한 session에서 함께 compile**한다 (`repeats={"decode": max_new_tokens - 1}`).
decode config는 요청 하나에서 decode를 그만큼 반복하는 비용으로 고른다.

```text
compile_model({"prefill": …, "decode": …})                       [측정 밖]
→ 요청 하나: prefill (PIM) → 첫 token → decode step × (max_new_tokens − 1)
     step = embed → decode layer (KV는 MRAM 상주, tail 위치는 CPU) → host cache 쓰기
            → TAIL step마다 cache 전송 대기열 → norm/lm_head → argmax
     첫 step: decode weight upload (prefill weight를 덮어씀) + prefill KV 전송 + layout 대조 1회
→ report.json, step_breakdown.csv, gemm.csv
```

### 4.2 결과

| 파일 | 내용 |
| --- | --- |
| `step_breakdown.csv` | step마다 한 행 + `mean_after_first`. PIM-LLM 항목과 `detail_*` |
| `gemm.csv` | step마다 PIM partition (첫 step 제외 평균) |
| `report.json` | `first_token_ms`, `step_ms`, `token_ids`, `transfers`, `decode_step_breakdown` |

### 4.3 실행

```bash
./cmd.sh e2e/benchmark_model_llama_generate.py --checkpoint <ckpt> --prompt-file e2e/prompt.txt \
    --prompt-tokens 512 --max-new-tokens 128 --local-capacity-limit-bytes 46680 \
    --report-dir e2e/llama2_s512_decode
```

---

## 5. U7: 길이 sweep

`run_upmem_compare.sh`는 llama2, llama3 × 128/256/512/1024 token마다 U5(`--repeat 3`)와 U6을
돌리고, 마지막에 `summarize_compare.py`로 PIM-LLM 결과와 함께 표를 만든다.

```bash
cd examples/upmem
./run_upmem_compare.sh          # 스크립트 안의 CHECKPOINTS 경로를 서버에 맞춘다
python summarize_compare.py --ours e2e/compare --pimllm <PIM-LLM>/results_run_llm
```

| 결과 | 위치 |
| --- | --- |
| prefill | `e2e/compare/<model>_s<seq>_prefill/{report.json,forward_breakdown.csv,gemm.csv,run.log}` |
| decode | `e2e/compare/<model>_s<seq>_decode/{report.json,step_breakdown.csv,gemm.csv,run.log}` |
| 비교표 | `e2e/compare/{summary.md,compare_prefill.csv,decode.csv}` |

`summarize_compare.py`의 prefill 비교는 둘 다 tokenize를 뺀 값이다 (PIM-LLM은 측정한 forward 평균,
TritonPIM은 `warm_mean` 행). PIM-LLM은 lm_head를 모든 prompt 행에 대해 DPU에서 계산하고 TritonPIM은
마지막 행만 CPU에서 계산하므로, layers 열은 decoder layer만 비교한다.

`run_upmem_llama.sh`는 1024 token prefill만 llama2/3에 대해 돌린다 (`e2e/<model>_ttft_upmem/`).

---

## 6. U8: 정확도 확인

| 스크립트 | 비교 |
| --- | --- |
| `e2e/verify_llama_layer.py --layer L` | layer 하나: UPMEM vs CPU W8A8 vs HF FP32 |
| `e2e/check_llama_accuracy.py` | CPU만: W8A8 layer가 FP32 모델과 어디서 달라지는지 (prefill·decode) |
| `e2e/check_llama_generate_cpu.py --report <U6 report.json>` | 같은 요청을 CPU layer로 생성해 token ID 비교 |
| `e2e/verify_decode_gemv.py` | decode shape(token 하나) GEMV: UPMEM vs CPU |

---

## 7. 다른 target과 비교할 때

| 항목 | UPMEM | 비교 시 |
| --- | --- | --- |
| prefill 기준 값 | `forward_breakdown.csv` `warm_mean` | weight가 전부 상주(groups = 1)면 warm 요청의 upload는 0. Silent-PIM은 `ttft_without_group0_load_ms`와 맞춘다 ([Silent-PIM 실험](Silent-PIM_Experiments.md)) |
| decode 기준 값 | `step_breakdown.csv` `mean_after_first` | 같은 행 이름 |
| prefill의 KV | layer마다 CPU에 보관 | Silent-PIM prefill은 KV를 출력하지 않는다 |
| decode의 prefill | 같은 session에서 PIM으로 실행 | Silent-PIM decode는 HF CPU prefill 뒤 decode만 compile |
