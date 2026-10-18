# TritonPIM Torch examples

모든 예제는 batch 1, sequence 1024, Llama-2-7B shape(hidden 4096, 32 heads × 128,
FFN 11008)을 쓴다. GEMM만 PIM에서 실행하고 나머지 연산은 CPU Inductor가 실행한다.
CPU reference와의 정확도 비교, calibration은 하지 않는다.

| 범위 | 연산 | PIM GEMM 수 | 상주 projection weight 수 |
|---|---|---:|---:|
| Attention kernel | QKᵀ → scale/mask/softmax → PV | 2 × heads | 0 |
| Attention layer | Q/K/V projection → RoPE → Attention → O projection | 68 | 4 |
| Transformer block | RMSNorm → Attention layer → residual → RMSNorm → SwiGLU FFN → residual | 71 | 7 |

## 디렉터리

```text
examples/
├── autotune_torch.py          # GEMM 하나의 offline autotuning (UPMEM / AiMX 공통)
├── fx_utils.py                # FX text/DOT 출력 (공통)
├── upmem/                     # INT8 x INT8 -> INT32, 2048 DPUs, 16 tasklets
│   ├── cmd.sh                 # UPMEM hw 실행 wrapper
│   ├── model_benchmark_common.py
│   ├── torch_model_session.py # 작은 session 예제 (tests가 사용)
│   ├── attention/             # int8_workloads.py, benchmark_model_attention_{kernel,layer}.py
│   ├── transformer/           # int8_workloads.py, benchmark_model_transformer_block.py
│   ├── e2e/                   # llama_workloads.py, benchmark_model_llama_ttft.py, prompt.txt
│   └── cpu/                   # 같은 INT8 workload의 CPU 전용 실행
└── aimx/                      # HF Llama-2-7b, BF16 GEMM on AiMX
    ├── llama_workloads.py                     # HF Llama-2-7b layer, BF16 GEMM
    ├── experiment_common.py                   # compile_model 호출, 요청별 breakdown
    ├── gemm/benchmark_gemm.py                 # GEMM 단독 breakdown (analytical / autotune)
    ├── layer/benchmark_layer.py               # layer 하나: 매 요청 업로드 vs weight-stationary
    ├── e2e/benchmark_e2e.py                   # 앞쪽 N layer prefill, TTFT
    ├── summarize_reports.py                   # layer/e2e report → CSV
    ├── verify_bf16_gemm.py                    # BF16 host GEMM = FP32 host 결과 (bit)
    ├── verify_k_alignment.py                  # A K pitch 정렬 검증
    ├── verify_llama_layer.py                  # layer 하나: HF FP32 vs CPU vs AiMX
    └── benchmark_llama_ttft.py                # 32 layer prefill TTFT, 첫 token
```

결과는 각 스크립트 옆의 `*_report/` 디렉터리에 `report.json`, `exported_fx.*`,
`partitioned_fx.*`로 저장된다.

## UPMEM

```bash
cd .../torch_backend/examples/upmem/attention
../cmd.sh benchmark_model_attention_kernel.py --heads 32 --gemm-config batched
../cmd.sh benchmark_model_attention_layer.py --gemm-config batched --warmup 3 --repeat 10

cd ../transformer
../cmd.sh benchmark_model_transformer_block.py --gemm-config batched --warmup 3 --repeat 10

cd ../e2e
../cmd.sh benchmark_model_llama_ttft.py --checkpoint <Llama-2-7b-hf snapshot> \
  --prompt-file prompt.txt --gemm-config batched --report-dir model_e2e_batched

cd ../cpu
python run_cpp_cpu.py attention-layer
```

`--gemm-config baseline`은 head 하나에 2048 DPUs, `batched`는 head당 64 DPUs로
32 heads를 한 batch round에 처리한다. Projection config와 물리 padding은 두 설정에서 같다.
Gate/Up은 물리 N=11264로 계산하고 11008열만 반환한다.

## AiMX

```bash
cd .../torch_backend/examples/aimx
# HF Llama-2-7b-hf checkpoint (BF16 GEMM, QK^T 2 head / PV 4 head block-diagonal)
# config는 compile_model이 layer 하나를 weight 상주로 튜닝해 고른다 (cache 재사용)
python verify_llama_layer.py   --checkpoint ~/models/Llama-2-7b-hf --prompt-file ../upmem/e2e/prompt.txt --fpgas 2
python benchmark_llama_ttft.py --checkpoint ~/models/Llama-2-7b-hf --prompt-file ../upmem/e2e/prompt.txt --fpgas 2
cd e2e && python benchmark_e2e.py --checkpoint ~/models/Llama-2-7b-hf --prompt-file ../../upmem/e2e/prompt.txt --layers 8
```

## 공통 autotuning

```bash
python autotune_torch.py --target aimx --num-units 1 --dtype float32 --m 1024 --n 4096 --k 4096
python autotune_torch.py --m 1024 --n 4096 --k 4096          # UPMEM, INT8
```

GEMM 하나를 `torch.compile` 경로로 측정한다. 측정에는 매 호출의 weight 전송이 포함된다.
weight 상주 모델의 config는 `compile_model`이 stage class마다 따로 측정한다.
