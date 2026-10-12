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
└── aimx/                      # FP32 host, BF16 on AiMX
    ├── fp32_workloads.py
    ├── aimx_benchmark_common.py
    ├── benchmark_aimx_attention_layer.py      # 고정 config (session-first)
    ├── benchmark_aimx_transformer_block.py    # 고정 config (session-first)
    ├── benchmark_aimx_model_compile.py        # policy 우선 model compile
    └── aimx_model_session.py                  # 작은 session 예제
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
# 고정 config
python benchmark_aimx_attention_layer.py   --fpgas 1
python benchmark_aimx_transformer_block.py --fpgas 1
# policy 우선 model compile: target과 unit 수는 입력, profile은 backend/target.py에 등록된 값
python benchmark_aimx_model_compile.py --scope attention   --units 1 --base-fpga 0
python benchmark_aimx_model_compile.py --scope transformer --units 1 --base-fpga 0
```

`benchmark_aimx_model_compile.py`의 compile time과 runtime 구조, policy 판정과 config 선택은
[Torch integration 설계](../../../design/TorchIntegration.md)에 있다.

## 공통 autotuning

```bash
python autotune_torch.py --target aimx --num-units 1 --dtype float32 --m 1024 --n 4096 --k 4096
python autotune_torch.py --m 1024 --n 4096 --k 4096          # UPMEM, INT8
```

GEMM 하나를 `torch.compile` 경로로 측정한다. 측정에는 매 호출의 weight 전송이 포함된다.
weight 상주 모델의 config는 `compile_model`이 policy 제약 안에서 따로 측정한다.
