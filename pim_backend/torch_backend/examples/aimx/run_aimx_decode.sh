#!/usr/bin/env bash
# Llama-2와 Llama-3 decode를 AiMX(FPGA 2장)에서 prompt 128/256/512/1024 token 뒤로 측정한다.
# decode phase만 compile한다 (AiMX는 phase 하나만 배치한다). prompt는 Hugging Face 모델이 CPU에서
# prefill해 key/value cache를 채우고(측정 밖), 32 layer decode가 prefill을 포함해 NEW_TOKENS개를
# 생성한다. 실행마다 autotune cache를 새로 둔다.
# conda activate pim_triton 후 실행한다.
# 결과: e2e/compare/<model>_s<seq>_decode_f<fpgas>/
#       {step_breakdown.csv,gemm.csv,attention.csv,autotune_cache.json,run.log}
set -euo pipefail

cd "$(dirname "$0")"

FPGAS=2
BASE_FPGA=0
NEW_TOKENS=128
MAX_BENCHMARKS=50
declare -A CHECKPOINTS=(
    [llama2]=/home/yonsei_park/models/Llama-2-7b-hf
    [llama3]=/home/yonsei_park/models/Meta-Llama-3-8B
)

for model in llama2 llama3; do
    for seq in 128 256 512 1024; do
        report=e2e/compare/${model}_s${seq}_decode_f${FPGAS}
        mkdir -p "$report"
        echo "=== $report"
        python e2e/benchmark_decode.py --checkpoint "${CHECKPOINTS[$model]}" \
            --prompt-file ../upmem/e2e/prompt.txt --prompt-tokens "$seq" --new-tokens "$NEW_TOKENS" \
            --fpgas "$FPGAS" --base-fpga "$BASE_FPGA" --max-benchmarks "$MAX_BENCHMARKS" \
            --cache "$report/autotune_cache.json" --report-dir "$report" 2>&1 | tee "$report/run.log"
    done
done
