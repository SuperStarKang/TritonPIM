#!/usr/bin/env bash
# Llama-2와 Llama-3를 UPMEM(2048 DPU, 16 tasklet)에서 prompt 128/256/512/1024 token으로 측정한다.
# prefill은 prefill만 담은 session(run_upmem.py, cold 1회 + warm 2회), decode는 prefill과 함께 compile한
# 생성 session(e2e/benchmark_model_llama_generate.py, 127 step)이다. CPU thread는 PIM-LLM과 같은 8개다.
# conda activate pim_triton 후 실행한다. cmd.sh가 sudo와 UPMEM_BACKEND=hw를 붙인다.
# 결과는 e2e/compare/<model>_s<seq>_prefill/{report.json,forward_breakdown.csv,gemm.csv}와
# e2e/compare/<model>_s<seq>_decode/{report.json,step_breakdown.csv,gemm.csv}이다. summarize_compare.py가
# PIM-LLM과 함께 표로 만들어 e2e/compare/{summary.md,compare_prefill.csv,decode.csv}에 둔다.
set -euo pipefail

cd "$(dirname "$0")"

export OMP_NUM_THREADS=8
LOCAL_CAPACITY_LIMIT_BYTES=46680
declare -A CHECKPOINTS=(
    [llama2]=/home/dlrkdals/models/llama-2-7b
    [llama3]=/home/dlrkdals/models/Meta-Llama-3-8B
)

for model in llama2 llama3; do
    for seq in 128 256 512 1024; do
        report=e2e/compare/${model}_s${seq}_prefill
        mkdir -p "$report"
        echo "=== $report"
        ./cmd.sh run_upmem.py llama --checkpoint "${CHECKPOINTS[$model]}" --seq "$seq" --repeat 3 \
            --local-capacity-limit-bytes "$LOCAL_CAPACITY_LIMIT_BYTES" \
            --report-dir "$report" 2>&1 | tee "$report/run.log"

        report=e2e/compare/${model}_s${seq}_decode
        mkdir -p "$report"
        echo "=== $report"
        ./cmd.sh e2e/benchmark_model_llama_generate.py --checkpoint "${CHECKPOINTS[$model]}" \
            --prompt-file e2e/prompt.txt --prompt-tokens "$seq" \
            --local-capacity-limit-bytes "$LOCAL_CAPACITY_LIMIT_BYTES" \
            --report-dir "$report" 2>&1 | tee "$report/run.log"
    done
done

python summarize_compare.py | tee e2e/compare/summary.md
