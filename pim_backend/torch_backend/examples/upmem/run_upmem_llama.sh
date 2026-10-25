#!/usr/bin/env bash
# Llama-2와 Llama-3를 UPMEM(2048 DPU, 16 tasklet)에서 prefill TTFT breakdown으로 측정한다.
# conda activate pim_triton 후 실행한다. cmd.sh가 sudo와 UPMEM_BACKEND=hw를 붙인다.
set -euo pipefail

cd "$(dirname "$0")"

LOCAL_CAPACITY_LIMIT_BYTES=46680
declare -A CHECKPOINTS=(
    [llama2]=/home/dlrkdals/models/llama-2-7b
    [llama3]=/home/dlrkdals/models/Meta-Llama-3-8B
)

for model in llama2 llama3; do
    report=e2e/${model}_ttft_upmem
    mkdir -p "$report"
    echo "=== $report"
    ./cmd.sh run_upmem.py llama --checkpoint "${CHECKPOINTS[$model]}" --repeat 3 \
        --local-capacity-limit-bytes "$LOCAL_CAPACITY_LIMIT_BYTES" \
        --report-dir "$report" 2>&1 | tee "$report/run.log"
done
