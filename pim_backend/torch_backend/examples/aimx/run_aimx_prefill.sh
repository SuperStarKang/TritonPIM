#!/usr/bin/env bash
# Llama-2와 Llama-3 prefill을 AiMX(FPGA 2장)에서 prompt 128/256/512/1024 token으로 측정한다.
# experiment 브랜치의 scripts/experiments/run_e2e_prefill.sh와 같은 인자에 길이만 더한다:
# 32 layer, kv share, head packing은 컴파일러에 맡기고(qk/pv 1), 실행마다 autotune cache를 새로 둔다.
# prefill phase만 compile한다. 요청은 첫 요청 1회 + 이후 요청 REPEAT회다.
# conda activate pim_triton 후 실행한다.
# 결과: e2e/compare/<model>_s<seq>_prefill_f<fpgas>/
#       {report.json,forward_breakdown.csv,gemm.csv,attention.csv,autotune_cache.json,run.log}
set -euo pipefail

cd "$(dirname "$0")"

FPGAS=2
BASE_FPGA=0
REPEAT=3
MAX_BENCHMARKS=50
declare -A CHECKPOINTS=(
    [llama2]=/home/yonsei_park/models/Llama-2-7b-hf
    [llama3]=/home/yonsei_park/models/Meta-Llama-3-8B
)

for model in llama2 llama3; do
    for seq in 128 256 512 1024; do
        report=e2e/compare/${model}_s${seq}_prefill_f${FPGAS}
        mkdir -p "$report"
        echo "=== $report"
        python e2e/benchmark_e2e.py --checkpoint "${CHECKPOINTS[$model]}" \
            --prompt-file ../upmem/e2e/prompt.txt --layers 32 --seq "$seq" --repeat "$REPEAT" \
            --fpgas "$FPGAS" --base-fpga "$BASE_FPGA" --max-benchmarks "$MAX_BENCHMARKS" \
            --cache "$report/autotune_cache.json" --report-dir "$report" 2>&1 | tee "$report/run.log"
    done
done
