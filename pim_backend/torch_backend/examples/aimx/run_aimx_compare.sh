#!/usr/bin/env bash
# Llama-2와 Llama-3를 AiMX(FPGA 2장)에서 prompt 128/256/512/1024 token으로 측정한다.
#   bash run_aimx_compare.sh prefill   e2e/benchmark_e2e.py, 32 layer, 첫 요청 1회 + 이후 요청 REPEAT회.
#                                   sweep: kv share/repeat x (QK^T head 묶음, PV head 묶음).
#                                   최선 설정은 결과를 보고 직접 고른다.
#   bash run_aimx_compare.sh decode    e2e/benchmark_decode.py, 32 layer, prefill을 포함해 NEW_TOKENS개 생성.
# conda activate pim_triton 후 실행한다.
# 결과:
#   prefill  e2e/compare/<model>_s<seq>_<kv>_qk<qk>_pv<pv>_f<fpgas>/
#            {report.json,forward_breakdown.csv,gemm.csv,attention.csv,run.log}
#   decode   e2e/compare/<model>_s<seq>_decode_f<fpgas>/run.log
set -euo pipefail

if [ "$#" -ne 1 ] || { [ "$1" != prefill ] && [ "$1" != decode ]; }; then
    echo "Usage: $0 prefill|decode" >&2
    exit 1
fi
mode=$1

cd "$(dirname "$0")"

FPGAS=2
BASE_FPGA=0
REPEAT=3
NEW_TOKENS=128
MAX_BENCHMARKS=20
declare -A CHECKPOINTS=(
    [llama2]=/home/dlrkdals/models/llama-2-7b
    [llama3]=/home/dlrkdals/models/Meta-Llama-3-8B
)
# "qk pv": 한 GEMM에 block-diagonal로 묶는 query head 수. 32 head를 나눠야 한다.
HEAD_PACKINGS=("1 1" "2 2" "2 4" "4 4")

for model in llama2 llama3; do
    for seq in 128 256 512 1024; do
        if [ "$mode" = decode ]; then
            report=e2e/compare/${model}_s${seq}_decode_f${FPGAS}
            mkdir -p "$report"
            echo "=== $report"
            python e2e/benchmark_decode.py --checkpoint "${CHECKPOINTS[$model]}" \
                --prompt-file ../upmem/e2e/prompt.txt --prompt-tokens "$seq" --new-tokens "$NEW_TOKENS" \
                --fpgas "$FPGAS" --base-fpga "$BASE_FPGA" --max-benchmarks "$MAX_BENCHMARKS" \
                2>&1 | tee "$report/run.log"
            continue
        fi
        for kv in share repeat; do
            for packing in "${HEAD_PACKINGS[@]}"; do
                read -r qk pv <<< "$packing"
                report=e2e/compare/${model}_s${seq}_${kv}_qk${qk}_pv${pv}_f${FPGAS}
                mkdir -p "$report"
                echo "=== $report"
                python e2e/benchmark_e2e.py --checkpoint "${CHECKPOINTS[$model]}" \
                    --prompt-file ../upmem/e2e/prompt.txt --layers 32 --seq "$seq" --kv "$kv" \
                    --qk-heads "$qk" --pv-heads "$pv" --repeat "$REPEAT" \
                    --fpgas "$FPGAS" --base-fpga "$BASE_FPGA" --max-benchmarks "$MAX_BENCHMARKS" \
                    --report-dir "$report" 2>&1 | tee "$report/run.log"
            done
        done
    done
done
