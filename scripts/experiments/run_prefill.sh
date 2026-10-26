#!/bin/bash
# 실험 6(LLM prefill, prompt 길이별): Llama-2와 Llama-3 32 layer prefill을 prompt 128/256/512/1024 token으로,
# 인자로 받은 FPGA 개수마다 측정한다. 인자는 run_e2e_prefill.sh와 같고(kv share, packing은 컴파일러,
# 실행마다 autotune cache) 길이만 더한다. prefill phase만 compile하고, 첫 요청 1회 + 이후 요청 3회를 잰다.
# 결과: $ROOT/results/prefill/<model>/f<N>/s<seq>/
#   report.json, forward_breakdown.csv(warm_median 행의 ttft_without_group0_load_ms), gemm.csv, attention.csv
set -eo pipefail
[ $# -gt 0 ] || { echo "usage: $0 <fpgas...>  (예: $0 4 8)" >&2; exit 1; }
source "$(dirname "$0")/../env.sh"
EXAMPLES=$ROOT/triton_shared/pim_backend/torch_backend/examples

for fpgas in "$@"; do
  for model in llama2 llama3; do
    case $model in
      llama2) checkpoint=Llama-2-7b-hf ;;
      llama3) checkpoint=Meta-Llama-3-8B ;;
    esac
    for seq in 128 256 512 1024; do
      out=$ROOT/results/prefill/$model/f$fpgas/s$seq
      mkdir -p $out
      python $EXAMPLES/aimx/e2e/benchmark_e2e.py --fpgas $fpgas --layers 32 --seq $seq \
        --checkpoint $MODELS_DIR/$checkpoint --prompt-file $EXAMPLES/upmem/e2e/prompt.txt \
        --cache $out/autotune_cache.json --report-dir $out 2>&1 | tee $out/log.txt
    done
  done
done
