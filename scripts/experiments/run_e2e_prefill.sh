#!/bin/bash
# 실험 5(LLM e2e prefill): 모델 32 layer, 1024 token prompt의 TTFT를 인자로 받은 FPGA 개수마다 측정한다.
# 결과: $ROOT/results/e2e_prefill/<model>/f<N>/report.json 의 summary.ttft_ms, per_layer_ms, kinds.
set -eo pipefail
[ $# -gt 1 ] || { echo "usage: $0 <llama2|llama3> <fpgas...>  (예: $0 llama3 4 8)" >&2; exit 1; }
model=$1
shift
case $model in
  llama2) checkpoint=Llama-2-7b-hf ;;
  llama3) checkpoint=Meta-Llama-3-8B ;;
  *) echo "model must be llama2 or llama3: $model" >&2; exit 1 ;;
esac
source "$(dirname "$0")/../env.sh"
EXAMPLES=$ROOT/triton_shared/pim_backend/torch_backend/examples

for fpgas in "$@"; do
  out=$ROOT/results/e2e_prefill/$model/f$fpgas
  mkdir -p $out
  python $EXAMPLES/aimx/e2e/benchmark_e2e.py --fpgas $fpgas --layers 32 \
    --checkpoint $MODELS_DIR/$checkpoint --prompt-file $EXAMPLES/upmem/e2e/prompt.txt \
    --cache $out/autotune_cache.json --report-dir $out 2>&1 | tee $out/log.txt
done
