#!/bin/bash
# 실험 4(attention 성능): 모델 layer 0의 QKV proj -> QK^T -> softmax(CPU) -> PV를 인자로 받은 FPGA 개수마다 측정한다.
# 결과: $ROOT/results/attention/<model>/f<N>/report.json 의 summary.attention_ms와 kinds(weight/activation GEMM, CPU).
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
  out=$ROOT/results/attention/$model/f$fpgas
  mkdir -p $out
  python $EXAMPLES/aimx/attention/benchmark_attention.py --fpgas $fpgas \
    --checkpoint $MODELS_DIR/$checkpoint --prompt-file $EXAMPLES/upmem/e2e/prompt.txt \
    --cache $out/autotune_cache.json --report-dir $out 2>&1 | tee $out/log.txt
done
