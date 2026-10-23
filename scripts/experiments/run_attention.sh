#!/bin/bash
# 실험 4(attention 성능): Llama-3-8B layer 0의 QKV proj -> QK^T -> softmax(CPU) -> PV를 인자로 받은 FPGA 개수마다 측정한다.
# 결과: $ROOT/results/attention/f<N>/report.json 의 summary.attention_ms와 kinds(weight/activation GEMM, CPU).
set -eo pipefail
[ $# -gt 0 ] || { echo "usage: $0 <fpgas...>  (예: $0 4 8)" >&2; exit 1; }
source "$(dirname "$0")/../env.sh"
EXAMPLES=$ROOT/triton_shared/pim_backend/torch_backend/examples

for fpgas in "$@"; do
  out=$ROOT/results/attention/f$fpgas
  mkdir -p $out
  python $EXAMPLES/aimx/attention/benchmark_attention.py --fpgas $fpgas \
    --checkpoint $MODELS_DIR/Meta-Llama-3-8B --prompt-file $EXAMPLES/upmem/e2e/prompt.txt \
    --cache $out/autotune_cache.json --report-dir $out 2>&1 | tee $out/log.txt
done
