#!/bin/bash
# 실험 7(LLM decode, prompt 길이별): Llama-2와 Llama-3 32 layer decode를 prompt 128/256/512/1024 token 뒤로,
# 인자로 받은 FPGA 개수마다 측정한다. decode phase만 compile한다(AiMX는 phase 하나만 배치한다).
# prompt는 Hugging Face 모델이 CPU에서 prefill해 key/value cache를 채우고(측정 밖), prefill을 포함해
# 128 token을 생성한다(127 step). 실행마다 autotune cache를 새로 둔다.
# 결과: $ROOT/results/decode/<model>/f<N>/s<seq>/
#   step_breakdown.csv(mean_after_first 행의 step_ms), gemm.csv, attention.csv
set -eo pipefail
[ $# -gt 0 ] || { echo "usage: $0 <fpgas...>  (예: $0 4 8)" >&2; exit 1; }
source "$(dirname "$0")/../env.sh"
# torch CPU thread와 OpenMP pool 크기를 고정한다. autotuning 측정 프로세스도 같은 값을 쓴다.
# FPGA job thread는 stub이 FPGA 수만큼 연다 (같은 pool의 worker를 다시 쓴다).
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=false
EXAMPLES=$ROOT/triton_shared/pim_backend/torch_backend/examples

for fpgas in "$@"; do
  for model in llama2 llama3; do
    case $model in
      llama2) checkpoint=Llama-2-7b-hf ;;
      llama3) checkpoint=Meta-Llama-3-8B ;;
    esac
    for seq in 128 256 512 1024; do
      out=$ROOT/results/decode/$model/f$fpgas/s$seq
      mkdir -p $out
      python $EXAMPLES/aimx/e2e/benchmark_decode.py --fpgas $fpgas --prompt-tokens $seq --new-tokens 128 \
        --checkpoint $MODELS_DIR/$checkpoint --prompt-file $EXAMPLES/upmem/e2e/prompt.txt \
        --cache $out/autotune_cache.json --report-dir $out 2>&1 | tee $out/log.txt
    done
  done
done
