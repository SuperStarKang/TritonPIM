#!/bin/bash
# 실험 8(ablation): weight·KV 상주(S)와 FX packing(P)을 켜고 끈 구성의 prefill 또는 decode를,
# 인자로 받은 FPGA 개수마다 측정한다. 구성마다 그 실행 방식으로 autotuning한 config를 쓴다.
#   base  S off, P off   GEMM마다 호출 하나, weight·activation(·KV)을 매번 보낸다
#   S     S on,  P off   GEMM마다 호출 하나, weight와 decode KV cache는 장치에 상주
#   P     S off, P on    FX packing(①②③)과 instance 묶음 후의 stage마다 호출, 매번 보낸다
#   all   S on,  P on    run_prefill.sh / run_decode.sh와 같다
# 환경 변수로 범위를 줄일 수 있다 (기본값은 괄호):
#   VARIANTS ("base S P all")  MODELS ("llama2 llama3")  SEQS ("1024")  NEW_TOKENS (128, decode)
# 결과: $ROOT/results/ablation/<phase>/<model>/f<N>/s<seq>/<variant>/
#   prefill  report.json, forward_breakdown.csv, gemm.csv, attention.csv, autotune_cache.json, log.txt
#   decode   step_breakdown.csv, gemm.csv, attention.csv, autotune_cache.json, log.txt
set -eo pipefail
[ $# -gt 1 ] && { [ "$1" = prefill ] || [ "$1" = decode ]; } || {
  echo "usage: $0 <prefill|decode> <fpgas...>  (예: $0 decode 4 8, VARIANTS=\"base all\" SEQS=128 $0 prefill 8)" >&2
  exit 1
}
phase=$1
shift
source "$(dirname "$0")/../env.sh"
EXAMPLES=$ROOT/triton_shared/pim_backend/torch_backend/examples
# torch CPU thread와 OpenMP pool 크기를 고정한다. autotuning 측정 프로세스도 같은 값을 쓴다.
# FPGA job thread는 stub이 FPGA 수만큼 연다 (같은 pool의 worker를 다시 쓴다).
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=false
VARIANTS=${VARIANTS:-"base S P all"}
MODELS=${MODELS:-"llama2 llama3"}
SEQS=${SEQS:-"1024"}
NEW_TOKENS=${NEW_TOKENS:-128}

for fpgas in "$@"; do
  for model in $MODELS; do
    case $model in
      llama2) checkpoint=Llama-2-7b-hf ;;
      llama3) checkpoint=Meta-Llama-3-8B ;;
      *) echo "model must be llama2 or llama3: $model" >&2; exit 1 ;;
    esac
    for seq in $SEQS; do
      for variant in $VARIANTS; do
        out=$ROOT/results/ablation/$phase/$model/f$fpgas/s$seq/$variant
        mkdir -p $out
        if [ $phase = prefill ]; then
          python $EXAMPLES/aimx/e2e/benchmark_e2e.py --fpgas $fpgas --layers 32 --seq $seq --variant $variant \
            --checkpoint $MODELS_DIR/$checkpoint --prompt-file $EXAMPLES/upmem/e2e/prompt.txt \
            --cache $out/autotune_cache.json --report-dir $out 2>&1 | tee $out/log.txt
        else
          python $EXAMPLES/aimx/e2e/benchmark_decode.py --fpgas $fpgas --prompt-tokens $seq \
            --new-tokens $NEW_TOKENS --variant $variant \
            --checkpoint $MODELS_DIR/$checkpoint --prompt-file $EXAMPLES/upmem/e2e/prompt.txt \
            --cache $out/autotune_cache.json --report-dir $out 2>&1 | tee $out/log.txt
        fi
      done
    done
  done
done
