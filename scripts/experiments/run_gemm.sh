#!/bin/bash
# 실험 1(GEMM 성능)과 3(Autotuning 수렴 곡선): 모델의 prefill GEMM(M=1024)을 인자로 받은 FPGA 개수마다 autotune 후 측정한다.
# 결과: $ROOT/results/gemm/<model>/f<N>/report.json 의 gemms[].wall_ms(best config 측정)와 gemms[].tuning(측정 순서별 후보).
# cache가 결과 폴더에 있어 다시 돌리면 탐색 없이 측정만 한다. 수렴 곡선을 다시 얻으려면 폴더를 지우고 실행한다.
set -eo pipefail
[ $# -gt 1 ] || { echo "usage: $0 <llama2|llama3> <fpgas...>  (예: $0 llama3 4 8)" >&2; exit 1; }
model=$1
shift
# 이름은 benchmark_gemm.py의 SHAPES. Llama-2는 MHA라 k/v가 q_proj와 같고, gate/up은 11008을 11264로 올린다.
case $model in
  llama2) shapes="q_proj gate_proj down_proj" ;;
  llama3) shapes="q_proj llama3_kv_proj llama3_gate_proj llama3_down_proj" ;;
  *) echo "model must be llama2 or llama3: $model" >&2; exit 1 ;;
esac
source "$(dirname "$0")/../env.sh"
EXAMPLES=$ROOT/triton_shared/pim_backend/torch_backend/examples/aimx

for fpgas in "$@"; do
  out=$ROOT/results/gemm/$model/f$fpgas
  mkdir -p $out
  python $EXAMPLES/gemm/benchmark_gemm.py --config-source autotune --fpgas $fpgas --shapes $shapes \
    --max-benchmarks 50 --cache $out/autotune_cache.json --report-dir $out 2>&1 | tee $out/log.txt
done
