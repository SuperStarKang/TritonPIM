#!/bin/bash
# TritonPIM GEMV를 shape마다 autotune해서 best config의 성능을 측정한다.
#
# 실행:
#   source ~/TritonPIM/set_env.sh && conda activate pim_triton
#   ./run_autotune_gemv.sh                  # tasklet 16(기본)
#   TASKLETS="8 16 24" ./run_autotune_gemv.sh   # tasklet 수도 바꿔가며 탐색
#
# 탐색 결과는 gemv_results/gemv_autotune_cache.json에 쌓이므로, 다시 실행하면 탐색 없이 best만 측정한다.
set -eo pipefail
cd "$(dirname "$0")"

OUT=gemv_results/$(date +%Y%m%d_%H%M%S)
CSV=$OUT/triton_autotune_gemv.csv

MAX_BENCHMARKS=${MAX_BENCHMARKS:-50}
TASKLETS=${TASKLETS:-0}

# M K  (ATiM에서 측정이 되는 shape)
SHAPES=(
  "1024 1024"
  "4096 4096"
  "8192 8192"
  "16384 4096"
  "4096 16384"
  "21504 7168"
  "7168 7168"
  "28672 7168"
  "7168 28672"
)

sudo -v
sudo mkdir -p "$OUT"

for s in "${SHAPES[@]}"; do
  read -r m k <<< "$s"
  for t in $TASKLETS; do
    ./cmd.sh autotune_gemv.py --m "$m" --k "$k" --nr-tasklets "$t" --max-benchmarks "$MAX_BENCHMARKS" \
      2>&1 | tee "$OUT/triton_autotune_gemv_${m}_${k}_t${t}.log"
  done
done

# sudo로 만든 cache 파일을 사용자 소유로 되돌린다
sudo chown -R "$(id -u):$(id -g)" gemv_results

grep -h -m1 '^HEADER,' "$OUT"/triton_autotune_gemv_*.log | head -1 | cut -d, -f2- > "$CSV"
grep -h '^RESULT,' "$OUT"/triton_autotune_gemv_*.log | cut -d, -f2- >> "$CSV"
column -t -s, "$CSV"
