#!/bin/bash
# TritonPIM GEMV를 ATiM FC shape과 같은 DPU 분할(p_m x p_k)로 측정한다.
#
# 실행:
#   source ~/TritonPIM/set_env.sh && conda activate pim_triton
#   ./run_benchmark_gemv.sh
#
# h2d stage는 매 호출의 A 전송을 포함한다 (ATiM H2D는 x만 포함).
set -eo pipefail
cd "$(dirname "$0")"

OUT=gemv_results/$(date +%Y%m%d_%H%M%S)
CSV=$OUT/triton_gemv.csv
mkdir -p "$OUT"

# M K BLOCK_M BLOCK_K p_m p_k tasklets   (ATiM pretuned 분할과 동일)
# BLOCK_M/BLOCK_K는 블록 수가 p_m/p_k로 나누어떨어지도록 골랐다
SHAPES=(
  "1024 1024 32 64 16 16 16"
  "4096 4096 32 64 64 32 8"
  "8192 8192 32 64 64 32 16"
  "12288 4096 32 64 64 32 24"
  "16384 4096 32 64 256 8 16"
  "4096 16384 32 64 32 64 16"
  "21504 7168 16 32 64 32 21"
  "7168 7168 32 64 112 16 16"
  "28672 7168 32 32 56 32 16"
  "7168 28672 16 64 64 32 14"
)

sudo -v
for s in "${SHAPES[@]}"; do
  read -r m k bm bk pm pk t <<< "$s"
  ./cmd.sh benchmark_gemv.py --m "$m" --k "$k" --block-m "$bm" --block-k "$bk" \
    --p-m "$pm" --p-k "$pk" --tasklets "$t" 2>&1 | tee "$OUT/triton_gemv_${m}_${k}.log"
done

grep -h -m1 '^HEADER,' "$OUT"/triton_gemv_*.log | head -1 | cut -d, -f2- > "$CSV"
grep -h '^RESULT,' "$OUT"/triton_gemv_*.log | cut -d, -f2- >> "$CSV"
column -t -s, "$CSV"
