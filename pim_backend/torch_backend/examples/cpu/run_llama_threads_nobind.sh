#!/usr/bin/env bash
# Llama-2와 Llama-3를 eager / Inductor C++로, thread 8~96에서 TTFT 측정한다.
# thread 배치를 고정하지 않는다(OS scheduler가 정한다). 고정한 결과와 구분하려고 _nobind를 붙인다.
# conda activate pim_triton 후 이 디렉터리에서 실행한다.
set -euo pipefail

cd "$(dirname "$0")"

declare -A CHECKPOINTS=(
    [llama2]=/home/dlrkdals/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
    [llama3]=/home/dlrkdals/models/Meta-Llama-3-8B
)
declare -A SCRIPTS=(
    [eager]=run_eager_cpu.py
    [cpp]=run_cpp_cpu.py
)

# 셸에 남아 있는 배치 설정도 지운다
unset OMP_PROC_BIND OMP_PLACES

for threads in 8 16 32 48 96; do
    export OMP_NUM_THREADS=$threads
    # Inductor C++은 compile 시점의 thread 수를 코드에 넣으므로 thread 수마다 cache를 나눈다
    export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/inductor_cpp_t$threads
    for model in llama2 llama3; do
        for mode in eager cpp; do
            report=${model}_ttft_${mode}_t${threads}_nobind
            mkdir -p "$report"
            echo "=== $report"
            python "${SCRIPTS[$mode]}" llama --checkpoint "${CHECKPOINTS[$model]}" --repeat 3 \
                --report-dir "$report" 2>&1 | tee "$report/run.log"
        done
    done
done
