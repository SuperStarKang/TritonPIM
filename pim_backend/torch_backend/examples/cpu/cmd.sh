# eager
python run_eager_cpu.py llama --checkpoint $C --repeat 3 --report-dir llama_ttft_eager_t$N

# inductor
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/inductor_cpp_t$N
python run_cpp_cpu.py llama --checkpoint $C --repeat 3 --report-dir llama_ttft_cpp_t$N

# triton_shared 
unset TORCHINDUCTOR_CACHE_DIR    # 3절에서 export했다면 해제
export TRITON_SHARED_NUM_THREADS=$N
python run_triton_shared_cpu.py llama --checkpoint $C --repeat 3 --report-dir llama_ttft_triton_shared_t$N
