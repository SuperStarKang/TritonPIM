export C=/home/dlrkdals/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
export N=16

export OMP_NUM_THREADS=16
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/inductor_cpp_t$OMP_NUM_THREADS