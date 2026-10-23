# AiMX 실험 환경 구성

## 사전 준비 (서버마다 1회)

| 항목 | 내용 |
|---|---|
| Miniconda | `~/miniconda3`에 설치 (다른 위치면 `env.sh`의 `CONDA_BASE`와 `1_setup_conda.sh` 수정) |
| gcc / g++ | 시스템 gcc (triton 빌드, AiMX stub `-fopenmp` 빌드) |
| AiMX SW | `$AIMX_PIM_SW`(기본 `~/pim_aimx_sw/pim_sw`)에 `lib/libpim.so`, `include/pim.h`, `src/pim_math_src/pim_math.h` |
| AiMX 드라이버 | `pim_mem_probe.c`의 PCI 버스를 `lspci -d 10ee:` 결과에 맞추고, 부팅 후 1회만 로드 |
| Hugging Face | meta-llama/Llama-2-7b-hf, meta-llama/Meta-Llama-3-8B 접근 승인 후 `hf auth login` |
| 디스크 | LLVM 약 16GB, 모델 약 30GB |

## 디렉터리 구조

```
TritonPIM/            ← ROOT
├── triton_shared/    ← 이 리포 (폴더 이름은 반드시 triton_shared)
├── triton/           ← 2_build_llvm.sh가 받음
├── llvm/             ← 2_build_llvm.sh가 빌드
└── dump/             ← TRITON_CACHE_DIR
```

## 순서

```bash
mkdir TritonPIM && cd TritonPIM
git clone -b experiment https://github.com/SuperStarKang/TritonPIM.git triton_shared

./triton_shared/scripts/1_setup_conda.sh       # pim_triton, llvm_toolchain env 생성
./triton_shared/scripts/2_build_llvm.sh        # triton clone + 패치, LLVM 빌드
./triton_shared/scripts/3_build_triton.sh      # triton + triton_shared + pim_backend 설치
./triton_shared/scripts/4_download_models.sh   # ~/models에 Llama 모델 다운로드
```

각 스크립트는 이미 만들어진 env/디렉터리가 있으면 실패한다. 다시 돌리려면 해당 env나 디렉터리를 지우고 실행한다.

## 실험

prefill 토큰 수는 1024. 첫 인자로 모델(`llama2` = Llama-2-7b-hf, `llama3` = Meta-Llama-3-8B), 그 뒤에 FPGA 개수들을 준다 (예: `llama3 4 8`, 시험용 `llama3 1 2`). 결과는 `results/<실험>/<model>/f<N>/{report.json,log.txt}`.

| # | 실험 | 스크립트 | 주요 결과 |
|---|---|---|---|
| 1 | GEMM 성능 | `experiments/run_gemm.sh` | `gemms[].wall_ms`, `config` |
| 2 | GEMV 성능 | (AiMX GEMV autotune 준비 후) | |
| 3 | Autotuning 수렴 곡선 | `experiments/run_gemm.sh` (1과 같은 실행) | `gemms[].tuning[].best_ms` |
| 4 | Attention | `experiments/run_attention.sh` | `summary.attention_ms`, `summary.kinds` |
| 5 | e2e prefill | `experiments/run_e2e_prefill.sh` | `summary.ttft_ms`, `per_layer_ms` |
| 5 | e2e decode | (decode 구현 후) | |

```bash
for model in llama2 llama3; do
  ./triton_shared/scripts/experiments/run_gemm.sh $model 4 8
  ./triton_shared/scripts/experiments/run_attention.sh $model 4 8
  ./triton_shared/scripts/experiments/run_e2e_prefill.sh $model 4 8
done
```

autotune cache가 결과 폴더에 저장되므로, 같은 폴더로 다시 돌리면 탐색 없이 측정만 한다. 수렴 곡선을 다시 얻으려면 폴더를 지우고 실행한다.

## 확인

```bash
source triton_shared/scripts/env.sh
python -c "import triton.backends as b; print(list(b.backends))"   # pim 포함
ls $TRITON_SHARED_OPT_PATH $TRITON_SHARED_TRANSLATE_PATH
ls $AIMX_PIM_SW/lib/libpim.so $AIMX_PIM_SW/include/pim.h $AIMX_PIM_SW/src/pim_math_src/pim_math.h
ls $MODELS_DIR/Llama-2-7b-hf $MODELS_DIR/Meta-Llama-3-8B
```
