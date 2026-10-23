# 빌드·실험 공통 환경. 사용법: source triton_shared/scripts/env.sh

# 서버마다 수정하는 값
export CONDA_BASE=$HOME/miniconda3
export MODELS_DIR=$HOME/models
export AIMX_PIM_SW=$HOME/pim_aimx_sw/pim_sw

# triton_shared/scripts/env.sh 기준으로 ../.. 가 ROOT (llvm/, triton/, triton_shared/, dump/)
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export ROOT
TRITON_BUILD=$ROOT/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate pim_triton

# llvm
export LLVM_TOOLCHAIN_PREFIX=$CONDA_BASE/envs/llvm_toolchain
export LLVM_BUILD_DIR=$ROOT/llvm/llvm-build
export LLVM_SYSPATH=$LLVM_BUILD_DIR
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_BINARY_DIR=$ROOT/llvm/llvm-install/bin
export LLVM_DIR=$ROOT/llvm/llvm-install/lib/cmake/llvm

# zstd
# LLVM은 llvm_toolchain env의 zstd로 빌드돼 find_package(zstd)가 그 헤더/라이브러리를 요구한다.
export CMAKE_PREFIX_PATH="$LLVM_TOOLCHAIN_PREFIX:$CMAKE_PREFIX_PATH"

# triton-shared (CPU backend) + pim_backend (PIM backend: device="pim", vendor=aimx)
export TRITON_PLUGIN_DIRS="$ROOT/triton_shared;$ROOT/triton_shared/pim_backend"

export TRITON_SHARED_OPT_PATH=$TRITON_BUILD/tools/triton-shared-opt/triton-shared-opt
export TRITON_SHARED_TRANSLATE_PATH=$TRITON_BUILD/tools/triton-shared-translate/triton-shared-translate
alias opt=$TRITON_SHARED_OPT_PATH
alias translate=$TRITON_SHARED_TRANSLATE_PATH

# triton settings
export TRITON_ALWAYS_COMPILE=1 # 캐시된 커널을 쓰지 않음
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=$ROOT/dump # 커널 IR 덤프 디렉토리
export TRITON_KERNEL_DUMP=$ROOT/dump # 커널 IR 덤프 디렉토리
export TRITON_CACHE_DIR=$ROOT/dump

# FileCheck·count·not·llvm-lit 이 들어있는 llvm 빌드 bin 을 PATH 앞에 추가 (FileCheck 때문에 필수)
export PATH=$LLVM_BUILD_DIR/bin:$PATH
alias lit="python $LLVM_BUILD_DIR/bin/llvm-lit -v $TRITON_BUILD/test"

# aimx pim sw path
export LIBRARY_PATH=$AIMX_PIM_SW/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$AIMX_PIM_SW/lib:$LD_LIBRARY_PATH
