#!/bin/bash
# pim_triton(빌드·실험)과 llvm_toolchain(LLVM 빌드용 clang) conda env를 만든다.
set -eo pipefail
SCRIPTS=$(cd "$(dirname "$0")" && pwd)

source $HOME/miniconda3/etc/profile.d/conda.sh
conda env create -f $SCRIPTS/environment.yml
conda env create -f $SCRIPTS/llvm_toolchain.yml
