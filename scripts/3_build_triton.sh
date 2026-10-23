#!/bin/bash
# 빌드한 LLVM으로 triton + triton_shared + pim_backend를 pim_triton env에 editable 설치한다.
set -eo pipefail
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
source $SCRIPTS/env.sh

# triton은 -Werror로 빌드되므로 llvm_toolchain의 clang이 PATH에 있어도 시스템 gcc를 쓰게 한다.
# cmake는 처음 configure할 때의 컴파일러를 build/의 CMakeCache.txt에 남긴다.
export CC=gcc CXX=g++
# setup.py의 기본 병렬 수는 2 * CPU 수다. MLIR 파일 하나가 cc1plus 1.3GB 이상을 쓰므로 메모리에 맞춰 줄인다.
export MAX_JOBS=8

cd $ROOT/triton
pip install -e . --no-build-isolation -v
