#!/bin/bash
# triton을 받아 패치하고, triton이 고정한 LLVM 커밋을 triton-san/script/build_llvm.sh로 빌드한다.
set -eo pipefail
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
source $SCRIPTS/env.sh

# LLVM 커밋(triton/cmake/llvm-hash.txt)을 triton에서 읽으므로 triton을 먼저 받는다
git clone https://github.com/triton-lang/triton.git $ROOT/triton
git -C $ROOT/triton checkout $(cat $ROOT/triton_shared/triton-hash.txt)
git -C $ROOT/triton apply $SCRIPTS/patches/triton-main-cc.patch

# build_llvm.sh는 PATH의 clang/clang++, cmake, ninja를 쓴다
export PATH=$LLVM_TOOLCHAIN_PREFIX/bin:$PATH
mkdir -p $ROOT/llvm
$ROOT/triton_shared/triton-san/script/build_llvm.sh $ROOT/triton $ROOT/llvm
