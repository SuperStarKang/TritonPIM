#!/bin/bash
# 빌드한 LLVM으로 triton + triton_shared + pim_backend를 pim_triton env에 editable 설치한다.
set -eo pipefail
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
source $SCRIPTS/env.sh

cd $ROOT/triton
pip install -e . --no-build-isolation -v
