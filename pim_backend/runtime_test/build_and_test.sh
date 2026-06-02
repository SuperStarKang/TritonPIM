#!/usr/bin/env bash
# 생성된 host.cpp/device.c를 컴파일해 실제 DPU에서 gemm 정확성을 검증한다.
# - UPMEM IR(test/Target/UPMEM/matmul.mlir)을 triton-shared-translate로 소스 생성
# - device.c는 dpu-upmem-dpurte-clang으로 .elf
# - host.cpp + main.cpp는 g++로 링크(libdpu)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED_ROOT="$(cd "$HERE/../.." && pwd)"
UPMEM_IR="$SHARED_ROOT/test/Target/UPMEM/matmul.mlir"

# triton-shared-translate 경로 (빌드 산출물에서 탐색)
TR="${TRITON_SHARED_TRANSLATE:-$(find /home/dlrkdals/TritonPIM/triton/build -name triton-shared-translate -type f 2>/dev/null | head -1)}"
if [ -z "$TR" ]; then
  echo "triton-shared-translate 를 찾지 못함. TRITON_SHARED_TRANSLATE 로 경로 지정." >&2
  exit 1
fi

cd "$HERE"
echo "[1/4] generate host.cpp / device.c"
"$TR" --upmem-to-host   "$UPMEM_IR" > host.cpp
"$TR" --upmem-to-device "$UPMEM_IR" > device.c

echo "[2/4] compile device.c -> matmul_kernel_device.elf (dpu-upmem-dpurte-clang)"
dpu-upmem-dpurte-clang -DNR_TASKLETS=1 device.c -o matmul_kernel_device.elf

echo "[3/4] compile host.cpp + main.cpp -> test_run (g++ + libdpu)"
# UPMEM host 빌드 플래그. dpu-pkg-config 가 표준; 없으면 명시 경로 사용.
if command -v dpu-pkg-config >/dev/null 2>&1; then
  DPU_FLAGS="$(dpu-pkg-config --cflags --libs dpu)"
else
  DPU_FLAGS="-I/usr/include/dpu -ldpu"
fi
g++ -O2 -std=c++17 host.cpp main.cpp $DPU_FLAGS -o test_run

echo "[4/4] run"
# dpu-upmem-dpurte-clang은 기본 v1B용 바이너리를 만든다. 시뮬레이터를 v1B로 맞춘다.
# 실제 하드웨어로 돌리려면 UPMEM_PROFILE="backend=hw" 로 덮어써서 실행.
: "${UPMEM_PROFILE:=chipId=0x42}"
sudo env "UPMEM_PROFILE=${UPMEM_PROFILE}" ./test_run
