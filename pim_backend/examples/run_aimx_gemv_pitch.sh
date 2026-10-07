#!/usr/bin/env bash
set -euo pipefail

cd /home/yonsei_park/kangmlee/TritonPIM/triton_shared/pim_backend/examples

g++ -O2 -std=c++17 -fPIC -shared -Wl,-z,defs \
    -I/home/yonsei_park/pim_aimx_sw/pim_sw/include \
    aimx_native_gemv_pitch.cpp \
    /home/yonsei_park/pim_aimx_sw/pim_sw/lib/libpim.so \
    -Wl,-rpath,/home/yonsei_park/pim_aimx_sw/pim_sw/lib \
    -lpthread -lm -o libaimx_native_gemv_pitch.so

python compare_aimx_gemv_pitch.py
