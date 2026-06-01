#include <pybind11/pybind11.h>

namespace py = pybind11;

// PIM backend는 Python에서 직접 IR을 컴파일하지 않고 triton-shared-opt subprocess로
// 변환을 수행한다. 따라서 pybind 모듈 init은 비워둔다 (triton_shared.cc와 동일 패턴).
//
// 이 심볼이 필요한 이유: Triton 본체(python/src/main.cc)가 TRITON_BACKENDS_TUPLE의
// 모든 backend 이름마다 init_triton_<name>을 선언·호출하므로, "pim"에 대한
// init_triton_pim이 없으면 링크에 실패한다.
void init_triton_pim(py::module &&m) {}
