# pim_backend는 Python 전용 backend 패키지다.
# PIM dialect / conversion / runtime의 C++ 코드는 triton_shared 트리(include/, lib/)에 들어가
# triton-shared-opt로 빌드·공유되므로, 여기서는 dialect를 빌드하지 않는다.
#
# 다만 Triton 본체(python/src/main.cc)가 TRITON_BACKENDS_TUPLE의 모든 이름마다
# init_triton_<name> 심볼을 선언·호출하므로, "pim"을 위한 빈 init_triton_pim을
# 제공해야 libtriton 링크가 성립한다. pim.cc가 그 역할만 한다.

include_directories(${Python3_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})

add_triton_plugin(TritonPim ${CMAKE_CURRENT_SOURCE_DIR}/pim.cc)
target_link_libraries(TritonPim PRIVATE Python3::Module pybind11::headers)
