#ifndef TRITON_SHARED_TARGET_UPMEM_UPMEMTRANSLATIONS_H
#define TRITON_SHARED_TARGET_UPMEM_UPMEMTRANSLATIONS_H

namespace mlir {
namespace triton {

// UPMEM IR → host C++ / device C 소스 번역을 mlir-translate 레지스트리에 등록한다.
// triton-shared-translate가 main에서 호출한다.
void registerUPMEMToHostTranslation();
void registerUPMEMToDeviceTranslation();

} // namespace triton
} // namespace mlir

#endif // TRITON_SHARED_TARGET_UPMEM_UPMEMTRANSLATIONS_H
