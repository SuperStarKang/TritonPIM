#ifndef TRITON_SHARED_TARGET_UPMEM_UPMEMTRANSLATIONS_H_
#define TRITON_SHARED_TARGET_UPMEM_UPMEMTRANSLATIONS_H_

namespace mlir {
namespace triton {

// UPMEM IR을 소스로 옮긴다. 둘 다 IR을 그대로 읽어 낸다 — op 하나가 코드 한 덩어리이고,
// 다루지 못하는 op을 만나면 에러를 낸다(모르는 것을 짐작해 채우지 않는다).
void registerUPMEMToHostTranslation();
void registerUPMEMToDeviceTranslation();

} // namespace triton
} // namespace mlir

#endif // TRITON_SHARED_TARGET_UPMEM_UPMEMTRANSLATIONS_H_
