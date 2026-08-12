#ifndef TRITON_SHARED_TARGET_AIMX_AIMXTRANSLATIONS_H_
#define TRITON_SHARED_TARGET_AIMX_AIMXTRANSLATIONS_H_

namespace mlir {
namespace triton {

// AiMX IR 을 소스로 옮긴다. 둘 다 IR 을 그대로 읽어 낸다 — op 하나가 코드 한 덩어리이고,
// 다루지 못하는 op 을 만나면 에러를 낸다(모르는 것을 짐작해 채우지 않는다).
//
// descriptors 쪽이 내는 것은 chain 이 아니라 **program** 이다. loop nest 라 크기가 작고,
// stub 이 launch 당 한 번만 부른다.
void registerAiMXToDescriptorsTranslation();
void registerAiMXToRuntimeStubTranslation();

} // namespace triton
} // namespace mlir

#endif // TRITON_SHARED_TARGET_AIMX_AIMXTRANSLATIONS_H_
