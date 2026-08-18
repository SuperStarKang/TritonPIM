#ifndef TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_TARGET_SPEC_H_
#define TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_TARGET_SPEC_H_

namespace mlir {
namespace aimx {

// AiMX 의 footprint·눈금 지식을 공통 target spec 레지스트리("aimx")에 등록한다.
// 도구 초기화(RegisterTritonSharedDialects)에서 명시 호출한다.
void registerAiMXTargetSpec();

} // namespace aimx
} // namespace mlir

#endif // TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_TARGET_SPEC_H_
