#ifndef TRITON_SHARED_DIALECT_PIMTARGET_UPMEM_IR_UPMEM_TARGET_SPEC_H_
#define TRITON_SHARED_DIALECT_PIMTARGET_UPMEM_IR_UPMEM_TARGET_SPEC_H_

namespace mlir {
namespace upmem {

// UPMEM 의 footprint 지식을 공통 target spec 레지스트리("upmem")에 등록한다.
// 도구 초기화(RegisterTritonSharedDialects)에서 명시 호출한다.
void registerUPMEMTargetSpec();

} // namespace upmem
} // namespace mlir

#endif // TRITON_SHARED_DIALECT_PIMTARGET_UPMEM_IR_UPMEM_TARGET_SPEC_H_
