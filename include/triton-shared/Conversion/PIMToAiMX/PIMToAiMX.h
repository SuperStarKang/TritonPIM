#ifndef TRITON_CONVERSION_PIMTOAIMX_PIMTOAIMX_H
#define TRITON_CONVERSION_PIMTOAIMX_PIMTOAIMX_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL_CONVERTPIMTOAIMX
#define GEN_PASS_DECL_AIMXLOWERMATMUL
#include "triton-shared/Conversion/PIMToAiMX/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertPIMToAiMXPass();
std::unique_ptr<OperationPass<ModuleOp>> createAiMXLowerMatmulPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_PIMTOAIMX_PIMTOAIMX_H
