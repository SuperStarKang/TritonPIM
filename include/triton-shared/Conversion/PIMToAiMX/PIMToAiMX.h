#ifndef TRITON_CONVERSION_PIMTOAIMX_PIMTOAIMX_H
#define TRITON_CONVERSION_PIMTOAIMX_PIMTOAIMX_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL_CONVERTPIMTOAIMX
#include "triton-shared/Conversion/PIMToAiMX/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertPIMToAiMXPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_PIMTOAIMX_PIMTOAIMX_H
