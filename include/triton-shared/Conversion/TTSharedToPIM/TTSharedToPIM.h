#ifndef TRITON_CONVERSION_TTSHAREDTOPIM_TTSHAREDTOPIM_H
#define TRITON_CONVERSION_TTSHAREDTOPIM_TTSHAREDTOPIM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL_CONVERTTTSHAREDTOPIM
#define GEN_PASS_DECL_REMOVEPIMBUFFERS
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTSharedToPIMPass();
std::unique_ptr<OperationPass<ModuleOp>> createRemovePIMBuffersPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TTSHAREDTOPIM_TTSHAREDTOPIM_H
