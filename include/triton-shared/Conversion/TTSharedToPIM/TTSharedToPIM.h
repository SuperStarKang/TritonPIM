#ifndef TRITON_CONVERSION_TTSHAREDTOPIM_TTSHAREDTOPIM_H
#define TRITON_CONVERSION_TTSHAREDTOPIM_TTSHAREDTOPIM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL_CONVERTTTSHAREDTOPIM
#define GEN_PASS_DECL_PIMBUILDGRID
#define GEN_PASS_DECL_PIMREMOVEREDUNDANTTRANSFERS
#define GEN_PASS_DECL_PIMANALYZEREUSE
#define GEN_PASS_DECL_PIMANALYZETRANSFER
#define GEN_PASS_DECL_PIMCLASSIFYCOMPUTE
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTSharedToPIMPass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMBuildGridPass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMRemoveRedundantTransfersPass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMAnalyzeReusePass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMAnalyzeTransferPass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMClassifyComputePass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TTSHAREDTOPIM_TTSHAREDTOPIM_H
