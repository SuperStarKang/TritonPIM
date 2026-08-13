#ifndef TRITON_CONVERSION_PIMTOUPMEM_PIMTOUPMEM_H
#define TRITON_CONVERSION_PIMTOUPMEM_PIMTOUPMEM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL_CONVERTPIMTOUPMEM
#include "triton-shared/Conversion/PIMToUPMEM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertPIMToUPMEMPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_PIMTOUPMEM_PIMTOUPMEM_H
