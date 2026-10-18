#ifndef TRITON_CONVERSION_TPTRTOLLVM_TPTRTOLLVM_H
#define TRITON_CONVERSION_TPTRTOLLVM_TPTRTOLLVM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTPtrToLLVMPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TPTRTOLLVM_TPTRTOLLVM_H
