#ifndef TPTR_TO_LLVM_CONVERSION_PASSES
#define TPTR_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TPtrToLLVM : Pass<"tptr-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower tptr ops and !ptr.ptr together with arith/math/complex/vector/index/memref/func/cf to LLVM";
  let description = [{
    !ptr.ptr appears as a memref element type, so it must be converted by the same
    type converter that lowers memref. This pass therefore runs the standard
    *-to-LLVM patterns in one conversion, with !ptr.ptr -> !llvm.ptr added.
  }];
  let constructor = "triton::createTPtrToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif
