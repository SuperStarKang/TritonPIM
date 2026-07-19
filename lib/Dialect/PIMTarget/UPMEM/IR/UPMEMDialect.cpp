#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.h"

// 생성된 op builder / type printer-parser가 사용하는 헤더들.
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::upmem;

void UPMEMDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMOps.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMTypes.cpp.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMOps.cpp.inc"

#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.cpp.inc"
