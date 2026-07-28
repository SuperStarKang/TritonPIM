#ifndef TRITON_SHARED_DIALECT_PIMTARGET_UPMEM_IR_UPMEM_DIALECT_H_
#define TRITON_SHARED_DIALECT_PIMTARGET_UPMEM_IR_UPMEM_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"  // 생성된 op이 ::mlir::BytecodeOpInterface::Trait를 상속
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.h.inc"

#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMTypes.h.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMOps.h.inc"

#endif // TRITON_SHARED_DIALECT_PIMTARGET_UPMEM_IR_UPMEM_DIALECT_H_
