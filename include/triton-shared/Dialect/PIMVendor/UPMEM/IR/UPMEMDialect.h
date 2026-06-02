#ifndef TRITON_SHARED_DIALECT_PIMVENDOR_UPMEM_IR_UPMEM_DIALECT_H_
#define TRITON_SHARED_DIALECT_PIMVENDOR_UPMEM_IR_UPMEM_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"  // 생성된 op이 ::mlir::BytecodeOpInterface::Trait를 상속
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton-shared/Dialect/PIMVendor/UPMEM/IR/UPMEMDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/PIMVendor/UPMEM/IR/UPMEMTypes.h.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/PIMVendor/UPMEM/IR/UPMEMOps.h.inc"

#endif // TRITON_SHARED_DIALECT_PIMVENDOR_UPMEM_IR_UPMEM_DIALECT_H_
