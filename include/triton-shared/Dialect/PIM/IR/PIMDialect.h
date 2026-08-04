#ifndef TRITON_SHARED_DIALECT_PIM_IR_PIM_DIALECT_H_
#define TRITON_SHARED_DIALECT_PIM_IR_PIM_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"  // 생성된 op이 ::mlir::BytecodeOpInterface::Trait를 상속
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// PIM dialect / attributes / types / ops 선언 (TableGen 생성물 포함)
//===----------------------------------------------------------------------===//
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/PIM/IR/PIMAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/PIM/IR/PIMTypes.h.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/PIM/IR/PIMOps.h.inc"

#endif // TRITON_SHARED_DIALECT_PIM_IR_PIM_DIALECT_H_
