#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

// 생성된 op builder / type printer-parser가 사용하는 헤더들.
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pim;

// Dialect 생성 시점에 type과 op를 등록한다.
void PIMDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton-shared/Dialect/PIM/IR/PIMTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/PIM/IR/PIMOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd type / op / dialect 정의
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/PIM/IR/PIMTypes.cpp.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/PIM/IR/PIMOps.cpp.inc"

#include "triton-shared/Dialect/PIM/IR/PIMDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Verifiers — clause 조합이 말이 되는지 확인한다.
//===----------------------------------------------------------------------===//

LogicalResult AccessOp::verify() {
  int64_t rank = getSizes().size();
  // R1: 축별 stride 개수 == 타일 rank.
  if (static_cast<int64_t>(getStrides().size()) != rank)
    return emitOpError("strides count must equal sizes rank");
  // R2: valid를 쓰면 축별로 하나씩(= rank).
  if (!getValid().empty() && static_cast<int64_t>(getValid().size()) != rank)
    return emitOpError("valid count must equal sizes rank");
  // R3: pad(채움값)는 valid(mask)와 함께만 의미가 있다. (저장은 valid만 OK.)
  if (getPad() && getValid().empty())
    return emitOpError("pad requires valid");
  // R4: gather는 축(gather_dim)과 인덱스 텐서가 쌍이고, 축은 범위 안이어야 한다.
  IntegerAttr dimAttr = getGatherDimAttr();
  if (static_cast<bool>(dimAttr) != static_cast<bool>(getIndex()))
    return emitOpError("gather_dim and index must be set together");
  if (dimAttr && (dimAttr.getInt() < 0 || dimAttr.getInt() >= rank))
    return emitOpError("gather_dim out of range");
  return success();
}

LogicalResult GatherOp::verify() {
  // masked gather는 masked-out을 채울 other가 있어야 한다.
  if (static_cast<bool>(getMask()) != static_cast<bool>(getOther()))
    return emitOpError("mask and other must be set together");
  return success();
}
