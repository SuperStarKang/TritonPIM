#ifndef TRITON_SHARED_DIALECT_PIM_IR_PIM_UTILS_H_
#define TRITON_SHARED_DIALECT_PIM_IR_PIM_UTILS_H_

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

namespace mlir {
namespace pim {

// op을 담은 module에 붙은 공통 profile. 없으면 널이다.
TargetProfileAttr targetProfile(Operation *op);

// scope 안에서 tile 버퍼를 새로 잡아야 하는 tensor 값(roots)과, 모든 tensor 값에서
// 그 값이 사는 버퍼의 대표 값으로 가는 대응(owner)을 만든다.
// pim.load 결과와 새 tile 을 만드는 linalg 계산의 outs 가 각각 버퍼 하나이고,
// scf.for 의 tensor iter_arg 는 초기값의 버퍼를 그대로 쓴다.
void collectLocalTiles(Operation *scope, SmallVectorImpl<Value> &roots,
                       llvm::DenseMap<Value, Value> &owner);

// lane 하나가 동시에 들고 있어야 하는 tile 버퍼의 바이트 합.
int64_t localTileBytes(Operation *scope);

} // namespace pim
} // namespace mlir

#endif // TRITON_SHARED_DIALECT_PIM_IR_PIM_UTILS_H_
