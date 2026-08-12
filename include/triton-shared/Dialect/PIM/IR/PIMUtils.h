#ifndef TRITON_SHARED_DIALECT_PIM_IR_PIM_UTILS_H_
#define TRITON_SHARED_DIALECT_PIM_IR_PIM_UTILS_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
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

// v 를 만드는 식의 잎까지 거슬러 올라가, loops 중 어느 것의 induction variable 이 이 값을
// 모는지 모은다. "이 좌표 축이 이 buffer 차원을 움직이는가" 를 가르는 판정이고,
// pim-map-devices(분할 눈금)와 pim-place-data(축별 region)가 같은 답을 써야 한다.
void collectDrivers(Value v, ArrayRef<scf::ForOp> loops,
                    llvm::SmallPtrSetImpl<Operation *> &drivers);

} // namespace pim
} // namespace mlir

#endif // TRITON_SHARED_DIALECT_PIM_IR_PIM_UTILS_H_
