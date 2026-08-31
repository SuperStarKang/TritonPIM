#ifndef TRITON_SHARED_DIALECT_PIM_IR_PIM_UTILS_H_
#define TRITON_SHARED_DIALECT_PIM_IR_PIM_UTILS_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

namespace mlir {
namespace pim {

// op을 담은 module에 붙은 공통 profile. 없으면 널이다.
TargetProfileAttr targetProfile(Operation *op);

// func/device에 기록된 3계층으로부터 원소 shape를 계산한다. 배열의 고정 순서는
// [m, n, k]다. 없는 축은 곱의 항등원인 1이다. 파생 shape를 attribute에 중복
// 저장하지 않는다.
SmallVector<int64_t, 3> blockShape(Operation *op);
SmallVector<int64_t, 3> groupShape(Operation *op);
SmallVector<int64_t, 3> executionTileShape(Operation *op);

// homogeneous bank topology의 device 전체 capacity limit. profile이 없거나
// 곱이 int64 범위를 넘으면 nullopt다.
std::optional<int64_t> unitCapacityLimitBytes(TargetProfileAttr profile);

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

// valid(mask) 산술에서 이 buffer 축의 원본 element extent를 복원한다. Triton
// lowering이 만드는 clamp 두 형태를 공통으로 해석하며, loop와 무관한 extent를
// 찾을 수 없으면 null Value를 돌려준다. Block Group valid 재구성과 placement tail
// clamp가 반드시 같은 원본 extent를 사용한다.
Value recoverExtentFromValid(Value valid, ArrayRef<scf::ForOp> loops);

// 이 축 loop 가 tile 하나에서 옮기는 원소 수. pim.access 의 sizes 가 그 값이고, 어느
// 차원인지는 origin 을 이 loop 가 단독으로 모는가로 가른다. 축을 모는 차원이 여럿인데
// 크기가 다르면 답이 하나가 아니므로 0 을 돌려준다. block-group 형성,
// device mapping, hierarchy 검증이 같은 판정을 쓴다.
int64_t tileElementsAlong(scf::ForOp axis, ArrayRef<scf::ForOp> loops);

} // namespace pim
} // namespace mlir

#endif // TRITON_SHARED_DIALECT_PIM_IR_PIM_UTILS_H_
