#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

mlir::pim::TargetProfileAttr mlir::pim::targetProfile(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    module = dyn_cast<ModuleOp>(op);
  return module ? module->getAttrOfType<TargetProfileAttr>("pim.target_profile")
                : TargetProfileAttr();
}

void mlir::pim::collectLocalTiles(Operation *scope,
                                  SmallVectorImpl<Value> &roots,
                                  llvm::DenseMap<Value, Value> &owner) {
  // 정의가 사용보다 앞이므로 pre-order 한 번으로 끝난다.
  scope->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto load = dyn_cast<pim::LoadOp>(op)) {
      roots.push_back(load.getResult());
      owner[load.getResult()] = load.getResult();
      return;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs()))
        if (Value o = owner.lookup(forOp.getInitArgs()[i])) {
          owner[arg] = o;
          owner[forOp.getResult(i)] = o;
        }
      return;
    }
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp || linalgOp.getNumDpsInits() != 1 || op->getNumResults() != 1)
      return;
    Value out = linalgOp.getDpsInits()[0];
    Value root = owner.lookup(out);
    // 누산기에 이어 쓰는 계산이면 그 버퍼를 그대로 쓰고, 새 tile 을 만드는 계산이면
    // 버퍼를 새로 잡는다.
    if (!root) {
      if (!isa<RankedTensorType>(op->getResult(0).getType()))
        return;
      root = out;
      roots.push_back(out);
      owner[out] = out;
    }
    owner[op->getResult(0)] = root;
  });
}

int64_t mlir::pim::localTileBytes(Operation *scope) {
  SmallVector<Value> roots;
  llvm::DenseMap<Value, Value> owner;
  collectLocalTiles(scope, roots, owner);
  int64_t bytes = 0;
  for (Value r : roots) {
    auto t = cast<RankedTensorType>(r.getType());
    bytes += t.getNumElements() *
             std::max<int64_t>(1, t.getElementType().getIntOrFloatBitWidth() / 8);
  }
  return bytes;
}

int64_t mlir::pim::tileElementsAlong(scf::ForOp axis,
                                     ArrayRef<scf::ForOp> loops) {
  int64_t elems = 0;
  bool conflict = false;
  axis.walk([&](pim::AccessOp a) {
    ArrayRef<int64_t> sizes = a.getSizes();
    for (unsigned d = 0, e = sizes.size(); d < e; d++) {
      llvm::SmallPtrSet<Operation *, 4> drivers;
      collectDrivers(a.getOrigin()[d], loops, drivers);
      if (!drivers.contains(axis))
        continue;
      // 공간 축(pim.axis 태그) 오프셋이 함께 실려 있어도 — split-K 커널의
      // K 주소 = pid_k·K_PER + k·BLOCK_K 가 그 예다 — 이 loop 한 회전이
      // 옮기는 원소 수는 access 크기 그대로다. 태그 없는 다른 loop 가 함께
      // 몰면 크기를 단정할 수 없으므로 그 차원은 건너뛴다.
      bool foreign = false;
      for (Operation *op : drivers)
        if (op != axis && !op->getAttrOfType<StringAttr>("pim.axis"))
          foreign = true;
      if (foreign)
        continue;
      if (elems && elems != sizes[d])
        conflict = true;
      elems = sizes[d];
    }
  });
  return conflict ? 0 : elems;
}

void mlir::pim::collectDrivers(Value v, ArrayRef<scf::ForOp> loops,
                               llvm::SmallPtrSetImpl<Operation *> &drivers) {
  SmallVector<Value> work{v};
  llvm::DenseSet<Value> seen;
  while (!work.empty()) {
    Value c = work.pop_back_val();
    if (!seen.insert(c).second)
      continue;
    if (auto ba = dyn_cast<BlockArgument>(c)) {
      for (scf::ForOp l : loops)
        if (ba == l.getInductionVar())
          drivers.insert(l);
      continue;
    }
    if (Operation *def = c.getDefiningOp())
      work.append(def->getOperands().begin(), def->getOperands().end());
  }
}
