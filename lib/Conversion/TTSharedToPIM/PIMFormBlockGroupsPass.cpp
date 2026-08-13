#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMFORMBLOCKGROUPS
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct PIMFormBlockGroupsPass
    : public mlir::triton::impl::PIMFormBlockGroupsBase<
          PIMFormBlockGroupsPass> {
  using PIMFormBlockGroupsBase::PIMFormBlockGroupsBase;

  void runOnOperation() override {
    if (groupM == 1 && groupN == 1)
      return; // 묶지 않으면 항등이다
    if (groupM < 1 || groupN < 1) {
      getOperation()->emitError(
          "pim-form-block-groups: group-m and group-n must be at least 1");
      return signalPassFailure();
    }
    WalkResult res = getOperation().walk([&](func::FuncOp func) {
      bool hasAccess = false;
      func.walk([&](pim::AccessOp) { hasAccess = true; });
      if (!hasAccess)
        return WalkResult::advance();
      return succeeded(formGroups(func)) ? WalkResult::advance()
                                         : WalkResult::interrupt();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }

  LogicalResult formGroups(func::FuncOp func) {
    // 묶을 축 loop 를 이름으로 찾는다.
    scf::ForOp rowLoop, colLoop;
    func.walk<WalkOrder::PreOrder>([&](scf::ForOp l) {
      auto a = l->getAttrOfType<StringAttr>("pim.axis");
      if (!a)
        return;
      if (a == "output_row_tile")
        rowLoop = l;
      else if (a == "output_column_tile")
        colLoop = l;
    });
    SmallVector<std::pair<scf::ForOp, int64_t>, 2> grouped;
    if (groupM != 1) {
      if (!rowLoop)
        return func->emitError("pim-form-block-groups: group-m is ")
               << groupM << " but the kernel has no output_row_tile axis";
      grouped.push_back({rowLoop, groupM});
    }
    if (groupN != 1) {
      if (!colLoop)
        return func->emitError("pim-form-block-groups: group-n is ")
               << groupN << " but the kernel has no output_column_tile axis";
      grouped.push_back({colLoop, groupN});
    }

    // loop 를 group 좌표로 바꾼다: 상한 = ceildiv(tiles, g), 몸체 첫머리에서 첫 tile
    // 좌표(iv·g)를 만들어 기존 사용처에 잇는다. origin 산술이 tile 좌표 위에 서 있으므로
    // 이 치환만으로 group 의 시작 위치가 맞는다.
    for (auto [loop, g] : grouped) {
      std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound());
      std::optional<int64_t> step = getConstantIntValue(loop.getStep());
      if (lb != 0 || step != 1)
        return loop->emitError("pim-form-block-groups: the axis loop is not "
                               "the canonical [0, tiles) step-1 form");
      Location loc = loop.getLoc();
      OpBuilder b(loop);
      Value gOut = b.create<arith::ConstantIndexOp>(loc, g);
      Value groups =
          b.create<arith::CeilDivSIOp>(loc, loop.getUpperBound(), gOut);
      loop.getUpperBoundMutable().assign(groups);
      OpBuilder fb(loop.getBody(), loop.getBody()->begin());
      Value gIn = fb.create<arith::ConstantIndexOp>(loc, g);
      auto first = fb.create<arith::MulIOp>(loc, loop.getInductionVar(), gIn);
      loop.getInductionVar().replaceAllUsesExcept(first, first);
    }

    // buffer 축의 tile 크기를 키운다. 어느 loop 가 그 축의 origin 을 모는지가 판정이고,
    // pim-place-data 가 영역을 만들 때 쓰는 판정(collectDrivers)과 같은 것이다.
    SmallVector<scf::ForOp, 2> loops;
    llvm::DenseMap<Operation *, int64_t> factorOf;
    for (auto [l, g] : grouped) {
      loops.push_back(l);
      factorOf[l.getOperation()] = g;
    }
    func.walk([&](pim::AccessOp a) {
      SmallVector<int64_t> sizes(a.getSizes());
      bool changed = false;
      for (unsigned d = 0, e = sizes.size(); d < e; d++) {
        llvm::SmallPtrSet<Operation *, 4> drivers;
        pim::collectDrivers(a.getOrigin()[d], loops, drivers);
        for (Operation *l : drivers) {
          sizes[d] *= factorOf[l];
          changed = true;
        }
      }
      if (changed)
        a.setSizesAttr(DenseI64ArrayAttr::get(a.getContext(), sizes));
    });
    // valid(mask)는 건드리지 않는다 — 소비자는 pim-place-data 의 원본 크기 복원뿐이고
    // 그 판정은 식의 구조만 보므로, 잘라내기는 새 sizes 로 다시 계산된다.

    // tile tensor 를 group 크기로 다시 적는다. collectLocalTiles 의 owner 사슬이 정확히
    // 다시 적어야 하는 값의 집합이다. load 사슬은 대응 access 의 sizes 가, 누산기 사슬은
    // 그것을 저장하는 access 의 sizes 가 새 shape 이다. 저장 access 의 rank 가 tensor 보다
    // 크면(split-K 의 출력면 축) 뒤쪽 축만 쓴다.
    SmallVector<Value> roots;
    llvm::DenseMap<Value, Value> owner;
    pim::collectLocalTiles(func, roots, owner);

    auto typeFrom = [](pim::AccessOp access, Value root) -> Type {
      auto old = cast<RankedTensorType>(root.getType());
      ArrayRef<int64_t> sizes = access.getSizes();
      return RankedTensorType::get(sizes.take_back(old.getRank()),
                                   old.getElementType());
    };
    llvm::DenseMap<Value, Type> newTypeOf;
    for (Value r : roots)
      if (auto load = r.getDefiningOp<pim::LoadOp>())
        newTypeOf[r] =
            typeFrom(load.getAccess().getDefiningOp<pim::AccessOp>(), r);
    func.walk([&](pim::StoreOp s) {
      Value root = owner.lookup(s.getValue());
      if (!root || newTypeOf.count(root))
        return;
      newTypeOf[root] =
          typeFrom(s.getAccess().getDefiningOp<pim::AccessOp>(), root);
    });
    for (Value r : roots)
      if (!newTypeOf.count(r))
        return func->emitError(
            "pim-form-block-groups: cannot decide the group shape of a tile "
            "that is neither loaded from nor stored to a pim.access");
    for (auto &[v, root] : owner)
      Value(v).setType(newTypeOf[root]);
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMFormBlockGroupsPass() {
  return std::make_unique<PIMFormBlockGroupsPass>();
}
