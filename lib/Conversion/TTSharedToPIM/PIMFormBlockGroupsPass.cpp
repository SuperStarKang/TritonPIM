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

static Value toIndex(OpBuilder &b, Location loc, Value value) {
  return isa<IndexType>(value.getType())
             ? value
             : b.create<arith::IndexCastOp>(loc, b.getIndexType(), value)
                   .getResult();
}

struct PIMFormBlockGroupsPass
    : public mlir::triton::impl::PIMFormBlockGroupsBase<
          PIMFormBlockGroupsPass> {
  using PIMFormBlockGroupsBase::PIMFormBlockGroupsBase;

  void runOnOperation() override {
    if (groupM < 0 || groupN < 0 || groupK < 0) {
      getOperation()->emitError("pim-form-block-groups: group-m, group-n and "
                                "group-k must be 0 (one block) or positive");
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

  // 계층 2는 주어진 Block 개수대로 IR을 재구성하고, target-neutral 묶음을
  // #pim.block_group으로 기록한다. 0은 가장 작은 항등 group인 Block 하나다.
  // 하드웨어 실행 눈금과 padding은 target lowering의 책임이다.
  LogicalResult formGroups(func::FuncOp func) {
    int64_t g[3] = {groupM ? groupM : 1, groupN ? groupN : 1,
                    groupK ? groupK : 1};
    if (g[0] != 1 || g[1] != 1 || g[2] != 1)
      if (failed(regroup(func, g[0], g[1], g[2])))
        return failure();
    return recordGroup(func, g);
  }

  LogicalResult regroup(func::FuncOp func, int64_t gM, int64_t gN,
                        int64_t gK) {
    // 묶을 축 loop 를 이름으로 찾는다.
    scf::ForOp rowLoop, colLoop;
    func.walk<WalkOrder::PreOrder>([&](scf::ForOp l) {
      auto a = l->getAttrOfType<pim::AxisAttr>("pim.axis");
      if (!a)
        return;
      if (a.getName().getValue() == "m")
        rowLoop = l;
      else if (a.getName().getValue() == "n")
        colLoop = l;
    });
    SmallVector<std::pair<scf::ForOp, int64_t>, 2> grouped;
    if (gM != 1) {
      if (!rowLoop)
        return func->emitError("pim-form-block-groups: group-m is ")
               << gM << " but the kernel has no m axis";
      grouped.push_back({rowLoop, gM});
    }
    if (gN != 1) {
      if (!colLoop)
        return func->emitError("pim-form-block-groups: group-n is ")
               << gN << " but the kernel has no n axis";
      grouped.push_back({colLoop, gN});
    }
    if (gK != 1) {
      // reduction loop는 PIMFormCompute가 linalg accumulator 의미에서 명시한
      // 표식으로만 찾는다. access origin은 연산 의미가 아니라 주소 정보다.
      SmallVector<scf::ForOp> all;
      func.walk<WalkOrder::PreOrder>([&](scf::ForOp l) { all.push_back(l); });
      SmallVector<scf::ForOp> reductions;
      for (scf::ForOp loop : all)
        if (loop->getAttrOfType<StringAttr>("pim.reduction"))
          reductions.push_back(loop);
      if (reductions.size() != 1)
        return func->emitError("pim-form-block-groups: group-k is ")
               << gK << " but " << reductions.size()
               << " semantic reduction loops are present; exactly one is "
                  "needed to know which loop to widen";
      grouped.push_back({reductions.front(), gK});
    }

    // m/n의 새 상한은 가장 바깥 축 loop 앞에 만든다. reduction 상한은 원래 runtime K
    // 식과 같은 위치인 reduction loop 앞에 둔다. 뒤의 pim-map-devices는 Common
    // scheduling boundary를 만들 때 이 불변 bound 식을 그 경계 앞으로 다시 만든다.
    scf::ForOp outermost;
    func.walk<WalkOrder::PreOrder>([&](scf::ForOp l) {
      if (!l->getAttrOfType<pim::AxisAttr>("pim.axis"))
        return WalkResult::advance();
      outermost = l;
      return WalkResult::interrupt();
    });

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
      // reduction loop 는 커널의 정수 타입(i32)일 수 있으므로 상수를 유도변수 타입으로
      // 만든다. 상한의 자리도 다르다: 축 loop 의 상한은 nest 밖(pim-map-devices 가
      // 밖에서 읽는다), reduction 의 상한은 그 loop 앞 — 원래 상한이 nest 안에서
      // 정의되기 때문이다(pim-place-data 의 rebuild 가 필요하면 밖으로 복제한다).
      Type ivType = loop.getInductionVar().getType();
      bool isAxis = loop->getAttrOfType<pim::AxisAttr>("pim.axis") != nullptr;
      OpBuilder b(isAxis ? outermost.getOperation() : loop.getOperation());
      Value gOut =
          b.create<arith::ConstantOp>(loc, b.getIntegerAttr(ivType, g));
      Value groups =
          b.create<arith::CeilDivSIOp>(loc, loop.getUpperBound(), gOut);
      loop.getUpperBoundMutable().assign(groups);
      OpBuilder fb(loop.getBody(), loop.getBody()->begin());
      Value gIn =
          fb.create<arith::ConstantOp>(loc, fb.getIntegerAttr(ivType, g));
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
    SmallVector<scf::ForOp> allLoops;
    func.walk<WalkOrder::PreOrder>(
        [&](scf::ForOp loop) { allLoops.push_back(loop); });
    SmallVector<pim::AccessOp> accesses;
    func.walk([&](pim::AccessOp access) { accesses.push_back(access); });
    for (pim::AccessOp a : accesses) {
      SmallVector<int64_t> sizes(a.getSizes());
      SmallVector<Value> valid(a.getValid());
      bool changed = false;
      for (unsigned d = 0, e = sizes.size(); d < e; d++) {
        llvm::SmallPtrSet<Operation *, 4> drivers;
        pim::collectDrivers(a.getOrigin()[d], loops, drivers);
        int64_t oldSize = sizes[d];
        for (Operation *l : drivers)
          sizes[d] *= factorOf[l];
        if (sizes[d] != oldSize) {
          changed = true;

          // valid는 원래 Block 시작과 Block 크기로 clamp되어 있다. Group 시작은
          // 위의 IV 치환으로 이미 맞았으므로, 원본 extent만 복원해 확대된 sizes로
          // 다시 clamp한다. pad 값은 같은 buffer 축의 invalid tail에 그대로 적용된다.
          if (!valid.empty()) {
            OpBuilder b(a);
            Location loc = a.getLoc();
            if (std::optional<int64_t> constant =
                    getConstantIntValue(valid[d])) {
              if (*constant != 0 && *constant != oldSize)
                return a->emitError(
                           "pim-form-block-groups: cannot extend a constant "
                           "partial valid extent on buffer axis ")
                       << d;
              valid[d] = b.create<arith::ConstantIndexOp>(
                  loc, *constant == 0 ? 0 : sizes[d]);
            } else {
              Value extent =
                  pim::recoverExtentFromValid(valid[d], allLoops);
              if (!extent)
                return a->emitError(
                           "pim-form-block-groups: cannot recover the original "
                           "extent from valid on buffer axis ")
                       << d;
              Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
              Value capacity =
                  b.create<arith::ConstantIndexOp>(loc, sizes[d]);
              Value remaining = b.create<arith::SubIOp>(
                  loc, toIndex(b, loc, extent), a.getOrigin()[d]);
              Value nonnegative =
                  b.create<arith::MaxSIOp>(loc, remaining, zero);
              valid[d] =
                  b.create<arith::MinSIOp>(loc, nonnegative, capacity);
            }
          }
        }
      }
      if (changed) {
        a.setSizesAttr(DenseI64ArrayAttr::get(a.getContext(), sizes));
        if (!valid.empty())
          a.getValidMutable().assign(valid);
      }
    }

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

  // regroup 뒤 IR에서 Group 원소 shape를 읽고, 이를 block 개수로 나눠 첫째 계층인
  // Block shape를 복원한다. Common IR에는 Block 원소 shape와 Group의 block 개수만
  // 기록하며 Group 원소 shape는 두 값의 곱으로 파생한다.
  LogicalResult recordGroup(func::FuncOp func, const int64_t g[3]) {
    SmallVector<scf::ForOp> loops;
    func.walk<WalkOrder::PreOrder>([&](scf::ForOp l) { loops.push_back(l); });
    scf::ForOp axis[3]; // m, n, k
    for (scf::ForOp l : loops) {
      auto a = l->getAttrOfType<pim::AxisAttr>("pim.axis");
      if (!a)
        continue;
      if (a.getName().getValue() == "m")
        axis[0] = l;
      else if (a.getName().getValue() == "n")
        axis[1] = l;
    }
    for (scf::ForOp loop : loops)
      if (loop->getAttrOfType<StringAttr>("pim.reduction")) {
        if (axis[2])
          return func->emitError(
              "pim-form-block-groups: more than one semantic reduction "
              "loop is present");
        axis[2] = loop;
      }
    static const char *name[3] = {"m", "n", "k"};
    int64_t blocks[3] = {g[0], g[1], g[2]};
    // 고정 [m,n,k] 표현에서 실제 compute loop가 없는 축은 곱의 항등원 1이다.
    int64_t blockShape[3] = {1, 1, 1};
    for (int i = 0; i < 3; i++) {
      if (!axis[i])
        continue;
      int64_t groupShape = pim::tileElementsAlong(axis[i], loops);
      if (groupShape <= 0 || groupShape % blocks[i] != 0)
        return func->emitError("pim-form-block-groups: cannot tell how many "
                               "elements one block moves on the ")
               << name[i] << " axis";
      blockShape[i] = groupShape / blocks[i];
    }
    OpBuilder b(func.getContext());
    func->setAttr("pim.block",
                  pim::BlockAttr::get(func.getContext(),
                                      b.getDenseI64ArrayAttr(blockShape)));
    func->setAttr("pim.block_group",
                  pim::BlockGroupAttr::get(func.getContext(),
                                           b.getDenseI64ArrayAttr(blocks)));
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMFormBlockGroupsPass() {
  return std::make_unique<PIMFormBlockGroupsPass>();
}
