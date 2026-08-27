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
    if (groupM < 0 || groupN < 0 || groupK < 0) {
      getOperation()->emitError("pim-form-block-groups: group-m, group-n and "
                                "group-k must be 0 (automatic) or positive");
      return signalPassFailure();
    }
    profile = pim::targetProfile(getOperation());
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

  pim::TargetProfileAttr profile;

  // 계층 2는 group 값(사용자 지정 또는 local-memory 기반 자동 산출)대로 IR을
  // 재구성하고, target-neutral block 묶음을 #pim.block_group으로 기록한다.
  // 하드웨어 실행 눈금과 padding은 target lowering의 책임이다.
  LogicalResult formGroups(func::FuncOp func) {
    int64_t g[3] = {groupM, groupN, groupK}; // 0 = 자동
    if (g[0] == 0 || g[1] == 0 || g[2] == 0)
      if (failed(autoGroups(func, g)))
        return failure();
    if (g[0] != 1 || g[1] != 1 || g[2] != 1)
      if (failed(regroup(func, g[0], g[1], g[2])))
        return failure();
    return recordGroup(func, g);
  }

  // 자동 group 산출 — "빠른 메모리에 들어갈 때까지 m → n → k 순서로 번갈아 1씩
  // 키운다". local 예산을 알 수 없거나 셈에 필요한 정보가 없으면 자동 축을
  // 1(항등)로 둔다. 하나의 block group은 intra-device 계층이 협력 실행하므로 local
  // buffer 한 벌을 기준으로 계산한다.
  LogicalResult autoGroups(func::FuncOp func, int64_t g[3]) {
    auto identity = [&] {
      for (int i = 0; i < 3; i++)
        if (g[i] == 0)
          g[i] = 1;
      return success();
    };
    if (!profile)
      return identity();
    IntegerAttr local = profile.getUsableLocalCapacityBytes();
    if (!local || local.getInt() <= 0)
      return identity();
    // 축 loop: m/n 은 pim.axis 이름으로, k 는 reduction loop(pim.axis 없이 origin
    // 을 모는 loop, 하나일 때만)로 찾는다 — regroup/fillUnits 와 같은 판정이다.
    SmallVector<scf::ForOp> loops;
    func.walk<WalkOrder::PreOrder>([&](scf::ForOp l) { loops.push_back(l); });
    scf::ForOp axis[3] = {};
    for (scf::ForOp l : loops) {
      auto a = l->getAttrOfType<pim::AxisAttr>("pim.axis");
      if (!a)
        continue;
      if (a.getName().getValue() == "m")
        axis[0] = l;
      else if (a.getName().getValue() == "n")
        axis[1] = l;
    }
    llvm::SmallPtrSet<Operation *, 4> reds;
    func.walk([&](pim::AccessOp a) {
      for (Value o : a.getOrigin()) {
        llvm::SmallPtrSet<Operation *, 4> ds;
        pim::collectDrivers(o, loops, ds);
        for (Operation *op : ds)
          if (!op->getAttrOfType<pim::AxisAttr>("pim.axis"))
            reds.insert(op);
      }
    });
    if (reds.size() == 1)
      axis[2] = cast<scf::ForOp>(*reds.begin());

    // access 마다 "각 차원을 어느 축이 모는가"와 원소 바이트를 모은다. 원소 타입은
    // 그 access 를 읽고 쓰는 load/store 의 tensor 타입에서 온다.
    struct Buf {
      int64_t elemBytes;
      SmallVector<std::pair<int, int64_t>, 3> dims; // {축 번호(-1=무관), tile 크기}
    };
    SmallVector<Buf> bufs;
    SmallVector<scf::ForOp, 3> axisLoops;
    for (int i = 0; i < 3; i++)
      if (axis[i])
        axisLoops.push_back(axis[i]);
    bool unknown = false;
    func.walk([&](pim::AccessOp a) {
      Type elem;
      for (Operation *user : a->getUsers()) {
        if (auto load = dyn_cast<pim::LoadOp>(user))
          elem = cast<RankedTensorType>(load.getResult().getType())
                     .getElementType();
        else if (auto store = dyn_cast<pim::StoreOp>(user))
          if (!elem)
            elem = cast<RankedTensorType>(store.getValue().getType())
                       .getElementType();
      }
      if (!elem) {
        unknown = true;
        return;
      }
      Buf b;
      b.elemBytes =
          std::max<int64_t>(1, elem.getIntOrFloatBitWidth() / 8);
      ArrayRef<int64_t> sizes = a.getSizes();
      for (unsigned d = 0, e = sizes.size(); d < e; d++) {
        llvm::SmallPtrSet<Operation *, 4> ds;
        pim::collectDrivers(a.getOrigin()[d], axisLoops, ds);
        int tag = -1;
        if (ds.size() == 1)
          for (int i = 0; i < 3; i++)
            if (axis[i] && ds.contains(axis[i]))
              tag = i;
        b.dims.push_back({tag, sizes[d]});
      }
      bufs.push_back(b);
    });
    if (unknown || bufs.empty())
      return identity();

    auto bytesFor = [&](const int64_t f[3]) {
      int64_t total = 0;
      for (const Buf &b : bufs) {
        int64_t elems = 1;
        for (auto [tag, size] : b.dims)
          elems *= size * (tag >= 0 ? f[tag] : 1);
        total += elems * b.elemBytes;
      }
      return total;
    };

    // 시작값: 사용자 고정 축은 그 값, 자동 축은 1. 없는 축은 못 키운다. 시작값부터
    // 예산을 넘으면 group 을 만들지 않는다.
    int64_t f[3];
    bool frozen[3];
    for (int i = 0; i < 3; i++) {
      f[i] = g[i] ? g[i] : 1;
      frozen[i] = g[i] != 0 || !axis[i];
    }
    int64_t budget = local.getInt();
    if (bytesFor(f) <= budget) {
      bool grew = true;
      while (grew) {
        grew = false;
        for (int i = 0; i < 3; i++) {
          if (frozen[i])
            continue;
          f[i]++;
          if (bytesFor(f) > budget) {
            f[i]--;
            frozen[i] = true;
          } else {
            grew = true;
          }
        }
      }
    }
    for (int i = 0; i < 3; i++)
      if (g[i] == 0)
        g[i] = f[i];
    return success();
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
      // reduction loop 를 찾는다: pim.axis 가 없으면서 access 의 origin 을 모는 loop.
      // 정확히 하나여야 어느 loop 를 넓힐지 유일하다.
      SmallVector<scf::ForOp> all;
      func.walk<WalkOrder::PreOrder>([&](scf::ForOp l) { all.push_back(l); });
      llvm::SmallPtrSet<Operation *, 4> reductions;
      func.walk([&](pim::AccessOp a) {
        for (Value o : a.getOrigin()) {
          llvm::SmallPtrSet<Operation *, 4> ds;
          pim::collectDrivers(o, all, ds);
          for (Operation *op : ds)
            if (!op->getAttrOfType<pim::AxisAttr>("pim.axis"))
              reductions.insert(op);
        }
      });
      if (reductions.size() != 1)
        return func->emitError("pim-form-block-groups: group-k is ")
               << gK << " but " << reductions.size()
               << " reduction loops drive memory accesses; exactly one is "
                  "needed to know which loop to widen";
      grouped.push_back({cast<scf::ForOp>(*reductions.begin()), gK});
    }

    // 새 상한은 전부 가장 바깥 축 loop 앞에 만든다. 안쪽 loop 앞에 두면 그 값이 바깥
    // loop 몸체 안에 갇혀서, 뒤의 pim-map-devices 가 nest 밖에서 상한을 읽을 때 지배가
    // 깨진다 — build-logical-grid 가 상한을 전부 entry 에 만드는 것과 같은 이유다.
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

  // 현재 IR에서 group의 useful shape을 읽어 target-neutral block group을 기록한다.
  // k가 없는 elementwise 연산은 k 축을 1로 둔다. target padding은 공통 attribute를
  // 변경하지 않고 TargetSpec과 target lowering이 target IR에 materialize한다.
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
    llvm::SmallPtrSet<Operation *, 4> reds;
    func.walk([&](pim::AccessOp a) {
      for (Value o : a.getOrigin()) {
        llvm::SmallPtrSet<Operation *, 4> ds;
        pim::collectDrivers(o, loops, ds);
        for (Operation *op : ds)
          if (!op->getAttrOfType<pim::AxisAttr>("pim.axis"))
            reds.insert(op);
      }
    });
    if (reds.size() == 1)
      axis[2] = cast<scf::ForOp>(*reds.begin());
    static const char *name[3] = {"m", "n", "k"};
    int64_t tiles[3] = {g[0], g[1], g[2]};
    int64_t useful[3] = {1, 1, 1};
    for (int i = 0; i < 3; i++) {
      if (!axis[i])
        continue;
      int64_t block = pim::tileElementsAlong(axis[i], loops);
      if (block <= 0)
        return func->emitError("pim-form-block-groups: cannot tell how many "
                               "elements one block moves on the ")
               << name[i] << " axis";
      useful[i] = block;
    }
    OpBuilder b(func.getContext());
    func->setAttr("pim.block_group",
                  pim::BlockGroupAttr::get(func.getContext(),
                                           b.getDenseI64ArrayAttr(tiles),
                                           b.getDenseI64ArrayAttr(useful)));
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMFormBlockGroupsPass() {
  return std::make_unique<PIMFormBlockGroupsPass>();
}
