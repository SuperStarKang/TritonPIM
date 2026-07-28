#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMPARTITION
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// v를 계산하는 식을 map대로 치환해 nest 밖에 다시 만든다. nest 밖에서 정의된 값은 이미
// 유닛 좌표와 무관하므로 그대로 쓴다. nest 안의 block argument(주소를 나르는 iter_arg 등)를
// 만나면 유닛 좌표로 바꿀 방법이 없으므로 널을 낸다.
static Value rebuild(OpBuilder &b, Value v, IRMapping &map, Operation *nest) {
  if (Value m = map.lookupOrNull(v))
    return m;
  Operation *def = v.getDefiningOp();
  if (!def)
    return nest->isAncestor(v.getParentBlock()->getParentOp()) ? Value() : v;
  if (!nest->isAncestor(def))
    return v;
  IRMapping local;
  for (Value o : def->getOperands()) {
    Value r = rebuild(b, o, map, nest);
    if (!r)
      return Value();
    local.map(o, r);
  }
  Value r = b.clone(*def, local)->getResult(cast<OpResult>(v).getResultNumber());
  map.map(v, r);
  return r;
}

// v가 유도변수를 쓰는 loop를 loops 중에서 찾는다. 하나도 없으면 널 — 그 축은 유닛이 몇 번을
// 돌든 타일 하나만큼이다. 둘 이상이면 ambiguous — 유닛 하나가 맡는 영역이 직사각형이 아니라
// origin/sizes로 못 적는다. 둘을 구별해야 한다: 널로 뭉뚱그리면 크기를 실제보다 작게 적어
// host가 필요한 데이터를 덜 보낸다.
static scf::ForOp driverLoop(Value v, ArrayRef<scf::ForOp> loops,
                             bool &ambiguous) {
  SmallVector<Value> work{v};
  llvm::DenseSet<Value> seen;
  scf::ForOp found;
  while (!work.empty()) {
    Value c = work.pop_back_val();
    if (!seen.insert(c).second)
      continue;
    if (auto ba = dyn_cast<BlockArgument>(c)) {
      for (scf::ForOp l : loops)
        if (ba == l.getInductionVar()) {
          if (found && found != l) {
            ambiguous = true;
            return nullptr;
          }
          found = l;
        }
      continue;
    }
    if (Operation *d = c.getDefiningOp())
      work.append(d->getOperands().begin(), d->getOperands().end());
  }
  return found;
}

// loop를 유닛 u(총 n개) 몫으로 좁히고, 그 유닛이 실제로 도는 횟수를 index로 낸다.
// 배정량은 올림 나눗셈이고 상한은 원래 상한으로 자르므로, 반복 횟수가 n으로
// 나누어떨어지지 않으면 마지막 유닛이 덜 맡고, 유닛이 반복보다 많으면 빈 범위가 된다.
// 그래서 배정량이 아니라 좁힌 범위에서 실제 횟수를 다시 구한다 — unit_data 크기가
// 여기서 나오므로, 배정량을 쓰면 마지막 유닛이 원본 버퍼 밖을 가리킨다.
static Value narrowLoop(OpBuilder &b, scf::ForOp loop, Value u, int64_t n) {
  Location loc = loop.getLoc();
  Value lb = loop.getLowerBound(), ub = loop.getUpperBound(), step = loop.getStep();
  Type t = lb.getType();
  Value uc = isa<IndexType>(t)
                 ? u
                 : b.create<arith::IndexCastOp>(loc, t, u).getResult();
  Value cn = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(t, n));
  Value trips = b.create<arith::CeilDivSIOp>(
      loc, b.create<arith::SubIOp>(loc, ub, lb), step);
  Value per = b.create<arith::CeilDivSIOp>(loc, trips, cn);
  Value chunk = b.create<arith::MulIOp>(loc, per, step);
  Value newLb = b.create<arith::AddIOp>(loc, lb,
                                        b.create<arith::MulIOp>(loc, uc, chunk));
  Value newUb = b.create<arith::MinSIOp>(
      loc, b.create<arith::AddIOp>(loc, newLb, chunk), ub);
  loop.getLowerBoundMutable().assign(newLb);
  loop.getUpperBoundMutable().assign(newUb);
  Value zero = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(t, 0));
  Value span = b.create<arith::MaxSIOp>(
      loc, b.create<arith::SubIOp>(loc, newUb, newLb), zero);
  Value mine = b.create<arith::CeilDivSIOp>(loc, span, step);
  return isa<IndexType>(t)
             ? mine
             : b.create<arith::IndexCastOp>(loc, b.getIndexType(), mine)
                   .getResult();
}

struct PIMPartitionPass
    : public mlir::triton::impl::PIMPartitionBase<PIMPartitionPass> {
  using PIMPartitionBase::PIMPartitionBase;

  void runOnOperation() override {
    getOperation().walk([&](func::FuncOp func) { partitionFunc(func); });
  }

  void partitionFunc(func::FuncOp func) {
    auto axesAttr = func->getAttrOfType<ArrayAttr>("pim.split_axes");
    if (!axesAttr || axesAttr.empty())
      return; // 쪼갤 축이 없으면 유닛 형태도 없다.
    if (splits.size() != axesAttr.size()) {
      func->emitError("pim-partition: splits ")
          << splits.size() << " but pim.split_axes has " << axesAttr.size();
      return signalPassFailure();
    }

    // reduction 축은 pim.split_axes와 같은 순서(바깥→안쪽)로 매긴다.
    SmallVector<scf::ForOp> reductionLoops;
    func.walk([&](scf::ForOp l) {
      if (l->hasAttr("pim.reduction_combine"))
        reductionLoops.push_back(l);
    });

    // 쪼개는 축(유닛 수 > 1)만 모은다. 이름·유닛 수·해당 loop가 자리 순서로 대응한다.
    SmallVector<Attribute> names;
    SmallVector<int64_t> counts;
    SmallVector<scf::ForOp> loops;
    SmallVector<Attribute> combineOver; // 부분 결과가 생기는 축
    StringAttr combine;
    unsigned reductionSeen = 0;
    for (auto [i, a] : llvm::enumerate(axesAttr)) {
      auto d = cast<DictionaryAttr>(a);
      bool isReduction =
          cast<StringAttr>(d.get("source")).getValue() == "reduction";
      unsigned rIdx = reductionSeen;
      if (isReduction)
        reductionSeen++;
      if (splits[i] <= 1)
        continue;
      auto axis = cast<StringAttr>(d.get("axis"));
      scf::ForOp loop;
      if (isReduction) {
        loop = reductionLoops[rIdx];
        auto c = cast<StringAttr>(d.get("combine"));
        // pim.unit_data는 결합 연산자를 하나만 나른다. 쪼갠 reduction 축들의 연산자가
        // 다르면 어느 것으로 합칠지 적을 수 없다.
        if (combine && combine != c) {
          func->emitError("pim-partition: split reduction axes use different "
                          "combine operators");
          return signalPassFailure();
        }
        combine = c;
        combineOver.push_back(axis);
      } else {
        func.walk([&](scf::ForOp l) {
          auto g = l->getAttrOfType<StringAttr>("pim.grid_dim");
          if (g && g == axis)
            loop = l;
        });
      }
      names.push_back(axis);
      counts.push_back(splits[i]);
      loops.push_back(loop);
    }
    if (names.empty())
      return; // 전부 1이면 나눌 것이 없다.

    int64_t total = 1;
    for (int64_t c : counts)
      total *= c;
    if (numUnits > 0 && total > numUnits) {
      func->emitError("pim-partition: splits need ")
          << total << " units but the machine has " << numUnits;
      return signalPassFailure();
    }
    // 유닛 하나가 한 순간에 들고 있어야 하는 tile 합계. 넘으면 어떤 target도 돌릴 수 없다.
    if (localMemoryBytes > 0)
      if (auto resident = func->getAttrOfType<ArrayAttr>("pim.co_resident")) {
        int64_t bytes = 0;
        for (Attribute e : resident)
          bytes += cast<IntegerAttr>(cast<DictionaryAttr>(e).get("bytes")).getInt();
        if (bytes > localMemoryBytes) {
          func->emitError("pim-partition: co-resident tiles need ")
              << bytes << " bytes but a unit has " << localMemoryBytes;
          return signalPassFailure();
        }
      }

    // 축 loop nest 전체를 감싸야 하므로 가장 바깥 축 loop를 찾는다(pre-order = 바깥부터).
    scf::ForOp outermost;
    func.walk<WalkOrder::PreOrder>([&](scf::ForOp l) {
      if (!outermost && llvm::is_contained(loops, l))
        outermost = l;
    });

    MLIRContext *ctx = func.getContext();
    Location loc = outermost.getLoc();
    OpBuilder b(outermost);
    auto units = b.create<pim::UnitsOp>(loc, ArrayAttr::get(ctx, names),
                                        b.getDenseI64ArrayAttr(counts));
    SmallVector<Type> argTys(names.size(), b.getIndexType());
    SmallVector<Location> argLocs(names.size(), loc);
    Block *body = b.createBlock(&units.getBody(), {}, argTys, argLocs);
    outermost->moveBefore(body, body->end());

    // 유닛 몫으로 loop를 좁히고, 좁힌 하한을 기억한다(unit_data의 시작 위치가 여기서 나온다).
    b.setInsertionPoint(outermost);
    IRMapping toUnitLb;
    llvm::DenseMap<Operation *, Value> perUnitTrips;
    for (auto [i, l] : llvm::enumerate(loops)) {
      Value per = narrowLoop(b, l, body->getArgument(i), counts[i]);
      perUnitTrips[l] = per;
      toUnitLb.map(l.getInductionVar(), l.getLowerBound());
    }

    // unit_data를 loop 앞에 끼워 넣으므로, 순회 중 블록을 고치지 않도록 먼저 모은다.
    SmallVector<pim::AccessOp> accesses;
    units.getBody().walk([&](pim::AccessOp a) { accesses.push_back(a); });
    auto gridDims = func->getAttrOfType<ArrayAttr>("pim.grid_dims");
    for (pim::AccessOp a : accesses)
      buildUnitData(b, a, outermost, loops, perUnitTrips, toUnitLb, names,
                    gridDims, combine, combineOver);
  }

  // access 하나에 대한 pim.unit_data를 만든다. 축별 시작 위치가 없거나 유닛 좌표로 되돌릴
  // 수 없는 형태면 만들지 않는다 — 없는 것이 틀린 것보다 낫다.
  void buildUnitData(OpBuilder &b, pim::AccessOp a, Operation *nest,
                     ArrayRef<scf::ForOp> loops,
                     const llvm::DenseMap<Operation *, Value> &perUnitTrips,
                     IRMapping &toUnitLb, ArrayRef<Attribute> names,
                     ArrayAttr gridDims, StringAttr combine,
                     ArrayRef<Attribute> combineOver) {
    auto role = a->getAttrOfType<StringAttr>("pim.role");
    auto id = a->getAttrOfType<IntegerAttr>("pim.access_id");
    if (a.getOrigin().empty() || !role || !id)
      return;

    Location loc = a.getLoc();
    MLIRContext *ctx = b.getContext();
    SmallVector<Value> origin, sizes, strides;
    for (auto [d, o] : llvm::enumerate(a.getOrigin())) {
      Value start = rebuild(b, o, toUnitLb, nest);
      Value stride = rebuild(b, a.getStrides()[d], toUnitLb, nest);
      if (!start || !stride)
        return;
      // 이 축을 움직이는 loop가 있으면 유닛이 그 loop를 도는 횟수만큼 타일이 이어 붙는다.
      bool ambiguous = false;
      scf::ForOp drv = driverLoop(o, loops, ambiguous);
      if (ambiguous)
        return;
      origin.push_back(start);
      strides.push_back(stride);
      Value tile = b.create<arith::ConstantIndexOp>(loc, a.getSizes()[d]);
      sizes.push_back(drv ? b.create<arith::MulIOp>(
                                loc, perUnitTrips.lookup(drv), tile)
                          : tile);
    }

    bool isOutput = role.getValue() == "output";
    // 입력이 어느 grid 축에 불변이면 그 축만 다른 유닛들이 같은 데이터를 받는다.
    SmallVector<Attribute> replicated;
    if (!isOutput)
      if (auto inv = a->getAttrOfType<DenseI64ArrayAttr>("pim.invariant_grid_dims"))
        for (int64_t d : inv.asArrayRef()) {
          Attribute nm = gridDims[d];
          if (llvm::is_contained(names, nm))
            replicated.push_back(nm);
        }

    b.create<pim::UnitDataOp>(
        loc, a.getBase(), origin, sizes, strides,
        StringAttr::get(ctx, isOutput ? "out" : "in"), id,
        replicated.empty() ? ArrayAttr() : ArrayAttr::get(ctx, replicated),
        isOutput && !combineOver.empty() ? combine : StringAttr(),
        isOutput && !combineOver.empty() ? ArrayAttr::get(ctx, combineOver)
                                         : ArrayAttr());
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMPartitionPass() {
  return std::make_unique<PIMPartitionPass>();
}
