#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMMAPDEVICES
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

static int64_t canonicalAxis(StringRef axis) {
  return axis == "m" ? 0 : axis == "n" ? 1 : 2;
}

static bool dependsOn(Value value, ValueRange roots) {
  SmallVector<Value> work{value};
  llvm::DenseSet<Value> seen;
  while (!work.empty()) {
    Value current = work.pop_back_val();
    if (!seen.insert(current).second)
      continue;
    if (llvm::is_contained(roots, current))
      return true;
    if (Operation *def = current.getDefiningOp())
      work.append(def->getOperands().begin(), def->getOperands().end());
  }
  return false;
}

// Device scheduling 산술은 index로 통일한다. Triton의 runtime reduction loop는
// i32일 수 있으므로 원래 scf.for 경계에서만 그 loop의 control type으로 되돌린다.
static Value castTo(OpBuilder &b, Location loc, Value value, Type type) {
  return value.getType() == type
             ? value
             : b.create<arith::IndexCastOp>(loc, type, value).getResult();
}

static Value toIndex(OpBuilder &b, Location loc, Value value) {
  return castTo(b, loc, value, b.getIndexType());
}

static Value constantOfType(OpBuilder &b, Location loc, Type type,
                            int64_t value) {
  return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(type, value))
      .getResult();
}

// scope 안에서 만들어진 loop-bound 식을 scope 앞에 다시 만든다. 함수 인자나 이미
// scope 밖에 있는 값은 그대로 쓰고, scope 안의 block argument나 side effect가 있는
// 식에 닿으면 scheduling 위치로 옮길 수 없으므로 실패한다.
static Value rebuildBefore(OpBuilder &b, Value value, IRMapping &made,
                           Operation *scope) {
  if (Value rebuilt = made.lookupOrNull(value))
    return rebuilt;
  Operation *def = value.getDefiningOp();
  if (!def) {
    Operation *owner = value.getParentBlock()->getParentOp();
    return owner == scope || scope->isAncestor(owner) ? Value() : value;
  }
  if (!scope->isAncestor(def))
    return value;
  if (def->getNumRegions() != 0 || !isMemoryEffectFree(def))
    return Value();

  IRMapping operands;
  for (Value operand : def->getOperands()) {
    Value rebuilt = rebuildBefore(b, operand, made, scope);
    if (!rebuilt)
      return Value();
    operands.map(operand, rebuilt);
  }
  Value rebuilt =
      b.clone(*def, operands)->getResult(cast<OpResult>(value).getResultNumber());
  made.map(value, rebuilt);
  return rebuilt;
}

struct SchedulingRange {
  Value lower;
  Value step;
  Value extent;
};

static LogicalResult buildSchedulingRange(OpBuilder &b, Location loc,
                                          scf::ForOp loop, Operation *scope,
                                          IRMapping &made,
                                          SchedulingRange &range) {
  range.lower = rebuildBefore(b, loop.getLowerBound(), made, scope);
  Value upper = rebuildBefore(b, loop.getUpperBound(), made, scope);
  range.step = rebuildBefore(b, loop.getStep(), made, scope);
  if (!range.lower || !upper || !range.step)
    return failure();
  Value count = b.create<arith::CeilDivSIOp>(
      loc, b.create<arith::SubIOp>(loc, upper, range.lower), range.step);
  range.extent = toIndex(b, loc, count);
  return success();
}

static void scheduleLoop(OpBuilder &b, scf::ForOp loop, Value round,
                         Value device, int64_t partitions,
                         int64_t unitGroups, const SchedulingRange &range) {
  Location loc = loop.getLoc();
  Type loopType = loop.getInductionVar().getType();
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value parts = b.create<arith::ConstantIndexOp>(loc, partitions);
  Value unit = b.create<arith::ConstantIndexOp>(loc, unitGroups);
  Value executionTile = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, round, parts), device);
  Value first = b.create<arith::MulIOp>(loc, executionTile, unit);
  Value left = b.create<arith::MaxSIOp>(
      loc, b.create<arith::SubIOp>(loc, range.extent, first), c0);
  Value trips = b.create<arith::MinSIOp>(loc, unit, left);
  Value firstInLoopType = castTo(b, loc, first, loopType);
  Value start = b.create<arith::AddIOp>(
      loc, range.lower,
      b.create<arith::MulIOp>(loc, firstInLoopType, range.step));

  loop.getLowerBoundMutable().assign(constantOfType(b, loc, loopType, 0));
  loop.getUpperBoundMutable().assign(castTo(b, loc, trips, loopType));
  loop.getStepMutable().assign(constantOfType(b, loc, loopType, 1));

  // With one partition, every entered scheduling round owns at least one
  // Group, including a partial final tile. Empty domains execute no rounds.
  // With multiple partitions the final round may contain inactive devices.
  loop->removeAttr("pim.min_trip_count");
  if (partitions == 1)
    loop->setAttr("pim.min_trip_count", b.getI64IntegerAttr(1));

  OpBuilder bb(loop.getBody(), loop.getBody()->begin());
  Value local = loop.getInductionVar();
  auto scaled = bb.create<arith::MulIOp>(loc, local, range.step);
  auto global = bb.create<arith::AddIOp>(loc, start, scaled);
  global->setAttr("pim.device_origin", bb.getUnitAttr());
  local.replaceUsesWithIf(global, [&](OpOperand &use) {
    return use.getOwner() != scaled && use.getOwner() != global;
  });
}

struct PIMMapDevicesPass
    : public mlir::triton::impl::PIMMapDevicesBase<PIMMapDevicesPass> {
  using PIMMapDevicesBase::PIMMapDevicesBase;

  void runOnOperation() override {
    pim::TargetProfileAttr profile = pim::targetProfile(getOperation());
    if (!profile) {
      getOperation()->emitError(
          "pim-map-devices: the module has no #pim.target_profile");
      return signalPassFailure();
    }
    if (getOperation()
            .walk([&](func::FuncOp func) {
              return failed(mapFunc(func, profile)) ? WalkResult::interrupt()
                                                   : WalkResult::advance();
            })
            .wasInterrupted())
      signalPassFailure();
  }

  LogicalResult mapFunc(func::FuncOp func, pim::TargetProfileAttr profile) {
    SmallVector<scf::ForOp> allLoops;
    func.walk<WalkOrder::PreOrder>(
        [&](scf::ForOp loop) { allLoops.push_back(loop); });

    SmallVector<scf::ForOp> loops;
    SmallVector<StringAttr> names;
    bool duplicateAxis = false;
    bool seen[3] = {false, false, false};
    for (scf::ForOp loop : allLoops) {
      auto axis = loop->getAttrOfType<pim::AxisAttr>("pim.axis");
      if (!axis)
        continue;
      int64_t number = canonicalAxis(axis.getName().getValue());
      if (seen[number]) {
        duplicateAxis = true;
        continue;
      }
      seen[number] = true;
      loops.push_back(loop);
      names.push_back(axis.getName());
    }
    if (duplicateAxis)
      return func->emitError(
          "pim-map-devices: each source pim.axis may occur at most once");

    // Execution Tile의 K는 source pim.axis="k"가 아니라 PIMFormCompute가
    // 명시한 하나의 semantic reduction을 자른다. 주소를 모는 loop를
    // 보고 연산 의미를 재해석하지 않는다.
    SmallVector<scf::ForOp> reductions;
    for (scf::ForOp loop : allLoops)
      if (auto reduction =
              loop->getAttrOfType<StringAttr>("pim.reduction")) {
        if (reduction.getValue() != "add")
          return loop->emitError("pim-map-devices: unsupported reduction '")
                 << reduction.getValue() << "'";
        reductions.push_back(loop);
      }
    if (reductions.size() > 1)
      return func->emitError(
          "pim-map-devices: multiple semantic reduction loops are present; "
          "Common IR cannot form one unambiguous temporal K Execution Tile");
    scf::ForOp reductionLoop =
        reductions.empty() ? scf::ForOp() : reductions.front();

    // K는 source grid의 독립 plane이 아니다. PIMFormCompute가 만든 하나의
    // semantic reduction domain을 이 pass가 p_k와 tile_k로 직접 분할한다.
    // 수동 작성 IR도 같은 계약을 지키게 하여 두 K domain이 공존하지 않게 한다.
    if (seen[2])
      return func->emitError(
          "pim-map-devices: source-level pim.axis=\"k\" is not supported; "
          "represent K with one pim.reduction loop and partition that "
          "semantic domain with p_k");

    if (loops.empty() && !reductionLoop)
      return success();

    auto execution =
        func->getAttrOfType<pim::ExecutionTileAttr>("pim.execution_tile");
    if (!execution)
      return func->emitError(
          "pim-map-devices: the function has no #pim.execution_tile; run "
          "--pim-form-execution-tiles before --pim-map-devices");

    SmallVector<int64_t, 3> partitions{pM, pN, pK};
    int64_t product = 1;
    for (auto [axis, partition] : llvm::enumerate(partitions)) {
      if (partition < 1)
        return func->emitError(
            "pim-map-devices: p_m, p_n, and p_k must be positive");
      const bool hasSemanticAxis =
          axis == 2 ? static_cast<bool>(reductionLoop) : seen[axis];
      if (!hasSemanticAxis && partition != 1) {
        if (axis == 2)
          return func->emitError(
              "pim-map-devices: p_k > 1 requires one semantic "
              "pim.reduction loop");
        return func->emitError(
            "pim-map-devices: a missing M/N axis must have partition 1");
      }
      product *= partition;
    }
    if (product > profile.getNumUnits())
      return func->emitError("pim-map-devices: [p_m,p_n,p_k] needs ")
             << product << " PIM units but the profile has "
             << profile.getNumUnits();

    ArrayRef<int64_t> tileGroups = execution.getGroups().asArrayRef();
    SmallVector<int64_t> granularity;
    for (StringAttr name : names) {
      int64_t axis = canonicalAxis(name.getValue());
      granularity.push_back(tileGroups[axis]);
    }

    scf::ForOp rootLoop;
    for (scf::ForOp candidate : allLoops) {
      if (candidate == reductionLoop || llvm::is_contained(loops, candidate)) {
        rootLoop = candidate;
        break;
      }
    }
    if (!rootLoop)
      return func->emitError(
          "pim-map-devices: cannot find the outermost PIM compute loop");

    // M/N이 없는 K-only reduction에서는 init과 최종 store가 reduction loop의
    // 형제다. loop만 pim.device로 옮기면 init은 scheduling round 밖에서 한 번만
    // 만들어지고 output write는 host에 남는다. 이 경우에만 loop의 single-use
    // tensor init chain과 direct result store를 함께 옮길 envelope로 모은다.
    SmallVector<Operation *> kOnlyInitOps;
    SmallVector<std::pair<pim::AccessOp, pim::StoreOp>> kOnlyOutputs;
    if (reductionLoop && rootLoop == reductionLoop) {
      SmallVector<Value> work(reductionLoop.getInitArgs());
      llvm::DenseSet<Operation *> seenInit;
      while (!work.empty()) {
        Operation *def = work.pop_back_val().getDefiningOp();
        if (!def || def->getNumResults() != 1 ||
            !isa<RankedTensorType>(def->getResult(0).getType()) ||
            !isMemoryEffectFree(def) || !def->getResult(0).hasOneUse() ||
            def->getBlock() != reductionLoop->getBlock() ||
            !seenInit.insert(def).second)
          continue;
        kOnlyInitOps.push_back(def);
        work.append(def->getOperands().begin(), def->getOperands().end());
      }

      pim::StoreOp invalidStore;
      StringRef invalidReason;
      func.walk([&](pim::StoreOp store) {
        if (invalidStore ||
            !dependsOn(store.getValue(), reductionLoop.getResults()))
          return;
        if (!llvm::is_contained(reductionLoop.getResults(),
                                store.getValue())) {
          invalidStore = store;
          invalidReason = "an epilogue consumes the K-only reduction result "
                          "before partial combine";
          return;
        }
        auto access = store.getAccess().getDefiningOp<pim::AccessOp>();
        if (!access || access->getBlock() != reductionLoop->getBlock() ||
            store->getBlock() != reductionLoop->getBlock() ||
            !access.getResult().hasOneUse()) {
          invalidStore = store;
          invalidReason = "the K-only reduction needs a direct output "
                          "pim.access and pim.store in the reduction block";
          return;
        }
        kOnlyOutputs.push_back({access, store});
      });
      if (invalidStore)
        return invalidStore->emitError("pim-map-devices: ") << invalidReason;
      if (reductionLoop.getNumResults() != 0 && kOnlyOutputs.size() != 1)
        return reductionLoop->emitError(
            "pim-map-devices: a K-only reduction result must have exactly "
            "one direct final output pim.store");
    }

    MLIRContext *ctx = func.getContext();
    Location loc = rootLoop.getLoc();
    OpBuilder b(rootLoop);

    // 안쪽 reduction bound는 원래 M/N loop 몸체 안에서 계산될 수 있다. 새 scheduling
    // loop는 rootLoop 밖에 생기므로 bound 식을 그 위치에 다시 만들어야 한다.
    IRMapping rebuiltBounds;
    SmallVector<SchedulingRange> ranges(loops.size());
    for (auto [i, loop] : llvm::enumerate(loops))
      if (failed(buildSchedulingRange(b, loc, loop, rootLoop, rebuiltBounds,
                                      ranges[i])))
        return loop->emitError(
            "pim-map-devices: loop bounds cannot be rebuilt at the Common "
            "scheduling boundary");

    SchedulingRange reductionRange;
    if (reductionLoop &&
        failed(buildSchedulingRange(b, loc, reductionLoop, rootLoop,
                                    rebuiltBounds, reductionRange)))
      return reductionLoop->emitError(
          "pim-map-devices: reduction bounds cannot be rebuilt at the Common "
          "scheduling boundary");

    SmallVector<unsigned> order;
    StringRef preferred[] = {"n", "m"};
    for (StringRef wanted : preferred)
      for (auto [i, name] : llvm::enumerate(names))
        if (name.getValue() == wanted)
          order.push_back(i);

    SmallVector<Value> roundIVs(loops.size());
    OpBuilder scheduled(rootLoop);
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    for (unsigned i : order) {
      int64_t axis = canonicalAxis(names[i].getValue());
      Value unit =
          scheduled.create<arith::ConstantIndexOp>(loc, granularity[i]);
      Value executionTiles =
          scheduled.create<arith::CeilDivSIOp>(loc, ranges[i].extent, unit);
      Value active =
          scheduled.create<arith::ConstantIndexOp>(loc, partitions[axis]);
      Value rounds =
          scheduled.create<arith::CeilDivSIOp>(loc, executionTiles, active);
      auto round = scheduled.create<scf::ForOp>(loc, c0, rounds, c1);
      round->setAttr("pim.scheduling_axis", names[i]);
      roundIVs[i] = round.getInductionVar();
      scheduled.setInsertionPointToStart(round.getBody());
    }

    // 하나의 semantic K domain을 (round, device_k)로 나눈다. 각 device는 한
    // round에 tile_k Block Group을 담당한다. 따라서 K round 수는
    // ceildiv(ceildiv(total_k_groups, tile_k), p_k)이다.
    scf::ForOp reductionRound;
    if (reductionLoop) {
      Value unit = scheduled.create<arith::ConstantIndexOp>(loc, tileGroups[2]);
      Value executionTiles = scheduled.create<arith::CeilDivSIOp>(
          loc, reductionRange.extent, unit);
      Value active =
          scheduled.create<arith::ConstantIndexOp>(loc, partitions[2]);
      Value rounds =
          scheduled.create<arith::CeilDivSIOp>(loc, executionTiles, active);
      reductionRound = scheduled.create<scf::ForOp>(loc, c0, rounds, c1);
      reductionRound->setAttr("pim.scheduling_axis",
                              scheduled.getStringAttr("reduction_k"));
      scheduled.setInsertionPointToStart(reductionRound.getBody());
    }

    auto mapping = pim::DeviceMappingAttr::get(
        ctx, b.getDenseI64ArrayAttr(partitions));
    auto device = scheduled.create<pim::DeviceOp>(loc, mapping);
    device->setAttr("pim.execution_tile", execution);
    func->removeAttr("pim.execution_tile");

    SmallVector<Type> argTypes(3, b.getIndexType());
    SmallVector<Location> argLocs(3, loc);
    Block *body = b.createBlock(&device.getBody(), {}, argTypes, argLocs);
    rootLoop->moveBefore(body, body->end());
    // kOnlyInitOps는 소비자에서 생산자 순서로 모였다. 생산자부터 rootLoop 앞에
    // 옮겨 zero-init이 Execution Tile마다 한 번 실행되게 한다.
    for (Operation *init : llvm::reverse(kOnlyInitOps))
      init->moveBefore(rootLoop);
    // output access/store는 reduction 뒤에 두고, 아래에서 같은 pim.device instance의
    // partial-to-final combine으로 바꾼다.
    Operation *afterReduction = rootLoop;
    for (auto [access, store] : kOnlyOutputs) {
      access->moveAfter(afterReduction);
      store->moveAfter(access);
      afterReduction = store;
    }

    b.setInsertionPoint(rootLoop);
    for (auto [i, loop] : llvm::enumerate(loops)) {
      int64_t axis = canonicalAxis(names[i].getValue());
      scheduleLoop(b, loop, roundIVs[i], body->getArgument(axis),
                   partitions[axis], granularity[i], ranges[i]);
    }

    if (reductionLoop) {
      scheduleLoop(b, reductionLoop, reductionRound.getInductionVar(),
                   body->getArgument(2), partitions[2], tileGroups[2],
                   reductionRange);
      reductionLoop->setAttr("pim.temporal_reduction", b.getUnitAttr());

      // Window마다 accumulator가 identity에서 다시 시작하므로 reduction 결과는
      // 하나의 partial contribution이다. 서로 다른 (round, device_k)는 같은 final
      // M/N destination의 contribution이다. 원래 store의
      // access가 operation의 최종 논리 destination이므로 pim.combine이 value와 그
      // access를 직접 연결한다. target은 destination이나 reduction 여부를 access
      // rank에서 다시 추론하지 않는다. combine 전에 epilogue가 끼면 partial들을
      // 먼저 결합한 뒤 epilogue를 실행하는 원래 의미와 달라지므로 거부한다.
      if (reductionLoop.getNumResults() != 0) {
        SmallVector<pim::StoreOp> partialStores;
        pim::StoreOp epilogueStore;
        device.walk([&](pim::StoreOp store) {
          if (!dependsOn(store.getValue(), reductionLoop.getResults()))
            return;
          if (!llvm::is_contained(reductionLoop.getResults(), store.getValue())) {
            epilogueStore = store;
            return;
          }
          partialStores.push_back(store);
        });
        if (epilogueStore)
          return epilogueStore->emitError(
              "pim-map-devices: an epilogue consumes a K partial "
              "result before it is combined; combine the reduction result "
              "before lowering the epilogue");
        if (partialStores.size() != 1)
          return reductionLoop->emitError(
              "pim-map-devices: one normalized K reduction result must have "
              "exactly one final output store");
        StringAttr combiner =
            reductionLoop->getAttrOfType<StringAttr>("pim.reduction");
        pim::StoreOp store = partialStores.front();
        OpBuilder combineBuilder(store);
        combineBuilder.create<pim::CombineOp>(
            store.getLoc(), store.getValue(), store.getAccess(), combiner);
        store.erase();
      }
    }
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMMapDevicesPass() {
  return std::make_unique<PIMMapDevicesPass>();
}
