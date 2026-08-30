#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMMAPDEVICES
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

static int64_t canonicalAxis(StringRef axis) {
  return axis == "m" ? 0 : axis == "n" ? 1 : axis == "k" ? 2 : -1;
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

static void scheduleLoop(OpBuilder &b, scf::ForOp loop, Value round,
                         Value device, int64_t partitions,
                         int64_t unitGroups, Value extent) {
  Location loc = loop.getLoc();
  Value lb = loop.getLowerBound(), step = loop.getStep();
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value parts = b.create<arith::ConstantIndexOp>(loc, partitions);
  Value unit = b.create<arith::ConstantIndexOp>(loc, unitGroups);
  Value executionTile = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, round, parts), device);
  Value first = b.create<arith::MulIOp>(loc, executionTile, unit);
  Value left = b.create<arith::MaxSIOp>(
      loc, b.create<arith::SubIOp>(loc, extent, first), c0);
  Value trips = b.create<arith::MinSIOp>(loc, unit, left);
  Value start =
      b.create<arith::AddIOp>(loc, lb,
                              b.create<arith::MulIOp>(loc, first, step));

  loop.getLowerBoundMutable().assign(c0);
  loop.getUpperBoundMutable().assign(trips);
  loop.getStepMutable().assign(c1);

  OpBuilder bb(loop.getBody(), loop.getBody()->begin());
  Value local = loop.getInductionVar();
  auto scaled = bb.create<arith::MulIOp>(loc, local, step);
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
    bool invalidAxis = false;
    bool seen[3] = {false, false, false};
    for (scf::ForOp loop : allLoops) {
      auto axis = loop->getAttrOfType<pim::AxisAttr>("pim.axis");
      if (!axis)
        continue;
      int64_t number = canonicalAxis(axis.getName().getValue());
      if (number < 0 || seen[number]) {
        invalidAxis = true;
        continue;
      }
      seen[number] = true;
      loops.push_back(loop);
      names.push_back(axis.getName());
    }
    if (invalidAxis)
      return func->emitError(
          "pim-map-devices: each canonical axis m, n, and k may occur at most once");

    // Execution Tile의 K는 spatial pim.axis="k"가 아니라 그 plane 안의
    // 이름 없는 reduction loop를 자른다. access origin을 모는 loop가 정확히
    // 하나일 때만 Common IR이 temporal K scheduling을 명시할 수 있다.
    llvm::SmallPtrSet<Operation *, 4> reductions;
    func.walk([&](pim::AccessOp access) {
      for (Value origin : access.getOrigin()) {
        llvm::SmallPtrSet<Operation *, 4> drivers;
        pim::collectDrivers(origin, allLoops, drivers);
        for (Operation *driver : drivers)
          if (!driver->getAttrOfType<pim::AxisAttr>("pim.axis"))
            reductions.insert(driver);
      }
    });
    if (reductions.size() > 1)
      return func->emitError(
          "pim-map-devices: multiple reduction loops drive PIM accesses; "
          "Common IR cannot form one unambiguous temporal K Execution Tile");
    scf::ForOp reductionLoop =
        reductions.size() == 1 ? cast<scf::ForOp>(*reductions.begin())
                               : scf::ForOp();

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
      if (!seen[axis] && partition != 1)
        return func->emitError(
            "pim-map-devices: a missing axis must have partition 1");
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
      int64_t groups = name.getValue() == "k" ? 1 : tileGroups[axis];
      if (groups <= 0)
        return func->emitError("pim-map-devices: execution_tile.groups[")
               << name.getValue()
               << "] must be positive for an existing axis";
      granularity.push_back(groups);
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

    MLIRContext *ctx = func.getContext();
    Location loc = rootLoop.getLoc();
    OpBuilder b(rootLoop);

    SmallVector<Value> extents;
    for (scf::ForOp loop : loops)
      extents.push_back(b.create<arith::CeilDivSIOp>(
          loc,
          b.create<arith::SubIOp>(loc, loop.getUpperBound(),
                                  loop.getLowerBound()),
          loop.getStep()));

    Value reductionExtent;
    if (reductionLoop)
      reductionExtent = b.create<arith::CeilDivSIOp>(
          loc,
          b.create<arith::SubIOp>(loc, reductionLoop.getUpperBound(),
                                  reductionLoop.getLowerBound()),
          reductionLoop.getStep());

    SmallVector<unsigned> order;
    StringRef preferred[] = {"n", "k", "m"};
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
          scheduled.create<arith::CeilDivSIOp>(loc, extents[i], unit);
      Value active =
          scheduled.create<arith::ConstantIndexOp>(loc, partitions[axis]);
      Value rounds =
          scheduled.create<arith::CeilDivSIOp>(loc, executionTiles, active);
      auto round = scheduled.create<scf::ForOp>(loc, c0, rounds, c1);
      round->setAttr("pim.scheduling_axis", names[i]);
      roundIVs[i] = round.getInductionVar();
      scheduled.setInsertionPointToStart(round.getBody());
    }

    // Spatial device grid가 정해진 뒤 같은 device set이 temporal K Execution
    // Tile들을 반복한다. 이 loop와 reduction loop의 local bound가 Common IR에
    // 남으므로 target은 K축을 다시 찾거나 window 수를 계산하지 않는다.
    scf::ForOp reductionRound;
    if (reductionLoop) {
      Value unit = scheduled.create<arith::ConstantIndexOp>(loc, tileGroups[2]);
      Value rounds = scheduled.create<arith::CeilDivSIOp>(loc, reductionExtent,
                                                          unit);
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

    b.setInsertionPoint(rootLoop);
    for (auto [i, loop] : llvm::enumerate(loops)) {
      int64_t axis = canonicalAxis(names[i].getValue());
      scheduleLoop(b, loop, roundIVs[i], body->getArgument(axis),
                   partitions[axis], granularity[i], extents[i]);
    }

    if (reductionLoop) {
      Value noDevicePartition = b.create<arith::ConstantIndexOp>(loc, 0);
      scheduleLoop(b, reductionLoop, reductionRound.getInductionVar(),
                   noDevicePartition, /*partitions=*/1, tileGroups[2],
                   reductionExtent);
      reductionLoop->setAttr("pim.temporal_reduction", b.getUnitAttr());

      // Window마다 accumulator가 identity에서 다시 시작하므로 최종 store는
      // partial 결과다. 결합 의미를 Common IR에 명시해 target이 p_k나 loop
      // 구조에서 add를 추론하지 않게 한다. 현재 Common compute가 정규화하는
      // matmul/matvec reduction은 add 결합이다.
      if (reductionLoop.getNumResults() != 0) {
        bool addCombine = false;
        reductionLoop.walk([&](Operation *op) {
          addCombine |= isa<linalg::MatmulOp, linalg::MatvecOp>(op);
        });
        SmallVector<pim::StoreOp> partialStores;
        device.walk([&](pim::StoreOp store) {
          if (dependsOn(store.getValue(), reductionLoop.getResults()))
            partialStores.push_back(store);
        });
        if (!partialStores.empty() && !addCombine)
          return func->emitError(
              "pim-map-devices: cannot materialize the temporal K partial "
              "combine; only normalized add reductions are supported");
        for (pim::StoreOp store : partialStores)
          store->setAttr("pim.combine", b.getStringAttr("add"));
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
