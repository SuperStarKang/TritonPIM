#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_PIMREUSEEXECUTIONTILEINPUTS
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace mlir::triton

namespace {

static bool definedInside(Value value, Operation *scope) {
  Operation *owner = value.getDefiningOp();
  if (!owner)
    owner = value.getParentBlock()->getParentOp();
  return owner == scope || scope->isAncestor(owner);
}

static bool knownNonempty(scf::ForOp loop) {
  auto lb = getConstantIntValue(loop.getLowerBound());
  auto ub = getConstantIntValue(loop.getUpperBound());
  auto step = getConstantIntValue(loop.getStep());
  if (lb && ub && step)
    return *step > 0 && *lb < *ub;
  auto minimum = loop->getAttrOfType<IntegerAttr>("pim.min_trip_count");
  return minimum && minimum.getInt() > 0;
}

static bool knownSingleton(scf::ForOp loop, pim::DeviceOp device) {
  auto lb = getConstantIntValue(loop.getLowerBound());
  auto ub = getConstantIntValue(loop.getUpperBound());
  auto step = getConstantIntValue(loop.getStep());
  if (lb && ub && step)
    return *step > 0 && *lb < *ub &&
           static_cast<__int128>(*ub) - *lb <= *step;
  if (!lb || *lb != 0 || !step || *step != 1 || !knownNonempty(loop))
    return false;
  auto tile = device->getAttrOfType<pim::ExecutionTileAttr>("pim.execution_tile");
  if (!tile)
    return false;
  int axis = -1;
  if (loop->hasAttr("pim.temporal_reduction"))
    axis = 2;
  else if (auto name = loop->getAttrOfType<pim::AxisAttr>("pim.axis"))
    axis = name.getName().getValue() == "m" ? 0 :
           name.getName().getValue() == "n" ? 1 : -1;
  return axis >= 0 && tile.getGroups().asArrayRef()[axis] == 1;
}

struct HoistPlan {
  pim::LoadOp load;
  pim::AccessOp access;
  IRMapping values;
  SmallVector<Operation *> scalarOps;
  llvm::SmallPtrSet<Operation *, 16> visited;

  Value remap(Value value) {
    while (Value replacement = values.lookupOrNull(value)) {
      if (replacement == value)
        break;
      value = replacement;
    }
    return value;
  }

  // Plan first, clone only after every operand has passed. Access IDs remain
  // unique: the original access/load are moved, never duplicated.
  bool require(Value value, scf::ForOp loop) {
    if (Value replacement = values.lookupOrNull(value))
      return require(replacement, loop);
    if (!definedInside(value, loop))
      return true;
    Operation *def = value.getDefiningOp();
    if (!def || def->getNumRegions() || !isPure(def) ||
        !llvm::all_of(def->getResultTypes(), [](Type t) {
          return t.isIntOrIndexOrFloat();
        }))
      return false;
    if (visited.contains(def))
      return true;
    for (Value operand : def->getOperands())
      if (!require(operand, loop))
        return false;
    visited.insert(def);
    scalarOps.push_back(def);
    return true;
  }
};

// Placement has already separated entry inputs from exit outputs for this
// device invocation. Check resident access IDs, not inequality of host pointers:
// different host arguments need not be disjoint. Never cross a scheduling round.
static bool inputIsReadOnly(pim::AccessOp access, pim::DeviceOp device) {
  llvm::DenseMap<int64_t, StringRef> directions;
  for (Operation &op : device.getBody().front())
    if (auto place = dyn_cast<pim::PlaceOp>(op))
      directions[place.getAccess()] = place.getDirection();
  if (directions.lookup(access.getId()) != "in")
    return false;
  WalkResult result = device.getBody().walk([&](Operation *op) {
    Value destination;
    if (auto store = dyn_cast<pim::StoreOp>(op))
      destination = store.getAccess();
    else if (auto combine = dyn_cast<pim::CombineOp>(op))
      destination = combine.getDestination();
    if (destination) {
      auto output = destination.getDefiningOp<pim::AccessOp>();
      return output && output.getId() != access.getId() &&
                     directions.lookup(output.getId()) == "out"
                 ? WalkResult::advance() : WalkResult::interrupt();
    }
    if (auto compute = dyn_cast<linalg::LinalgOp>(op)) {
      // The nested scalar body is still visited to reject unknown effects.
      if (!compute.hasPureTensorSemantics())
        return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (isa<pim::PlaceOp, pim::LoadOp, scf::ForOp, scf::YieldOp>(op) ||
        isMemoryEffectFree(op))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });
  return !result.wasInterrupted();
}

static bool buildPlan(pim::LoadOp load, scf::ForOp loop,
                      pim::DeviceOp device, HoistPlan &plan) {
  auto access = load.getAccess().getDefiningOp<pim::AccessOp>();
  if (!access || !access->hasOneUse() || load.getResult().use_empty() ||
      !inputIsReadOnly(access, device))
    return false;
  // The current local-buffer lowering can overwrite a tensor used as a DPS
  // initializer. Only read-only linalg input uses may share this resident value.
  for (OpOperand &use : load.getResult().getUses()) {
    auto compute = dyn_cast<linalg::LinalgOp>(use.getOwner());
    if (!compute || !compute.hasPureTensorSemantics() ||
        !llvm::is_contained(compute.getDpsInputOperands(), &use))
      return false;
  }

  for (Operation *parent = load->getParentOp(); parent != loop;
       parent = parent->getParentOp()) {
    auto inner = dyn_cast<scf::ForOp>(parent);
    // No conditional/speculative loads and no multi-K loop interchange.
    if (!inner || !knownSingleton(inner, device))
      return false;
    plan.values.map(inner.getInductionVar(), inner.getLowerBound());
  }
  for (Value operand : access->getOperands())
    if (!plan.require(operand, loop))
      return false;
  plan.load = load;
  plan.access = access;
  return true;
}

struct PIMReuseExecutionTileInputsPass
    : mlir::triton::impl::PIMReuseExecutionTileInputsBase<
          PIMReuseExecutionTileInputsPass> {
  using PIMReuseExecutionTileInputsBase::PIMReuseExecutionTileInputsBase;

  void runOnOperation() override {
    getOperation().walk([&](pim::DeviceOp device) {
      if (!device->hasAttr("pim.execution_tile"))
        return;
      SmallVector<scf::ForOp> loops;
      device.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
        if (auto axis = loop->getAttrOfType<pim::AxisAttr>("pim.axis"))
          if (axis.getName().getValue() == "m" ||
              axis.getName().getValue() == "n")
            loops.push_back(loop);
      });
      for (scf::ForOp loop : loops)
        reuseInputs(loop, device);
    });
  }

  void reuseInputs(scf::ForOp loop, pim::DeviceOp device) {
    // Group loops have no carried state; reduction loops keep their accumulator.
    // A result-bearing loop would require a result-bearing execution guard.
    auto step = getConstantIntValue(loop.getStep());
    if (loop.getNumResults() || !step || *step <= 0 ||
        knownSingleton(loop, device))
      return;
    auto lb = getConstantIntValue(loop.getLowerBound());
    auto ub = getConstantIntValue(loop.getUpperBound());
    if (lb && ub && *lb >= *ub)
      return;

    SmallVector<pim::LoadOp> loads;
    loop.walk([&](pim::LoadOp load) { loads.push_back(load); });
    bool guarded = false;
    for (pim::LoadOp load : loads) {
      HoistPlan plan;
      if (!buildPlan(load, loop, device, plan))
        continue;
      if (!guarded && !knownNonempty(loop)) {
        // Keep empty Group loops empty without introducing optional tensor
        // values or duplicating access IDs. This ordinary SCF loop runs 0/1 times.
        OpBuilder b(loop);
        Location loc = loop.getLoc();
        Value nonempty = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, loop.getLowerBound(),
            loop.getUpperBound());
        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        Value one = b.create<arith::ConstantIndexOp>(loc, 1);
        Value count = b.create<arith::SelectOp>(loc, nonempty, one, zero);
        auto guard = b.create<scf::ForOp>(loc, zero, count, one);
        loop->moveBefore(guard.getBody()->getTerminator());
        guarded = true;
      }
      OpBuilder b(loop);
      for (Operation *op : plan.scalarOps) {
        for (Value operand : op->getOperands())
          if (Value replacement = plan.values.lookupOrNull(operand))
            plan.values.map(operand, plan.remap(replacement));
        b.clone(*op, plan.values);
      }
      for (OpOperand &operand : plan.access->getOpOperands())
        operand.set(plan.remap(operand.get()));
      // The global origin is still the Common origin. A Target must localize
      // this input from its placement, since singleton-IV folding can erase
      // the original in-loop device_origin arithmetic.
      plan.access->setAttr("pim.reused_input", b.getUnitAttr());
      plan.access->moveBefore(loop);
      plan.load->moveBefore(loop);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMReuseExecutionTileInputsPass() {
  return std::make_unique<PIMReuseExecutionTileInputsPass>();
}
