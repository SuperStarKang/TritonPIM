#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMVERIFY
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

static int64_t canonicalAxis(StringRef axis) {
  return axis == "m" ? 0 : axis == "n" ? 1 : axis == "k" ? 2 : -1;
}

// allocation_shape는 상수와 상수 곱까지 허용한다. Target은 이 값을 고정 MRAM
// slot으로 사용하므로 런타임 값은 허용하지 않는다.
static std::optional<int64_t> constantIndexExpr(Value value) {
  if (std::optional<int64_t> constant = getConstantIntValue(value))
    return constant;
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return constantIndexExpr(cast.getIn());
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    std::optional<int64_t> lhs = constantIndexExpr(mul.getLhs());
    std::optional<int64_t> rhs = constantIndexExpr(mul.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    __int128 product = static_cast<__int128>(*lhs) * *rhs;
    if (product < std::numeric_limits<int64_t>::min() ||
        product > std::numeric_limits<int64_t>::max())
      return std::nullopt;
    return static_cast<int64_t>(product);
  }
  return std::nullopt;
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

struct PIMVerifyPass
    : public mlir::triton::impl::PIMVerifyBase<PIMVerifyPass> {
  using PIMVerifyBase::PIMVerifyBase;

  void runOnOperation() override {
    pim::TargetProfileAttr profile = pim::targetProfile(getOperation());
    if (!profile) {
      getOperation()->emitError(
          "pim-verify: the module has no #pim.target_profile");
      return signalPassFailure();
    }
    if (failed(verifyFunctions(profile)))
      return signalPassFailure();
    WalkResult result = getOperation().walk([&](pim::DeviceOp device) {
      return succeeded(verifyDevice(device, profile)) ? WalkResult::advance()
                                                      : WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }

  // Common hierarchy와 실제 local compute tile의 크기만 맞춘다. Scheduling 식의
  // 정확한 산술 모양은 그 식을 만드는 pass의 단위 테스트가 담당한다.
  LogicalResult verifyFunctions(pim::TargetProfileAttr profile) {
    int64_t localLimit = profile.getLocalCapacityLimitBytes();
    WalkResult result = getOperation().walk([&](func::FuncOp func) {
      bool hasAccess = false;
      func.walk([&](pim::AccessOp) { hasAccess = true; });
      if (!hasAccess)
        return WalkResult::advance();

      auto block = func->getAttrOfType<pim::BlockAttr>("pim.block");
      auto group =
          func->getAttrOfType<pim::BlockGroupAttr>("pim.block_group");
      if (!block || !group) {
        func->emitError(
            "pim-verify: every PIM function needs #pim.block and "
            "#pim.block_group");
        return WalkResult::interrupt();
      }

      pim::DeviceOp device;
      func.walk([&](pim::DeviceOp candidate) {
        if (!device)
          device = candidate;
      });
      auto execution =
          device ? device->getAttrOfType<pim::ExecutionTileAttr>(
                       "pim.execution_tile")
                 : pim::ExecutionTileAttr();
      if (!execution) {
        func->emitError(
            "pim-verify: every PIM function needs a pim.device with "
            "#pim.execution_tile");
        return WalkResult::interrupt();
      }

      SmallVector<scf::ForOp> loops;
      func.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
        if (loop->getParentOfType<pim::DeviceOp>())
          loops.push_back(loop);
      });
      scf::ForOp axes[3];
      for (scf::ForOp loop : loops) {
        if (auto axis = loop->getAttrOfType<pim::AxisAttr>("pim.axis")) {
          int64_t number = canonicalAxis(axis.getName().getValue());
          if (number >= 0 && number < 2)
            axes[number] = loop;
        }
        if (auto reduction =
                loop->getAttrOfType<StringAttr>("pim.reduction")) {
          if (reduction.getValue() != "add") {
            loop->emitError("pim-verify: unsupported reduction '")
                << reduction.getValue() << "'";
            return WalkResult::interrupt();
          }
          axes[2] = loop;
        }
      }

      SmallVector<int64_t, 3> groupShape = pim::groupShape(func);
      if (groupShape.size() != 3) {
        func->emitError(
            "pim-verify: cannot derive the three-dimensional Block Group "
            "shape");
        return WalkResult::interrupt();
      }
      ArrayRef<int64_t> tileGroups = execution.getGroups().asArrayRef();
      static const char *axisNames[3] = {"m", "n", "k"};
      for (int64_t axis = 0; axis < 3; ++axis) {
        if (!axes[axis]) {
          if (groupShape[axis] != 1 || tileGroups[axis] != 1) {
            func->emitError("pim-verify: inactive ")
                << axisNames[axis]
                << " axis must have Block Group and Execution Tile extents "
                   "of 1";
            return WalkResult::interrupt();
          }
          continue;
        }
        if (axis == 2 && !axes[axis]->hasAttr("pim.temporal_reduction")) {
          func->emitError(
              "pim-verify: the semantic reduction is not marked as a "
              "temporal K reduction");
          return WalkResult::interrupt();
        }
        int64_t actual = pim::tileElementsAlong(axes[axis], loops);
        if (actual != groupShape[axis]) {
          func->emitError("pim-verify: local ")
              << axisNames[axis] << " tile has " << actual
              << " elements but its Block Group has " << groupShape[axis];
          return WalkResult::interrupt();
        }
      }

      if (localLimit > 0) {
        int64_t bytes = pim::localTileBytes(func);
        if (bytes > localLimit) {
          func->emitError("pim-verify: tile buffers need ")
              << bytes << " byte but local capacity is " << localLimit;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult verifyDevice(pim::DeviceOp device,
                             pim::TargetProfileAttr profile) {
    auto execution =
        device->getAttrOfType<pim::ExecutionTileAttr>("pim.execution_tile");
    if (!execution)
      return device->emitError(
          "pim-verify: every pim.device needs #pim.execution_tile");

    SmallVector<scf::ForOp> temporalReductions;
    SmallVector<scf::ForOp> semanticReductions;
    SmallVector<scf::ForOp> spatialAxes[2];
    SmallVector<scf::ForOp> deviceLoops;
    scf::ForOp spatialK;
    device.getBody().walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
      deviceLoops.push_back(loop);
      if (loop->hasAttr("pim.temporal_reduction"))
        temporalReductions.push_back(loop);
      if (loop->getAttrOfType<StringAttr>("pim.reduction"))
        semanticReductions.push_back(loop);
      if (auto axis = loop->getAttrOfType<pim::AxisAttr>("pim.axis")) {
        int64_t number = canonicalAxis(axis.getName().getValue());
        if (number >= 0 && number < 2)
          spatialAxes[number].push_back(loop);
        else if (number == 2)
          spatialK = loop;
      }
    });
    if (spatialK)
      return spatialK->emitError(
          "pim-verify: K must be a semantic reduction, not a spatial axis");
    if (temporalReductions.size() > 1)
      return device->emitError(
          "pim-verify: one device may have only one temporal K reduction");
    if (semanticReductions.size() > 1)
      return device->emitError(
          "pim-verify: one device may have only one semantic reduction");
    for (int64_t axis = 0; axis < 2; ++axis)
      if (spatialAxes[axis].size() > 1)
        return device->emitError("pim-verify: one device may have only one ")
               << (axis == 0 ? "M" : "N") << " axis";

    SmallVector<scf::ForOp> temporalRounds;
    SmallVector<scf::ForOp> spatialRounds[2];
    for (Operation *parent = device->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto loop = dyn_cast<scf::ForOp>(parent);
      if (!loop)
        continue;
      auto axis = loop->getAttrOfType<StringAttr>("pim.scheduling_axis");
      if (!axis)
        continue;
      if (axis.getValue() == "reduction_k") {
        temporalRounds.push_back(loop);
        continue;
      }
      int64_t number = canonicalAxis(axis.getValue());
      if (number >= 0 && number < 2)
        spatialRounds[number].push_back(loop);
      else if (number == 2)
        return loop->emitError(
            "pim-verify: source spatial K scheduling is unsupported");
    }

    if (temporalReductions.empty() != temporalRounds.empty())
      return device->emitError(
          "pim-verify: temporal K reduction and reduction_k round must "
          "occur together");
    if (!temporalReductions.empty() && temporalRounds.size() != 1)
      return device->emitError(
          "pim-verify: one temporal K reduction needs one reduction_k round");
    if (semanticReductions.empty() != temporalReductions.empty()) {
      if (!temporalReductions.empty())
        return temporalReductions.front()->emitError(
            "pim-verify: a temporal K loop needs a semantic reduction");
      return semanticReductions.front()->emitError(
          "pim-verify: a semantic reduction must be the temporal K loop");
    }
    if (!temporalReductions.empty() &&
        temporalReductions.front() != semanticReductions.front())
      return device->emitError(
          "pim-verify: the temporal K loop must carry the semantic "
          "reduction");

    for (int64_t axis = 0; axis < 2; ++axis) {
      StringRef name = axis == 0 ? "M" : "N";
      if (spatialAxes[axis].empty() && !spatialRounds[axis].empty())
        return spatialRounds[axis].front()->emitError("pim-verify: spatial ")
               << name << " scheduling round has no local axis";
      if (!spatialAxes[axis].empty() && spatialRounds[axis].size() != 1)
        return spatialAxes[axis].front()->emitError("pim-verify: spatial ")
               << name << " axis needs one scheduling round";
    }

    ArrayRef<int64_t> partitions =
        device.getMapping().getPartitions().asArrayRef();
    if (partitions[0] > 1 && spatialAxes[0].empty())
      return device->emitError(
          "pim-verify: p_m > 1 requires one spatial M axis");
    if (partitions[1] > 1 && spatialAxes[1].empty())
      return device->emitError(
          "pim-verify: p_n > 1 requires one spatial N axis");
    if (partitions[2] > 1 && semanticReductions.empty())
      return device->emitError(
          "pim-verify: p_k > 1 requires one semantic K reduction");

    int64_t usedUnits = 1;
    for (int64_t count : partitions) {
      if (usedUnits > std::numeric_limits<int64_t>::max() / count)
        return device->emitError(
            "pim-verify: partition count product overflows i64");
      usedUnits *= count;
    }
    if (usedUnits > profile.getNumUnits())
      return device->emitError("pim-verify: partition counts need ")
             << usedUnits << " units but the profile has "
             << profile.getNumUnits();

    llvm::DenseMap<int64_t, pim::AccessOp> accesses;
    WalkResult accessResult = device.walk([&](pim::AccessOp access) {
      if (access.getOrigin().size() != access.getSizes().size()) {
        access->emitError(
            "pim-verify: origin rank must equal access sizes rank");
        return WalkResult::interrupt();
      }
      if (Value pad = access.getPad()) {
        if (!matchPattern(pad, m_Zero()) &&
            !matchPattern(pad, m_AnyZeroFloat())) {
          access->emitError("pim-verify: only zero padding is supported");
          return WalkResult::interrupt();
        }
      }
      if (!accesses.insert({access.getId(), access}).second) {
        access->emitError("pim-verify: duplicate pim.access id");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (accessResult.wasInterrupted())
      return failure();

    llvm::DenseMap<int64_t, pim::PlaceOp> placements;
    int64_t staticAllocationBytes = 0;
    for (Operation &operation : device.getBody().front()) {
      auto place = dyn_cast<pim::PlaceOp>(&operation);
      if (!place)
        continue;
      if (!placements.insert({place.getAccess(), place}).second)
        return place->emitError(
            "pim-verify: two placements refer to one access");
      auto found = accesses.find(place.getAccess());
      if (found == accesses.end())
        return place->emitError("pim-verify: no pim.access with id ")
               << place.getAccess();
      pim::AccessOp access = found->second;
      if (place.getAllocationShape().size() != access.getSizes().size())
        return place->emitError(
            "pim-verify: placement rank must equal access rank");

      int64_t elements = 1;
      for (auto [axis, allocationValue] :
           llvm::enumerate(place.getAllocationShape())) {
        std::optional<int64_t> allocation =
            constantIndexExpr(allocationValue);
        if (!allocation || *allocation < 1)
          return place->emitError("pim-verify: allocation_shape[")
                 << axis << "] must be a positive compile-time constant";
        int64_t expected = access.getSizes()[axis];
        if (expected < 1)
          return access->emitError("pim-verify: access size[")
                 << axis << "] must be positive";
        llvm::SmallPtrSet<Operation *, 4> drivers;
        pim::collectDrivers(access.getOrigin()[axis], deviceLoops, drivers);
        for (Operation *driver : drivers) {
          int64_t hierarchyAxis = 2;
          if (auto spatial =
                  driver->getAttrOfType<pim::AxisAttr>("pim.axis"))
            hierarchyAxis = canonicalAxis(spatial.getName().getValue());
          else if (!driver->hasAttr("pim.temporal_reduction"))
            return place->emitError("pim-verify: allocation axis ")
                   << axis << " is driven by a non-hierarchy loop";
          int64_t factor = execution.getGroups().asArrayRef()[hierarchyAxis];
          if (expected > std::numeric_limits<int64_t>::max() / factor)
            return place->emitError(
                "pim-verify: resident allocation extent overflows i64");
          expected *= factor;
        }
        if (*allocation != expected)
          return place->emitError("pim-verify: allocation_shape[")
                 << axis << "] = " << *allocation
                 << " but the Execution Tile requires " << expected;
        if (std::optional<int64_t> size =
                getConstantIntValue(place.getSize()[axis]))
          if (*size < 0 || *size > *allocation)
            return place->emitError("pim-verify: static place size[")
                   << axis << "] = " << *size << " is outside [0, "
                   << *allocation << "]";
        if (elements > std::numeric_limits<int64_t>::max() / *allocation)
          return place->emitError(
              "pim-verify: allocation element count overflows i64");
        elements *= *allocation;
      }

      Type elementType =
          cast<BaseMemRefType>(place.getBuffer().getType()).getElementType();
      int64_t elementBytes =
          std::max<int64_t>(1, elementType.getIntOrFloatBitWidth() / 8);
      if (elements > std::numeric_limits<int64_t>::max() / elementBytes)
        return place->emitError(
            "pim-verify: allocation byte count overflows i64");
      int64_t bytes = elements * elementBytes;
      if (staticAllocationBytes >
          std::numeric_limits<int64_t>::max() - bytes)
        return place->emitError(
            "pim-verify: total allocation byte count overflows i64");
      staticAllocationBytes += bytes;
    }
    if (accesses.size() != placements.size())
      return device->emitError("pim-verify: ")
             << accesses.size() << " accesses but " << placements.size()
             << " placements";

    SmallVector<Value> kRoots;
    kRoots.push_back(device.getBody().front().getArgument(2));
    for (scf::ForOp loop : temporalReductions)
      kRoots.push_back(loop.getInductionVar());
    for (scf::ForOp loop : temporalRounds)
      kRoots.push_back(loop.getInductionVar());

    auto placementDependsOnK = [&](pim::PlaceOp place) {
      for (Value value : place.getStart())
        if (dependsOn(value, kRoots))
          return true;
      for (Value value : place.getSize())
        if (dependsOn(value, kRoots))
          return true;
      for (Value value : place.getStrides())
        if (dependsOn(value, kRoots))
          return true;
      return false;
    };

    auto verifyFinalAccess = [&](pim::AccessOp access,
                                 Operation *write) -> LogicalResult {
      auto found = placements.find(access.getId());
      if (found == placements.end())
        return write->emitError(
            "pim-verify: final output has no matching placement");
      if (found->second.getDirection() != "out")
        return write->emitError(
            "pim-verify: final output placement must have direction out");
      if (placementDependsOnK(found->second))
        return write->emitError(
            "pim-verify: final output placement must not depend on K "
            "scheduling");
      return success();
    };

    WalkResult storeResult = device.walk([&](pim::StoreOp store) {
      auto access = store.getAccess().getDefiningOp<pim::AccessOp>();
      if (!access) {
        store->emitError(
            "pim-verify: store destination must be defined by pim.access");
        return WalkResult::interrupt();
      }
      if (failed(verifyFinalAccess(access, store)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (storeResult.wasInterrupted())
      return failure();
    WalkResult combineResult = device.walk([&](pim::CombineOp combine) {
      auto access =
          combine.getDestination().getDefiningOp<pim::AccessOp>();
      if (!access || failed(verifyFinalAccess(access, combine)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (combineResult.wasInterrupted())
      return failure();

    if (!semanticReductions.empty() &&
        semanticReductions.front().getNumResults() != 0) {
      scf::ForOp reduction = semanticReductions.front();
      int64_t combineCount = 0;
      pim::StoreOp directStore;
      pim::CombineOp unrelatedCombine;
      bool epilogue = false;
      bool wrongKind = false;
      StringAttr reductionKind =
          reduction->getAttrOfType<StringAttr>("pim.reduction");
      device.walk([&](pim::StoreOp store) {
        if (!directStore && dependsOn(store.getValue(), reduction.getResults()))
          directStore = store;
      });
      device.walk([&](pim::CombineOp combine) {
        if (!dependsOn(combine.getPartial(), reduction.getResults())) {
          if (!unrelatedCombine)
            unrelatedCombine = combine;
          return;
        }
        ++combineCount;
        epilogue |= !llvm::is_contained(reduction.getResults(),
                                        combine.getPartial());
        wrongKind |= combine.getKind() != reductionKind.getValue();
      });
      if (directStore)
        return directStore->emitError(
            "pim-verify: a K partial must use pim.combine, not pim.store");
      if (unrelatedCombine)
        return unrelatedCombine->emitError(
            "pim-verify: pim.combine partial must be the K reduction result");
      if (reduction.getRegionIterArgs().size() != 1 ||
          reduction.getNumResults() != 1)
        return reduction->emitError(
            "pim-verify: a K reduction must carry one accumulator result");
      if (combineCount != 1)
        return device->emitError(
            "pim-verify: one K reduction result needs one pim.combine");
      if (epilogue)
        return device->emitError(
            "pim-verify: a K partial cannot have an epilogue before combine");
      if (wrongKind)
        return device->emitError(
            "pim-verify: reduction and combine kinds must match");
    } else if (semanticReductions.empty()) {
      pim::CombineOp unexplained;
      device.walk([&](pim::CombineOp combine) {
        if (!unexplained)
          unexplained = combine;
      });
      if (unexplained)
        return unexplained->emitError(
            "pim-verify: pim.combine requires a semantic K reduction");
    }

    std::optional<int64_t> unitLimit = pim::unitCapacityLimitBytes(profile);
    if (unitLimit && *unitLimit > 0 && staticAllocationBytes > *unitLimit)
      return device->emitError("pim-verify: allocation needs ")
             << staticAllocationBytes << " byte but one unit is limited to "
             << *unitLimit << " byte";
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMVerifyPass() {
  return std::make_unique<PIMVerifyPass>();
}
