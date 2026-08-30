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

// 배치 pass가 만드는 clamp 식인가. 런타임 값이면 부등식을 증명할 수 없으므로 clamp가
// 있었는지를 구조로 확인한다.
static bool isClamped(Value v) {
  Operation *def = v.getDefiningOp();
  return getConstantIntValue(v).has_value() ||
         isa_and_nonnull<arith::SelectOp, arith::MinSIOp, arith::MaxSIOp>(def);
}

// pim-map-devices가 만드는 local window의 upper bound는
// min(execution_tile.groups[k], remaining)이다. operand 순서가 바뀌어도 같은
// 계약으로 보고, 다른 임의의 clamp는 temporal K window로 인정하지 않는다.
static bool isClampedTo(Value value, int64_t limit) {
  auto min = value.getDefiningOp<arith::MinSIOp>();
  if (!min)
    return false;
  return getConstantIntValue(min.getLhs()) == limit ||
         getConstantIntValue(min.getRhs()) == limit;
}

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
    if (*lhs < 0 || *rhs < 0 ||
        (*lhs != 0 &&
         *rhs > std::numeric_limits<int64_t>::max() / *lhs))
      return std::nullopt;
    return *lhs * *rhs;
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
      getOperation()->emitError("pim-verify: the module has no #pim.target_profile");
      return signalPassFailure();
    }
    if (failed(verifyCores(profile)))
      return signalPassFailure();
    WalkResult res = getOperation().walk([&](pim::DeviceOp d) {
      return succeeded(verifyDevice(d, profile)) ? WalkResult::advance()
                                                 : WalkResult::interrupt();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }

  LogicalResult verifyCores(pim::TargetProfileAttr profile) {
    int64_t local = profile.getLocalCapacityLimitBytes();
    WalkResult res = getOperation().walk([&](func::FuncOp func) {
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

      // hierarchy metadata가 실제 local compute tile과 같은 크기를 말하는지 확인한다.
      // m/n은 이름 붙은 logical axis, k는 access origin을 모는 유일한 reduction
      // loop다. 이 검사가 없으면 block_group.blocks를 바꿔도 linalg/access shape는
      // 예전 값인 모순된 Common IR이 target lowering까지 내려간다.
      SmallVector<scf::ForOp> loops;
      func.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
        // pim.device 바깥의 scheduling round도 tag 없는 scf.for다. 그것을
        // reduction loop로 오인하지 않고 device-local compute loop만 본다.
        if (loop->getParentOfType<pim::DeviceOp>())
          loops.push_back(loop);
      });
      scf::ForOp hierarchyAxes[3];
      for (scf::ForOp loop : loops) {
        auto axis = loop->getAttrOfType<pim::AxisAttr>("pim.axis");
        if (!axis)
          continue;
        if (axis.getName().getValue() == "m")
          hierarchyAxes[0] = loop;
        else if (axis.getName().getValue() == "n")
          hierarchyAxes[1] = loop;
      }
      llvm::SmallPtrSet<Operation *, 4> reductions;
      func.walk([&](pim::AccessOp access) {
        for (Value origin : access.getOrigin()) {
          llvm::SmallPtrSet<Operation *, 4> drivers;
          pim::collectDrivers(origin, loops, drivers);
          for (Operation *driver : drivers)
            if (!driver->getAttrOfType<pim::AxisAttr>("pim.axis"))
              reductions.insert(driver);
        }
      });
      if (reductions.size() == 1)
        hierarchyAxes[2] = cast<scf::ForOp>(*reductions.begin());

      SmallVector<int64_t, 3> groupShape = pim::groupShape(func);
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
      ArrayRef<int64_t> tileGroups = execution.getGroups().asArrayRef();
      static const char *axisNames[3] = {"m", "n", "k"};
      for (int i = 0; i < 3; ++i) {
        bool hasComputeAxis = static_cast<bool>(hierarchyAxes[i]);
        if (!hasComputeAxis &&
            (groupShape.size() != 3 || groupShape[i] != 1 ||
             tileGroups[i] != 1)) {
          func->emitError("pim-verify: inactive ")
              << axisNames[i]
              << " axis must have identity Block, Block Group, and "
                 "Execution Tile extents of 1";
          return WalkResult::interrupt();
        }
        if (!hasComputeAxis)
          continue;
        if (i == 2 &&
            !hierarchyAxes[i]->hasAttr("pim.temporal_reduction")) {
          func->emitError(
              "pim-verify: the reduction loop is not materialized as a "
              "temporal K Execution Tile");
          return WalkResult::interrupt();
        }
        int64_t actual = pim::tileElementsAlong(hierarchyAxes[i], loops);
        if (actual != groupShape[i]) {
          func->emitError("pim-verify: #pim.block × #pim.block_group gives ")
              << groupShape[i] << " elements on " << axisNames[i]
              << " but the local compute tile uses " << actual;
          return WalkResult::interrupt();
        }
      }
      // bank/core/lane 분해와 물리 padding은 target lowering의 책임이다. Common IR은
      // block group의 working set이 공통 local capacity 상한을 넘는지만 검사한다.
      if (local == 0)
        return WalkResult::advance();
      int64_t bytes = pim::localTileBytes(func);
      if (bytes > local) {
        func->emitError("pim-verify: tile buffers need ")
            << bytes << " byte of local storage but a PIM Core has " << local;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return failure(res.wasInterrupted());
  }

  LogicalResult verifyDevice(pim::DeviceOp device,
                             pim::TargetProfileAttr profile) {
    auto execution =
        device->getAttrOfType<pim::ExecutionTileAttr>("pim.execution_tile");
    if (!execution)
      return device->emitError(
          "pim-verify: every pim.device needs #pim.execution_tile");
    SmallVector<int64_t, 3> executionShape =
        pim::executionTileShape(device);
    if (executionShape.size() != 3)
      return device->emitError(
          "pim-verify: cannot derive the execution tile shape from the "
          "three hierarchy attributes");
    if (llvm::any_of(executionShape,
                     [](int64_t extent) { return extent < 1; }))
      return device->emitError(
          "pim-verify: execution tile extents must be positive");

    SmallVector<scf::ForOp> deviceLoops;
    device.getBody().walk<WalkOrder::PreOrder>(
        [&](scf::ForOp loop) { deviceLoops.push_back(loop); });
    SmallVector<scf::ForOp> temporalReductions;
    for (scf::ForOp loop : deviceLoops)
      if (loop->hasAttr("pim.temporal_reduction"))
        temporalReductions.push_back(loop);
    if (temporalReductions.size() > 1)
      return device->emitError(
          "pim-verify: one Execution Tile may have only one temporal K "
          "reduction loop");

    if (!temporalReductions.empty()) {
      scf::ForOp reduction = temporalReductions.front();
      int64_t tileK = execution.getGroups().asArrayRef()[2];
      if (!matchPattern(reduction.getLowerBound(), m_Zero()) ||
          getConstantIntValue(reduction.getStep()) != 1 ||
          !isClampedTo(reduction.getUpperBound(), tileK))
        return reduction->emitError(
            "pim-verify: the temporal K loop must be a zero-based unit-step "
            "window clamped to execution_tile.groups[k] = ")
               << tileK;
    }

    bool hasTemporalRound = false;
    for (Operation *parent = device->getParentOp(); parent;
         parent = parent->getParentOp())
      if (auto loop = dyn_cast<scf::ForOp>(parent))
        if (auto axis = loop->getAttrOfType<StringAttr>("pim.scheduling_axis"))
          hasTemporalRound |= axis.getValue() == "reduction_k";
    if (hasTemporalRound != !temporalReductions.empty())
      return device->emitError(
          "pim-verify: temporal K scheduling round and device-local "
          "reduction window must occur together");

    int64_t used = 1;
    for (int64_t p : device.getMapping().getPartitions().asArrayRef())
      used *= p;
    if (used > profile.getNumUnits())
      return device->emitError("pim-verify: the partition counts need ")
             << used << " PIM units but the profile has "
             << profile.getNumUnits();

    llvm::DenseMap<int64_t, pim::AccessOp> accessIds;
    llvm::DenseSet<int64_t> placeIds;
    WalkResult res = device.walk([&](pim::AccessOp a) {
      if (a.getOrigin().size() != a.getSizes().size()) {
        a->emitError("pim-verify: origin rank does not match sizes rank");
        return WalkResult::interrupt();
      }
      // 단계 1은 pad = 0만 지원한다. 0이 아닌 pad는 calloc으로 나오지 않아 별도 채우기가
      // 필요하다.
      if (Value pad = a.getPad())
        if (!matchPattern(pad, m_Zero()) && !matchPattern(pad, m_AnyZeroFloat())) {
          a->emitError("pim-verify: only a zero pad value is supported");
          return WalkResult::interrupt();
        }
      if (!accessIds.insert({(int64_t)a.getId(), a}).second) {
        a->emitError("pim-verify: duplicate pim.access id");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted())
      return failure();

    int64_t staticAllocationBytes = 0;
    bool allAllocationsStatic = true;
    for (Operation &op : device.getBody().front()) {
      auto place = dyn_cast<pim::PlaceOp>(&op);
      if (!place)
        continue;
      if (!placeIds.insert(place.getAccess()).second)
        return place->emitError("pim-verify: two placements for one pim.access");
      auto it = accessIds.find((int64_t)place.getAccess());
      if (it == accessIds.end())
        return place->emitError("pim-verify: no pim.access with id ")
               << place.getAccess() << " in this device";
      pim::AccessOp access = it->second;
      int64_t elements = 1;
      bool allocationStatic = true;
      for (auto [d, s] : llvm::enumerate(place.getSize())) {
        // mask 가 있는 축만 잘라야 한다. 없으면 도는 범위가 그대로 영역이다.
        bool masked = !access.getValid().empty() &&
                      !getConstantIntValue(access.getValid()[d]);
        if (masked && !isClamped(s))
          return place->emitError("pim-verify: the size of buffer axis ")
                 << d << " is not clamped to the buffer extent";
        std::optional<int64_t> size = getConstantIntValue(s);
        std::optional<int64_t> allocation =
            getConstantIntValue(place.getAllocationShape()[d]);

        // placement가 실제 Execution Tile 하나의 resident buffer인지 hierarchy와
        // 독립적으로 맞춰 본다. access dimension을 모는 m/n loop는 해당 Tile의
        // Group 수만큼, spatial split-K는 plane 하나만큼, temporal reduction은
        // tile_k Group 수만큼 상주해야 한다.
        llvm::SmallPtrSet<Operation *, 4> drivers;
        pim::collectDrivers(access.getOrigin()[d], deviceLoops, drivers);
        int64_t expected = access.getSizes()[d];
        bool hierarchyOnly = true;
        for (Operation *driver : drivers) {
          if (auto axis = driver->getAttrOfType<pim::AxisAttr>("pim.axis")) {
            StringRef name = axis.getName().getValue();
            ArrayRef<int64_t> groups = execution.getGroups().asArrayRef();
            int64_t factor = name == "m"   ? groups[0]
                             : name == "n" ? groups[1]
                             : name == "k" ? 1
                                           : 0;
            if (factor < 1 ||
                expected > std::numeric_limits<int64_t>::max() / factor)
              return place->emitError(
                  "pim-verify: resident allocation extent overflows i64");
            expected *= factor;
          } else if (driver->hasAttr("pim.temporal_reduction")) {
            int64_t factor = execution.getGroups().asArrayRef()[2];
            if (expected > std::numeric_limits<int64_t>::max() / factor)
              return place->emitError(
                  "pim-verify: resident allocation extent overflows i64");
            expected *= factor;
          } else {
            hierarchyOnly = false;
          }
        }
        if (hierarchyOnly) {
          allocation = constantIndexExpr(place.getAllocationShape()[d]);
          if (!allocation)
            return place->emitError("pim-verify: resident allocation_shape[")
                   << d << "] must be a static Execution Tile extent";
          if (*allocation != expected)
            return place->emitError("pim-verify: allocation_shape[")
                   << d << "] is " << *allocation
                   << " but the Execution Tile requires " << expected;
        }
        if (size && *size < 0)
          return place->emitError("pim-verify: negative size on buffer axis ")
                 << d;
        if (allocation && *allocation < 0)
          return place->emitError(
                     "pim-verify: negative allocation_shape on buffer axis ")
                 << d;
        if (size && allocation && *size > *allocation)
          return place->emitError("pim-verify: buffer axis ")
                 << d << " needs " << *size << " elements but only " << *allocation
                 << " are allocated per device";
        if (!allocation) {
          allocationStatic = false;
        } else if (*allocation != 0 &&
                   elements > std::numeric_limits<int64_t>::max() /
                                  *allocation) {
          return place->emitError(
              "pim-verify: allocation_shape element count overflows i64");
        } else {
          elements *= *allocation;
        }
      }
      if (!allocationStatic) {
        allAllocationsStatic = false;
      } else {
        Type elementType =
            cast<BaseMemRefType>(place.getBuffer().getType()).getElementType();
        int64_t elementBytes = std::max<int64_t>(
            1, elementType.getIntOrFloatBitWidth() / 8);
        if (elements != 0 &&
            elementBytes > std::numeric_limits<int64_t>::max() / elements)
          return place->emitError(
              "pim-verify: allocation byte count overflows i64");
        int64_t bytes = elements * elementBytes;
        if (staticAllocationBytes >
            std::numeric_limits<int64_t>::max() - bytes)
          return place->emitError(
              "pim-verify: total allocation byte count overflows i64");
        staticAllocationBytes += bytes;
      }
    }
    if (accessIds.size() != placeIds.size())
      return device->emitError("pim-verify: ")
             << accessIds.size() << " accesses but " << placeIds.size()
             << " placements";

    if (!temporalReductions.empty() &&
        temporalReductions.front().getNumResults() != 0) {
      bool sawPartialStore = false;
      bool missingCombine = false;
      device.walk([&](pim::StoreOp store) {
        if (!dependsOn(store.getValue(), temporalReductions.front().getResults()))
          return;
        sawPartialStore = true;
        missingCombine |=
            !store->getAttrOfType<StringAttr>("pim.combine");
      });
      if (!sawPartialStore)
        return device->emitError(
            "pim-verify: temporal K reduction has no partial output store");
      if (missingCombine)
        return device->emitError(
            "pim-verify: every temporal K partial store needs an explicit "
            "pim.combine");
    }
    std::optional<int64_t> unitLimit = pim::unitCapacityLimitBytes(profile);
    if (allAllocationsStatic && unitLimit && *unitLimit > 0 &&
        staticAllocationBytes > *unitLimit)
      return device->emitError("pim-verify: the execution tile needs ")
             << staticAllocationBytes << " byte but one PIM unit is limited to "
             << *unitLimit << " byte";
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMVerifyPass() {
  return std::make_unique<PIMVerifyPass>();
}
