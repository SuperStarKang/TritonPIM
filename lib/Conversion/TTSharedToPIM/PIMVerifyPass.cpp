#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

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
    IntegerAttr local = profile.getUsableLocalCapacityBytes();
    WalkResult res = getOperation().walk([&](func::FuncOp func) {
      bool hasAccess = false;
      func.walk([&](pim::AccessOp) { hasAccess = true; });
      if (!hasAccess)
        return WalkResult::advance();
      auto plan = func->getAttrOfType<pim::IntraDevicePlanAttr>(
          "pim.intra_device_plan");
      if (!plan) {
        func->emitError(
            "pim-verify: every PIM function needs #pim.intra_device_plan");
        return WalkResult::interrupt();
      }
      auto product = [](ArrayRef<int64_t> values) {
        int64_t value = 1;
        for (int64_t v : values)
          value *= v;
        return value;
      };
      int64_t banks = product(plan.getBankPartitions().asArrayRef());
      int64_t cores = product(plan.getCorePartitions().asArrayRef());
      if (banks > profile.getNumberOfBanksPerPimDevice()) {
        func->emitError("pim-verify: intra-device plan needs ")
            << banks << " banks but a PIM Device has "
            << profile.getNumberOfBanksPerPimDevice();
        return WalkResult::interrupt();
      }
      if (cores > profile.getNumberOfPimCoresPerBank()) {
        func->emitError("pim-verify: intra-device plan needs ")
            << cores << " cores per bank but a bank has "
            << profile.getNumberOfPimCoresPerBank();
        return WalkResult::interrupt();
      }
      if (plan.getLanes() > profile.getNumberOfParallelLanesPerPimCore()) {
        func->emitError("pim-verify: intra-device plan needs ")
            << plan.getLanes() << " lanes per core but a core has "
            << profile.getNumberOfParallelLanesPerPimCore();
        return WalkResult::interrupt();
      }
      ArrayRef<int64_t> bp = plan.getBankPartitions().asArrayRef();
      ArrayRef<int64_t> cp = plan.getCorePartitions().asArrayRef();
      bool laneReducesK =
          plan.getLaneAxis().getValue() == "k" && plan.getLanes() > 1;
      if (bp[2] > 1 || cp[2] > 1 || laneReducesK) {
        func->emitError(
            "pim-verify: intra-device k partitioning needs an explicit "
            "bank/core/lane reduction contract; it is not materialized yet");
        return WalkResult::interrupt();
      }
      // 축별 divisibility와 padding은 target의 물리 실행 눈금에 의존한다. 공통
      // verifier는 topology 사용량만 검사하고, target lowering이 실제 subtile을
      // materialize하면서 나눗셈과 padding legality를 검사한다.
      // 하나의 group을 협력 실행하므로 local buffer는 lane마다 복제되지 않는다.
      if (!local) {
        func->emitError("pim-verify: usable_local_capacity_bytes is not "
                        "recorded, so the lane budget cannot be checked");
        return WalkResult::interrupt();
      }
      if (local.getInt() == 0)
        return WalkResult::advance();
      int64_t bytes = pim::localTileBytes(func);
      int64_t need = bytes;
      if (need > local.getInt()) {
        func->emitError("pim-verify: tile buffers need ")
            << need << " byte of local storage but a PIM Core has "
            << local.getInt();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return failure(res.wasInterrupted());
  }

  LogicalResult verifyDevice(pim::DeviceOp device,
                             pim::TargetProfileAttr profile) {
    auto plan =
        device->getAttrOfType<pim::InterDevicePlanAttr>(
            "pim.inter_device_plan");
    if (!plan)
      return device->emitError(
          "pim-verify: every pim.device needs #pim.inter_device_plan");
    auto execution =
        device->getAttrOfType<pim::ExecutionTileAttr>("pim.execution_tile");
    if (!execution)
      return device->emitError(
          "pim-verify: every pim.device needs #pim.execution_tile");
    auto reduction =
        device->getAttrOfType<pim::ReductionPlanAttr>("pim.reduction_plan");
    if (!reduction)
      return device->emitError(
          "pim-verify: every pim.device needs #pim.reduction_plan");
    ArrayRef<int64_t> tile = plan.getExecutionTile().asArrayRef();
    if (tile.size() != 3 || tile[0] != execution.getM() ||
        tile[1] != execution.getN() || tile[2] != execution.getK())
      return device->emitError(
          "pim-verify: device plan and execution tile disagree");

    bool hasSpatialK = llvm::any_of(
        device.getMapping().getAxes(), [](Attribute axis) {
          return cast<StringAttr>(axis).getValue() == "k";
        });
    if (hasSpatialK !=
        (reduction.getSpatialPartials().getValue() ==
         "separate_output_planes"))
      return device->emitError(
          "pim-verify: reduction plan spatial_partials disagrees with the "
          "k axis");
    if (execution.getK() > 0 &&
        reduction.getTemporalAccumulation().getValue() != "host")
      return device->emitError(
          "pim-verify: a finite tile-k requires host temporal accumulation");

    int64_t used = 1;
    for (int64_t p : device.getMapping().getPartitions().asArrayRef())
      used *= p;
    if (used > profile.getNumberOfPimDevices())
      return device->emitError("pim-verify: the partition counts need ")
             << used << " PIM Devices but the machine has "
             << profile.getNumberOfPimDevices();

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

    ArrayAttr axes = device.getMapping().getAxes();
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
      for (Attribute a : place.getMapping().getAxisSources()) {
        StringRef source = cast<StringAttr>(a).getValue();
        if (!source.empty() && !llvm::is_contained(axes, a))
          return place->emitError("pim-verify: axis source ")
                 << source << " is not a device mapping axis";
      }
      for (auto [d, s] : llvm::enumerate(place.getSize())) {
        // mask 가 있는 축만 잘라야 한다. 없으면 도는 범위가 그대로 영역이다.
        bool masked = !access.getValid().empty() &&
                      !getConstantIntValue(access.getValid()[d]);
        if (masked && !isClamped(s))
          return place->emitError("pim-verify: the size of buffer axis ")
                 << d << " is not clamped to the buffer extent";
        std::optional<int64_t> size = getConstantIntValue(s);
        std::optional<int64_t> capacity =
            getConstantIntValue(place.getCapacity()[d]);
        if (size && *size < 0)
          return place->emitError("pim-verify: negative size on buffer axis ")
                 << d;
        if (size && capacity && *size > *capacity)
          return place->emitError("pim-verify: buffer axis ")
                 << d << " needs " << *size << " elements but only " << *capacity
                 << " are allocated per device";
      }
    }
    if (accessIds.size() != placeIds.size())
      return device->emitError("pim-verify: ")
             << accessIds.size() << " accesses but " << placeIds.size()
             << " placements";
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMVerifyPass() {
  return std::make_unique<PIMVerifyPass>();
}
