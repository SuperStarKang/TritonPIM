#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMPLANINTRADEVICE
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// Materialize the common bank -> core -> lane decomposition selected by the
// profile seed or autotuner pass options.
struct PIMPlanIntraDevicePass
    : public mlir::triton::impl::PIMPlanIntraDeviceBase<
          PIMPlanIntraDevicePass> {
  using PIMPlanIntraDeviceBase::PIMPlanIntraDeviceBase;

  void runOnOperation() override {
    pim::TargetProfileAttr profile = pim::targetProfile(getOperation());
    if (!profile) {
      getOperation()->emitError(
          "pim-plan-intra-device: the module has no #pim.target_profile");
      return signalPassFailure();
    }
    WalkResult res = getOperation().walk([&](func::FuncOp func) {
      bool hasAccess = false;
      func.walk([&](pim::AccessOp) { hasAccess = true; });
      if (!hasAccess)
        return WalkResult::advance();
      return succeeded(planFunc(func, profile)) ? WalkResult::advance()
                                                : WalkResult::interrupt();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }

  LogicalResult planFunc(func::FuncOp func, pim::TargetProfileAttr profile) {
    OpBuilder b(func.getContext());
    auto partitions = [&](ArrayRef<int64_t> values, StringRef option,
                          SmallVectorImpl<int64_t> &result) -> LogicalResult {
      if (values.empty()) {
        result.assign({1, 1, 1});
        return success();
      }
      if (values.size() != 3)
        return func->emitError("pim-plan-intra-device: ")
               << option << " must contain [m, n, k]";
      result.assign(values.begin(), values.end());
      if (llvm::any_of(result, [](int64_t p) { return p < 1; }))
        return func->emitError("pim-plan-intra-device: ")
               << option << " entries must be positive";
      return success();
    };
    SmallVector<int64_t> banks, cores;
    if (failed(partitions(bankPartitions, "bank-partitions", banks)) ||
        failed(partitions(corePartitions, "core-partitions", cores)))
      return failure();
    if (laneAxis != "m" && laneAxis != "n" && laneAxis != "k")
      return func->emitError(
          "pim-plan-intra-device: lane-axis must be m, n, or k");
    int64_t selectedLanes =
        lanes == 0 ? profile.getNumberOfParallelLanesPerPimCore() : lanes;
    if (selectedLanes < 1)
      return func->emitError(
          "pim-plan-intra-device: lanes must be positive or zero (auto)");
    auto group = func->getAttrOfType<pim::BlockGroupAttr>("pim.block_group");
    int axis = laneAxis == "m" ? 0 : laneAxis == "n" ? 1 : 2;
    if (lanes == 0 && group) {
      int64_t extent = group.getUseful().asArrayRef()[axis];
      extent /= banks[axis] * cores[axis];
      while (selectedLanes > 1 && extent % selectedLanes != 0)
        --selectedLanes;
    }

    func->setAttr(
        "pim.intra_device_plan",
        pim::IntraDevicePlanAttr::get(
            func.getContext(), b.getDenseI64ArrayAttr(banks),
            b.getDenseI64ArrayAttr(cores), selectedLanes,
            b.getStringAttr(laneAxis)));
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMPlanIntraDevicePass() {
  return std::make_unique<PIMPlanIntraDevicePass>();
}
