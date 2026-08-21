#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMPLANGROUPEXECUTION
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct PIMPlanGroupExecutionPass
    : public mlir::triton::impl::PIMPlanGroupExecutionBase<
          PIMPlanGroupExecutionPass> {
  using PIMPlanGroupExecutionBase::PIMPlanGroupExecutionBase;

  void runOnOperation() override {
    pim::TargetProfileAttr profile = pim::targetProfile(getOperation());
    if (!profile) {
      getOperation()->emitError(
          "pim-plan-group-execution: the module has no #pim.target_profile");
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
    bool simd = profile.getParallelExecutionKind() == "simd";
    int64_t lanes = profile.getNumberOfParallelLanesPerPimCore();
    IntegerAttr local = profile.getUsableLocalCapacityBytes();
    // simd 는 lane 협력이 하드웨어에 내장돼 있어 역할이 언제나 output_row 다.
    // multithread 는 profile 의 lane_mode 가 역할을 정한다 — pim-form-block-groups
    // 와 같은 값을 봐야 하므로 pass 옵션이 아니라 profile 필드다. 모르는 값은
    // 진단이다(기본값으로 때우지 않는다).
    StringRef role = "output_row";
    if (!simd) {
      StringAttr mode = profile.getLaneMode();
      if (!mode)
        return func->emitError(
            "pim-plan-group-execution: a multithread profile carries no "
            "lane_mode — give it to --pim-set-target-profile");
      if (mode.getValue() != "whole_tile" && mode.getValue() != "output_row")
        return func->emitError(
                   "pim-plan-group-execution: unknown lane_mode \"")
               << mode.getValue()
               << "\"; expected \"whole_tile\" or \"output_row\"";
      role = mode.getValue();
      if (role == "whole_tile") {
        // lane 마다 group 버퍼를 복제하므로 실제 lane 수를 예산이 정한다.
        if (!local)
          return func->emitError(
              "pim-plan-group-execution: usable_local_capacity_bytes is not "
              "recorded, so the lane budget of a multithread target cannot "
              "be decided");
        if (local.getInt() > 0) {
          int64_t bytes = std::max<int64_t>(1, pim::localTileBytes(func));
          lanes = std::max<int64_t>(1, std::min(lanes, local.getInt() / bytes));
        }
      } else {
        // output_row 는 버퍼가 한 벌이라 lane 수가 예산과 무관하다. 대신 lane 들이
        // 출력 행을 나눠 가지므로 행 수가 lanes 로 나누어떨어져야 한다 — 능력값
        // 이하의 약수 중 최대값을 고른다. 행 수를 모르면(축이 없거나 모호) 능력값을
        // 그대로 두고 target lowering 의 진단에 맡긴다.
        SmallVector<scf::ForOp> loops;
        func.walk<WalkOrder::PreOrder>(
            [&](scf::ForOp l) { loops.push_back(l); });
        scf::ForOp rowLoop;
        for (scf::ForOp l : loops) {
          auto a = l->getAttrOfType<StringAttr>("pim.axis");
          if (a && a.getValue() == "output_row_tile")
            rowLoop = l;
        }
        int64_t rows =
            rowLoop ? pim::tileElementsAlong(rowLoop, loops) : 0;
        if (rows > 0)
          while (lanes > 1 && rows % lanes != 0)
            lanes--;
      }
      // 한 벌(또는 lanes 벌)이 예산에 드는지는 pim-verify 가 확인한다.
    }

    func->setAttr("pim.group_execution",
                  pim::GroupExecutionAttr::get(func.getContext(), lanes,
                                               b.getStringAttr(role)));
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMPlanGroupExecutionPass() {
  return std::make_unique<PIMPlanGroupExecutionPass>();
}
