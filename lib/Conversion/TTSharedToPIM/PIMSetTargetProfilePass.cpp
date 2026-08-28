#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMSETTARGETPROFILE
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct PIMSetTargetProfilePass
    : public mlir::triton::impl::PIMSetTargetProfileBase<
          PIMSetTargetProfilePass> {
  using PIMSetTargetProfileBase::PIMSetTargetProfileBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder b(&getContext());
    module->setAttr(
        "pim.target_profile",
        pim::TargetProfileAttr::get(
            &getContext(), b.getStringAttr(target), numUnits, banksPerUnit,
            coresPerBank, lanesPerCore, bankCapacityLimitBytes,
            localCapacityLimitBytes));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMSetTargetProfilePass() {
  return std::make_unique<PIMSetTargetProfilePass>();
}
