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
    // 음수는 미측정이라는 뜻이라 필드를 만들지 않는다. 0은 그 저장소가 존재하지 않는다는
    // 뜻이므로 값 0을 그대로 싣는다.
    auto optional = [&](int64_t v) {
      return v < 0 ? IntegerAttr() : b.getI64IntegerAttr(v);
    };
    // 실행 단위는 [m, n, k] 셋이거나 아예 없다. 없으면 눈금이 없는 기계다.
    if (!executionUnit.empty() && executionUnit.size() != 3) {
      module->emitError(
          "pim-set-target-profile: execution-unit must be three values (m, n, "
          "k) or absent");
      return signalPassFailure();
    }
    DenseI64ArrayAttr unit;
    if (!executionUnit.empty())
      unit = b.getDenseI64ArrayAttr(
          SmallVector<int64_t>(executionUnit.begin(), executionUnit.end()));

    module->setAttr(
        "pim.target_profile",
        pim::TargetProfileAttr::get(
            &getContext(), b.getStringAttr(target), numberOfPIMDevices,
            lanesPerCore, b.getStringAttr(parallelExecutionKind), unit,
            optional(usableBankCapacityBytes),
            optional(usableLocalCapacityBytes)));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMSetTargetProfilePass() {
  return std::make_unique<PIMSetTargetProfilePass>();
}
