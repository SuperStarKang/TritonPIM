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
    SmallVector<int64_t> coreShape(pimCoreShape.begin(), pimCoreShape.end());
    if (coreShape.empty())
      coreShape.push_back(1);

    module->setAttr(
        "pim.target_profile",
        pim::TargetProfileAttr::get(
            &getContext(), b.getStringAttr(target), numberOfPIMDevices,
            b.getDenseI64ArrayAttr(coreShape), lanesPerCore,
            b.getStringAttr(parallelExecutionKind),
            b.getBoolAttr(uniformInstructionSequence),
            optional(usableBankCapacityBytes),
            optional(usableLocalCapacityBytes)));

    if (devicesPerTransferGroup > 0)
      module->setAttr("pim.upmem_ext",
                      pim::UPMEMExtAttr::get(&getContext(),
                                             devicesPerTransferGroup));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMSetTargetProfilePass() {
  return std::make_unique<PIMSetTargetProfilePass>();
}
