#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMMAPCORES
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct PIMMapCoresPass
    : public mlir::triton::impl::PIMMapCoresBase<PIMMapCoresPass> {
  using PIMMapCoresBase::PIMMapCoresBase;

  void runOnOperation() override {
    pim::TargetProfileAttr profile = pim::targetProfile(getOperation());
    if (!profile) {
      getOperation()->emitError(
          "pim-map-cores: the module has no #pim.target_profile");
      return signalPassFailure();
    }
    WalkResult res = getOperation().walk([&](func::FuncOp func) {
      bool hasAccess = false;
      func.walk([&](pim::AccessOp) { hasAccess = true; });
      if (!hasAccess)
        return WalkResult::advance();
      return succeeded(mapFunc(func, profile)) ? WalkResult::advance()
                                               : WalkResult::interrupt();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }

  LogicalResult mapFunc(func::FuncOp func, pim::TargetProfileAttr profile) {
    OpBuilder b(func.getContext());
    ArrayRef<int64_t> coreShape = profile.getPimCoreShape().asArrayRef();
    int64_t cores = 1;
    for (int64_t c : coreShape)
      cores *= c;
    if (cores != 1)
      return func->emitError("pim-map-cores: PIM Group with ")
             << cores
             << " PIM Cores is not mapped yet; only one core per group is "
                "implemented";

    // core 축에 맡길 logical 차원이 없으므로 역할은 전부 "none" 이고 logical tile 하나가
    // core subset 전체를 차지한다.
    SmallVector<Attribute> roles(coreShape.size(), b.getStringAttr("none"));
    SmallVector<int64_t> footprint(coreShape.size(), 1);

    bool simd = profile.getParallelExecutionKind() == "simd";
    int64_t lanes = profile.getNumberOfParallelLanesPerPimCore();
    IntegerAttr local = profile.getUsableLocalCapacityBytes();
    if (!simd) {
      // multithread 는 lane 마다 tile 버퍼를 복제하므로 실제 lane 수를 예산이 정한다.
      if (!local)
        return func->emitError(
            "pim-map-cores: usable_local_capacity_bytes is not recorded, so the "
            "lane budget of a multithread target cannot be decided");
      if (local.getInt() > 0) {
        int64_t bytes = std::max<int64_t>(1, pim::localTileBytes(func));
        lanes = std::max<int64_t>(1, std::min(lanes, local.getInt() / bytes));
      }
    }

    func->setAttr("pim.core_mapping",
                  pim::CoreMappingAttr::get(
                      func.getContext(), b.getArrayAttr(roles),
                      b.getDenseI64ArrayAttr(footprint), lanes,
                      b.getStringAttr(simd ? "output_row" : "whole_tile")));
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMMapCoresPass() {
  return std::make_unique<PIMMapCoresPass>();
}
