#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMFORMEXECUTIONWAVES
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct PIMFormExecutionWavesPass
    : public mlir::triton::impl::PIMFormExecutionWavesBase<
          PIMFormExecutionWavesPass> {
  using PIMFormExecutionWavesBase::PIMFormExecutionWavesBase;

  void runOnOperation() override {
    pim::TargetProfileAttr profile = pim::targetProfile(getOperation());
    if (!profile) {
      getOperation()->emitError(
          "pim-form-execution-waves: the module has no #pim.target_profile");
      return signalPassFailure();
    }
    ArrayRef<int64_t> coreShape = profile.getPimCoreShape().asArrayRef();
    WalkResult res = getOperation().walk([&](func::FuncOp func) {
      auto cores = func->getAttrOfType<pim::CoreMappingAttr>("pim.core_mapping");
      if (!cores)
        return WalkResult::advance();
      return succeeded(formWaves(func, coreShape, cores))
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }

  // footprint 를 core shape 위에 겹치지 않게 놓아 wave 하나의 logical tile 수를 구한다.
  LogicalResult formWaves(func::FuncOp func, ArrayRef<int64_t> coreShape,
                          pim::CoreMappingAttr cores) {
    ArrayRef<int64_t> footprint = cores.getTileFootprint().asArrayRef();
    if (footprint.size() != coreShape.size())
      return func->emitError("pim-form-execution-waves: tile_footprint rank ")
             << footprint.size() << " does not match pim_core_shape rank "
             << coreShape.size();
    int64_t tilesPerWave = 1;
    for (auto [f, c] : llvm::zip(footprint, coreShape)) {
      if (f <= 0 || c % f != 0)
        return func->emitError("pim-form-execution-waves: tile_footprint ")
               << f << " does not divide the PIM Core axis extent " << c;
      tilesPerWave *= c / f;
    }
    if (tilesPerWave != 1)
      return func->emitError("pim-form-execution-waves: a PIM Group holds ")
             << tilesPerWave
             << " logical tiles per wave; lockstep waves are not built yet";
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMFormExecutionWavesPass() {
  return std::make_unique<PIMFormExecutionWavesPass>();
}
