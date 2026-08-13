#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMFORMEXECUTIONTILES
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct PIMFormExecutionTilesPass
    : public mlir::triton::impl::PIMFormExecutionTilesBase<
          PIMFormExecutionTilesPass> {
  using PIMFormExecutionTilesBase::PIMFormExecutionTilesBase;

  void runOnOperation() override {
    if (tileM == 0 && tileN == 0 && tileK == 0)
      return; // 상한이 없으면 항등이다 — 배정 전체가 execution tile 하나
    if (tileM < 0 || tileN < 0 || tileK < 0) {
      getOperation()->emitError("pim-form-execution-tiles: caps must not be "
                                "negative (0 means unlimited)");
      return signalPassFailure();
    }
    WalkResult res = getOperation().walk([&](pim::DeviceOp device) {
      return succeeded(attachCaps(device)) ? WalkResult::advance()
                                           : WalkResult::interrupt();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }

  LogicalResult attachCaps(pim::DeviceOp device) {
    // 상한이 붙는 축의 loop 와 그 축의 group tile 원소수를 찾아 배수 조건을 검사한다.
    // 눈금보다 잘게 자르면 창 경계가 tile 을 가르므로 진단이다.
    SmallVector<scf::ForOp> loops;
    device.getBody().walk<WalkOrder::PreOrder>(
        [&](scf::ForOp l) { loops.push_back(l); });
    auto axisLoop = [&](StringRef name) -> scf::ForOp {
      for (scf::ForOp l : loops)
        if (l->getAttrOfType<StringAttr>("pim.axis") == name)
          return l;
      return scf::ForOp();
    };
    struct Cap {
      int64_t cap;
      StringRef axis;
      StringRef option;
    };
    Cap caps[] = {{tileM, "output_row_tile", "tile-m"},
                  {tileN, "output_column_tile", "tile-n"},
                  {tileK, "explicit_k_partition", "tile-k"}};
    for (const Cap &c : caps) {
      if (c.cap == 0)
        continue;
      scf::ForOp loop = axisLoop(c.axis);
      if (!loop)
        return device->emitError("pim-form-execution-tiles: ")
               << c.option << " is given but the kernel has no " << c.axis
               << " axis";
      int64_t elems = pim::tileElementsAlong(loop, loops);
      if (elems <= 0)
        return device->emitError(
                   "pim-form-execution-tiles: cannot tell how many elements "
                   "one ")
               << c.axis << " tile moves, so the cap cannot be validated";
      if (c.cap % elems != 0)
        return device->emitError("pim-form-execution-tiles: ")
               << c.option << "(" << c.cap << ") is not a multiple of the "
               << c.axis << " tile (" << elems << " elements)";
    }
    device->setAttr("pim.execution_tile",
                    pim::ExecutionTileAttr::get(device.getContext(), tileM,
                                                tileN, tileK));
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMFormExecutionTilesPass() {
  return std::make_unique<PIMFormExecutionTilesPass>();
}
