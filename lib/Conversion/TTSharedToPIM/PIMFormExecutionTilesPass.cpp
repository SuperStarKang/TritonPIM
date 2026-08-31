#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
    if (tileM < 0 || tileN < 0 || tileK < 0) {
      getOperation()->emitError(
          "pim-form-execution-tiles: tile-m, tile-n and tile-k must be "
          "0 (default) or positive");
      return signalPassFailure();
    }
    WalkResult res = getOperation().walk([&](func::FuncOp func) {
      bool hasAccess = false;
      func.walk([&](pim::AccessOp) { hasAccess = true; });
      if (!hasAccess)
        return WalkResult::advance();
      return succeeded(attachTile(func)) ? WalkResult::advance()
                                         : WalkResult::interrupt();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }

  LogicalResult attachTile(func::FuncOp func) {
    // Execution Tile은 원소 상한이 아니라 Block Group의 정수 grid다. 이 pass는
    // 비용을 계산하거나 PIM DRAM을 채우지 않고, 주어진 group 개수를 그대로
    // Common IR에 materialize한다. 0은 존재하는 축의 최소값인 group 하나다.
    SmallVector<scf::ForOp> loops;
    func.walk<WalkOrder::PreOrder>(
        [&](scf::ForOp l) { loops.push_back(l); });
    auto axisLoop = [&](StringRef name) -> scf::ForOp {
      for (scf::ForOp l : loops) {
        // reduction loop 는 pim.axis 가 없다 — null StringAttr 를 StringRef 와
        // 비교하면 getValue() 가 터지므로 걸러서 지난다.
        auto a = l->getAttrOfType<pim::AxisAttr>("pim.axis");
        if (a && a.getName().getValue() == name)
          return l;
      }
      return scf::ForOp();
    };
    // tile-k는 spatial split 여부와 무관하게 각 output plane 내부 reduction loop의
    // temporal 상주 window다. k은 별도의 spatial output 축이다.
    // reduction loop는 PIMFormCompute가 linalg/accumulator 의미에서 붙인
    // pim.reduction으로만 찾는다.
    SmallVector<scf::ForOp> reductionLoops;
    for (scf::ForOp loop : loops)
      if (loop->getAttrOfType<StringAttr>("pim.reduction"))
        reductionLoops.push_back(loop);
    if (reductionLoops.size() > 1)
      return func->emitError(
          "pim-form-execution-tiles: more than one semantic reduction loop "
          "is present");
    scf::ForOp mLoop = axisLoop("m");
    scf::ForOp nLoop = axisLoop("n");
    scf::ForOp redLoop =
        reductionLoops.empty() ? scf::ForOp() : reductionLoops.front();
    if (!func->getAttrOfType<pim::BlockAttr>("pim.block") ||
        !func->getAttrOfType<pim::BlockGroupAttr>("pim.block_group"))
      return func->emitError(
          "pim-form-execution-tiles: #pim.block and #pim.block_group are "
          "required; run --pim-form-block-groups first");

    scf::ForOp axisLoops[3] = {mLoop, nLoop, redLoop};
    int64_t requested[3] = {tileM, tileN, tileK};
    StringRef options[3] = {"tile-m", "tile-n", "tile-k"};
    StringRef axes[3] = {"m", "n", "reduction"};
    // 모든 hierarchy vector는 고정 [m,n,k] 양수 표현이다. 실제 loop가 없는
    // 축도 곱의 항등원인 Group 하나로 기록한다.
    int64_t tileGroups[3] = {1, 1, 1};
    SmallVector<int64_t, 3> groupShape = pim::groupShape(func);
    for (int i = 0; i < 3; ++i) {
      if (!axisLoops[i]) {
        // 0은 pass option의 기본값이고 1은 Common IR의 항등값이다. 둘 다
        // 비활성 축의 Group 하나로 정규화한다. 2 이상은 묶을 loop가 없다.
        if (requested[i] > 1)
          return func->emitError("pim-form-execution-tiles: ")
                 << options[i] << " is given but the kernel has no " << axes[i]
                 << (i == 2 ? " loop" : " axis");
        continue;
      }
      if (groupShape.size() != 3 || groupShape[i] <= 0)
        return func->emitError(
                   "pim-form-execution-tiles: cannot derive the block-group "
                   "extent on the ")
               << axes[i] << " axis";
      tileGroups[i] = requested[i] == 0 ? 1 : requested[i];
    }
    OpBuilder b(func.getContext());
    func->setAttr(
        "pim.execution_tile",
        pim::ExecutionTileAttr::get(func.getContext(),
                                    b.getDenseI64ArrayAttr(tileGroups)));
    return success();
  }

};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMFormExecutionTilesPass() {
  return std::make_unique<PIMFormExecutionTilesPass>();
}
