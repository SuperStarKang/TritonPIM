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
      for (scf::ForOp l : loops) {
        // reduction loop 는 pim.axis 가 없다 — null StringAttr 를 StringRef 와
        // 비교하면 getValue() 가 터지므로 걸러서 지난다.
        auto a = l->getAttrOfType<StringAttr>("pim.axis");
        if (a && a.getValue() == name)
          return l;
      }
      return scf::ForOp();
    };
    // k 상한의 축은 둘 중 하나다. explicit_k_partition 이 있으면 공간 분할의 눈금이고,
    // 없으면 reduction loop 다 — "한 번에 상주하는 k 구간의 상한"이라는 뜻은 같고
    // 소비자만 다르다. reduction loop 는 pim-form-block-groups 의 group-k 와 같은
    // 판정으로 찾는다: pim.axis 없이 access 의 origin 을 모는 loop, 하나일 때만 유일.
    auto reductionLoop = [&]() -> scf::ForOp {
      llvm::SmallPtrSet<Operation *, 4> reds;
      device.getBody().walk([&](pim::AccessOp a) {
        for (Value o : a.getOrigin()) {
          llvm::SmallPtrSet<Operation *, 4> ds;
          pim::collectDrivers(o, loops, ds);
          for (Operation *op : ds)
            if (!op->getAttrOfType<StringAttr>("pim.axis"))
              reds.insert(op);
        }
      });
      return reds.size() == 1 ? cast<scf::ForOp>(*reds.begin()) : scf::ForOp();
    };
    struct Cap {
      int64_t cap;
      StringRef axis;
      StringRef option;
    };
    Cap caps[] = {{tileM, "output_row_tile", "tile-m"},
                  {tileN, "output_column_tile", "tile-n"},
                  {tileK, "explicit_k_partition", "tile-k"}};
    // execution tile 은 block group 정수 개다. group 이 없으면(눈금 없는 기계) block
    // 정수 개이고, explicit_k_partition 은 공간 분할의 눈금이라 언제나 block 기준이다.
    auto func = device->getParentOfType<func::FuncOp>();
    auto group = func ? func->getAttrOfType<pim::BlockGroupAttr>("pim.block_group")
                      : pim::BlockGroupAttr();
    for (const Cap &c : caps) {
      if (c.cap == 0)
        continue;
      scf::ForOp loop = axisLoop(c.axis);
      bool isK = c.axis == "explicit_k_partition";
      bool viaReduction = false;
      if (!loop && isK) {
        loop = reductionLoop();
        viaReduction = true;
      }
      if (!loop)
        return device->emitError("pim-form-execution-tiles: ")
               << c.option << " is given but the kernel has no " << c.axis
               << " axis" << (isK ? " and no single reduction loop" : "");
      int idx = isK ? (viaReduction ? 2 : -1)
                    : (c.axis == "output_row_tile" ? 0 : 1);
      int64_t elems;
      StringRef grain;
      if (group && idx >= 0) {
        elems = group.getUseful().asArrayRef()[idx];
        grain = "block group";
      } else {
        elems = pim::tileElementsAlong(loop, loops);
        grain = "tile";
      }
      if (elems <= 0)
        return device->emitError(
                   "pim-form-execution-tiles: cannot tell how many elements "
                   "one ")
               << c.axis << " tile moves, so the cap cannot be validated";
      if (c.cap % elems != 0)
        return device->emitError("pim-form-execution-tiles: ")
               << c.option << "(" << c.cap << ") is not a multiple of the "
               << c.axis << " " << grain << " (" << elems << " elements)";
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
