#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_PIMFORMHOSTPLAN
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace mlir::triton

namespace {

struct PIMFormHostPlanPass
    : mlir::triton::impl::PIMFormHostPlanBase<PIMFormHostPlanPass> {
  using PIMFormHostPlanBase::PIMFormHostPlanBase;

  void runOnOperation() override {
    SmallVector<pim::DeviceOp> devices;
    getOperation().walk([&](pim::DeviceOp device) {
      devices.push_back(device);
    });
    for (pim::DeviceOp device : devices)
      if (failed(formPlan(device)))
        return signalPassFailure();
  }

  LogicalResult formPlan(pim::DeviceOp device) {
    SmallVector<pim::PlaceOp> inputs, outputs;
    SmallVector<Operation *> compute;
    llvm::DenseSet<Operation *> computeSet;
    for (Operation &op : device.getBody().front()) {
      bool usesCompute = llvm::any_of(op.getOperands(), [&](Value operand) {
        return computeSet.contains(operand.getDefiningOp());
      });
      if (auto place = dyn_cast<pim::PlaceOp>(&op)) {
        if (usesCompute)
          return place->emitError(
              "pim-form-host-plan: placement depends on a device compute result");
        (place.getDirection() == "in" ? inputs : outputs).push_back(place);
        continue;
      }
      bool scalarSetup = isa<arith::ArithDialect>(op.getDialect()) &&
                         isMemoryEffectFree(&op) &&
                         llvm::all_of(op.getResultTypes(), [](Type type) {
                           return type.isIntOrIndex() || isa<FloatType>(type);
                         });
      if (scalarSetup && !usesCompute)
        continue;
      compute.push_back(&op);
      computeSet.insert(&op);
    }
    if (compute.empty())
      return device->emitError("pim-form-host-plan: no device compute body");
    OpBuilder b(device);
    auto partition = b.create<pim::PartitionOp>(device.getLoc(),
                                              device.getMapping());
    for (NamedAttribute attr : device->getAttrs())
      partition->setAttr(attr.getName(), attr.getValue());
    partition.getBody().takeBody(device.getBody());
    device.erase();

    b.setInsertionPointToEnd(&partition.getBody().front());
    for (pim::PlaceOp input : inputs) {
      if (auto shared = input.getSharedOverAttr(); shared && !shared.empty())
        b.create<pim::BroadcastOp>(input.getLoc(), input.getAccessAttr());
      else
        b.create<pim::ScatterOp>(input.getLoc(), input.getAccessAttr());
    }

    auto launch = b.create<pim::LaunchOp>(partition.getLoc());
    Block *body = b.createBlock(&launch.getBody());
    for (Operation *op : compute)
      op->moveBefore(body, body->end());

    b.setInsertionPointAfter(launch);
    for (pim::PlaceOp output : outputs)
      b.create<pim::GatherOp>(output.getLoc(), output.getAccessAttr());
    return partition.verify();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMFormHostPlanPass() {
  return std::make_unique<PIMFormHostPlanPass>();
}
