#include "triton-shared/Conversion/PIMToUPMEM/PIMToUPMEM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIMVendor/UPMEM/IR/UPMEMDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

// GEN_PASS_DEF가 생성하는 impl::...Base는 mlir::triton 네임스페이스 안에 있어야 한다.
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_LEGALIZEPIMTOUPMEM
#include "triton-shared/Conversion/PIMToUPMEM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct LegalizePIMToUPMEMPass
    : public mlir::triton::impl::LegalizePIMToUPMEMBase<LegalizePIMToUPMEMPass> {
  using LegalizePIMToUPMEMBase::LegalizePIMToUPMEMBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // device_func를 module에 추가하므로 walk 중 변형을 피하려 func을 먼저 모은다.
    SmallVector<func::FuncOp> funcs;
    module.walk([&](func::FuncOp f) {
      bool hasGemm = false;
      f.walk([&](pim::ComputeOp) { hasGemm = true; });
      if (hasGemm)
        funcs.push_back(f);
    });
    for (func::FuncOp f : funcs)
      legalize(f, module);
  }

  // no-split 단일 tile: A/B는 MRAM으로 올리고, DPU 1개가 gemm을 수행한 뒤 C를 내린다.
  void legalize(func::FuncOp func, ModuleOp module) {
    pim::MemoryRegionOp aReg, bReg, cReg;
    func.walk([&](pim::MemoryRegionOp r) {
      StringRef role = r.getRole();
      if (role == "input_activation")
        aReg = r;
      else if (role == "weight")
        bReg = r;
      else if (role == "output")
        cReg = r;
    });
    pim::TilePlanOp tp;
    func.walk([&](pim::TilePlanOp t) { tp = t; });

    // memory_region의 buffer는 함수 인자(block arg)라 본문을 지워도 유효하다.
    Value A = aReg.getBuffer();
    Value B = bReg.getBuffer();
    Value C = cReg.getBuffer();
    uint64_t bm = tp.getBlockM();
    uint64_t bn = tp.getBlockN();
    uint64_t bk = tp.getBlockK();
    std::string devName = (func.getName() + "_device").str();

    Block &entry = func.getBody().front();
    while (!entry.empty())
      entry.back().erase();

    OpBuilder b = OpBuilder::atBlockBegin(&entry);
    Location loc = func.getLoc();
    auto setTy = upmem::DPUSetType::get(b.getContext());

    Value units = b.create<upmem::AllocUnitsOp>(loc, setTy, /*num_units=*/(uint64_t)1);
    b.create<upmem::LoadProgramOp>(loc, units, devName);
    b.create<upmem::CopyToGlobalOp>(loc, units, A, "mram_a");
    b.create<upmem::CopyToGlobalOp>(loc, units, B, "mram_b");
    b.create<upmem::LaunchOp>(loc, units);
    b.create<upmem::CopyFromGlobalOp>(loc, units, C, "mram_c");
    b.create<func::ReturnOp>(loc);

    // 같은 module에 device_func를 추가하고 그 안에 per-DPU gemm을 둔다.
    OpBuilder mb(module.getBody(), module.getBody()->end());
    auto dev = mb.create<upmem::DeviceFuncOp>(loc, devName);
    Block *deviceBlock = mb.createBlock(&dev.getBody());
    mb.setInsertionPointToStart(deviceBlock);
    mb.create<upmem::LocalGemmOp>(loc, bm, bn, bk, "fp32");
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createLegalizePIMToUPMEMPass() {
  return std::make_unique<LegalizePIMToUPMEMPass>();
}
