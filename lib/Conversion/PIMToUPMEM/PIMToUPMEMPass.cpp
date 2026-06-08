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

  // A/B를 MRAM으로 올리고, split_n개 DPU가 gemm을 수행한 뒤 C를 내린다.
  // split_n=1이면 단일 유닛(기존 동작). >1이면 출력 N축을 유닛에 분배(n-split) —
  // 실제 per-DPU 분배·gather는 vendor codegen(UPMEMToHost)이 num_units를 보고 생성한다.
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
    // faithful-body(O1.4): pim.compute의 본문(scf/linalg/memref)을 device_func로 옮긴다.
    pim::ComputeOp compute;
    func.walk([&](pim::ComputeOp c) { compute = c; });

    // memory_region의 buffer는 함수 인자(block arg)라 본문을 지워도 유효하다.
    Value A = aReg.getBuffer();
    Value B = bReg.getBuffer();
    Value C = cReg.getBuffer();
    uint64_t splitM = tp.getSplitM();
    uint64_t splitN = tp.getSplitN();
    uint64_t splitK = tp.getSplitK();
    uint64_t numUnits = splitM * splitN * splitK;  // 총 DPU 수 (2D 출력 분배 × K 분담)
    std::string devName = (func.getName() + "_device").str();
    Location loc = func.getLoc();

    // device_func를 만들고 pim.compute 본문을 그대로 옮긴다(BLOCK·계산 구조 보존).
    // device codegen(UPMEMToDevice)이 이 본문에서 BLOCK·gemm 구조를 읽어 device.c를 생성한다.
    // 본문을 옮긴 뒤 entry를 지워야 하므로 erase보다 먼저 수행한다(takeBody는 compute의 region을 비운다).
    OpBuilder mb(module.getBody(), module.getBody()->end());
    auto dev = mb.create<upmem::DeviceFuncOp>(loc, devName);
    dev.getBody().takeBody(compute.getBody());

    // host 시퀀스로 함수 본문 교체 (compute 포함 entry op 전부 제거; 본문은 이미 device_func로 이동됨).
    Block &entry = func.getBody().front();
    while (!entry.empty())
      entry.back().erase();

    OpBuilder b = OpBuilder::atBlockBegin(&entry);
    auto setTy = upmem::DPUSetType::get(b.getContext());

    Value units = b.create<upmem::AllocUnitsOp>(loc, setTy, /*num_units=*/numUnits,
                                                /*split_m=*/splitM, /*split_n=*/splitN,
                                                /*split_k=*/splitK);
    b.create<upmem::LoadProgramOp>(loc, units, devName);
    b.create<upmem::CopyToGlobalOp>(loc, units, A, "mram_a");
    b.create<upmem::CopyToGlobalOp>(loc, units, B, "mram_b");
    b.create<upmem::LaunchOp>(loc, units);
    b.create<upmem::CopyFromGlobalOp>(loc, units, C, "mram_c");
    b.create<func::ReturnOp>(loc);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createLegalizePIMToUPMEMPass() {
  return std::make_unique<LegalizePIMToUPMEMPass>();
}
