#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

// GEN_PASS_DEF가 생성하는 impl::...Base는 mlir::triton 네임스페이스 안에 있어야 한다.
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTTSHAREDTOPIM
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// tt.dot은 triton-to-linalg가 linalg.matmul로 낮춘다(별도 marker attribute는 붙지 않는다).
// MVP는 커널당 dot 1개(단일 tile)를 가정하므로 첫 linalg.matmul을 그대로 집는다.
static linalg::MatmulOp findMatmul(func::FuncOp func) {
  linalg::MatmulOp found;
  func.walk([&](linalg::MatmulOp op) {
    found = op;
    return WalkResult::interrupt();
  });
  return found;
}

struct ConvertTTSharedToPIMPass
    : public mlir::triton::impl::ConvertTTSharedToPIMBase<ConvertTTSharedToPIMPass> {
  using ConvertTTSharedToPIMBase::ConvertTTSharedToPIMBase;

  void runOnOperation() override {
    getOperation().walk([&](func::FuncOp func) {
      if (linalg::MatmulOp matmul = findMatmul(func))
        rewriteToGemm(func, matmul);
    });
  }

  // tiled gemm 변환 (grid-as-SPLIT):
  // - BLOCK_M/N/K는 matmul operand/result shape에서 읽는다 (한 program이 처리하는 타일 크기).
  //   (operand0 = A tile [BLOCK_M, BLOCK_K], result = C tile [BLOCK_M, BLOCK_N])
  // - 전체 M/N/K는 런타임 인자다. Triton 커널 시그니처 (A,B,C,M,N,K,...) 규약상 인자 3,4,5.
  //   Triton은 program_id로 타일을 나누지만(SPMD) 전체 크기는 dynamic이라 IR에서 상수로 못 읽는다.
  // - A/B/C는 함수 인자 앞 3개.
  // - 함수 본문 전체를 pim 시퀀스로 교체한다. grid/program_id/K-loop는 여기서 버리되,
  //   BLOCK(tile_plan)+전체 M/N/K(compute operand)를 남겨 두면 grid=ceildiv(전체,BLOCK)로
  //   복원 가능하다 (SPLIT은 O4에서 이 정보로 타일을 유닛에 배정).
  void rewriteToGemm(func::FuncOp func, linalg::MatmulOp matmul) {
    auto lhsTy = cast<RankedTensorType>(matmul->getOperand(0).getType());
    auto resTy = cast<RankedTensorType>(matmul->getResult(0).getType());
    uint64_t blockM = resTy.getDimSize(0);
    uint64_t blockN = resTy.getDimSize(1);
    uint64_t blockK = lhsTy.getDimSize(1);

    Value A = func.getArgument(0);
    Value B = func.getArgument(1);
    Value C = func.getArgument(2);
    // 전체 문제 크기 (dynamic): matmul(A,B,C,M,N,K) 규약상 i32 인자 3,4,5.
    Value fullM = func.getArgument(3);
    Value fullN = func.getArgument(4);
    Value fullK = func.getArgument(5);

    // 기존 본문을 뒤에서부터 모두 지운다 (사용처를 먼저 지워 use-def 위반 방지).
    // block 인자(A,B,C,...)는 유지된다.
    Block &entry = func.getBody().front();
    while (!entry.empty())
      entry.back().erase();

    OpBuilder b = OpBuilder::atBlockBegin(&entry);
    Location loc = func.getLoc();
    auto regionTy = pim::RegionType::get(b.getContext());
    auto tpTy = pim::TilePlanType::get(b.getContext());

    Value aReg = b.create<pim::MemoryRegionOp>(loc, regionTy, A, "host", "row_major",
                                               "input_activation");
    Value bReg = b.create<pim::MemoryRegionOp>(loc, regionTy, B, "device_global",
                                               "row_major", "weight");
    Value cReg = b.create<pim::MemoryRegionOp>(loc, regionTy, C, "host", "row_major",
                                               "output");

    // split_n은 pass 옵션(=compile option). >1이면 vendor가 N축을 유닛에 분배한다.
    Value tp = b.create<pim::TilePlanOp>(loc, tpTy, blockM, blockN, blockK,
                                         /*split_m=*/1, /*split_n=*/(uint64_t)splitN,
                                         /*split_k=*/1);

    b.create<pim::TransferOp>(loc, aReg, "host", "device_global");
    b.create<pim::TransferOp>(loc, bReg, "host", "device_global");
    b.create<pim::ComputeOp>(loc, regionTy, aReg, bReg, cReg, tp, fullM, fullN,
                             fullK, "gemm");
    b.create<pim::TransferOp>(loc, cReg, "device_global", "host");
    b.create<pim::LaunchOp>(loc, func.getName(), vendor, /*device_id=*/0);

    b.create<func::ReturnOp>(loc);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTTSharedToPIMPass() {
  return std::make_unique<ConvertTTSharedToPIMPass>();
}
