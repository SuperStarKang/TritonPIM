#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMCLASSIFYCOMPUTE
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// 연산의 target-무관 "종류"만 판별한다(합법성·opcode·분해는 target 몫). "" = 태그 없음.
// generic은 iterator/indexing_map/body 신호로 세분한다.
static StringRef classifyComputeOp(Operation *op) {
  if (isa<linalg::MatmulOp>(op))
    return "gemm";
  if (isa<linalg::MatvecOp>(op))
    return "gemv";
  if (isa<linalg::TransposeOp>(op))
    return "transpose";
  if (isa<linalg::ReduceOp>(op))
    return "reduce";
  if (isa<linalg::FillOp>(op))
    return "fill";
  auto g = dyn_cast<linalg::GenericOp>(op);
  if (!g)
    return "";
  // reduction iterator가 있으면 reduce.
  if (cast<linalg::LinalgOp>(op).getNumReductionLoops() > 0)
    return "reduce";
  // body에 linalg.index가 있으면 index→value 생성(iota).
  bool hasIndex = false;
  g.walk([&](linalg::IndexOp) { hasIndex = true; });
  if (hasIndex)
    return "index_generation";
  // 입력 인덱스 맵이 차원을 drop하면(결과 수 < 도메인 수) broadcast.
  SmallVector<AffineMap> maps = g.getIndexingMapsArray();
  for (unsigned i = 0, n = g.getNumDpsInputs(); i < n; i++)
    if (maps[i].getNumResults() < maps[i].getNumDims())
      return "broadcast";
  return "elementwise"; // 전부 parallel + identity map = pointwise
}

struct PIMClassifyComputePass
    : public mlir::triton::impl::PIMClassifyComputeBase<PIMClassifyComputePass> {
  using PIMClassifyComputeBase::PIMClassifyComputeBase;

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      StringRef cls = classifyComputeOp(op);
      if (!cls.empty())
        op->setAttr("pim.op", StringAttr::get(op->getContext(), cls));
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMClassifyComputePass() {
  return std::make_unique<PIMClassifyComputePass>();
}
