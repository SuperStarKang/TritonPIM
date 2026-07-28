#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMNORMALIZEACCUMULATOR
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// generic이 두 입력을 그대로 더해 내놓기만 하는가. 축이 전부 parallel이고 세 map이 항등이며
// body가 두 입력 인자의 덧셈 하나여야 한다 — 다른 계산이 섞이면 누산기에 접어 넣을 수 없다.
static bool isElementwiseAdd(linalg::GenericOp gen) {
  if (gen.getInputs().size() != 2 || gen.getOutputs().size() != 1)
    return false;
  for (AffineMap m : gen.getIndexingMapsArray())
    if (!m.isIdentity())
      return false;
  for (utils::IteratorType t : gen.getIteratorTypesArray())
    if (t != utils::IteratorType::parallel)
      return false;
  auto yield = cast<linalg::YieldOp>(gen.getBody()->getTerminator());
  Operation *add = yield->getOperand(0).getDefiningOp();
  if (!add || !isa<arith::AddFOp, arith::AddIOp>(add))
    return false;
  Value a0 = gen.getBody()->getArgument(0), a1 = gen.getBody()->getArgument(1);
  return (add->getOperand(0) == a0 && add->getOperand(1) == a1) ||
         (add->getOperand(0) == a1 && add->getOperand(1) == a0);
}

// 0으로 채운 tile인가. 0이라야 "0에 A·B를 더한 것"을 누산기에 더하는 것과 누산기에 바로
// 더하는 것이 같다.
static bool isZeroFill(Value v) {
  auto fill = v.getDefiningOp<linalg::FillOp>();
  if (!fill)
    return false;
  Value f = fill.getInputs()[0];
  return matchPattern(f, m_Zero()) || matchPattern(f, m_AnyZeroFloat());
}

// 누산기 초기값을 만드는 op을 그 loop 바로 앞으로 내린다.
// tensor는 값 의미라 -loop-invariant-code-motion이 이걸 grid loop 밖까지 빼 둔다. 값으로는
// 맞지만 버퍼로 내리면 틀린다 — 초기화가 한 번만 돌아 두 번째 instance부터는 앞 결과가
// 남는다. loop 바로 앞으로 내려 두면 instance마다 새로 채워진다.
// 쓰는 곳이 그 loop 하나여야 옮긴다(여러 곳에서 쓰면 공유하겠다는 뜻이다).
static void sinkInit(scf::ForOp loop) {
  SmallVector<Operation *> chain;
  SmallVector<Value> work(loop.getInitArgs());
  while (!work.empty()) {
    Operation *def = work.pop_back_val().getDefiningOp();
    if (!def || def->getNumResults() != 1 ||
        !isa<RankedTensorType>(def->getResult(0).getType()) ||
        !isMemoryEffectFree(def) || !def->getResult(0).hasOneUse())
      continue;
    chain.push_back(def);
    work.append(def->getOperands().begin(), def->getOperands().end());
  }
  // 소비하는 쪽부터 옮기므로 생산하는 쪽이 앞에 오게 된다.
  for (Operation *op : chain)
    op->moveBefore(loop);
}

struct PIMNormalizeAccumulatorPass
    : public mlir::triton::impl::PIMNormalizeAccumulatorBase<
          PIMNormalizeAccumulatorPass> {
  using PIMNormalizeAccumulatorBase::PIMNormalizeAccumulatorBase;

  void runOnOperation() override {
    // 안쪽 op을 지우므로 loop를 먼저 모은 뒤 돈다.
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp l) { loops.push_back(l); });
    for (scf::ForOp l : loops)
      normalizeLoop(l);
    // 접은 뒤라야 초기값이 iter_args에서만 쓰여 내릴 수 있다.
    for (scf::ForOp l : loops)
      sinkInit(l);
  }

  // 누산기 iter_arg마다 matmul(outs=zero) + generic(add) 형태를 matmul(outs=누산기)로 접는다.
  void normalizeLoop(scf::ForOp loop) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [i, acc] : llvm::enumerate(loop.getRegionIterArgs())) {
      if (!isa<RankedTensorType>(acc.getType()))
        continue;
      auto gen = yield.getOperand(i).getDefiningOp<linalg::GenericOp>();
      if (!gen || gen.getOutputs()[0] != acc || !isElementwiseAdd(gen))
        continue;
      // 입력 둘 중 하나가 누산기, 나머지가 이번 반복의 계산 결과.
      Value other;
      if (gen.getInputs()[0] == acc)
        other = gen.getInputs()[1];
      else if (gen.getInputs()[1] == acc)
        other = gen.getInputs()[0];
      else
        continue;
      auto compute = other.getDefiningOp<linalg::LinalgOp>();
      // 결과가 여기서만 쓰여야 outs를 바꿔도 다른 곳이 영향받지 않는다.
      if (!compute || !other.hasOneUse() || compute.getNumDpsInits() != 1)
        continue;
      if (!isZeroFill(compute.getDpsInits()[0]))
        continue;
      compute.setDpsInitOperand(0, acc);
      gen.getResult(0).replaceAllUsesWith(other);
      gen.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMNormalizeAccumulatorPass() {
  return std::make_unique<PIMNormalizeAccumulatorPass>();
}
