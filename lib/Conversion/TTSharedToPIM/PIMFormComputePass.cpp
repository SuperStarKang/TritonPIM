#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMFORMCOMPUTE
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// generic이 두 입력을 그대로 더해 내놓기만 하는가. 축이 전부 parallel이고 세
// map이 항등이며 body가 두 입력 인자의 덧셈 하나여야 누산기에 접어 넣을 수
// 있다.
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

// generic이 두 입력의 elementwise 곱이기만 하는가. GEMV frontend 패턴의
// A * broadcast(x)를 named linalg.matvec로 올리기 전에 의미가 바뀌지 않음을
// 확인하는 검사다.
static bool isElementwiseMul(linalg::GenericOp gen) {
  if (gen.getInputs().size() != 2 || gen.getOutputs().size() != 1)
    return false;
  for (AffineMap m : gen.getIndexingMapsArray())
    if (!m.isIdentity())
      return false;
  for (utils::IteratorType t : gen.getIteratorTypesArray())
    if (t != utils::IteratorType::parallel)
      return false;
  auto yield = cast<linalg::YieldOp>(gen.getBody()->getTerminator());
  Operation *mul = yield->getOperand(0).getDefiningOp();
  if (!mul || !isa<arith::MulFOp, arith::MulIOp>(mul))
    return false;
  Value a0 = gen.getBody()->getArgument(0), a1 = gen.getBody()->getArgument(1);
  return (mul->getOperand(0) == a0 && mul->getOperand(1) == a1) ||
         (mul->getOperand(0) == a1 && mul->getOperand(1) == a0);
}

// reduce body가 input + accumulator 하나인 sum reduction인가.
static bool isAddReduction(linalg::ReduceOp reduce) {
  if (reduce.getInputs().size() != 1 || reduce.getInits().size() != 1 ||
      reduce->getNumResults() != 1)
    return false;
  Block *body = reduce.getBody();
  if (body->getNumArguments() != 2)
    return false;
  auto yield = cast<linalg::YieldOp>(body->getTerminator());
  Operation *add = yield->getOperand(0).getDefiningOp();
  if (!add || !isa<arith::AddFOp, arith::AddIOp>(add))
    return false;
  Value input = body->getArgument(0), acc = body->getArgument(1);
  return (add->getOperand(0) == input && add->getOperand(1) == acc) ||
         (add->getOperand(0) == acc && add->getOperand(1) == input);
}

// tensor<K> -> tensor<1xK> -> tensor<MxK> broadcast 패턴에서 원래 vector를
// 얻는다. map을 정확히 (m, k) -> (0, k)로 제한해 다른 broadcast가
// 우연히 GEMV로 바뀌는 것을 막는다.
static Value matchBroadcastedVector(Value value, RankedTensorType matrixType) {
  auto broadcast = value.getDefiningOp<linalg::GenericOp>();
  if (!broadcast || broadcast.getInputs().size() != 1 ||
      broadcast.getOutputs().size() != 1 || broadcast->getNumResults() != 1)
    return {};
  for (utils::IteratorType t : broadcast.getIteratorTypesArray())
    if (t != utils::IteratorType::parallel)
      return {};
  ArrayRef<AffineMap> maps = broadcast.getIndexingMapsArray();
  if (maps.size() != 2 || !maps[1].isIdentity() || maps[0].getNumDims() != 2 ||
      maps[0].getNumResults() != 2)
    return {};
  auto zero = dyn_cast<AffineConstantExpr>(maps[0].getResult(0));
  auto kDim = dyn_cast<AffineDimExpr>(maps[0].getResult(1));
  if (!zero || zero.getValue() != 0 || !kDim || kDim.getPosition() != 1)
    return {};
  auto resultType = dyn_cast<RankedTensorType>(value.getType());
  if (!resultType || resultType != matrixType)
    return {};
  auto yield = cast<linalg::YieldOp>(broadcast.getBody()->getTerminator());
  if (yield.getValues().size() != 1 ||
      yield.getValues()[0] != broadcast.getBody()->getArgument(0))
    return {};

  auto expand = broadcast.getInputs()[0].getDefiningOp<tensor::ExpandShapeOp>();
  if (!expand)
    return {};
  auto expandedType = dyn_cast<RankedTensorType>(expand.getResult().getType());
  auto vectorType = dyn_cast<RankedTensorType>(expand.getSrc().getType());
  if (!expandedType || !vectorType || expandedType.getRank() != 2 ||
      vectorType.getRank() != 1 || expandedType.getShape()[0] != 1 ||
      expandedType.getShape()[1] != matrixType.getShape()[1] ||
      vectorType.getShape()[0] != matrixType.getShape()[1])
    return {};
  return expand.getSrc();
}

// INT8 GEMV는 곱하기 전에 행렬과 vector를 INT32로 넓힌다. 그 sign extension
// (elementwise generic 또는 tensor arith.extsi) 앞의 좁은 값을 돌려준다.
// linalg.matvec는 입력을 출력 type으로 sign extension해 곱하므로(cast_signed)
// 넓히기 전 값을 바로 입력으로 써도 의미가 같다. extension이 아니면 그대로 둔다.
static Value stripSignExtension(Value value) {
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    return ext.getIn();
  auto gen = value.getDefiningOp<linalg::GenericOp>();
  if (!gen || gen.getInputs().size() != 1 || gen.getOutputs().size() != 1)
    return value;
  for (AffineMap m : gen.getIndexingMapsArray())
    if (!m.isIdentity())
      return value;
  for (utils::IteratorType t : gen.getIteratorTypesArray())
    if (t != utils::IteratorType::parallel)
      return value;
  auto yield = cast<linalg::YieldOp>(gen.getBody()->getTerminator());
  auto ext = yield->getOperand(0).getDefiningOp<arith::ExtSIOp>();
  if (!ext || ext.getIn() != gen.getBody()->getArgument(0))
    return value;
  return gen.getInputs()[0];
}

static bool allResultsUnused(Operation *op) {
  return llvm::all_of(op->getResults(), [](Value v) { return v.use_empty(); });
}

// 0으로 채운 tile인가. 0이라야 "0에 A·B를 더한 것"을 누산기에 더하는 것과
// 누산기에 바로 더하는 것이 같다.
static bool isZeroFill(Value v) {
  auto fill = v.getDefiningOp<linalg::FillOp>();
  if (!fill)
    return false;
  Value f = fill.getInputs()[0];
  return matchPattern(f, m_Zero()) || matchPattern(f, m_AnyZeroFloat());
}

// Common IR이 현재 표현할 수 있는 reduction combiner를 모든 linalg op의 scalar
// body에서 읽는다. op 이름을 열거하지 않으므로 matmul/matvec/reduce뿐 아니라 같은
// add-accumulator 계약의 convolution/generic도 동일하게 처리한다. access 주소를
// 모는 loop인지 여부는 데이터 이동 정보일 뿐 연산 의미가 아니므로 사용하지 않는다.
static bool dependsOn(Value value, Value root) {
  SmallVector<Value> work{value};
  llvm::DenseSet<Value> seen;
  while (!work.empty()) {
    Value current = work.pop_back_val();
    if (!seen.insert(current).second)
      continue;
    if (current == root)
      return true;
    if (Operation *def = current.getDefiningOp())
      work.append(def->getOperands().begin(), def->getOperands().end());
  }
  return false;
}

enum class AddComputeKind {
  NotAddAccumulator,
  IndependentPartial,
  AccumulatorDependentPartial,
};

static AddComputeKind classifyAddCompute(linalg::LinalgOp compute) {
  if (compute.getNumDpsInits() != 1 || compute->getNumRegions() != 1 ||
      compute->getNumResults() != 1)
    return AddComputeKind::NotAddAccumulator;
  Region &region = compute->getRegion(0);
  if (region.empty())
    return AddComputeKind::NotAddAccumulator;
  Block &body = region.front();
  unsigned accumulatorIndex = compute.getDpsInputs().size();
  if (body.getNumArguments() <= accumulatorIndex)
    return AddComputeKind::NotAddAccumulator;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return AddComputeKind::NotAddAccumulator;
  Operation *add = yield.getValues()[0].getDefiningOp();
  if (!add || !isa<arith::AddFOp, arith::AddIOp>(add))
    return AddComputeKind::NotAddAccumulator;
  Value accumulator = body.getArgument(accumulatorIndex);
  Value partial;
  if (add->getOperand(0) == accumulator)
    partial = add->getOperand(1);
  else if (add->getOperand(1) == accumulator)
    partial = add->getOperand(0);
  else
    return AddComputeKind::NotAddAccumulator;
  return dependsOn(partial, accumulator)
             ? AddComputeKind::AccumulatorDependentPartial
             : AddComputeKind::IndependentPartial;
}

// 누산기 초기값을 만드는 op을 그 loop 바로 앞으로 내린다. 값 의미로는 밖에
// 있어도 맞지만 버퍼로 내리면 초기화가 한 번만 돌아 두 번째 instance부터 앞
// 결과가 남는다. 쓰는 곳이 그 loop 하나여야 옮긴다.
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
  // chain은 소비하는 쪽부터 담겼다. 생산하는 쪽부터 옮겨야 순서가 맞는다.
  for (Operation *op : llvm::reverse(chain))
    op->moveBefore(loop);
}

struct PIMFormComputePass
    : public mlir::triton::impl::PIMFormComputeBase<PIMFormComputePass> {
  using PIMFormComputeBase::PIMFormComputeBase;

  void runOnOperation() override {
    // 안쪽 op을 지우므로 loop를 먼저 모은 뒤 돈다.
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp l) { loops.push_back(l); });
    for (scf::ForOp l : loops)
      foldIntoAccumulator(l);
    // Triton GEMV의 broadcast -> mul -> transpose -> reduce를 target에 의존하지
    // 않는 named op으로 정규화한다. frontend가 linalg.matvec를 바로 내면
    // 이 단계는 그 op을 그대로 둔다.
    SmallVector<linalg::ReduceOp> reductions;
    getOperation().walk(
        [&](linalg::ReduceOp reduce) { reductions.push_back(reduce); });
    for (linalg::ReduceOp reduce : reductions)
      formMatvec(reduce);
    // linalg compute가 loop-carried accumulator를 직접 갱신하고 그 결과가 같은
    // iter_arg 위치로 yield될 때만 이 scf.for를 의미상 reduction으로 기록한다.
    // 뒤 패스는 이 표식을 소비하며 access-driving loop를 reduction으로 추측하지
    // 않는다.
    for (scf::ForOp l : loops)
      if (failed(markReduction(l)))
        return signalPassFailure();
    // 접은 뒤라야 초기값이 iter_args에서만 쓰여 내릴 수 있다.
    for (scf::ForOp l : loops)
      sinkInit(l);
  }

  LogicalResult markReduction(scf::ForOp loop) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    bool found = false;
    for (auto [i, acc] : llvm::enumerate(loop.getRegionIterArgs())) {
      if (!isa<RankedTensorType>(acc.getType()))
        continue;
      Operation *def = yield.getOperand(i).getDefiningOp();
      auto compute = dyn_cast_or_null<linalg::LinalgOp>(def);
      if (!compute || compute.getNumDpsInits() != 1 ||
          compute.getDpsInits()[0] != acc)
        continue;
      AddComputeKind kind = classifyAddCompute(compute);
      if (kind == AddComputeKind::AccumulatorDependentPartial)
        return compute->emitError(
            "pim-form-compute: the add contribution must not depend on the "
            "previous accumulator before it is split into Execution Tiles");
      if (kind != AddComputeKind::IndependentPartial)
        continue;
      if (!isZeroFill(loop.getInitArgs()[i]))
        return loop->emitError(
            "pim-form-compute: an add reduction must start from a zero "
            "identity before it can be split into Execution Tiles");
      found = true;
    }
    if (found) {
      if (loop.getRegionIterArgs().size() != 1 || loop.getNumResults() != 1)
        return loop->emitError(
            "pim-form-compute: a normalized PIM reduction loop must carry "
            "exactly one accumulator result");
      loop->setAttr("pim.reduction",
                    StringAttr::get(loop.getContext(), "add"));
    }
    return success();
  }

  // TTShared가 GEMV를 내는 두 reduce 배치를 모두 받는다.
  //   A[M,K] * broadcast(x[K]) -> transpose[K,M] -> reduce(dim=0)
  //   A[M,K] * broadcast(x[K])                  -> reduce(dim=1)
  // accumulator가 reduce init에 이미 접힌 뒤에 실행하므로 생성된
  // linalg.matvec도 같은 K-loop 누산 의미를 가진다.
  void formMatvec(linalg::ReduceOp reduce) {
    if (!isAddReduction(reduce) || reduce.getDimensions().size() != 1)
      return;

    Value product = reduce.getInputs()[0];
    linalg::TransposeOp transpose;
    int64_t reductionDim = reduce.getDimensions()[0];
    if (reductionDim == 0) {
      transpose = product.getDefiningOp<linalg::TransposeOp>();
      if (!transpose || transpose.getPermutation().size() != 2 ||
          transpose.getPermutation()[0] != 1 ||
          transpose.getPermutation()[1] != 0)
        return;
      product = transpose.getInput();
    } else if (reductionDim != 1) {
      return;
    }

    auto productType = dyn_cast<RankedTensorType>(product.getType());
    auto resultType = dyn_cast<RankedTensorType>(reduce.getResult(0).getType());
    if (!productType || !resultType || productType.getRank() != 2 ||
        resultType.getRank() != 1 ||
        productType.getShape()[0] != resultType.getShape()[0] ||
        productType.getElementType() != resultType.getElementType())
      return;

    auto mul = product.getDefiningOp<linalg::GenericOp>();
    if (!mul || !isElementwiseMul(mul))
      return;
    Value matrix, vector;
    for (unsigned broadcastOperand = 0; broadcastOperand < 2;
         ++broadcastOperand) {
      Value input = mul.getInputs()[broadcastOperand];
      Value candidate = matchBroadcastedVector(input, productType);
      // INT8 vector를 먼저 broadcast하고 넓힌 경우: extension 앞의 broadcast를 본다.
      if (!candidate) {
        Value narrow = stripSignExtension(input);
        if (narrow != input)
          candidate = matchBroadcastedVector(
              narrow, cast<RankedTensorType>(narrow.getType()));
      }
      if (!candidate)
        continue;
      Value other = mul.getInputs()[1 - broadcastOperand];
      if (other.getType() != productType)
        continue;
      matrix = other;
      vector = candidate;
      break;
    }
    if (!matrix || !vector)
      return;
    // 넓히기 전 INT8 행렬과 vector를 matvec 입력으로 쓴다. WRAM에 INT32 사본을
    // 만들지 않고, device 곱셈도 좁은 값으로 한다.
    matrix = stripSignExtension(matrix);
    vector = stripSignExtension(vector);
    SmallVector<Value> mulInputs(mul.getInputs().begin(), mul.getInputs().end());

    OpBuilder b(reduce);
    auto matvec =
        b.create<linalg::MatvecOp>(reduce.getLoc(), ValueRange{matrix, vector},
                                   ValueRange{reduce.getInits()[0]});
    reduce.getResult(0).replaceAllUsesWith(matvec.getResult(0));

    // 다른 사용자가 있으면 원래 계산을 보존한다. GEMV 전용 연쇄인
    // 경우에만 소비자에서 생산자 순으로 지워 중간 tile을 없앤다. mul 입력에서
    // matvec가 쓰는 값까지 거슬러 오르는 broadcast, expand_shape, extension이 대상이다.
    Operation *transposeOp = transpose ? transpose.getOperation() : nullptr;
    SmallVector<Operation *> deadCandidates{reduce, transposeOp, mul.getOperation()};
    for (Value input : mulInputs) {
      Value value = input;
      while (value != matrix && value != vector) {
        Operation *def = value.getDefiningOp();
        if (!def || !isa<linalg::GenericOp, tensor::ExpandShapeOp, arith::ExtSIOp>(def))
          break;
        deadCandidates.push_back(def);
        value = def->getOperand(0);
      }
    }
    for (Operation *op : deadCandidates)
      if (op && allResultsUnused(op))
        op->erase();
  }

  // 누산기 iter_arg마다 matmul(outs=zero) + generic(add) 형태를
  // matmul(outs=누산기)로 접는다.
  void foldIntoAccumulator(scf::ForOp loop) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [i, acc] : llvm::enumerate(loop.getRegionIterArgs())) {
      if (!isa<RankedTensorType>(acc.getType()))
        continue;
      auto gen = yield.getOperand(i).getDefiningOp<linalg::GenericOp>();
      if (!gen || gen.getOutputs()[0] != acc || !isElementwiseAdd(gen))
        continue;
      // 입력 둘 중 하나가 누산기, 나머지가 이번 반복의 계산 결과다.
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
mlir::triton::createPIMFormComputePass() {
  return std::make_unique<PIMFormComputePass>();
}
