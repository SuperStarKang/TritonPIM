#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMANALYZETRANSFER
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// 원소 개수를 바이트로 바꾼다. 8비트 미만 타입에서 0이 되지 않도록 올림 나눗셈을 쓴다.
static int64_t bytesOf(int64_t elems, Type elemType) {
  return (elems * elemType.getIntOrFloatBitWidth() + 7) / 8;
}

// 상수 lb/ub/step에서 반복 횟수를 구한다. 하나라도 상수가 아니면 실패.
static bool constTripCount(scf::ForOp loop, int64_t &trips) {
  std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lb || !ub || !step || *step <= 0)
    return false;
  trips = *ub <= *lb ? 0 : (*ub - *lb + *step - 1) / *step;
  return true;
}

// op이 instance 하나 안에서 몇 번 실행되는지. 감싸는 loop 중 grid loop가 아닌 것들의
// 반복 횟수 곱이다. grid loop는 instance를 세는 것이라 여기서 빼고, 전송량 식의 grid
// 인자로 따로 들어간다. scf.if 안이면 한 반복에 최대 한 번 실행되므로 1로 본다.
static bool repeatCount(Operation *op, int64_t &count) {
  count = 1;
  for (Operation *p = op->getParentOp(); p && !isa<func::FuncOp>(p);
       p = p->getParentOp()) {
    if (auto loop = dyn_cast<scf::ForOp>(p)) {
      if (loop->hasAttr("pim.grid_dim"))
        continue;
      int64_t trips;
      if (!constTripCount(loop, trips))
        return false;
      count *= trips;
      continue;
    }
    if (isa<scf::IfOp>(p))
      continue;
    return false; // 모르는 구조 안이면 횟수를 셀 수 없다.
  }
  return true;
}

// access가 한 번 옮기는 tile의 바이트. sizes는 원소 개수라 element type으로 환산한다.
// element type은 base memref에서 읽는다. valid는 동적 값이라 못 쓰므로 sizes 전체를
// 세며, 따라서 경계 tile에서는 실제보다 크게 잡힌 상한이다.
static bool tileBytes(pim::AccessOp access, int64_t &bytes) {
  auto baseTy = dyn_cast<BaseMemRefType>(access.getBase().getType());
  if (!baseTy || !baseTy.getElementType().isIntOrFloat())
    return false;
  int64_t elems = 1;
  for (int64_t s : access.getSizes())
    elems *= s;
  bytes = bytesOf(elems, baseTy.getElementType());
  return true;
}

// access가 instance 하나 안에서 전송되는 횟수. 전송 op(load/store/copy)마다 그것을
// 감싸는 비-grid loop 반복 횟수를 더한다. access op이 아니라 전송 op의 위치를 봐야
// 한다 — hoisting이 access는 두고 load만 loop 밖으로 옮길 수 있다.
// tile 바이트와 곱하지 않고 따로 두는 이유는 셋이다: 곱해 두면 전송 건수를 복원할 수
// 없고, 유닛당 상주량 계산에 tile 바이트가 따로 필요하며, 감싸는 loop 반복이 상수가
// 아니어도 tile 바이트는 남길 수 있다.
static bool transferCount(pim::AccessOp access, int64_t &count) {
  count = 0;
  for (Operation *user : access.getResult().getUsers()) {
    if (!isa<pim::LoadOp, pim::StoreOp, pim::CopyOp>(user))
      continue;
    int64_t repeats;
    if (!repeatCount(user, repeats))
      return false;
    count += repeats;
  }
  return true;
}

// tensor 타입의 바이트. shape이 동적이면 실패.
static bool tensorBytes(Type type, int64_t &bytes) {
  auto tensorTy = dyn_cast<RankedTensorType>(type);
  if (!tensorTy || !tensorTy.hasStaticShape() ||
      !tensorTy.getElementType().isIntOrFloat())
    return false;
  bytes = bytesOf(tensorTy.getNumElements(), tensorTy.getElementType());
  return true;
}

// 블록이 func 몸체에서 몇 단계나 안쪽인지.
static unsigned blockDepth(Block *block) {
  unsigned depth = 0;
  for (Operation *p = block->getParentOp(); p && !isa<func::FuncOp>(p);
       p = p->getParentOp())
    depth++;
  return depth;
}

// 한 순간에 유닛 메모리에 함께 있어야 하는 tile 목록. 가장 깊이 있는 pim.load의 블록을
// 잡아, 그 블록의 load 결과 tensor들과 그 블록이 scf.for 몸체라면 tensor 타입 iter_arg를
// 모은다. GEMM이면 A tile + B tile + 누산기가 된다.
// lifetime은 loop 구조에서 나온다. 블록 안에서 매 반복 새로 읽히면 streaming(버퍼 하나를
// 돌려쓴다), loop를 관통하는 iter_arg면 persistent(자리를 계속 차지한다).
// 합계가 아니라 목록으로 두는 이유는 target의 메모리 계층이 하나가 아니기 때문이다.
// 어느 tile을 빠른 쪽에 두고 어느 것을 스트리밍할지는 개별 크기를 알아야 정한다.
// access 필드는 함수 안 pim.access의 walk 순서 인덱스다(누산기에는 없다).
// pim.load는 !pim.access를 거치고 pim.gather는 base를 직접 받지만, 둘 다 유닛 메모리에
// tile 하나를 만든다. 전송량은 index 텐서 값에 달려 못 세지만(§16.2) 상주량은 결과
// tensor 타입에서 나오므로 gather도 목록에 넣는다. gather에는 access 인덱스가 없다.
static Value residentTile(Operation *op) {
  if (auto load = dyn_cast<pim::LoadOp>(op))
    return load.getResult();
  if (auto gather = dyn_cast<pim::GatherOp>(op))
    return gather.getResult();
  return {};
}

static ArrayAttr coResident(func::FuncOp func) {
  Block *deepest = nullptr;
  unsigned deepestDepth = 0;
  func.walk([&](Operation *op) {
    if (!residentTile(op))
      return;
    unsigned depth = blockDepth(op->getBlock());
    if (!deepest || depth > deepestDepth) {
      deepest = op->getBlock();
      deepestDepth = depth;
    }
  });
  if (!deepest)
    return {};

  MLIRContext *ctx = func.getContext();
  auto i64 = IntegerType::get(ctx, 64);
  auto accessName = StringAttr::get(ctx, "access");
  auto bytesName = StringAttr::get(ctx, "bytes");
  auto lifetimeName = StringAttr::get(ctx, "lifetime");

  SmallVector<Attribute> entries;
  for (Operation &op : *deepest) {
    Value tileValue = residentTile(&op);
    if (!tileValue)
      continue;
    int64_t tile;
    // 하나라도 못 세면 목록이 불완전해져 용량 판정에 쓸 수 없으므로 아예 붙이지 않는다.
    if (!tensorBytes(tileValue.getType(), tile))
      return {};
    SmallVector<NamedAttribute> fields;
    if (auto load = dyn_cast<pim::LoadOp>(&op))
      if (auto id = load.getAccess().getDefiningOp()->getAttrOfType<IntegerAttr>(
              "pim.access_id"))
        fields.push_back(NamedAttribute(accessName, id));
    fields.push_back(NamedAttribute(bytesName, IntegerAttr::get(i64, tile)));
    fields.push_back(
        NamedAttribute(lifetimeName, StringAttr::get(ctx, "streaming")));
    entries.push_back(DictionaryAttr::get(ctx, fields));
  }
  if (auto loop = dyn_cast<scf::ForOp>(deepest->getParentOp()))
    for (BlockArgument arg : loop.getRegionIterArgs()) {
      int64_t acc;
      if (!tensorBytes(arg.getType(), acc))
        continue;
      entries.push_back(DictionaryAttr::get(
          ctx, {NamedAttribute(bytesName, IntegerAttr::get(i64, acc)),
                NamedAttribute(lifetimeName,
                               StringAttr::get(ctx, "persistent"))}));
    }
  if (entries.empty())
    return {};
  return ArrayAttr::get(ctx, entries);
}

// 계산 op 한 번 실행이 도는 반복 공간에서 원소 하나당 하는 산술 연산 개수.
// named op은 본문이 IR에 출력되지 않으므로 종류로 정한다.
static int64_t arithPerIteration(linalg::LinalgOp op) {
  if (isa<linalg::MatmulOp, linalg::MatvecOp>(op.getOperation()))
    return 2; // 곱해서 더한다
  Region &body = op.getOperation()->getRegion(0);
  if (body.empty())
    return 0;
  int64_t n = 0;
  for (Operation &inner : body.front())
    if (isa<arith::ArithDialect>(inner.getDialect()))
      n++;
  return n;
}

// instance 하나가 하는 산술 연산 횟수. 계산 op마다 (반복 공간 크기) × (원소당 산술 개수)
// × (감싸는 비-grid loop 반복 횟수)를 더한다. matmul tile [M,K]×[K,N]이면 반복 공간이
// M·N·K이고 원소당 둘이라 2·M·N·K가 된다. tile 크기와 loop 횟수만 쓰므로 target과
// 무관하다. 하나라도 못 세면 부분 합계는 오해를 부르므로 전체를 포기한다.
static bool computeOps(func::FuncOp func, int64_t &ops) {
  ops = 0;
  bool ok = true;
  func.walk([&](linalg::LinalgOp op) {
    int64_t iters = 1;
    for (int64_t r : op.getStaticLoopRanges()) {
      if (ShapedType::isDynamic(r)) {
        ok = false;
        return;
      }
      iters *= r;
    }
    int64_t repeats;
    if (!repeatCount(op.getOperation(), repeats)) {
      ok = false;
      return;
    }
    ops += iters * arithPerIteration(op) * repeats;
  });
  return ok;
}

// pim.grid_dim이 붙은 scf.for를 바깥에서 안쪽 순서로 모은다. walk 기본값은 post-order라
// 안쪽 loop부터 방문하므로 pre-order를 명시한다.
static void collectGridLoops(func::FuncOp func,
                             SmallVectorImpl<scf::ForOp> &grid) {
  func.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
    if (loop->hasAttr("pim.grid_dim"))
      grid.push_back(loop);
  });
}

struct PIMAnalyzeTransferPass
    : public mlir::triton::impl::PIMAnalyzeTransferBase<PIMAnalyzeTransferPass> {
  using PIMAnalyzeTransferBase::PIMAnalyzeTransferBase;

  void runOnOperation() override {
    getOperation().walk([&](func::FuncOp func) { analyzeFunc(func); });
  }

  void analyzeFunc(func::FuncOp func) {
    SmallVector<pim::AccessOp> accesses;
    func.walk([&](pim::AccessOp a) { accesses.push_back(a); });
    if (accesses.empty())
      return; // PIM Common 함수가 아니면 건드리지 않는다.

    MLIRContext *ctx = func.getContext();
    auto i64 = IntegerType::get(ctx, 64);

    // tile 바이트는 sizes와 element type만 쓰므로 언제나 붙는다. 전송 횟수는 감싸는
    // 비-grid loop 반복이 상수일 때만 붙는다.
    for (pim::AccessOp a : accesses) {
      int64_t tile;
      if (!tileBytes(a, tile))
        continue;
      a->setAttr("pim.tile_bytes", IntegerAttr::get(i64, tile));
      int64_t count;
      if (transferCount(a, count))
        a->setAttr("pim.transfer_count", IntegerAttr::get(i64, count));
    }

    if (ArrayAttr resident = coResident(func))
      func->setAttr("pim.co_resident", resident);

    int64_t ops;
    if (computeOps(func, ops))
      func->setAttr("pim.compute_ops", IntegerAttr::get(i64, ops));

    SmallVector<scf::ForOp> grid;
    collectGridLoops(func, grid);
    if (grid.empty())
      return; // grid 축이 없으면 grid 크기도 없다.

    SmallVector<int64_t> extents;
    for (scf::ForOp g : grid) {
      int64_t trips;
      // grid 크기가 runtime 값이면 붙이지 않는다. 근거 없는 수를 IR에 넣지 않는다.
      // 0이면 도는 instance가 없다.
      if (!constTripCount(g, trips) || trips <= 0)
        return;
      extents.push_back(trips);
    }
    func->setAttr("pim.grid_extents", DenseI64ArrayAttr::get(ctx, extents));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMAnalyzeTransferPass() {
  return std::make_unique<PIMAnalyzeTransferPass>();
}
