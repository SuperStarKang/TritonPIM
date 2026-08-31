#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMBUILDLOGICALGRID
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// triton-shared ABI: 커널 인자 끝 6개 = num_programs(축 0,1,2) 다음 program_id(축 0,1,2).
static constexpr unsigned kGridRank = 3;

// Source program grid는 서로 독립적인 output tile 좌표만 표현한다. Reduction K는
// PIMFormCompute가 linalg accumulator 의미에서 찾은 semantic reduction loop이며,
// source program_id로 미리 만든 split-K plane이 아니다.
static constexpr StringRef kAxisNames[] = {"m", "n"};

// 복원된 logical 축 하나. 축의 tile 개수는 extent 이거나, divideBy 가 있으면
// ceildiv(extent, divideBy) 다. 둘 다 func 인자와 상수만으로 계산된다.
struct LogicalAxis {
  Value coordinate; // 이 축의 tile 번호를 내던 값. loop 유도변수로 바뀐다.
  Value extent;
  Value divideBy;
};

// 타입이 다를 때만 index_cast를 넣는다.
static Value castTo(OpBuilder &b, Location loc, Value v, Type t) {
  return v.getType() == t ? v : b.create<arith::IndexCastOp>(loc, t, v);
}

// v를 계산하는 식을 삽입 지점에 다시 만든다. 잎이 entry block 인자나 상수라야 한다 —
// 그래야 그 식이 grid 전체를 지배하고 loop 상한으로 쓸 수 있다.
static Value cloneInvariant(OpBuilder &b, Value v, Block &entry) {
  if (auto ba = dyn_cast<BlockArgument>(v))
    return ba.getOwner() == &entry ? v : Value();
  Operation *def = v.getDefiningOp();
  if (!def || def->getNumResults() != 1 || !isMemoryEffectFree(def))
    return Value();
  IRMapping map;
  for (Value o : def->getOperands()) {
    Value r = cloneInvariant(b, o, entry);
    if (!r)
      return Value();
    map.map(o, r);
  }
  return b.clone(*def, map)->getResult(0);
}

// v를 피연산자 idx 자리에 쓰는 OpTy 사용처.
template <typename OpTy>
static OpTy userAt(Value v, unsigned idx) {
  for (Operation *u : v.getUsers())
    if (auto op = dyn_cast<OpTy>(u))
      if (op->getOperand(idx) == v)
        return op;
  return OpTy();
}

// binary op의 두 피연산자 중 other가 아닌 쪽.
static Value otherOperand(Operation *op, Value one) {
  return op->getOperand(0) == one ? op->getOperand(1) : op->getOperand(0);
}

// GROUP_M 순회 순서를 (m, n)로 되돌린다.
//   tiles_per_group  = column_tiles * G
//   group            = pid / tiles_per_group
//   first_row        = group * G
//   rows_in_group    = min(row_tiles - first_row, G)
//   within           = pid % tiles_per_group
//   row_tile         = first_row + within % rows_in_group
//   column_tile      = within / rows_in_group
// 순서는 배치에 영향이 없으므로 이 사슬은 통째로 죽는다.
static bool matchGroupOrder(Value pid, SmallVectorImpl<LogicalAxis> &axes) {
  auto within = userAt<arith::RemSIOp>(pid, 0);
  auto group = userAt<arith::DivSIOp>(pid, 0);
  if (!within || !group || within.getRhs() != group.getRhs())
    return false;
  auto tilesPerGroup = group.getRhs().getDefiningOp<arith::MulIOp>();
  auto firstRow = userAt<arith::MulIOp>(group, 0);
  if (!tilesPerGroup || !firstRow ||
      !llvm::is_contained(tilesPerGroup->getOperands(), firstRow.getRhs()))
    return false;

  auto columnTile = userAt<arith::DivSIOp>(within, 0);
  auto rowRem = userAt<arith::RemSIOp>(within, 0);
  if (!columnTile || !rowRem || columnTile.getRhs() != rowRem.getRhs())
    return false;
  auto rowsInGroup = columnTile.getRhs().getDefiningOp<arith::MinSIOp>();
  if (!rowsInGroup)
    return false;

  // rows_in_group = min(row_tiles - first_row, G) 에서 row_tiles를 꺼낸다.
  auto remaining = rowsInGroup.getLhs().getDefiningOp<arith::SubIOp>();
  if (!remaining)
    remaining = rowsInGroup.getRhs().getDefiningOp<arith::SubIOp>();
  if (!remaining || remaining.getRhs() != firstRow.getResult())
    return false;

  Operation *rowTile = nullptr;
  for (Operation *u : rowRem->getUsers())
    if (isa<arith::AddIOp>(u) && otherOperand(u, rowRem) == firstRow.getResult())
      rowTile = u;
  if (!rowTile)
    return false;

  axes.push_back({rowTile->getResult(0), remaining.getLhs(), Value()});
  axes.push_back({columnTile.getResult(),
                  otherOperand(tilesPerGroup, firstRow.getRhs()), Value()});
  return true;
}

// pid / d 와 pid % d 쌍을 두 logical 축으로 가른다. 두 나눗셈의 제수가 같은 SSA 값인
// 것이 이 쌍의 판정이다. 행 tile 수는 IR에 없으므로 ceildiv(grid, d)로 유도한다 —
// launcher가 grid == row_tiles * column_tiles 를 보장한다.
static bool matchDivRem(Value pid, Value numPrograms,
                        SmallVectorImpl<LogicalAxis> &axes) {
  auto rowTile = userAt<arith::DivSIOp>(pid, 0);
  auto columnTile = userAt<arith::RemSIOp>(pid, 0);
  if (!rowTile || !columnTile || rowTile.getRhs() != columnTile.getRhs())
    return false;
  Value divisor = rowTile.getRhs();
  axes.push_back({rowTile.getResult(), numPrograms, divisor});
  axes.push_back({columnTile.getResult(), divisor, Value()});
  return true;
}

struct PIMBuildLogicalGridPass
    : public mlir::triton::impl::PIMBuildLogicalGridBase<
          PIMBuildLogicalGridPass> {
  using PIMBuildLogicalGridBase::PIMBuildLogicalGridBase;

  void runOnOperation() override {
    WalkResult res = getOperation().walk([&](func::FuncOp func) {
      return succeeded(buildGrid(func)) ? WalkResult::advance()
                                        : WalkResult::interrupt();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }

  LogicalResult buildGrid(func::FuncOp func) {
    // 단일 블록만 다룬다. early return이 여러 개면 옮길 몸체 범위를 하나로 못 자른다.
    unsigned n = func.getNumArguments();
    if (!func.getBody().hasOneBlock() || n < 2 * kGridRank)
      return success();

    SmallVector<BlockArgument, kGridRank> numPrograms, programId;
    for (unsigned axis = 0; axis < kGridRank; axis++) {
      numPrograms.push_back(func.getArgument(n - 2 * kGridRank + axis));
      programId.push_back(func.getArgument(n - kGridRank + axis));
    }
    // 끝 6개가 전부 스칼라 정수여야 이 ABI다.
    for (unsigned axis = 0; axis < kGridRank; axis++)
      if (!numPrograms[axis].getType().isIntOrIndex() ||
          !programId[axis].getType().isIntOrIndex())
        return success();
    if (llvm::all_of(programId, [](Value v) { return v.use_empty(); }))
      return success();

    Block &entry = func.getBody().front();
    Location loc = func.getLoc();
    OpBuilder b(&entry, entry.begin());

    // source grid 축을 logical 축으로 푼다. 첫 축만 갈라질 수 있고, 나머지는 그대로다.
    SmallVector<LogicalAxis> axes;
    for (unsigned axis = 0; axis < kGridRank; axis++) {
      if (programId[axis].use_empty())
        continue;
      if (axes.empty() &&
          (matchGroupOrder(programId[axis], axes) ||
           matchDivRem(programId[axis], numPrograms[axis], axes)))
        continue;
      axes.push_back({programId[axis], numPrograms[axis], Value()});
    }
    if (axes.size() > 2)
      return func->emitError(
          "pim-build-logical-grid: source-level split-K is not a Common IR "
          "logical axis; use one semantic reduction loop and let "
          "pim-map-devices partition it with p_k");

    // 상한을 먼저 전부 만든다. loop를 쌓으면서 만들면 안쪽 loop의 상한이 바깥 loop 안에
    // 들어가 loop-invariant인 채로 남는다.
    SmallVector<Value> upperBounds;
    for (const LogicalAxis &a : axes) {
      Value extent = cloneInvariant(b, a.extent, entry);
      Value divisor = a.divideBy ? cloneInvariant(b, a.divideBy, entry) : Value();
      if (!extent || (a.divideBy && !divisor))
        return func->emitError("pim-build-logical-grid: the tile count of axis ")
               << kAxisNames[upperBounds.size()]
               << " is not a function of the kernel arguments alone";
      if (divisor)
        extent = b.create<arith::CeilDivSIOp>(
            loc, castTo(b, loc, extent, divisor.getType()), divisor);
      upperBounds.push_back(castTo(b, loc, extent, b.getIndexType()));
    }

    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<scf::ForOp> loops;
    for (auto [i, ub] : llvm::enumerate(upperBounds)) {
      auto forOp = b.create<scf::ForOp>(loc, c0, ub, c1);
      forOp->setAttr(
          "pim.axis",
          pim::AxisAttr::get(func.getContext(), b.getStringAttr(kAxisNames[i])));
      loops.push_back(forOp);
      b.setInsertionPointToStart(forOp.getBody());
    }

    // 원래 몸체(바깥 loop 다음 ~ return 앞)를 가장 안쪽 loop로 옮긴다.
    Block *inner = loops.back().getBody();
    inner->getOperations().splice(
        inner->begin(), entry.getOperations(),
        std::next(Block::iterator(loops.front().getOperation())),
        Block::iterator(entry.getTerminator()));

    // tile 좌표를 유도변수로 바꾼다. 몸체 맨 앞에 두면 몸체 전체를 지배한다.
    b.setInsertionPointToStart(inner);
    for (auto [i, a] : llvm::enumerate(axes))
      a.coordinate.replaceAllUsesWith(
          castTo(b, loc, loops[i].getInductionVar(), a.coordinate.getType()));

    // 좌표를 풀던 사슬은 이제 아무도 쓰지 않는다. 지워야 program_id 인자가 자유로워진다.
    eraseDeadPure(func);

    // program_id 인자를 시그니처에서 지운다. 남겨 두면 인자 끝 3개를 program ID로 보는
    // 뒤 패스가 num_programs를 program ID로 오인한다.
    llvm::BitVector toErase(n);
    for (unsigned axis = 0; axis < kGridRank; axis++)
      toErase.set(n - kGridRank + axis);
    if (failed(func.eraseArguments(toErase)))
      return func.emitError("pim-build-logical-grid: program_id arguments are "
                            "still used after grid construction");
    return success();
  }

  void eraseDeadPure(func::FuncOp func) {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> dead;
      func.walk([&](Operation *op) {
        if (op->getNumRegions() == 0 && !op->hasTrait<OpTrait::IsTerminator>() &&
            op->use_empty() && isMemoryEffectFree(op))
          dead.push_back(op);
      });
      for (Operation *op : dead) {
        op->erase();
        changed = true;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMBuildLogicalGridPass() {
  return std::make_unique<PIMBuildLogicalGridPass>();
}
