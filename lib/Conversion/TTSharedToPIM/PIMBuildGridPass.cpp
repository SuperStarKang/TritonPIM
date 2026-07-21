#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMBUILDGRID
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// triton-shared ABI: 커널 인자 끝 6개 = num_programs(axis 0,1,2) 다음 program_id(axis 0,1,2).
// TritonArithToLinalg의 GetProgramIDConverter가 이 순서로 붙인다.
static constexpr unsigned kGridRank = 3;

// axis 번호(0,1,2) → 이름. grid_dim 속성에 쓴다(triton launch 축 그대로).
static StringRef axisName(unsigned axis) {
  static constexpr StringRef names[] = {"x", "y", "z"};
  return names[axis];
}

// 타입이 다를 때만 index_cast를 넣는다(i32 ABI 인자 ↔ index 유도변수).
static Value castTo(OpBuilder &b, Location loc, Value v, Type t) {
  return v.getType() == t ? v : b.create<arith::IndexCastOp>(loc, t, v);
}

struct PIMBuildGridPass
    : public mlir::triton::impl::PIMBuildGridBase<PIMBuildGridPass> {
  using PIMBuildGridBase::PIMBuildGridBase;

  void runOnOperation() override {
    getOperation().walk([&](func::FuncOp func) { buildGrid(func); });
  }

  void buildGrid(func::FuncOp func) {
    // 단일 블록만 다룬다. early return이 여러 개면 옮길 몸체 범위를 하나로 못 자른다.
    // 선언(블록 0개)도 여기서 걸린다.
    if (!func.getBody().hasOneBlock())
      return;

    unsigned n = func.getNumArguments();
    if (n < 2 * kGridRank)
      return;

    SmallVector<BlockArgument, kGridRank> numPrograms, programId;
    for (unsigned axis = 0; axis < kGridRank; axis++) {
      numPrograms.push_back(func.getArgument(n - 2 * kGridRank + axis));
      programId.push_back(func.getArgument(n - kGridRank + axis));
    }
    // 끝 6개가 전부 스칼라 정수여야 이 ABI다. 아니면 손으로 쓴 함수이므로 건드리지 않는다.
    for (unsigned axis = 0; axis < kGridRank; axis++)
      if (!numPrograms[axis].getType().isIntOrIndex() ||
          !programId[axis].getType().isIntOrIndex())
        return;

    // 실제로 쓰이는 축만 loop로 만든다. 안 쓰는 축은 범위가 1이라 loop가 필요 없다.
    SmallVector<unsigned, kGridRank> usedAxes;
    for (unsigned axis = 0; axis < kGridRank; axis++)
      if (!programId[axis].use_empty())
        usedAxes.push_back(axis);
    if (usedAxes.empty())
      return;

    Block &entry = func.getBody().front();
    Operation *terminator = entry.getTerminator();
    Location loc = func.getLoc();
    OpBuilder b(&entry, entry.begin());

    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);

    // 상한을 먼저 전부 만든다. loop를 쌓으면서 만들면 안쪽 loop의 상한 cast가
    // 바깥 loop 안에 들어가 loop-invariant인 채로 남는다.
    SmallVector<Value, kGridRank> upperBounds;
    for (unsigned axis : usedAxes)
      upperBounds.push_back(
          castTo(b, loc, numPrograms[axis], b.getIndexType()));

    // 축 순서대로 바깥에서 안으로 grid loop를 쌓는다.
    SmallVector<scf::ForOp, kGridRank> loops;
    for (auto [i, axis] : llvm::enumerate(usedAxes)) {
      auto forOp = b.create<scf::ForOp>(loc, c0, upperBounds[i], c1);
      forOp->setAttr("pim.grid_dim", b.getStringAttr(axisName(axis)));
      loops.push_back(forOp);
      b.setInsertionPointToStart(forOp.getBody());
    }

    // 원래 몸체(바깥 loop 다음 ~ return 앞)를 가장 안쪽 loop로 옮긴다.
    // 목적지 begin()은 scf.yield 자리라, 그 앞에 들어간다.
    Block *inner = loops.back().getBody();
    inner->getOperations().splice(
        inner->begin(), entry.getOperations(),
        std::next(Block::iterator(loops.front().getOperation())),
        Block::iterator(terminator));

    // 유도변수를 원래 program_id 자리에 넣는다. 몸체 맨 앞에 두면 몸체 전체를 지배한다.
    b.setInsertionPointToStart(inner);
    for (auto [i, axis] : llvm::enumerate(usedAxes)) {
      Value pid = castTo(b, loc, loops[i].getInductionVar(),
                         programId[axis].getType());
      programId[axis].replaceAllUsesWith(pid);
    }

    // program_id 인자는 이제 아무도 쓰지 않으므로 시그니처에서 지운다. 남겨 두면 인자
    // 끝 3개를 program ID로 보는 뒤 패스가 num_programs를 program ID로 오인한다.
    // num_programs는 loop 상한으로 쓰이므로 남긴다.
    llvm::BitVector toErase(n);
    for (unsigned axis = 0; axis < kGridRank; axis++)
      toErase.set(n - kGridRank + axis);
    if (failed(func.eraseArguments(toErase))) {
      func.emitError("failed to erase program_id arguments");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMBuildGridPass() {
  return std::make_unique<PIMBuildGridPass>();
}
