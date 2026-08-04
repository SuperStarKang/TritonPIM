#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMNORMALIZEADDRESS
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// v 가 주소 산술만 거쳐 reinterpret_cast 의 offset 에 닿는가. 주소와 무관한 index
// iter_arg 까지 잡지 않도록 범위를 이 판정으로 한정한다.
static bool carriesAddress(Value v) {
  SmallVector<Value> work{v};
  llvm::DenseSet<Value> seen;
  while (!work.empty()) {
    Value c = work.pop_back_val();
    if (!seen.insert(c).second)
      continue;
    for (Operation *user : c.getUsers()) {
      if (auto rc = dyn_cast<memref::ReinterpretCastOp>(user)) {
        for (OpFoldResult o : rc.getMixedOffsets())
          if (dyn_cast<Value>(o) == c)
            return true;
        continue;
      }
      if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp, arith::IndexCastOp>(
              user))
        work.push_back(user->getResult(0));
    }
  }
  return false;
}

// v 가 loop 밖에서 정의되었는가.
static bool definedOutside(scf::ForOp loop, Value v) {
  if (Operation *def = v.getDefiningOp())
    return !loop->isAncestor(def);
  return !loop->isAncestor(v.getParentBlock()->getParentOp());
}

struct PIMNormalizeAddressPass
    : public mlir::triton::impl::PIMNormalizeAddressBase<
          PIMNormalizeAddressPass> {
  using PIMNormalizeAddressBase::PIMNormalizeAddressBase;

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp l) { loops.push_back(l); });
    for (scf::ForOp l : loops)
      if (failed(normalizeLoop(l)))
        return signalPassFailure();
  }

  // 주소를 나르는 iter_arg 를 offset(i) = initial + i*advance 로 다시 쓴다.
  LogicalResult normalizeLoop(scf::ForOp loop) {
    Value step = loop.getStep();
    auto stepConst = getConstantIntValue(step);
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    Value iv = loop.getInductionVar();

    for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
      if (!carriesAddress(arg))
        continue;
      auto add = yield.getOperand(i).getDefiningOp<arith::AddIOp>();
      Value advance;
      if (add) {
        if (add.getLhs() == arg)
          advance = add.getRhs();
        else if (add.getRhs() == arg)
          advance = add.getLhs();
      }
      if (!advance || !definedOutside(loop, advance) ||
          !arg.getType().isIntOrIndex() || (stepConst && *stepConst == 0))
        return emitClosedFormError(loop, i);

      OpBuilder b(loop.getBody(), loop.getBody()->begin());
      Location loc = loop.getLoc();
      Value iteration = b.create<arith::DivSIOp>(
          loc, b.create<arith::SubIOp>(loc, iv, loop.getLowerBound()), step);
      if (arg.getType() != iv.getType())
        iteration =
            b.create<arith::IndexCastOp>(loc, arg.getType(), iteration);
      // yield 의 addi 도 함께 바뀌어 다음 반복 값이 initial + (i+1)*advance 가 되므로
      // loop 의미가 그대로 보존된다.
      arg.replaceAllUsesWith(b.create<arith::AddIOp>(
          loc, loop.getInitArgs()[i],
          b.create<arith::MulIOp>(loc, iteration, advance)));
    }
    return success();
  }

  LogicalResult emitClosedFormError(scf::ForOp loop, unsigned index) {
    InFlightDiagnostic err =
        loop->emitError("pim-normalize-address: address recurrence carried by "
                        "iter_arg #")
        << index
        << " cannot be put in closed form: the yielded value is not "
           "addi(<iter_arg>, advance) with advance defined outside the loop";
    err.attachNote() << "PIM placement needs offset(i) = initial + i*advance to "
                        "recover the per-core region of this buffer";
    err.attachNote() << "index with an affine expression of the loop variable "
                        "(a_ptr + (k*BLOCK_K)*stride_ak) instead of "
                        "accumulating the pointer (a_ptrs += BLOCK_K*stride_ak)";
    return failure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMNormalizeAddressPass() {
  return std::make_unique<PIMNormalizeAddressPass>();
}
