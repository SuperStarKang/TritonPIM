#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMRECOVERACCESSORIGIN
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// v가 어떤 loop 유도변수에 의존하는가. 좌표 쪽과 간격 쪽을 가르는 기준이다.
static bool dependsOnLoop(Value v) {
  SmallVector<Value> work{v};
  llvm::DenseSet<Value> seen;
  while (!work.empty()) {
    Value c = work.pop_back_val();
    if (!seen.insert(c).second)
      continue;
    if (auto ba = dyn_cast<BlockArgument>(c)) {
      if (isa<scf::ForOp>(ba.getOwner()->getParentOp()))
        return true;
      continue;
    }
    if (Operation *def = c.getDefiningOp())
      work.append(def->getOperands().begin(), def->getOperands().end());
  }
  return false;
}

static Value peelCast(Value v) {
  while (auto c = v.getDefiningOp<arith::IndexCastOp>())
    v = c.getIn();
  return v;
}

// v를 (기준 값, 정수 배수)로 정규화한다. index_cast와 상수 곱을 벗기므로 같은 간격을
// 다른 식으로 쓴 값들이 같은 기준 값으로 모인다. 순수 상수면 기준 값이 널이다.
static void canonicalStride(Value v, Value &base, int64_t &coeff) {
  coeff = 1;
  v = peelCast(v);
  while (true) {
    if (std::optional<int64_t> c = getConstantIntValue(v)) {
      coeff *= *c;
      base = Value();
      return;
    }
    auto mul = v.getDefiningOp<arith::MulIOp>();
    if (!mul)
      break;
    if (std::optional<int64_t> c = getConstantIntValue(mul.getRhs())) {
      coeff *= *c;
      v = peelCast(mul.getLhs());
      continue;
    }
    if (std::optional<int64_t> c = getConstantIntValue(mul.getLhs())) {
      coeff *= *c;
      v = peelCast(mul.getRhs());
      continue;
    }
    break;
  }
  base = v;
}

// offset의 addi 항들을 펼친다.
static SmallVector<Value> flattenTerms(Value offset) {
  SmallVector<Value> work{offset}, terms;
  while (!work.empty()) {
    Value t = work.pop_back_val();
    if (auto add = t.getDefiningOp<arith::AddIOp>()) {
      work.append({add.getLhs(), add.getRhs()});
      continue;
    }
    terms.push_back(t);
  }
  return terms;
}

static Value castToIndex(OpBuilder &b, Location loc, Value v) {
  return isa<IndexType>(v.getType())
             ? v
             : b.create<arith::IndexCastOp>(loc, b.getIndexType(), v)
                   .getResult();
}

struct PIMRecoverAccessOriginPass
    : public mlir::triton::impl::PIMRecoverAccessOriginBase<
          PIMRecoverAccessOriginPass> {
  using PIMRecoverAccessOriginBase::PIMRecoverAccessOriginBase;

  void runOnOperation() override {
    SmallVector<pim::AccessOp> accesses;
    getOperation().walk([&](pim::AccessOp a) {
      if (a.getOrigin().empty())
        accesses.push_back(a);
    });
    for (pim::AccessOp a : accesses)
      if (failed(recover(a)))
        return signalPassFailure();
  }

  LogicalResult recover(pim::AccessOp access) {
    OpBuilder b(access);
    Location loc = access.getLoc();
    unsigned rank = access.getSizes().size();

    // 축별 기준 간격. 항의 간격을 이것과 대조해 축을 정한다.
    SmallVector<Value> strideBase(rank);
    SmallVector<int64_t> strideCoeff(rank);
    for (unsigned d = 0; d < rank; d++)
      canonicalStride(access.getStrides()[d], strideBase[d], strideCoeff[d]);

    SmallVector<Value> origin(rank);
    // tile에는 없고 원본 버퍼에는 있는 축. 크기 1로 앞에 새로 만든다.
    SmallVector<Value> extraStride, extraOrigin;

    for (Value term : flattenTerms(access.getOffset())) {
      if (std::optional<int64_t> c = getConstantIntValue(term); c && *c == 0)
        continue;

      // 곱의 두 인자 중 유도변수에 의존하는 쪽이 좌표이고 나머지가 간격이다.
      Value index = peelCast(term), stride;
      if (auto mul = index.getDefiningOp<arith::MulIOp>()) {
        bool lhs = dependsOnLoop(mul.getLhs()), rhs = dependsOnLoop(mul.getRhs());
        if (lhs != rhs) {
          index = lhs ? mul.getLhs() : mul.getRhs();
          stride = lhs ? mul.getRhs() : mul.getLhs();
        }
      }
      Value base;
      int64_t coeff = 1;
      if (stride)
        canonicalStride(stride, base, coeff);

      unsigned axis = rank;
      for (unsigned d = 0; d < rank; d++)
        if (base == strideBase[d] && strideCoeff[d] != 0 &&
            coeff % strideCoeff[d] == 0) {
          axis = d;
          coeff /= strideCoeff[d];
          break;
        }

      Value contribution = castToIndex(b, loc, index);
      if (coeff != 1)
        contribution = b.create<arith::MulIOp>(
            loc, contribution, b.create<arith::ConstantIndexOp>(loc, coeff));

      if (axis == rank) {
        extraStride.push_back(castToIndex(b, loc, stride));
        extraOrigin.push_back(contribution);
        continue;
      }
      // 같은 축에 두 항이 오면 두 기여를 더한다.
      origin[axis] = origin[axis]
                         ? b.create<arith::AddIOp>(loc, origin[axis], contribution)
                               .getResult()
                         : contribution;
    }

    for (Value &o : origin)
      if (!o)
        o = b.create<arith::ConstantIndexOp>(loc, 0);
    if (extraStride.empty()) {
      access.getOriginMutable().assign(origin);
      return success();
    }
    return promote(b, access, extraStride, extraOrigin, origin);
  }

  // 새 축을 앞에 붙인 access로 바꾼다. split-K의 출력면처럼 원본 버퍼에는 있는데
  // tile에는 없는 축이 이 경우다.
  LogicalResult promote(OpBuilder &b, pim::AccessOp access,
                        ArrayRef<Value> extraStride, ArrayRef<Value> extraOrigin,
                        ArrayRef<Value> origin) {
    Location loc = access.getLoc();
    SmallVector<int64_t> sizes(extraStride.size(), 1);
    llvm::append_range(sizes, access.getSizes());
    SmallVector<Value> strides(extraStride);
    llvm::append_range(strides, access.getStrides());
    SmallVector<Value> newOrigin(extraOrigin);
    llvm::append_range(newOrigin, origin);
    SmallVector<Value> valid;
    if (!access.getValid().empty()) {
      Value one = b.create<arith::ConstantIndexOp>(loc, 1);
      valid.assign(extraStride.size(), one);
      llvm::append_range(valid, access.getValid());
    }
    auto fresh = b.create<pim::AccessOp>(
        loc, access.getResult().getType(), access.getBase(), access.getOffset(),
        newOrigin, strides, b.getDenseI64ArrayAttr(sizes), valid,
        access.getPad(), access.getIdAttr());
    access.getResult().replaceAllUsesWith(fresh.getResult());
    access.erase();
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMRecoverAccessOriginPass() {
  return std::make_unique<PIMRecoverAccessOriginPass>();
}
