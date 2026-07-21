#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMANALYZELOGICALWORK
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// triton-shared ABI: 커널 인자 끝에 6개가 붙는다 — num_programs(axis 0,1,2) 다음에
// program_id(axis 0,1,2). 즉 program ID는 맨 끝 3개 인자다(GetProgramIDConverter 참고).
static constexpr unsigned kLaunchGridRank = 3;

// axis 번호(0,1,2) → 이름. program_dims 속성에 쓴다(triton launch 축 그대로).
static StringRef axisName(unsigned axis) {
  static constexpr StringRef names[] = {"x", "y", "z"};
  return names[axis];
}

// func의 program-ID 인자를 axis→BlockArgument로 모은다. 끝 3개 중 스칼라 정수/index만
// 유효 axis로 본다(hand-written·비정형 시그니처에서 memref 등이 끝에 오면 그 axis는 없음).
static void collectProgramIdArgs(func::FuncOp func,
                                 SmallVectorImpl<BlockArgument> &pidByAxis) {
  pidByAxis.assign(kLaunchGridRank, BlockArgument());
  unsigned n = func.getNumArguments();
  if (n < kLaunchGridRank)
    return;
  for (unsigned axis = 0; axis < kLaunchGridRank; axis++) {
    BlockArgument arg = func.getArgument(n - kLaunchGridRank + axis);
    if (arg.getType().isIntOrIndex())
      pidByAxis[axis] = arg;
  }
}

// offset이 의존하는 program 축 집합을 backward slice로 구한다. slice가 불투명하면(추적 불가한
// leaf) 안전하게 모든 유효 axis에 의존한다고 보고 opaque=true를 남긴다(재사용 과대주장 금지).
static void collectOffsetDeps(Value root, ArrayRef<BlockArgument> pidByAxis,
                              llvm::DenseSet<unsigned> &deps, bool &opaque) {
  SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;

    if (auto ba = dyn_cast<BlockArgument>(v)) {
      // program-ID 인자면 해당 축을 기록.
      bool isPid = false;
      for (unsigned axis = 0; axis < pidByAxis.size(); axis++)
        if (pidByAxis[axis] == ba) {
          deps.insert(axis);
          isPid = true;
          break;
        }
      if (isPid)
        continue;
      // scf.for의 iter_arg/induction은 따라간다. iter_arg는 init과 yield 양쪽으로.
      if (auto forOp = dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp())) {
        unsigned argNo = ba.getArgNumber();
        if (argNo == 0)
          continue; // induction variable → loop-local, program 무관
        unsigned iterIdx = argNo - 1;
        worklist.push_back(forOp.getInitArgs()[iterIdx]);
        worklist.push_back(
            forOp.getBody()->getTerminator()->getOperand(iterIdx));
        continue;
      }
      // 그 외 func 인자(stride/size 등, 인스턴스 불변)는 안전한 leaf, 나머지 block arg는 불투명.
      if (isa<func::FuncOp>(ba.getOwner()->getParentOp()))
        continue;
      opaque = true;
      continue;
    }

    Operation *def = v.getDefiningOp();
    if (!def) {
      opaque = true;
      continue;
    }
    if (isa<arith::ConstantOp>(def))
      continue; // 상수 leaf
    for (Value o : def->getOperands())
      worklist.push_back(o);
  }
}

// access를 소비하는 방식으로 role을 정한다. store/copy-dst → output, 그 외(load/copy-src) → input.
// load/store/copy 어느 것도 안 쓰면 "".
static StringRef accessRole(Value access) {
  bool used = false;
  for (Operation *u : access.getUsers()) {
    if (isa<pim::StoreOp>(u))
      return "output";
    if (auto copy = dyn_cast<pim::CopyOp>(u)) {
      if (copy.getDst() == access)
        return "output";
      used = true; // src
      continue;
    }
    if (isa<pim::LoadOp>(u))
      used = true;
  }
  return used ? "input" : StringRef();
}

// 결합 arith op → target-무관 combine 이름. 판별 불가면 "". op 이름으로 봐 dialect 버전차를 피한다.
static StringRef arithToCombine(Operation *op) {
  StringRef n = op->getName().getStringRef();
  if (n == "arith.addf" || n == "arith.addi")
    return "add";
  if (n == "arith.mulf" || n == "arith.muli")
    return "mul";
  if (n == "arith.maximumf" || n == "arith.maxnumf" || n == "arith.maxsi" ||
      n == "arith.maxui")
    return "max";
  if (n == "arith.minimumf" || n == "arith.minnumf" || n == "arith.minsi" ||
      n == "arith.minui")
    return "min";
  return "";
}

// 누적 계산 op의 결합 연산자. matmul/matvec는 add(곱을 더해 누적). reduce/generic은
// body의 yield 값을 만든 arith op로 판정. 판별 불가면 "".
static StringRef reductionCombine(Operation *compute) {
  if (isa<linalg::MatmulOp, linalg::MatvecOp>(compute))
    return "add";
  if (!isa<linalg::ReduceOp, linalg::GenericOp>(compute))
    return "";
  Region &r = compute->getRegion(0);
  if (r.empty())
    return "";
  auto yield = dyn_cast<linalg::YieldOp>(r.front().getTerminator());
  if (!yield || yield->getNumOperands() != 1)
    return "";
  Operation *def = yield->getOperand(0).getDefiningOp();
  return def ? arithToCombine(def) : StringRef();
}

struct PIMAnalyzeLogicalWorkPass
    : public mlir::triton::impl::PIMAnalyzeLogicalWorkBase<
          PIMAnalyzeLogicalWorkPass> {
  using PIMAnalyzeLogicalWorkBase::PIMAnalyzeLogicalWorkBase;

  void runOnOperation() override {
    getOperation().walk([&](func::FuncOp func) { analyzeFunc(func); });
  }

  void analyzeFunc(func::FuncOp func) {
    SmallVector<pim::AccessOp> accesses;
    func.walk([&](pim::AccessOp a) { accesses.push_back(a); });
    if (accesses.empty())
      return; // PIM Common 함수가 아니면 건드리지 않는다.

    MLIRContext *ctx = func.getContext();
    func->setAttr("pim.execution_model", StringAttr::get(ctx, "spmd"));

    SmallVector<BlockArgument> pidByAxis;
    collectProgramIdArgs(func, pidByAxis);

    // 각 access의 program-축 의존 집합을 구하고, 실제로 쓰인 축(active)을 모은다.
    SmallVector<llvm::DenseSet<unsigned>> depsPerAccess(accesses.size());
    llvm::DenseSet<unsigned> activeAxes;
    for (auto [i, a] : llvm::enumerate(accesses)) {
      bool opaque = false;
      collectOffsetDeps(a.getOffset(), pidByAxis, depsPerAccess[i], opaque);
      if (opaque)
        for (unsigned axis = 0; axis < pidByAxis.size(); axis++)
          if (pidByAxis[axis])
            depsPerAccess[i].insert(axis);
      for (unsigned axis : depsPerAccess[i])
        activeAxes.insert(axis);
    }

    // role은 program 축과 무관하게 항상 단다.
    for (pim::AccessOp a : accesses)
      if (StringRef role = accessRole(a.getResult()); !role.empty())
        a->setAttr("pim.role", StringAttr::get(ctx, role));

    // ⑥ reduction loop: tensor iter_arg 누산기를 계산 op의 outs로 받아 누적하는 scf.for에
    // combine 연산자를 단다(target이 split 시 부분합을 이 연산으로 결합). program 축과 무관.
    func.walk([&](scf::ForOp forOp) {
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        if (!isa<RankedTensorType>(arg.getType()))
          continue;
        Operation *compute = yield.getOperand(i).getDefiningOp();
        auto linalgOp = compute ? dyn_cast<linalg::LinalgOp>(compute) : nullptr;
        if (!linalgOp || !llvm::is_contained(linalgOp.getDpsInits(), arg))
          continue; // 누산기(iter_arg)를 outs로 받는 계산의 결과가 yield 되어야 한다.
        if (StringRef c = reductionCombine(compute); !c.empty()) {
          forOp->setAttr("pim.reduction_combine", StringAttr::get(ctx, c));
          break;
        }
      }
    });

    if (activeAxes.empty())
      return; // program ID에 의존하는 접근이 없으면 여기까지(예: loop-carried offset).

    // active 축을 axis 순서로 정렬해 program_dims 순서를 정한다. invariant 인덱스는 이 순서 기준.
    SmallVector<unsigned> orderedAxes(activeAxes.begin(), activeAxes.end());
    llvm::sort(orderedAxes);

    SmallVector<Attribute> dimNames;
    for (unsigned axis : orderedAxes)
      dimNames.push_back(StringAttr::get(ctx, axisName(axis)));
    func->setAttr("pim.program_dims", ArrayAttr::get(ctx, dimNames));

    llvm::DenseSet<unsigned> reduceAcross; // ⑦ 출력이 불변인 축 = 부분합 결합 필요 축
    for (auto [i, a] : llvm::enumerate(accesses)) {
      SmallVector<int64_t> invariant;
      for (auto [pos, axis] : llvm::enumerate(orderedAxes))
        if (!depsPerAccess[i].contains(axis))
          invariant.push_back(pos);
      a->setAttr("pim.invariant_program_dims",
                 DenseI64ArrayAttr::get(ctx, invariant));
      // ⑦ 출력 접근이 어떤 program 축에 불변이면, 그 축만 다른 program들이 같은 출력에 씀
      // → 부분합을 합쳐야 한다(Split-K류). 그 축들을 모은다.
      if (accessRole(a.getResult()) == "output")
        for (unsigned axis : orderedAxes)
          if (!depsPerAccess[i].contains(axis))
            reduceAcross.insert(axis);
    }

    if (!reduceAcross.empty()) {
      SmallVector<unsigned> ra(reduceAcross.begin(), reduceAcross.end());
      llvm::sort(ra);
      SmallVector<Attribute> raNames;
      for (unsigned axis : ra)
        raNames.push_back(StringAttr::get(ctx, axisName(axis)));
      func->setAttr("pim.reduce_across_programs", ArrayAttr::get(ctx, raNames));
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMAnalyzeLogicalWorkPass() {
  return std::make_unique<PIMAnalyzeLogicalWorkPass>();
}
