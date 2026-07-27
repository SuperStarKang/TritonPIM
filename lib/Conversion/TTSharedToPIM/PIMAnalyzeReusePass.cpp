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
#define GEN_PASS_DEF_PIMANALYZEREUSE
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// pim.grid_dim이 붙은 scf.for를 바깥에서 안쪽 순서로 모은다. 이것이 program 차원이다.
// walk 기본값은 post-order라 안쪽 loop부터 방문하므로 pre-order를 명시한다.
static void collectGridLoops(func::FuncOp func,
                             SmallVectorImpl<scf::ForOp> &grid) {
  func.walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
    if (forOp->hasAttr("pim.grid_dim"))
      grid.push_back(forOp);
  });
}

// 값이 의존하는 grid 차원 집합을 backward slice로 구한다. 추적할 수 없는 leaf를 만나면
// opaque=true를 남겨, 호출자가 모든 차원에 의존한다고 보수적으로 처리하게 한다
// (재사용 가능성을 과대주장하지 않는다).
// grid가 아닌 loop의 유도변수에 의존하면 inner=true를 남긴다. 그러면 pim.access가 그
// loop 안에서 만들어져 전송을 밖으로 뺄 수 없고(§15.6의 첫 조건), 불변 축이 약속하는
// 재사용이 loop 순서를 바꾸지 않는 한 실현되지 않는다.
static void collectDeps(Value root, ArrayRef<scf::ForOp> grid,
                        llvm::DenseSet<unsigned> &deps, bool &opaque,
                        bool &inner) {
  SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;

    if (auto ba = dyn_cast<BlockArgument>(v)) {
      auto forOp = dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp());
      if (!forOp) {
        // func 인자(stride/size 등)는 instance 사이에서 값이 같은 안전한 leaf.
        // 다른 region의 block arg는 추적할 수 없다.
        if (!isa<func::FuncOp>(ba.getOwner()->getParentOp()))
          opaque = true;
        continue;
      }
      // grid가 아닌 loop 안에서 변하는 값이면(유도변수든 iter_arg든) 그 loop마다 주소가
      // 달라져 pim.access를 loop 밖으로 뺄 수 없다.
      bool isGrid = llvm::is_contained(grid, forOp);
      if (!isGrid)
        inner = true;
      if (ba == forOp.getInductionVar()) {
        // grid loop의 유도변수면 그 차원에 의존한다. grid가 아닌 loop(K tile 등)의
        // 유도변수는 instance 내부의 반복이라 grid 차원과 무관하다.
        if (isGrid)
          for (auto [d, g] : llvm::enumerate(grid))
            if (g == forOp) {
              deps.insert(d);
              break;
            }
        continue;
      }
      // iter_arg는 초기값과 yield 값 양쪽으로 따라간다.
      unsigned iterIdx = ba.getArgNumber() - 1;
      worklist.push_back(forOp.getInitArgs()[iterIdx]);
      worklist.push_back(forOp.getBody()->getTerminator()->getOperand(iterIdx));
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

// access가 읽거나 쓰는 데이터가 의존하는 grid 차원 집합.
// offset만으로는 부족하다. strides는 축 간격을, valid는 경계 tile의 유효 범위를,
// index는 row gather가 실제로 읽는 행을 정한다. pad는 주소는 아니지만 유효 범위 밖
// 원소의 값이 되어 load 결과를 바꾸므로 함께 본다.
static void collectAccessDeps(pim::AccessOp access, ArrayRef<scf::ForOp> grid,
                              llvm::DenseSet<unsigned> &deps, bool &opaque,
                              bool &inner) {
  collectDeps(access.getOffset(), grid, deps, opaque, inner);
  for (Value stride : access.getStrides())
    collectDeps(stride, grid, deps, opaque, inner);
  for (Value valid : access.getValid())
    collectDeps(valid, grid, deps, opaque, inner);
  if (Value pad = access.getPad())
    collectDeps(pad, grid, deps, opaque, inner);
  if (Value index = access.getIndex())
    collectDeps(index, grid, deps, opaque, inner);
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

struct PIMAnalyzeReusePass
    : public mlir::triton::impl::PIMAnalyzeReuseBase<PIMAnalyzeReusePass> {
  using PIMAnalyzeReuseBase::PIMAnalyzeReuseBase;

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

    SmallVector<scf::ForOp> grid;
    collectGridLoops(func, grid);

    // 각 access의 grid 차원 의존 집합을 구하고, 어떤 access든 실제로 의존하는 차원을 모은다.
    SmallVector<llvm::DenseSet<unsigned>> depsPerAccess(accesses.size());
    SmallVector<bool> innerDep(accesses.size(), false);
    llvm::DenseSet<unsigned> activeDims;
    for (auto [i, a] : llvm::enumerate(accesses)) {
      bool opaque = false;
      bool inner = false;
      collectAccessDeps(a, grid, depsPerAccess[i], opaque, inner);
      if (opaque)
        for (unsigned d = 0; d < grid.size(); d++)
          depsPerAccess[i].insert(d);
      innerDep[i] = inner;
      for (unsigned d : depsPerAccess[i])
        activeDims.insert(d);
    }

    // role과 재사용 실현 가능성은 grid 차원과 무관하게 항상 단다.
    for (auto [i, a] : llvm::enumerate(accesses)) {
      if (StringRef role = accessRole(a.getResult()); !role.empty())
        a->setAttr("pim.role", StringAttr::get(ctx, role));
      a->setAttr("pim.reuse_realizable", BoolAttr::get(ctx, !innerDep[i]));
    }

    // reduction loop: tensor iter_arg 누산기를 계산 op의 outs로 받아 누적하는 scf.for에
    // combine 연산자를 단다(유닛에 나눌 때 부분 결과를 이 연산으로 합친다). grid 축과 무관.
    SmallVector<scf::ForOp> reductionLoops;
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
          reductionLoops.push_back(forOp);
          break;
        }
      }
    });

    if (grid.empty())
      return; // grid loop가 없으면 grid 차원도 없다(예: loop-carried offset만 있는 함수).

    // grid_dims는 grid loop의 중첩 순서 그대로다. invariant 인덱스도 이 순서를 가리킨다.
    SmallVector<Attribute> dimNames;
    for (scf::ForOp g : grid)
      dimNames.push_back(g->getAttr("pim.grid_dim"));
    func->setAttr("pim.grid_dims", ArrayAttr::get(ctx, dimNames));

    llvm::DenseSet<unsigned> needCombine;
    for (auto [i, a] : llvm::enumerate(accesses)) {
      bool isOutput = accessRole(a.getResult()) == "output";
      SmallVector<int64_t> invariant;
      for (unsigned d = 0; d < grid.size(); d++) {
        if (depsPerAccess[i].contains(d))
          continue;
        invariant.push_back(d);
        // 출력이 차원 d에 불변이면 d만 다른 instance들이 같은 출력에 쓰므로 부분 결과를
        // 합쳐야 한다. 단 어떤 access도 d에 의존하지 않으면 모든 instance가 같은 값을
        // 만드는 중복 실행이라, 합치면 오히려 결과가 틀린다. 활성 차원일 때만 표시한다.
        if (isOutput && activeDims.contains(d))
          needCombine.insert(d);
      }
      a->setAttr("pim.invariant_grid_dims",
                 DenseI64ArrayAttr::get(ctx, invariant));
    }

    func->setAttr("pim.split_axes",
                  buildSplitAxes(func, ctx, grid, reductionLoops, needCombine));
  }

  // 쪼갤 수 있는 축을 모은다. grid 축과 reduction loop 둘 다 후보다.
  // - pim.scatter가 있으면 인덱스 텐서의 값을 컴파일 시점에 몰라 서로 다른 유닛이 같은
  //   원소에 쓰지 않는다고 보장할 수 없으므로 grid 축을 올리지 않는다.
  // - 쪼갤 때 부분 결과가 생기는 축은 combine 연산자를 함께 적는다. 연산자를 모르면 그
  //   축을 올리지 않는다 — 모르는 채 쪼개면 target이 덮어쓰는 코드를 만든다.
  // - reduction loop는 몸체에 pim.store/pim.scatter가 있으면 반복마다 부수 효과가
  //   있다는 뜻이라 올리지 않는다.
  ArrayAttr buildSplitAxes(func::FuncOp func, MLIRContext *ctx,
                           ArrayRef<scf::ForOp> grid,
                           ArrayRef<scf::ForOp> reductionLoops,
                           const llvm::DenseSet<unsigned> &needCombine) {
    auto axisName = StringAttr::get(ctx, "axis");
    auto sourceName = StringAttr::get(ctx, "source");
    auto combineName = StringAttr::get(ctx, "combine");

    bool hasScatter = false;
    func.walk([&](pim::ScatterOp) { hasScatter = true; });

    // grid 축이 결합을 요구할 때 쓸 연산자. reduction loop가 정확히 하나일 때만 정해진다.
    StringAttr gridCombine;
    if (reductionLoops.size() == 1)
      gridCombine =
          reductionLoops[0]->getAttrOfType<StringAttr>("pim.reduction_combine");

    SmallVector<Attribute> axes;
    if (!hasScatter)
      for (auto [d, g] : llvm::enumerate(grid)) {
        SmallVector<NamedAttribute> fields{
            NamedAttribute(axisName, g->getAttr("pim.grid_dim")),
            NamedAttribute(sourceName, StringAttr::get(ctx, "grid"))};
        if (needCombine.contains(d)) {
          if (!gridCombine)
            continue;
          fields.push_back(NamedAttribute(combineName, gridCombine));
        }
        axes.push_back(DictionaryAttr::get(ctx, fields));
      }

    for (unsigned i = 0, e = reductionLoops.size(); i < e; i++) {
      // ArrayRef가 주는 참조는 const라, 값 의미인 Op을 복사해 non-const 멤버를 부른다.
      scf::ForOp loop = reductionLoops[i];
      bool sideEffect = false;
      loop.getBody()->walk([&](Operation *op) {
        if (isa<pim::StoreOp, pim::ScatterOp>(op))
          sideEffect = true;
      });
      if (sideEffect)
        continue;
      StringAttr name = reductionLoops.size() == 1
                            ? StringAttr::get(ctx, "k")
                            : StringAttr::get(ctx, "k" + llvm::Twine(i));
      axes.push_back(DictionaryAttr::get(
          ctx,
          {NamedAttribute(axisName, name),
           NamedAttribute(sourceName, StringAttr::get(ctx, "reduction")),
           NamedAttribute(combineName, loop->getAttrOfType<StringAttr>(
                                           "pim.reduction_combine"))}));
    }
    return ArrayAttr::get(ctx, axes);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMAnalyzeReusePass() {
  return std::make_unique<PIMAnalyzeReusePass>();
}
