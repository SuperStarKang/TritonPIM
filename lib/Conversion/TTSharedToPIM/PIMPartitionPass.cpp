#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMPARTITION
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

// access를 통해 instance 하나가 옮기는 총 바이트. 전송 op(load/store/copy)마다 그것을
// 감싸는 비-grid loop 반복 횟수를 곱해 더한다. access op이 아니라 전송 op의 위치를
// 봐야 한다 — hoisting이 access는 두고 load만 loop 밖으로 옮길 수 있다.
static bool accessBytes(pim::AccessOp access, int64_t &bytes) {
  int64_t tile;
  if (!tileBytes(access, tile))
    return false;
  bytes = 0;
  for (Operation *user : access.getResult().getUsers()) {
    if (!isa<pim::LoadOp, pim::StoreOp, pim::CopyOp>(user))
      continue;
    int64_t repeats;
    if (!repeatCount(user, repeats))
      return false;
    bytes += tile * repeats;
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

// 한 순간에 유닛 메모리에 있어야 하는 양. 가장 깊이 있는 pim.load의 블록을 잡아, 그
// 블록의 load 결과 tensor들과 그 블록이 scf.for 몸체라면 tensor 타입 iter_arg를 더한다.
// GEMM이면 A tile + B tile + 누산기가 된다.
static bool workingSetBytes(func::FuncOp func, int64_t &bytes) {
  Block *deepest = nullptr;
  unsigned deepestDepth = 0;
  func.walk([&](pim::LoadOp load) {
    unsigned depth = blockDepth(load->getBlock());
    if (!deepest || depth > deepestDepth) {
      deepest = load->getBlock();
      deepestDepth = depth;
    }
  });
  if (!deepest)
    return false;

  bytes = 0;
  for (Operation &op : *deepest) {
    auto load = dyn_cast<pim::LoadOp>(&op);
    if (!load)
      continue;
    int64_t tile;
    if (!tensorBytes(load.getResult().getType(), tile))
      return false;
    bytes += tile;
  }
  if (auto loop = dyn_cast<scf::ForOp>(deepest->getParentOp()))
    for (BlockArgument arg : loop.getRegionIterArgs()) {
      int64_t acc;
      if (tensorBytes(arg.getType(), acc))
        bytes += acc;
    }
  return true;
}

// n의 약수를 오름차순으로. 분할 후보에서 각 축이 가질 수 있는 유닛 수다.
static SmallVector<int64_t> divisorsOf(int64_t n) {
  SmallVector<int64_t> result;
  for (int64_t i = 1; i <= n; i++)
    if (n % i == 0)
      result.push_back(i);
  return result;
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

// 전송량을 계산할 access 하나의 정보.
struct AccessCost {
  int64_t bytes;
  llvm::SmallDenseSet<int64_t> invariant;
};

struct PIMPartitionPass
    : public mlir::triton::impl::PIMPartitionBase<PIMPartitionPass> {
  using PIMPartitionBase::PIMPartitionBase;

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

    // access별 전송 바이트. grid와 무관하므로 표를 못 만드는 경우에도 단다.
    for (pim::AccessOp a : accesses) {
      int64_t bytes;
      if (accessBytes(a, bytes))
        a->setAttr("pim.access_bytes", IntegerAttr::get(i64, bytes));
    }

    int64_t ws = 0;
    if (workingSetBytes(func, ws))
      func->setAttr("pim.working_set_bytes", IntegerAttr::get(i64, ws));

    SmallVector<scf::ForOp> grid;
    collectGridLoops(func, grid);
    if (grid.empty())
      return; // grid 축이 없으면 분할 후보도 없다.

    // 순서를 바꿔도 되는지. 서로 다른 instance가 같은 출력에 쓰면 안 된다.
    func->setAttr("pim.grid_permutable",
                  BoolAttr::get(ctx, !func->hasAttr("pim.reduce_across_programs")));

    SmallVector<int64_t> extents;
    for (scf::ForOp g : grid) {
      int64_t trips;
      // grid 크기가 runtime 값이면 표를 만들지 않는다. 0이면 도는 instance가 없다.
      if (!constTripCount(g, trips) || trips <= 0)
        return;
      extents.push_back(trips);
    }
    func->setAttr("pim.grid_extents", DenseI64ArrayAttr::get(ctx, extents));

    // 전송량 식에 넣을 access별 정보를 모은다. 하나라도 빠지면 표를 만들지 않는다.
    SmallVector<AccessCost> costs;
    for (pim::AccessOp a : accesses) {
      auto bytes = a->getAttrOfType<IntegerAttr>("pim.access_bytes");
      auto invariant =
          a->getAttrOfType<DenseI64ArrayAttr>("pim.invariant_program_dims");
      if (!bytes || !invariant)
        return;
      ArrayRef<int64_t> dims = invariant.asArrayRef();
      AccessCost cost;
      cost.bytes = bytes.getInt();
      cost.invariant.insert(dims.begin(), dims.end());
      costs.push_back(cost);
    }

    func->setAttr("pim.partition_cost", buildCostTable(ctx, extents, costs));
  }

  // 각 축의 약수를 모두 조합해 후보를 만들고 전송량을 계산한다. 축 d의 유닛 수가 s_d,
  // grid 크기가 g_d일 때
  //   bytes(s) = Σ_a  access_bytes_a × Π{ s_j : j ∈ I_a } × Π{ g_j : j ∉ I_a }
  // 이다. access가 축 j에 불변이면 j가 달라져도 같은 tile이라 유닛 수만큼만 옮기면 되고,
  // 불변이 아니면 grid 크기만큼 서로 다른 tile을 옮긴다.
  ArrayAttr buildCostTable(MLIRContext *ctx, ArrayRef<int64_t> extents,
                           ArrayRef<AccessCost> costs) {
    unsigned rank = extents.size();
    SmallVector<SmallVector<int64_t>> divisors;
    for (int64_t g : extents)
      divisors.push_back(divisorsOf(g));

    auto i64 = IntegerType::get(ctx, 64);
    auto splitsName = StringAttr::get(ctx, "splits");
    auto bytesName = StringAttr::get(ctx, "bytes");

    SmallVector<Attribute> table;
    SmallVector<size_t> pick(rank, 0);
    SmallVector<int64_t> splits(rank);
    while (true) {
      for (unsigned d = 0; d < rank; d++)
        splits[d] = divisors[d][pick[d]];

      int64_t total = 0;
      for (const AccessCost &cost : costs) {
        int64_t bytes = cost.bytes;
        for (unsigned d = 0; d < rank; d++)
          bytes *= cost.invariant.contains(d) ? splits[d] : extents[d];
        total += bytes;
      }
      table.push_back(DictionaryAttr::get(
          ctx, {NamedAttribute(splitsName, DenseI64ArrayAttr::get(ctx, splits)),
                NamedAttribute(bytesName, IntegerAttr::get(i64, total))}));

      // 마지막 축부터 올려 다음 조합으로 넘어간다. 전부 넘치면 끝이다.
      unsigned d = rank;
      while (d > 0 && ++pick[d - 1] == divisors[d - 1].size())
        pick[--d] = 0;
      if (d == 0)
        break;
    }
    return ArrayAttr::get(ctx, table);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMPartitionPass() {
  return std::make_unique<PIMPartitionPass>();
}
