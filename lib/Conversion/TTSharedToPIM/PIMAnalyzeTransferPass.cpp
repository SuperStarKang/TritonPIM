#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// pim.load는 !pim.access를 거치고 pim.gather는 base를 직접 받지만, 둘 다 유닛 메모리에
// tile 하나를 만든다.
static Value residentTile(Operation *op) {
  if (auto load = dyn_cast<pim::LoadOp>(op))
    return load.getResult();
  if (auto gather = dyn_cast<pim::GatherOp>(op))
    return gather.getResult();
  return {};
}

// 한 순간에 유닛 메모리에 함께 있어야 하는 tile 목록. 가장 깊이 있는 전송의 블록을 잡고,
// 그 블록에서 매 반복 새로 읽는 것(streaming)과 그 블록을 감싸는 loop를 관통하는
// 것(persistent)을 모은다. GEMM이면 A tile + B tile + 누산기가 된다.
// 합계가 아니라 목록으로 두는 이유는 target의 메모리 계층이 하나가 아니기 때문이다.
// 어느 tile을 빠른 쪽에 두고 어느 것을 스트리밍할지는 개별 크기를 알아야 정한다.
// access 필드는 pim.access_id다. access를 거치지 않는 gather 결과와 누산기에는 없다.
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

  bool complete = true;
  SmallVector<Attribute> entries;
  auto add = [&](Operation *op, Value tile, StringRef lifetime) {
    int64_t bytes;
    if (!tensorBytes(tile.getType(), bytes)) {
      complete = false;
      return;
    }
    SmallVector<NamedAttribute> fields;
    if (auto load = dyn_cast<pim::LoadOp>(op))
      if (auto id = load.getAccess().getDefiningOp()->getAttrOfType<IntegerAttr>(
              "pim.access_id"))
        fields.push_back(NamedAttribute(accessName, id));
    fields.push_back(NamedAttribute(bytesName, IntegerAttr::get(i64, bytes)));
    fields.push_back(
        NamedAttribute(lifetimeName, StringAttr::get(ctx, lifetime)));
    entries.push_back(DictionaryAttr::get(ctx, fields));
  };

  // 매 반복 새로 읽히는 tile — 버퍼 하나를 돌려쓴다.
  for (Operation &op : *deepest)
    if (Value tile = residentTile(&op))
      add(&op, tile, "streaming");

  Operation *nest = deepest->getParentOp();
  // loop를 관통하는 누산기 — 자리를 계속 차지한다.
  if (auto loop = dyn_cast<scf::ForOp>(nest))
    for (BlockArgument arg : loop.getRegionIterArgs()) {
      int64_t bytes;
      if (tensorBytes(arg.getType(), bytes))
        entries.push_back(DictionaryAttr::get(
            ctx, {NamedAttribute(bytesName, IntegerAttr::get(i64, bytes)),
                  NamedAttribute(lifetimeName,
                                 StringAttr::get(ctx, "persistent"))}));
    }
  // 밖에서 한 번 읽어 안에서 쓰는 tile — 역시 loop 내내 자리를 차지한다.
  // --pim-remove-redundant-transfers가 loop 밖으로 뺀 load가 여기 걸린다. 빠뜨리면
  // 용량을 실제보다 작게 보고 들어가지 않는 분할을 고른다.
  func.walk([&](Operation *op) {
    Value tile = residentTile(op);
    if (!tile || op->getBlock() == deepest)
      return;
    if (llvm::any_of(tile.getUsers(),
                     [&](Operation *u) { return nest->isAncestor(u); }))
      add(op, tile, "persistent");
  });

  // 하나라도 못 세면 목록이 불완전해져 용량 판정에 쓸 수 없으므로 아예 붙이지 않는다.
  if (!complete || entries.empty())
    return {};
  return ArrayAttr::get(ctx, entries);
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

    auto i64 = IntegerType::get(func.getContext(), 64);
    for (pim::AccessOp a : accesses) {
      int64_t tile;
      if (tileBytes(a, tile))
        a->setAttr("pim.tile_bytes", IntegerAttr::get(i64, tile));
    }
    if (ArrayAttr resident = coResident(func))
      func->setAttr("pim.co_resident", resident);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMAnalyzeTransferPass() {
  return std::make_unique<PIMAnalyzeTransferPass>();
}
