#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_REMOVEPIMBUFFERS
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// op이 메모리를 쓸 수 있으면(=전송 재사용의 장벽) true. 읽기 전용/무효과는 false,
// 효과를 모르는 op(함수 호출·루프 등)은 보수적으로 true.
static bool mayWrite(Operation *op) {
  if (isMemoryEffectFree(op))
    return false;
  auto mem = dyn_cast<MemoryEffectOpInterface>(op);
  if (!mem)
    return true;
  SmallVector<MemoryEffects::EffectInstance> effects;
  mem.getEffects(effects);
  for (auto &e : effects)
    if (isa<MemoryEffects::Write>(e.getEffect()))
      return true;
  return false;
}

// a와 b 사이(같은 블록, a가 앞)에 쓰기 op이 있으면 true.
static bool writeBetween(Operation *a, Operation *b) {
  for (Operation *cur = a->getNextNode(); cur && cur != b; cur = cur->getNextNode())
    if (mayWrite(cur))
      return true;
  return false;
}

// load와 같은 access를 가진, 같은 블록의 더 앞선 값(앞 store의 값 / 앞 load의 결과)을 찾는다.
// 사이에 쓰기가 나오면 중단한다. 없으면 널.
static Value earlierValue(pim::LoadOp load) {
  Value d = load.getAccess();
  for (Operation *cur = load->getPrevNode(); cur; cur = cur->getPrevNode()) {
    if (auto st = dyn_cast<pim::StoreOp>(cur))
      if (st.getAccess() == d)
        return st.getValue(); // 쓰고 바로 읽기 → 그 값 전달
    if (auto ld = dyn_cast<pim::LoadOp>(cur))
      if (ld.getAccess() == d)
        return ld.getResult(); // 같은 영역 반복 load → 앞 결과 재사용
    if (mayWrite(cur))
      return Value(); // 그 영역을 바꿨을 수 있으므로 중단
  }
  return Value();
}

struct RemovePIMBuffersPass
    : public mlir::triton::impl::RemovePIMBuffersBase<RemovePIMBuffersPass> {
  using RemovePIMBuffersBase::RemovePIMBuffersBase;

  void runOnOperation() override {
    // 1) passthrough: store(load(%ds), %dd) → copy(%ds, %dd).
    SmallVector<Operation *> dead;
    getOperation().walk([&](pim::StoreOp store) {
      auto load = store.getValue().getDefiningOp<pim::LoadOp>();
      if (!load || !load.getResult().hasOneUse())
        return; // 로드 값이 이 store에만 쓰여야 한다.
      if (load->getBlock() != store->getBlock() || writeBetween(load, store))
        return;
      OpBuilder(store).create<pim::CopyOp>(store.getLoc(), load.getAccess(),
                                           store.getAccess());
      dead.append({store.getOperation(), load.getOperation()});
    });
    for (Operation *op : dead)
      op->erase();

    // 2) load 재사용(dedup + store→load forwarding). 대체한 load는 바로 지워, 연쇄(load 3개+)일 때
    // 뒤 스캔이 중간 load가 아니라 최초 정의를 찾도록 한다. walk 중 현재 op 삭제는 안전하다.
    getOperation().walk([&](pim::LoadOp load) {
      if (Value v = earlierValue(load)) {
        load.getResult().replaceAllUsesWith(v);
        load.erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createRemovePIMBuffersPass() {
  return std::make_unique<RemovePIMBuffersPass>();
}
