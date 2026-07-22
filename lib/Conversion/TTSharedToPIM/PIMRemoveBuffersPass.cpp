#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
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

// access가 가리키는 base memref. pim.access가 만든 값이 아니면 어느 버퍼인지 모르므로 널.
static Value accessBase(Value access) {
  auto op = access.getDefiningOp<pim::AccessOp>();
  return op ? op.getBase() : Value();
}

// 두 base가 같은 버퍼를 가리킬 수 있으면 true. base가 다르면 겹치지 않는다고 본다.
// 이것은 가정이다 — Triton은 커널 포인터 인자가 서로 다른 버퍼임을 보장하지 않는다.
// 한쪽이라도 모르면 보수적으로 겹친다고 본다.
static bool mayAlias(Value a, Value b) { return !a || !b || a == b; }

// op이 base 버퍼를 쓸 수 있으면 true. pim 쓰기 op은 대상 버퍼를 알 수 있으므로 base를
// 비교하고, 그 외에는 메모리 효과만 보고 판단한다.
static bool writesBuffer(Operation *op, Value base) {
  if (auto store = dyn_cast<pim::StoreOp>(op))
    return mayAlias(accessBase(store.getAccess()), base);
  if (auto copy = dyn_cast<pim::CopyOp>(op))
    return mayAlias(accessBase(copy.getDst()), base);
  if (auto scatter = dyn_cast<pim::ScatterOp>(op))
    return mayAlias(scatter.getBase(), base);
  return mayWrite(op);
}

// loop 몸체 어디에서든(중첩 region 포함) base 버퍼를 쓰면 true.
static bool writesInLoop(scf::ForOp loop, Value base) {
  WalkResult r = loop.getBody()->walk([&](Operation *op) {
    // scf.for/scf.if 같은 컨테이너 자체의 효과는 그 안의 op을 방문하며 따로 본다.
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return WalkResult::advance();
    return writesBuffer(op, base) ? WalkResult::interrupt()
                                  : WalkResult::advance();
  });
  return r.wasInterrupted();
}

// loop 안에서 매번 같은 데이터를 읽는 pim.load를 loop 앞으로 옮긴다. 조건은 세 가지다.
//  - access가 loop 밖에서 정의된다. 즉 유도변수에 의존하지 않는다.
//  - loop 안에 그 버퍼를 쓰는 op이 없다.
//  - loop 몸체 블록의 직속 자식이다. scf.if 안의 load를 빼면 실행되지 않았을 읽기가
//    실행되므로 옮기지 않는다.
// grid loop에서는 빼지 않는다. instance 사이에 공유되는 전송인지는 분할 구조가 정해진
// 뒤에야 알 수 있어서, 여기서 빼면 유닛마다 사본이 필요하다는 사실이 사라진다.
static void hoistLoads(scf::ForOp loop) {
  if (loop->hasAttr("pim.grid_dim"))
    return;
  SmallVector<pim::LoadOp> hoistable;
  for (Operation &op : *loop.getBody()) {
    auto load = dyn_cast<pim::LoadOp>(&op);
    if (!load)
      continue;
    Value access = load.getAccess();
    // access가 loop 안에서 만들어지면 반복마다 읽는 영역이 달라질 수 있다.
    if (loop->isAncestor(access.getParentRegion()->getParentOp()))
      continue;
    if (writesInLoop(loop, accessBase(access)))
      continue;
    hoistable.push_back(load);
  }
  // 앞에서부터 loop 바로 앞으로 옮기면 원래 순서가 유지된다.
  for (pim::LoadOp load : hoistable)
    load->moveBefore(loop.getOperation());
}

struct RemovePIMBuffersPass
    : public mlir::triton::impl::RemovePIMBuffersBase<RemovePIMBuffersPass> {
  using RemovePIMBuffersBase::RemovePIMBuffersBase;

  void runOnOperation() override {
    // 0) loop-invariant load hoisting. 뒤의 두 재작성보다 먼저 한다. loop 밖으로 나온
    // load들이 같은 블록에 모여야 그 자리에서 재사용 판정을 받을 수 있다.
    // walk 기본값이 post-order라 안쪽 loop부터 처리되고, 안쪽에서 빠져나온 load가
    // 바깥 loop의 직속 자식이 되어 다시 후보가 된다(여러 겹을 타고 올라간다).
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : loops)
      hoistLoads(loop);

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
