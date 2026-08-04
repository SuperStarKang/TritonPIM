#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

// GEN_PASS_DEF가 생성하는 impl::...Base는 mlir::triton 네임스페이스 안에 있어야 한다.
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTTSHAREDTOPIM
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// OpFoldResult(상수 또는 SSA)를 index Value로. 상수면 arith.constant를 만든다.
static Value toIndex(OpBuilder &b, Location loc, OpFoldResult ofr) {
  if (auto v = dyn_cast<Value>(ofr))
    return v;
  int64_t c = cast<IntegerAttr>(cast<Attribute>(ofr)).getInt();
  return b.create<arith::ConstantIndexOp>(loc, c);
}

static SmallVector<Value> toIndices(OpBuilder &b, Location loc,
                                    ArrayRef<OpFoldResult> ofrs) {
  SmallVector<Value> vs;
  for (OpFoldResult o : ofrs)
    vs.push_back(toIndex(b, loc, o));
  return vs;
}

// alloc 사용처 중 bufferization.to_tensor를 찾는다(로드 관용구의 끝).
// bufferization dialect에 링크 의존을 만들지 않으려 op 이름으로 확인한다.
static Operation *findToTensor(memref::AllocOp alloc) {
  for (Operation *u : alloc->getUsers())
    if (u->getName().getStringRef() == "bufferization.to_tensor")
      return u;
  return nullptr;
}

// alloc을 outs로 채우는 linalg.fill을 찾는다(mask의 padding 값).
static linalg::FillOp findFill(memref::AllocOp alloc) {
  for (Operation *u : alloc->getUsers())
    if (auto f = dyn_cast<linalg::FillOp>(u))
      return f;
  return nullptr;
}

// alloc이 두 개 이상의 copy 대상으로 채워지면 wrap-around 로드다. 하나의 pim.access로
// 표현할 수 없고, masked walk가 두 copy를 각각 매치해 공유 alloc을 이중 erase하면
// double free가 나므로 여기서 감지한다.
static bool isWrapAroundBuffer(memref::AllocOp alloc) {
  int copyTargets = 0;
  for (Operation *u : alloc->getUsers()) {
    auto sv = dyn_cast<memref::SubViewOp>(u);
    if (!sv)
      continue;
    for (Operation *su : sv->getUsers())
      if (auto c = dyn_cast<memref::CopyOp>(su))
        if (c.getTarget() == sv.getResult()) {
          copyTargets++;
          break;
        }
  }
  return copyTargets > 1;
}

struct ConvertTTSharedToPIMPass
    : public mlir::triton::impl::ConvertTTSharedToPIMBase<ConvertTTSharedToPIMPass> {
  using ConvertTTSharedToPIMBase::ConvertTTSharedToPIMBase;

  void runOnOperation() override {
    WalkResult res = getOperation().walk([&](func::FuncOp func) {
      if (func.getBody().empty())
        return WalkResult::advance();
      raiseFunc(func);
      return succeeded(checkResidual(func)) ? WalkResult::advance()
                                            : WalkResult::interrupt();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }

  // 메모리 접근 관용구를 pim.access + pim.load/store로 끌어올려 staging 버퍼를 없앤다.
  // scf.for와 program_id 식은 건드리지 않고, 계산(named linalg)도 그대로 둔다.
  void raiseFunc(func::FuncOp func) {
    OpBuilder b(func.getContext());
    int64_t nextId = 0;

    auto makeAccess = [&](Location loc, Value base, Value offset,
                          ValueRange strides, ArrayRef<int64_t> sizes,
                          ValueRange valid, Value pad) {
      // origin 은 --pim-recover-access-origin 이 채운다.
      return b.create<pim::AccessOp>(loc, pim::AccessType::get(b.getContext()),
                                     base, offset, ValueRange{}, strides,
                                     b.getDenseI64ArrayAttr(sizes), valid, pad,
                                     b.getI64IntegerAttr(nextId++));
    };

    // 같은 reinterpret_cast를 load와 store가 함께 쓰면 하나의 pim.access를 공유한다.
    llvm::DenseMap<Operation *, Value> access;
    auto getAccess = [&](memref::ReinterpretCastOp rc) -> Value {
      auto it = access.find(rc);
      if (it != access.end())
        return it->second;
      // 삽입 위치는 guard로 되돌린다. 여기서 옮긴 위치가 남으면 바깥 create가 rc 자리에
      // op을 만들고, store에서는 그것이 곧 dominance 위반이다.
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(rc);
      Location loc = rc.getLoc();
      Value a = makeAccess(loc, rc.getSource(),
                           toIndex(b, loc, rc.getMixedOffsets()[0]),
                           toIndices(b, loc, rc.getMixedStrides()),
                           rc.getStaticSizes(), {}, Value());
      access[rc] = a;
      return a;
    };

    SmallVector<Operation *> dead;  // 반드시 죽는 관용구 op (사용처→정의 순으로 쌓는다)
    SmallVector<Operation *> roots; // reinterpret_cast 등 use_empty일 때만 지운다

    // mask 로드: reinterpret_cast + alloc + (fill?) + subview×2 + copy
    //          → pim.access{valid, pad?} + pim.load.
    func.walk([&](memref::CopyOp copy) {
      auto vsub = copy.getSource().getDefiningOp<memref::SubViewOp>();
      auto bsub = copy.getTarget().getDefiningOp<memref::SubViewOp>();
      if (!vsub || !bsub)
        return;
      auto rc = vsub.getSource().getDefiningOp<memref::ReinterpretCastOp>();
      auto alloc = bsub.getSource().getDefiningOp<memref::AllocOp>();
      if (!rc || !alloc)
        return;
      if (isWrapAroundBuffer(alloc)) {
        copy->emitError("wrap-around load (two disjoint regions from `% M`/`% N` "
                        "index wrapping) is unsupported on PIM; rewrite the kernel "
                        "to mask the boundary (`mask=`/`other=`) instead of modulo "
                        "indexing");
        signalPassFailure();
        return;
      }
      Operation *toTensor = findToTensor(alloc);
      if (!toTensor)
        return;
      linalg::FillOp fill = findFill(alloc); // 없을 수 있다(채움값 없는 마스크 로드).
      b.setInsertionPoint(toTensor);
      Location loc = toTensor->getLoc();
      Value d = makeAccess(loc, rc.getSource(),
                           toIndex(b, loc, rc.getMixedOffsets()[0]),
                           toIndices(b, loc, rc.getMixedStrides()),
                           rc.getStaticSizes(),
                           toIndices(b, loc, vsub.getMixedSizes()),
                           fill ? fill.getInputs()[0] : Value());
      Value v = b.create<pim::LoadOp>(loc, toTensor->getResult(0).getType(), d);
      toTensor->getResult(0).replaceAllUsesWith(v);
      dead.append({toTensor, copy.getOperation(), vsub.getOperation(),
                   bsub.getOperation(), alloc.getOperation()});
      if (fill)
        dead.push_back(isa<scf::IfOp>(fill->getParentOp()) ? fill->getParentOp()
                                                           : fill.getOperation());
      roots.push_back(rc);
    });

    // 기본 로드: copy(reinterpret_cast → alloc) + to_tensor(alloc) → pim.load.
    func.walk([&](memref::CopyOp copy) {
      auto rc = copy.getSource().getDefiningOp<memref::ReinterpretCastOp>();
      auto alloc = copy.getTarget().getDefiningOp<memref::AllocOp>();
      if (!rc || !alloc)
        return;
      Operation *toTensor = findToTensor(alloc);
      if (!toTensor)
        return;
      b.setInsertionPoint(toTensor);
      Value v = b.create<pim::LoadOp>(toTensor->getLoc(),
                                      toTensor->getResult(0).getType(),
                                      getAccess(rc));
      toTensor->getResult(0).replaceAllUsesWith(v);
      dead.append({toTensor, copy.getOperation(), alloc.getOperation()});
    });

    // 저장: materialize_in_destination(value, dest) → pim.store.
    // dest가 reinterpret_cast면 tile 저장, subview(reinterpret_cast)면 masked 저장이다.
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef() !=
          "bufferization.materialize_in_destination")
        return;
      Value dest = op->getOperand(1);
      if (auto rc = dest.getDefiningOp<memref::ReinterpretCastOp>()) {
        b.setInsertionPoint(op);
        b.create<pim::StoreOp>(op->getLoc(), op->getOperand(0), getAccess(rc));
        dead.push_back(op);
        return;
      }
      auto sub = dest.getDefiningOp<memref::SubViewOp>();
      auto es = op->getOperand(0).getDefiningOp<tensor::ExtractSliceOp>();
      if (!sub || !es)
        return;
      auto rc = sub.getSource().getDefiningOp<memref::ReinterpretCastOp>();
      if (!rc)
        return;
      b.setInsertionPoint(op);
      Location loc = op->getLoc();
      Value d = makeAccess(loc, rc.getSource(),
                           toIndex(b, loc, rc.getMixedOffsets()[0]),
                           toIndices(b, loc, rc.getMixedStrides()),
                           rc.getStaticSizes(),
                           toIndices(b, loc, es.getMixedSizes()), Value());
      b.create<pim::StoreOp>(loc, es.getSource(), d);
      dead.append({op, es.getOperation(), sub.getOperation()});
      roots.push_back(rc);
    });

    // 죽은 관용구 op을 먼저 지우고(사용처→정의 순), 그다음 빈 root를 지운다.
    // erased 집합으로 이중 erase를 막는다.
    llvm::SmallPtrSet<Operation *, 32> erased;
    for (Operation *op : dead)
      if (erased.insert(op).second)
        op->erase();
    for (Operation *op : roots)
      if (!erased.contains(op) && op->use_empty()) {
        erased.insert(op);
        op->erase();
      }
    for (auto &kv : access)
      if (!erased.contains(kv.first) && kv.first->use_empty())
        kv.first->erase();
  }

  // 변환하지 못한 메모리 접근이 남았는지 본다. 남으면 뒤의 배치 pass가 그 버퍼를
  // 보지 못한 채 조용히 부분 배치를 만든다.
  LogicalResult checkResidual(func::FuncOp func) {
    WalkResult res = func.walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      StringRef d = dialect ? dialect->getNamespace() : StringRef();
      if (d != "memref" && d != "bufferization")
        return WalkResult::advance();
      op->emitError("convert-ttshared-to-pim: this memory access pattern is not "
                    "supported on PIM and was left unconverted");
      return WalkResult::interrupt();
    });
    return failure(res.wasInterrupted());
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTTSharedToPIMPass() {
  return std::make_unique<ConvertTTSharedToPIMPass>();
}
