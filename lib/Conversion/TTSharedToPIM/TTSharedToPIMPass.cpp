#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseMap.h"

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

// reinterpret_cast의 offset/sizes/strides를 그대로 pim.access로 옮긴다(무비용 주소 명세).
static Value buildAccess(OpBuilder &b, memref::ReinterpretCastOp rc) {
  Location loc = rc.getLoc();
  Value offset = toIndex(b, loc, rc.getMixedOffsets()[0]);
  SmallVector<Value> strides;
  for (OpFoldResult s : rc.getMixedStrides())
    strides.push_back(toIndex(b, loc, s));
  // structured 접근의 sizes는 항상 static. dynamic이면 getStaticSizes에 kDynamic이 그대로 드러난다.
  return b.create<pim::AccessOp>(loc, pim::AccessType::get(b.getContext()),
                                 rc.getSource(), offset, strides,
                                 b.getDenseI64ArrayAttr(rc.getStaticSizes()));
}

// alloc 사용처 중 bufferization.to_tensor를 찾는다(로드 관용구의 끝).
// bufferization dialect에 링크 의존을 만들지 않으려 op 이름으로 확인한다.
static Operation *findToTensor(memref::AllocOp alloc) {
  for (Operation *u : alloc->getUsers())
    if (u->getName().getStringRef() == "bufferization.to_tensor")
      return u;
  return nullptr;
}

// 계산 op을 종류 태그로 분류한다. named op은 그대로 두고 pim.class만 단다(치환 아님).
// control/glue(fill/for/subview 등)는 "" 반환 → 태그 없음.
static StringRef classifyComputeOp(Operation *op) {
  if (isa<linalg::MatmulOp>(op))
    return "gemm";
  if (isa<linalg::MatvecOp>(op))
    return "gemv";
  if (isa<linalg::TransposeOp>(op))
    return "transpose";
  if (isa<linalg::ReduceOp>(op))
    return "reduce";
  if (isa<linalg::GenericOp>(op)) {
    // reduction 루프가 있으면 reduce, 전부 parallel이면 elementwise.
    if (cast<linalg::LinalgOp>(op).getNumReductionLoops() > 0)
      return "reduce";
    return "elementwise";
  }
  return "";
}

struct ConvertTTSharedToPIMPass
    : public mlir::triton::impl::ConvertTTSharedToPIMBase<ConvertTTSharedToPIMPass> {
  using ConvertTTSharedToPIMBase::ConvertTTSharedToPIMBase;

  void runOnOperation() override {
    getOperation().walk([&](func::FuncOp func) {
      if (!func.getBody().empty())
        raiseFunc(func);
    });
  }

  // 메모리 3중구조(reinterpret_cast+alloc+copy+to_tensor / materialize_in_destination)를
  // pim.access + pim.load/store로 끌어올려 staging 버퍼를 없앤다. scf.for와 program_id 식은
  // 건드리지 않는다(SPMD, symbolic). 계산(named linalg)은 유지하고 pim.class 태그만 단다.
  void raiseFunc(func::FuncOp func) {
    OpBuilder b(func.getContext());

    // 같은 reinterpret_cast를 load와 store가 함께 쓰면 하나의 pim.access를 공유한다.
    llvm::DenseMap<Operation *, Value> access;
    auto getAccess = [&](memref::ReinterpretCastOp rc) -> Value {
      auto it = access.find(rc);
      if (it != access.end())
        return it->second;
      b.setInsertionPoint(rc);
      Value a = buildAccess(b, rc);
      access[rc] = a;
      return a;
    };

    SmallVector<Operation *> dead;

    // 로드: copy(reinterpret_cast -> alloc) + to_tensor(alloc) -> pim.load
    func.walk([&](memref::CopyOp copy) {
      auto rc = copy.getSource().getDefiningOp<memref::ReinterpretCastOp>();
      auto alloc = copy.getTarget().getDefiningOp<memref::AllocOp>();
      if (!rc || !alloc)
        return; // 로드 관용구가 아니면 그대로 둔다(예: wraparound의 subview 대상).
      Operation *toTensor = findToTensor(alloc);
      if (!toTensor)
        return;
      b.setInsertionPoint(toTensor);
      Value v = b.create<pim::LoadOp>(toTensor->getLoc(),
                                      toTensor->getResult(0).getType(),
                                      getAccess(rc));
      toTensor->getResult(0).replaceAllUsesWith(v);
      dead.push_back(toTensor);
      dead.push_back(copy);
      dead.push_back(alloc);
    });

    // 저장: materialize_in_destination(value, reinterpret_cast) -> pim.store
    // (dest=operand 1). bufferization 링크 의존을 피하려 op 이름으로 확인한다.
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef() !=
          "bufferization.materialize_in_destination")
        return;
      auto rc = op->getOperand(1).getDefiningOp<memref::ReinterpretCastOp>();
      if (!rc)
        return;
      b.setInsertionPoint(op);
      b.create<pim::StoreOp>(op->getLoc(), op->getOperand(0), getAccess(rc));
      dead.push_back(op);
    });

    // 죽은 관용구 op을 먼저 지우고(사용처 → 정의 순으로 쌓임), 그다음 빈 reinterpret_cast를 지운다.
    for (Operation *op : dead)
      op->erase();
    for (auto &kv : access)
      if (kv.first->use_empty())
        kv.first->erase();

    // 분류: 각 계산 op에 pim.class 태그(치환 아님).
    func.walk([&](Operation *op) {
      StringRef cls = classifyComputeOp(op);
      if (!cls.empty())
        op->setAttr("pim.class", b.getStringAttr(cls));
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTTSharedToPIMPass() {
  return std::make_unique<ConvertTTSharedToPIMPass>();
}
