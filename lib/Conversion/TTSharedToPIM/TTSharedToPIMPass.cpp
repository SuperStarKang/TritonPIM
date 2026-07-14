#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

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

// 여러 OpFoldResult를 index Value 리스트로.
static SmallVector<Value> toIndices(OpBuilder &b, Location loc,
                                    ArrayRef<OpFoldResult> ofrs) {
  SmallVector<Value> vs;
  for (OpFoldResult o : ofrs)
    vs.push_back(toIndex(b, loc, o));
  return vs;
}

// pim.access 생성. clause(valid/pad/index/gather_dim)는 없으면 빈 값/널로 둔다.
static Value makeAccess(OpBuilder &b, Location loc, Value base, Value offset,
                        ValueRange strides, ArrayRef<int64_t> sizes,
                        ValueRange valid, Value pad, Value index,
                        IntegerAttr gatherDim) {
  return b.create<pim::AccessOp>(loc, pim::AccessType::get(b.getContext()), base,
                                 offset, strides, b.getDenseI64ArrayAttr(sizes),
                                 valid, pad, index, gatherDim);
}

// reinterpret_cast의 offset/sizes/strides를 그대로 pim.access로 옮긴다(①②③).
static Value buildAccess(OpBuilder &b, memref::ReinterpretCastOp rc) {
  Location loc = rc.getLoc();
  Value offset = toIndex(b, loc, rc.getMixedOffsets()[0]);
  SmallVector<Value> strides = toIndices(b, loc, rc.getMixedStrides());
  return makeAccess(b, loc, rc.getSource(), offset, strides, rc.getStaticSizes(),
                    {}, Value(), Value(), IntegerAttr());
}

// alloc 사용처 중 bufferization.to_tensor를 찾는다(로드 관용구의 끝).
// bufferization dialect에 링크 의존을 만들지 않으려 op 이름으로 확인한다.
static Operation *findToTensor(memref::AllocOp alloc) {
  for (Operation *u : alloc->getUsers())
    if (u->getName().getStringRef() == "bufferization.to_tensor")
      return u;
  return nullptr;
}

// alloc을 outs로 채우는 linalg.fill을 찾는다(⑤ mask의 padding).
static linalg::FillOp findFill(memref::AllocOp alloc) {
  for (Operation *u : alloc->getUsers())
    if (auto f = dyn_cast<linalg::FillOp>(u))
      return f;
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

  // 메모리 접근(reinterpret_cast+alloc+copy+to_tensor / materialize / mask / gather)을
  // pim.access + pim.load/store/gather/scatter로 끌어올려 staging 버퍼를 없앤다. scf.for와
  // program_id 식은 건드리지 않는다(SPMD, symbolic). 계산(named linalg)은 유지하고 태그만 단다.
  void raiseFunc(func::FuncOp func) {
    OpBuilder b(func.getContext());

    // 같은 reinterpret_cast를 load와 store가 함께 쓰면 하나의 pim.access를 공유한다(①②③).
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

    SmallVector<Operation *> dead;     // 반드시 죽는 관용구 op (사용처→정의 순으로 쌓는다)
    SmallVector<Operation *> roots;    // reinterpret_cast/cast 등 use_empty일 때만 지운다

    // ⑤ mask 로드: reinterpret_cast + alloc + fill + subview×2 + copy → pim.access{valid,pad}+load.
    // (판별: copy의 양쪽이 subview) valid = subview 크기, pad = fill 입력값.
    func.walk([&](memref::CopyOp copy) {
      auto vsub = copy.getSource().getDefiningOp<memref::SubViewOp>();
      auto bsub = copy.getTarget().getDefiningOp<memref::SubViewOp>();
      if (!vsub || !bsub)
        return;
      auto rc = vsub.getSource().getDefiningOp<memref::ReinterpretCastOp>();
      auto alloc = bsub.getSource().getDefiningOp<memref::AllocOp>();
      if (!rc || !alloc)
        return;
      Operation *toTensor = findToTensor(alloc);
      linalg::FillOp fill = findFill(alloc);
      if (!toTensor || !fill)
        return;
      b.setInsertionPoint(toTensor);
      Location loc = toTensor->getLoc();
      SmallVector<Value> valid = toIndices(b, loc, vsub.getMixedSizes());
      SmallVector<Value> strides = toIndices(b, loc, rc.getMixedStrides());
      Value offset = toIndex(b, loc, rc.getMixedOffsets()[0]);
      Value d = makeAccess(b, loc, rc.getSource(), offset, strides,
                           rc.getStaticSizes(), valid, fill.getInputs()[0],
                           Value(), IntegerAttr());
      Value v = b.create<pim::LoadOp>(loc, toTensor->getResult(0).getType(), d);
      toTensor->getResult(0).replaceAllUsesWith(v);
      // fill은 scf.if로 감싸여 있으면 그 if째로 지운다.
      Operation *fillDead = isa<scf::IfOp>(fill->getParentOp()) ? fill->getParentOp()
                                                                : fill.getOperation();
      dead.append({toTensor, copy.getOperation(), vsub.getOperation(),
                   bsub.getOperation(), fillDead, alloc.getOperation()});
      roots.push_back(rc);
    });

    // ⑥ row-gather 로드: alloc + scf.for{extract+per-row reinterpret_cast+subview+copy}
    // → pim.access{gather_dim,index}+load. (판별: copy source=reinterpret_cast,
    // target=subview(alloc), rc offset = index_cast(tensor.extract)).
    func.walk([&](memref::CopyOp copy) {
      auto rc = copy.getSource().getDefiningOp<memref::ReinterpretCastOp>();
      auto sub = copy.getTarget().getDefiningOp<memref::SubViewOp>();
      if (!rc || !sub)
        return;
      auto alloc = sub.getSource().getDefiningOp<memref::AllocOp>();
      if (!alloc)
        return;
      auto ofsVal = dyn_cast<Value>(rc.getMixedOffsets()[0]);
      if (!ofsVal)
        return;
      auto icast = ofsVal.getDefiningOp<arith::IndexCastOp>();
      if (!icast)
        return;
      auto extract = icast.getIn().getDefiningOp<tensor::ExtractOp>();
      if (!extract)
        return;
      auto forOp = dyn_cast<scf::ForOp>(copy->getParentOp());
      Operation *toTensor = findToTensor(alloc);
      if (!forOp || !toTensor)
        return;
      // gather 축 = subview offset이 동적(루프 인덱스)인 첫 축.
      int64_t gdim = 0;
      for (auto [i, o] : llvm::enumerate(sub.getMixedOffsets()))
        if (isa<Value>(o)) {
          gdim = i;
          break;
        }
      b.setInsertionPoint(forOp);
      Location loc = forOp.getLoc();
      SmallVector<Value> strides = toIndices(b, loc, rc.getMixedStrides());
      Value offset = b.create<arith::ConstantIndexOp>(loc, 0);
      Value d = makeAccess(b, loc, rc.getSource(), offset, strides,
                           alloc.getType().getShape(), {}, Value(),
                           extract.getTensor(), b.getI64IntegerAttr(gdim));
      Value v = b.create<pim::LoadOp>(loc, toTensor->getResult(0).getType(), d);
      toTensor->getResult(0).replaceAllUsesWith(v);
      dead.append({toTensor, forOp.getOperation(), alloc.getOperation()});
    });

    // ① 기본 로드: copy(reinterpret_cast → alloc) + to_tensor(alloc) → pim.load.
    func.walk([&](memref::CopyOp copy) {
      auto rc = copy.getSource().getDefiningOp<memref::ReinterpretCastOp>();
      auto alloc = copy.getTarget().getDefiningOp<memref::AllocOp>();
      if (!rc || !alloc)
        return; // 로드 관용구가 아니면 그대로 둔다(⑤/⑥/wraparound 등).
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
    // dest가 reinterpret_cast면 ①③ store, subview(reinterpret_cast)면 ⑤ masked store.
    // bufferization 링크 의존을 피하려 op 이름으로 확인한다.
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
      // ⑤ masked store: dest=subview(reinterpret_cast), value=extract_slice(full).
      auto sub = dest.getDefiningOp<memref::SubViewOp>();
      auto es = op->getOperand(0).getDefiningOp<tensor::ExtractSliceOp>();
      if (!sub || !es)
        return;
      auto rc = sub.getSource().getDefiningOp<memref::ReinterpretCastOp>();
      if (!rc)
        return;
      b.setInsertionPoint(op);
      Location loc = op->getLoc();
      SmallVector<Value> valid = toIndices(b, loc, es.getMixedSizes());
      SmallVector<Value> strides = toIndices(b, loc, rc.getMixedStrides());
      Value offset = toIndex(b, loc, rc.getMixedOffsets()[0]);
      Value d = makeAccess(b, loc, rc.getSource(), offset, strides,
                           rc.getStaticSizes(), valid, Value(), Value(),
                           IntegerAttr());
      b.create<pim::StoreOp>(loc, es.getSource(), d);
      dead.append({op, es.getOperation(), sub.getOperation()});
      roots.push_back(rc);
    });

    // ⑦ 원소별 gather: linalg.generic 안의 tensor.extract(to_tensor(memref.cast(base))) → pim.gather.
    func.walk([&](linalg::GenericOp g) {
      tensor::ExtractOp extract;
      g.walk([&](tensor::ExtractOp e) { extract = e; });
      if (!extract)
        return;
      // extract가 body에 직접 있어야 한다(무조건 전송). scf.if 안이면 mask 있는 gather라
      // pim.gather로는 표현 못 하므로 그대로 둔다(잘못 변환 금지).
      if (extract->getParentOp() != g.getOperation())
        return;
      Operation *toT = extract.getTensor().getDefiningOp();
      if (!toT || toT->getName().getStringRef() != "bufferization.to_tensor")
        return;
      auto cast = toT->getOperand(0).getDefiningOp<memref::CastOp>();
      if (!cast)
        return;
      b.setInsertionPoint(g);
      Value v = b.create<pim::GatherOp>(g.getLoc(), g.getResult(0).getType(),
                                        cast.getSource(), g.getInputs()[0],
                                        Value(), Value());
      g.getResult(0).replaceAllUsesWith(v);
      dead.push_back(g);
      roots.append({toT, cast.getOperation()});
      // generic의 outs(init tensor)도 이제 안 쓰이면 정리한다.
      if (Operation *init = g.getOutputs()[0].getDefiningOp())
        roots.push_back(init);
    });

    // ⑦ masked gather: linalg.generic(ins=index,mask) body가 scf.if{extract}else{default}.
    // masked-out은 default로 채우므로 mask+default clause를 달아 pim.gather로.
    func.walk([&](linalg::GenericOp g) {
      if (g.getInputs().size() != 2)
        return;
      scf::IfOp ifOp;
      g.walk([&](scf::IfOp i) { ifOp = i; });
      if (!ifOp || ifOp.getElseRegion().empty())
        return;
      tensor::ExtractOp extract;
      ifOp.getThenRegion().walk([&](tensor::ExtractOp e) { extract = e; });
      if (!extract)
        return;
      Operation *toT = extract.getTensor().getDefiningOp();
      if (!toT || toT->getName().getStringRef() != "bufferization.to_tensor")
        return;
      auto castOp = toT->getOperand(0).getDefiningOp<memref::CastOp>();
      if (!castOp)
        return;
      // default = else 분기가 내는 값. generic 밖에서 정의돼야(지배) 그대로 쓸 수 있다.
      auto elseYield =
          cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
      Value def = elseYield.getOperand(0);
      if (Operation *d = def.getDefiningOp())
        if (d->getParentOfType<linalg::GenericOp>().getOperation() == g.getOperation())
          return;
      b.setInsertionPoint(g);
      Value v = b.create<pim::GatherOp>(g.getLoc(), g.getResult(0).getType(),
                                        castOp.getSource(), g.getInputs()[0],
                                        g.getInputs()[1], def);
      g.getResult(0).replaceAllUsesWith(v);
      dead.push_back(g);
      roots.append({toT, castOp.getOperation()});
      if (Operation *init = g.getOutputs()[0].getDefiningOp())
        roots.push_back(init);
    });

    // ⑦ 원소별 scatter: linalg.generic 안의 memref.store(memref.cast(base)) → pim.scatter.
    func.walk([&](linalg::GenericOp g) {
      memref::StoreOp store;
      g.walk([&](memref::StoreOp s) { store = s; });
      if (!store)
        return;
      // store가 body에 직접 있어야 한다(무조건 전송). scf.if 안이면 mask 있는 scatter라 그대로 둔다.
      if (store->getParentOp() != g.getOperation())
        return;
      auto cast = store.getMemref().getDefiningOp<memref::CastOp>();
      if (!cast)
        return;
      b.setInsertionPoint(g);
      b.create<pim::ScatterOp>(g.getLoc(), g.getInputs()[1], cast.getSource(),
                               g.getInputs()[0], Value());
      dead.push_back(g);
      roots.push_back(cast.getOperation());
    });

    // ⑦ masked scatter: linalg.generic(ins=index,value,mask) body가 scf.if{store}.
    // masked-out은 안 쓰므로 mask clause를 달아 pim.scatter로.
    func.walk([&](linalg::GenericOp g) {
      if (g.getInputs().size() != 3)
        return;
      scf::IfOp ifOp;
      g.walk([&](scf::IfOp i) { ifOp = i; });
      if (!ifOp)
        return;
      memref::StoreOp store;
      ifOp.getThenRegion().walk([&](memref::StoreOp s) { store = s; });
      if (!store)
        return;
      auto cast = store.getMemref().getDefiningOp<memref::CastOp>();
      if (!cast)
        return;
      b.setInsertionPoint(g);
      b.create<pim::ScatterOp>(g.getLoc(), g.getInputs()[1], cast.getSource(),
                               g.getInputs()[0], g.getInputs()[2]);
      dead.push_back(g);
      roots.push_back(cast.getOperation());
    });

    // 죽은 관용구 op을 먼저 지우고(사용처→정의 순), 그다음 빈 root(reinterpret_cast/cast)를 지운다.
    for (Operation *op : dead)
      op->erase();
    for (Operation *op : roots)
      if (op->use_empty())
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
