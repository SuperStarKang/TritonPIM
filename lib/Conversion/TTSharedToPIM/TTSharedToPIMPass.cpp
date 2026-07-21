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

// alloc이 두 개 이상의 copy 대상(=subview)으로 채워지면 wrap-around 로드다(두 view + 두 copy가
// 한 버퍼로). 이 경우 하나의 pim.access로 온전히 표현할 수 없다(§8 "추후 지원"). ⑤ masked walk가
// 두 copy를 각각 매치해 공유 alloc/fill/to_tensor를 이중 erase하면 double free가 나므로,
// 여기서 감지해 변환을 건너뛰고 관용구를 그대로 둔다(크래시·오변환 모두 방지).
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

    // ⑤ mask 로드: reinterpret_cast + alloc + (fill?) + subview×2 + copy → pim.access{valid, pad?}+load.
    // 판별: copy 양쪽이 subview. valid = subview 크기. 채움값(other) 있으면 fill에서 pad, 없으면 valid만
    // (fill 없는 마스크 로드: vec_add/reduce/layernorm 등. pim.access는 pad 없이 valid만도 허용).
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
        // §8 wrap-around: 단일 pim.access로 표현 불가. 조용히 두면 target에서 깨지므로 여기서 실패시킨다.
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
      SmallVector<Value> valid = toIndices(b, loc, vsub.getMixedSizes());
      SmallVector<Value> strides = toIndices(b, loc, rc.getMixedStrides());
      Value offset = toIndex(b, loc, rc.getMixedOffsets()[0]);
      Value pad = fill ? fill.getInputs()[0] : Value(); // 채움값 있으면 pad, 없으면 널
      Value d = makeAccess(b, loc, rc.getSource(), offset, strides,
                           rc.getStaticSizes(), valid, pad, Value(), IntegerAttr());
      Value v = b.create<pim::LoadOp>(loc, toTensor->getResult(0).getType(), d);
      toTensor->getResult(0).replaceAllUsesWith(v);
      dead.append({toTensor, copy.getOperation(), vsub.getOperation(),
                   bsub.getOperation(), alloc.getOperation()});
      // 채움이 있을 때만 fill(scf.if로 감싸였으면 그 if째)을 죽은 목록에 넣는다.
      if (fill) {
        Operation *fillDead = isa<scf::IfOp>(fill->getParentOp())
                                  ? fill->getParentOp()
                                  : fill.getOperation();
        dead.push_back(fillDead);
      }
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

    // 스칼라 저장: affine.store %scalar, %reinterpret_cast[0] (sizes [1]) → pim.access + pim.store.
    // reduce/layernorm의 mean/rstd 등 한 칸짜리 결과. 스칼라를 tensor<1xT>로 감싸 pim.store로 넘긴다.
    // affine dialect 링크 의존을 피하려 op 이름으로 확인한다. index는 크기 1이라 항상 0이므로 무시.
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "affine.store")
        return;
      auto rc = op->getOperand(1).getDefiningOp<memref::ReinterpretCastOp>();
      if (!rc)
        return;
      ArrayRef<int64_t> sizes = rc.getStaticSizes();
      if (sizes.size() != 1 || sizes[0] != 1)
        return; // 한 칸(크기 1) 저장만 다룬다.
      Value scalar = op->getOperand(0);
      Value acc = getAccess(rc); // rc 위치에 pim.access 생성(offset/strides/sizes)
      b.setInsertionPoint(op);
      Value tv = b.create<tensor::FromElementsOp>(
          op->getLoc(), RankedTensorType::get({1}, scalar.getType()),
          ValueRange{scalar});
      b.create<pim::StoreOp>(op->getLoc(), tv, acc);
      dead.push_back(op);
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
    // erased 집합으로 이중 erase를 막는다(같은 op이 여러 walk에서 dead/roots/access에 겹쳐 들어올 수
    // 있음 — 이미 지운 포인터엔 use_empty()도 부르지 않는다).
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
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTTSharedToPIMPass() {
  return std::make_unique<ConvertTTSharedToPIMPass>();
}
