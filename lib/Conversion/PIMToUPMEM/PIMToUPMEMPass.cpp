#include "triton-shared/Conversion/PIMToUPMEM/PIMToUPMEM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTPIMTOUPMEM
#include "triton-shared/Conversion/PIMToUPMEM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// v를 계산하는 식을 map대로 치환해 지금 삽입 지점에 다시 만든다. scope 밖에서 정의된
// 값은 유닛 좌표와 무관하므로 그대로 쓴다. map에 없는 scope 안의 block argument를
// 만나면 널 — 그 값은 이 자리에서 만들 수 없다.
static Value cloneInto(OpBuilder &b, Value v, IRMapping &map, Operation *scope) {
  if (Value m = map.lookupOrNull(v))
    return m;
  Operation *def = v.getDefiningOp();
  if (!def)
    return scope->isAncestor(v.getParentBlock()->getParentOp()) ? Value() : v;
  if (!scope->isAncestor(def))
    return v;
  IRMapping local;
  for (Value o : def->getOperands()) {
    Value r = cloneInto(b, o, map, scope);
    if (!r)
      return Value();
    local.map(o, r);
  }
  Value r = b.clone(*def, local)->getResult(cast<OpResult>(v).getResultNumber());
  map.map(v, r);
  return r;
}

// v가 scope 안에서 정의되었는가.
static bool definedIn(Operation *scope, Value v) {
  if (Operation *def = v.getDefiningOp())
    return scope->isAncestor(def);
  return scope->isAncestor(v.getParentBlock()->getParentOp());
}

// 밖에서 정의된 값을 top 자리에 다시 만든다. 상수처럼 뿌리까지 값이 이어지면 되고,
// func 인자에 닿으면 그 자리에서 만들 수 없으므로 널을 낸다.
static Value rebuildIn(Operation *scope, OpBuilder &top, Value v,
                       llvm::DenseMap<Value, Value> &made) {
  if (definedIn(scope, v))
    return v;
  auto it = made.find(v);
  if (it != made.end())
    return it->second;
  Operation *def = v.getDefiningOp();
  if (!def)
    return Value();
  IRMapping local;
  for (Value o : def->getOperands()) {
    Value r = rebuildIn(scope, top, o, made);
    if (!r)
      return Value();
    local.map(o, r);
  }
  Value r =
      top.clone(*def, local)->getResult(cast<OpResult>(v).getResultNumber());
  made[v] = r;
  return r;
}

// 축 하나에 대응하는 loop와, 그 loop가 유닛 몫으로 좁혀지며 생긴 값들.
struct Axis {
  StringAttr name;
  scf::ForOp loop;
  Value trips;             // 이 유닛이 실제로 도는 횟수 (loop 상한)
  arith::AddIOp unitOrigin; // 전역 인덱스를 복원하던 op (pim.unit_origin)
};

// MRAM 버퍼 하나. pim.access_id로 찾는다.
struct Buffer {
  FlatSymbolRefAttr mram;
  // capacity[1..rank-1]. 유닛이 배정받은 축별 크기다. host가 넣어 준다.
  SmallVector<FlatSymbolRefAttr> pitch;
  Type elem;
  // 이 버퍼를 읽고 쓰는 타일의 shape. 유닛 버퍼가 이 단위로 쓰여 있다.
  SmallVector<int64_t> tile;
};

struct ConvertPIMToUPMEMPass
    : public mlir::triton::impl::ConvertPIMToUPMEMBase<ConvertPIMToUPMEMPass> {
  using ConvertPIMToUPMEMBase::ConvertPIMToUPMEMBase;

  void runOnOperation() override {
    SmallVector<pim::UnitsOp> units;
    getOperation().walk([&](pim::UnitsOp u) { units.push_back(u); });
    for (pim::UnitsOp u : units)
      convert(u);
  }

  void convert(pim::UnitsOp units) {
    auto func = units->getParentOfType<func::FuncOp>();
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    Location loc = units.getLoc();
    Block *unitsBody = &units.getBody().front();

    // ── 입력 모으기 ────────────────────────────────────────────────────────
    SmallVector<pim::UnitDataOp> inputs, outputs;
    for (Operation &op : *unitsBody)
      if (auto ud = dyn_cast<pim::UnitDataOp>(&op))
        (ud.getDirection() == "out" ? outputs : inputs).push_back(ud);

    scf::ForOp nest;
    for (Operation &op : *unitsBody)
      if (auto f = dyn_cast<scf::ForOp>(&op)) {
        nest = f;
        break;
      }
    if (!nest) {
      units->emitError("convert-pim-to-upmem: no loop nest inside pim.units");
      return signalPassFailure();
    }

    SmallVector<Axis> axes;
    if (failed(collectAxes(units, nest, axes)))
      return signalPassFailure();

    int64_t numUnits = 1;
    for (int64_t c : units.getCounts())
      numUnits *= c;

    // ── 모듈 수준 선언 ─────────────────────────────────────────────────────
    OpBuilder mb(module.getBody(), module.getBody()->begin());
    StringRef base = func.getName();
    int64_t bufferBytes = unitMemoryBytes / std::max<int64_t>(
                              1, inputs.size() + outputs.size());

    // 타일 shape. 유닛 버퍼를 이 단위로 쓰면 타일 하나가 MRAM 에서 연속이 되어 DMA 가
    // 한 번으로 끝난다. pim.access 가 정적으로 들고 있다.
    llvm::DenseMap<int64_t, SmallVector<int64_t>> tileOf;
    nest.walk([&](pim::AccessOp a) {
      auto id = a->getAttrOfType<IntegerAttr>("pim.access_id");
      tileOf[id.getInt()] = SmallVector<int64_t>(a.getSizes());
    });

    llvm::DenseMap<Operation *, FlatSymbolRefAttr> mramOf;
    llvm::DenseMap<int64_t, Buffer> byAccess;
    SmallVector<pim::UnitDataOp> all(inputs);
    all.append(outputs);
    for (pim::UnitDataOp ud : all) {
      int64_t id = ud.getAccess();
      auto name = StringAttr::get(ctx, (base + "_mram_" + Twine(id)).str());
      mramOf[ud] = FlatSymbolRefAttr::get(name);
      mb.create<upmem::MRAMOp>(loc, name,
                               TypeAttr::get(mramType(ud, bufferBytes)));
      Buffer info{FlatSymbolRefAttr::get(name),
                  {},
                  cast<BaseMemRefType>(ud.getBase().getType()).getElementType(),
                  tileOf[id]};
      if (info.tile.size() != ud.getCapacity().size()) {
        ud->emitError("no pim.access with this id inside pim.units");
        return signalPassFailure();
      }
      // 축 하나를 건너뛰는 간격. device가 MRAM 안의 위치를 계산하는 데 쓴다.
      for (unsigned d = 1, e = ud.getCapacity().size(); d < e; d++) {
        auto pn = StringAttr::get(
            ctx, (base + "_pitch_" + Twine(id) + "_" + Twine(d)).str());
        info.pitch.push_back(FlatSymbolRefAttr::get(pn));
        mb.create<upmem::HostVarOp>(loc, pn,
                                    TypeAttr::get(IndexType::get(ctx)));
      }
      byAccess[id] = info;
    }
    SmallVector<FlatSymbolRefAttr> tilesVar;
    for (const Axis &a : axes) {
      auto name =
          StringAttr::get(ctx, (base + "_tiles_" + a.name.getValue()).str());
      tilesVar.push_back(FlatSymbolRefAttr::get(name));
      mb.create<upmem::HostVarOp>(loc, name,
                                  TypeAttr::get(IndexType::get(ctx)));
    }

    // ── device: loop nest를 옮기고 유닛 로컬로 만든다 ──────────────────────
    auto devName = StringAttr::get(ctx, (base + "_device").str());
    auto dev = mb.create<upmem::DeviceFuncOp>(loc, devName);
    Block *devBody = mb.createBlock(&dev.getBody());
    nest->moveBefore(devBody, devBody->end());
    OpBuilder db(devBody, devBody->begin());
    for (auto [i, a] : llvm::enumerate(axes)) {
      // 상한은 host가 넣어 주는 값에서 온다.
      Value n = db.create<upmem::GetHostVarOp>(
          loc, a.loop.getUpperBound().getType(), tilesVar[i]);
      a.loop.getUpperBoundMutable().assign(n);
      // 유닛의 시작을 0으로 놓으면 주소 계산이 유닛 안의 위치를 낸다.
      if (a.unitOrigin) {
        Type t = a.unitOrigin.getResult().getType();
        a.unitOrigin.getLhsMutable().assign(
            db.create<arith::ConstantOp>(loc, db.getIntegerAttr(t, 0))
                .getResult());
      }
    }

    // ── host: func 몸체를 UPMEM 호출로 바꾼다 ──────────────────────────────
    // 영역을 모으고 되돌리는 일을 몇 갈래로 나눌지. codegen이 이 값을 읽는다.
    func->setAttr("upmem.host_threads", mb.getI64IntegerAttr(hostThreads));
    OpBuilder hb(units);
    Value set = hb.create<upmem::AcquireOp>(loc, upmem::SetType::get(ctx),
                                            hb.getI64IntegerAttr(numUnits),
                                            FlatSymbolRefAttr::get(devName));

    auto foreach = hb.create<upmem::ForeachUnitOp>(loc, set, units.getAxes(),
                                                   units.getCountsAttr());
    SmallVector<Type> argTys{upmem::UnitType::get(ctx)};
    SmallVector<Location> argLocs{loc};
    for (unsigned i = 0, e = units.getAxes().size(); i < e; i++) {
      argTys.push_back(hb.getIndexType());
      argLocs.push_back(loc);
    }
    Block *fuBody = hb.createBlock(&foreach.getBody(), {}, argTys, argLocs);
    // 좌표 계산을 통째로 옮기고 block argument를 새 것으로 잇는다.
    fuBody->getOperations().splice(fuBody->end(), unitsBody->getOperations(),
                                   unitsBody->begin(), unitsBody->end());
    for (unsigned i = 0, e = units.getAxes().size(); i < e; i++)
      unitsBody->getArgument(i).replaceAllUsesWith(fuBody->getArgument(i + 1));

    Value dpu = fuBody->getArgument(0);
    for (pim::UnitDataOp ud : inputs) {
      OpBuilder b(ud);
      Value staged = b.create<upmem::StageOp>(
          loc, stagedType(ud), ud.getBase(), ud.getOrigin(), ud.getSizes(),
          ud.getStrides(), ud.getCapacity(),
          b.getDenseI64ArrayAttr(byAccess[ud.getAccess()].tile),
          ud.getReplicatedOverAttr());
      b.create<upmem::ToMRAMOp>(loc, dpu, staged, mramOf[ud]);
    }
    // 유닛마다 다른 반복 횟수를 그 DPU에 보낸다.
    OpBuilder tb(fuBody, fuBody->end());
    for (auto [i, a] : llvm::enumerate(axes))
      tb.create<upmem::ToHostVarOp>(loc, dpu, a.trips, tilesVar[i]);
    // MRAM 안의 축 간격도 함께 보낸다. capacity는 유닛마다 같지만 이 자리에서
    // 계산되므로 같은 통로를 쓴다.
    for (pim::UnitDataOp ud : all) {
      const Buffer &info = byAccess[ud.getAccess()];
      for (auto [d, sym] : llvm::enumerate(info.pitch))
        tb.create<upmem::ToHostVarOp>(loc, dpu, ud.getCapacity()[d + 1], sym);
    }

    hb.setInsertionPointAfter(foreach);
    hb.create<upmem::LaunchOp>(loc, set);
    for (pim::UnitDataOp ud : outputs) {
      // buildOutput이 region을 만들며 삽입 지점을 옮기므로 매번 되돌린다.
      OpBuilder::InsertionGuard guard(hb);
      if (failed(buildOutput(hb, ud, set, numUnits, units, mramOf[ud],
                             foreach.getBody().front(), foreach,
                             byAccess[ud.getAccess()].tile)))
        return signalPassFailure();
    }

    // pim.unit_data 는 여기서 한 번에 지운다. 위의 stage/to_host_var 가 이 op 의
    // origin·sizes·capacity 를 계속 읽으므로 그 전에 지우면 해제된 op 을 읽게 된다.
    for (pim::UnitDataOp ud : all)
      ud.erase();

    units.erase();
    if (failed(convertDevice(dev, byAccess)))
      return signalPassFailure();
  }

  // 축 목록. pim.units의 axes 순서대로 loop를 찾고, 그 loop가 유닛 몫으로 좁혀지며
  // 생긴 상한과 전역 인덱스 복원 op을 함께 모은다.
  LogicalResult collectAxes(pim::UnitsOp units, scf::ForOp nest,
                            SmallVectorImpl<Axis> &axes) {
    SmallVector<scf::ForOp> reduction;
    units.getBody().walk([&](scf::ForOp l) {
      if (l->hasAttr("pim.reduction_combine"))
        reduction.push_back(l);
    });
    unsigned seen = 0;
    for (Attribute a : units.getAxes()) {
      auto name = cast<StringAttr>(a);
      scf::ForOp loop;
      units.getBody().walk([&](scf::ForOp l) {
        auto g = l->getAttrOfType<StringAttr>("pim.grid_dim");
        if (g && g == name)
          loop = l;
      });
      if (!loop && seen < reduction.size())
        loop = reduction[seen++];
      if (!loop)
        return units->emitError("no loop for split axis ") << name;
      // 이 loop의 몸체에 **직접** 있는 것만 본다. 안쪽 loop에도 같은 표시가 있으므로
      // walk로 찾으면 가장 안쪽 것이 잡힌다.
      arith::AddIOp origin;
      for (Operation &op : *loop.getBody())
        if (auto add = dyn_cast<arith::AddIOp>(&op))
          if (add->hasAttr("pim.unit_origin")) {
            origin = add;
            break;
          }
      axes.push_back({name, loop, loop.getUpperBound(), origin});
    }
    return success();
  }

  // MRAM 버퍼 타입. capacity가 런타임 값이라 정확히 잡을 수 없고 device의
  // __mram_noinit은 상수를 요구하므로, 유닛 메모리를 버퍼 수로 나눈 상한을 쓴다.
  MemRefType mramType(pim::UnitDataOp ud, int64_t bytes) {
    auto elem = cast<BaseMemRefType>(ud.getBase().getType()).getElementType();
    int64_t elemBytes = std::max<int64_t>(1, elem.getIntOrFloatBitWidth() / 8);
    int64_t elems = bytes / elemBytes;
    return MemRefType::get(
        {elems}, elem, MemRefLayoutAttrInterface(),
        upmem::MemorySpaceAttr::get(&getContext(), upmem::MemorySpace::MRAM));
  }

  // staging 버퍼 타입. capacity가 런타임 값이라 shape은 전부 동적이다.
  MemRefType stagedType(pim::UnitDataOp ud) {
    auto elem = cast<BaseMemRefType>(ud.getBase().getType()).getElementType();
    SmallVector<int64_t> shape(ud.getCapacity().size(), ShapedType::kDynamic);
    return MemRefType::get(shape, elem);
  }

  //===--------------------------------------------------------------------===//
  // device 쪽
  //===--------------------------------------------------------------------===//

  // 유닛 메모리 위로 내린다.
  //  - tile 은 WRAM 버퍼가 되고 계산은 그 위에서 memref 형태 linalg 로 돈다.
  //  - pim.load/store 는 행마다 DMA 하나가 된다. MRAM DMA 가 연속 영역만 다루는데
  //    타일의 행끼리는 간격만큼 떨어져 있어 한 번에 옮길 수 없다.
  //  - pim.access 는 MRAM 안의 위치를 정하는 식이 되고 op 자체는 사라진다.
  // 지금은 tasklet 하나를 전제한다(local_lanes = 1). 여러 tasklet 으로 나누는 것은
  // 행 loop 와 계산을 함께 쪼개는 별도 단계다.
  LogicalResult convertDevice(upmem::DeviceFuncOp dev,
                              const llvm::DenseMap<int64_t, Buffer> &byAccess) {
    Block *body = &dev.getBody().front();
    OpBuilder top(body, body->begin());
    Location loc = dev.getLoc();
    auto wram = upmem::MemorySpaceAttr::get(&getContext(), upmem::MemorySpace::WRAM);

    // 축 간격을 한 번씩만 읽어 둔다.
    llvm::DenseMap<StringAttr, Value> pitchValue;
    auto pitchOf = [&](FlatSymbolRefAttr sym) {
      Value &v = pitchValue[sym.getAttr()];
      if (!v)
        v = top.create<upmem::GetHostVarOp>(loc, top.getIndexType(), sym);
      return v;
    };
    // tasklet 이 여럿이면 tile 버퍼를 그 수만큼 복제하고 자기 번호로 잘라 쓴다.
    // tasklet 마다 다른 tile 을 맡으므로 버퍼를 공유하면 서로 덮어쓴다.
    Value taskletId;
    if (localLanes > 1)
      taskletId = top.create<upmem::TaskletIdOp>(loc, top.getIndexType());
    int64_t wramBytes = 0;
    auto allocFor = [&](RankedTensorType t) -> Value {
      SmallVector<int64_t> shape(t.getShape());
      if (localLanes > 1)
        shape.insert(shape.begin(), localLanes);
      int64_t n = 1;
      for (int64_t s : shape)
        n *= s;
      wramBytes +=
          n * std::max<int64_t>(1, t.getElementType().getIntOrFloatBitWidth() / 8);
      Value all = top.create<memref::AllocOp>(
          loc, MemRefType::get(shape, t.getElementType(),
                               MemRefLayoutAttrInterface(), wram));
      if (localLanes == 1)
        return all;
      SmallVector<OpFoldResult> offsets(shape.size(), top.getIndexAttr(0));
      offsets[0] = taskletId;
      SmallVector<OpFoldResult> sizes, strides(shape.size(),
                                               top.getIndexAttr(1));
      for (int64_t s : shape)
        sizes.push_back(top.getIndexAttr(s));
      sizes[0] = top.getIndexAttr(1);
      MemRefType mine = memref::SubViewOp::inferRankReducedResultType(
          t.getShape(), cast<MemRefType>(all.getType()), offsets, sizes,
          strides);
      return top.create<memref::SubViewOp>(loc, mine, all, offsets, sizes,
                                           strides);
    };

    // tensor 값 → WRAM 버퍼 대응. 정의가 사용보다 앞이므로 pre-order 로 한 번에 만든다.
    llvm::DenseMap<Value, Value> buf;
    dev.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto load = dyn_cast<pim::LoadOp>(op)) {
        buf[load.getResult()] = allocFor(cast<RankedTensorType>(load.getType()));
        return;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs()))
          if (Value b = buf.lookup(forOp.getInitArgs()[i])) {
            buf[arg] = b;
            buf[forOp.getResult(i)] = b;
          }
        return;
      }
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!linalgOp || linalgOp.getNumDpsInits() != 1 ||
          op->getNumResults() != 1)
        return;
      Value out = linalgOp.getDpsInits()[0];
      // 누산기에 이어 쓰는 계산이면 그 버퍼를 그대로 쓰고, 새 tile 을 만드는
      // 계산(초기화 등)이면 버퍼를 새로 잡는다.
      Value target = buf.lookup(out);
      if (!target) {
        auto t = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!t)
          return;
        target = allocFor(t);
        buf[out] = target;
      }
      buf[op->getResult(0)] = target;
    });

    // tile 버퍼는 tasklet 마다 하나씩이라 tasklet 수만큼 곱해진다. 여기서 막지 않으면
    // dpu-clang 이 링크 단계에서 wram 구역이 넘쳤다고 수십 줄을 쏟아 낸다. IR 을 고치기
    // 전에 본다 — 반쯤 바뀐 IR 이 오류에 딸려 나오면 읽기 어렵다.
    // pass option 은 그대로 스트림에 넣으면 아무것도 안 찍히므로 값을 꺼내 쓴다.
    int64_t lanes = localLanes, budget = localMemoryBytes;
    if (wramBytes > budget)
      return dev->emitError("tile buffers need ")
             << wramBytes << " byte of unit fast memory (" << lanes
             << " tasklet(s)) but the budget is " << budget
             << " byte; lower local_lanes or the BLOCK sizes";

    // linalg 를 memref 형태로 다시 만든다(결과가 없어지므로 op 을 새로 만들어야 한다).
    // 옛 op 은 아직 지우지 않는다 — 그 결과를 scf.yield 가 아직 쓰고 있다.
    SmallVector<Operation *> oldCompute;
    dev.walk([&](linalg::LinalgOp linalgOp) {
      Operation *op = linalgOp;
      if (op->getNumResults() != 1 || !buf.count(op->getResult(0)))
        return;
      OpBuilder b(op);
      SmallVector<Value> operands;
      for (Value v : op->getOperands())
        operands.push_back(buf.lookup(v) ? buf.lookup(v) : v);
      OperationState st(op->getLoc(), op->getName());
      st.addOperands(operands);
      st.addAttributes(op->getAttrs());
      IRMapping map;
      for (Region &r : op->getRegions())
        r.cloneInto(st.addRegion(), map);
      b.create(st);
      oldCompute.push_back(op);
    });

    // 전송을 DMA 로 내린다.
    SmallVector<Operation *> oldLoads, oldStores;
    dev.walk([&](Operation *op) {
      if (auto load = dyn_cast<pim::LoadOp>(op)) {
        buildDMA(load, load.getAccess(), buf.lookup(load.getResult()), byAccess,
                 pitchOf, /*isRead=*/true);
        oldLoads.push_back(op);
      } else if (auto store = dyn_cast<pim::StoreOp>(op)) {
        buildDMA(store, store.getAccess(), buf.lookup(store.getValue()),
                 byAccess, pitchOf, /*isRead=*/false);
        oldStores.push_back(op);
      }
    });

    SmallVector<scf::ForOp> loops;
    dev.walk([&](scf::ForOp l) { loops.push_back(l); });

    // 소비하는 쪽부터 지운다. 순서를 지키지 않으면 아직 쓰이는 값을 지우게 되고,
    // 그러면 남은 사용처가 해제된 op 을 가리킨다.
    //   store  → loop 결과를 놓아 준다
    //   loop   → scf.yield 가 옛 계산 결과를, init 이 누산기 초기화 결과를 놓아 준다
    //   계산   → iter_arg 대신 놓아 둔 초기값과 pim.load 결과를 놓아 준다.
    //            walk 가 바깥 쪽을 먼저 담으므로 거꾸로 돌아야 쓰는 쪽이 먼저다.
    //   load   → pim.access 를 놓아 준다
    for (Operation *op : oldStores)
      op->erase();
    for (scf::ForOp l : loops)
      if (l.getNumRegionIterArgs() > 0)
        dropIterArgs(l);
    for (Operation *op : llvm::reverse(oldCompute))
      op->erase();
    for (Operation *op : oldLoads)
      op->erase();

    // 남은 pim.access 와 죽은 tensor.empty 정리.
    SmallVector<Operation *> leftover;
    dev.walk([&](Operation *op) {
      if (isa<pim::AccessOp, tensor::EmptyOp>(op) && op->use_empty())
        leftover.push_back(op);
    });
    for (Operation *op : llvm::reverse(leftover))
      op->erase();

    // pim.access가 사라지면서 원본 버퍼의 전역 간격(stride)을 쓰던 주소 계산이 죽는다.
    // device는 유닛 안의 간격만 쓰므로 그 값들은 host 쪽에 남아 있고, 죽은 계산을 두면
    // device 몸체가 host 값을 참조해 지배 관계가 깨진다. 더 지울 것이 없을 때까지 돈다.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> unused;
      dev.walk([&](Operation *op) {
        if (op->getNumRegions() == 0 && !op->hasTrait<OpTrait::IsTerminator>() &&
            op->use_empty() && isMemoryEffectFree(op))
          unused.push_back(op);
      });
      for (Operation *op : unused) {
        op->erase();
        changed = true;
      }
    }

    if (localLanes > 1)
      distributeTasklets(dev);

    // 옮겨 온 loop nest 는 loop 경계와 fill 값 같은 상수를 host func 안의 것으로
    // 가리킨다. device_func 은 그 func 의 형제라 그 값들이 보이지 않으므로 다시 만든다.
    if (failed(materializeExternals(dev)))
      return failure();

    WalkResult remain = dev.walk([&](Operation *op) {
      return isa<pim::PIMDialect>(op->getDialect()) ? WalkResult::interrupt()
                                                    : WalkResult::advance();
    });
    if (remain.wasInterrupted())
      return dev->emitError("device body still contains pim ops");
    return success();
  }

  // 유닛이 맡은 타일을 tasklet 에 나눈다. 병렬 grid loop 들을 하나로 합친 뒤 tasklet
  // 번호부터 tasklet 수만큼 건너뛰며 돈다. 타일마다 쓰는 자리가 겹치지 않으므로 barrier 가
  // 없다. loop 하나만 나누지 않는 이유는 그 loop 의 반복이 tasklet 수보다 적으면 나머지
  // tasklet 이 놀기 때문이다 — 유닛이 맡은 타일 수는 합쳐야 제대로 나온다.
  void distributeTasklets(upmem::DeviceFuncOp dev) {
    SmallVector<scf::ForOp> grid;
    dev.walk([&](scf::ForOp l) {
      if (l->hasAttr("pim.grid_dim"))
        grid.push_back(l);
    });
    if (grid.empty())
      return;
    // walk 는 안쪽을 먼저 담는다. 바깥부터 안쪽 순서로 되돌린다.
    std::reverse(grid.begin(), grid.end());

    OpBuilder b(grid.front());
    Location loc = grid.front().getLoc();
    Value total = grid.front().getUpperBound();
    for (unsigned i = 1, e = grid.size(); i < e; i++)
      total = b.create<arith::MulIOp>(loc, total, grid[i].getUpperBound());
    Value tid = b.create<upmem::TaskletIdOp>(loc, b.getIndexType());
    Value lanes = b.create<upmem::NumTaskletsOp>(loc, b.getIndexType());
    auto flat = b.create<scf::ForOp>(loc, tid, total, lanes);

    // 선형 번호를 축별 좌표로 푼다. 안쪽 축부터 나머지와 몫으로 벗겨 낸다.
    OpBuilder fb(flat.getBody(), flat.getBody()->begin());
    Value rest = flat.getInductionVar();
    for (int i = grid.size() - 1; i >= 0; i--) {
      Value ub = grid[i].getUpperBound();
      grid[i].getInductionVar().replaceAllUsesWith(
          fb.create<arith::RemSIOp>(loc, rest, ub));
      rest = fb.create<arith::DivSIOp>(loc, rest, ub);
    }

    // 몸체를 바깥부터 이어 붙인다. 안쪽 loop 는 그대로 딸려 오고, 그 몸체가 다음 차례에
    // 그 뒤로 붙은 다음 빈 loop 만 사라진다.
    Block *to = flat.getBody();
    for (scf::ForOp l : grid) {
      Block *from = l.getBody();
      to->getOperations().splice(Block::iterator(to->getTerminator()),
                                 from->getOperations(), from->begin(),
                                 Block::iterator(from->getTerminator()));
    }
    for (scf::ForOp l : grid)
      l.erase();
  }

  LogicalResult materializeExternals(upmem::DeviceFuncOp dev) {
    Block *body = &dev.getBody().front();
    OpBuilder top(body, body->begin());
    llvm::DenseMap<Value, Value> made;
    WalkResult res = dev.walk([&](Operation *op) {
      for (OpOperand &use : op->getOpOperands()) {
        Value v = rebuildIn(dev, top, use.get(), made);
        if (!v) {
          op->emitError("device body uses a host value it cannot rebuild");
          return WalkResult::interrupt();
        }
        use.set(v);
      }
      return WalkResult::advance();
    });
    return failure(res.wasInterrupted());
  }

  // 타일 하나를 옮긴다. upmem.stage 가 유닛 버퍼를 타일 단위로 썼으므로 타일 하나가
  // MRAM 에서 연속이고, DMA 한 번이면 된다. 타일이 2048 byte 를 넘을 때만 행 묶음으로
  // 쪼갠다. MRAM 안의 타일 시작은 축별 타일 번호에서 나온다:
  //   번호[d] = origin[d] / tile[d],  축별 타일 수[d] = capacity[d] / tile[d]
  //   시작 = ((번호[0] * 수[1] + 번호[1]) * 수[2] + ...) * 타일 원소 수
  template <typename PitchFn>
  void buildDMA(Operation *xfer, Value accessValue, Value wbuf,
                const llvm::DenseMap<int64_t, Buffer> &byAccess, PitchFn pitchOf,
                bool isRead) {
    auto access = accessValue.getDefiningOp<pim::AccessOp>();
    auto id = access->getAttrOfType<IntegerAttr>("pim.access_id");
    const Buffer &info = byAccess.at(id.getInt());
    auto wtype = cast<MemRefType>(wbuf.getType());
    ArrayRef<int64_t> shape = wtype.getShape();
    unsigned rank = wtype.getRank();

    int64_t elemBytes = std::max<int64_t>(
        1, wtype.getElementType().getIntOrFloatBitWidth() / 8);
    int64_t rowElems = 1;
    for (unsigned d = 1; d < rank; d++)
      rowElems *= shape[d];
    // 한 번에 옮길 행 수. 2048 byte 안에 들어가면서 행 수를 나누어떨어지게 하는 가장 큰 값.
    int64_t perDMA = shape[0];
    while (perDMA > 1 &&
           (perDMA * rowElems * elemBytes > 2048 || shape[0] % perDMA))
      perDMA--;
    int64_t chunks = shape[0] / perDMA;

    OpBuilder b(xfer);
    Location loc = xfer->getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value nchunks = b.create<arith::ConstantIndexOp>(loc, chunks);

    auto chunkLoop = b.create<scf::ForOp>(loc, zero, nchunks, one);
    OpBuilder rb(chunkLoop.getBody(), chunkLoop.getBody()->begin());
    Value c = chunkLoop.getInductionVar();

    auto constant = [&](int64_t v) {
      return rb.create<arith::ConstantIndexOp>(loc, v).getResult();
    };
    Value off = rb.create<arith::DivSIOp>(loc, access.getOrigin()[0],
                                          constant(shape[0]));
    for (unsigned d = 1; d < rank; d++) {
      Value count = rb.create<arith::DivSIOp>(loc, pitchOf(info.pitch[d - 1]),
                                              constant(shape[d]));
      off = rb.create<arith::AddIOp>(
          loc, rb.create<arith::MulIOp>(loc, off, count),
          rb.create<arith::DivSIOp>(loc, access.getOrigin()[d],
                                    constant(shape[d])));
    }
    off = rb.create<arith::AddIOp>(
        loc, rb.create<arith::MulIOp>(loc, off, constant(shape[0] * rowElems)),
        rb.create<arith::MulIOp>(loc, c, constant(perDMA * rowElems)));

    // WRAM 쪽도 이어진 행 묶음이라 rank 를 줄이지 않는다.
    SmallVector<OpFoldResult> offsets(rank, rb.getIndexAttr(0));
    offsets[0] = rb.create<arith::MulIOp>(loc, c, constant(perDMA)).getResult();
    SmallVector<OpFoldResult> sizes, strides(rank, rb.getIndexAttr(1));
    for (int64_t s : shape)
      sizes.push_back(rb.getIndexAttr(s));
    sizes[0] = rb.getIndexAttr(perDMA);
    SmallVector<int64_t> partShape(shape);
    partShape[0] = perDMA;
    MemRefType partType = memref::SubViewOp::inferRankReducedResultType(
        partShape, wtype, offsets, sizes, strides);
    Value part = rb.create<memref::SubViewOp>(loc, partType, wbuf, offsets,
                                              sizes, strides);
    if (isRead)
      rb.create<upmem::MRAMReadOp>(loc, info.mram, off, part);
    else
      rb.create<upmem::MRAMWriteOp>(loc, part, info.mram, off);
  }

  // tensor iter_arg 를 없앤 loop 로 바꾼다. 버퍼가 loop 를 관통하므로 값을 나르지 않는다.
  void dropIterArgs(scf::ForOp loop) {
    OpBuilder b(loop);
    auto fresh = b.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(),
                                      loop.getUpperBound(), loop.getStep());
    Block *from = loop.getBody(), *to = fresh.getBody();
    loop.getInductionVar().replaceAllUsesWith(fresh.getInductionVar());
    to->getOperations().splice(to->begin(), from->getOperations(),
                               from->begin(),
                               Block::iterator(from->getTerminator()));
    // 옛 계산이 몸체와 함께 밖으로 나왔는데 아직 iter_arg 를 쓴다. 그 자리를 처음 값으로
    // 돌려놓아야 옛 loop 를 지울 수 있다 — 옛 계산은 바로 뒤에서 함께 사라진다.
    for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs()))
      arg.replaceAllUsesWith(loop.getInitArgs()[i]);
    loop.erase();
  }

  // 출력 하나: 전 유닛에서 회수 → 그룹마다 축약 → 원본으로 되돌리기.
  LogicalResult buildOutput(OpBuilder &b, pim::UnitDataOp ud, Value set,
                            int64_t numUnits, pim::UnitsOp units,
                            FlatSymbolRefAttr mram, Block &unitBlock,
                            Operation *foreach, ArrayRef<int64_t> tile) {
    MLIRContext *ctx = b.getContext();
    Location loc = ud.getLoc();
    unsigned rank = ud.getCapacity().size();
    auto elem = cast<BaseMemRefType>(ud.getBase().getType()).getElementType();

    // 회수할 바이트 수는 배정량에서 나온다. 유닛 좌표와 무관하므로 foreach 밖으로
    // 옮겨 만들 수 있다 — 못 옮기면 유닛마다 크기가 다르다는 뜻이라 회수가 성립하지 않는다.
    IRMapping empty;
    SmallVector<Value> capacity;
    for (Value c : ud.getCapacity()) {
      Value v = cloneInto(b, c, empty, foreach);
      if (!v)
        return ud->emitError("output capacity depends on the unit coordinate; "
                             "every unit must hand back the same size");
      capacity.push_back(v);
    }

    SmallVector<int64_t> partsShape{numUnits};
    partsShape.append(rank, ShapedType::kDynamic);
    Value parts = b.create<upmem::GatherOp>(
        loc, MemRefType::get(partsShape, elem), set, mram, capacity);

    // 결합 축은 그룹 축에서 빠진다. 그 축만 다른 유닛들이 한 그룹이다.
    SmallVector<Attribute> groupAxes;
    SmallVector<int64_t> groupCounts;
    int64_t reduceCount = 1;
    StringAttr reduceAxis;
    ArrayAttr over = ud.getCombineOverAttr();
    for (auto [d, a] : llvm::enumerate(units.getAxes())) {
      bool isReduce = over && llvm::is_contained(over, a);
      if (isReduce) {
        reduceAxis = cast<StringAttr>(a);
        reduceCount *= units.getCounts()[d];
        continue;
      }
      groupAxes.push_back(a);
      groupCounts.push_back(units.getCounts()[d]);
    }

    auto group = b.create<upmem::ForeachGroupOp>(
        loc, parts, ArrayAttr::get(ctx, groupAxes),
        b.getDenseI64ArrayAttr(groupCounts), reduceAxis,
        b.getI64IntegerAttr(reduceCount), ud.getCombineAttr());
    SmallVector<int64_t> partShape(rank, ShapedType::kDynamic);
    SmallVector<Type> argTys{MemRefType::get(partShape, elem)};
    SmallVector<Location> argLocs{loc};
    for (size_t i = 0; i < groupAxes.size(); i++) {
      argTys.push_back(b.getIndexType());
      argLocs.push_back(loc);
    }
    Block *gb = b.createBlock(&group.getBody(), {}, argTys, argLocs);

    // 영역은 그룹 좌표로 다시 만든다. 결합 축 좌표는 그룹에 없으므로, 출력이 그 축에
    // 의존하면 애초에 결합할 수 없다 — 그 경우 여기서 실패한다.
    IRMapping map;
    unsigned g = 1;
    for (auto [d, a] : llvm::enumerate(units.getAxes()))
      if (!(over && llvm::is_contained(over, a)))
        map.map(unitBlock.getArgument(d + 1), gb->getArgument(g++));

    OpBuilder gbuilder(gb, gb->end());
    SmallVector<Value> origin, sizes, strides;
    for (unsigned d = 0; d < rank; d++) {
      // 좌표 계산은 이미 foreach_unit 몸체로 옮겨졌으므로 그 범위를 기준으로 본다.
      Value o = cloneInto(gbuilder, ud.getOrigin()[d], map, foreach);
      Value s = cloneInto(gbuilder, ud.getSizes()[d], map, foreach);
      Value t = cloneInto(gbuilder, ud.getStrides()[d], map, foreach);
      if (!o || !s || !t)
        return ud->emitError(
            "output region depends on the combined axis; cannot rebuild it "
            "from group coordinates");
      origin.push_back(o);
      sizes.push_back(s);
      strides.push_back(t);
    }
    // capacity는 gather가 회수한 버퍼의 shape 그대로다. 유닛 좌표와 무관해서 위에서
    // foreach 밖으로 옮겨 만들었으므로 여기서 그대로 쓸 수 있다.
    gbuilder.create<upmem::UnstageOp>(loc, gb->getArgument(0), ud.getBase(),
                                      origin, sizes, strides, capacity,
                                      gbuilder.getDenseI64ArrayAttr(tile));
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertPIMToUPMEMPass() {
  return std::make_unique<ConvertPIMToUPMEMPass>();
}
