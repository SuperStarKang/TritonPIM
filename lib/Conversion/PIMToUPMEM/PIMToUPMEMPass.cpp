#include "triton-shared/Conversion/PIMToUPMEM/PIMToUPMEM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
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
// 값은 device 좌표와 무관하므로 그대로 쓴다. map에 없는 scope 안의 block argument를
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

// v 가 정의 사슬로 block argument arg 에 닿는가. 어느 place 가 round 창 축에 매여
// 있는지(그 축 좌표가 영역 식에 들어가는지)를 가르는 판정이다.
static bool dependsOnArg(Value v, Value arg) {
  SmallVector<Value> work{v};
  llvm::DenseSet<Value> seen;
  while (!work.empty()) {
    Value c = work.pop_back_val();
    if (!seen.insert(c).second)
      continue;
    if (c == arg)
      return true;
    if (Operation *def = c.getDefiningOp())
      work.append(def->getOperands().begin(), def->getOperands().end());
  }
  return false;
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

// 축 하나에 대응하는 loop와, 그 loop가 device 몫으로 좁혀지며 생긴 값들.
struct Axis {
  StringAttr name;
  scf::ForOp loop;
  Value trips;              // 이 device가 실제로 도는 횟수 (loop 상한)
  arith::AddIOp deviceOrigin; // 전역 좌표를 복원하던 op
};

// MRAM 버퍼 하나. pim.access의 id로 찾는다.
struct Buffer {
  FlatSymbolRefAttr mram;
  // capacity[1..rank-1]. device가 배정받은 축별 크기다. host가 넣어 준다.
  SmallVector<FlatSymbolRefAttr> pitch;
  Type elem;
  // 이 버퍼를 읽고 쓰는 타일의 shape. device 버퍼가 이 단위로 쓰여 있다.
  SmallVector<int64_t> tile;
};

struct ConvertPIMToUPMEMPass
    : public mlir::triton::impl::ConvertPIMToUPMEMBase<ConvertPIMToUPMEMPass> {
  using ConvertPIMToUPMEMBase::ConvertPIMToUPMEMBase;

  void runOnOperation() override {
    profile = pim::targetProfile(getOperation());
    if (!profile) {
      getOperation()->emitError(
          "convert-pim-to-upmem: the module has no #pim.target_profile");
      return signalPassFailure();
    }
    if (failed(checkLegality()))
      return signalPassFailure();

    // round(시간 창) 옵션. 값이 없으면 창 1개 — 배정 전체가 한 번에 상주(기존 동작).
    roundAxis = StringRef();
    roundWindow = 0;
    if (!roundTiles.empty()) {
      auto [name, count] = StringRef(roundTiles).split(':');
      if (count.empty() || count.getAsInteger(10, roundWindow) ||
          roundWindow < 1) {
        getOperation()->emitError(
            "convert-pim-to-upmem: round-tiles must be <axis>:<tiles>, got \"")
            << roundTiles << "\"";
        return signalPassFailure();
      }
      if (name == "explicit_k_partition") {
        getOperation()->emitError(
            "convert-pim-to-upmem: rounds along explicit_k_partition are not "
            "supported — partial sums would have to be reloaded across "
            "rounds");
        return signalPassFailure();
      }
      roundAxis = name;
    }

    SmallVector<pim::DeviceOp> devices;
    getOperation().walk([&](pim::DeviceOp d) { devices.push_back(d); });
    for (pim::DeviceOp d : devices)
      convert(d);

    // profile은 여기서 소비된다. 남겨 두면 target IR이 여전히 PIM을 가리키게 되고,
    // 뒤 단계가 기계 값을 다시 읽을 수 있다는 오해를 만든다.
    getOperation()->removeAttr("pim.target_profile");
    getOperation().walk(
        [](func::FuncOp f) { f->removeAttr("pim.group_execution"); });
  }

  pim::TargetProfileAttr profile;
  StringRef roundAxis;   // 창 축 이름. 비어 있으면 round 없음.
  int64_t roundWindow;   // 창 하나의 tile 수 (창 축 방향)

  // UPMEM으로 내릴 수 있는 profile인지 진입 즉시 확인한다.
  LogicalResult checkLegality() {
    if (profile.getTarget() != "upmem")
      return getOperation()->emitError(
                 "convert-pim-to-upmem: the module targets ")
             << profile.getTarget().getValue();
    if (profile.getParallelExecutionKind() != "multithread")
      return getOperation()->emitError(
          "convert-pim-to-upmem: UPMEM tasklets are independent threads, so "
          "parallel_execution_kind must be \"multithread\"");
    if (!profile.getUsableBankCapacityBytes())
      return getOperation()->emitError(
          "convert-pim-to-upmem: usable_bank_capacity_bytes is not recorded, so "
          "the MRAM buffer size cannot be decided");
    return success();
  }

  void convert(pim::DeviceOp device) {
    auto func = device->getParentOfType<func::FuncOp>();
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    Location loc = device.getLoc();
    Block *deviceBody = &device.getBody().front();

    auto cores =
        func->getAttrOfType<pim::GroupExecutionAttr>("pim.group_execution");
    if (!cores) {
      func->emitError("convert-pim-to-upmem: the function has no "
                      "#pim.group_execution");
      return signalPassFailure();
    }
    int64_t lanes = cores.getLanes();
    // 협력 lane 모드: lane들이 tile 하나를 출력 행으로 나눠 함께 계산한다. 버퍼는 한 벌,
    // 적재는 분담, 경계는 barrier다. lane이 하나면 나눌 것이 없어 기존 경로와 같다.
    bool rowSplit =
        lanes > 1 && cores.getLaneAxisRole().getValue() == "output_row";

    // ── 입력 모으기 ────────────────────────────────────────────────────────
    SmallVector<pim::PlaceOp> inputs, outputs;
    for (Operation &op : *deviceBody)
      if (auto p = dyn_cast<pim::PlaceOp>(&op))
        (p.getDirection() == "out" ? outputs : inputs).push_back(p);

    scf::ForOp nest;
    for (Operation &op : *deviceBody)
      if (auto f = dyn_cast<scf::ForOp>(&op)) {
        nest = f;
        break;
      }
    if (!nest) {
      device->emitError("convert-pim-to-upmem: no loop nest inside pim.device");
      return signalPassFailure();
    }

    SmallVector<Axis> axes;
    if (failed(collectAxes(device, axes)))
      return signalPassFailure();

    int64_t numUnits = 1;
    for (int64_t c : device.getMapping().getPartitions().asArrayRef())
      numUnits *= c;

    // ── round(시간 창) 문맥 ────────────────────────────────────────────────
    // 창 축에 매인 place(B·C)는 round 마다 창을 재전송하고, 매이지 않은 place(A)는
    // 한 번만 보낸다. 매였는지는 영역 식이 창 축 좌표 인자에 닿는가로 가른다.
    int64_t roundAxisIdx = -1;
    llvm::DenseMap<Operation *, int> windowAxisOf; // place → 창에 매인 buffer 축
    if (roundWindow) {
      for (auto [i, a] : llvm::enumerate(device.getMapping().getAxes()))
        if (cast<StringAttr>(a).getValue() == roundAxis)
          roundAxisIdx = i;
      if (roundAxisIdx < 0) {
        device->emitError("convert-pim-to-upmem: round axis ")
            << roundAxis << " is not an axis of this kernel";
        return signalPassFailure();
      }
      Value warg = deviceBody->getArgument(roundAxisIdx);
      SmallVector<pim::PlaceOp> places(inputs);
      places.append(outputs);
      for (pim::PlaceOp place : places) {
        int found = -1;
        for (unsigned d = 0, e = place.getCapacity().size(); d < e; d++) {
          if (!dependsOnArg(place.getStart()[d], warg) &&
              !dependsOnArg(place.getSize()[d], warg) &&
              !dependsOnArg(place.getCapacity()[d], warg))
            continue;
          if (found >= 0) {
            place->emitError(
                "convert-pim-to-upmem: two buffer axes follow the round axis, "
                "so a round window is not a single rectangle");
            return signalPassFailure();
          }
          found = d;
        }
        if (place.getDirection() == "out" && found < 0) {
          place->emitError(
              "convert-pim-to-upmem: the output does not follow the round "
              "axis, so every round would rewrite the same region");
          return signalPassFailure();
        }
        windowAxisOf[place] = found;
      }
    }
    auto windowAxis = [&](pim::PlaceOp p) -> int {
      auto it = windowAxisOf.find(p);
      return it == windowAxisOf.end() ? -1 : it->second;
    };

    // ── 모듈 수준 선언 ─────────────────────────────────────────────────────
    OpBuilder mb(module.getBody(), module.getBody()->begin());
    StringRef base = func.getName();
    SmallVector<pim::PlaceOp> all(inputs);
    all.append(outputs);
    int64_t bufferBytes = profile.getUsableBankCapacityBytes().getInt() /
                          std::max<int64_t>(1, all.size());

    // 타일 shape. device 버퍼를 이 단위로 쓰면 타일 하나가 MRAM에서 연속이 되어 DMA가
    // 한 번으로 끝난다. pim.access가 정적으로 들고 있다.
    llvm::DenseMap<int64_t, SmallVector<int64_t>> tileOf;
    nest.walk([&](pim::AccessOp a) {
      tileOf[a.getId()] = SmallVector<int64_t>(a.getSizes());
    });

    llvm::DenseMap<Operation *, FlatSymbolRefAttr> mramOf;
    llvm::DenseMap<int64_t, Buffer> byAccess;
    for (pim::PlaceOp place : all) {
      int64_t id = place.getAccess();
      auto name = StringAttr::get(ctx, (base + "_mram_" + Twine(id)).str());
      mramOf[place] = FlatSymbolRefAttr::get(name);
      Buffer info{FlatSymbolRefAttr::get(name),
                  {},
                  cast<BaseMemRefType>(place.getBuffer().getType()).getElementType(),
                  tileOf[id]};
      if (info.tile.size() != place.getCapacity().size()) {
        place->emitError("no pim.access with this id inside pim.device");
        return signalPassFailure();
      }
      if (failed(checkCapacity(place, info.elem, bufferBytes,
                               windowAxis(place), roundWindow, info.tile)))
        return signalPassFailure();
      mb.create<upmem::MRAMOp>(loc, name,
                               TypeAttr::get(mramType(info.elem, bufferBytes)));
      // 축 하나를 건너뛰는 간격. device가 MRAM 안의 위치를 계산하는 데 쓴다.
      for (unsigned d = 1, e = place.getCapacity().size(); d < e; d++) {
        auto pn = StringAttr::get(
            ctx, (base + "_pitch_" + Twine(id) + "_" + Twine(d)).str());
        info.pitch.push_back(FlatSymbolRefAttr::get(pn));
        mb.create<upmem::HostVarOp>(loc, pn, TypeAttr::get(IndexType::get(ctx)));
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

    // ── device: loop nest를 옮기고 device 로컬로 만든다 ────────────────────
    auto devName = StringAttr::get(ctx, (base + "_device").str());
    auto dev = mb.create<upmem::DeviceFuncOp>(loc, devName);
    // 실제로 쓰는 tasklet 수. launcher가 이 값을 -DNR_TASKLETS로 쓴다.
    dev->setAttr("upmem.lanes", mb.getI64IntegerAttr(lanes));
    Block *devBody = mb.createBlock(&dev.getBody());
    nest->moveBefore(devBody, devBody->end());
    OpBuilder db(devBody, devBody->begin());
    for (auto [i, a] : llvm::enumerate(axes)) {
      // 상한은 host가 넣어 주는 값에서 온다.
      Value n = db.create<upmem::GetHostVarOp>(
          loc, a.loop.getUpperBound().getType(), tilesVar[i]);
      a.loop.getUpperBoundMutable().assign(n);
      // device의 시작을 0으로 놓으면 주소 계산이 device 안의 위치를 낸다.
      if (a.deviceOrigin) {
        Type t = a.deviceOrigin.getResult().getType();
        a.deviceOrigin.getLhsMutable().assign(
            db.create<arith::ConstantOp>(loc, db.getIntegerAttr(t, 0))
                .getResult());
      }
    }

    // ── host: func 몸체를 UPMEM 호출로 바꾼다 ──────────────────────────────
    func->setAttr("upmem.host_threads", mb.getI64IntegerAttr(hostThreads));
    OpBuilder hb(device);
    Value set = hb.create<upmem::AcquireOp>(loc, upmem::SetType::get(ctx),
                                            hb.getI64IntegerAttr(numUnits),
                                            FlatSymbolRefAttr::get(devName));

    ArrayAttr axisNames = device.getMapping().getAxes();
    auto counts =
        hb.getDenseI64ArrayAttr(device.getMapping().getPartitions().asArrayRef());
    auto foreach = hb.create<upmem::ForeachUnitOp>(loc, set, axisNames, counts);
    SmallVector<Type> argTys{upmem::UnitType::get(ctx)};
    SmallVector<Location> argLocs{loc};
    for (unsigned i = 0, e = axisNames.size(); i < e; i++) {
      argTys.push_back(hb.getIndexType());
      argLocs.push_back(loc);
    }
    Block *fuBody = hb.createBlock(&foreach.getBody(), {}, argTys, argLocs);
    // 좌표 계산을 통째로 옮기고 block argument를 새 것으로 잇는다.
    fuBody->getOperations().splice(fuBody->end(), deviceBody->getOperations(),
                                   deviceBody->begin(), deviceBody->end());
    for (unsigned i = 0, e = axisNames.size(); i < e; i++)
      deviceBody->getArgument(i).replaceAllUsesWith(fuBody->getArgument(i + 1));

    Value dpu = fuBody->getArgument(0);
    for (pim::PlaceOp place : inputs) {
      if (windowAxis(place) >= 0)
        continue; // 창에 매인 입력은 round 마다 창을 다시 모은다
      OpBuilder b(place);
      Value staged = b.create<upmem::StageOp>(
          loc, stagedType(place), place.getBuffer(), place.getStart(),
          place.getSize(), place.getStrides(), place.getCapacity(),
          b.getDenseI64ArrayAttr(byAccess[place.getAccess()].tile),
          place.getReplicatedOverAttr());
      b.create<upmem::ToMRAMOp>(loc, dpu, staged, mramOf[place]);
    }
    // device마다 다른 반복 횟수를 그 DPU에 보낸다. 창 축의 횟수는 round 마다 다르므로
    // round loop 안에서 보낸다.
    OpBuilder tb(fuBody, fuBody->end());
    for (auto [i, a] : llvm::enumerate(axes))
      if ((int64_t)i != roundAxisIdx)
        tb.create<upmem::ToHostVarOp>(loc, dpu, a.trips, tilesVar[i]);
    // MRAM 안의 축 간격도 함께 보낸다. 창에 매인 축의 간격은 창 크기다 — 그 buffer 는
    // round 마다 창 하나만 상주하기 때문이다.
    for (pim::PlaceOp place : all) {
      const Buffer &info = byAccess[place.getAccess()];
      int wd = windowAxis(place);
      for (auto [d, sym] : llvm::enumerate(info.pitch)) {
        Value v = place.getCapacity()[d + 1];
        if ((int)(d + 1) == wd)
          v = tb.create<arith::ConstantIndexOp>(loc,
                                                roundWindow * info.tile[d + 1]);
        tb.create<upmem::ToHostVarOp>(loc, dpu, v, sym);
      }
    }

    hb.setInsertionPointAfter(foreach);
    if (!roundWindow) {
      hb.create<upmem::LaunchOp>(loc, set);
      for (pim::PlaceOp place : outputs) {
        // buildOutput이 region을 만들며 삽입 지점을 옮기므로 매번 되돌린다.
        OpBuilder::InsertionGuard guard(hb);
        if (failed(buildOutput(hb, place, set, numUnits, axisNames, counts,
                               mramOf[place], foreach,
                               byAccess[place.getAccess()].tile,
                               /*windowAxis=*/-1, /*round=*/Value(),
                               /*windowTiles=*/0)))
          return signalPassFailure();
      }
    } else {
      // round 수 = ceildiv(창 축 배정량, 창). 배정량(capacity)은 전 device 공통이라
      // round 수도 공통이고, 덜 맡은 device 는 꼬리 round 에서 0 회를 돈다.
      Value c0 = hb.create<arith::ConstantIndexOp>(loc, 0);
      Value c1 = hb.create<arith::ConstantIndexOp>(loc, 1);
      Value w = hb.create<arith::ConstantIndexOp>(loc, roundWindow);
      Value parts = hb.create<arith::ConstantIndexOp>(
          loc, device.getMapping().getPartitions().asArrayRef()[roundAxisIdx]);
      Value capTiles = hb.create<arith::CeilDivSIOp>(
          loc, device.getAxisExtents()[roundAxisIdx], parts);
      Value rounds = hb.create<arith::CeilDivSIOp>(loc, capTiles, w);
      auto roundLoop = hb.create<scf::ForOp>(loc, c0, rounds, c1);
      Value round = roundLoop.getInductionVar();
      OpBuilder lb(roundLoop.getBody(),
                   std::prev(roundLoop.getBody()->end()));

      // 창 foreach: 창에 매인 입력의 재배치와 창 축 반복 횟수. 좌표 산술은 첫 foreach
      // 의 식을 복제해 새 좌표 인자에 잇는다.
      auto fu2 = lb.create<upmem::ForeachUnitOp>(loc, set, axisNames, counts);
      Block *fu2Body;
      {
        OpBuilder blockBuilder(ctx);
        fu2Body = blockBuilder.createBlock(&fu2.getBody(), fu2.getBody().end(),
                                           argTys, argLocs);
      }
      IRMapping toRound;
      for (unsigned i = 0, e = fu2Body->getNumArguments(); i < e; i++)
        toRound.map(fuBody->getArgument(i), fu2Body->getArgument(i));
      Value dpu2 = fu2Body->getArgument(0);
      OpBuilder fb(fu2Body, fu2Body->end());
      Value zero = fb.create<arith::ConstantIndexOp>(loc, 0);
      // 이 round 의 창 축 반복 횟수 = clamp(trips − round·창, 0, 창).
      Value tripsW = cloneInto(fb, axes[roundAxisIdx].trips, toRound, foreach);
      if (!tripsW) {
        device->emitError("convert-pim-to-upmem: the round axis trip count "
                          "cannot be rebuilt in the round loop");
        return signalPassFailure();
      }
      Value done = fb.create<arith::MulIOp>(loc, round, w);
      Value left = fb.create<arith::SubIOp>(loc, tripsW, done);
      Value winTrips = fb.create<arith::MinSIOp>(
          loc, fb.create<arith::MaxSIOp>(loc, left, zero), w);
      fb.create<upmem::ToHostVarOp>(loc, dpu2, winTrips,
                                    tilesVar[roundAxisIdx]);
      for (pim::PlaceOp place : inputs) {
        int wd = windowAxis(place);
        if (wd < 0)
          continue;
        const Buffer &info = byAccess[place.getAccess()];
        Value winElems = fb.create<arith::ConstantIndexOp>(
            loc, roundWindow * info.tile[wd]);
        Value shift = fb.create<arith::MulIOp>(loc, round, winElems);
        SmallVector<Value> start, size, strides, capacity;
        for (unsigned d = 0, e = place.getCapacity().size(); d < e; d++) {
          Value o = cloneInto(fb, place.getStart()[d], toRound, foreach);
          Value s = cloneInto(fb, place.getSize()[d], toRound, foreach);
          Value st = cloneInto(fb, place.getStrides()[d], toRound, foreach);
          Value cp = cloneInto(fb, place.getCapacity()[d], toRound, foreach);
          if (!o || !s || !st || !cp) {
            place->emitError("convert-pim-to-upmem: the window region cannot "
                             "be rebuilt in the round loop");
            return signalPassFailure();
          }
          if ((int)d == wd) {
            o = fb.create<arith::AddIOp>(loc, o, shift);
            s = fb.create<arith::MinSIOp>(
                loc,
                fb.create<arith::MaxSIOp>(
                    loc, fb.create<arith::SubIOp>(loc, s, shift), zero),
                winElems);
            cp = winElems;
          }
          start.push_back(o);
          size.push_back(s);
          strides.push_back(st);
          capacity.push_back(cp);
        }
        Value staged = fb.create<upmem::StageOp>(
            loc, stagedType(place), place.getBuffer(), start, size, strides,
            capacity, fb.getDenseI64ArrayAttr(info.tile),
            place.getReplicatedOverAttr());
        fb.create<upmem::ToMRAMOp>(loc, dpu2, staged, mramOf[place]);
      }

      lb.create<upmem::LaunchOp>(loc, set);
      for (pim::PlaceOp place : outputs) {
        OpBuilder::InsertionGuard guard(lb);
        if (failed(buildOutput(lb, place, set, numUnits, axisNames, counts,
                               mramOf[place], foreach,
                               byAccess[place.getAccess()].tile,
                               windowAxis(place), round, roundWindow)))
          return signalPassFailure();
      }
    }

    // pim.place는 여기서 한 번에 지운다. 위의 stage/to_host_var가 이 op의
    // start·size·capacity를 계속 읽으므로 그 전에 지우면 해제된 op을 읽게 된다.
    for (pim::PlaceOp place : all)
      place.erase();

    device.erase();
    if (failed(convertDevice(dev, byAccess, lanes, rowSplit, tb, dpu, base)))
      return signalPassFailure();
  }

  // 축 목록. device mapping의 axes 순서대로 loop를 찾고, 그 loop가 device 몫으로
  // 좁혀지며 생긴 상한과 전역 좌표 복원 op을 함께 모은다.
  LogicalResult collectAxes(pim::DeviceOp device, SmallVectorImpl<Axis> &axes) {
    for (Attribute a : device.getMapping().getAxes()) {
      auto name = cast<StringAttr>(a);
      scf::ForOp loop;
      device.getBody().walk([&](scf::ForOp l) {
        if (l->getAttrOfType<StringAttr>("pim.axis") == name)
          loop = l;
      });
      if (!loop)
        return device->emitError("no loop for axis ") << name;
      // 이 loop의 몸체에 직접 있는 것만 본다. 안쪽 loop에도 같은 표시가 있으므로
      // walk로 찾으면 가장 안쪽 것이 잡힌다.
      arith::AddIOp origin;
      for (Operation &op : *loop.getBody())
        if (auto add = dyn_cast<arith::AddIOp>(&op))
          if (add->hasAttr("pim.device_origin")) {
            origin = add;
            break;
          }
      axes.push_back({name, loop, loop.getUpperBound(), origin});
    }
    return success();
  }

  // UPMEM은 bank 예산을 버퍼 개수로 균등 분할한다. 총합만 보면 개별 버퍼가 넘치는
  // 구성이 통과하므로 버퍼마다 본다. 크기가 런타임이면 여기서 증명할 수 없고, host
  // 코드가 전송 직전에 검사한다. 창에 매인 축은 round 마다 창 하나만 상주한다.
  LogicalResult checkCapacity(pim::PlaceOp place, Type elem, int64_t bufferBytes,
                              int windowAxis, int64_t windowTiles,
                              ArrayRef<int64_t> tile) {
    int64_t elems = 1;
    for (auto [d, c] : llvm::enumerate(place.getCapacity())) {
      if ((int)d == windowAxis) {
        elems *= windowTiles * tile[d];
        continue;
      }
      std::optional<int64_t> v = getConstantIntValue(c);
      if (!v)
        return success();
      elems *= *v;
    }
    int64_t bytes = elems * std::max<int64_t>(1, elem.getIntOrFloatBitWidth() / 8);
    if (bytes > bufferBytes)
      return place->emitError("convert-pim-to-upmem: this buffer needs ")
             << bytes << " byte of MRAM but the per-buffer share is "
             << bufferBytes << " byte";
    return success();
  }

  // MRAM 버퍼 타입. capacity가 런타임 값이라 정확히 잡을 수 없고 device의
  // __mram_noinit은 상수를 요구하므로, bank 예산을 버퍼 수로 나눈 상한을 쓴다.
  MemRefType mramType(Type elem, int64_t bytes) {
    int64_t elemBytes = std::max<int64_t>(1, elem.getIntOrFloatBitWidth() / 8);
    return MemRefType::get(
        {bytes / elemBytes}, elem, MemRefLayoutAttrInterface(),
        upmem::MemorySpaceAttr::get(&getContext(), upmem::MemorySpace::MRAM));
  }

  // staging 버퍼 타입. capacity가 런타임 값이라 shape은 전부 동적이다.
  MemRefType stagedType(pim::PlaceOp place) {
    auto elem =
        cast<BaseMemRefType>(place.getBuffer().getType()).getElementType();
    SmallVector<int64_t> shape(place.getCapacity().size(), ShapedType::kDynamic);
    return MemRefType::get(shape, elem);
  }

  //===--------------------------------------------------------------------===//
  // device 쪽
  //===--------------------------------------------------------------------===//

  // device 메모리 위로 내린다.
  //  - tile은 WRAM 버퍼가 되고 계산은 그 위에서 memref 형태 linalg로 돈다.
  //  - pim.load/store는 타일 하나를 옮기는 DMA가 된다.
  //  - pim.access는 MRAM 안의 위치를 정하는 식이 되고 op 자체는 사라진다.
  LogicalResult convertDevice(upmem::DeviceFuncOp dev,
                              const llvm::DenseMap<int64_t, Buffer> &byAccess,
                              int64_t lanes, bool rowSplit, OpBuilder &tb,
                              Value dpu, StringRef base) {
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
    // whole_tile: tasklet마다 다른 tile을 맡으므로 tile 버퍼를 그 수만큼 복제하고 자기
    // 번호로 잘라 쓴다 — 공유하면 서로 덮어쓴다. rowSplit(output_row): 전원이 같은 tile을
    // 계산하므로 버퍼는 한 벌이고, 겹쳐 쓰지 않도록 계산과 저장을 출력 행으로 가른다.
    Value taskletId;
    if (lanes > 1)
      taskletId = top.create<upmem::TaskletIdOp>(loc, top.getIndexType());
    auto allocFor = [&](RankedTensorType t) -> Value {
      SmallVector<int64_t> shape(t.getShape());
      if (lanes > 1 && !rowSplit)
        shape.insert(shape.begin(), lanes);
      Value all = top.create<memref::AllocOp>(
          loc, MemRefType::get(shape, t.getElementType(),
                               MemRefLayoutAttrInterface(), wram));
      if (lanes == 1 || rowSplit)
        return all;
      SmallVector<OpFoldResult> offsets(shape.size(), top.getIndexAttr(0));
      offsets[0] = taskletId;
      SmallVector<OpFoldResult> sizes, strides(shape.size(), top.getIndexAttr(1));
      for (int64_t s : shape)
        sizes.push_back(top.getIndexAttr(s));
      sizes[0] = top.getIndexAttr(1);
      MemRefType mine = memref::SubViewOp::inferRankReducedResultType(
          t.getShape(), cast<MemRefType>(all.getType()), offsets, sizes, strides);
      return top.create<memref::SubViewOp>(loc, mine, all, offsets, sizes,
                                           strides);
    };

    // tensor 값 → WRAM 버퍼 대응. 버퍼를 새로 잡을 자리는 공통 규칙이 정한다.
    SmallVector<Value> roots;
    llvm::DenseMap<Value, Value> owner;
    pim::collectLocalTiles(dev, roots, owner);
    llvm::DenseMap<Value, Value> buf;
    for (Value r : roots)
      buf[r] = allocFor(cast<RankedTensorType>(r.getType()));
    auto bufferOf = [&](Value v) -> Value {
      Value root = owner.lookup(v);
      return root ? buf.lookup(root) : Value();
    };

    // 협력 모드의 출력 행 slice. 버퍼마다 한 번만 만들어 재사용한다. 행 수가 lane 수로
    // 나누어떨어져야 slice가 정적 shape이 된다 — 아니면 진단을 낸다(자투리 행을 특정
    // tasklet에 몰아주는 동적 shape은 codegen이 다루지 않는다).
    llvm::DenseMap<Value, Value> rowSlices;
    auto rowSliceOf = [&](Value bufv, Operation *at) -> Value {
      Value &slot = rowSlices[bufv];
      if (slot)
        return slot;
      auto t = dyn_cast<MemRefType>(bufv.getType());
      if (!t) {
        at->emitError("convert-pim-to-upmem: expected a WRAM tile buffer "
                      "operand when splitting by output rows");
        return Value();
      }
      int64_t rows = t.getShape()[0];
      if (rows % lanes != 0) {
        at->emitError("convert-pim-to-upmem: the tile has ")
            << rows << " output rows, which " << lanes
            << " cooperating tasklets cannot split evenly; pick a lane count "
               "that divides the row count";
        return Value();
      }
      int64_t rowsPerLane = rows / lanes;
      Value offset0 = top.create<arith::MulIOp>(
          loc, taskletId,
          top.create<arith::ConstantIndexOp>(loc, rowsPerLane));
      SmallVector<OpFoldResult> offsets(t.getRank(), top.getIndexAttr(0));
      offsets[0] = offset0;
      SmallVector<OpFoldResult> sizes, strides(t.getRank(), top.getIndexAttr(1));
      for (int64_t s : t.getShape())
        sizes.push_back(top.getIndexAttr(s));
      sizes[0] = top.getIndexAttr(rowsPerLane);
      SmallVector<int64_t> shape(t.getShape());
      shape[0] = rowsPerLane;
      MemRefType mine = memref::SubViewOp::inferRankReducedResultType(
          shape, t, offsets, sizes, strides);
      slot = top.create<memref::SubViewOp>(loc, mine, bufv, offsets, sizes,
                                           strides);
      return slot;
    };

    // linalg를 memref 형태로 다시 만든다(결과가 없어지므로 op을 새로 만들어야 한다).
    // 옛 op은 아직 지우지 않는다 — 그 결과를 scf.yield가 아직 쓰고 있다.
    SmallVector<Operation *> oldCompute;
    WalkResult computeRes = dev.walk([&](linalg::LinalgOp linalgOp) {
      Operation *op = linalgOp;
      if (op->getNumResults() != 1 || !bufferOf(op->getResult(0)))
        return WalkResult::advance();
      OpBuilder b(op);
      SmallVector<Value> operands;
      for (Value v : op->getOperands())
        operands.push_back(bufferOf(v) ? bufferOf(v) : v);
      if (rowSplit) {
        // 전원이 같은 tile을 계산하므로 각자 자기 출력 행만 맡는다. 어느 피연산자를
        // 가르는지는 op마다 정해져 있다 — matmul은 A와 누산기의 행이 출력 행과 같은
        // 축이고 B는 전 행이 필요하다. 그 밖의 계산은 가르는 규칙이 없으므로 진단이다.
        if (isa<linalg::FillOp>(op)) {
          if (!(operands[1] = rowSliceOf(operands[1], op)))
            return WalkResult::interrupt();
        } else if (isa<linalg::MatmulOp>(op)) {
          if (!(operands[0] = rowSliceOf(operands[0], op)))
            return WalkResult::interrupt();
          if (!(operands[2] = rowSliceOf(operands[2], op)))
            return WalkResult::interrupt();
          // 공유 피연산자(B)의 적재가 전부 끝난 뒤에만 계산을 시작한다.
          b.create<upmem::BarrierOp>(op->getLoc());
        } else {
          op->emitError(
              "convert-pim-to-upmem: cooperating tasklets "
              "(lane_axis_role = output_row) can split only matmul and fill "
              "by output rows");
          return WalkResult::interrupt();
        }
      }
      OperationState st(op->getLoc(), op->getName());
      st.addOperands(operands);
      st.addAttributes(op->getAttrs());
      IRMapping map;
      for (Region &r : op->getRegions())
        r.cloneInto(st.addRegion(), map);
      b.create(st);
      // 다음 반복의 적재가 공유 피연산자를 덮어쓰기 전에 전원의 소비가 끝나야 한다.
      if (rowSplit && isa<linalg::MatmulOp>(op))
        b.create<upmem::BarrierOp>(op->getLoc());
      oldCompute.push_back(op);
      return WalkResult::advance();
    });
    if (computeRes.wasInterrupted())
      return failure();

    // 전송을 DMA로 내린다. 협력 모드에서는 chunk를 tasklet이 나눠 옮긴다.
    SmallVector<Operation *> oldLoads, oldStores;
    dev.walk([&](Operation *op) {
      if (auto load = dyn_cast<pim::LoadOp>(op)) {
        buildDMA(load, load.getAccess(), bufferOf(load.getResult()), byAccess,
                 pitchOf, /*isRead=*/true, rowSplit, taskletId, lanes);
        oldLoads.push_back(op);
      } else if (auto store = dyn_cast<pim::StoreOp>(op)) {
        // 협력 모드의 저장도 chunk 분담이라 다른 tasklet의 행을 읽는다. 앞 barrier는
        // 전 행의 계산이 끝났음을(matmul이 없는 적재→저장 커널의 경로도 여기서 닫힌다),
        // 뒤 barrier는 다음 tile의 fill이 누산기를 덮기 전에 저장이 끝났음을 보장한다.
        if (rowSplit) {
          OpBuilder b(store);
          b.create<upmem::BarrierOp>(store.getLoc());
        }
        buildDMA(store, store.getAccess(), bufferOf(store.getValue()), byAccess,
                 pitchOf, /*isRead=*/false, rowSplit, taskletId, lanes);
        if (rowSplit) {
          OpBuilder b(store);
          b.create<upmem::BarrierOp>(store.getLoc());
        }
        oldStores.push_back(op);
      }
    });

    SmallVector<scf::ForOp> loops;
    dev.walk([&](scf::ForOp l) { loops.push_back(l); });

    // 소비하는 쪽부터 지운다. 순서를 지키지 않으면 아직 쓰이는 값을 지우게 된다.
    for (Operation *op : oldStores)
      op->erase();
    for (scf::ForOp l : loops)
      if (l.getNumRegionIterArgs() > 0)
        dropIterArgs(l);
    for (Operation *op : llvm::reverse(oldCompute))
      op->erase();
    for (Operation *op : oldLoads)
      op->erase();

    // 남은 pim.access와 죽은 tensor.empty 정리.
    SmallVector<Operation *> leftover;
    dev.walk([&](Operation *op) {
      if (isa<pim::AccessOp, tensor::EmptyOp>(op) && op->use_empty())
        leftover.push_back(op);
    });
    for (Operation *op : llvm::reverse(leftover))
      op->erase();

    // pim.access가 사라지면서 원본 버퍼의 전역 간격을 쓰던 주소 계산이 죽는다.
    // device는 자기 안의 간격만 쓰므로 그 값들은 host 쪽에 남아 있고, 죽은 계산을 두면
    // device 몸체가 host 값을 참조해 지배 관계가 깨진다.
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

    // whole_tile만 tile을 tasklet에 나눈다. 협력 모드는 전원이 같은 tile 순서를 돌며
    // 안에서 행과 chunk를 나눴다.
    if (lanes > 1 && !rowSplit)
      distributeTasklets(dev);

    // 옮겨 온 loop nest는 loop 경계와 fill 값 같은 상수를 host func 안의 것으로
    // 가리킨다. device_func은 그 func의 형제라 그 값들이 보이지 않으므로 다시 만든다.
    if (failed(materializeExternals(dev, tb, dpu, base)))
      return failure();

    WalkResult remain = dev.walk([&](Operation *op) {
      return isa<pim::PIMDialect>(op->getDialect()) ? WalkResult::interrupt()
                                                    : WalkResult::advance();
    });
    if (remain.wasInterrupted())
      return dev->emitError("device body still contains pim ops");
    return success();
  }

  // device가 맡은 타일을 tasklet에 나눈다. 병렬 축 loop들을 하나로 합친 뒤 tasklet
  // 번호부터 tasklet 수만큼 건너뛰며 돈다. 타일마다 쓰는 자리가 겹치지 않으므로
  // barrier가 없다. 합치는 이유는 loop 하나만 나누면 그 loop의 반복이 tasklet 수보다
  // 적을 때 나머지 tasklet이 놀기 때문이다.
  void distributeTasklets(upmem::DeviceFuncOp dev) {
    SmallVector<scf::ForOp> axisLoops;
    dev.walk([&](scf::ForOp l) {
      if (l->hasAttr("pim.axis"))
        axisLoops.push_back(l);
    });
    if (axisLoops.empty())
      return;
    // walk는 안쪽을 먼저 담는다. 바깥부터 안쪽 순서로 되돌린다.
    std::reverse(axisLoops.begin(), axisLoops.end());

    OpBuilder b(axisLoops.front());
    Location loc = axisLoops.front().getLoc();
    Value total = axisLoops.front().getUpperBound();
    for (unsigned i = 1, e = axisLoops.size(); i < e; i++)
      total = b.create<arith::MulIOp>(loc, total, axisLoops[i].getUpperBound());
    Value tid = b.create<upmem::TaskletIdOp>(loc, b.getIndexType());
    Value taskletCount = b.create<upmem::NumTaskletsOp>(loc, b.getIndexType());
    auto flat = b.create<scf::ForOp>(loc, tid, total, taskletCount);

    // 선형 번호를 축별 좌표로 푼다. 안쪽 축부터 나머지와 몫으로 벗겨 낸다.
    OpBuilder fb(flat.getBody(), flat.getBody()->begin());
    Value rest = flat.getInductionVar();
    for (int i = axisLoops.size() - 1; i >= 0; i--) {
      Value ub = axisLoops[i].getUpperBound();
      axisLoops[i].getInductionVar().replaceAllUsesWith(
          fb.create<arith::RemSIOp>(loc, rest, ub));
      rest = fb.create<arith::DivSIOp>(loc, rest, ub);
    }

    // 몸체를 바깥부터 이어 붙인다.
    Block *to = flat.getBody();
    for (scf::ForOp l : axisLoops) {
      Block *from = l.getBody();
      to->getOperations().splice(Block::iterator(to->getTerminator()),
                                 from->getOperations(), from->begin(),
                                 Block::iterator(from->getTerminator()));
    }
    for (scf::ForOp l : axisLoops)
      l.erase();
  }

  // 옮겨 온 loop nest가 host func 안의 값을 가리키면 device 자리에 다시 만든다. 상수처럼
  // 뿌리까지 값이 이어지면 복제하고, 커널 인자에서 나오는 수(reduction loop 반복 횟수 등)는
  // device 안에서 만들 수 없으므로 host 변수로 통로를 낸다.
  LogicalResult materializeExternals(upmem::DeviceFuncOp dev, OpBuilder &tb,
                                     Value dpu, StringRef base) {
    Block *body = &dev.getBody().front();
    OpBuilder top(body, body->begin());
    // 선언은 module 바로 아래에 둔다. host/device가 이름으로 공유하는 심볼이다.
    Block *moduleBody = getOperation().getBody();
    OpBuilder mb(moduleBody, moduleBody->begin());
    Location loc = dev.getLoc();
    llvm::DenseMap<Value, Value> made, viaHost;
    unsigned next = 0;

    WalkResult res = dev.walk([&](Operation *op) {
      for (OpOperand &use : op->getOpOperands()) {
        if (Value v = rebuildIn(dev, top, use.get(), made)) {
          use.set(v);
          continue;
        }
        Value host = use.get();
        if (!host.getType().isIntOrIndex()) {
          op->emitError("device body uses a host value it cannot rebuild");
          return WalkResult::interrupt();
        }
        Value &slot = viaHost[host];
        if (!slot) {
          auto name = StringAttr::get(&getContext(),
                                      (base + "_host_" + Twine(next++)).str());
          mb.create<upmem::HostVarOp>(loc, name, TypeAttr::get(host.getType()));
          auto sym = FlatSymbolRefAttr::get(name);
          tb.create<upmem::ToHostVarOp>(loc, dpu, host, sym);
          slot = top.create<upmem::GetHostVarOp>(loc, host.getType(), sym);
        }
        use.set(slot);
      }
      return WalkResult::advance();
    });
    return failure(res.wasInterrupted());
  }

  // 타일 하나를 옮긴다. upmem.stage가 device 버퍼를 타일 단위로 썼으므로 타일 하나가
  // MRAM에서 연속이다. 타일이 2048 byte를 넘을 때만 행 묶음으로 쪼갠다. MRAM 안의
  // 타일 시작은 축별 타일 번호에서 나온다:
  //   번호[d] = origin[d] / tile[d],  축별 타일 수[d] = capacity[d] / tile[d]
  //   시작 = ((번호[0] * 수[1] + 번호[1]) * 수[2] + ...) * 타일 원소 수
  // 타일 축은 원본 버퍼의 rank라 WRAM 버퍼보다 클 수 있다 — 앞쪽 축의 크기가 1이면
  // 그 축은 자리 계산에만 나타난다(split-K의 출력면이 그 경우다).
  template <typename PitchFn>
  void buildDMA(Operation *xfer, Value accessValue, Value wbuf,
                const llvm::DenseMap<int64_t, Buffer> &byAccess, PitchFn pitchOf,
                bool isRead, bool distribute, Value taskletId, int64_t lanes) {
    auto access = accessValue.getDefiningOp<pim::AccessOp>();
    const Buffer &info = byAccess.at(access.getId());
    ArrayRef<int64_t> tile = info.tile;
    unsigned rank = tile.size();
    auto wtype = cast<MemRefType>(wbuf.getType());
    ArrayRef<int64_t> shape = wtype.getShape();
    unsigned wrank = wtype.getRank();

    int64_t elemBytes = std::max<int64_t>(
        1, wtype.getElementType().getIntOrFloatBitWidth() / 8);
    int64_t rowElems = 1;
    for (unsigned d = 1; d < wrank; d++)
      rowElems *= shape[d];
    int64_t tileElems = 1;
    for (int64_t t : tile)
      tileElems *= t;
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

    // 협력 모드에서는 chunk를 tasklet 번호부터 tasklet 수만큼 건너뛰며 나눠 옮긴다.
    // 서로 다른 chunk라 겹쳐 쓰지 않고, 적재/소비의 경계는 호출자가 barrier로 닫는다.
    Value lb = zero, step = one;
    if (distribute) {
      lb = taskletId;
      step = b.create<arith::ConstantIndexOp>(loc, lanes);
    }
    auto chunkLoop = b.create<scf::ForOp>(loc, lb, nchunks, step);
    OpBuilder rb(chunkLoop.getBody(), chunkLoop.getBody()->begin());
    Value c = chunkLoop.getInductionVar();

    auto constant = [&](int64_t v) {
      return rb.create<arith::ConstantIndexOp>(loc, v).getResult();
    };
    Value off = rb.create<arith::DivSIOp>(loc, access.getOrigin()[0],
                                          constant(tile[0]));
    for (unsigned d = 1; d < rank; d++) {
      Value count = rb.create<arith::DivSIOp>(loc, pitchOf(info.pitch[d - 1]),
                                              constant(tile[d]));
      off = rb.create<arith::AddIOp>(
          loc, rb.create<arith::MulIOp>(loc, off, count),
          rb.create<arith::DivSIOp>(loc, access.getOrigin()[d],
                                    constant(tile[d])));
    }
    off = rb.create<arith::AddIOp>(
        loc, rb.create<arith::MulIOp>(loc, off, constant(tileElems)),
        rb.create<arith::MulIOp>(loc, c, constant(perDMA * rowElems)));

    // WRAM 쪽도 이어진 행 묶음이라 rank를 줄이지 않는다.
    SmallVector<OpFoldResult> offsets(wrank, rb.getIndexAttr(0));
    offsets[0] = rb.create<arith::MulIOp>(loc, c, constant(perDMA)).getResult();
    SmallVector<OpFoldResult> sizes, strides(wrank, rb.getIndexAttr(1));
    for (int64_t s : shape)
      sizes.push_back(rb.getIndexAttr(s));
    sizes[0] = rb.getIndexAttr(perDMA);
    SmallVector<int64_t> partShape(shape);
    partShape[0] = perDMA;
    MemRefType partType = memref::SubViewOp::inferRankReducedResultType(
        partShape, wtype, offsets, sizes, strides);
    Value part =
        rb.create<memref::SubViewOp>(loc, partType, wbuf, offsets, sizes, strides);
    if (isRead)
      rb.create<upmem::MRAMReadOp>(loc, info.mram, off, part);
    else
      rb.create<upmem::MRAMWriteOp>(loc, part, info.mram, off);
  }

  // tensor iter_arg를 없앤 loop로 바꾼다. 버퍼가 loop를 관통하므로 값을 나르지 않는다.
  void dropIterArgs(scf::ForOp loop) {
    OpBuilder b(loop);
    auto fresh = b.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(),
                                      loop.getUpperBound(), loop.getStep());
    if (Attribute axis = loop->getAttr("pim.axis"))
      fresh->setAttr("pim.axis", axis);
    Block *from = loop.getBody(), *to = fresh.getBody();
    loop.getInductionVar().replaceAllUsesWith(fresh.getInductionVar());
    to->getOperations().splice(to->begin(), from->getOperations(), from->begin(),
                               Block::iterator(from->getTerminator()));
    // 옛 계산이 몸체와 함께 밖으로 나왔는데 아직 iter_arg를 쓴다. 그 자리를 처음 값으로
    // 돌려놓아야 옛 loop를 지울 수 있다.
    for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs()))
      arg.replaceAllUsesWith(loop.getInitArgs()[i]);
    loop.erase();
  }

  // 출력 하나: 전 device에서 회수 → 그룹마다 원본으로 되돌리기. round 가 있으면
  // 창에 매인 축의 회수량은 창 크기이고, 되돌릴 자리는 round 만큼 밀린다.
  LogicalResult buildOutput(OpBuilder &b, pim::PlaceOp place, Value set,
                            int64_t numUnits, ArrayAttr axes,
                            DenseI64ArrayAttr counts, FlatSymbolRefAttr mram,
                            Operation *foreach, ArrayRef<int64_t> tile,
                            int windowAxis, Value round, int64_t windowTiles) {
    Location loc = place.getLoc();
    unsigned rank = place.getCapacity().size();
    auto elem =
        cast<BaseMemRefType>(place.getBuffer().getType()).getElementType();

    // 회수할 바이트 수는 배정량에서 나온다. device 좌표와 무관하므로 foreach 밖으로
    // 옮겨 만들 수 있다 — 못 옮기면 device마다 크기가 다르다는 뜻이라 회수가 성립하지 않는다.
    IRMapping empty;
    SmallVector<Value> capacity;
    for (auto [d, c] : llvm::enumerate(place.getCapacity())) {
      if ((int)d == windowAxis) {
        capacity.push_back(
            b.create<arith::ConstantIndexOp>(loc, windowTiles * tile[d]));
        continue;
      }
      Value v = cloneInto(b, c, empty, foreach);
      if (!v)
        return place->emitError("output capacity depends on the device "
                                "coordinate; every device must hand back the "
                                "same size");
      capacity.push_back(v);
    }

    SmallVector<int64_t> partsShape{numUnits};
    partsShape.append(rank, ShapedType::kDynamic);
    Value parts = b.create<upmem::GatherOp>(
        loc, MemRefType::get(partsShape, elem), set, mram, capacity);

    auto group = b.create<upmem::ForeachGroupOp>(loc, parts, axes, counts);
    SmallVector<int64_t> partShape(rank, ShapedType::kDynamic);
    SmallVector<Type> argTys{MemRefType::get(partShape, elem)};
    SmallVector<Location> argLocs{loc};
    for (size_t i = 0, e = axes.size(); i < e; i++) {
      argTys.push_back(b.getIndexType());
      argLocs.push_back(loc);
    }
    Block *gb = b.createBlock(&group.getBody(), {}, argTys, argLocs);

    // 영역은 그룹 좌표로 다시 만든다.
    IRMapping map;
    Block &unitBlock = cast<upmem::ForeachUnitOp>(foreach).getBody().front();
    for (unsigned d = 0, e = axes.size(); d < e; d++)
      map.map(unitBlock.getArgument(d + 1), gb->getArgument(d + 1));

    OpBuilder gbuilder(gb, gb->end());
    Value zero, winElems, shift;
    if (windowAxis >= 0) {
      zero = gbuilder.create<arith::ConstantIndexOp>(loc, 0);
      winElems = gbuilder.create<arith::ConstantIndexOp>(
          loc, windowTiles * tile[windowAxis]);
      shift = gbuilder.create<arith::MulIOp>(loc, round, winElems);
    }
    SmallVector<Value> start, size, strides;
    for (unsigned d = 0; d < rank; d++) {
      // 좌표 계산은 이미 foreach_unit 몸체로 옮겨졌으므로 그 범위를 기준으로 본다.
      Value o = cloneInto(gbuilder, place.getStart()[d], map, foreach);
      Value s = cloneInto(gbuilder, place.getSize()[d], map, foreach);
      Value t = cloneInto(gbuilder, place.getStrides()[d], map, foreach);
      if (!o || !s || !t)
        return place->emitError(
            "output region cannot be rebuilt from group coordinates");
      if ((int)d == windowAxis) {
        o = gbuilder.create<arith::AddIOp>(loc, o, shift);
        s = gbuilder.create<arith::MinSIOp>(
            loc,
            gbuilder.create<arith::MaxSIOp>(
                loc, gbuilder.create<arith::SubIOp>(loc, s, shift), zero),
            winElems);
      }
      start.push_back(o);
      size.push_back(s);
      strides.push_back(t);
    }
    gbuilder.create<upmem::UnstageOp>(loc, gb->getArgument(0), place.getBuffer(),
                                      start, size, strides, capacity,
                                      gbuilder.getDenseI64ArrayAttr(tile));
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertPIMToUPMEMPass() {
  return std::make_unique<ConvertPIMToUPMEMPass>();
}
