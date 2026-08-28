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
#include "mlir/IR/AffineExpr.h"
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

// UPMEM의 첫 elementwise 지원 범위. 모든 operand가 같은 logical index를 쓰고 모든
// iterator가 parallel이면 첫 차원(m)을 tasklet별 연속 행으로 안전하게 자를 수 있다.
static bool isIdentityPointwise(linalg::GenericOp gen) {
  if (gen.getOutputs().size() != 1)
    return false;
  for (AffineMap map : gen.getIndexingMapsArray())
    if (!map.isIdentity())
      return false;
  return llvm::all_of(gen.getIteratorTypesArray(), [](utils::IteratorType t) {
    return t == utils::IteratorType::parallel;
  });
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

// #pim.execution_tile의 reduction 크기로 정해지는 temporal K window.
// M/N execution-tile round는 PIM Common IR이 pim.device 바깥에 이미 만들었으므로
// target lowering은 다시 자르지 않는다. K만 launch마다 한 구간을 실행하고 host가
// window 사이의 부분합을 누산한다(unstage accumulate).
struct Window {
  scf::ForOp kLoop;         // temporal reduction loop
  int64_t cap = 0;          // 상한 (원소)
  int64_t elems = 0;        // 창 축 tile 하나의 원소 수
  int64_t tiles = 0;        // 창의 tile 수 = cap / elems
  Value trips;              // 축 전체 반복 횟수 (foreach 몸체 기준 값)
  FlatSymbolRefAttr tilesVar; // 이 축의 반복 횟수 host 변수
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

    SmallVector<pim::DeviceOp> devices;
    getOperation().walk([&](pim::DeviceOp d) { devices.push_back(d); });
    for (pim::DeviceOp d : devices)
      convert(d);

    // 공통 PIM planning 속성은 여기서 모두 소비된다. 남겨 두면 target
    // IR이 여전히 PIM dialect에 의존하고, UPMEM translator가 완전히 lowered된
    // IR만 받는다는 경계가 깨진다.
    getOperation()->removeAttr("pim.target_profile");
    getOperation().walk([](func::FuncOp f) {
      f->removeAttr("pim.block");
      f->removeAttr("pim.block_group");
      f->removeAttr("pim.execution_tile");
    });
    getOperation().walk([](Operation *op) {
      op->removeAttr("pim.axis");
      op->removeAttr("pim.device_origin");
    });
  }

  pim::TargetProfileAttr profile;

  // UPMEM으로 내릴 수 있는 profile인지 진입 즉시 확인한다.
  LogicalResult checkLegality() {
    if (profile.getTarget() != "upmem")
      return getOperation()->emitError(
                 "convert-pim-to-upmem: the module targets ")
             << profile.getTarget().getValue();
    if (profile.getBanksPerUnit() != 1 || profile.getCoresPerBank() != 1)
      return getOperation()->emitError(
          "convert-pim-to-upmem: UPMEM topology must be one MRAM bank and "
          "one DPU core per PIM unit");
    if (profile.getBankCapacityLimitBytes() <= 0)
      return getOperation()->emitError(
          "convert-pim-to-upmem: bank_capacity_limit_bytes must be positive, so "
          "the MRAM buffer size cannot be decided");
    int64_t lanes = nrTasklets == 0 ? profile.getLanesPerCore() : nrTasklets;
    if (lanes < 1 || lanes > profile.getLanesPerCore())
      return getOperation()->emitError(
                 "convert-pim-to-upmem: nr-tasklets must be between 1 and ")
             << profile.getLanesPerCore();
    return success();
  }

  void convert(pim::DeviceOp device) {
    auto func = device->getParentOfType<func::FuncOp>();
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    Location loc = device.getLoc();
    Block *deviceBody = &device.getBody().front();

    int64_t lanes = nrTasklets == 0 ? profile.getLanesPerCore() : nrTasklets;
    if (func->getAttrOfType<pim::BlockGroupAttr>("pim.block_group")) {
      SmallVector<int64_t, 3> shape = pim::groupShape(func);
      if (shape.size() != 3) {
        func->emitError("convert-pim-to-upmem: cannot derive block-group "
                        "shape from #pim.block and #pim.block_group");
        return signalPassFailure();
      }
      int64_t rows = shape[0];
      if (rows % lanes != 0) {
        func->emitError("convert-pim-to-upmem: block-group m extent ")
            << rows << " is not divisible by " << lanes << " tasklets";
        return signalPassFailure();
      }
    }
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

    // 타일 shape. device 버퍼를 이 단위로 쓰면 타일 하나가 MRAM에서 연속이 되어 DMA가
    // 한 번으로 끝난다. pim.access가 정적으로 들고 있다.
    llvm::DenseMap<int64_t, SmallVector<int64_t>> tileOf;
    nest.walk([&](pim::AccessOp a) {
      tileOf[a.getId()] = SmallVector<int64_t>(a.getSizes());
    });

    // ── temporal K window ─────────────────────────────────────────────────
    // Common IR이 만든 M/N scheduling round는 그대로 소비한다. target에서 추가로
    // 만드는 시간 분할은 한 device의 execution tile 안에서 reduction 범위가 K
    // residency보다 클 때뿐이다. 각 launch의 partial은 host가 같은 output에 누산한다.
    auto tiles =
        device->getAttrOfType<pim::ExecutionTileAttr>("pim.execution_tile");
    SmallVector<int64_t, 3> tileShape = pim::executionTileShape(device);
    if (tiles && tileShape.size() != 3) {
      device->emitError("convert-pim-to-upmem: cannot derive execution tile "
                        "shape from the Common IR hierarchy");
      return signalPassFailure();
    }
    // reduction loop: pim.axis 없이 access 의 origin 을 모는 loop, 하나일 때만.
    scf::ForOp kLoop;
    {
      SmallVector<scf::ForOp> allLoops;
      device.getBody().walk([&](scf::ForOp l) { allLoops.push_back(l); });
      llvm::SmallPtrSet<Operation *, 4> reds;
      device.getBody().walk([&](pim::AccessOp a) {
        for (Value o : a.getOrigin()) {
          llvm::SmallPtrSet<Operation *, 4> ds;
          pim::collectDrivers(o, allLoops, ds);
          for (Operation *op : ds)
            if (!op->getAttrOfType<pim::AxisAttr>("pim.axis"))
              reds.insert(op);
        }
      });
      if (reds.size() == 1)
        kLoop = cast<scf::ForOp>(*reds.begin());
    }

    // M/N execution-tile round는 공통 pim-map-devices가 이미 pim.device 바깥에
    // 만들었다. target lowering이 같은 축을 다시 windowing하면 각 execution tile을
    // 전체 문제의 round 수만큼 중복 실행하게 된다. 여기에는 reduction loop 안에서만
    // 생기는 temporal K window만 남긴다.
    SmallVector<Window> windows;
    if (tiles) {
      if (tileShape[2] > 0) {
        if (!kLoop) {
          device->emitError(
              "convert-pim-to-upmem: the execution tile has a k cap but the "
              "kernel has neither a k axis nor a single "
              "reduction loop to window");
          return signalPassFailure();
        }
        Window w;
        w.kLoop = kLoop;
        w.cap = tileShape[2];
        w.trips = kLoop.getUpperBound();
        windows.push_back(w);
      }
    }

    // 각 place에서 reduction loop가 움직이는 buffer 축을 찾는다. 한 place의 두 축이
    // 같은 reduction에 함께 매이면 직사각형 temporal window로 표현할 수 없다.
    SmallVector<pim::PlaceOp> allPlaces(inputs);
    allPlaces.append(outputs);
    llvm::DenseMap<Operation *, SmallVector<int, 4>> dimWinOf;
    for (pim::PlaceOp place : allPlaces)
      dimWinOf[place].assign(place.getAllocationShape().size(), -1);
    for (unsigned wi = 0; wi < windows.size(); ++wi) {
      Window &w = windows[wi];
      int64_t elems = 0;
      for (pim::PlaceOp place : allPlaces) {
        SmallVector<int, 4> &dims = dimWinOf[place];
        int found = -1;
        pim::AccessOp access;
        nest.walk([&](pim::AccessOp a) {
          if (a.getId() == place.getAccess())
            access = a;
        });
        if (access) {
          SmallVector<scf::ForOp, 1> only{w.kLoop};
          for (unsigned d = 0, e = access.getOrigin().size(); d < e; d++) {
            llvm::SmallPtrSet<Operation *, 4> ds;
            pim::collectDrivers(access.getOrigin()[d], only, ds);
            if (!ds.contains(w.kLoop))
              continue;
            if (found >= 0) {
              place->emitError(
                  "convert-pim-to-upmem: two buffer axes follow the "
                  "reduction window, so the window is not a single "
                  "rectangle");
              return signalPassFailure();
            }
            found = d;
          }
        }
        if (found >= 0) {
          dims[found] = wi;
          int64_t sz = tileOf[place.getAccess()][found];
          if (elems && elems != sz) {
            place->emitError(
                "convert-pim-to-upmem: the windowed buffers disagree on the "
                "tile size along the window axis");
            return signalPassFailure();
          }
          elems = sz;
        }
      }
      if (!elems) {
        device->emitError(
            "convert-pim-to-upmem: no buffer follows a window axis");
        return signalPassFailure();
      }
      if (w.cap % elems != 0) {
        device->emitError("convert-pim-to-upmem: the execution tile cap (")
            << w.cap << " elements) is not a multiple of the group tile ("
            << elems << " elements) along the window axis";
        return signalPassFailure();
      }
      w.elems = elems;
      w.tiles = w.cap / elems;
    }

    // ── 모듈 수준 선언 ─────────────────────────────────────────────────────
    OpBuilder mb(module.getBody(), module.getBody()->begin());
    StringRef base = func.getName();
    SmallVector<pim::PlaceOp> all(inputs);
    all.append(outputs);

    // MRAM buffer 크기. 모든 차원이 창(정적 tile 수 × tile)이나 상수로 묶이면 그
    // 곱이 정확한 크기다 — 전 buffer 가 그러면 그 크기로 잡고 합이 예산에 드는지
    // 여기서 확인한다. 하나라도 런타임 차원이 남으면 기존대로 예산 균등 분할 상한을
    // 쓰고 host 가 전송 직전에 검사한다.
    auto staticBytes = [&](pim::PlaceOp place) -> int64_t {
      const SmallVector<int, 4> &dims = dimWinOf[place];
      ArrayRef<int64_t> tile = tileOf[place.getAccess()];
      int64_t elems = 1;
      for (unsigned d = 0, e = place.getAllocationShape().size(); d < e; d++) {
        if (dims[d] >= 0) {
          elems *= windows[dims[d]].tiles * tile[d];
          continue;
        }
        std::optional<int64_t> v =
            getConstantIntValue(place.getAllocationShape()[d]);
        if (!v)
          return -1;
        elems *= *v;
      }
      Type elem =
          cast<BaseMemRefType>(place.getBuffer().getType()).getElementType();
      return elems * std::max<int64_t>(1, elem.getIntOrFloatBitWidth() / 8);
    };
    int64_t bankBudget = profile.getBankCapacityLimitBytes();
    int64_t fallbackBytes = bankBudget / std::max<int64_t>(1, all.size());
    llvm::DenseMap<Operation *, int64_t> bytesOf;
    bool allStatic = true;
    int64_t staticTotal = 0;
    for (pim::PlaceOp place : all) {
      int64_t b = staticBytes(place);
      bytesOf[place] = b;
      if (b < 0)
        allStatic = false;
      else
        staticTotal += b;
    }
    if (allStatic && staticTotal > bankBudget) {
      device->emitError("convert-pim-to-upmem: the execution tile needs ")
          << staticTotal << " byte of MRAM but the budget is " << bankBudget
          << " byte — lower tile_m, tile_n, or tile_k";
      return signalPassFailure();
    }

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
      if (info.tile.size() != place.getAllocationShape().size()) {
        place->emitError("no pim.access with this id inside pim.device");
        return signalPassFailure();
      }
      int64_t bufBytes = allStatic ? bytesOf[place] : fallbackBytes;
      if (!allStatic &&
          failed(checkCapacity(place, info.elem, fallbackBytes,
                               dimWinOf[place], windows, info.tile)))
        return signalPassFailure();
      mb.create<upmem::MRAMOp>(loc, name,
                               TypeAttr::get(mramType(info.elem, bufBytes)));
      // 축 하나를 건너뛰는 간격. device가 MRAM 안의 위치를 계산하는 데 쓴다.
      for (unsigned d = 1, e = place.getAllocationShape().size(); d < e; d++) {
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
    // Temporal reduction 반복 횟수 변수. split-K kernel에는 spatial k축과 그 plane
    // 내부 temporal reduction이 동시에 있으므로 `_tiles_reduction_k`로 구분한다.
    for (Window &w : windows) {
      auto name =
          StringAttr::get(ctx, (base + "_tiles_reduction_k").str());
      w.tilesVar = FlatSymbolRefAttr::get(name);
      mb.create<upmem::HostVarOp>(
          loc, name, TypeAttr::get(w.kLoop.getUpperBound().getType()));
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
    // k 창이 있으면 reduction loop 도 창의 tile 수만 돌게 상한을 host 변수로 돌린다.
    // 원래 상한 사슬은 죽은 값 정리가 걷어 간다.
    for (Window &w : windows) {
      Value n = db.create<upmem::GetHostVarOp>(
          loc, w.kLoop.getUpperBound().getType(), w.tilesVar);
      w.kLoop.getUpperBoundMutable().assign(n);
    }

    // ── host: func 몸체를 UPMEM 호출로 바꾼다 ──────────────────────────────
    func->setAttr("upmem.host_threads", mb.getI64IntegerAttr(hostThreads));
    // 같은 PIM Device set은 모든 scheduling round에서 재사용한다. acquire를 공통
    // scheduling loop 바깥에 두고, round 안에는 stage/launch/gather만 남긴다.
    Operation *scheduleRoot = device;
    while (auto round = dyn_cast_or_null<scf::ForOp>(scheduleRoot->getParentOp())) {
      if (!round->hasAttr("pim.scheduling_axis"))
        break;
      scheduleRoot = round;
    }
    OpBuilder setup(scheduleRoot);
    Value set = setup.create<upmem::AcquireOp>(
        loc, upmem::SetType::get(ctx), setup.getI64IntegerAttr(numUnits),
        FlatSymbolRefAttr::get(devName));
    OpBuilder hb(device);

    ArrayAttr axisNames = hb.getStrArrayAttr({"m", "n", "k"});
    auto counts =
        hb.getDenseI64ArrayAttr(device.getMapping().getPartitions().asArrayRef());
    auto foreach = hb.create<upmem::ForeachUnitOp>(loc, set, axisNames, counts);
    SmallVector<Type> argTys{upmem::UnitType::get(ctx)};
    SmallVector<Location> argLocs{loc};
    for (unsigned i = 0; i < 3; i++) {
      argTys.push_back(hb.getIndexType());
      argLocs.push_back(loc);
    }
    Block *fuBody = hb.createBlock(&foreach.getBody(), {}, argTys, argLocs);
    // 좌표 계산을 통째로 옮기고 block argument를 새 것으로 잇는다.
    fuBody->getOperations().splice(fuBody->end(), deviceBody->getOperations(),
                                   deviceBody->begin(), deviceBody->end());
    for (unsigned i = 0; i < 3; i++)
      deviceBody->getArgument(i).replaceAllUsesWith(fuBody->getArgument(i + 1));

    Value dpu = fuBody->getArgument(0);
    auto hasWindow = [&](pim::PlaceOp place) {
      for (int wi : dimWinOf[place])
        if (wi >= 0)
          return true;
      return false;
    };
    for (pim::PlaceOp place : inputs) {
      if (hasWindow(place))
        continue; // 창에 매인 입력은 자기 창 level 에서 창을 다시 모은다
      OpBuilder b(place);
      Value staged = b.create<upmem::StageOp>(
          loc, stagedType(place), place.getBuffer(), place.getStart(),
          place.getSize(), place.getStrides(), place.getAllocationShape(),
          b.getDenseI64ArrayAttr(byAccess[place.getAccess()].tile),
          place.getReplicatedOverAttr());
      b.create<upmem::ToMRAMOp>(loc, dpu, staged, mramOf[place]);
    }
    // Common IR mapping 축의 device별 반복 횟수를 그 DPU에 보낸다. Temporal K
    // 반복 횟수만 아래 window loop에서 launch마다 따로 보낸다.
    OpBuilder tb(fuBody, fuBody->end());
    for (auto [i, a] : llvm::enumerate(axes))
      tb.create<upmem::ToHostVarOp>(loc, dpu, a.trips, tilesVar[i]);
    // MRAM 안의 축 간격도 함께 보낸다. 창에 매인 축의 간격은 창 크기다 — 그 buffer 는
    // launch 마다 창 하나만 상주하기 때문이다.
    for (pim::PlaceOp place : all) {
      const Buffer &info = byAccess[place.getAccess()];
      const SmallVector<int, 4> &dims = dimWinOf[place];
      for (auto [d, sym] : llvm::enumerate(info.pitch)) {
        Value v = place.getAllocationShape()[d + 1];
        if (dims[d + 1] >= 0)
          v = tb.create<arith::ConstantIndexOp>(
              loc, windows[dims[d + 1]].tiles * info.tile[d + 1]);
        tb.create<upmem::ToHostVarOp>(loc, dpu, v, sym);
      }
    }

    hb.setInsertionPointAfter(foreach);
    if (windows.empty()) {
      hb.create<upmem::LaunchOp>(loc, set);
      for (pim::PlaceOp place : outputs) {
        // buildOutput이 region을 만들며 삽입 지점을 옮기므로 매번 되돌린다.
        OpBuilder::InsertionGuard guard(hb);
        if (failed(buildOutput(hb, place, set, numUnits, axisNames, counts,
                               mramOf[place], foreach,
                               byAccess[place.getAccess()].tile, windows,
                               dimWinOf[place], /*roundIVs=*/{},
                               /*accRound=*/Value())))
          return signalPassFailure();
      }
    } else {
      // Temporal K loop. 창 수 = ceildiv(reduction 반복 횟수, K residency)이며,
      // 각 round에서 남은 trip 수를 DPU에 보내고 K를 쓰는 입력만 다시 전송한다.
      Value c0 = hb.create<arith::ConstantIndexOp>(loc, 0);
      Value c1 = hb.create<arith::ConstantIndexOp>(loc, 1);
      SmallVector<Value> roundIVs(windows.size());
      Value accRound; // k 창 번호 — 출력 누산의 스위치
      OpBuilder lb = hb;
      for (unsigned wi = 0; wi < windows.size(); ++wi) {
        Window &w = windows[wi];
        // K window 수는 device 좌표와 무관해야 모든 DPU가 같은 횟수로 launch된다.
        IRMapping empty;
        Value kTrips = cloneInto(lb, w.trips, empty, foreach);
        if (!kTrips) {
          device->emitError(
              "convert-pim-to-upmem: the reduction trip count depends on "
              "the device coordinate, so the k window count cannot be "
              "computed on the host");
          return signalPassFailure();
        }
        if (!isa<IndexType>(kTrips.getType()))
          kTrips =
              lb.create<arith::IndexCastOp>(loc, lb.getIndexType(), kTrips);
        Value wTiles = lb.create<arith::ConstantIndexOp>(loc, w.tiles);
        Value rounds = lb.create<arith::CeilDivSIOp>(loc, kTrips, wTiles);
        auto roundLoop = lb.create<scf::ForOp>(loc, c0, rounds, c1);
        roundIVs[wi] = roundLoop.getInductionVar();
        accRound = roundIVs[wi];
        lb = OpBuilder(roundLoop.getBody(),
                       std::prev(roundLoop.getBody()->end()));

        // 이 level 의 foreach: 창 반복 횟수 갱신 + 재전송. 좌표 산술은 첫 foreach
        // 의 식을 복제해 새 좌표 인자에 잇는다.
        auto fu2 = lb.create<upmem::ForeachUnitOp>(loc, set, axisNames, counts);
        Block *fu2Body;
        {
          OpBuilder blockBuilder(ctx);
          fu2Body = blockBuilder.createBlock(&fu2.getBody(),
                                             fu2.getBody().end(), argTys,
                                             argLocs);
        }
        IRMapping toRound;
        for (unsigned i = 0, e = fu2Body->getNumArguments(); i < e; i++)
          toRound.map(fuBody->getArgument(i), fu2Body->getArgument(i));
        Value dpu2 = fu2Body->getArgument(0);
        OpBuilder fb(fu2Body, fu2Body->end());
        Value zero = fb.create<arith::ConstantIndexOp>(loc, 0);

        // 이 창의 반복 횟수 = clamp(trips − round·창, 0, 창).
        Value tripsW = cloneInto(fb, w.trips, toRound, foreach);
        if (!tripsW) {
          device->emitError("convert-pim-to-upmem: the window axis trip count "
                            "cannot be rebuilt in the round loop");
          return signalPassFailure();
        }
        Value winTrips;
        {
          // reduction 상한이 index 가 아니면 index 로 셈하고 원 타입으로 돌린다.
          Type origTy = tripsW.getType();
          Value t = tripsW;
          if (!isa<IndexType>(origTy))
            t = fb.create<arith::IndexCastOp>(loc, fb.getIndexType(), t);
          Value wTiles = fb.create<arith::ConstantIndexOp>(loc, w.tiles);
          Value done = fb.create<arith::MulIOp>(loc, roundIVs[wi], wTiles);
          Value left = fb.create<arith::SubIOp>(loc, t, done);
          winTrips = fb.create<arith::MinSIOp>(
              loc, fb.create<arith::MaxSIOp>(loc, left, zero), wTiles);
          if (!isa<IndexType>(origTy))
            winTrips =
                fb.create<arith::IndexCastOp>(loc, origTy, winTrips);
        }
        fb.create<upmem::ToHostVarOp>(loc, dpu2, winTrips, w.tilesVar);

        for (pim::PlaceOp place : inputs) {
          const SmallVector<int, 4> &dims = dimWinOf[place];
          // 이 place 의 가장 안쪽 창 level 이 여기인가 — 여기여야 매인 창의 좌표가
          // 전부 정해진 상태에서 한 번만 재전송한다.
          int deepest = -1;
          for (int v : dims)
            deepest = std::max(deepest, v);
          if (deepest != (int)wi)
            continue;
          const Buffer &info = byAccess[place.getAccess()];
          SmallVector<Value> start, size, strides, capacity;
          for (unsigned d = 0, e = place.getAllocationShape().size(); d < e;
               d++) {
            Value o = cloneInto(fb, place.getStart()[d], toRound, foreach);
            Value s = cloneInto(fb, place.getSize()[d], toRound, foreach);
            Value st = cloneInto(fb, place.getStrides()[d], toRound, foreach);
            Value cp =
                cloneInto(fb, place.getAllocationShape()[d], toRound, foreach);
            if (!o || !s || !st || !cp) {
              place->emitError("convert-pim-to-upmem: the window region cannot "
                               "be rebuilt in the round loop");
              return signalPassFailure();
            }
            if (dims[d] >= 0) {
              const Window &dw = windows[dims[d]];
              Value winElems = fb.create<arith::ConstantIndexOp>(
                  loc, dw.tiles * info.tile[d]);
              Value shift =
                  fb.create<arith::MulIOp>(loc, roundIVs[dims[d]], winElems);
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
      }

      lb.create<upmem::LaunchOp>(loc, set);
      for (pim::PlaceOp place : outputs) {
        OpBuilder::InsertionGuard guard(lb);
        if (failed(buildOutput(lb, place, set, numUnits, axisNames, counts,
                               mramOf[place], foreach,
                               byAccess[place.getAccess()].tile, windows,
                               dimWinOf[place], roundIVs, accRound)))
          return signalPassFailure();
      }
    }

    // pim.place는 여기서 한 번에 지운다. 위의 stage/to_host_var가 이 op의
    // start·size·capacity를 계속 읽으므로 그 전에 지우면 해제된 op을 읽게 된다.
    for (pim::PlaceOp place : all)
      place.erase();

    device.erase();
    if (failed(convertDevice(dev, byAccess, lanes, tb, dpu, base)))
      return signalPassFailure();
  }

  // 실제로 존재하는 m/n/k loop와, 그 loop가 device 몫으로 좁혀지며 생긴 상한 및
  // 전역 좌표 복원 op을 canonical 순서로 모은다.
  LogicalResult collectAxes(pim::DeviceOp device, SmallVectorImpl<Axis> &axes) {
    StringRef canonical[] = {"m", "n", "k"};
    for (StringRef axisName : canonical) {
      auto name = StringAttr::get(&getContext(), axisName);
      scf::ForOp loop;
      device.getBody().walk([&](scf::ForOp l) {
        if (auto axis = l->getAttrOfType<pim::AxisAttr>("pim.axis"))
          if (axis.getName() == name)
            loop = l;
      });
      if (!loop)
        continue;
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

  // 균등 분할(fallback) 경로의 버퍼별 검사. 총합만 보면 개별 버퍼가 넘치는 구성이
  // 통과하므로 버퍼마다 본다. 크기가 런타임이면 여기서 증명할 수 없고, host 코드가
  // 전송 직전에 검사한다. 창에 매인 축은 launch 마다 창 하나만 상주한다.
  LogicalResult checkCapacity(pim::PlaceOp place, Type elem, int64_t bufferBytes,
                              ArrayRef<int> dimWin, ArrayRef<Window> windows,
                              ArrayRef<int64_t> tile) {
    int64_t elems = 1;
    for (auto [d, c] : llvm::enumerate(place.getAllocationShape())) {
      if (dimWin[d] >= 0) {
        elems *= windows[dimWin[d]].tiles * tile[d];
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
    SmallVector<int64_t> shape(place.getAllocationShape().size(),
                               ShapedType::kDynamic);
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
                              int64_t lanes, OpBuilder &tb, Value dpu,
                              StringRef base) {
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
    // Block Group 하나는 모든 tasklet이 함께 계산한다. WRAM buffer는 한 벌이고,
    // lanes > 1이면 compute의 output m과 DMA chunk를 tasklet 사이에 나눈다.
    bool cooperative = lanes > 1;
    Value taskletId;
    if (cooperative)
      taskletId = top.create<upmem::TaskletIdOp>(loc, top.getIndexType());
    auto allocFor = [&](RankedTensorType t) -> Value {
      SmallVector<int64_t> shape(t.getShape());
      return top.create<memref::AllocOp>(
          loc, MemRefType::get(shape, t.getElementType(),
                               MemRefLayoutAttrInterface(), wram));
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
      if (auto gen = dyn_cast<linalg::GenericOp>(op)) {
        if (!isIdentityPointwise(gen)) {
          op->emitError(
              "convert-pim-to-upmem: UPMEM supports only identity-map, "
              "all-parallel linalg.generic compute");
          return WalkResult::interrupt();
        }
      } else if (!isa<linalg::FillOp, linalg::MatmulOp, linalg::MatvecOp>(op)) {
        op->emitError(
            "convert-pim-to-upmem: UPMEM compute lowering currently supports "
            "fill, matmul, matvec, and identity pointwise generic");
        return WalkResult::interrupt();
      }
      OpBuilder b(op);
      SmallVector<Value> operands;
      for (Value v : op->getOperands())
        operands.push_back(bufferOf(v) ? bufferOf(v) : v);
      bool hasTileInput = llvm::any_of(
          ArrayRef<Value>(operands).drop_back(),
          [](Value v) { return isa<MemRefType>(v.getType()); });
      if (cooperative) {
        // 어떤 operand가 output m축을 따르는지는 연산 이름이 아니라 linalg indexing
        // map으로 판정한다. d0이 leading dimension이면 그 WRAM buffer만 행 slice하고,
        // d0을 쓰지 않는 B/x 같은 reduction operand는 전체 tile을 공유한다.
        SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
        if (maps.size() != operands.size()) {
          op->emitError("convert-pim-to-upmem: linalg indexing map count does "
                        "not match the operand count");
          return WalkResult::interrupt();
        }
        AffineExpr m = getAffineDimExpr(0, &getContext());
        for (auto [i, map] : llvm::enumerate(maps)) {
          if (!map.isProjectedPermutation()) {
            op->emitError(
                "convert-pim-to-upmem: output-row tasklet partition requires "
                "projected-permutation linalg indexing maps");
            return WalkResult::interrupt();
          }
          int mResult = -1;
          for (unsigned r = 0; r < map.getNumResults(); ++r)
            if (map.getResult(r) == m)
              mResult = r;
          if (mResult > 0) {
            op->emitError(
                "convert-pim-to-upmem: output m must be the leading dimension "
                "of every operand that follows m");
            return WalkResult::interrupt();
          }
          if (mResult == 0 &&
              !(operands[i] = rowSliceOf(operands[i], op)))
            return WalkResult::interrupt();
        }
        // Cooperative DMA는 tile 전체를 tasklet 사이에 나누므로 compute 전에 모든
        // input chunk가 도착해야 한다. Fill처럼 shaped input이 없으면 barrier가 없다.
        if (hasTileInput)
          b.create<upmem::BarrierOp>(op->getLoc());
      }
      OperationState st(op->getLoc(), op->getName());
      st.addOperands(operands);
      st.addAttributes(op->getAttrs());
      IRMapping map;
      for (Region &r : op->getRegions())
        r.cloneInto(st.addRegion(), map);
      b.create(st);
      // Reduction iter_arg loop 안에서는 다음 반복의 DMA가 공유 buffer를 덮기 전에
      // 모든 tasklet의 소비가 끝나야 한다. 연산 종류가 아니라 반복 구조로 판정한다.
      bool repeatedCompute = false;
      for (Operation *parent = op->getParentOp();
           parent && parent != dev.getOperation();
           parent = parent->getParentOp())
        if (auto loop = dyn_cast<scf::ForOp>(parent))
          repeatedCompute |= loop.getNumRegionIterArgs() > 0;
      if (cooperative && hasTileInput && repeatedCompute)
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
                 pitchOf, /*isRead=*/true, taskletId, lanes);
        oldLoads.push_back(op);
      } else if (auto store = dyn_cast<pim::StoreOp>(op)) {
        // 협력 모드의 저장도 chunk 분담이라 다른 tasklet의 행을 읽는다. 앞 barrier는
        // 전 행의 계산이 끝났음을(matmul이 없는 적재→저장 커널의 경로도 여기서 닫힌다),
        // 뒤 barrier는 다음 tile의 fill이 누산기를 덮기 전에 저장이 끝났음을 보장한다.
        if (cooperative) {
          OpBuilder b(store);
          b.create<upmem::BarrierOp>(store.getLoc());
        }
        buildDMA(store, store.getAccess(), bufferOf(store.getValue()), byAccess,
                 pitchOf, /*isRead=*/false, taskletId, lanes);
        if (cooperative) {
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
                bool isRead, Value taskletId, int64_t lanes) {
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
    if (lanes > 1) {
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

  // 출력 하나: 전 device에서 회수 → 그룹마다 원본으로 되돌리기. 창에 매인 축의
  // 회수량은 창 크기이고, 되돌릴 자리는 그 창의 round 만큼 밀린다. k 창이 있으면
  // 회수한 것이 부분합이므로 unstage 가 첫 창(accRound == 0)은 대입, 그 뒤는
  // 누산한다.
  LogicalResult buildOutput(OpBuilder &b, pim::PlaceOp place, Value set,
                            int64_t numUnits, ArrayAttr axes,
                            DenseI64ArrayAttr counts, FlatSymbolRefAttr mram,
                            Operation *foreach, ArrayRef<int64_t> tile,
                            ArrayRef<Window> windows, ArrayRef<int> dimWin,
                            ArrayRef<Value> roundIVs, Value accRound) {
    Location loc = place.getLoc();
    unsigned rank = place.getAllocationShape().size();
    auto elem =
        cast<BaseMemRefType>(place.getBuffer().getType()).getElementType();

    // 회수할 바이트 수는 배정량에서 나온다. device 좌표와 무관하므로 foreach 밖으로
    // 옮겨 만들 수 있다 — 못 옮기면 device마다 크기가 다르다는 뜻이라 회수가 성립하지 않는다.
    IRMapping empty;
    SmallVector<Value> capacity;
    for (auto [d, c] : llvm::enumerate(place.getAllocationShape())) {
      if (dimWin[d] >= 0) {
        capacity.push_back(b.create<arith::ConstantIndexOp>(
            loc, windows[dimWin[d]].tiles * tile[d]));
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
    Value zero;
    SmallVector<Value> start, size, strides;
    for (unsigned d = 0; d < rank; d++) {
      // 좌표 계산은 이미 foreach_unit 몸체로 옮겨졌으므로 그 범위를 기준으로 본다.
      Value o = cloneInto(gbuilder, place.getStart()[d], map, foreach);
      Value s = cloneInto(gbuilder, place.getSize()[d], map, foreach);
      Value t = cloneInto(gbuilder, place.getStrides()[d], map, foreach);
      if (!o || !s || !t)
        return place->emitError(
            "output region cannot be rebuilt from group coordinates");
      if (dimWin[d] >= 0) {
        if (!zero)
          zero = gbuilder.create<arith::ConstantIndexOp>(loc, 0);
        Value winElems = gbuilder.create<arith::ConstantIndexOp>(
            loc, windows[dimWin[d]].tiles * tile[d]);
        Value shift = gbuilder.create<arith::MulIOp>(
            loc, roundIVs[dimWin[d]], winElems);
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
                                      start, size, strides, capacity, accRound,
                                      gbuilder.getDenseI64ArrayAttr(tile));
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertPIMToUPMEMPass() {
  return std::make_unique<ConvertPIMToUPMEMPass>();
}
