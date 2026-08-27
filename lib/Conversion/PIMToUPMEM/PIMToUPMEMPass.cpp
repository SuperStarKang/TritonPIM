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

// 시간 창 하나 — #pim.execution_tile 의 축 상한 하나가 창 하나다.
// 출력 축 창(mappingIdx ≥ 0)은 launch 를 반복하며 배정 몫을 창 단위로 상주시키고,
// k 창(mappingIdx < 0, kLoop)은 launch 마다 K 의 한 구간만 누적해 부분합을 낸다 —
// host 가 창을 건너 누산한다(unstage accumulate).
struct Window {
  int mappingIdx = -1;      // device mapping 축 번호. k 창은 -1
  scf::ForOp kLoop;         // k 창일 때 reduction loop
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
      f->removeAttr("pim.intra_device_plan");
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
    if (profile.getParallelExecutionKind() != "multithread")
      return getOperation()->emitError(
          "convert-pim-to-upmem: UPMEM tasklets are independent threads, so "
          "parallel_execution_kind must be \"multithread\"");
    if (profile.getNumberOfBanksPerPimDevice() != 1 ||
        profile.getNumberOfPimCoresPerBank() != 1)
      return getOperation()->emitError(
          "convert-pim-to-upmem: UPMEM topology must be one MRAM bank and "
          "one DPU core per PIM Device");
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

    if (!device->getAttrOfType<pim::InterDevicePlanAttr>(
            "pim.inter_device_plan")) {
      device->emitError(
          "convert-pim-to-upmem: the device has no #pim.inter_device_plan");
      return signalPassFailure();
    }
    auto reduction =
        device->getAttrOfType<pim::ReductionPlanAttr>("pim.reduction_plan");
    if (!reduction) {
      device->emitError(
          "convert-pim-to-upmem: the device has no #pim.reduction_plan");
      return signalPassFailure();
    }
    if (reduction.getTemporalAccumulation().getValue() != "none" &&
        reduction.getTemporalAccumulation().getValue() != "host") {
      device->emitError(
          "convert-pim-to-upmem: only host temporal accumulation is supported");
      return signalPassFailure();
    }

    auto intra = func->getAttrOfType<pim::IntraDevicePlanAttr>(
        "pim.intra_device_plan");
    if (!intra) {
      func->emitError("convert-pim-to-upmem: the function has no "
                      "#pim.intra_device_plan");
      return signalPassFailure();
    }
    int64_t lanes = intra.getLanes();
    ArrayRef<int64_t> bp = intra.getBankPartitions().asArrayRef();
    ArrayRef<int64_t> cp = intra.getCorePartitions().asArrayRef();
    if (!llvm::all_of(bp, [](int64_t p) { return p == 1; }) ||
        !llvm::all_of(cp, [](int64_t p) { return p == 1; })) {
      func->emitError(
          "convert-pim-to-upmem: UPMEM intra-device bank/core partitions "
          "must both be [1, 1, 1]");
      return signalPassFailure();
    }
    if (lanes > 1 && intra.getLaneAxis().getValue() != "m") {
      func->emitError(
          "convert-pim-to-upmem: cooperative tasklets can split only the m "
          "axis");
      return signalPassFailure();
    }
    if (auto group =
            func->getAttrOfType<pim::BlockGroupAttr>("pim.block_group")) {
      int64_t rows = group.getUseful().asArrayRef()[0];
      if (rows % lanes != 0) {
        func->emitError("convert-pim-to-upmem: block-group m extent ")
            << rows << " is not divisible by " << lanes << " tasklets";
        return signalPassFailure();
      }
    }
    // 협력 lane 모드: lane들이 tile 하나를 출력 행으로 나눠 함께 계산한다. 버퍼는 한 벌,
    // 적재는 분담, 경계는 barrier다. lane이 하나면 나눌 것이 없어 기존 경로와 같다.
    bool rowSplit = lanes > 1 && intra.getLaneAxis().getValue() == "m";

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

    // ── 창(시간 분할) 문맥 ─────────────────────────────────────────────────
    // 창은 공통 IR 의 #pim.execution_tile 에서 온다. 출력 축(m·n) 창은 launch 를
    // 반복하며 배정 몫을 창 단위로 상주시키고, k(reduction) 창은 launch 마다 K 의
    // 한 구간만 누적해 부분합을 내며 host 가 누산한다(unstage accumulate). 배정이
    // 상한 안이면 창 수가 런타임에 1 로 퇴화해 반복이 없던 것과 같다.
    // host loop 순서는 n → k → m 이고, 각 place 는 자기가 매인 창 중 가장 안쪽
    // level 에서 창을 재전송한다 — B(k·n)는 k level, A(m·k)는 m level. k 상한이
    // k이 있어도 각 spatial plane 안의 reduction 범위가 tile-k보다
    // 크면 temporal K window를 추가한다. temporal partial은 같은 output plane에 host
    // 누산되고, 서로 다른 spatial plane의 최종 합산은 그 출력의 소비자가 수행한다.
    auto tiles =
        device->getAttrOfType<pim::ExecutionTileAttr>("pim.execution_tile");
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
      if (tiles.getK() > 0) {
        if (!kLoop) {
          device->emitError(
              "convert-pim-to-upmem: the execution tile has a k cap but the "
              "kernel has neither a k axis nor a single "
              "reduction loop to window");
          return signalPassFailure();
        }
        Window w;
        w.kLoop = kLoop;
        w.cap = tiles.getK();
        w.trips = kLoop.getUpperBound();
        windows.push_back(w);
      }
    }

    // 각 place 가 어느 창의 어느 buffer 축에 매였는가. 출력 축 창은 영역 식이 그 축
    // 좌표 인자에 닿는가로, k 창은 access 의 origin 을 reduction loop 가 모는가로
    // 가른다. 창마다 place 의 축은 최대 하나여야 창이 직사각형 하나로 남는다.
    SmallVector<pim::PlaceOp> allPlaces(inputs);
    allPlaces.append(outputs);
    llvm::DenseMap<Operation *, SmallVector<int, 4>> dimWinOf;
    for (pim::PlaceOp place : allPlaces)
      dimWinOf[place].assign(place.getCapacity().size(), -1);
    for (unsigned wi = 0; wi < windows.size(); ++wi) {
      Window &w = windows[wi];
      int64_t elems = 0;
      for (pim::PlaceOp place : allPlaces) {
        SmallVector<int, 4> &dims = dimWinOf[place];
        int found = -1;
        if (w.mappingIdx >= 0) {
          Value warg = deviceBody->getArgument(w.mappingIdx);
          for (unsigned d = 0, e = place.getCapacity().size(); d < e; d++) {
            if (!dependsOnArg(place.getStart()[d], warg) &&
                !dependsOnArg(place.getSize()[d], warg) &&
                !dependsOnArg(place.getCapacity()[d], warg))
              continue;
            if (found >= 0) {
              place->emitError(
                  "convert-pim-to-upmem: two buffer axes follow the same "
                  "window axis, so the window is not a single rectangle");
              return signalPassFailure();
            }
            found = d;
          }
        } else {
          // k 창: 이 place 의 access origin 중 reduction loop 가 모는 차원.
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
        }
        if (place.getDirection() == "out" && w.mappingIdx >= 0 && found < 0) {
          place->emitError(
              "convert-pim-to-upmem: the output does not follow the window "
              "axis, so every round would rewrite the same region");
          return signalPassFailure();
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
      for (unsigned d = 0, e = place.getCapacity().size(); d < e; d++) {
        if (dims[d] >= 0) {
          elems *= windows[dims[d]].tiles * tile[d];
          continue;
        }
        std::optional<int64_t> v = getConstantIntValue(place.getCapacity()[d]);
        if (!v)
          return -1;
        elems *= *v;
      }
      Type elem =
          cast<BaseMemRefType>(place.getBuffer().getType()).getElementType();
      return elems * std::max<int64_t>(1, elem.getIntOrFloatBitWidth() / 8);
    };
    int64_t bankBudget = profile.getUsableBankCapacityBytes().getInt();
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
      if (info.tile.size() != place.getCapacity().size()) {
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
    // 창 축의 반복 횟수 변수. mapping 축 창은 위의 tilesVar 를 그대로 쓰고,
    // reduction 창은 별도 이름으로 만든다. split-K kernel에는 spatial k축과 그
    // plane 내부 temporal reduction 창이 동시에 존재하므로 둘 다 `_tiles_k`를 쓰면
    // module symbol이 충돌한다.
    for (Window &w : windows) {
      if (w.mappingIdx >= 0) {
        w.tilesVar = tilesVar[w.mappingIdx];
        continue;
      }
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
      if (w.mappingIdx >= 0)
        continue;
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
          place.getSize(), place.getStrides(), place.getCapacity(),
          b.getDenseI64ArrayAttr(byAccess[place.getAccess()].tile),
          place.getReplicatedOverAttr());
      b.create<upmem::ToMRAMOp>(loc, dpu, staged, mramOf[place]);
    }
    // device마다 다른 반복 횟수를 그 DPU에 보낸다. 창 축의 횟수는 창마다 다르므로
    // 그 창의 level 에서 보낸다.
    OpBuilder tb(fuBody, fuBody->end());
    auto axisWindowed = [&](size_t i) {
      for (const Window &w : windows)
        if (w.mappingIdx == (int)i)
          return true;
      return false;
    };
    for (auto [i, a] : llvm::enumerate(axes))
      if (!axisWindowed(i))
        tb.create<upmem::ToHostVarOp>(loc, dpu, a.trips, tilesVar[i]);
    // MRAM 안의 축 간격도 함께 보낸다. 창에 매인 축의 간격은 창 크기다 — 그 buffer 는
    // launch 마다 창 하나만 상주하기 때문이다.
    for (pim::PlaceOp place : all) {
      const Buffer &info = byAccess[place.getAccess()];
      const SmallVector<int, 4> &dims = dimWinOf[place];
      for (auto [d, sym] : llvm::enumerate(info.pitch)) {
        Value v = place.getCapacity()[d + 1];
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
      // 창 loop nest — 바깥 창부터 하나씩(n → k → m). 창 수 = ceildiv(축 반복
      // 횟수, 창)이라 배정이 상한 안이면 런타임에 1 로 퇴화한다. 창마다 [창 반복
      // 횟수 갱신 + 그 level 이 가장 안쪽 창인 place 의 재전송] foreach 를 놓고,
      // 가장 안쪽에서 launch 와 회수를 한다.
      Value c0 = hb.create<arith::ConstantIndexOp>(loc, 0);
      Value c1 = hb.create<arith::ConstantIndexOp>(loc, 1);
      SmallVector<Value> roundIVs(windows.size());
      Value accRound; // k 창 번호 — 출력 누산의 스위치
      OpBuilder lb = hb;
      for (unsigned wi = 0; wi < windows.size(); ++wi) {
        Window &w = windows[wi];
        Value rounds;
        if (w.mappingIdx >= 0) {
          // 배정량(capacity)은 전 device 공통이라 창 수도 공통이고, 덜 맡은
          // device 는 꼬리 창에서 0 회를 돈다.
          Value wTiles = lb.create<arith::ConstantIndexOp>(loc, w.tiles);
          Value parts = lb.create<arith::ConstantIndexOp>(
              loc,
              device.getMapping().getPartitions().asArrayRef()[w.mappingIdx]);
          Value capTiles = lb.create<arith::CeilDivSIOp>(
              loc, device.getAxisExtents()[w.mappingIdx], parts);
          rounds = lb.create<arith::CeilDivSIOp>(loc, capTiles, wTiles);
        } else {
          // k 창 수 = ceildiv(reduction 반복 횟수, 창). device 좌표에 무관해야
          // 전 device 가 같은 횟수의 launch 를 돈다.
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
          rounds = lb.create<arith::CeilDivSIOp>(loc, kTrips, wTiles);
        }
        auto roundLoop = lb.create<scf::ForOp>(loc, c0, rounds, c1);
        roundIVs[wi] = roundLoop.getInductionVar();
        if (w.mappingIdx < 0)
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
        if (auto axis = l->getAttrOfType<pim::AxisAttr>("pim.axis"))
          if (axis.getName() == name)
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

  // 균등 분할(fallback) 경로의 버퍼별 검사. 총합만 보면 개별 버퍼가 넘치는 구성이
  // 통과하므로 버퍼마다 본다. 크기가 런타임이면 여기서 증명할 수 없고, host 코드가
  // 전송 직전에 검사한다. 창에 매인 축은 launch 마다 창 하나만 상주한다.
  LogicalResult checkCapacity(pim::PlaceOp place, Type elem, int64_t bufferBytes,
                              ArrayRef<int> dimWin, ArrayRef<Window> windows,
                              ArrayRef<int64_t> tile) {
    int64_t elems = 1;
    for (auto [d, c] : llvm::enumerate(place.getCapacity())) {
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
        // 축이고 B는 전 행이 필요하다. identity pointwise generic은 모든 shaped
        // operand의 첫 축이 출력 m과 같으므로 전부 같은 행 slice로 가른다.
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
        } else if (auto gen = dyn_cast<linalg::GenericOp>(op)) {
          if (!isIdentityPointwise(gen)) {
            op->emitError(
                "convert-pim-to-upmem: cooperating tasklets require an "
                "identity-map, all-parallel linalg.generic");
            return WalkResult::interrupt();
          }
          for (Value &operand : operands) {
            auto type = dyn_cast<MemRefType>(operand.getType());
            if (!type || type.getRank() == 0 ||
                !(operand = rowSliceOf(operand, op)))
              return WalkResult::interrupt();
          }
          // DMA chunk는 tasklet 사이에 분담되므로 계산 전에 모든 입력 적재가 끝나야 한다.
          b.create<upmem::BarrierOp>(op->getLoc());
        } else {
          op->emitError(
              "convert-pim-to-upmem: cooperating tasklets "
              "(lane_axis = m) can split only matmul, fill, and identity "
              "pointwise generic operations by output rows");
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
    unsigned rank = place.getCapacity().size();
    auto elem =
        cast<BaseMemRefType>(place.getBuffer().getType()).getElementType();

    // 회수할 바이트 수는 배정량에서 나온다. device 좌표와 무관하므로 foreach 밖으로
    // 옮겨 만들 수 있다 — 못 옮기면 device마다 크기가 다르다는 뜻이라 회수가 성립하지 않는다.
    IRMapping empty;
    SmallVector<Value> capacity;
    for (auto [d, c] : llvm::enumerate(place.getCapacity())) {
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
