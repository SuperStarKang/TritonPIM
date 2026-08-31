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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <functional>

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

// Common scheduling이 좁힌 loop의 전역 좌표 복원 op. target은 loop dependency를
// 재분석하지 않고 명시적인 pim.device_origin만 소비한다.
static arith::AddIOp deviceOrigin(scf::ForOp loop) {
  for (Operation &op : *loop.getBody())
    if (auto add = dyn_cast<arith::AddIOp>(&op))
      if (add->hasAttr("pim.device_origin"))
        return add;
  return {};
}

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
      op->removeAttr("pim.temporal_reduction");
      op->removeAttr("pim.scheduling_axis");
      op->removeAttr("pim.combine");
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
    bool hasM = llvm::any_of(axes, [](const Axis &axis) {
      return axis.name.getValue() == "m";
    });
    bool hasSpatialK = llvm::any_of(axes, [](const Axis &axis) {
      return axis.name.getValue() == "k";
    });
    bool hasReduce = false;
    device.walk([&](linalg::ReduceOp) { hasReduce = true; });

    // UPMEM의 tasklet 협력은 Common의 m축 Block Group을 행 단위로 나누는
    // 구현이다. 실제 m축이 있을 때만 그 크기에 lane divisibility를 적용한다.
    if (hasM && func->getAttrOfType<pim::BlockGroupAttr>("pim.block_group")) {
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
    // M축이 없는 K-only reduction은 모든 tasklet이 같은 accumulator를 쓰게 된다.
    // tasklet-local partial과 DPU 내부 combine이 IR에 없는 동안에는 한 tasklet만이
    // Common이 정한 reduction 순서를 그대로 실행할 수 있다.
    if (!hasM && hasReduce && lanes != 1) {
      device->emitError(
          "convert-pim-to-upmem: K-only linalg.reduce requires "
          "nr-tasklets=1 until tasklet-local reduction combine is represented");
      return signalPassFailure();
    }

    int64_t numUnits = 1;
    for (int64_t c : device.getMapping().getPartitions().asArrayRef())
      numUnits *= c;

    // 타일 shape. device 버퍼를 이 단위로 쓰면 타일 하나가 MRAM에서 연속이 되어 DMA가
    // 한 번으로 끝난다. pim.access가 정적으로 들고 있다. K-only output access는
    // temporal loop의 뒤에 있으므로 첫 loop가 아니라 device 전체에서 수집한다.
    llvm::DenseMap<int64_t, SmallVector<int64_t>> tileOf;
    device.walk([&](pim::AccessOp a) {
      tileOf[a.getId()] = SmallVector<int64_t>(a.getSizes());
    });

    // ── Common temporal K contract ─────────────────────────────────────────
    // K축이나 window 크기를 access에서 다시 찾지 않는다. Common IR이 표시한 local
    // reduction loop와 그 바깥 scheduling round를 그대로 소비한다.
    SmallVector<scf::ForOp> temporalLoops;
    device.getBody().walk([&](scf::ForOp loop) {
      if (loop->hasAttr("pim.temporal_reduction"))
        temporalLoops.push_back(loop);
    });
    if (temporalLoops.size() > 1) {
      device->emitError("convert-pim-to-upmem: Common IR has more than one "
                        "pim.temporal_reduction loop");
      return signalPassFailure();
    }
    scf::ForOp temporalLoop =
        temporalLoops.empty() ? scf::ForOp() : temporalLoops.front();
    scf::ForOp temporalRound, spatialKRound;
    for (Operation *parent = device->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto loop = dyn_cast<scf::ForOp>(parent);
      if (!loop)
        continue;
      auto axis = loop->getAttrOfType<StringAttr>("pim.scheduling_axis");
      if (!axis)
        continue;
      if (axis.getValue() == "reduction_k") {
        if (temporalRound) {
          device->emitError("convert-pim-to-upmem: Common IR has more than one "
                            "reduction_k scheduling round");
          return signalPassFailure();
        }
        temporalRound = loop;
      } else if (axis.getValue() == "k") {
        if (spatialKRound) {
          device->emitError("convert-pim-to-upmem: Common IR has more than one "
                            "spatial k scheduling round");
          return signalPassFailure();
        }
        spatialKRound = loop;
      }
    }
    if (static_cast<bool>(temporalLoop) != static_cast<bool>(temporalRound)) {
      device->emitError("convert-pim-to-upmem: pim.temporal_reduction and "
                        "reduction_k scheduling round must occur together");
      return signalPassFailure();
    }
    if (hasSpatialK != static_cast<bool>(spatialKRound)) {
      device->emitError("convert-pim-to-upmem: a spatial pim.axis=\"k\" and "
                        "its Common k scheduling round must occur together");
      return signalPassFailure();
    }
    arith::AddIOp temporalOrigin;
    if (temporalLoop) {
      temporalOrigin = deviceOrigin(temporalLoop);
      if (!temporalOrigin) {
        temporalLoop->emitError("convert-pim-to-upmem: Common temporal "
                                "reduction has no pim.device_origin");
        return signalPassFailure();
      }
    }

    // partial 결합도 loop 구조에서 추론하지 않는다. Common pim.store의 명시적인
    // pim.combine을 access id별로 옮긴다.
    llvm::DenseMap<int64_t, StringAttr> combineOf;
    llvm::DenseMap<int64_t, int64_t> combinedValueRankOf;
    WalkResult combineResult = device.walk([&](pim::StoreOp store) {
      auto combine = store->getAttrOfType<StringAttr>("pim.combine");
      if (!combine)
        return WalkResult::advance();
      auto access = store.getAccess().getDefiningOp<pim::AccessOp>();
      if (!access) {
        store->emitError("convert-pim-to-upmem: store access is not defined by "
                         "pim.access");
        return WalkResult::interrupt();
      }
      if (combine.getValue() != "add") {
        store->emitError("convert-pim-to-upmem: UPMEM supports only the Common "
                         "add combiner");
        return WalkResult::interrupt();
      }
      StringAttr &known = combineOf[access.getId()];
      if (known && known != combine) {
        store->emitError("convert-pim-to-upmem: stores for one access disagree "
                         "on pim.combine");
        return WalkResult::interrupt();
      }
      known = combine;
      int64_t valueRank =
          cast<RankedTensorType>(store.getValue().getType()).getRank();
      auto [rankIt, inserted] =
          combinedValueRankOf.try_emplace(access.getId(), valueRank);
      if (!inserted && rankIt->second != valueRank) {
        store->emitError("convert-pim-to-upmem: stores for one combined access "
                         "have different value ranks");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (combineResult.wasInterrupted())
      return signalPassFailure();
    if (!combineOf.empty() && !temporalRound) {
      device->emitError("convert-pim-to-upmem: pim.combine requires the Common "
                        "reduction_k scheduling round");
      return signalPassFailure();
    }

    // ── 모듈 수준 선언 ─────────────────────────────────────────────────────
    OpBuilder mb(module.getBody(), module.getBody()->begin());
    StringRef base = func.getName();
    SmallVector<pim::PlaceOp> all(inputs);
    all.append(outputs);

    // MRAM buffer 크기는 Common pim.place.allocation_shape 그대로다. Common verifier가
    // 이 값을 Block-Group-Execution Tile에서 파생한 정적 fixed slot으로 확정했으므로,
    // target은 K축을 다시 자르거나 runtime trip count/균등 분할 상한으로 대체하지 않는다.
    auto fixedBytes = [&](pim::PlaceOp place) -> std::optional<int64_t> {
      int64_t elems = 1;
      for (Value capacity : place.getAllocationShape()) {
        std::optional<int64_t> v = getConstantIntValue(capacity);
        if (!v) {
          place->emitError(
              "convert-pim-to-upmem: allocation_shape must be the static "
              "fixed slot decided by PIM Common IR");
          return std::nullopt;
        }
        elems *= *v;
      }
      Type elem =
          cast<BaseMemRefType>(place.getBuffer().getType()).getElementType();
      return elems * std::max<int64_t>(1, elem.getIntOrFloatBitWidth() / 8);
    };
    int64_t bankBudget = profile.getBankCapacityLimitBytes();
    llvm::DenseMap<Operation *, int64_t> bytesOf;
    int64_t staticTotal = 0;
    for (pim::PlaceOp place : all) {
      std::optional<int64_t> bytes = fixedBytes(place);
      if (!bytes)
        return signalPassFailure();
      bytesOf[place] = *bytes;
      staticTotal += *bytes;
    }
    if (staticTotal > bankBudget) {
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
      mb.create<upmem::MRAMOp>(loc, name,
                               TypeAttr::get(mramType(info.elem, bytesOf[place])));
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

    // ── device: loop nest를 옮기고 device 로컬로 만든다 ────────────────────
    auto devName = StringAttr::get(ctx, (base + "_device").str());
    auto dev = mb.create<upmem::DeviceFuncOp>(loc, devName);
    // 실제로 쓰는 tasklet 수. launcher가 이 값을 -DNR_TASKLETS로 쓴다.
    dev->setAttr("upmem.lanes", mb.getI64IntegerAttr(lanes));
    Block *devBody = mb.createBlock(&dev.getBody());

    // 보통 zero-init과 store는 m/n loop 안에 있지만, K-only reduction에는 그 바깥
    // 축이 없어 `fill → temporal K loop → store`가 device body의 같은 레벨에 놓인다.
    // 첫 loop만 떼어 내면 init은 host에 남고 store도 device에서 빠진다. 연산 이름으로
    // 이 모양을 다시 판정하지 않고, loop의 SSA 입력과 결과에 연결된 top-level compute
    // slice를 양방향으로 모아 원래 순서 그대로 옮긴다.
    llvm::DenseSet<Operation *> computeSlice;
    std::function<void(Value)> requireValue;
    std::function<void(Operation *)> requireOperation = [&](Operation *op) {
      if (!op || op->getBlock() != deviceBody ||
          op->hasTrait<OpTrait::IsTerminator>() ||
          !computeSlice.insert(op).second)
        return;
      for (Value operand : op->getOperands())
        requireValue(operand);
    };
    // index/integer 값은 device 분배와 host staging도 함께 사용한다. 이 정의를
    // 이동하면 host IR에 dangling SSA use가 생긴다. scalar control 값은 제자리에
    // 두고 materializeExternals가 device용 복제/host_var를 만든다. tensor와
    // !pim.access처럼 device compute가 직접 소유해야 하는 값만 정의와 함께 옮긴다.
    requireValue = [&](Value value) {
      if (value.getType().isIntOrIndex())
        return;
      requireOperation(value.getDefiningOp());
    };
    requireOperation(nest);
    SmallVector<Value> produced(nest->getResults());
    while (!produced.empty()) {
      Value value = produced.pop_back_val();
      for (OpOperand &use : llvm::make_early_inc_range(value.getUses())) {
        Operation *user = use.getOwner();
        if (user->getBlock() != deviceBody ||
            user->hasTrait<OpTrait::IsTerminator>())
          continue;
        bool inserted = computeSlice.insert(user).second;
        if (!inserted)
          continue;
        for (Value operand : user->getOperands())
          requireValue(operand);
        produced.append(user->getResults().begin(), user->getResults().end());
      }
    }
    SmallVector<Operation *> orderedCompute;
    for (Operation &op : *deviceBody)
      if (computeSlice.contains(&op))
        orderedCompute.push_back(&op);
    for (Operation *op : orderedCompute)
      op->moveBefore(devBody, devBody->end());
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
    // Common이 계산한 global origin은 host staging에 이미 반영되어 있다. DPU MRAM에는
    // Execution Tile 하나만 0부터 놓이므로 local temporal loop의 origin도 0 기준으로
    // 바꾼다. loop 상한은 Common 값 그대로 두고 materializeExternals가 전달한다.
    if (temporalOrigin) {
      Type t = temporalOrigin.getResult().getType();
      temporalOrigin.getLhsMutable().assign(
          db.create<arith::ConstantOp>(loc, db.getIntegerAttr(t, 0)).getResult());
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
    for (pim::PlaceOp place : inputs) {
      OpBuilder b(place);
      Value staged = b.create<upmem::StageOp>(
          loc, stagedType(place), place.getBuffer(), place.getStart(),
          place.getSize(), place.getStrides(), place.getAllocationShape(),
          b.getDenseI64ArrayAttr(byAccess[place.getAccess()].tile),
          place.getReplicatedOverAttr());
      b.create<upmem::ToMRAMOp>(loc, dpu, staged, mramOf[place]);
    }
    // Common IR mapping 축의 device별 반복 횟수를 그 DPU에 보낸다. temporal K
    // local bound도 Common IR의 외부값으로 남아 materializeExternals가 그대로 보낸다.
    OpBuilder tb(fuBody, fuBody->end());
    for (auto [i, a] : llvm::enumerate(axes))
      tb.create<upmem::ToHostVarOp>(loc, dpu, a.trips, tilesVar[i]);
    // MRAM 안의 축 간격도 Common allocation_shape 그대로 보낸다.
    for (pim::PlaceOp place : all) {
      const Buffer &info = byAccess[place.getAccess()];
      for (auto [d, sym] : llvm::enumerate(info.pitch)) {
        Value v = place.getAllocationShape()[d + 1];
        tb.create<upmem::ToHostVarOp>(loc, dpu, v, sym);
      }
    }

    hb.setInsertionPointAfter(foreach);
    hb.create<upmem::LaunchOp>(loc, set);
    for (pim::PlaceOp place : outputs) {
      // Common store가 add partial이라고 명시한 output만 partial 순서를 만든다.
      // 0인 경우는 spatial round=0, temporal round=0, device_k=0뿐이다.
      // device_k는 foreach_group 안에서 더하고, 여기서는 scheduling round만 합친다.
      Value partialRound;
      bool combined = combineOf.count(place.getAccess());
      if (combined) {
        partialRound = temporalRound.getInductionVar();
        if (spatialKRound)
          partialRound = hb.create<arith::AddIOp>(
              loc, partialRound, spatialKRound.getInductionVar());
      }
      OpBuilder::InsertionGuard guard(hb);
      if (failed(buildOutput(hb, place, set, numUnits, axisNames, counts,
                             mramOf[place], foreach,
                             byAccess[place.getAccess()].tile, partialRound,
                             hasSpatialK && combined,
                             combined ? combinedValueRankOf[place.getAccess()]
                                      : -1)))
        return signalPassFailure();
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
      axes.push_back({name, loop, loop.getUpperBound(), deviceOrigin(loop)});
    }
    return success();
  }

  // Common fixed slot의 정확한 byte 수를 UPMEM __mram_noinit 배열 타입으로 옮긴다.
  MemRefType mramType(Type elem, int64_t bytes) {
    int64_t elemBytes = std::max<int64_t>(1, elem.getIntOrFloatBitWidth() / 8);
    return MemRefType::get(
        {bytes / elemBytes}, elem, MemRefLayoutAttrInterface(),
        upmem::MemorySpaceAttr::get(&getContext(), upmem::MemorySpace::MRAM));
  }

  // Host staging도 Common allocation_shape와 같은 fixed slot이다. start/size는
  // round마다 달라도 staging buffer의 shape는 바뀌지 않는다.
  MemRefType stagedType(pim::PlaceOp place) {
    auto elem =
        cast<BaseMemRefType>(place.getBuffer().getType()).getElementType();
    SmallVector<int64_t> shape;
    for (Value extent : place.getAllocationShape())
      shape.push_back(*getConstantIntValue(extent));
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
      } else if (!isa<linalg::FillOp, linalg::MatmulOp, linalg::MatvecOp,
                      linalg::ReduceOp>(op)) {
        op->emitError(
            "convert-pim-to-upmem: UPMEM compute lowering currently supports "
            "fill, matmul, matvec, reduce, and identity pointwise generic");
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

  // 출력 하나: 전 device에서 회수 → 그룹마다 원본으로 되돌린다. 영역과 capacity는
  // Common pim.place를 그대로 쓴다. Common pim.combine이 있으면 spatial/temporal
  // partial 좌표가 모두 0인 첫 partial만 대입하고 나머지는 누산한다.
  LogicalResult buildOutput(OpBuilder &b, pim::PlaceOp place, Value set,
                            int64_t numUnits, ArrayAttr axes,
                            DenseI64ArrayAttr counts, FlatSymbolRefAttr mram,
                            Operation *foreach, ArrayRef<int64_t> tile,
                            Value partialRound, bool spatialPartial,
                            int64_t combinedValueRank) {
    Location loc = place.getLoc();
    unsigned rank = place.getAllocationShape().size();
    auto elem =
        cast<BaseMemRefType>(place.getBuffer().getType()).getElementType();

    // 회수할 바이트 수는 배정량에서 나온다. device 좌표와 무관하므로 foreach 밖으로
    // 옮겨 만들 수 있다 — 못 옮기면 device마다 크기가 다르다는 뜻이라 회수가 성립하지 않는다.
    IRMapping empty;
    SmallVector<Value> capacity;
    for (Value c : place.getAllocationShape()) {
      Value v = cloneInto(b, c, empty, foreach);
      if (!v)
        return place->emitError("output capacity depends on the device "
                                "coordinate; every device must hand back the "
                                "same size");
      capacity.push_back(v);
    }

    SmallVector<int64_t> fixedShape;
    for (Value extent : place.getAllocationShape())
      fixedShape.push_back(*getConstantIntValue(extent));
    SmallVector<int64_t> partsShape{numUnits};
    llvm::append_range(partsShape, fixedShape);
    Value parts = b.create<upmem::GatherOp>(
        loc, MemRefType::get(partsShape, elem), set, mram, capacity);

    auto group = b.create<upmem::ForeachGroupOp>(loc, parts, axes, counts);
    SmallVector<Type> argTys{MemRefType::get(fixedShape, elem)};
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
    Value accumulate = partialRound;
    if (spatialPartial) {
      // axisNames는 Common의 canonical [m,n,k] 순서다. 같은 launch 안의 p_k
      // device partial도 temporal/spatial round와 같은 add combine source다.
      unsigned k = 0;
      for (; k < axes.size(); ++k)
        if (cast<StringAttr>(axes[k]).getValue() == "k")
          break;
      if (k == axes.size())
        return place->emitError(
            "output combine cannot find the canonical k device coordinate");
      accumulate = gbuilder.create<arith::AddIOp>(
          loc, accumulate, gb->getArgument(k + 1));
    }

    unsigned partialDims = 0;
    if (spatialPartial) {
      if (combinedValueRank < 0 ||
          combinedValueRank > static_cast<int64_t>(rank))
        return place->emitError(
            "combined output value rank exceeds its placement rank");
      // RecoverAccessOrigin이 spatial split-K의 partial plane을 buffer 앞축에
      // 명시한다. host combine은 그 축의 시작만 0으로 두고 size/capacity는
      // 보존한다. 따라서 inactive tail device(size=0)는 그대로 no-op이고,
      // 모든 plane partial은 같은 최종 output 영역에 더해진다.
      partialDims = rank - combinedValueRank;
    }
    SmallVector<Value> start, size, strides;
    for (unsigned d = 0; d < rank; d++) {
      // 좌표 계산은 이미 foreach_unit 몸체로 옮겨졌으므로 그 범위를 기준으로 본다.
      Value o = d < partialDims
                    ? gbuilder.create<arith::ConstantIndexOp>(loc, 0).getResult()
                    : cloneInto(gbuilder, place.getStart()[d], map, foreach);
      Value s = cloneInto(gbuilder, place.getSize()[d], map, foreach);
      Value t = cloneInto(gbuilder, place.getStrides()[d], map, foreach);
      if (!o || !s || !t)
        return place->emitError(
            "output region cannot be rebuilt from group coordinates");
      start.push_back(o);
      size.push_back(s);
      strides.push_back(t);
    }
    gbuilder.create<upmem::UnstageOp>(loc, gb->getArgument(0), place.getBuffer(),
                                      start, size, strides, capacity, accumulate,
                                      gbuilder.getDenseI64ArrayAttr(tile));
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertPIMToUPMEMPass() {
  return std::make_unique<ConvertPIMToUPMEMPass>();
}
