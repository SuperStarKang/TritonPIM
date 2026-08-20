#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMMAPDEVICES
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// n을 두 인수로 가른다. tile 개수가 런타임이라 컴파일 시점에는 정사각에 가까운 것 말고
// 고를 근거가 없다.
static std::pair<int64_t, int64_t> factorize(int64_t n) {
  int64_t bestRow = n, bestColumn = 1;
  for (int64_t c = 1; c * c <= n; c++) {
    if (n % c)
      continue;
    int64_t r = n / c;
    if (r - c < bestRow - bestColumn) {
      bestRow = r;
      bestColumn = c;
    }
  }
  return {bestRow, bestColumn};
}

// 한 scheduling round에서 이 PIM Device가 맡는 execution tile 하나로 loop를 좁힌다.
//
//   executionTile = round * partitions + device
//   first         = executionTile * unitTiles
//
// 따라서 6개 execution tile을 3개 device에 배치하면 round 0은 ET0/1/2, round 1은
// ET3/4/5가 된다. 마지막 round의 빈 device는 trips=0인 no-op이다.
static void scheduleLoop(OpBuilder &b, scf::ForOp loop, Value round,
                         Value device, int64_t partitions,
                         int64_t unitTiles, Value extent) {
  Location loc = loop.getLoc();
  Value lb = loop.getLowerBound(), step = loop.getStep();
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value parts = b.create<arith::ConstantIndexOp>(loc, partitions);
  Value unit = b.create<arith::ConstantIndexOp>(loc, unitTiles);
  Value executionTile = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, round, parts), device);
  Value first =
      b.create<arith::MulIOp>(loc, executionTile, unit).getResult();
  Value left = b.create<arith::MaxSIOp>(
      loc, b.create<arith::SubIOp>(loc, extent, first), c0);
  Value trips = b.create<arith::MinSIOp>(loc, unit, left);
  Value start =
      b.create<arith::AddIOp>(loc, lb, b.create<arith::MulIOp>(loc, first, step));

  loop.getLowerBoundMutable().assign(c0);
  loop.getUpperBoundMutable().assign(trips);
  loop.getStepMutable().assign(c1);

  // 전역 좌표 = device의 시작 + 지역 좌표 × 원래 step. target은 첫 피연산자를 0으로
  // 바꿔 device 안의 위치를 얻으므로 그 자리에 표시를 남긴다.
  OpBuilder bb(loop.getBody(), loop.getBody()->begin());
  Value local = loop.getInductionVar();
  auto scaled = bb.create<arith::MulIOp>(loc, local, step);
  auto global = bb.create<arith::AddIOp>(loc, start, scaled);
  global->setAttr("pim.device_origin", bb.getUnitAttr());
  local.replaceUsesWithIf(global, [&](OpOperand &use) {
    return use.getOwner() != scaled && use.getOwner() != global;
  });
}

struct PIMMapDevicesPass
    : public mlir::triton::impl::PIMMapDevicesBase<PIMMapDevicesPass> {
  using PIMMapDevicesBase::PIMMapDevicesBase;

  void runOnOperation() override {
    pim::TargetProfileAttr profile = pim::targetProfile(getOperation());
    if (!profile) {
      getOperation()->emitError(
          "pim-map-devices: the module has no #pim.target_profile");
      return signalPassFailure();
    }
    if (getOperation()
            .walk([&](func::FuncOp func) {
              return failed(mapFunc(func, profile)) ? WalkResult::interrupt()
                                                   : WalkResult::advance();
            })
            .wasInterrupted())
      return signalPassFailure();
  }

  LogicalResult mapFunc(func::FuncOp func, pim::TargetProfileAttr profile) {
    // walk 기본값은 post-order라 안쪽 loop부터 방문하므로 pre-order를 명시한다.
    SmallVector<scf::ForOp> loops;
    SmallVector<Attribute> names;
    func.walk<WalkOrder::PreOrder>([&](scf::ForOp l) {
      if (auto a = l->getAttrOfType<StringAttr>("pim.axis")) {
        loops.push_back(l);
        names.push_back(a);
      }
    });
    if (loops.empty())
      return success();

    auto execution =
        func->getAttrOfType<pim::ExecutionTileAttr>("pim.execution_tile");
    if (!execution)
      return func->emitError(
          "pim-map-devices: the function has no #pim.execution_tile; run "
          "--pim-form-execution-tiles before --pim-map-devices");

    // 자동 인수분해에서 explicit K 분할은 device 축이 아니다 — A·B의 K 영역이 연속으로
    // 남아 일반 cyclic placement 없이 배치된다. 명시 분할(device-axis-partitions)로는
    // 나눌 수 있고, 그때의 연속성은 pim-place-data가 판정한다.
    SmallVector<unsigned> tileAxes;
    for (auto [i, n] : llvm::enumerate(names))
      if (cast<StringAttr>(n) != "explicit_k_partition")
        tileAxes.push_back(i);

    // 어느 축을 먼저 쪼갤지는 target 이 안다 — 축마다 복제되는 피연산자가 다르기
    // 때문이다. matmul 에서 output_row_tile 을 쪼개면 B 가 device 수만큼 복제되고,
    // output_column_tile 을 쪼개면 A 가 복제된다. 목록에 적힌 축이 앞으로 오고 아래에서
    // factorize 의 **큰** 인수를 받는다. 목록이 비어 있으면 지금까지의 순서 그대로다.
    if (!deviceAxisPriority.empty()) {
      SmallVector<StringRef> wanted;
      StringRef(deviceAxisPriority).split(wanted, ',', -1, /*KeepEmpty=*/false);
      SmallVector<unsigned> ordered;
      for (StringRef w : wanted) {
        w = w.trim();
        for (unsigned i : tileAxes)
          if (cast<StringAttr>(names[i]).getValue() == w &&
              !llvm::is_contained(ordered, i))
            ordered.push_back(i);
      }
      for (unsigned i : tileAxes)
        if (!llvm::is_contained(ordered, i))
          ordered.push_back(i);
      tileAxes = ordered;
    }

    SmallVector<int64_t> partitions(loops.size(), 1);
    int64_t devices = profile.getNumberOfPimDevices();
    if (!deviceAxisPartitions.empty()) {
      // 명시 분할은 자동 인수분해를 통째로 대체한다. tile 수가 런타임이라 자동으로는
      // 형상에 맞출 수 없는 것(작은 M, k 분할)을 형상을 아는 쪽이 지정하는 통로다.
      SmallVector<StringRef> entries;
      StringRef(deviceAxisPartitions).split(entries, ',', -1, /*KeepEmpty=*/false);
      for (StringRef entry : entries) {
        auto [name, count] = entry.trim().split(':');
        int64_t n = 0;
        if (count.trim().getAsInteger(10, n) || n < 1)
          return func->emitError("pim-map-devices: device-axis-partitions "
                                 "entry \"")
                 << entry << "\" is not <axis>:<count>";
        for (auto [i, nm] : llvm::enumerate(names))
          if (cast<StringAttr>(nm).getValue() == name.trim())
            partitions[i] = n;
        // 이 커널에 없는 축은 넘어간다 (granularity 와 같은 규약).
      }
      int64_t product = 1;
      for (int64_t p : partitions)
        product *= p;
      if (product > devices)
        return func->emitError("pim-map-devices: the explicit partitions need ")
               << product << " PIM Devices but the machine has " << devices;
    } else if (tileAxes.size() == 1) {
      partitions[tileAxes[0]] = devices;
    } else if (tileAxes.size() >= 2) {
      auto [row, column] = factorize(devices);
      partitions[tileAxes[0]] = row;
      partitions[tileAxes[1]] = column;
    }

    // 축별 분할 눈금. 옵션은 **원소 수**로 받는다 — 하드웨어가 아는 것이 그것이고, tile
    // 하나가 몇 원소인지는 커널의 BLOCK 이 정하므로 target 이 알 수 없다. 여기서 tile 수로
    // 바꾼다.
    SmallVector<int64_t> granularity(loops.size(), 1);
    for (auto [i, name] : llvm::enumerate(names)) {
      StringRef axis = cast<StringAttr>(name).getValue();
      int64_t cap = axis == "output_row_tile"      ? execution.getM()
                    : axis == "output_column_tile" ? execution.getN()
                                                    : 0;
      if (cap <= 0)
        continue;
      int64_t perTile = pim::tileElementsAlong(loops[i], loops);
      if (perTile <= 0 || cap % perTile != 0)
        return func->emitError("pim-map-devices: execution tile cap ")
               << cap << " is not a multiple of the " << axis
               << " block-group tile size " << perTile;
      granularity[i] = cap / perTile;
    }
    if (!deviceAxisGranularity.empty()) {
      SmallVector<StringRef> entries;
      StringRef(deviceAxisGranularity).split(entries, ',', -1, /*KeepEmpty=*/false);
      for (StringRef entry : entries) {
        auto [name, count] = entry.trim().split(':');
        int64_t elems = 0;
        if (count.trim().getAsInteger(10, elems) || elems <= 0)
          return func->emitError("pim-map-devices: device-axis-granularity "
                                 "entry \"")
                 << entry << "\" is not <axis>:<elements>";
        int64_t axis = -1;
        for (unsigned i : tileAxes)
          if (cast<StringAttr>(names[i]).getValue() == name.trim())
            axis = i;
        if (axis < 0)
          continue; // 이 커널에 없는 축이다
        if (partitions[axis] < 2)
          continue; // 눈금은 실제로 나눌 때만 의미가 있다 — 나누지 않는 축은 검사도 없다
        int64_t perTile = pim::tileElementsAlong(loops[axis], loops);
        if (perTile <= 0)
          return func->emitError("pim-map-devices: cannot tell how many "
                                 "elements one ")
                 << name.trim() << " tile moves, so it cannot be split in "
                 << elems << "-element units";
        if (elems % perTile != 0)
          return func->emitError("pim-map-devices: the ")
                 << name.trim() << " tile is " << perTile
                 << " elements, which does not divide the requested split unit "
                 << elems
                 << " — pick a BLOCK size that divides the hardware unit";
        int64_t requested = elems / perTile;
        if (granularity[axis] != 1 && granularity[axis] != requested)
          return func->emitError("pim-map-devices: device-axis-granularity for ")
                 << name.trim() << " conflicts with the execution tile";
        granularity[axis] = requested;
      }
    }

    MLIRContext *ctx = func.getContext();
    Location loc = loops.front().getLoc();
    OpBuilder b(loops.front());

    // 나누기 전의 축별 block-group 개수다.
    SmallVector<Value> extents;
    for (scf::ForOp l : loops)
      extents.push_back(b.create<arith::CeilDivSIOp>(
          loc,
          b.create<arith::SubIOp>(loc, l.getUpperBound(), l.getLowerBound()),
          l.getStep()));

    SmallVector<Attribute> schedulingOrder;
    StringRef preferred[] = {"output_column_tile", "explicit_k_partition",
                             "output_row_tile"};
    for (StringRef wanted : preferred)
      for (Attribute name : names)
        if (cast<StringAttr>(name).getValue() == wanted)
          schedulingOrder.push_back(name);
    for (Attribute name : names)
      if (!llvm::is_contained(schedulingOrder, name))
        schedulingOrder.push_back(name);

    // execution-tile grid를 먼저 만들고 그 바깥에 scheduling round를 둔다. 각 축의
    // round 수는 ceil(execution_tiles / active_devices)다. 이 loop들은 host 쪽에 남고,
    // 그 안의 pim.device 한 개가 해당 round의 모든 PIM Device를 병렬 실행한다.
    SmallVector<Value> roundIVs(loops.size());
    OpBuilder scheduled(loops.front());
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    for (Attribute orderedName : schedulingOrder) {
      unsigned axis = 0;
      while (names[axis] != orderedName)
        ++axis;
      Value unit = scheduled.create<arith::ConstantIndexOp>(loc,
                                                             granularity[axis]);
      Value executionTiles = scheduled.create<arith::CeilDivSIOp>(
          loc, extents[axis], unit);
      Value active = scheduled.create<arith::ConstantIndexOp>(
          loc, partitions[axis]);
      Value rounds = scheduled.create<arith::CeilDivSIOp>(
          loc, executionTiles, active);
      auto round = scheduled.create<scf::ForOp>(loc, c0, rounds, c1);
      round->setAttr("pim.scheduling_axis", orderedName);
      roundIVs[axis] = round.getInductionVar();
      scheduled.setInsertionPointToStart(round.getBody());
    }

    auto device = scheduled.create<pim::DeviceOp>(
        loc,
        pim::DeviceMappingAttr::get(ctx, ArrayAttr::get(ctx, names),
                                    b.getDenseI64ArrayAttr(partitions)),
        extents);
    auto plan = pim::DevicePlanAttr::get(
        ctx, devices, ArrayAttr::get(ctx, names),
        b.getDenseI64ArrayAttr(partitions),
        b.getDenseI64ArrayAttr(
            {execution.getM(), execution.getN(), execution.getK()}),
        ArrayAttr::get(ctx, schedulingOrder));
    device->setAttr("pim.execution_tile", execution);
    device->setAttr("pim.device_plan", plan);
    bool hasSpatialK = llvm::any_of(names, [](Attribute name) {
      return cast<StringAttr>(name).getValue() == "explicit_k_partition";
    });
    device->setAttr(
        "pim.reduction_plan",
        pim::ReductionPlanAttr::get(
            ctx,
            StringAttr::get(ctx, hasSpatialK ? "separate_output_planes"
                                             : "none"),
            StringAttr::get(ctx, execution.getK() > 0 ? "host" : "none"),
            StringAttr::get(ctx, hasSpatialK ? "external" : "none")));
    SmallVector<Type> argTys(loops.size(), b.getIndexType());
    SmallVector<Location> argLocs(loops.size(), loc);
    Block *body = b.createBlock(&device.getBody(), {}, argTys, argLocs);
    loops.front()->moveBefore(body, body->end());

    b.setInsertionPoint(loops.front());
    for (auto [i, l] : llvm::enumerate(loops))
      scheduleLoop(b, l, roundIVs[i], body->getArgument(i), partitions[i],
                   granularity[i], extents[i]);
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMMapDevicesPass() {
  return std::make_unique<PIMMapDevicesPass>();
}
