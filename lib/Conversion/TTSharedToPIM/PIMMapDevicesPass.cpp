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

// loop를 device 하나의 몫으로 좁힌다(balanced block). 몫과 나머지로 나누어 나머지
// partition이 하나씩 더 맡으므로 꼬리에서 device가 놀지 않고, 축마다 연속 구간 하나가
// 유지된다. 좁힌 뒤 loop는 [0, trips)를 1씩 돌고, 전역 좌표는 몸체 첫머리에서 다시 만든다.
static void narrowLoop(OpBuilder &b, scf::ForOp loop, Value index,
                       int64_t partitions, int64_t granularity, Value extent) {
  Location loc = loop.getLoc();
  Value lb = loop.getLowerBound(), step = loop.getStep();
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value parts = b.create<arith::ConstantIndexOp>(loc, partitions);
  // 나누는 단위. 기본은 tile 하나지만, target이 눈금을 주면 그 배수로만 자른다. 눈금보다
  // 잘게 자르면 경계에 걸친 실행 단위가 양쪽 device에서 각각 통째로 실행되어 총 일감이
  // 늘어난다 — 7개짜리를 둘로 나눌 때 3.5가 올림되어 4+4=8이 되는 식이다. 대신 3+4로
  // **불균등하게** 나눈다. 눈금이 tile을 나누지 못하면 마지막 group이 부분이므로 아래에서
  // extent로 자른다.
  int64_t unitTiles = std::max<int64_t>(granularity, 1);
  Value unit, units = extent;
  if (unitTiles != 1) {
    unit = b.create<arith::ConstantIndexOp>(loc, unitTiles);
    units = b.create<arith::CeilDivSIOp>(loc, extent, unit);
  }
  Value minUnits = b.create<arith::DivSIOp>(loc, units, parts);
  Value remainder = b.create<arith::RemSIOp>(loc, units, parts);
  Value firstUnit = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, index, minUnits),
      b.create<arith::MinSIOp>(loc, index, remainder));
  Value tripUnits = b.create<arith::AddIOp>(
      loc, minUnits,
      b.create<arith::SelectOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, index,
                                  remainder),
          c1, c0));
  Value first = firstUnit, trips = tripUnits;
  if (unitTiles != 1) {
    first = b.create<arith::MulIOp>(loc, firstUnit, unit);
    // 마지막 group이 부분일 수 있고, extent 뒤에서 시작하는 device는 아무것도 맡지 않는다.
    Value left = b.create<arith::MaxSIOp>(
        loc, b.create<arith::SubIOp>(loc, extent, first), c0);
    trips = b.create<arith::MinSIOp>(
        loc, b.create<arith::MulIOp>(loc, tripUnits, unit), left);
  }
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
  local.replaceAllUsesExcept(global, scaled);
}

// 이 축이 tile 하나에서 옮기는 원소 수. pim.access 의 sizes 가 그 값이고, 어느 차원인지는
// origin 을 이 loop 가 모는가로 가른다 — pim-place-data 가 region 을 만들 때 쓰는 판정과
// 같은 것이다. 축을 모는 차원이 여럿인데 크기가 다르면 답이 하나가 아니므로 0 을 돌려준다.
static int64_t tileElements(scf::ForOp axis, ArrayRef<scf::ForOp> loops) {
  int64_t elems = 0;
  bool conflict = false;
  axis.walk([&](pim::AccessOp a) {
    ArrayRef<int64_t> sizes = a.getSizes();
    for (unsigned d = 0, e = sizes.size(); d < e; d++) {
      llvm::SmallPtrSet<Operation *, 4> drivers;
      pim::collectDrivers(a.getOrigin()[d], loops, drivers);
      if (drivers.size() != 1 || !drivers.contains(axis))
        continue;
      if (elems && elems != sizes[d])
        conflict = true;
      elems = sizes[d];
    }
  });
  return conflict ? 0 : elems;
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
        int64_t perTile = tileElements(loops[axis], loops);
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
        granularity[axis] = elems / perTile;
      }
    }

    MLIRContext *ctx = func.getContext();
    Location loc = loops.front().getLoc();
    OpBuilder b(loops.front());

    // 나누기 전의 축별 tile 개수. 배치 pass가 여기서 배정량을 다시 계산한다.
    SmallVector<Value> extents;
    for (scf::ForOp l : loops)
      extents.push_back(b.create<arith::CeilDivSIOp>(
          loc,
          b.create<arith::SubIOp>(loc, l.getUpperBound(), l.getLowerBound()),
          l.getStep()));

    auto device = b.create<pim::DeviceOp>(
        loc,
        pim::DeviceMappingAttr::get(ctx, ArrayAttr::get(ctx, names),
                                    b.getDenseI64ArrayAttr(partitions)),
        extents);
    SmallVector<Type> argTys(loops.size(), b.getIndexType());
    SmallVector<Location> argLocs(loops.size(), loc);
    Block *body = b.createBlock(&device.getBody(), {}, argTys, argLocs);
    loops.front()->moveBefore(body, body->end());

    b.setInsertionPoint(loops.front());
    for (auto [i, l] : llvm::enumerate(loops))
      narrowLoop(b, l, body->getArgument(i), partitions[i], granularity[i],
                 extents[i]);
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMMapDevicesPass() {
  return std::make_unique<PIMMapDevicesPass>();
}
