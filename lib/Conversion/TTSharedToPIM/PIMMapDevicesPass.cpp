#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

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
                       int64_t partitions, Value extent) {
  Location loc = loop.getLoc();
  Value lb = loop.getLowerBound(), step = loop.getStep();
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value parts = b.create<arith::ConstantIndexOp>(loc, partitions);
  Value minTiles = b.create<arith::DivSIOp>(loc, extent, parts);
  Value remainder = b.create<arith::RemSIOp>(loc, extent, parts);
  Value first = b.create<arith::AddIOp>(
      loc, b.create<arith::MulIOp>(loc, index, minTiles),
      b.create<arith::MinSIOp>(loc, index, remainder));
  Value trips = b.create<arith::AddIOp>(
      loc, minTiles,
      b.create<arith::SelectOp>(
          loc,
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, index,
                                  remainder),
          c1, c0));
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
    getOperation().walk(
        [&](func::FuncOp func) { mapFunc(func, profile); });
  }

  void mapFunc(func::FuncOp func, pim::TargetProfileAttr profile) {
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
      return;

    // explicit K 분할은 device 축이 아니다. A·B의 K 영역이 연속으로 남아 일반 cyclic
    // placement 없이 배치된다.
    SmallVector<unsigned> tileAxes;
    for (auto [i, n] : llvm::enumerate(names))
      if (cast<StringAttr>(n) != "explicit_k_partition")
        tileAxes.push_back(i);

    SmallVector<int64_t> partitions(loops.size(), 1);
    int64_t devices = profile.getNumberOfPimDevices();
    if (tileAxes.size() == 1) {
      partitions[tileAxes[0]] = devices;
    } else if (tileAxes.size() >= 2) {
      auto [row, column] = factorize(devices);
      partitions[tileAxes[0]] = row;
      partitions[tileAxes[1]] = column;
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
      narrowLoop(b, l, body->getArgument(i), partitions[i], extents[i]);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMMapDevicesPass() {
  return std::make_unique<PIMMapDevicesPass>();
}
