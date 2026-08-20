#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"
#include "triton-shared/Dialect/PIM/IR/TargetSpec.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMFORMEXECUTIONTILES
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct PIMFormExecutionTilesPass
    : public mlir::triton::impl::PIMFormExecutionTilesBase<
          PIMFormExecutionTilesPass> {
  using PIMFormExecutionTilesBase::PIMFormExecutionTilesBase;

  void runOnOperation() override {
    if (tileM < 0 || tileN < 0 || tileK < 0) {
      getOperation()->emitError("pim-form-execution-tiles: caps must not be "
                                "negative (0 means automatic)");
      return signalPassFailure();
    }
    profile = pim::targetProfile(getOperation());
    WalkResult res = getOperation().walk([&](func::FuncOp func) {
      bool hasAccess = false;
      func.walk([&](pim::AccessOp) { hasAccess = true; });
      if (!hasAccess)
        return WalkResult::advance();
      return succeeded(attachCaps(func)) ? WalkResult::advance()
                                         : WalkResult::interrupt();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }

  pim::TargetProfileAttr profile;

  LogicalResult attachCaps(func::FuncOp func) {
    // 상한이 붙는 축의 loop 와 그 축의 group tile 원소수를 찾아 배수 조건을 검사한다.
    // 눈금보다 잘게 자르면 창 경계가 tile 을 가르므로 진단이다.
    SmallVector<scf::ForOp> loops;
    func.walk<WalkOrder::PreOrder>(
        [&](scf::ForOp l) { loops.push_back(l); });
    auto axisLoop = [&](StringRef name) -> scf::ForOp {
      for (scf::ForOp l : loops) {
        // reduction loop 는 pim.axis 가 없다 — null StringAttr 를 StringRef 와
        // 비교하면 getValue() 가 터지므로 걸러서 지난다.
        auto a = l->getAttrOfType<StringAttr>("pim.axis");
        if (a && a.getValue() == name)
          return l;
      }
      return scf::ForOp();
    };
    // tile-k는 spatial split 여부와 무관하게 각 output plane 내부 reduction loop의
    // temporal 상주 window다. explicit_k_partition은 별도의 spatial output 축이다.
    // reduction loop는 pim.axis 없이 access origin을 모는 loop 하나로 찾는다.
    auto reductionLoop = [&]() -> scf::ForOp {
      llvm::SmallPtrSet<Operation *, 4> reds;
      func.walk([&](pim::AccessOp a) {
        for (Value o : a.getOrigin()) {
          llvm::SmallPtrSet<Operation *, 4> ds;
          pim::collectDrivers(o, loops, ds);
          for (Operation *op : ds)
            if (!op->getAttrOfType<StringAttr>("pim.axis"))
              reds.insert(op);
        }
      });
      return reds.size() == 1 ? cast<scf::ForOp>(*reds.begin()) : scf::ForOp();
    };
    scf::ForOp mLoop = axisLoop("output_row_tile");
    scf::ForOp nLoop = axisLoop("output_column_tile");
    scf::ForOp explicitK = axisLoop("explicit_k_partition");
    scf::ForOp redLoop = reductionLoop();

    // execution tile 은 block group 정수 개다. group 이 없으면(눈금 없는 기계) block
    // 정수 개이고(useful = phys), explicit_k_partition 은 공간 분할의 눈금이라 언제나
    // block 기준이다.
    auto group = func->getAttrOfType<pim::BlockGroupAttr>("pim.block_group");
    auto unitFor = [&](int idx, scf::ForOp loop) -> std::pair<int64_t, int64_t> {
      if (group && idx >= 0)
        return {group.getUseful().asArrayRef()[idx],
                group.getPhys().asArrayRef()[idx]};
      int64_t e = loop ? pim::tileElementsAlong(loop, loops) : 0;
      return {e, e};
    };

    // ── 사용자 값 검증 — 상한은 그 축의 group(없으면 tile) 원소수의 배수다 ────────
    struct Cap {
      int64_t cap;
      StringRef axis;
      StringRef option;
    };
    Cap caps[] = {{tileM, "output_row_tile", "tile-m"},
                  {tileN, "output_column_tile", "tile-n"},
                  {tileK, "reduction", "tile-k"}};
    for (const Cap &c : caps) {
      if (c.cap == 0)
        continue;
      bool isK = c.axis == "reduction";
      scf::ForOp loop = isK ? redLoop : axisLoop(c.axis);
      if (!loop)
        return func->emitError("pim-form-execution-tiles: ")
               << c.option << " is given but the kernel has no " << c.axis
               << (isK ? " loop" : " axis");
      int idx = isK ? 2 : (c.axis == "output_row_tile" ? 0 : 1);
      int64_t elems = unitFor(idx, loop).first;
      StringRef grain = (group && idx >= 0) ? "block group" : "tile";
      if (elems <= 0)
        return func->emitError(
                   "pim-form-execution-tiles: cannot tell how many elements "
                   "one ")
               << c.axis << " tile moves, so the cap cannot be validated";
      if (c.cap % elems != 0)
        return func->emitError("pim-form-execution-tiles: ")
               << c.option << "(" << c.cap << ") is not a multiple of the "
               << c.axis << " " << grain << " (" << elems << " elements)";
    }

    int64_t capM = tileM, capN = tileN, capK = tileK;

    // ── 자동 산출 — device 배치 전에 execution tile shape 을 먼저 고른다 ──
    // 조건: matmul 모양(m·n 축 + reduction loop 하나, explicit_k_partition 없음),
    // target 의 footprint spec 등록됨, bank 예산 알려짐. 조건 밖이면 0 을 그대로
    // 둔다 — 전부 0 이면 attr 없이 두는 기존 동작이다.
    bool matmulShaped = mLoop && nLoop && redLoop && !explicitK;
    const pim::TargetSpec *spec =
        profile ? pim::lookupTargetSpec(profile.getTarget().getValue())
                : nullptr;
    IntegerAttr bank =
        profile ? profile.getUsableBankCapacityBytes() : IntegerAttr();
    if ((capM == 0 || capN == 0 || capK == 0) && matmulShaped && spec &&
        bank && bank.getInt() > 0) {
      auto [mU, mP] = unitFor(0, mLoop);
      auto [nU, nP] = unitFor(1, nLoop);
      auto [kU, kP] = unitFor(2, redLoop);
      if (mU <= 0 || nU <= 0 || kU <= 0)
        return func->emitError(
            "pim-form-execution-tiles: cannot tell the per-axis tile grain, "
            "so the execution tile cannot be computed automatically");
      auto [kGrainU, kGrainP] = spec->kGrain(profile, kU, kP);

      if (capM == 0)
        capM = mU; // group 1개 — 두 target 공통 기본
      if (capN == 0) {
        int64_t nGroups = spec->preferredNGroups(profile, nU, nP);
        capN = std::max<int64_t>(1, nGroups) * nU;
      }
      // footprint 는 물리 원소 위에서 계산한다 (검증을 통과했으므로 나눗셈은 정확).
      int64_t mPhys = capM / mU * mP;
      int64_t nPhys = capN / nU * nP;

      if (capK == 0) {
        SmallVector<pim::BufferShape> buffers;
        collectBufferShapes(func, mLoop, nLoop, redLoop, buffers);
        int64_t budget = bank.getInt();
        auto fits = [&](int64_t u) {
          return spec->deviceFootprintBytes(profile, buffers, mPhys, nPhys,
                                            u * kGrainP) <= budget;
        };
        if (!fits(1))
          return func->emitError(
                     "pim-form-execution-tiles: even one k unit (")
                 << kGrainP << " elements) does not fit the device budget of "
                 << budget << " byte";
        // footprint 는 k 에 단조증가라 "예산에 드는 최대 배수" 는 이분 탐색이 정확하다.
        int64_t lo = 1;
        while (fits(lo * 2))
          lo *= 2;
        int64_t hi = lo * 2;
        while (hi - lo > 1) {
          int64_t mid = lo + (hi - lo) / 2;
          (fits(mid) ? lo : hi) = mid;
        }
        capK = lo * kGrainU;
      }
    }

    if (explicitK && (capM == 0 || capN == 0 || capK == 0))
      return func->emitError(
          "pim-form-execution-tiles: a kernel with explicit_k_partition "
          "requires concrete tile-m, tile-n, and tile-k; automatic spatial "
          "plus temporal K planning is not implemented yet");

    if (capM == 0 && capN == 0 && capK == 0)
      return success(); // 상한 없음 — 배정 전체가 execution tile 하나 (기존 동작)
    func->setAttr("pim.execution_tile",
                  pim::ExecutionTileAttr::get(func.getContext(), capM, capN,
                                              capK));
    return success();
  }

  // footprint 계산에 쓸 buffer 모양: access 마다 "각 차원이 m/n/k 상한을 따라가는가"
  // 와 원소 바이트. 원소 타입은 그 access 를 읽고 쓰는 load/store 의 tensor 타입에서
  // 온다. 상한 축이 몰지 않는 차원(split-K 출력면 같은 것)은 정적 tile 크기 그대로다.
  void collectBufferShapes(func::FuncOp func, scf::ForOp mLoop,
                           scf::ForOp nLoop, scf::ForOp redLoop,
                           SmallVectorImpl<pim::BufferShape> &buffers) {
    SmallVector<scf::ForOp, 3> axisLoops;
    scf::ForOp axes[3] = {mLoop, nLoop, redLoop};
    for (scf::ForOp l : axes)
      if (l)
        axisLoops.push_back(l);
    func.walk([&](pim::AccessOp a) {
      Type elem;
      for (Operation *user : a->getUsers()) {
        if (auto load = dyn_cast<pim::LoadOp>(user))
          elem = cast<RankedTensorType>(load.getResult().getType())
                     .getElementType();
        else if (auto store = dyn_cast<pim::StoreOp>(user))
          if (!elem)
            elem = cast<RankedTensorType>(store.getValue().getType())
                       .getElementType();
      }
      if (!elem)
        return;
      pim::BufferShape shape;
      shape.elemBytes =
          std::max<int64_t>(1, elem.getIntOrFloatBitWidth() / 8);
      ArrayRef<int64_t> sizes = a.getSizes();
      for (unsigned d = 0, e = sizes.size(); d < e; d++) {
        llvm::SmallPtrSet<Operation *, 4> ds;
        pim::collectDrivers(a.getOrigin()[d], axisLoops, ds);
        pim::BufferDim dim;
        dim.staticSize = sizes[d];
        if (ds.size() == 1) {
          if (mLoop && ds.contains(mLoop))
            dim.axis = pim::BufferDim::M;
          else if (nLoop && ds.contains(nLoop))
            dim.axis = pim::BufferDim::N;
          else if (redLoop && ds.contains(redLoop))
            dim.axis = pim::BufferDim::K;
        }
        shape.dims.push_back(dim);
      }
      buffers.push_back(shape);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMFormExecutionTilesPass() {
  return std::make_unique<PIMFormExecutionTilesPass>();
}
