#include "triton-shared/Conversion/PIMToAiMX/PIMToAiMX.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTPIMTOAIMX
#include "triton-shared/Conversion/PIMToAiMX/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
int64_t roundUp(int64_t a, int64_t b) { return ceilDiv(a, b) * b; }

bool isBatch(int64_t b) { return b == 1 || b == 2 || b == 4 || b == 8; }

// execution tile 하나가 PIM DRAM 에 요구하는 바이트. §6.2 의 표 그대로다.
struct Footprint {
  int64_t paddedK, paddedN;
  int64_t a, arel, b, c, chain;
  int64_t groups; // descriptor group 수

  int64_t total() const {
    return a + arel + b + c + chain + aimx::kPMMSize;
  }
};

Footprint footprintOf(int64_t batch, int64_t mT, int64_t kT, int64_t nT) {
  using namespace aimx;
  Footprint f;
  f.paddedK = kChunk * ceilDiv(kT, kChunk);
  f.paddedN = kChunk * ceilDiv(nT, kChunk);
  f.a = mT * f.paddedK * kTypeSize;
  // Arel 은 A 를 16 배로 복제한다 — vecA 가 reduction 방향으로 늘어선 A 를 원한다.
  f.arel = mT * kT * kTypeSize * kNumChannels;
  f.b = kT * f.paddedN * kTypeSize;
  f.c = mT * f.paddedN * kTypeSize;
  int64_t rows = mT / batch;                              // output block 행
  int64_t cols = f.paddedN * batch / kLanes;              // output block 열
  int64_t steps = kT * batch / kRegSize;                  // reduction step
  f.groups = 2 * (kT / kRegSize) * rows + rows * cols * (2 * steps + 1);
  f.chain = f.groups * kGroupBytes;
  return f;
}

// linalg.matmul 의 operand 를 pim.access 의 id 로 되짚은 결과.
struct Roles {
  int64_t a = -1, b = -1, c = -1;
  // source 의 Triton block. A tile 이 BLOCK_M x BLOCK_K, B tile 이 BLOCK_K x BLOCK_N 이다.
  // loop nest 를 지우기 전에 읽어야 한다 — 지운 뒤에는 이 값이 IR 어디에도 남지 않는다.
  int64_t blockM = 0, blockN = 0, blockK = 0;
};

struct ConvertPIMToAiMXPass
    : public mlir::triton::impl::ConvertPIMToAiMXBase<ConvertPIMToAiMXPass> {
  using ConvertPIMToAiMXBase::ConvertPIMToAiMXBase;

  void runOnOperation() override {
    profile = pim::targetProfile(getOperation());
    if (!profile) {
      getOperation()->emitError(
          "convert-pim-to-aimx: the module has no #pim.target_profile");
      return signalPassFailure();
    }
    if (failed(checkLegality()))
      return signalPassFailure();

    SmallVector<pim::DeviceOp> devices;
    getOperation().walk([&](pim::DeviceOp d) { devices.push_back(d); });
    for (pim::DeviceOp d : devices)
      if (failed(convert(d)))
        return signalPassFailure();

    // profile 은 여기서 소비된다. 남겨 두면 target IR 이 여전히 PIM 을 가리키게 되고,
    // 뒤 단계가 기계 값을 다시 읽을 수 있다는 오해를 만든다.
    getOperation()->removeAttr("pim.target_profile");
    getOperation().walk(
        [](func::FuncOp f) { f->removeAttr("pim.group_execution"); });
  }

  pim::TargetProfileAttr profile;

  LogicalResult checkLegality() {
    Operation *m = getOperation();
    if (profile.getTarget() != "aimx")
      return m->emitError("convert-pim-to-aimx: the module targets ")
             << profile.getTarget().getValue();
    if (profile.getParallelExecutionKind() != "simd")
      return m->emitError(
          "convert-pim-to-aimx: AiMX lanes are address-decoded SIMD slots, so "
          "parallel_execution_kind must be \"simd\"");
    ArrayRef<int64_t> shape = profile.getPimCoreShape().asArrayRef();
    if (shape.size() != 1 || shape[0] != 1)
      return m->emitError(
          "convert-pim-to-aimx: AiMX has one PIM Core per PIM Device, so "
          "pim_core_shape must be [1]");
    if (profile.getNumberOfParallelLanesPerPimCore() != aimx::kLanes)
      return m->emitError("convert-pim-to-aimx: AiMX has ")
             << aimx::kLanes << " lanes per PIM Core, not "
             << profile.getNumberOfParallelLanesPerPimCore();
    if (batch != 0 && !isBatch(batch))
      return m->emitError(
          "convert-pim-to-aimx: batch must be 1, 2, 4, or 8 — or 0 to let the "
          "pass pick the largest value the Triton block allows");
    return success();
  }

  //===--------------------------------------------------------------------===//
  // batch
  //===--------------------------------------------------------------------===//

  // batch 는 arithmetic intensity 그 자체다. B tile 은 출력 행 그룹마다 다시 읽히고
  // 그 횟수가 m_t/batch 이므로 invoke 당 전송량은 1/batch 로 줄고, 대신 descriptor
  // group 수는 batch 배로 는다. 실측 계수(group 하나 0.44 us, 대역폭 56 GB/s)로 두 항을
  // 같게 놓으면 batch = sqrt(2 * 4096 * 128 / (a*B)) = 6.5 이고, legal 한 값 중 가장
  // 가까운 것이 8 이다. batch=1 대비 3.4 배로 실측된다.
  //
  // 그런데 8 을 그냥 기본값으로 박을 수는 없다. 세 조건이 전부 Triton block 에 달려 있다.
  //
  //   m_t % batch == 0              출력 block 은 batch 행이다
  //   BLOCK_N | (4096/batch)        출력 block 폭이 Triton block 의 정수 배여야 한다
  //   BLOCK_K | (256/batch)         reduction step 이 Triton block 의 정수 배여야 한다
  //
  // batch 를 키울수록 뒤의 둘이 빡빡해진다 — BLOCK_K=64 는 batch 8 에서 32 를 나누지
  // 못한다. 그래서 기본값은 값이 아니라 **규칙**이다: 세 조건을 만족하는 최대값.
  int64_t pickBatch(const Roles &roles, int64_t mT) {
    if (batch != 0)
      return batch;
    int64_t best = 1;
    for (int64_t b : {2, 4, 8})
      if (mT % b == 0 && (aimx::kLanes / b) % roles.blockN == 0 &&
          (aimx::kRegSize / b) % roles.blockK == 0)
        best = b;
    return best;
  }

  //===--------------------------------------------------------------------===//
  // execution tile
  //===--------------------------------------------------------------------===//

  // 예산은 pass option 이 이기고, 없으면 profile 을 쓴다. 둘 다 없으면 tile 을 고를 수
  // 없다 — 할당해 보고 물러서는 길이 없으므로(§1.8 (h)) 여기서 멈추는 것이 맞다.
  LogicalResult budgetBytes(Operation *at, int64_t &budget) {
    if (bankCapacityBytes >= 0) {
      budget = bankCapacityBytes;
      return success();
    }
    if (auto bank = profile.getUsableBankCapacityBytes()) {
      budget = bank.getInt();
      return success();
    }
    return at->emitError(
        "convert-pim-to-aimx: neither usable_bank_capacity_bytes nor "
        "bank-capacity-bytes gives the PIM DRAM budget, so the execution tile "
        "cannot be chosen");
  }

  // §6.1. 예산은 **buffer 상한** (k_max, n_max) 만 정한다. 실제 tile 모양 (k_t, n_t) 은
  // 런타임 값이고 descriptor program 이 인자로 받는다. m_t 는 이미 Triton block 묶음으로
  // 정해져서 들어온다 — 예산에 맞추어 줄일 대상이 아니다.
  LogicalResult pickTile(Operation *at, int64_t batchV, int64_t mT,
                         int64_t &kMax, int64_t &nMax, Footprint &fp) {
    int64_t budget;
    if (failed(budgetBytes(at, budget)))
      return failure();

    nMax = roundUp(std::max<int64_t>(tileN, aimx::kChunk), aimx::kChunk);

    if (tileK > 0) {
      kMax = roundUp(tileK, aimx::kRegSize);
      fp = footprintOf(batchV, mT, kMax, nMax);
      if (fp.total() > budget)
        return at->emitError("convert-pim-to-aimx: the requested buffer bound "
                             "(m_t=")
               << mT << ", tile-k=" << kMax << ", tile-n=" << nMax
               << ") needs " << fp.total()
               << " byte of PIM DRAM but the budget is " << budget
               << " byte — lower group-m, tile-k, or tile-n";
      return success();
    }

    // footprint 는 k 에 대해 단조 증가하므로 이분 탐색이면 된다. 단위는 256 이다.
    int64_t lo = 1, hi = (1 << 20) / aimx::kRegSize;
    if (footprintOf(batchV, mT, lo * aimx::kRegSize, nMax).total() > budget)
      return at->emitError("convert-pim-to-aimx: even k_max=")
             << aimx::kRegSize << " with m_t=" << mT << ", n_max=" << nMax
             << " needs "
             << footprintOf(batchV, mT, aimx::kRegSize, nMax).total()
             << " byte of PIM DRAM but the budget is " << budget << " byte";
    while (lo < hi) {
      int64_t mid = lo + (hi - lo + 1) / 2;
      if (footprintOf(batchV, mT, mid * aimx::kRegSize, nMax).total() <= budget)
        lo = mid;
      else
        hi = mid - 1;
    }
    kMax = lo * aimx::kRegSize;
    fp = footprintOf(batchV, mT, kMax, nMax);
    return success();
  }

  //===--------------------------------------------------------------------===//
  // device 하나
  //===--------------------------------------------------------------------===//

  LogicalResult convert(pim::DeviceOp device) {
    auto func = device->getParentOfType<func::FuncOp>();
    MLIRContext *ctx = &getContext();
    Location loc = device.getLoc();
    Block *deviceBody = &device.getBody().front();

    auto cores =
        func->getAttrOfType<pim::GroupExecutionAttr>("pim.group_execution");
    if (!cores)
      return func->emitError(
          "convert-pim-to-aimx: the function has no #pim.group_execution");
    if (cores.getLanes() != aimx::kLanes)
      return func->emitError(
                 "convert-pim-to-aimx: --pim-plan-group-execution chose ")
             << cores.getLanes() << " lanes, but AiMX always drives "
             << aimx::kLanes;

    Roles roles;
    if (failed(readRoles(device, roles)))
      return failure();

    llvm::DenseMap<int64_t, pim::PlaceOp> byAccess;
    SmallVector<pim::PlaceOp> all;
    for (Operation &op : *deviceBody)
      if (auto p = dyn_cast<pim::PlaceOp>(&op)) {
        byAccess[p.getAccess()] = p;
        all.push_back(p);
      }
    pim::PlaceOp aPlace = byAccess.lookup(roles.a);
    pim::PlaceOp bPlace = byAccess.lookup(roles.b);
    pim::PlaceOp cPlace = byAccess.lookup(roles.c);
    if (!aPlace || !bPlace || !cPlace)
      return device->emitError("convert-pim-to-aimx: the matmul operands have "
                               "no matching pim.place");
    if (cPlace.getDirection() != "out")
      return cPlace->emitError(
          "convert-pim-to-aimx: the matmul result must be an \"out\" place");
    // 출력에 축이 하나 더 있으면 그 축은 reduction 이 아니라 **분리**다 — slice 마다
    // 독립된 GEMM 이고 host 가 나중에 더한다(전형적인 split-K). AiMX 로 못 내릴 이유는
    // 없지만 host loop 한 겹과 k_t 기준 변경이 필요하고, 아직 설계에 없다.
    if (cPlace.getStart().size() != 2)
      return cPlace->emitError("convert-pim-to-aimx: the output has rank ")
             << cPlace.getStart().size()
             << " — an extra output axis is host split-K (one independent "
                "matmul per slice), which this pass does not lower yet";
    if (aPlace.getStart().size() != 2 || bPlace.getStart().size() != 2)
      return device->emitError(
          "convert-pim-to-aimx: A and B must be two-dimensional");

    // ── execution tile 은 Triton block 묶음이다 ────────────────────────────────
    // m 방향만 우리가 고른다: m_t = BLOCK_M * group-m. 반대 방향(요청한 m_t 를 BLOCK_M
    // 경계로 올리기)은 m_t 가 언제나 BLOCK_M 의 배수로 튀어 BLOCK_M=8 에서 40, 48, 96 을
    // 만들 수 없었다.
    //
    // n·k 방향은 묶음 수가 입력이 될 수 없다. 그쪽 눈금(4096/batch, 256/batch)은 우리가
    // 고르는 값이 아니라 하드웨어 상수다. 대신 그 상수를 BLOCK_N·BLOCK_K 의 정수 배로
    // **만드는** 것이 batch 선택이고, 그 결과가 아래 group[1], group[2] 다.
    //
    // groupM 은 Pass::Option 이라 값으로 받아 둔다 — 진단 stream 과 산술이 옵션 타입을
    // 거치지 않게 한다.
    int64_t gm = std::max<int64_t>(groupM, 1);
    int64_t mT = roles.blockM * gm;
    int64_t batchV = pickBatch(roles, mT);
    if (mT % batchV != 0)
      return device->emitError("convert-pim-to-aimx: m_t = BLOCK_M(")
             << roles.blockM << ") * group-m(" << gm << ") = " << mT
             << " is not a multiple of batch(" << batchV
             << ") — the output block is batch rows tall, so an execution tile "
                "has to hold a whole number of them. Pick a group-m that makes "
                "the product a multiple of "
             << batchV << ", or leave batch at 0 to let the pass pick";
    int64_t hwN = aimx::kLanes / batchV, hwK = aimx::kRegSize / batchV;
    if (hwN % roles.blockN != 0)
      return device->emitError("convert-pim-to-aimx: BLOCK_N=")
             << roles.blockN << " does not divide the output block width "
             << hwN << " (= 4096/batch, batch=" << batchV
             << "), so the n axis cannot be tiled by whole Triton blocks — pick "
                "a divisor of "
             << hwN << ", or leave batch at 0 to let the pass pick";
    if (hwK % roles.blockK != 0)
      return device->emitError("convert-pim-to-aimx: BLOCK_K=")
             << roles.blockK << " does not divide the reduction step " << hwK
             << " (= 256/batch, batch=" << batchV << ") — pick a divisor of "
             << hwK << ", or leave batch at 0 to let the pass pick";

    int64_t kMax, nMax;
    Footprint fp;
    if (failed(pickTile(device, batchV, mT, kMax, nMax, fp)))
      return failure();

    // 이 attribute 는 **할당된 buffer 의 모양**이다. 실제 tile 은 런타임 값이고
    // --aimx-lower-matmul 이 compile time 상수로 읽는 것은 batch 하나뿐이다.
    func->setAttr("aimx.tiling",
                  aimx::TilingAttr::get(ctx, batchV, mT, kMax, nMax, fp.paddedK,
                                        fp.paddedN, /*transposed=*/false));
    // 위 attribute 가 buffer 모양이라면 이쪽은 source 와 하드웨어의 관계다. execution 은
    // 전부 compile time 상수이고, n_t 는 런타임 값이라 여기 들어가지 않는다.
    auto i64s = [&](std::initializer_list<int64_t> v) {
      return DenseI64ArrayAttr::get(ctx, v);
    };
    func->setAttr(
        "aimx.tile_group",
        aimx::TileGroupAttr::get(
            ctx, i64s({roles.blockM, roles.blockN, roles.blockK}),
            i64s({mT / roles.blockM, hwN / roles.blockN, hwK / roles.blockK}),
            i64s({mT, hwN, hwK})));

    int64_t numDevices = 1;
    for (int64_t c : device.getMapping().getPartitions().asArrayRef())
      numDevices *= c;

    // ── host: pim.device 자리에 acquire + foreach_device 를 놓는다 ────────────
    OpBuilder hb(device);
    auto setTy = aimx::DeviceSetType::get(ctx);
    Value set = hb.create<aimx::AcquireOp>(loc, setTy,
                                           hb.getI64IntegerAttr(numDevices));
    ArrayAttr axes = device.getMapping().getAxes();
    auto counts = hb.getDenseI64ArrayAttr(
        device.getMapping().getPartitions().asArrayRef());
    auto foreach = hb.create<aimx::ForeachDeviceOp>(loc, set, axes, counts);

    SmallVector<Type> argTys{aimx::DeviceType::get(ctx)};
    SmallVector<Location> argLocs{loc};
    for (unsigned i = 0, e = axes.size(); i < e; i++) {
      argTys.push_back(hb.getIndexType());
      argLocs.push_back(loc);
    }
    Block *fdBody = hb.createBlock(&foreach.getBody(), {}, argTys, argLocs);
    // 좌표 계산을 통째로 옮기고 block argument 를 새 것으로 잇는다.
    fdBody->getOperations().splice(fdBody->end(), deviceBody->getOperations(),
                                   deviceBody->begin(), deviceBody->end());
    for (unsigned i = 0, e = axes.size(); i < e; i++)
      deviceBody->getArgument(i).replaceAllUsesWith(fdBody->getArgument(i + 1));

    // 논리 tile loop nest 는 통째로 사라진다. AiMX 에는 device program 이 없고
    // 계산 자체가 descriptor chain 이다.
    SmallVector<scf::ForOp> nests;
    for (Operation &op : *fdBody)
      if (auto f = dyn_cast<scf::ForOp>(&op))
        nests.push_back(f);
    for (scf::ForOp f : nests)
      f.erase();

    buildPlacement(fdBody, func.getName(), aPlace, bPlace, cPlace, mT, kMax,
                   nMax, hwN, fp);

    for (pim::PlaceOp p : all)
      p.erase();
    device.erase();
    // loop nest 를 지우면서 그것만 쓰던 값들이 죽은 채로 남는다. 만든 쪽이 치운다 —
    // 남겨 두면 codegen 이 옮길 수 없는 값(누산기 초기값 f32 상수 같은 것)을 만나
    // 실패한다.
    eraseDead(func);
    return success();
  }

  // 더 이상 줄지 않을 때까지 죽은 op 을 지운다. aimx op 들은 부수 효과가 선언돼 있지
  // 않아 isOpTriviallyDead 가 살려 두므로 이 순회에 걸리지 않는다.
  void eraseDead(func::FuncOp func) {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> dead;
      func.walk([&](Operation *op) {
        // region 을 가진 op 은 건드리지 않는다. 부모와 자식이 같은 목록에 들어가면
        // 부모를 지운 뒤 자식을 다시 지우게 된다.
        if (op->getNumRegions() == 0 && isOpTriviallyDead(op))
          dead.push_back(op);
      });
      for (Operation *op : llvm::reverse(dead)) {
        op->erase();
        changed = true;
      }
    }
  }

  // matmul 의 두 입력과 결과가 어느 pim.access 에서 왔는지 읽는다.
  LogicalResult readRoles(pim::DeviceOp device, Roles &roles) {
    SmallVector<linalg::MatmulOp> matmuls;
    SmallVector<pim::StoreOp> stores;
    device.getBody().walk([&](Operation *op) {
      if (auto mm = dyn_cast<linalg::MatmulOp>(op))
        matmuls.push_back(mm);
      if (auto st = dyn_cast<pim::StoreOp>(op))
        stores.push_back(st);
    });
    if (matmuls.size() != 1)
      return device->emitError("convert-pim-to-aimx: expected exactly one "
                               "linalg.matmul inside pim.device, found ")
             << matmuls.size();
    if (stores.size() != 1)
      return device->emitError("convert-pim-to-aimx: expected exactly one "
                               "pim.store inside pim.device, found ")
             << stores.size();

    auto idOf = [&](Value tile, int64_t &id) -> LogicalResult {
      auto load = tile.getDefiningOp<pim::LoadOp>();
      if (!load)
        return device->emitError("convert-pim-to-aimx: a matmul operand does "
                                 "not come from pim.load");
      auto access = load.getAccess().getDefiningOp<pim::AccessOp>();
      if (!access)
        return device->emitError(
            "convert-pim-to-aimx: a pim.load does not read a pim.access");
      id = access.getId();
      return success();
    };
    if (failed(idOf(matmuls[0].getInputs()[0], roles.a)) ||
        failed(idOf(matmuls[0].getInputs()[1], roles.b)))
      return failure();

    auto outAccess = stores[0].getAccess().getDefiningOp<pim::AccessOp>();
    if (!outAccess)
      return device->emitError(
          "convert-pim-to-aimx: the pim.store does not write a pim.access");
    roles.c = outAccess.getId();

    // 원소 타입. MAC 은 bfloat16 곱셈기이고 누산기도 부동소수점이다 — 정수 matmul 을
    // 옮길 자리가 없다.
    auto tileElem = [](Value v) {
      return cast<RankedTensorType>(v.getType()).getElementType();
    };
    if (!isa<FloatType>(tileElem(matmuls[0].getInputs()[0])) ||
        !isa<FloatType>(tileElem(matmuls[0].getInputs()[1])) ||
        !isa<FloatType>(tileElem(matmuls[0].getOutputs()[0])))
      return device->emitError("convert-pim-to-aimx: the AiMX MAC is a "
                               "bfloat16 multiplier, so the matmul operands "
                               "must be floating point");

    auto aTy = cast<RankedTensorType>(matmuls[0].getInputs()[0].getType());
    auto bTy = cast<RankedTensorType>(matmuls[0].getInputs()[1].getType());
    if (!aTy.hasStaticShape() || !bTy.hasStaticShape())
      return device->emitError("convert-pim-to-aimx: the matmul operands must "
                               "have a static shape — the Triton block size has "
                               "to be known to relate it to the hardware grid");
    roles.blockM = aTy.getDimSize(0);
    roles.blockK = aTy.getDimSize(1);
    roles.blockN = bTy.getDimSize(1);
    if (bTy.getDimSize(0) != roles.blockK)
      return device->emitError("convert-pim-to-aimx: the matmul operands "
                               "disagree on BLOCK_K");
    // 0 이면 아래에서 lcm 과 나눗셈이 0 으로 나눈다.
    if (roles.blockM <= 0 || roles.blockN <= 0 || roles.blockK <= 0)
      return device->emitError("convert-pim-to-aimx: the Triton block has a "
                               "non-positive dimension");

    // 누산기는 vACC 다. 하드웨어가 0 에서 시작하므로 초기값이 0 이 아니면 답이 달라진다.
    // --pim-form-compute 가 만든 모양에서 누산기는 reduction scf.for 의 iter_arg 이고
    // 그 초기값이 linalg.fill 이다.
    auto acc = dyn_cast<BlockArgument>(matmuls[0].getOutputs()[0]);
    auto loop = acc ? dyn_cast<scf::ForOp>(acc.getOwner()->getParentOp())
                    : scf::ForOp();
    if (!loop)
      return device->emitError("convert-pim-to-aimx: the accumulator is not a "
                               "reduction loop iter_arg");
    Value init = loop.getInitArgs()[acc.getArgNumber() - 1];
    auto fill = init.getDefiningOp<linalg::FillOp>();
    if (!fill)
      return device->emitError("convert-pim-to-aimx: the accumulator is not "
                               "initialized by linalg.fill");
    if (!matchPattern(fill.getInputs()[0], m_AnyZeroFloat()))
      return device->emitError(
          "convert-pim-to-aimx: the accumulator must start at zero — vACC does");
    return success();
  }

  //===--------------------------------------------------------------------===//
  // level 1 배치
  //===--------------------------------------------------------------------===//

  void buildPlacement(Block *body, StringRef base, pim::PlaceOp aPlace,
                      pim::PlaceOp bPlace, pim::PlaceOp cPlace, int64_t mT,
                      int64_t kMax, int64_t nMax, int64_t hwN,
                      const Footprint &fp) {
    OpBuilder b(body, body->end());
    Location loc = body->getParentOp()->getLoc();
    MLIRContext *ctx = b.getContext();
    Value dev = body->getArgument(0);
    auto bufTy = aimx::BufferType::get(ctx);
    auto elem = TypeAttr::get(b.getBF16Type());
    // 이 pass 는 아직 전치를 고르지 않는다. IR 은 표현할 수 있고 규칙이 런타임 값을
    // 필요로 할 뿐이다.
    UnitAttr noTranspose;

    // 상수는 loop 를 만들기 전에 전부 만들어 둔다. loop 뒤에 만들면 loop 안의 사용을
    // 지배하지 못한다.
    auto idx = [&](int64_t v) {
      return b.create<arith::ConstantIndexOp>(loc, v).getResult();
    };
    auto roundUpVal = [&](Value v, int64_t m) {
      Value mv = idx(m);
      Value q = b.create<arith::CeilDivSIOp>(loc, v, mv).getResult();
      return b.create<arith::MulIOp>(loc, q, mv).getResult();
    };
    Value c0 = idx(0), c1 = idx(1);
    Value mTv = idx(mT);

    // ── 런타임 tile 모양 ────────────────────────────────────────────────────
    // buffer 는 (k_max, n_max) 로 잡혔고 descriptor program 은 여기서 만든 값을 인자로
    // 받는다. 세 축 모두 상한에서 clamp 하고 넘치는 만큼 바깥 loop 이 돈다.
    //
    // k 를 자르는 것은 **손해**이고 마지막 수단이다. k tile 하나가 통째로 한 chain 에
    // 들어가면 reduction 이 vACC 안에서 끝나고 bf16 반올림이 출력당 한 번뿐인데, 자르면
    // 부분합이 tile 마다 PIM 밖으로 나와 C 전송이 k tile 수만큼 곱해진다. vACC 는 출력
    // block 하나 깊이(주소가 pmm+VACC_OFFSET 하나)라 invoke 를 건너 들고 있을 수 없으므로
    // 이것은 피할 수 있는 비용이 아니다. 그래서 k_max 는 예산이 허락하는 한 크게 두고
    // (1 GiB, batch 8, m_t 16, n_max 4096 이면 115712), 자르는 일은 그것을 넘을 때만
    // 일어난다 — 현실적인 k 는 전부 한 번에 들어간다.
    Value kRt = b.create<arith::MinSIOp>(
                     loc, roundUpVal(aPlace.getSize()[1], aimx::kRegSize),
                     idx(kMax))
                    .getResult();
    Value padK = roundUpVal(kRt, aimx::kChunk);
    // n 의 올림 단위는 **출력 block 폭 4096/batch** 다. 4096 으로 올리면 batch > 1 에서
    // 남는 열이 그대로 노는 lane 이 된다 — N = 2560, batch = 8 이면 n_t 가 4096(1536 열
    // 낭비)이 아니라 2560(낭비 0)이다. N >= n_max 면 어느 쪽이든 n_max 로 잘린다.
    Value nRt = b.create<arith::MinSIOp>(
                     loc, roundUpVal(cPlace.getSize()[1], hwN), idx(nMax))
                    .getResult();

    // PIM DRAM buffer 는 언제나 PIM_WIDTH 경계에 둔다. descriptor 가 그 단위로 주소를
    // 흩뿌리므로 그보다 거친 눈금에서 시작해야 한다.
    auto alloc = [&](int64_t bytes, aimx::AllocKind kind) {
      int64_t align =
          kind == aimx::AllocKind::PMM ? aimx::kPMMAlign : aimx::kPIMWidth;
      return b
          .create<aimx::AllocOp>(loc, bufTy, dev, b.getI64IntegerAttr(bytes),
                                 b.getI64IntegerAttr(align),
                                 aimx::AllocKindAttr::get(ctx, kind))
          .getResult();
    };

    // PMM 이 가장 먼저다. pim_private_malloc 이 pim_malloc 이 이미 내준 물리 주소를
    // 그대로 돌려주는 경우가 있어, 데이터 buffer 가 그 뒤에 와야 겹침을 검사할 수 있다.
    Value pmm = alloc(aimx::kPMMSize, aimx::AllocKind::PMM);
    Value aBuf = alloc(fp.a, aimx::AllocKind::Data);
    Value arel = alloc(fp.arel, aimx::AllocKind::Data);
    Value bBuf = alloc(fp.b, aimx::AllocKind::Data);
    Value cBuf = alloc(fp.c, aimx::AllocKind::Data);
    Value chain = alloc(fp.chain, aimx::AllocKind::Descriptor);

    // chain 은 tile loop 밖에서 한 번만 만든다. tile 모양과 base 주소가 모두 같으므로
    // 모든 tile 에서 byte 단위로 동일하다. 재배치가 먼저, 계산이 다음이고 그 순서가
    // 곧 chain 안의 순서다.
    //
    // 인자는 런타임 값(k_t, n_t)이다. group 수는 k 와 padded_n 에 대해 단조 증가하고
    // k_t <= k_max, n_t <= n_max 이므로 chain buffer 를 넘지 않는다.
    b.create<aimx::BuildDescriptorsOp>(
        loc, FlatSymbolRefAttr::get(ctx, (base + "_relayout").str()),
        ValueRange{mTv, kRt, padK, aBuf, arel}, chain);
    b.create<aimx::BuildDescriptorsOp>(
        loc, FlatSymbolRefAttr::get(ctx, (base + "_compute").str()),
        ValueRange{mTv, kRt, nRt, arel, bBuf, cBuf, pmm}, chain);

    Value mTiles =
        b.create<arith::CeilDivSIOp>(loc, cPlace.getSize()[0], mTv).getResult();
    Value nTiles =
        b.create<arith::CeilDivSIOp>(loc, cPlace.getSize()[1], nRt).getResult();
    // k 가 상한 안이면 이 값이 1 이고 아래 loop 은 한 번만 돈다 — 그때 나오는 코드는
    // k 를 나누기 전과 같다.
    Value kTiles =
        b.create<arith::CeilDivSIOp>(loc, aPlace.getSize()[1], kRt).getResult();

    // 이 tile 이 실제로 맡는 크기. 마지막 tile 이 덜 맡으면 stage 가 그만큼만 모으고
    // 나머지 자리는 0 이 된다 — 그 0 이 누산에 그대로 더해지므로 정확성 요구사항이다.
    auto remaining = [&](OpBuilder &ib, Value total, Value off, Value cap) {
      Value left = ib.create<arith::SubIOp>(loc, total, off).getResult();
      return ib.create<arith::MinSIOp>(loc, left, cap).getResult();
    };

    // ── loop 순서: n → k → m ────────────────────────────────────────────────
    // B 가 가장 큰 buffer(k_t * n_t * 2)이고 (k, n) 에만 의존하므로 두 축을 바깥에 둔다.
    // A 는 (m, k) 에 의존하지만 m_t * padded_k * 2 로 훨씬 작아 안쪽이 맞다. k 를 가장
    // 안쪽에 두면 B 가 m tile 마다 다시 올라가 전송량이 m tile 수만큼 곱해진다.
    // n 과 m 의 상대 순서를 바꾸는 것은 §8 I1 로 chain 을 재배치용과 계산용으로 나눈
    // 뒤에만 의미가 있다 — 지금은 chain 하나에 둘 다 붙어 있어 invoke 마다 재배치가
    // 다시 돈다.
    auto nLoop = b.create<scf::ForOp>(loc, c0, nTiles, c1);
    OpBuilder nb = OpBuilder::atBlockBegin(nLoop.getBody());
    Value nOff =
        nb.create<arith::MulIOp>(loc, nLoop.getInductionVar(), nRt).getResult();

    auto kLoop = nb.create<scf::ForOp>(loc, c0, kTiles, c1);
    OpBuilder kb = OpBuilder::atBlockBegin(kLoop.getBody());
    Value kOff =
        kb.create<arith::MulIOp>(loc, kLoop.getInductionVar(), kRt).getResult();

    Value bRow0 =
        kb.create<arith::AddIOp>(loc, bPlace.getStart()[0], kOff).getResult();
    Value bCol0 =
        kb.create<arith::AddIOp>(loc, bPlace.getStart()[1], nOff).getResult();
    kb.create<aimx::StageOp>(
        loc, bPlace.getBuffer(), ValueRange{bRow0, bCol0},
        ValueRange{remaining(kb, bPlace.getSize()[0], kOff, kRt),
                   remaining(kb, bPlace.getSize()[1], nOff, nRt)},
        bPlace.getStrides(), ValueRange{kRt, nRt}, nRt, bBuf, elem,
        noTranspose);

    // 두 번째 k tile 부터는 C 를 덮는 것이 아니라 이전 부분합에 더한다. k tile 이
    // 하나뿐이면 이 값은 언제나 0 이고 unstage 는 예전 그대로 대입만 한다.
    Value acc =
        kb.create<arith::SelectOp>(
              loc,
              kb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                       kLoop.getInductionVar(), c0)
                  .getResult(),
              c1, c0)
            .getResult();

    auto mLoop = kb.create<scf::ForOp>(loc, c0, mTiles, c1);
    OpBuilder mb = OpBuilder::atBlockBegin(mLoop.getBody());
    Value mOff =
        mb.create<arith::MulIOp>(loc, mLoop.getInductionVar(), mTv).getResult();

    Value aRow0 =
        mb.create<arith::AddIOp>(loc, aPlace.getStart()[0], mOff).getResult();
    Value aCol0 =
        mb.create<arith::AddIOp>(loc, aPlace.getStart()[1], kOff).getResult();
    mb.create<aimx::StageOp>(
        loc, aPlace.getBuffer(), ValueRange{aRow0, aCol0},
        ValueRange{remaining(mb, aPlace.getSize()[0], mOff, mTv),
                   remaining(mb, aPlace.getSize()[1], kOff, kRt)},
        aPlace.getStrides(), ValueRange{mTv, kRt}, padK, aBuf, elem,
        noTranspose);

    mb.create<aimx::InvokeOp>(loc, dev, chain);

    Value cRow0 =
        mb.create<arith::AddIOp>(loc, cPlace.getStart()[0], mOff).getResult();
    Value cCol0 =
        mb.create<arith::AddIOp>(loc, cPlace.getStart()[1], nOff).getResult();
    mb.create<aimx::UnstageOp>(
        loc, cBuf, cPlace.getBuffer(), ValueRange{cRow0, cCol0},
        ValueRange{remaining(mb, cPlace.getSize()[0], mOff, mTv),
                   remaining(mb, cPlace.getSize()[1], nOff, nRt)},
        cPlace.getStrides(), nRt, acc, elem, noTranspose);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertPIMToAiMXPass() {
  return std::make_unique<ConvertPIMToAiMXPass>();
}
