#include "triton-shared/Conversion/PIMToAiMX/PIMToAiMX.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXFootprint.h"

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
#include <numeric>

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTPIMTOAIMX
#include "triton-shared/Conversion/PIMToAiMX/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

bool isBatch(int64_t b) { return b == 1 || b == 2 || b == 4 || b == 8; }

// block group 한 축 — 공통 #pim.block_group 에서 읽는다. Triton block 정수 개(tiles)를
// 하드웨어 실행 단위의 정수 배에 채운 것이고, 딱 맞지 않으면 남는 자리가 padding 이다 —
// stage 가 0 을 채우고 unstage 가 읽지 않으므로 정확성에는 무해하고, 그 몫의 lane 과
// reduction step 이 노는 것이 비용이다.
struct BlockGroup {
  int64_t tiles;  // Triton block 수
  int64_t useful; // tiles * BLOCK — 실제 데이터
  int64_t phys;   // 하드웨어가 도는 크기
  int64_t pad() const { return phys - useful; }
};

// 한 AiMX Device에서 scheduling round 전체가 공유하는 PIM DRAM 주소들이다.
// descriptor chain은 이 주소와 execution-tile 최대 모양만 참조하므로 한 번 만든다.
struct PlacementBuffers {
  Value device;
  Value pmm;
  Value a;
  Value arel;
  Value b;
  Value c;
  Value chain;
};

// execution tile 하나가 PIM DRAM 에 요구하는 바이트 — 공통 --pim-form-execution-tiles
// 의 k 상한 탐색과 같은 식을 써야 하므로 AiMXFootprint.h 로 옮겼다.
using aimx::Footprint;
using aimx::footprintOf;

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

    // profile·group_execution·block_group 은 여기서 소비된다. 남겨 두면 target IR 이
    // 여전히 PIM 을 가리키게 되고, PIM dialect 을 모르는 translate 도구가 파싱에서
    // 멈춘다.
    getOperation()->removeAttr("pim.target_profile");
    getOperation().walk([](func::FuncOp f) {
      f->removeAttr("pim.group_execution");
      f->removeAttr("pim.block_group");
    });
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
    if (profile.getNumberOfParallelLanesPerPimCore() != aimx::kLanes)
      return m->emitError("convert-pim-to-aimx: AiMX has ")
             << aimx::kLanes << " lanes per PIM Core, not "
             << profile.getNumberOfParallelLanesPerPimCore();
    // 실행 단위가 batch 를 정한다: 출력 block 은 batch 행 x 4096/batch 열이고
    // reduction step 은 256/batch 다. 셋이 이 관계가 아니면 AiMX 가 아니다.
    auto unit = profile.getExecutionUnit();
    if (!unit || unit.asArrayRef().size() != 3)
      return m->emitError(
          "convert-pim-to-aimx: the profile carries no execution_unit "
          "[m, n, k], so the block group grid is unknown");
    ArrayRef<int64_t> u = unit.asArrayRef();
    if (!isBatch(u[0]) || u[1] != aimx::kLanes / u[0] ||
        u[2] != aimx::kRegSize / u[0])
      return m->emitError("convert-pim-to-aimx: the execution unit must be "
                          "[batch, 4096/batch, 256/batch] with batch 1, 2, 4, "
                          "or 8 — got [")
             << u[0] << ", " << u[1] << ", " << u[2] << "]";
    return success();
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

  // §6.1. (m_t, k_max, n_max)는 buffer와 재사용 descriptor chain의 고정 모양이다.
  // tail execution tile은 stage에서 이 모양까지 0 padding하고 unstage는 유효 영역만
  // 회수한다. 할당해 보고 물러서는 길이가 없으므로 예산을 넘으면 여기서 멈춘다.
  LogicalResult checkBudget(Operation *at, int64_t batchV, int64_t mT,
                            int64_t kMax, int64_t nMax, Footprint &fp) {
    int64_t budget;
    if (failed(budgetBytes(at, budget)))
      return failure();
    fp = footprintOf(batchV, mT, kMax, nMax);
    if (fp.total() > budget)
      return at->emitError("convert-pim-to-aimx: the buffer bound (m_t=")
             << mT << ", k_max=" << kMax << ", n_max=" << nMax << ") needs "
             << fp.total() << " byte of PIM DRAM but the budget is " << budget
             << " byte — lower tile_m, tile_k, or tile_n";
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

    if (!device->getAttrOfType<pim::DevicePlanAttr>("pim.device_plan"))
      return device->emitError(
          "convert-pim-to-aimx: the device has no #pim.device_plan");
    auto reduction =
        device->getAttrOfType<pim::ReductionPlanAttr>("pim.reduction_plan");
    if (!reduction)
      return device->emitError(
          "convert-pim-to-aimx: the device has no #pim.reduction_plan");
    if (reduction.getTemporalAccumulation().getValue() != "none" &&
        reduction.getTemporalAccumulation().getValue() != "host")
      return device->emitError(
          "convert-pim-to-aimx: only host temporal accumulation is supported");

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
    // 마지막 두 축이 PIM의 2-D C tile이다. 앞축이 있으면 split-K 같은 독립 partial
    // output 축이며 host buffer의 origin/stride로 보존한다.
    if (cPlace.getStart().size() < 2)
      return cPlace->emitError(
          "convert-pim-to-aimx: the output needs at least row and column axes");
    if (aPlace.getStart().size() != 2 || bPlace.getStart().size() != 2)
      return device->emitError(
          "convert-pim-to-aimx: A and B must be two-dimensional");

    // ── block → block group → execution tile ─────────────────────────────────
    // 세 계층은 전부 공통 IR 이 정했다. block 은 matmul operand 의 크기(readRoles)로
    // 보이고, block group 은 --pim-form-block-groups 가 실행 단위에 block 을 채워
    // #pim.block_group 으로 적어 두었으며, execution tile 상한은 #pim.execution_tile
    // 이다. 이 pass 가 하는 일은 번역뿐이다 — group 을 상한까지 모아 buffer 를 잡고,
    // padding(useful < phys)을 stage/unstage 의 group 배치로 옮긴다.
    auto bg = func->getAttrOfType<pim::BlockGroupAttr>("pim.block_group");
    if (!bg)
      return func->emitError(
          "convert-pim-to-aimx: the function has no #pim.block_group — run "
          "--pim-form-block-groups after --pim-set-target-profile");
    auto groupAt = [&](int i) {
      BlockGroup g;
      g.tiles = bg.getTiles().asArrayRef()[i];
      g.useful = bg.getUseful().asArrayRef()[i];
      g.phys = bg.getPhys().asArrayRef()[i];
      return g;
    };
    BlockGroup mGroup = groupAt(0), nGroup = groupAt(1), kGroup = groupAt(2);
    // batch 는 실행 단위의 m 이다 (gate 가 [batch, 4096/batch, 256/batch] 를 검증했다).
    int64_t batchV = profile.getExecutionUnit().asArrayRef()[0];
    // k tile 은 block group 과 vecA 적재 단위(256, §2.4 의 k_t % 256)를 동시에 채워야
    // 한다. 물리 k 의 눈금은 그래서 둘의 lcm 이고, padding 이 없으면 256 그대로다.
    int64_t kUnit = std::lcm(kGroup.phys, aimx::kRegSize);
    int64_t kUnitUseful = kUnit / kGroup.phys * kGroup.useful;

    // 상한(유효 원소)이 담는 group 수는 내림이다(최소 1) — 상한은 넘지 않는 약속이다.
    // host loop 은 유효 크기로 걷고 buffer 는 물리 크기로 잡는다. 세 축의 구체 수치는
    // 공통 --pim-form-execution-tiles 가 확정했다(사용자 값 또는 예산 탐색 — footprint
    // 식은 AiMXFootprint.h 로 공유). 이 pass 는 읽기만 하고, 예산 재검증(checkBudget)
    // 만 방어적으로 유지한다.
    auto cap =
        device->getAttrOfType<pim::ExecutionTileAttr>("pim.execution_tile");
    if (!cap || cap.getM() <= 0 || cap.getN() <= 0 || cap.getK() <= 0)
      return device->emitError(
          "convert-pim-to-aimx: #pim.execution_tile with concrete m, n, k "
          "caps is required — run --pim-form-execution-tiles (it computes "
          "missing caps from the profile budget)");

    int64_t mCount = std::max<int64_t>(cap.getM() / mGroup.useful, 1);
    int64_t mUse = mCount * mGroup.useful;
    int64_t mPhys = mCount * mGroup.phys;
    int64_t nMax =
        std::max<int64_t>(cap.getN() / nGroup.useful, 1) * nGroup.phys;
    int64_t kMax = std::max<int64_t>(cap.getK() / kUnitUseful, 1) * kUnit;

    Footprint fp;
    if (failed(checkBudget(device, batchV, mPhys, kMax, nMax, fp)))
      return failure();

    // 이 attribute는 할당된 buffer와 descriptor chain의 고정 모양이다.
    func->setAttr("aimx.tiling",
                  aimx::TilingAttr::get(ctx, batchV, mPhys, kMax, nMax,
                                        fp.paddedK, fp.paddedN,
                                        /*transposed=*/false));
    // 위 attribute 가 buffer 모양이라면 이쪽은 source 와 하드웨어의 관계다. group 이
    // block group 당 Triton block 수, execution 이 block group 의 물리 크기다. 전부
    // compile time 상수이고, execution tile 안의 실제 tail 크기는 place가 들고 있다.
    auto i64s = [&](std::initializer_list<int64_t> v) {
      return DenseI64ArrayAttr::get(ctx, v);
    };
    func->setAttr(
        "aimx.tile_group",
        aimx::TileGroupAttr::get(
            ctx, i64s({roles.blockM, roles.blockN, roles.blockK}),
            i64s({mGroup.tiles, nGroup.tiles, kGroup.tiles}),
            i64s({mGroup.phys, nGroup.phys, kGroup.phys})));

    int64_t numDevices = 1;
    for (int64_t c : device.getMapping().getPartitions().asArrayRef())
      numDevices *= c;

    // ── host: 공통 scheduling round 바깥에 foreach_device 를 놓는다 ─────────
    // PIM Device set은 공통 scheduling round 전체에서 동일하다. acquire와
    // foreach_device를 가장 바깥 scheduling loop 앞으로 옮긴다. 그 결과 각 device
    // thread 안에서 allocation/descriptor build를 한 번 하고 모든 round를 돈다.
    Operation *scheduleRoot = device;
    while (auto round = dyn_cast_or_null<scf::ForOp>(scheduleRoot->getParentOp())) {
      if (!round->hasAttr("pim.scheduling_axis"))
        break;
      scheduleRoot = round;
    }
    OpBuilder setup(scheduleRoot);
    auto setTy = aimx::DeviceSetType::get(ctx);
    Value set = setup.create<aimx::AcquireOp>(
        loc, setTy, setup.getI64IntegerAttr(numDevices));
    OpBuilder hb(scheduleRoot);
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

    // round 계산과 pim.device를 함께 foreach_device 안으로 옮긴다. 원래 round IV와
    // device 좌표를 그대로 보존하므로 place의 전역 origin 식을 다시 만들 필요가 없다.
    scheduleRoot->moveBefore(fdBody, fdBody->end());

    Block *roundBody = device->getBlock();
    auto deviceIt = device->getIterator();
    roundBody->getOperations().splice(deviceIt, deviceBody->getOperations(),
                                      deviceBody->begin(), deviceBody->end());
    for (unsigned i = 0, e = axes.size(); i < e; i++)
      deviceBody->getArgument(i).replaceAllUsesWith(fdBody->getArgument(i + 1));

    // 논리 tile loop nest 는 통째로 사라진다. AiMX 에는 device program 이 없고
    // 계산 자체가 descriptor chain 이다.
    SmallVector<scf::ForOp> nests;
    for (Operation &op : *roundBody)
      if (auto f = dyn_cast<scf::ForOp>(&op))
        nests.push_back(f);
    for (scf::ForOp f : nests)
      f.erase();

    PlacementBuffers buffers = buildSetup(
        fdBody, func.getName(), mPhys, kMax, nMax, fp);
    buildPlacement(device.getOperation(), aPlace, bPlace, cPlace, mUse, mPhys,
                   kMax, nMax, mGroup, nGroup, kGroup, kUnit, fp, buffers);

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
    // 0 이면 아래의 나눗셈들이 0 으로 나눈다.
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

  PlacementBuffers buildSetup(Block *body, StringRef base, int64_t mPhys,
                              int64_t kMax, int64_t nMax,
                              const Footprint &fp) {
    OpBuilder b(body, body->begin());
    Location loc = body->getParentOp()->getLoc();
    MLIRContext *ctx = b.getContext();
    Value dev = body->getArgument(0);
    auto bufTy = aimx::BufferType::get(ctx);

    auto alloc = [&](int64_t bytes, aimx::AllocKind kind) {
      int64_t align =
          kind == aimx::AllocKind::PMM ? aimx::kPMMAlign : aimx::kPIMWidth;
      return b
          .create<aimx::AllocOp>(loc, bufTy, dev, b.getI64IntegerAttr(bytes),
                                 b.getI64IntegerAttr(align),
                                 aimx::AllocKindAttr::get(ctx, kind))
          .getResult();
    };

    PlacementBuffers buffers;
    buffers.device = dev;
    // PMM이 먼저여야 pim_private_malloc과 일반 pim_malloc 영역의 겹침을 검사할 수 있다.
    buffers.pmm = alloc(aimx::kPMMSize, aimx::AllocKind::PMM);
    buffers.a = alloc(fp.a, aimx::AllocKind::Data);
    buffers.arel = alloc(fp.arel, aimx::AllocKind::Data);
    buffers.b = alloc(fp.b, aimx::AllocKind::Data);
    buffers.c = alloc(fp.c, aimx::AllocKind::Data);
    buffers.chain = alloc(fp.chain, aimx::AllocKind::Descriptor);

    // descriptor는 execution-tile 최대 모양과 고정 PIM-side base address만 사용한다.
    // tail round도 같은 chain을 실행하며, 부족한 데이터는 stage가 0으로 채우고
    // unstage가 실제 유효 영역만 회수한다.
    Value m = b.create<arith::ConstantIndexOp>(loc, mPhys);
    Value k = b.create<arith::ConstantIndexOp>(loc, kMax);
    Value paddedK = b.create<arith::ConstantIndexOp>(loc, fp.paddedK);
    Value n = b.create<arith::ConstantIndexOp>(loc, nMax);
    b.create<aimx::BuildDescriptorsOp>(
        loc, FlatSymbolRefAttr::get(ctx, (base + "_relayout").str()),
        ValueRange{m, k, paddedK, buffers.a, buffers.arel}, buffers.chain);
    b.create<aimx::BuildDescriptorsOp>(
        loc, FlatSymbolRefAttr::get(ctx, (base + "_compute").str()),
        ValueRange{m, k, n, buffers.arel, buffers.b, buffers.c, buffers.pmm},
        buffers.chain);
    return buffers;
  }

  void buildPlacement(Operation *at, pim::PlaceOp aPlace,
                      pim::PlaceOp bPlace, pim::PlaceOp cPlace, int64_t mUse,
                      int64_t mPhys, int64_t kMax, int64_t nMax,
                      const BlockGroup &mGroup, const BlockGroup &nGroup,
                      const BlockGroup &kGroup, int64_t kUnit,
                      const Footprint &fp,
                      const PlacementBuffers &buffers) {
    OpBuilder b(at);
    Location loc = at->getLoc();
    MLIRContext *ctx = b.getContext();
    Value dev = buffers.device;
    auto elem = TypeAttr::get(b.getBF16Type());
    // 이 pass 는 아직 전치를 고르지 않는다. IR 은 표현할 수 있고 규칙이 런타임 값을
    // 필요로 할 뿐이다.
    UnitAttr noTranspose;

    // 상수는 loop 를 만들기 전에 전부 만들어 둔다. loop 뒤에 만들면 loop 안의 사용을
    // 지배하지 못한다.
    auto idx = [&](int64_t v) {
      return b.create<arith::ConstantIndexOp>(loc, v).getResult();
    };
    Value c0 = idx(0), c1 = idx(1);
    // m 도 n·k 와 같다: host loop 은 유효 크기로 걷고, buffer 와 descriptor program 은
    // 물리 크기를 본다. padding 이 없으면 둘이 같다.
    Value mUseV = idx(mUse), mPhysV = idx(mPhys);
    Value fixedK = idx(kMax), fixedN = idx(nMax);
    Value fixedPaddedK = idx(fp.paddedK);

    // ── execution tile 안의 실제 유효 범위 ─────────────────────────────────
    // buffer와 descriptor는 고정 최대 모양이고, 여기서는 현재 round의 유효 크기를
    // 상한에서 clamp한다. tail은 stage가 최대 모양까지 0으로 채운다.
    //
    // block group 이 끼면 **유효(useful)** 크기와 **물리(phys)** 크기가 갈린다. host
    // 가 소비하는 것(loop 전진 폭, stage/unstage 의 origin·sizes)은 유효 크기이고,
    // PIM DRAM·chain·descriptor program 이 보는 것(pad_to, pitch, program 인자)은
    // 물리 크기다. padding 이 없으면 둘이 같아 예전과 같은 값이 나온다.
    //
    // k 를 자르는 것은 **손해**이고 마지막 수단이다. k tile 하나가 통째로 한 chain 에
    // 들어가면 reduction 이 vACC 안에서 끝나고 bf16 반올림이 출력당 한 번뿐인데, 자르면
    // 부분합이 tile 마다 PIM 밖으로 나와 C 전송이 k tile 수만큼 곱해진다. vACC 는 출력
    // block 하나 깊이(주소가 pmm+VACC_OFFSET 하나)라 invoke 를 건너 들고 있을 수 없으므로
    // 이것은 피할 수 있는 비용이 아니다. 그래서 k_max 는 예산이 허락하는 한 크게 두고
    // (1 GiB, batch 8, m_t 16, n_max 4096 이면 115712), 자르는 일은 그것을 넘을 때만
    // 일어난다 — 현실적인 k 는 전부 한 번에 들어간다.
    int64_t kUnitUseful = (kUnit / kGroup.phys) * kGroup.useful;
    // 단위 수는 최소 1 이다. device 분배가 이 device 에 일을 주지 않으면(범위 0)
    // 단위 수가 0 이 되고 그 곱 k_t/n_t 가 tile 수 나눗셈의 분모로 들어가 SIGFPE 다.
    // 1 로 받치면 tile 수가 ceildiv(0, 단위) = 0 이라 loop 이 돌지 않는 no-op 이 된다.
    Value kUnits = b.create<arith::MaxSIOp>(
        loc,
        b.create<arith::MinSIOp>(
             loc,
             b.create<arith::CeilDivSIOp>(loc, aPlace.getSize()[1],
                                          idx(kUnitUseful))
                 .getResult(),
             idx(kMax / kUnit))
            .getResult(),
        c1);
    Value kRt = b.create<arith::MulIOp>(loc, kUnits, idx(kUnit)).getResult();
    Value kUse = kGroup.pad()
                     ? b.create<arith::MulIOp>(loc, kUnits, idx(kUnitUseful))
                           .getResult()
                     : kRt;
    // n 의 눈금은 block group 의 물리 폭이다 (padding 이 없으면 출력 block 폭
    // 4096/batch 그대로). 4096 으로 올리면 batch > 1 에서 남는 열이 그대로 노는
    // lane 이 된다 — N = 2560, batch = 8 이면 n_t 가 4096(1536 열 낭비)이 아니라
    // 2560(낭비 0)이다. N >= n_max 면 어느 쪽이든 n_max 로 잘린다.
    Value nGroups = b.create<arith::MaxSIOp>(
        loc,
        b.create<arith::MinSIOp>(
             loc, b.create<arith::CeilDivSIOp>(
                      loc, cPlace.getSize()[cPlace.getSize().size() - 1],
                                          idx(nGroup.useful))
                 .getResult(),
             idx(nMax / nGroup.phys))
            .getResult(),
        c1);
    Value nRt =
        b.create<arith::MulIOp>(loc, nGroups, idx(nGroup.phys)).getResult();
    Value nUse = nGroup.pad()
                     ? b.create<arith::MulIOp>(loc, nGroups, idx(nGroup.useful))
                           .getResult()
                     : nRt;
    // padding 이 있는 축만 stage / unstage 에 block group 배치를 싣는다. 없는 축은
    // 항등이라 attr 를 아예 내지 않는다 — 그래야 stub 이 예전 코드를 그대로 낸다.
    auto grpAttr = [&](const BlockGroup &g) {
      return g.pad() ? b.getDenseI64ArrayAttr({g.useful, g.phys})
                     : DenseI64ArrayAttr();
    };

    // tile 수와 loop 전진 폭은 **유효** 크기로 센다 — host 좌표는 padding 을 모른다.
    Value mTiles =
        b.create<arith::CeilDivSIOp>(
             loc, cPlace.getSize()[cPlace.getSize().size() - 2], mUseV)
            .getResult();
    Value nTiles = b.create<arith::CeilDivSIOp>(
        loc, cPlace.getSize()[cPlace.getSize().size() - 1], nUse);
    // k 가 상한 안이면 이 값이 1 이고 아래 loop 은 한 번만 돈다 — 그때 나오는 코드는
    // k 를 나누기 전과 같다.
    Value kTiles =
        b.create<arith::CeilDivSIOp>(loc, aPlace.getSize()[1], kUse).getResult();

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
        nb.create<arith::MulIOp>(loc, nLoop.getInductionVar(), nUse).getResult();

    auto kLoop = nb.create<scf::ForOp>(loc, c0, kTiles, c1);
    OpBuilder kb = OpBuilder::atBlockBegin(kLoop.getBody());
    Value kOff =
        kb.create<arith::MulIOp>(loc, kLoop.getInductionVar(), kUse).getResult();

    Value bRow0 =
        kb.create<arith::AddIOp>(loc, bPlace.getStart()[0], kOff).getResult();
    Value bCol0 =
        kb.create<arith::AddIOp>(loc, bPlace.getStart()[1], nOff).getResult();
    kb.create<aimx::StageOp>(
        loc, bPlace.getBuffer(), ValueRange{bRow0, bCol0},
        ValueRange{remaining(kb, bPlace.getSize()[0], kOff, kUse),
                   remaining(kb, bPlace.getSize()[1], nOff, nUse)},
        bPlace.getStrides(), ValueRange{fixedK, fixedN}, fixedN, buffers.b,
        elem,
        noTranspose, /*group_rows=*/grpAttr(kGroup),
        /*group_cols=*/grpAttr(nGroup));

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
        mb.create<arith::MulIOp>(loc, mLoop.getInductionVar(), mUseV)
            .getResult();

    Value aRow0 =
        mb.create<arith::AddIOp>(loc, aPlace.getStart()[0], mOff).getResult();
    Value aCol0 =
        mb.create<arith::AddIOp>(loc, aPlace.getStart()[1], kOff).getResult();
    mb.create<aimx::StageOp>(
        loc, aPlace.getBuffer(), ValueRange{aRow0, aCol0},
        ValueRange{remaining(mb, aPlace.getSize()[0], mOff, mUseV),
                   remaining(mb, aPlace.getSize()[1], kOff, kUse)},
        aPlace.getStrides(), ValueRange{mPhysV, fixedK}, fixedPaddedK,
        buffers.a, elem,
        noTranspose, /*group_rows=*/grpAttr(mGroup),
        /*group_cols=*/grpAttr(kGroup));

    mb.create<aimx::InvokeOp>(loc, dev, buffers.chain);

    unsigned cRank = cPlace.getStart().size();
    SmallVector<Value> cOrigin(cPlace.getStart());
    SmallVector<Value> cSizes(cPlace.getSize());
    cOrigin[cRank - 2] = mb.create<arith::AddIOp>(
        loc, cOrigin[cRank - 2], mOff);
    cOrigin[cRank - 1] = mb.create<arith::AddIOp>(
        loc, cOrigin[cRank - 1], nOff);
    cSizes[cRank - 2] =
        remaining(mb, cSizes[cRank - 2], mOff, mUseV);
    cSizes[cRank - 1] = remaining(mb, cSizes[cRank - 1], nOff, nUse);
    mb.create<aimx::UnstageOp>(
        loc, buffers.c, cPlace.getBuffer(), cOrigin, cSizes,
        cPlace.getStrides(), fixedN, acc, elem, noTranspose,
        /*group_rows=*/grpAttr(mGroup), /*group_cols=*/grpAttr(nGroup));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertPIMToAiMXPass() {
  return std::make_unique<ConvertPIMToAiMXPass>();
}
