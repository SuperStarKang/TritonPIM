#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMPLACEDATA
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// v를 계산하는 식을 map대로 치환해 nest 밖에 다시 만든다. nest 밖에서 정의된 값은 이미
// device 좌표와 무관하므로 그대로 쓴다. map에 없는 nest 안의 block argument를 만나면
// 그 자리에서 만들 수 없으므로 널을 낸다.
static Value rebuild(OpBuilder &b, Value v, IRMapping &map, Operation *nest) {
  if (Value m = map.lookupOrNull(v))
    return m;
  Operation *def = v.getDefiningOp();
  if (!def)
    return nest->isAncestor(v.getParentBlock()->getParentOp()) ? Value() : v;
  if (!nest->isAncestor(def))
    return v;
  IRMapping local;
  for (Value o : def->getOperands()) {
    Value r = rebuild(b, o, map, nest);
    if (!r)
      return Value();
    local.map(o, r);
  }
  Value r = b.clone(*def, local)->getResult(cast<OpResult>(v).getResultNumber());
  map.map(v, r);
  return r;
}

// loop 몸체에 직접 있는 전역 좌표 복원 op. 안쪽 loop에도 같은 표시가 있으므로 walk로
// 찾으면 가장 안쪽 것이 잡힌다.
static Operation *deviceOrigin(scf::ForOp loop) {
  for (Operation &op : *loop.getBody())
    if (op.hasAttr("pim.device_origin"))
      return &op;
  return nullptr;
}

// origin 식을 덧셈 항으로 가른다.
static void collectAddends(Value v, SmallVectorImpl<Value> &terms) {
  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    collectAddends(add.getLhs(), terms);
    collectAddends(add.getRhs(), terms);
    return;
  }
  terms.push_back(v);
}

// 항이 유도변수에 곱하는 정적 계수. index_cast 를 벗기고 muli 사슬의 상수들을 곱해
// 내려간다 — group-k 가 끼면 유도변수 사용처가 muli(muli(iv, g), tile) 꼴이 된다.
// 유도변수까지 상수 곱으로만 이어지지 않으면 알 수 없다.
static std::optional<int64_t> stepOfTerm(Value term) {
  while (auto c = term.getDefiningOp<arith::IndexCastOp>())
    term = c.getIn();
  if (isa<BlockArgument>(term))
    return 1;
  if (Operation *origin = term.getDefiningOp();
      origin && origin->hasAttr("pim.device_origin")) {
    auto add = dyn_cast<arith::AddIOp>(origin);
    if (!add)
      return std::nullopt;
    // start + local*step에서 scheduling round에 의존하는 start는 loop 한
    // 회전의 전진 폭과 무관하다. local 쪽 scaled 항만 따라간다.
    return stepOfTerm(add.getRhs());
  }
  auto mul = term.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return std::nullopt;
  if (std::optional<int64_t> c = getConstantIntValue(mul.getRhs())) {
    std::optional<int64_t> rest = stepOfTerm(mul.getLhs());
    return rest ? std::optional<int64_t>(*c * *rest) : std::nullopt;
  }
  if (std::optional<int64_t> c = getConstantIntValue(mul.getLhs())) {
    std::optional<int64_t> rest = stepOfTerm(mul.getRhs());
    return rest ? std::optional<int64_t>(*c * *rest) : std::nullopt;
  }
  return std::nullopt;
}

// 좌표는 loop 고유 타입으로 두고 배치 산술은 index로 한다. 축 loop는 index이고 reduction
// loop는 커널이 쓰던 정수 타입이라 섞으면 타입이 어긋난다.
static Value toIndex(OpBuilder &b, Location loc, Value v) {
  return isa<IndexType>(v.getType())
             ? v
             : b.create<arith::IndexCastOp>(loc, b.getIndexType(), v)
                   .getResult();
}

// device 하나가 loop 하나를 도는 범위.
struct LoopRange {
  Value key;           // 몸체 안에서 전역 좌표를 나르는 값
  Value first;         // 첫 반복의 좌표
  Value trips;         // 실제로 도는 횟수
  Value capacityTiles; // 배정량. device 불변이다.
  StringAttr axis;     // canonical device axis. semantic reduction은 k.
  int64_t partitions;  // 이 축의 device partition 수
};

// 나뉜 축 P 가 다른 loop 와 함께 버퍼 축을 움직여도 연속인 경우가 하나 있다: 함께 모는
// loop 가 정확히 tile 씩 전진하는 안쪽 loop 이고(blocked 분할), P 가 그 전체 구간을
// 단위로 전진할 때다. 그때 device 영역은 시작 하나에 trips 곱 × tile 길이의 연속
// 구간이라 기존 span/capacity 식이 그대로 맞는다. 함께 모는 loop 가 tile 보다 크게
// 전진하면(cyclic 분할) 영역이 띄엄띄엄이므로 진단한다.
// P 의 전진 폭(= 안쪽 loop 의 전체 구간)은 런타임 값이라 여기서 증명하지 못한다 —
// partitioned logical coordinate가 연속 구간을 만들도록 정규화됐다는 것이 계약이다.
static LogicalResult
checkBlockedComposite(pim::AccessOp access, unsigned d, Operation *partitioned,
                      StringAttr axisName, ArrayRef<scf::ForOp> loops,
                      const llvm::DenseMap<Operation *, LoopRange> &ranges,
                      const llvm::SmallPtrSetImpl<Operation *> &drivers) {
  auto fail = [&]() {
    return access->emitError("pim-place-data: buffer axis ")
           << d << " is driven by the partitioned axis " << axisName.getValue()
           << " together with another loop, and the composite is not a blocked "
              "split (the co-driving loop must advance by exactly one tile), "
              "so the device region is not a single contiguous range";
  };
  if (drivers.size() != 2)
    return fail();
  Operation *other = nullptr;
  for (Operation *o : drivers)
    if (o != partitioned)
      other = o;
  // 나뉜 축 둘이 한 버퍼 축을 몰면 어느 쪽 구간도 연속일 수 없다.
  if (ranges.find(other)->second.partitions != 1)
    return fail();
  SmallVector<Value> terms;
  collectAddends(access.getOrigin()[d], terms);
  bool sawInner = false;
  for (Value t : terms) {
    llvm::SmallPtrSet<Operation *, 4> termDrivers;
    pim::collectDrivers(t, loops, termDrivers);
    if (termDrivers.empty())
      continue; // 좌표 무관 base
    if (termDrivers.size() != 1)
      return fail(); // 한 항에 두 loop 가 섞이면 분해할 수 없다
    if (termDrivers.contains(partitioned))
      continue; // P 의 시작 항 — 전진 폭은 런타임이라 위의 계약에 맡긴다
    std::optional<int64_t> step = stepOfTerm(t);
    if (!step || *step != access.getSizes()[d])
      return fail();
    sawInner = true;
  }
  return sawInner ? success() : fail();
}

struct PIMPlaceDataPass
    : public mlir::triton::impl::PIMPlaceDataBase<PIMPlaceDataPass> {
  using PIMPlaceDataBase::PIMPlaceDataBase;

  void runOnOperation() override {
    SmallVector<pim::DeviceOp> devices;
    getOperation().walk([&](pim::DeviceOp d) { devices.push_back(d); });
    for (pim::DeviceOp d : devices)
      if (failed(placeDevice(d)))
        return signalPassFailure();
  }

  LogicalResult placeDevice(pim::DeviceOp device) {
    Block *body = &device.getBody().front();
    scf::ForOp nest;
    for (Operation &op : *body)
      if (auto f = dyn_cast<scf::ForOp>(&op)) {
        nest = f;
        break;
      }
    if (!nest)
      return device->emitError("pim-place-data: no loop nest inside pim.device");

    // walk 기본값은 post-order라 안쪽 loop부터 방문하므로 pre-order를 명시한다.
    SmallVector<scf::ForOp> loops;
    body->walk<WalkOrder::PreOrder>([&](scf::ForOp l) { loops.push_back(l); });

    OpBuilder b(nest);
    Location loc = device.getLoc();
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);

    ArrayRef<int64_t> partitions =
        device.getMapping().getPartitions().asArrayRef();
    auto execution =
        device->getAttrOfType<pim::ExecutionTileAttr>("pim.execution_tile");
    if (!execution)
      return device->emitError(
          "pim-place-data: the device has no #pim.execution_tile");
    ArrayRef<int64_t> tileGroups = execution.getGroups().asArrayRef();
    llvm::DenseMap<Operation *, LoopRange> ranges;
    IRMapping firstMap;
    for (scf::ForOp l : loops) {
      Value lb = rebuild(b, l.getLowerBound(), firstMap, nest);
      Value ub = rebuild(b, l.getUpperBound(), firstMap, nest);
      Value step = rebuild(b, l.getStep(), firstMap, nest);
      if (!lb || !ub || !step)
        return l->emitError("pim-place-data: loop bounds depend on a value that "
                            "cannot be rebuilt outside the loop nest");

      // 좁혀진 loop는 전역 좌표를 몸체 첫머리에서 다시 만든다. 그 op의 첫 피연산자가
      // 이 device의 시작이다.
      Operation *origin = deviceOrigin(l);
      LoopRange r;
      r.key = origin ? origin->getResult(0) : l.getInductionVar();
      r.first = origin ? origin->getOperand(0) : lb;
      r.trips = toIndex(b, loc,
                        b.create<arith::CeilDivSIOp>(
                            loc, b.create<arith::SubIOp>(loc, ub, lb), step));

      // 모든 hierarchy loop의 capacity는 Execution Tile 하나의 고정 크기다.
      // 여러 scheduling round가 같은 buffer를 재사용하므로 마지막 round의 실제
      // trips가 작더라도 allocation_shape는 변하지 않는다. 계층으로 분류되지 않은
      // loop에는 runtime trip count를 capacity로 쓰지 않는다. 그런 loop가 resident
      // origin을 움직이면 아래에서 fixed slot을 만들 수 없다고 진단한다.
      r.capacityTiles = Value();
      r.axis = StringAttr();
      r.partitions = 1;
      if (l->hasAttr("pim.temporal_reduction")) {
        r.axis = b.getStringAttr("k");
        r.partitions = partitions[2];
        r.capacityTiles =
            b.create<arith::ConstantIndexOp>(loc, tileGroups[2]);
      }
      if (auto name = l->getAttrOfType<pim::AxisAttr>("pim.axis")) {
        r.axis = name.getName();
        StringRef axis = name.getName().getValue();
        if (axis == "k")
          return l->emitError(
              "pim-place-data: source-level pim.axis=\"k\" is not part of "
              "the Common IR hierarchy; use the semantic pim.reduction loop");
        int axisNumber = axis == "m" ? 0 : 1;
        r.partitions = partitions[axisNumber];
        r.capacityTiles =
            b.create<arith::ConstantIndexOp>(loc, tileGroups[axisNumber]);
      }
      ranges[l] = r;
      firstMap.map(r.key, r.first);
    }

    SmallVector<pim::AccessOp> accesses;
    body->walk([&](pim::AccessOp a) { accesses.push_back(a); });
    for (pim::AccessOp a : accesses)
      if (failed(placeAccess(b, a, nest, loops, ranges, c0)))
        return failure();
    return success();
  }

  LogicalResult placeAccess(OpBuilder &b, pim::AccessOp access, scf::ForOp nest,
                            ArrayRef<scf::ForOp> loops,
                            const llvm::DenseMap<Operation *, LoopRange> &ranges,
                            Value c0) {
    MLIRContext *ctx = b.getContext();
    Location loc = access.getLoc();
    unsigned rank = access.getSizes().size();
    bool isOutput = llvm::any_of(access.getResult().getUsers(), [](Operation *u) {
      return isa<pim::StoreOp, pim::CombineOp>(u);
    });

    // 치환표는 access마다 새로 시작한다. 앞 access가 만든 클론을 물려받으면 그 값이
    // 이 access의 삽입 지점보다 뒤에 있을 수 있다.
    IRMapping firstMap;
    for (scf::ForOp l : loops) {
      const LoopRange &r = ranges.find(l)->second;
      firstMap.map(r.key, r.first);
    }

    SmallVector<Value> start, size, allocationShape, strides;
    for (unsigned d = 0; d < rank; d++) {
      Value origin = rebuild(b, access.getOrigin()[d], firstMap, nest);
      Value stride = rebuild(b, access.getStrides()[d], firstMap, nest);
      if (!origin || !stride)
        return access->emitError("pim-place-data: the region of buffer axis ")
               << d << " cannot be rebuilt from the device coordinate";

      llvm::SmallPtrSet<Operation *, 4> drivers;
      pim::collectDrivers(access.getOrigin()[d], loops, drivers);
      Value tile = b.create<arith::ConstantIndexOp>(loc, access.getSizes()[d]);
      Value trips, capacityTiles;
      bool first = true;
      for (scf::ForOp l : loops) {
        if (!drivers.contains(l))
          continue;
        const LoopRange &r = ranges.find(l)->second;
        if (!r.capacityTiles)
          return access->emitError("pim-place-data: buffer axis ")
                 << d
                 << " is driven by a loop outside the Block-Group-Execution "
                    "Tile hierarchy, so a fixed resident slot cannot be "
                    "derived";
        // 나뉜 축이 다른 loop와 함께 한 버퍼 축을 움직이면, blocked 분할(함께 모는
        // loop가 정확히 tile씩 전진)일 때만 연속 구간 하나다. 아니면 진단이다.
        if (r.axis && drivers.size() > 1 && r.partitions != 1 &&
            failed(checkBlockedComposite(access, d, l, r.axis, loops, ranges,
                                         drivers)))
          return failure();
        trips = first ? r.trips
                      : b.create<arith::MulIOp>(loc, trips, r.trips).getResult();
        capacityTiles = first ? r.capacityTiles
                              : b.create<arith::MulIOp>(loc, capacityTiles,
                                                        r.capacityTiles)
                                    .getResult();
        first = false;
      }
      // 움직이는 loop가 없으면 device마다 타일 하나뿐이다.
      Value span = first ? tile : b.create<arith::MulIOp>(loc, trips, tile);
      Value capacityElems =
          first ? tile : b.create<arith::MulIOp>(loc, capacityTiles, tile);

      // 도는 범위를 원본 버퍼의 크기로 자른다. 자르지 않으면 grid가 실제 크기보다 넓게
      // 잡힌 경우(마지막 tile이 통째로 무효인 경우) 버퍼 밖을 읽는다.
      Value length = span;
      if (!access.getValid().empty() &&
          !getConstantIntValue(access.getValid()[d])) {
        Value extent =
            pim::recoverExtentFromValid(access.getValid()[d], loops);
        if (!extent)
          return access->emitError("pim-place-data: cannot recover the extent of "
                                   "buffer axis ")
                 << d << " from its mask";
        Value rebuilt = rebuild(b, extent, firstMap, nest);
        if (!rebuilt)
          return access->emitError("pim-place-data: the extent of buffer axis ")
                 << d << " cannot be rebuilt outside the loop nest";
        Value remaining = b.create<arith::SubIOp>(
            loc, toIndex(b, loc, rebuilt), origin);
        length = b.create<arith::MinSIOp>(
            loc, b.create<arith::MaxSIOp>(loc, remaining, c0), span);
      }

      start.push_back(origin);
      strides.push_back(stride);
      size.push_back(length);
      allocationShape.push_back(capacityElems);
    }

    // 입력이 어느 logical 축에 불변이면 그 축만 다른 device들은 같은 데이터를 받는다.
    SmallVector<Attribute> replicated;
    if (!isOutput) {
      llvm::SmallPtrSet<Operation *, 4> used;
      // origin은 각 buffer 축의 실제 시작 좌표이며 device별 slice dependency의
      // 기준이다. offset은 origin·stride의 중복 선형화 값이므로 다시 해석하지
      // 않는다.
      for (Value v : access.getOrigin())
        pim::collectDrivers(v, loops, used);
      // stride 자체가 device 축에 따라 달라지는 access도 같은 slice가 아니다.
      for (Value v : access.getStrides())
        pim::collectDrivers(v, loops, used);
      // mask가 origin 외의 축에 추가로 의존하면 그 device들은 서로 다른 유효
      // 영역을 받으므로 replicated로 볼 수 없다.
      for (Value v : access.getValid())
        pim::collectDrivers(v, loops, used);
      if (Value pad = access.getPad())
        pim::collectDrivers(pad, loops, used);
      for (scf::ForOp l : loops) {
        const LoopRange &r = ranges.find(l)->second;
        if (r.axis && !used.contains(l))
          replicated.push_back(r.axis);
      }
    }

    b.create<pim::PlaceOp>(
        loc, access.getBase(), start, size, allocationShape, strides,
        StringAttr::get(ctx, isOutput ? "out" : "in"), access.getIdAttr(),
        replicated.empty() ? ArrayAttr() : ArrayAttr::get(ctx, replicated));
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMPlaceDataPass() {
  return std::make_unique<PIMPlaceDataPass>();
}
