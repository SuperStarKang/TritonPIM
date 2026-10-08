#include "triton-shared/Conversion/PIMToAiMX/PIMToAiMX.h"
#include "triton-shared/Conversion/PIMToAiMX/DescriptorBuilder.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXFootprint.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>

using namespace mlir;
namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTPIMTOAIMX
#include "triton-shared/Conversion/PIMToAiMX/Passes.h.inc"
}

namespace {
struct Layout {
  int64_t usefulRows, usefulCols, rows, cols;
};
struct View {
  Value buffer, offset, pitch;
  Layout layout;
};
struct Placement {
  pim::PlaceOp source;
  View view;
  int64_t rows, cols;
  bool combine = false;
  SmallVector<Value> sizes;
};

// SSA bindings are compiler-local. There is no serialized compute template,
// kernel-kind plan, or runtime interpreter of Common computations.
class DeviceLowering {
public:
  DeviceLowering(ModuleOp module, pim::PartitionOp device,
                 pim::TargetProfileAttr profile, int64_t batch,
                 int64_t matvecRowAlignment, bool matvecFirstNOnly,
                 int64_t matmulRowAlignment, bool matmulUsefulNOnly,
                 bool persistent, unsigned site)
      : module(module), device(device), profile(profile), batch(batch),
        matvecRowAlignment(matvecRowAlignment),
        matvecFirstNOnly(matvecFirstNOnly),
        matmulRowAlignment(matmulRowAlignment),
        matmulUsefulNOnly(matmulUsefulNOnly), persistent(persistent),
        prefix((device->getParentOfType<func::FuncOp>().getName() +
                "_device" + Twine(site)).str()),
        setup(module.getContext()), code(module.getContext()), loc(device.getLoc()) {}

  LogicalResult run() {
    auto partitions = device.getMapping().getPartitions().asArrayRef();
    if (partitions[2] != 1)
      return device.emitError("AiMX cross-device pim.combine is not implemented");
    if (failed(collectLayouts())) return failure();
    int64_t count;
    if (!aimx::checkedProduct({partitions[0], partitions[1], partitions[2]}, count) ||
        count > profile.getNumUnits())
      return device.emitError("AiMX mapping exceeds the target device count");
    Operation *schedule = device;
    while (auto loop = dyn_cast_or_null<scf::ForOp>(schedule->getParentOp())) {
      if (!loop->hasAttr("pim.scheduling_axis")) break;
      schedule = loop;
    }
    OpBuilder host(schedule);
    Value set = host.create<aimx::AcquireOp>(
        loc, aimx::DeviceSetType::get(module.getContext()), host.getI64IntegerAttr(count));
    host.setInsertionPoint(device);
    auto foreach = host.create<aimx::ForeachDeviceOp>(
        loc, set, host.getStrArrayAttr({"m", "n", "k"}), host.getDenseI64ArrayAttr(partitions));
    if (persistent) foreach->setAttr("aimx.persistent", host.getUnitAttr());
    SmallVector<Type> arguments{aimx::DeviceType::get(module.getContext()),
                                host.getIndexType(), host.getIndexType(), host.getIndexType()};
    Block *body = host.createBlock(&foreach.getBody(), {}, arguments,
                                   SmallVector<Location>(4, loc));
    for (unsigned i = 0; i < 3; ++i)
      indices.map(device.getBody().front().getArgument(i), body->getArgument(i + 1));
    setup.setInsertionPointToStart(body);
    deviceValue = body->getArgument(0);
    if (failed(allocate(aimx::kPMMSize, aimx::AllocKind::PMM, pmm))) return failure();
    auto chainAllocation = setup.create<aimx::AllocOp>(
        loc, aimx::BufferType::get(module.getContext()), deviceValue,
        setup.getI64IntegerAttr(aimx::kGroupBytes), setup.getI64IntegerAttr(aimx::kPIMWidth),
        aimx::AllocKindAttr::get(module.getContext(), aimx::AllocKind::Descriptor));
    chain = chainAllocation.getBuffer();

    // The whole descriptor-construction boundary: index arithmetic, layout
    // checks, SCF and build_descriptors, never retained linalg/PIM operations.
    auto construction = setup.create<scf::ExecuteRegionOp>(loc, TypeRange{});
    construction->setAttr("no_inline", setup.getUnitAttr());
    Block *constructionBody = code.createBlock(&construction.getRegion());
    code.create<scf::YieldOp>(loc);
    code.setInsertionPoint(constructionBody->getTerminator());
    setup.setInsertionPoint(construction);
    // Placement coordinates may be computed before chain construction, but
    // arithmetic using a local loop result must stay after that lowered loop.
    for (Operation &op : device.getBody().front())
      if (isIndexOperation(&op) && llvm::all_of(op.getOperands(), [&](Value value) {
            return indices.contains(value) ||
                   !device->isAncestor(value.getParentBlock()->getParentOp());
          }))
        cloneIndex(op, setup);
    if (failed(createPlacements()) || failed(prepareMasks()) || failed(initializeCombines()) ||
        failed(lowerBlock(device.getBody().front(), true)))
      return failure();
    int64_t chainBytes;
    if (!aimx::checkedProduct({std::max<int64_t>(groups, 1), aimx::kGroupBytes}, chainBytes))
      return device.emitError("AiMX descriptor chain extent overflows");
    if (failed(account(chainBytes))) return failure();
    chainAllocation.setBytesAttr(setup.getI64IntegerAttr(chainBytes));
    if (bankBytes > profile.getBankCapacityLimitBytes())
      return device.emitError("AiMX physical buffers and descriptors exceed per-bank capacity");
    SmallVector<int64_t> allocationBytes;
    body->walk([&](aimx::AllocOp allocation) { allocationBytes.push_back(allocation.getBytes()); });
    foreach->setAttr("aimx.allocations", setup.getDenseI64ArrayAttr(allocationBytes));
    foreach->setAttr("aimx.bank_bytes", setup.getI64IntegerAttr(bankBytes));
    foreach->setAttr("aimx.descriptor_bytes", setup.getI64IntegerAttr(chainBytes));
    host.setInsertionPointAfter(construction);
    // Common 전송 계획 순서 그대로: broadcast/scatter → launch → gather.
    // placement가 없는 access는 계산이 쓰지 않아 target 할당도 없다.
    Block &plan = device.getBody().front();
    for (Operation &op : plan) {
      if (auto broadcast = dyn_cast<pim::BroadcastOp>(op)) {
        auto found = placements.find(broadcast.getAccess());
        if (found != placements.end())
          stage(host, found->second, found->second.source.getSharedOverAttr());
      } else if (auto scatter = dyn_cast<pim::ScatterOp>(op)) {
        auto found = placements.find(scatter.getAccess());
        if (found != placements.end()) stage(host, found->second, ArrayAttr());
      }
    }
    host.create<aimx::InvokeOp>(loc, deviceValue, chain);
    for (auto gather : plan.getOps<pim::GatherOp>()) {
      auto found = placements.find(gather.getAccess());
      if (found != placements.end()) unstage(host, found->second);
    }
    device.erase();
    return success();
  }

private:
  ModuleOp module;
  pim::PartitionOp device;
  pim::TargetProfileAttr profile;
  int64_t batch;
  int64_t matvecRowAlignment;
  bool matvecFirstNOnly;
  int64_t matmulRowAlignment;
  bool matmulUsefulNOnly;
  bool persistent;
  std::string prefix;
  unsigned symbol = 0;
  OpBuilder setup, code;
  Location loc;
  Value deviceValue, pmm, chain;
  IRMapping indices;
  llvm::DenseMap<Value, Layout> layouts;
  llvm::DenseMap<Value, Value> layoutParents;
  llvm::DenseMap<int64_t, Value> accessValues;
  llvm::DenseMap<Value, View> values;
  llvm::MapVector<int64_t, Placement> placements;
  llvm::DenseSet<Operation *> assignedCombines;
  int64_t groups = 0, repetitions = 1, bankBytes = 0;

  std::string name(StringRef role) {
    return (Twine(prefix) + "_" + role + "_" + Twine(symbol++)).str();
  }
  Value c(OpBuilder &b, int64_t value) { return b.create<arith::ConstantIndexOp>(loc, value); }
  Value mapped(Value value) { return indices.lookupOrDefault(value); }
  SmallVector<Value> mapped(ValueRange operands) {
    SmallVector<Value> result;
    for (Value value : operands) result.push_back(mapped(value));
    return result;
  }
  Value add(Value a, Value b) { return code.create<arith::AddIOp>(loc, a, b); }
  Value mul(Value a, Value b) { return code.create<arith::MulIOp>(loc, a, b); }
  static bool sameIndex(Value a, Value b) {
    if (a == b) return true;
    auto lhs = getConstantIntValue(a), rhs = getConstantIntValue(b);
    return lhs && rhs && *lhs == *rhs;
  }
  Value sub(Value a, Value b) {
    if (sameIndex(a, b)) return c(code, 0);
    // Simplify local addresses, without a second proof of Common traversal.
    if (auto sum = a.getDefiningOp<arith::AddIOp>()) {
      if (sameIndex(sum.getLhs(), b)) return sum.getRhs();
      if (sameIndex(sum.getRhs(), b)) return sum.getLhs();
      if (auto base = b.getDefiningOp<arith::AddIOp>())
        for (unsigned i = 0; i != 2; ++i)
          for (unsigned j = 0; j != 2; ++j)
            if (sameIndex(sum->getOperand(i), base->getOperand(j)))
              return sub(sum->getOperand(1 - i), base->getOperand(1 - j));
    }
    // Common placement origins often scale a group coordinate after adding
    // its scheduling start: (start + local) * block - start * block. Cancel
    // the common scale before subtraction so only the resident-local offset
    // reaches descriptors. This also handles chained block/group scales.
    if (auto product = a.getDefiningOp<arith::MulIOp>())
      if (auto base = b.getDefiningOp<arith::MulIOp>())
        for (unsigned i = 0; i != 2; ++i)
          for (unsigned j = 0; j != 2; ++j)
            if (sameIndex(product->getOperand(i), base->getOperand(j)))
              return mul(sub(product->getOperand(1 - i), base->getOperand(1 - j)),
                         product->getOperand(i));
    return code.create<arith::SubIOp>(loc, a, b);
  }
  static bool isIndexOperation(Operation *op) {
    return op->getName().getDialectNamespace() == "arith" && op->getNumRegions() == 0 &&
           op->getNumResults() != 0 && llvm::all_of(op->getResultTypes(), [](Type t) {
             return t.isIntOrIndex();
           });
  }
  void cloneIndex(Operation &op, OpBuilder &builder) {
    Operation *clone = builder.clone(op, indices);
    SmallVector<Value> folded;
    // Use each arith op's own folding semantics on the target copy. In
    // particular, Common allocation_shape may be a product of constants.
    // Do not canonicalize away Common loops or their hierarchy attributes.
    if (succeeded(builder.tryFold(clone, folded)) && !folded.empty()) {
      for (auto [from, to] : llvm::zip(op.getResults(), folded)) indices.map(from, to);
      indices.erase(&op);
      clone->erase();
    }
  }
  LogicalResult addLayout(Operation *op, Value value) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || layouts.count(value)) return success();
    if (!type.hasStaticShape() || (type.getRank() != 1 && type.getRank() != 2))
      return op->emitError("AiMX needs static rank-one or rank-two tensor tiles");
    Type element = type.getElementType();
    if (!element.isBF16() && !element.isF16() && !element.isF32())
      return op->emitError("AiMX tensor storage supports BF16/F16/F32 source types");
    int64_t rows = type.getDimSize(0), cols = type.getRank() == 1 ? 1 : type.getDimSize(1);
    if (rows <= 0 || cols <= 0)
      return op->emitError("AiMX tensor extent cannot be represented");
    layouts[value] = {rows, cols, rows, cols};
    layoutParents[value] = value;
    return success();
  }
  Value layoutRoot(Value value) {
    while (layoutParents.lookup(value) != value)
      value = layoutParents.lookup(value);
    return value;
  }
  LogicalResult joinLayouts(Operation *op, Value lhs, Value rhs) {
    if (!isa<RankedTensorType>(lhs.getType()) ||
        !isa<RankedTensorType>(rhs.getType())) return success();
    lhs = layoutRoot(lhs);
    rhs = layoutRoot(rhs);
    if (lhs == rhs) return success();
    const Layout &a = layouts.lookup(lhs), &b = layouts.lookup(rhs);
    if (a.usefulRows != b.usefulRows || a.usefulCols != b.usefulCols)
      return op->emitError("AiMX connected tensor values need equal logical shapes");
    layoutParents[rhs] = lhs;
    return success();
  }
  LogicalResult collectLayouts() {
    WalkResult result = device.walk([&](Operation *op) {
      for (Value value : op->getOperands())
        if (failed(addLayout(op, value))) return WalkResult::interrupt();
      for (Value value : op->getResults())
        if (failed(addLayout(op, value))) return WalkResult::interrupt();
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            if (failed(addLayout(op, argument))) return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) return failure();
    // Layout compatibility follows SSA dependencies, not coincident tensor types.
    WalkResult connected = device.walk([&](Operation *op) {
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
        for (unsigned i = 0; i < loop.getNumRegionIterArgs(); ++i) {
          Value init = loop.getInitArgs()[i];
          if (failed(joinLayouts(op, init, loop.getRegionIterArgs()[i])) ||
              failed(joinLayouts(op, init, loop.getResult(i))) ||
              failed(joinLayouts(op, init, yield.getOperand(i))))
            return WalkResult::interrupt();
        }
      } else if (isa<linalg::MatmulOp, linalg::MatvecOp>(op)) {
        if (op->getNumResults() != 1 || op->getNumOperands() != 3 ||
            !llvm::all_of(op->getOperandTypes(), [](Type type) {
              return isa<RankedTensorType>(type);
            })) {
          op->emitError("AiMX MAC lowering expects tensor operands and one tensor result");
          return WalkResult::interrupt();
        }
        if (failed(joinLayouts(op, op->getOperand(2), op->getResult(0))))
          return WalkResult::interrupt();
      } else if (isa<linalg::GenericOp, linalg::FillOp, tensor::CastOp,
                     arith::AddFOp, arith::SubFOp, arith::MulFOp>(op)) {
        Value first;
        for (Value value : op->getOperands())
          if (isa<RankedTensorType>(value.getType())) {
            if (first && failed(joinLayouts(op, first, value)))
              return WalkResult::interrupt();
            first = value;
          }
        for (Value value : op->getResults())
          if (isa<RankedTensorType>(value.getType())) {
            if (first && failed(joinLayouts(op, first, value)))
              return WalkResult::interrupt();
            first = value;
          }
      }
      pim::AccessOp access;
      Value value;
      if (auto load = dyn_cast<pim::LoadOp>(op)) {
        access = load.getAccess().getDefiningOp<pim::AccessOp>();
        value = load.getResult();
      } else if (auto store = dyn_cast<pim::StoreOp>(op)) {
        access = store.getAccess().getDefiningOp<pim::AccessOp>();
        value = store.getValue();
      } else if (auto combine = dyn_cast<pim::CombineOp>(op)) {
        access = combine.getDestination().getDefiningOp<pim::AccessOp>();
        value = combine.getPartial();
      }
      if (access) {
        auto inserted = accessValues.try_emplace(access.getId(), value);
        if (!inserted.second && failed(joinLayouts(op, inserted.first->second, value)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (connected.wasInterrupted()) return failure();
    for (auto place : device.getBody().front().getOps<pim::PlaceOp>())
      for (auto other : device.getBody().front().getOps<pim::PlaceOp>())
        if (place.getBuffer() == other.getBuffer() &&
            llvm::equal(place.getStart(), other.getStart()) &&
            llvm::equal(place.getSize(), other.getSize()) &&
            llvm::equal(place.getStrides(), other.getStrides()) &&
            accessValues.count(place.getAccess()) && accessValues.count(other.getAccess()))
          if (failed(joinLayouts(place, accessValues.lookup(place.getAccess()),
                                 accessValues.lookup(other.getAccess())))) return failure();

    llvm::DenseSet<Value> macValues;
    WalkResult macLayouts = device.walk([&](Operation *op) {
      if (!isa<linalg::MatmulOp, linalg::MatvecOp>(op)) return WalkResult::advance();
      for (Value operand : op->getOperands()) macValues.insert(layoutRoot(operand));
      auto &a = layouts[layoutRoot(op->getOperand(0))];
      auto &b = layouts[layoutRoot(op->getOperand(1))];
      auto &out = layouts[layoutRoot(op->getResult(0))];
      int64_t columnAlignment = aimx::kLanes / batch;
      // Matvec storage defaults to 4096 elements, independently of MAC width.
      if (isa<linalg::MatvecOp>(op) && matvecRowAlignment != 0)
        columnAlignment = matvecRowAlignment;
      if (isa<linalg::MatmulOp>(op) && matmulRowAlignment != 0)
        columnAlignment = matmulRowAlignment;
      if (!aimx::checkedRoundUp(a.rows, batch, a.rows) ||
          !aimx::checkedRoundUp(a.cols, aimx::kLanes, a.cols) ||
          !aimx::checkedRoundUp(b.rows, aimx::kRegSize, b.rows) ||
          !aimx::checkedRoundUp(b.cols, columnAlignment, b.cols) ||
          !aimx::checkedRoundUp(out.rows, batch, out.rows) ||
          !aimx::checkedRoundUp(out.cols, columnAlignment, out.cols)) {
        op->emitError("AiMX MAC physical extent overflows");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (macLayouts.wasInterrupted()) return failure();
    for (auto &entry : layouts) {
      Value root = layoutRoot(entry.first);
      auto &layout = layouts[root];
      if (!macValues.contains(root) &&
          !aimx::checkedRoundUp(layout.cols, aimx::kLanes, layout.cols))
        return device.emitError("AiMX elementwise physical extent overflows");
    }
    for (auto &entry : layouts) {
      entry.second = layouts.lookup(layoutRoot(entry.first));
      int64_t elements;
      if (!aimx::checkedProduct({entry.second.rows, entry.second.cols}, elements) ||
          elements % aimx::kLanes != 0)
        return device.emitError("AiMX contiguous groups need a representable multiple of 4096 elements");
    }
    return success();
  }
  LogicalResult account(int64_t bytes) {
    if (bytes <= 0 || bytes > aimx::kMaxRuntimeAllocationBytes)
      return device.emitError("AiMX allocation exceeds the runtime's representable size");
    int64_t used = ((bytes + aimx::kDeviceWidth - 1) / aimx::kDeviceWidth) * aimx::kBurstSize;
    if (llvm::AddOverflow(bankBytes, used, bankBytes))
      return device.emitError("AiMX physical allocation extent overflows");
    return success();
  }
  LogicalResult allocate(int64_t bytes, aimx::AllocKind kind, Value &buffer) {
    if (failed(account(bytes))) return failure();
    buffer = setup.create<aimx::AllocOp>(
        loc, aimx::BufferType::get(module.getContext()), deviceValue,
        setup.getI64IntegerAttr(bytes),
        setup.getI64IntegerAttr(kind == aimx::AllocKind::PMM ? aimx::kPMMAlign : aimx::kDeviceWidth),
        aimx::AllocKindAttr::get(module.getContext(), kind));
    return success();
  }
  LogicalResult scratch(Layout layout, View &view) {
    int64_t bytes;
    if (!aimx::checkedProduct({layout.rows, layout.cols, aimx::kTypeSize}, bytes))
      return device.emitError("AiMX tensor allocation extent overflows");
    if (failed(allocate(bytes, aimx::AllocKind::Data, view.buffer))) return failure();
    view.offset = c(setup, 0);
    view.pitch = c(setup, layout.cols * aimx::kTypeSize);
    view.layout = layout;
    return success();
  }
  bool hasFullMask(pim::AccessOp access) {
    for (auto [axis, valid] : llvm::enumerate(access.getValid()))
      if (getConstantIntValue(mapped(valid)) != access.getSizes()[axis]) return false;
    return true;
  }
  // Common spells coordinates in both index and kernel integer arithmetic.
  // Compare only the casts, sums and scales used by those coordinates; this is
  // not a rewrite of the original arithmetic or of the descriptor addresses.
  static bool sameMaskOrigin(Value a, Value b) {
    if (sameIndex(a, b)) return true;
    if (auto cast = a.getDefiningOp<arith::IndexCastOp>())
      return sameMaskOrigin(cast.getIn(), b);
    if (auto cast = b.getDefiningOp<arith::IndexCastOp>())
      return sameMaskOrigin(a, cast.getIn());
    Operation *lhs = a.getDefiningOp(), *rhs = b.getDefiningOp();
    if (!lhs || !rhs || lhs->getName() != rhs->getName() ||
        !isa<arith::AddIOp, arith::MulIOp>(lhs)) return false;
    return (sameMaskOrigin(lhs->getOperand(0), rhs->getOperand(0)) &&
            sameMaskOrigin(lhs->getOperand(1), rhs->getOperand(1))) ||
           (sameMaskOrigin(lhs->getOperand(0), rhs->getOperand(1)) &&
            sameMaskOrigin(lhs->getOperand(1), rhs->getOperand(0)));
  }
  // Extract the shared upper boundary from an access-local rectangular clip.
  // Only these two equivalent forms describe one placement-wide rectangle:
  //   min(max(end - origin, 0), size)
  //   max(min(origin + size, end), origin) - origin
  // Common can retain redundant min(size)/max(0) wrappers. A repeated partial
  // mask or a clip relative to another coordinate must not become a full DMA.
  static Value maskEnd(Value valid, Value origin, int64_t size) {
    bool lower = false, upper = false;
    while (true) {
      if (auto cast = valid.getDefiningOp<arith::IndexCastOp>()) {
        valid = cast.getIn();
        continue;
      }
      Operation *op = valid.getDefiningOp();
      if (!op || !isa<arith::MinSIOp, arith::MaxSIOp>(op)) break;
      bool minimum = isa<arith::MinSIOp>(op);
      int64_t bound = minimum ? size : 0;
      if (getConstantIntValue(op->getOperand(0)) == bound)
        valid = op->getOperand(1);
      else if (getConstantIntValue(op->getOperand(1)) == bound)
        valid = op->getOperand(0);
      else break;
      upper |= minimum;
      lower |= !minimum;
    }
    auto sub = valid.getDefiningOp<arith::SubIOp>();
    if (!sub || !sameMaskOrigin(sub.getRhs(), origin)) return {};
    auto maximum = sub.getLhs().getDefiningOp<arith::MaxSIOp>();
    Value clipped;
    if (maximum) {
      if (sameMaskOrigin(maximum.getLhs(), origin)) clipped = maximum.getRhs();
      else if (sameMaskOrigin(maximum.getRhs(), origin)) clipped = maximum.getLhs();
    }
    auto minimum = clipped ? clipped.getDefiningOp<arith::MinSIOp>() : arith::MinSIOp();
    if (minimum)
      for (unsigned i = 0; i != 2; ++i) {
        auto end = minimum->getOperand(i).getDefiningOp<arith::AddIOp>();
        if (end &&
            ((sameMaskOrigin(end.getLhs(), origin) && getConstantIntValue(end.getRhs()) == size) ||
             (sameMaskOrigin(end.getRhs(), origin) && getConstantIntValue(end.getLhs()) == size)))
          return minimum->getOperand(1 - i);
      }
    return lower && upper ? sub.getLhs() : Value();
  }
  LogicalResult createPlacements() {
    llvm::DenseMap<int64_t, pim::AccessOp> accesses;
    device.walk([&](pim::AccessOp access) { accesses[access.getId()] = access; });
    for (Operation &op : device.getBody().front()) {
      auto place = dyn_cast<pim::PlaceOp>(op);
      if (!place) continue;
      auto value = accessValues.lookup(place.getAccess());
      if (!value) continue; // Unconsumed placement has no target allocation.
      auto type = cast<RankedTensorType>(value.getType());
      Layout layout = layouts.lookup(value);
      auto rows = getConstantIntValue(mapped(place.getAllocationShape()[0]));
      auto cols = type.getRank() == 1 ? std::optional<int64_t>(1)
                                     : getConstantIntValue(mapped(place.getAllocationShape()[1]));
      int64_t physicalRows, physicalCols, bytes;
      if (!rows || !cols || *rows % layout.usefulRows || *cols % layout.usefulCols ||
          !aimx::checkedProduct({*rows / layout.usefulRows, layout.rows}, physicalRows) ||
          !aimx::checkedProduct({*cols / layout.usefulCols, layout.cols}, physicalCols) ||
          !aimx::checkedProduct({physicalRows, physicalCols, aimx::kTypeSize}, bytes))
        return place.emitError("AiMX needs a representable fixed Common allocation_shape");
      Placement placement{place, {}, physicalRows, physicalCols};
      placement.sizes = mapped(place.getSize());
      // Identical Common input/output placements of one memref share storage.
      // This is necessary for a later pim.load to observe an earlier store.
      for (auto &entry : placements) {
        Placement &previous = entry.second;
        if (previous.source.getBuffer() != place.getBuffer()) continue;
        auto previousAccess = accesses.lookup(previous.source.getAccess());
        auto currentAccess = accesses.lookup(place.getAccess());
        // Two input masks can stage different rectangles of the same host
        // buffer. They must not overwrite one shared resident allocation.
        if (llvm::equal(previous.source.getStart(), place.getStart()) &&
            llvm::equal(previous.source.getSize(), place.getSize()) &&
            llvm::equal(previous.source.getStrides(), place.getStrides()) &&
            ((hasFullMask(previousAccess) && hasFullMask(currentAccess)) ||
             (llvm::equal(previousAccess.getValid(), currentAccess.getValid()) &&
              previousAccess.getPad() == currentAccess.getPad())) &&
            previous.rows == physicalRows && previous.cols == physicalCols &&
            previous.view.layout.usefulRows == layout.usefulRows &&
            previous.view.layout.usefulCols == layout.usefulCols &&
            previous.view.layout.rows == layout.rows && previous.view.layout.cols == layout.cols) {
          placement.view = previous.view;
          break;
        }
        if (previous.source.getDirection() == "out" || place.getDirection() == "out")
          return place.emitError("AiMX non-identical read/write placements of one memref need a shared address mapping");
      }
      if (placement.view.buffer) {
        placements[place.getAccess()] = placement;
        continue;
      }
      if (failed(allocate(bytes, aimx::AllocKind::Data, placement.view.buffer))) return failure();
      placement.view.offset = c(setup, 0);
      placement.view.pitch = c(setup, layout.cols * aimx::kTypeSize);
      placement.view.layout = layout;
      placements[place.getAccess()] = placement;
    }
    return success();
  }
  FailureOr<Value> firstMaskValue(Value value, IRMapping &first) {
    if (Value result = first.lookupOrNull(value)) return result;
    Operation *owner = value.getParentBlock()->getParentOp();
    if (!device->isAncestor(owner)) return value;
    if (owner == device && indices.contains(value)) return mapped(value);
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto loop = dyn_cast<scf::ForOp>(owner);
      if (!loop) {
        owner->emitError("AiMX mask value cannot be evaluated at the placement boundary");
        return failure();
      }
      auto capacity = loopCapacity(loop);
      if (failed(capacity)) return failure();
      if (*capacity != 1) {
        loop.emitError("AiMX varying masks across multiple groups need per-group masked transfers");
        return failure();
      }
      Value initial = argument.getArgNumber() == 0
                          ? loop.getLowerBound()
                          : loop.getInitArgs()[argument.getArgNumber() - 1];
      auto result = firstMaskValue(initial, first);
      if (succeeded(result)) first.map(value, *result);
      return result;
    }
    Operation *definition = value.getDefiningOp();
    if (!definition || !isIndexOperation(definition) || !isSpeculatable(definition)) {
      owner->emitError("AiMX mask value must be an integer/index expression safe before a possibly empty group");
      return failure();
    }
    for (Value operand : definition->getOperands()) {
      auto result = firstMaskValue(operand, first);
      if (failed(result)) return failure();
      first.map(operand, *result);
    }
    Operation *clone = setup.clone(*definition, first);
    SmallVector<Value> folded;
    if (succeeded(setup.tryFold(clone, folded)) && !folded.empty()) {
      for (auto [from, to] : llvm::zip(definition->getResults(), folded)) first.map(from, to);
      first.erase(definition);
      clone->erase();
    }
    return first.lookup(value);
  }
  LogicalResult prepareMasks() {
    llvm::DenseSet<Value> read, written;
    device.walk([&](pim::LoadOp load) {
      auto access = load.getAccess().getDefiningOp<pim::AccessOp>();
      auto found = placements.find(access.getId());
      if (found != placements.end()) read.insert(found->second.view.buffer);
    });
    device.walk([&](pim::StoreOp store) {
      auto access = store.getAccess().getDefiningOp<pim::AccessOp>();
      auto found = placements.find(access.getId());
      if (found != placements.end()) written.insert(found->second.view.buffer);
    });
    device.walk([&](pim::CombineOp combine) {
      auto access = combine.getDestination().getDefiningOp<pim::AccessOp>();
      auto found = placements.find(access.getId());
      if (found != placements.end()) written.insert(found->second.view.buffer);
    });
    WalkResult result = device.walk([&](pim::AccessOp access) {
      auto found = placements.find(access.getId());
      if (found == placements.end()) return WalkResult::advance();
      if (hasFullMask(access)) return WalkResult::advance();
      Placement &placement = found->second;
      IRMapping first;
      bool empty = llvm::any_of(access.getValid(), [&](Value valid) {
        return getConstantIntValue(mapped(valid)) == 0;
      });
      for (auto [axis, valid] : llvm::enumerate(access.getValid())) {
        if (getConstantIntValue(mapped(valid)) == access.getSizes()[axis]) continue;
        // A statically empty rectangle needs no per-group mask expansion.
        // Keep one empty dimension; the other bounds cannot expose any data.
        if (empty && getConstantIntValue(mapped(valid)) != 0) continue;
        // Across resident groups, project an access-local clip to the same
        // global end at the placement start. Keep Common's actual size (which
        // also accounts for scheduling tails), never the allocation capacity
        // or the first group's valid size. The existing stage/unstage layout
        // then handles the whole rectangle, including per-group physical pad.
        if (!empty && getConstantIntValue(mapped(placement.source.getAllocationShape()[axis])) !=
            access.getSizes()[axis]) {
          Value end = maskEnd(valid, access.getOrigin()[axis], access.getSizes()[axis]);
          if (!end) {
            access.emitError("AiMX masks spanning multiple resident groups need a shared rectangular bound; per-group masked transfers are not implemented");
            return WalkResult::interrupt();
          }
          auto extent = firstMaskValue(end, first);
          if (failed(extent)) return WalkResult::interrupt();
          end = *extent;
          if (!end.getType().isIndex())
            end = setup.createOrFold<arith::IndexCastOp>(loc, setup.getIndexType(), end);
          end = setup.createOrFold<arith::SubIOp>(
              loc, end, mapped(placement.source.getStart()[axis]));
          end = setup.createOrFold<arith::MaxSIOp>(loc, end, c(setup, 0));
          placement.sizes[axis] = setup.createOrFold<arith::MinSIOp>(
              loc, placement.sizes[axis], end);
          continue;
        }
        // A single resident group still uses its explicit valid value.
        auto extent = firstMaskValue(valid, first);
        if (failed(extent)) return WalkResult::interrupt();
        placement.sizes[axis] = setup.createOrFold<arith::MinSIOp>(
            loc, placement.sizes[axis], *extent);
      }
      // A masked store followed by a load must preserve the old inactive
      // elements. Zero-filled staging alone cannot implement that dependency.
      if (read.contains(placement.view.buffer) && written.contains(placement.view.buffer)) {
        access.emitError("AiMX masked read/write aliases need a masked descriptor merge");
        return WalkResult::interrupt();
      }
      if (read.contains(placement.view.buffer) && access.getPad()) {
        Attribute pad;
        if (!matchPattern(access.getPad(), m_Constant(&pad)) ||
            !isa<FloatAttr>(pad) || !cast<FloatAttr>(pad).getValue().isZero() ||
            cast<FloatAttr>(pad).getValue().isNegative()) {
          access.emitError("AiMX masked input staging supports only positive-zero or unspecified padding");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }
  LogicalResult initializeCombines() {
    device.walk([&](pim::CombineOp combine) {
      auto access = combine.getDestination().getDefiningOp<pim::AccessOp>();
      auto found = placements.find(access.getId());
      if (found != placements.end()) found->second.combine = true;
      if (canAssignMACCombine(combine)) assignedCombines.insert(combine);
    });
    for (auto &entry : placements) {
      Placement &placement = entry.second;
      if (!placement.combine) continue;
      for (auto &other : placements)
        if (other.second.source.getDirection() == "in" &&
            other.second.view.buffer == placement.view.buffer)
          return placement.source.emitError("AiMX combine identity storage cannot alias a staged input");
      View zero;
      Layout extent{placement.rows, placement.cols, placement.rows, placement.cols};
      View whole = placement.view;
      whole.layout = extent;
      whole.pitch = c(setup, extent.cols * aimx::kTypeSize);
      if (failed(constant(setup.getFloatAttr(setup.getBF16Type(), 0.0), extent, zero)) ||
          failed(copy(zero, whole))) return failure();
    }
    return success();
  }
  SmallVector<Value> hostValues(OpBuilder &b, ValueRange operands, int64_t extra) {
    auto result = mapped(operands);
    if (result.size() == 1) result.push_back(c(b, extra));
    return result;
  }
  void stage(OpBuilder &b, const Placement &placement, ArrayAttr sharedOver) {
    auto place = placement.source;
    const auto &layout = placement.view.layout;
    b.create<aimx::StageOp>(
        loc, mapped(place.getBuffer()), hostValues(b, place.getStart(), 0),
        hostValues(b, placement.sizes, 1), hostValues(b, place.getStrides(), 0),
        ValueRange{c(b, placement.rows), c(b, placement.cols)}, c(b, placement.cols),
        placement.view.buffer, TypeAttr::get(b.getBF16Type()), UnitAttr(),
        DenseI64ArrayAttr(), DenseI64ArrayAttr(), sharedOver, b.getDenseI64ArrayAttr({
        layout.usefulRows, layout.usefulCols, layout.rows, layout.cols,
        placement.rows / layout.rows, placement.cols / layout.cols}));
  }
  void unstage(OpBuilder &b, const Placement &placement) {
    Value accumulate = c(b, 0);
    if (placement.combine)
      for (Operation *parent = device->getParentOp(); parent; parent = parent->getParentOp()) {
        auto loop = dyn_cast<scf::ForOp>(parent);
        auto axis = loop ? loop->getAttrOfType<StringAttr>("pim.scheduling_axis") : StringAttr();
        if (axis && axis.getValue() == "reduction_k") {
          Value later = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                               loop.getInductionVar(), loop.getLowerBound());
          accumulate = b.create<arith::SelectOp>(loc, later, c(b, 1), accumulate);
          break;
        }
      }
    auto place = placement.source;
    const auto &layout = placement.view.layout;
    b.create<aimx::UnstageOp>(
        loc, placement.view.buffer, mapped(place.getBuffer()),
        hostValues(b, place.getStart(), 0), hostValues(b, placement.sizes, 1),
        hostValues(b, place.getStrides(), 0), c(b, placement.cols), accumulate,
        TypeAttr::get(b.getBF16Type()), UnitAttr(), DenseI64ArrayAttr(),
        DenseI64ArrayAttr(), b.getDenseI64ArrayAttr({
        layout.usefulRows, layout.usefulCols, layout.rows, layout.cols,
        placement.rows / layout.rows, placement.cols / layout.cols}));
  }
  FailureOr<View> accessView(pim::AccessOp access) {
    auto found = placements.find(access.getId());
    if (found == placements.end()) {
      access.emitError("AiMX access has no resident placement"); return failure();
    }
    Placement &placement = found->second;
    View view = placement.view;
    if (access.getOrigin().empty()) {
      access.emitError("AiMX blocked placement needs Common access origins"); return failure();
    }
    // Origin/stride is Common's coordinate representation of its linear offset.
    // Translate these coordinates; do not reverse-engineer a kernel loop pattern.
    Value row = sub(mapped(access.getOrigin()[0]), mapped(placement.source.getStart()[0]));
    Value col = c(code, 0);
    if (access.getOrigin().size() == 2)
      col = sub(mapped(access.getOrigin()[1]), mapped(placement.source.getStart()[1]));
    Value rowGroup = code.create<arith::DivSIOp>(loc, row, c(code, view.layout.usefulRows));
    Value colGroup = code.create<arith::DivSIOp>(loc, col, c(code, view.layout.usefulCols));
    Value panel = add(mul(rowGroup, c(code, placement.cols / view.layout.cols)), colGroup);
    Value innerRow = code.create<arith::RemSIOp>(loc, row, c(code, view.layout.usefulRows));
    Value innerCol = code.create<arith::RemSIOp>(loc, col, c(code, view.layout.usefulCols));
    for (auto [coordinate, remainder, extent] : {
             std::tuple{row, innerRow, placement.rows / view.layout.rows * view.layout.usefulRows},
             std::tuple{col, innerCol, placement.cols / view.layout.cols * view.layout.usefulCols}}) {
      code.create<aimx::RequireOp>(loc,
          code.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, remainder, c(code, 0)),
          code.getStringAttr("AiMX access must start at a contiguous group boundary"));
      code.create<aimx::RequireOp>(loc,
          code.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, coordinate, c(code, 0)),
          code.getStringAttr("AiMX access precedes its resident placement"));
      code.create<aimx::RequireOp>(loc,
          code.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, coordinate, c(code, extent)),
          code.getStringAttr("AiMX access exceeds its resident placement"));
    }
    view.offset = mul(panel, c(code, view.layout.rows * view.layout.cols * aimx::kTypeSize));
    return view;
  }
  LogicalResult append(aimx::DescriptorSequence sequence, ArrayRef<View> views,
                       ValueRange buffers = {}) {
    int64_t count;
    if (llvm::MulOverflow(sequence.groups, repetitions, count) ||
        llvm::AddOverflow(groups, count, groups))
      return device.emitError("AiMX descriptor extent overflows");
    SmallVector<Value> args;
    for (const View &view : views) args.append({view.buffer, view.offset, view.pitch});
    llvm::append_range(args, buffers);
    code.create<aimx::BuildDescriptorsOp>(
        loc, FlatSymbolRefAttr::get(module.getContext(), sequence.program.getSymName()), args, chain);
    return success();
  }
  LogicalResult copy(const View &from, const View &to) {
    if (from.buffer == to.buffer && from.offset == to.offset && from.pitch == to.pitch)
      return success();
    if (from.layout.rows != to.layout.rows || from.layout.cols != to.layout.cols)
      return device.emitError("AiMX copy requires matching physical group layouts");
    return append(aimx::buildCopy(module, name("copy"), from.layout.rows, from.layout.cols),
                  {from, to});
  }
  LogicalResult binary(aimx::Opcode opcode, const View &a, const View &b, const View &out) {
    if (a.layout.rows != out.layout.rows || a.layout.cols != out.layout.cols ||
        b.layout.rows != out.layout.rows || b.layout.cols != out.layout.cols)
      return device.emitError("AiMX elementwise operands need matching physical group layouts");
    return append(aimx::buildElementwise(module, name("elementwise"), opcode,
                                         out.layout.rows, out.layout.cols),
                  {a, b, out}, ValueRange{pmm});
  }
  // Standard memref globals provide scalar fill data to existing staging.
  LogicalResult constant(Attribute value, Layout layout, View &view) {
    auto floating = dyn_cast<FloatAttr>(value);
    if (!floating)
      return device.emitError("AiMX tensor constants require a scalar floating-point value");
    if (failed(scratch(layout, view))) return failure();
    OpBuilder globals(module.getBody(), module.getBody()->begin());
    std::string symbol = name("constant");
    auto type = MemRefType::get({1}, floating.getType());
    auto tensorType = RankedTensorType::get({1}, floating.getType());
    auto initial = DenseElementsAttr::get(tensorType, ArrayRef<Attribute>{floating});
    globals.create<memref::GlobalOp>(
        loc, globals.getStringAttr(symbol), globals.getStringAttr("private"),
        TypeAttr::get(type), initial, globals.getUnitAttr(), IntegerAttr());
    Value source = setup.create<memref::GetGlobalOp>(loc, type, symbol);
    Value zero = c(setup, 0);
    setup.create<aimx::StageOp>(
        loc, source, ValueRange{zero, zero},
        ValueRange{c(setup, layout.usefulRows), c(setup, layout.usefulCols)},
        ValueRange{zero, zero}, ValueRange{c(setup, layout.rows), c(setup, layout.cols)},
        c(setup, layout.cols), view.buffer, TypeAttr::get(setup.getBF16Type()), UnitAttr(),
        DenseI64ArrayAttr(), DenseI64ArrayAttr(), ArrayAttr(), DenseI64ArrayAttr());
    return success();
  }
  FailureOr<View> valueView(Value value) {
    auto found = values.find(value);
    if (found != values.end()) return found->second;
    if (value.getDefiningOp<tensor::EmptyOp>()) {
      View view;
      if (failed(scratch(layouts.lookup(value), view))) return failure();
      values[value] = view;
      return view;
    }
    value.getParentBlock()->getParentOp()->emitError("AiMX operand has no lowered SSA value");
    return failure();
  }
  LogicalResult lowerBinary(Operation *op, Layout layout) {
    aimx::Opcode opcode;
    if (isa<arith::AddFOp>(op)) opcode = aimx::Opcode::add;
    else if (isa<arith::SubFOp>(op)) opcode = aimx::Opcode::sub;
    else if (isa<arith::MulFOp>(op)) opcode = aimx::Opcode::mul;
    else return op->emitError("AiMX has no descriptor lowering for this scalar operation");
    auto operand = [&](Value value) -> FailureOr<View> {
      auto found = values.find(value);
      if (found != values.end()) return found->second;
      Attribute attribute;
      if (isa<FloatType>(value.getType()) && matchPattern(value, m_Constant(&attribute))) {
        View view;
        if (failed(constant(attribute, layout, view))) return failure();
        return view;
      }
      return valueView(value);
    };
    auto a = operand(op->getOperand(0)), b = operand(op->getOperand(1));
    if (failed(a) || failed(b)) return failure();
    View out;
    // A/B are read before C drains in each native elementwise chunk. Reuse
    // only a private scalar-body intermediate whose last use is this op.
    // Region arguments, staged constants, and loop-carried buffers are not
    // owned by this scalar body and must remain intact for other consumers.
    if (isa<linalg::GenericOp>(op->getParentOp()))
      for (Value value : op->getOperands()) {
        Operation *definition = value.getDefiningOp();
        if (definition && definition->getBlock() == op->getBlock() &&
            isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(definition) &&
            value.hasOneUse()) {
          out = values.lookup(value);
          break;
        }
      }
    if (!out.buffer && failed(scratch(layout, out))) return failure();
    if (failed(binary(opcode, *a, *b, out))) return failure();
    values[op->getResult(0)] = out;
    return success();
  }
  LogicalResult lowerGeneric(linalg::GenericOp generic) {
    for (auto iterator : generic.getIteratorTypesArray())
      if (iterator != utils::IteratorType::parallel)
        return generic.emitError("AiMX generic reduction lowering is not implemented");
    for (AffineMap map : generic.getIndexingMapsArray())
      if (!map.isIdentity())
        return generic.emitError("AiMX generic projected/transpose indexing maps are not implemented");
    if (generic->getNumResults() == 0)
      return generic.emitError("AiMX generic lowering expects tensor results");
    Layout layout = layouts.lookup(generic->getResult(0));
    for (auto [argument, operand] : llvm::zip(generic.getBody()->getArguments(), generic->getOperands())) {
      if (argument.use_empty()) continue;
      auto view = valueView(operand);
      if (failed(view)) return failure();
      values[argument] = *view;
    }
    for (Operation &op : *generic.getBody()) {
      if (auto yield = dyn_cast<linalg::YieldOp>(op)) {
        for (auto [result, value] : llvm::zip(generic->getResults(), yield.getValues())) {
          auto view = valueView(value);
          if (failed(view)) return failure();
          values[result] = *view;
        }
      } else if (auto scalar = dyn_cast<arith::ConstantOp>(op)) {
        View view;
        if (failed(constant(scalar.getValue(), layout, view))) return failure();
        values[scalar.getResult()] = view;
      } else if (failed(lowerBinary(&op, layout))) return failure();
    }
    return success();
  }
  LogicalResult validateMAC(Operation *op) {
    auto linalg = cast<linalg::LinalgOp>(op);
    bool vector = isa<linalg::MatvecOp>(op);
    Block &body = op->getRegion(0).front();
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    auto addition = yield && yield.getValues().size() == 1
                        ? yield.getValues()[0].getDefiningOp<arith::AddFOp>() : arith::AddFOp();
    Value product;
    if (addition) {
      if (addition.getLhs() == body.getArgument(2)) product = addition.getRhs();
      if (addition.getRhs() == body.getArgument(2)) product = addition.getLhs();
    }
    auto multiply = product ? product.getDefiningOp<arith::MulFOp>() : arith::MulFOp();
    llvm::SmallPtrSet<Operation *, 8> matched;
    auto input = [&](Value value, unsigned i) {
      if (auto ext = value.getDefiningOp<arith::ExtFOp>()) {
        matched.insert(ext);
        value = ext.getIn();
      }
      return value == body.getArgument(i);
    };
    auto ctx = module.getContext();
    auto m = getAffineDimExpr(0, ctx), n = getAffineDimExpr(1, ctx);
    auto k = getAffineDimExpr(vector ? 1 : 2, ctx);
    unsigned rank = vector ? 2 : 3;
    SmallVector<AffineMap> maps{
        AffineMap::get(rank, 0, {m, k}, ctx),
        vector ? AffineMap::get(rank, 0, {k}, ctx) : AffineMap::get(rank, 0, {k, n}, ctx),
        vector ? AffineMap::get(rank, 0, {m}, ctx) : AffineMap::get(rank, 0, {m, n}, ctx)};
    if (!multiply || !input(multiply.getLhs(), 0) || !input(multiply.getRhs(), 1) ||
        linalg.getIndexingMapsArray() != maps)
      return op->emitError("AiMX MAC cannot represent this operation body/indexing map");
    matched.insert(addition);
    matched.insert(multiply);
    matched.insert(yield);
    for (Operation &scalar : body)
      if (!matched.contains(&scalar) && !isOpTriviallyDead(&scalar))
        return scalar.emitError("AiMX MAC has an unlowered scalar operation");
    return success();
  }
  LogicalResult lowerMAC(Operation *op) {
    if (failed(validateMAC(op))) return failure();
    bool vector = isa<linalg::MatvecOp>(op);
    auto a = valueView(op->getOperand(0)), b = valueView(op->getOperand(1));
    auto acc = valueView(op->getOperand(2));
    if (failed(a) || failed(b) || failed(acc)) return failure();
    Layout layout = layouts.lookup(op->getResult(0));
    View partial, out, zero;
    if (failed(scratch(layout, partial)) || failed(scratch(layout, out)) ||
        failed(constant(setup.getFloatAttr(setup.getBF16Type(), 0.0), layout, zero)) ||
        failed(copy(zero, partial))) return failure();
    int64_t kPhysical, arelBytes;
    if (!aimx::checkedRoundUp(a->layout.usefulCols, aimx::kRegSize, kPhysical) ||
        !aimx::checkedProduct({layout.rows, kPhysical, aimx::kTypeSize, aimx::kNumChannels}, arelBytes))
      return op->emitError("AiMX MAC relayout extent overflows");
    Value arel;
    if (failed(allocate(arelBytes, aimx::AllocKind::Data, arel))) return failure();
    int64_t columns = layout.usefulCols;
    if (!vector && !matmulUsefulNOnly) columns = layout.cols;
    if (failed(append(aimx::buildMatmul(module, name(vector ? "matvec" : "matmul"),
                                       batch, layout.usefulRows, a->layout.usefulCols,
                                       columns), {*a, *b, partial}, ValueRange{arel, pmm})) ||
        failed(binary(aimx::Opcode::add, *acc, partial, out))) return failure();
    values[op->getResult(0)] = out;
    return success();
  }
  static bool isMACReduction(scf::ForOp loop) {
    if (!loop->hasAttr("pim.temporal_reduction")) return false;
    bool hasMAC = false;
    loop.walk([&](Operation *op) {
      if (isa<linalg::MatmulOp, linalg::MatvecOp>(op)) hasMAC = true;
    });
    return hasMAC;
  }
  static std::optional<int64_t> coefficient(Value value, BlockArgument induction) {
    if (value == induction) return 1;
    Operation *owner = induction.getOwner()->getParentOp();
    if (!owner->isAncestor(value.getParentBlock()->getParentOp())) return 0;
    if (auto argument = dyn_cast<BlockArgument>(value))
      if (isa<scf::ForOp>(argument.getOwner()->getParentOp()) && argument.getArgNumber() == 0)
        return 0;
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
      return coefficient(cast.getIn(), induction);
    Operation *op = value.getDefiningOp();
    if (!op || !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
      if (getConstantIntValue(value)) return 0;
      return std::nullopt;
    }
    auto lhs = coefficient(op->getOperand(0), induction);
    auto rhs = coefficient(op->getOperand(1), induction);
    if (!lhs || !rhs) return std::nullopt;
    int64_t result;
    if (isa<arith::AddIOp>(op)) {
      if (llvm::AddOverflow(*lhs, *rhs, result)) return std::nullopt;
    } else if (isa<arith::SubIOp>(op)) {
      if (llvm::SubOverflow(*lhs, *rhs, result)) return std::nullopt;
    } else {
      if (*lhs == 0 && *rhs == 0) return 0;
      auto scale = getConstantIntValue(op->getOperand(*lhs == 0 ? 0 : 1));
      if (!scale || (*lhs != 0 && *rhs != 0) ||
          llvm::MulOverflow(*scale, *lhs == 0 ? *rhs : *lhs, result))
        return std::nullopt;
    }
    return result;
  }
  bool canAssignMACCombine(pim::CombineOp combine) {
    auto reduction = combine.getPartial().getDefiningOp<scf::ForOp>();
    if (combine.getKind() != "add" || !reduction || !isMACReduction(reduction))
      return false;
    auto access = combine.getDestination().getDefiningOp<pim::AccessOp>();
    if (access.getOrigin().size() != 1 && access.getOrigin().size() != 2)
      return false;
    bool exclusive = true;
    device.walk([&](Operation *op) {
      Value destination;
      if (auto other = dyn_cast<pim::CombineOp>(op)) {
        if (other == combine) return;
        destination = other.getDestination();
      } else if (auto store = dyn_cast<pim::StoreOp>(op)) {
        destination = store.getAccess();
      } else if (auto load = dyn_cast<pim::LoadOp>(op)) {
        destination = load.getAccess();
      }
      if (destination && destination.getDefiningOp<pim::AccessOp>().getBase() ==
                             access.getBase()) exclusive = false;
    });
    if (!exclusive) return false;
    bool seen[2] = {false, false};
    for (Operation *parent = combine->getParentOp(); parent != device;
         parent = parent->getParentOp()) {
      if (isa<pim::LaunchOp>(parent)) continue;
      auto loop = dyn_cast<scf::ForOp>(parent);
      auto axis = loop ? loop->getAttrOfType<pim::AxisAttr>("pim.axis") : pim::AxisAttr();
      if (!axis || (axis.getName().getValue() != "m" && axis.getName().getValue() != "n"))
        return false;
      unsigned dimension = axis.getName().getValue() == "m" ? 0 : 1;
      if (dimension >= access.getOrigin().size() || seen[dimension]) return false;
      seen[dimension] = true;
      auto step = getConstantIntValue(loop.getStep());
      auto induction = llvm::cast<BlockArgument>(loop.getInductionVar());
      auto stride = coefficient(access.getOrigin()[dimension], induction);
      int64_t distance;
      if (!step || *step <= 0 || !stride ||
          llvm::MulOverflow(*step, *stride, distance) || distance < access.getSizes()[dimension])
        return false;
      for (unsigned axis = 0; axis < access.getOrigin().size(); ++axis) {
        if (axis == dimension) continue;
        auto other = coefficient(access.getOrigin()[axis], induction);
        if (!other || *other != 0) return false;
      }
    }
    return true;
  }
  FailureOr<int64_t> loopCapacity(scf::ForOp loop) {
    auto lower = getConstantIntValue(mapped(loop.getLowerBound()));
    auto upper = getConstantIntValue(mapped(loop.getUpperBound()));
    auto step = getConstantIntValue(mapped(loop.getStep()));
    if (lower && upper && step && *step > 0) {
      if (loop.getUnsignedCmp()) {
        Type type = loop.getLowerBound().getType();
        unsigned width = isa<IndexType>(type) ? 64 : cast<IntegerType>(type).getWidth();
        if (width > 64) {
          loop.emitError("AiMX descriptor loop bounds must fit 64 bits");
          return failure();
        }
        uint64_t start = APInt(width, *lower).getZExtValue();
        uint64_t end = APInt(width, *upper).getZExtValue();
        if (end <= start) return int64_t(0);
        uint64_t count = (end - start) / *step + ((end - start) % *step != 0);
        if (count > uint64_t(std::numeric_limits<int64_t>::max())) {
          loop.emitError("AiMX descriptor loop extent overflows");
          return failure();
        }
        return static_cast<int64_t>(count);
      }
      int64_t span;
      if (llvm::SubOverflow(*upper, *lower, span)) {
        loop.emitError("AiMX descriptor loop extent overflows");
        return failure();
      }
      return span <= 0 ? 0 : span / *step + (span % *step != 0);
    }
    auto execution = device->getAttrOfType<pim::ExecutionTileAttr>("pim.execution_tile");
    auto axis = loop->getAttrOfType<pim::AxisAttr>("pim.axis");
    if (execution && axis) {
      StringRef name = axis.getName().getValue();
      return execution.getGroups().asArrayRef()[name == "m" ? 0 : name == "n" ? 1 : 2];
    }
    if (execution && loop->hasAttr("pim.temporal_reduction"))
      return execution.getGroups().asArrayRef()[2];
    // Common reuse의 0/1회 보호 loop처럼 상한이 두 상수 중 하나면 큰 쪽이 상한이다.
    if (auto select = loop.getUpperBound().getDefiningOp<arith::SelectOp>()) {
      auto t = getConstantIntValue(select.getTrueValue());
      auto f = getConstantIntValue(select.getFalseValue());
      if (lower == 0 && step == 1 && t && f && !loop.getUnsignedCmp())
        return std::max({*t, *f, int64_t(0)});
    }
    loop.emitError("AiMX descriptor storage needs a bounded Common loop extent");
    return failure();
  }
  LogicalResult lowerLoop(scf::ForOp loop) {
    if (isMACReduction(loop)) return lowerMACReduction(loop);
    SmallVector<View> carried;
    SmallVector<Value> scalarInitial;
    for (auto [argument, init] : llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs())) {
      if (argument.getType().isIntOrIndex()) {
        scalarInitial.push_back(mapped(init));
        continue;
      }
      if (!isa<RankedTensorType>(argument.getType()))
        return loop.emitError("AiMX loop-carried values must be tensor data or integer/index control values");
      View view;
      auto initial = valueView(init);
      if (failed(initial) || failed(scratch(layouts.lookup(argument), view)) ||
          failed(copy(*initial, view))) return failure();
      carried.push_back(view);
      values[argument] = view;
    }
    auto capacity = loopCapacity(loop);
    if (failed(capacity)) return failure();
    int64_t outer = repetitions;
    if (llvm::MulOverflow(repetitions, *capacity, repetitions))
      return loop.emitError("AiMX descriptor loop extent overflows");
    auto lowered = code.create<scf::ForOp>(loc, mapped(loop.getLowerBound()),
                                          mapped(loop.getUpperBound()), mapped(loop.getStep()),
                                          scalarInitial);
    lowered->setAttrs(loop->getAttrs());
    indices.map(loop.getInductionVar(), lowered.getInductionVar());
    unsigned scalarPosition = 0;
    for (BlockArgument argument : loop.getRegionIterArgs())
      if (argument.getType().isIntOrIndex())
        indices.map(argument, lowered.getRegionIterArgs()[scalarPosition++]);
    // SCF builds an implicit terminator only for loops without iter_args.
    if (!scalarInitial.empty()) {
      code.setInsertionPointToEnd(lowered.getBody());
      code.create<scf::YieldOp>(loc, lowered.getRegionIterArgs());
    }
    code.setInsertionPoint(lowered.getBody()->getTerminator());
    if (failed(lowerBlock(*loop.getBody(), false))) return failure();
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    SmallVector<View> yielded;
    SmallVector<Value> scalarYielded;
    for (Value value : yield.getOperands()) {
      if (value.getType().isIntOrIndex()) {
        scalarYielded.push_back(mapped(value));
        continue;
      }
      auto view = valueView(value);
      if (failed(view)) return failure();
      yielded.push_back(*view);
    }
    // Preserve parallel loop-carried assignment, including swapping tensors.
    if (yielded.size() > 1)
      for (View &view : yielded) {
        View snapshot;
        if (failed(scratch(view.layout, snapshot)) || failed(copy(view, snapshot))) return failure();
        view = snapshot;
      }
    for (auto [from, to] : llvm::zip(yielded, carried))
      if (failed(copy(from, to))) return failure();
    lowered.getBody()->getTerminator()->setOperands(scalarYielded);
    code.setInsertionPointAfter(lowered);
    repetitions = outer;
    unsigned tensorPosition = 0;
    scalarPosition = 0;
    for (Value result : loop.getResults()) {
      if (result.getType().isIntOrIndex())
        indices.map(result, lowered.getResult(scalarPosition++));
      else
        values[result] = carried[tensorPosition++];
    }
    return success();
  }
  LogicalResult lowerMACReduction(scf::ForOp loop) {
    auto reduction = loop->getAttrOfType<StringAttr>("pim.reduction");
    auto step = getConstantIntValue(mapped(loop.getStep()));
    if (!reduction || reduction.getValue() != "add" ||
        loop.getNumRegionIterArgs() != 1 || !step || *step <= 0 || loop.getUnsignedCmp())
      return loop.emitError("AiMX vACC reduction needs one tensor accumulator and a positive constant signed step");
    auto fill = loop.getInitArgs()[0].getDefiningOp<linalg::FillOp>();
    Attribute initial;
    if (!fill || !matchPattern(fill.getInputs()[0], m_Constant(&initial)) ||
        !isa<FloatAttr>(initial) || !cast<FloatAttr>(initial).getValue().isZero() ||
        cast<FloatAttr>(initial).getValue().isNegative())
      return loop.emitError("AiMX vACC reduction requires a positive-zero accumulator");
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    Operation *mac = yield.getOperand(0).getDefiningOp();
    if (!mac || !isa<linalg::MatmulOp, linalg::MatvecOp>(mac) ||
        mac->getBlock() != loop.getBody() ||
        mac->getOperand(2) != loop.getRegionIterArgs()[0] ||
        !loop.getRegionIterArgs()[0].hasOneUse() || !mac->getResult(0).hasOneUse())
      return loop.emitError("AiMX vACC reduction requires an unobserved MAC accumulator carried directly to the next K group");
    if (failed(validateMAC(mac))) return failure();
    auto loadA = mac->getOperand(0).getDefiningOp<pim::LoadOp>();
    auto loadB = mac->getOperand(1).getDefiningOp<pim::LoadOp>();
    if (!loadA || !loadB || loadA->getBlock() != loop.getBody() ||
        loadB->getBlock() != loop.getBody())
      return loop.emitError("AiMX vACC reduction requires direct resident A/B loads in each K group");
    for (Operation &op : *loop.getBody()) {
      if (&op == mac || &op == loadA || &op == loadB || &op == yield ||
          isa<pim::AccessOp>(op) || isIndexOperation(&op)) continue;
      if (auto constant = dyn_cast<arith::ConstantOp>(op))
        if (isa<FloatAttr>(constant.getValue())) continue;
      return op.emitError("AiMX vACC reduction cannot move this operation across K groups");
    }
    auto capacity = loopCapacity(loop);
    if (failed(capacity)) return failure();
    Layout layout = layouts.lookup(loop.getResult(0));
    int64_t logicalK = layouts.lookup(loadA.getResult()).usefulCols;
    int64_t physicalK, arelGroupBytes, arelBytes;
    if (!aimx::checkedRoundUp(logicalK, aimx::kRegSize, physicalK) ||
        !aimx::checkedProduct({layout.rows, physicalK, aimx::kTypeSize,
                              aimx::kNumChannels}, arelGroupBytes) ||
        !aimx::checkedProduct({std::max<int64_t>(*capacity, 1), arelGroupBytes}, arelBytes))
      return loop.emitError("AiMX resident K-group relayout extent overflows");
    View out;
    auto zero = valueView(loop.getInitArgs()[0]);
    Value arel;
    if (failed(zero) || failed(scratch(layout, out)) || failed(copy(*zero, out)) ||
        failed(allocate(arelBytes, aimx::AllocKind::Data, arel))) return failure();
    values[loop.getResult(0)] = out;
    Value lower = mapped(loop.getLowerBound()), upper = mapped(loop.getUpperBound());
    Value stride = mapped(loop.getStep());
    auto asIndex = [&](Value value) -> Value {
      if (value.getType().isIndex()) return value;
      return code.create<arith::IndexCastOp>(loc, code.getIndexType(), value);
    };
    Value count = code.create<arith::CeilDivSIOp>(loc,
        code.create<arith::MaxSIOp>(loc, sub(asIndex(upper), asIndex(lower)), c(code, 0)),
        c(code, *step));
    code.create<aimx::RequireOp>(loc,
        code.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, count, c(code, *capacity)),
        code.getStringAttr("AiMX K-group count exceeds its resident Arel capacity"));
    Value active = code.create<arith::SelectOp>(loc,
        code.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, count, c(code, 0)),
        c(code, 1), c(code, 0));
    auto nonempty = code.create<scf::ForOp>(loc, c(code, 0), active, c(code, 1));
    code.setInsertionPoint(nonempty.getBody()->getTerminator());
    IRMapping outerIndices = indices;
    int64_t outer = repetitions;
    auto mapGroup = [&](scf::ForOp lowered) -> View {
      indices = outerIndices;
      indices.map(loop.getInductionVar(), lowered.getInductionVar());
      for (Operation &op : *loop.getBody())
        if (isIndexOperation(&op)) cloneIndex(op, code);
      Value slot = code.create<arith::DivSIOp>(loc,
          sub(asIndex(lowered.getInductionVar()), asIndex(lower)), c(code, *step));
      return {arel, mul(slot, c(code, arelGroupBytes)), c(code, 0), layout};
    };
    // Complete every transpose before starting a live vACC accumulation.
    auto prepare = code.create<scf::ForOp>(loc, lower, upper, stride);
    prepare->setAttrs(loop->getAttrs());
    prepare->setAttr("aimx.matmul_phase", code.getStringAttr("relayout"));
    code.setInsertionPoint(prepare.getBody()->getTerminator());
    if (llvm::MulOverflow(outer, *capacity, repetitions))
      return loop.emitError("AiMX descriptor loop extent overflows");
    View relayout = mapGroup(prepare);
    auto a = accessView(loadA.getAccess().getDefiningOp<pim::AccessOp>());
    if (failed(a) || failed(append(aimx::buildMatmulRelayout(
            module, name("matmul_relayout"), batch, layout.usefulRows, logicalK),
            {*a, relayout}))) return failure();
    code.setInsertionPointAfter(prepare);
    repetitions = outer;
    auto row = code.create<scf::ForOp>(loc, c(code, 0), c(code, layout.rows), c(code, batch));
    code.setInsertionPoint(row.getBody()->getTerminator());
    if (llvm::MulOverflow(repetitions, layout.rows / batch, repetitions))
      return loop.emitError("AiMX output row extent overflows");
    int64_t width = aimx::kLanes / batch;
    // N=1 matvec needs one MAC/drain microtile even with wider storage rows.
    int64_t columns = matvecFirstNOnly && isa<linalg::MatvecOp>(mac)
                          ? width : layout.cols;
    if (isa<linalg::MatmulOp>(mac) && matmulUsefulNOnly &&
        !aimx::checkedRoundUp(layout.usefulCols, width, columns))
      return loop.emitError("AiMX useful GEMM N extent overflows");
    auto col = code.create<scf::ForOp>(loc, c(code, 0), c(code, columns), c(code, width));
    code.setInsertionPoint(col.getBody()->getTerminator());
    if (llvm::MulOverflow(repetitions, columns / width, repetitions))
      return loop.emitError("AiMX output column extent overflows");
    int64_t outputRepetitions = repetitions;
    auto accumulate = code.create<scf::ForOp>(loc, lower, upper, stride);
    accumulate->setAttrs(loop->getAttrs());
    accumulate->setAttr("aimx.matmul_phase", code.getStringAttr("accumulate"));
    code.setInsertionPoint(accumulate.getBody()->getTerminator());
    if (llvm::MulOverflow(repetitions, *capacity, repetitions))
      return loop.emitError("AiMX MAC K-group extent overflows");
    relayout = mapGroup(accumulate);
    relayout.offset = add(relayout.offset, mul(row.getInductionVar(),
        c(code, physicalK * aimx::kTypeSize * aimx::kNumChannels)));
    auto b = accessView(loadB.getAccess().getDefiningOp<pim::AccessOp>());
    if (failed(b)) return failure();
    b->offset = add(b->offset, mul(col.getInductionVar(), c(code, aimx::kTypeSize)));
    if (failed(append(aimx::buildMatmulAccumulate(
            module, name("matmul_accumulate"), batch, logicalK), {relayout, *b})))
      return failure();
    int64_t stepK = aimx::kRegSize / batch;
    int64_t macK = ((logicalK + stepK - 1) / stepK) * stepK;
    if (macK % aimx::kRegSize != 0) {
      // Only the last group pads the complete MAC stream to a native K block.
      Value totalK = mul(count, c(code, macK));
      Value paddedK = mul(code.create<arith::CeilDivSIOp>(
          loc, totalK, c(code, aimx::kRegSize)), c(code, aimx::kRegSize));
      Value paddingSteps = code.create<arith::DivSIOp>(
          loc, sub(paddedK, totalK), c(code, stepK));
      Value slot = code.create<arith::DivSIOp>(loc,
          sub(asIndex(accumulate.getInductionVar()), asIndex(lower)), c(code, *step));
      Value last = code.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             slot, sub(count, c(code, 1)));
      Value paddingCount = code.create<arith::SelectOp>(loc, last, paddingSteps, c(code, 0));
      auto padding = code.create<scf::ForOp>(loc, c(code, 0), paddingCount, c(code, 1));
      code.setInsertionPoint(padding.getBody()->getTerminator());
      View zeroA = relayout, zeroB = *b;
      zeroA.offset = add(zeroA.offset, c(code, (macK / stepK) * aimx::kDeviceWidth));
      zeroB.offset = add(zeroB.offset, mul(c(code, macK), zeroB.pitch));
      // Each repeated pair reads the same known-zero portion of the last panel.
      if (llvm::MulOverflow(outputRepetitions, batch - 1, repetitions))
        return loop.emitError("AiMX final K padding extent overflows");
      if (failed(append(aimx::buildMatmulAccumulate(
              module, name("matmul_padding"), batch, stepK), {zeroA, zeroB})))
        return failure();
    }
    code.setInsertionPointAfter(accumulate);
    repetitions = outputRepetitions;
    View output = out;
    output.offset = add(out.offset,
        add(mul(row.getInductionVar(), out.pitch),
            mul(col.getInductionVar(), c(code, aimx::kTypeSize))));
    if (failed(append(aimx::buildMatmulDrain(module, name("matmul_drain"), batch),
                      {output}, ValueRange{pmm}))) return failure();
    code.setInsertionPointAfter(nonempty);
    indices = outerIndices;
    repetitions = outer;
    return success();
  }
  LogicalResult lowerBlock(Block &body, bool root) {
    for (Operation &op : body) {
      if (isa<pim::PlaceOp, pim::AccessOp, scf::YieldOp>(op)) continue;
      // 전송은 run()이 stage/unstage로 내린다.
      if (isa<pim::BroadcastOp, pim::ScatterOp, pim::GatherOp>(op)) continue;
      if (auto launch = dyn_cast<pim::LaunchOp>(op)) {
        if (failed(lowerBlock(launch.getBody().front(), false))) return failure();
        continue;
      }
      if (isIndexOperation(&op)) {
        if (!root || !indices.contains(op.getResult(0))) cloneIndex(op, code);
        continue;
      }
      if (isa<arith::ConstantOp>(op) && !isa<RankedTensorType>(op.getResult(0).getType()))
        continue; // Used as data by fill/generic, not host arithmetic.
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (failed(lowerLoop(loop))) return failure();
      } else if (auto load = dyn_cast<pim::LoadOp>(op)) {
        auto access = load.getAccess().getDefiningOp<pim::AccessOp>();
        auto view = accessView(access);
        if (failed(view)) return failure();
        View snapshot;
        if (failed(scratch(view->layout, snapshot)) || failed(copy(*view, snapshot))) return failure();
        values[load.getResult()] = snapshot;
      } else if (isa<tensor::EmptyOp>(op)) {
        continue; // Allocate only if an operation actually reads the undefined tensor.
      } else if (auto fill = dyn_cast<linalg::FillOp>(op)) {
        Attribute scalar;
        if (!matchPattern(fill.getInputs()[0], m_Constant(&scalar)))
          return fill.emitError("AiMX dynamic fill operands are not implemented");
        for (Value result : fill->getResults()) {
          View view;
          if (failed(constant(scalar, layouts.lookup(result), view))) return failure();
          values[result] = view;
        }
      } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
        if (failed(lowerGeneric(generic))) return failure();
      } else if (isa<linalg::MatmulOp, linalg::MatvecOp>(op)) {
        if (failed(lowerMAC(&op))) return failure();
      } else if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(op)) {
        if (failed(lowerBinary(&op, layouts.lookup(op.getResult(0))))) return failure();
      } else if (auto cast = dyn_cast<tensor::CastOp>(op)) {
        auto view = valueView(cast.getSource());
        if (failed(view)) return failure();
        values[cast.getResult()] = *view;
      } else if (auto store = dyn_cast<pim::StoreOp>(op)) {
        auto source = valueView(store.getValue());
        auto destination = accessView(store.getAccess().getDefiningOp<pim::AccessOp>());
        if (failed(source) || failed(destination) || failed(copy(*source, *destination))) return failure();
      } else if (auto combine = dyn_cast<pim::CombineOp>(op)) {
        if (combine.getKind() != "add")
          return combine.emitError("AiMX does not implement this pim.combine kind");
        auto access = combine.getDestination().getDefiningOp<pim::AccessOp>();
        auto source = valueView(combine.getPartial());
        auto destination = accessView(access);
        if (failed(source) || failed(destination)) return failure();
        if (assignedCombines.contains(combine)) {
          if (failed(copy(*source, *destination))) return failure();
        } else if (failed(binary(aimx::Opcode::add, *destination, *source, *destination))) {
          return failure();
        }
      } else return op.emitError("AiMX has no lowering for this Common operation");
    }
    return success();
  }
};

struct ConvertPIMToAiMXPass
    : mlir::triton::impl::ConvertPIMToAiMXBase<ConvertPIMToAiMXPass> {
  using ConvertPIMToAiMXBase::ConvertPIMToAiMXBase;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto profile = pim::targetProfile(module);
    if (!profile || profile.getTarget() != "aimx" ||
        profile.getBanksPerUnit() != aimx::kNumBanks * aimx::kNumChannels ||
        profile.getBankCapacityLimitBytes() <= 0) {
      module.emitError("convert-pim-to-aimx requires an AiMX target profile");
      return signalPassFailure();
    }
    if (batch != 1 && batch != 2 && batch != 4 && batch != 8) {
      module.emitError("AiMX batch must be 1, 2, 4, or 8");
      return signalPassFailure();
    }
    if (matvecRowAlignment < 0 ||
        matvecRowAlignment % (aimx::kLanes / batch) != 0) {
      module.emitError("AiMX matvec row alignment must be zero or a positive multiple of 4096/batch");
      return signalPassFailure();
    }
    if (matmulRowAlignment < 0 ||
        matmulRowAlignment % (aimx::kLanes / batch) != 0) {
      module.emitError("AiMX matmul row alignment must be zero or a positive multiple of 4096/batch");
      return signalPassFailure();
    }
    WalkResult unplanned = module.walk([&](pim::DeviceOp device) {
      device.emitOpError("requires pim-form-host-plan before AiMX lowering");
      return WalkResult::interrupt();
    });
    if (unplanned.wasInterrupted())
      return signalPassFailure();
    SmallVector<pim::PartitionOp> devices;
    module.walk([&](pim::PartitionOp device) { devices.push_back(device); });
    unsigned site = 0;
    for (pim::PartitionOp device : devices)
      if (failed(DeviceLowering(module, device, profile, batch, matvecRowAlignment,
                                matvecFirstNOnly, matmulRowAlignment,
                                matmulUsefulNOnly, persistent, site++).run()))
        return signalPassFailure();
    module->removeAttr("pim.target_profile");
    module.walk([](func::FuncOp function) {
      function->removeAttr("pim.block");
      function->removeAttr("pim.block_group");
      bool changed;
      do {
        changed = false;
        SmallVector<Operation *> dead;
        function.walk([&](Operation *op) {
          if (op->getNumRegions() == 0 && isOpTriviallyDead(op)) dead.push_back(op);
        });
        for (Operation *op : llvm::reverse(dead)) { op->erase(); changed = true; }
      } while (changed);
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createConvertPIMToAiMXPass() {
  return std::make_unique<ConvertPIMToAiMXPass>();
}
