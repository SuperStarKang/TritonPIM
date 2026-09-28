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
  DeviceLowering(ModuleOp module, pim::DeviceOp device,
                 pim::TargetProfileAttr profile, int64_t batch, bool persistent,
                 unsigned site)
      : module(module), device(device), profile(profile), batch(batch),
        persistent(persistent),
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

    // The whole descriptor-construction boundary. Only arith, SCF and
    // build_descriptors are emitted here, never retained linalg/PIM operations.
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
    host.setInsertionPointAfter(construction);
    for (auto &entry : placements)
      if (entry.second.source.getDirection() == "in") stage(host, entry.second);
    host.create<aimx::InvokeOp>(loc, deviceValue, chain);
    for (auto &entry : placements)
      if (entry.second.source.getDirection() == "out") unstage(host, entry.second);
    device.erase();
    return success();
  }

private:
  ModuleOp module;
  pim::DeviceOp device;
  pim::TargetProfileAttr profile;
  int64_t batch;
  bool persistent;
  std::string prefix;
  unsigned symbol = 0;
  OpBuilder setup, code;
  Location loc;
  Value deviceValue, pmm, chain;
  IRMapping indices;
  llvm::DenseMap<Type, Layout> layouts;
  llvm::DenseMap<Value, View> values;
  llvm::MapVector<int64_t, Placement> placements;
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
  Value physicalIndex(Value logical, int64_t useful, int64_t physical) {
    if (useful == physical) return logical;
    Value extent = c(code, useful);
    Value group = code.create<arith::DivSIOp>(loc, logical, extent);
    Value within = code.create<arith::RemSIOp>(loc, logical, extent);
    return add(mul(group, c(code, physical)), within);
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
  LogicalResult addLayout(Operation *op, RankedTensorType type) {
    if (layouts.count(type)) return success();
    if (!type.hasStaticShape() || (type.getRank() != 1 && type.getRank() != 2))
      return op->emitError("AiMX needs static rank-one or rank-two tensor tiles");
    Type element = type.getElementType();
    if (!element.isBF16() && !element.isF16() && !element.isF32())
      return op->emitError("AiMX tensor storage supports BF16/F16/F32 source types");
    int64_t rows = type.getDimSize(0), cols = type.getRank() == 1 ? 1 : type.getDimSize(1);
    int64_t physicalCols;
    if (rows <= 0 || !aimx::checkedRoundUp(cols, aimx::kLanes, physicalCols))
      return op->emitError("AiMX tensor extent cannot be represented");
    // Shared elementwise-compatible row layout. This is physical padding, not
    // a change to the number of Common blocks, groups or execution tiles.
    layouts[type] = {rows, cols, rows, physicalCols};
    return success();
  }
  LogicalResult collectLayouts() {
    WalkResult result = device.walk([&](Operation *op) {
      for (Type type : op->getOperandTypes())
        if (auto tensor = dyn_cast<RankedTensorType>(type))
          if (failed(addLayout(op, tensor))) return WalkResult::interrupt();
      for (Type type : op->getResultTypes())
        if (auto tensor = dyn_cast<RankedTensorType>(type))
          if (failed(addLayout(op, tensor))) return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) return failure();
    WalkResult macLayouts = device.walk([&](Operation *op) {
      if (!isa<linalg::MatmulOp, linalg::MatvecOp>(op)) return WalkResult::advance();
      if (op->getNumResults() != 1 || op->getNumOperands() != 3 ||
          !llvm::all_of(op->getOperandTypes(), [](Type type) {
            return isa<RankedTensorType>(type);
          })) {
        op->emitError("AiMX MAC lowering expects tensor operands and one tensor result");
        return WalkResult::interrupt();
      }
      auto &a = layouts[op->getOperand(0).getType()];
      auto &b = layouts[op->getOperand(1).getType()];
      auto &acc = layouts[op->getOperand(2).getType()];
      auto &out = layouts[op->getResult(0).getType()];
      if (!aimx::checkedRoundUp(a.rows, batch, a.rows) ||
          !aimx::checkedRoundUp(b.rows, aimx::kRegSize, b.rows) ||
          !aimx::checkedRoundUp(acc.rows, batch, acc.rows) ||
          !aimx::checkedRoundUp(out.rows, batch, out.rows)) {
        op->emitError("AiMX MAC physical row extent overflows");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (macLayouts.wasInterrupted()) return failure();
    // Equal logical shapes must compose even when the source dtype differs.
    // A result reused as a MAC input needs the same row padding as that input.
    for (auto &entry : layouts)
      for (auto &other : layouts)
        if (cast<RankedTensorType>(entry.first).getShape() ==
            cast<RankedTensorType>(other.first).getShape()) {
          int64_t rows = std::max(entry.second.rows, other.second.rows);
          entry.second.rows = other.second.rows = rows;
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
        setup.getI64IntegerAttr(kind == aimx::AllocKind::PMM ? aimx::kPMMAlign : aimx::kPIMWidth),
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
    llvm::DenseMap<int64_t, RankedTensorType> types;
    llvm::DenseMap<int64_t, pim::AccessOp> accesses;
    device.walk([&](pim::AccessOp access) { accesses[access.getId()] = access; });
    device.walk([&](pim::LoadOp load) {
      types[load.getAccess().getDefiningOp<pim::AccessOp>().getId()] =
          cast<RankedTensorType>(load.getResult().getType());
    });
    device.walk([&](pim::StoreOp store) {
      types[store.getAccess().getDefiningOp<pim::AccessOp>().getId()] =
          cast<RankedTensorType>(store.getValue().getType());
    });
    device.walk([&](pim::CombineOp combine) {
      types[combine.getDestination().getDefiningOp<pim::AccessOp>().getId()] =
          cast<RankedTensorType>(combine.getPartial().getType());
    });
    for (Operation &op : device.getBody().front()) {
      auto place = dyn_cast<pim::PlaceOp>(op);
      if (!place) continue;
      auto type = types.lookup(place.getAccess());
      if (!type) continue; // Unconsumed placement has no target allocation.
      Layout layout = layouts.lookup(type);
      auto rows = getConstantIntValue(mapped(place.getAllocationShape()[0]));
      auto cols = type.getRank() == 1 ? std::optional<int64_t>(1)
                                     : getConstantIntValue(mapped(place.getAllocationShape()[1]));
      int64_t physicalRows, physicalCols, bytes;
      if (!rows || !cols ||
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
      placement.view.pitch = c(setup, physicalCols * aimx::kTypeSize);
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
  DenseI64ArrayAttr padding(OpBuilder &b, int64_t useful, int64_t physical) {
    return useful == physical ? DenseI64ArrayAttr()
                              : b.getDenseI64ArrayAttr({useful, physical});
  }
  LogicalResult initializeCombines() {
    device.walk([&](pim::CombineOp combine) {
      auto access = combine.getDestination().getDefiningOp<pim::AccessOp>();
      auto found = placements.find(access.getId());
      if (found != placements.end()) found->second.combine = true;
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
      if (failed(constant(setup.getFloatAttr(setup.getBF16Type(), 0.0), extent, zero)) ||
          failed(copy(zero, placement.view))) return failure();
    }
    return success();
  }
  SmallVector<Value> hostValues(OpBuilder &b, ValueRange operands, int64_t extra) {
    auto result = mapped(operands);
    if (result.size() == 1) result.push_back(c(b, extra));
    return result;
  }
  void stage(OpBuilder &b, const Placement &placement) {
    auto place = placement.source;
    const auto &layout = placement.view.layout;
    b.create<aimx::StageOp>(
        loc, mapped(place.getBuffer()), hostValues(b, place.getStart(), 0),
        hostValues(b, placement.sizes, 1), hostValues(b, place.getStrides(), 0),
        ValueRange{c(b, placement.rows), c(b, placement.cols)}, c(b, placement.cols),
        placement.view.buffer, TypeAttr::get(b.getBF16Type()), UnitAttr(),
        padding(b, layout.usefulRows, layout.rows), padding(b, layout.usefulCols, layout.cols));
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
        TypeAttr::get(b.getBF16Type()), UnitAttr(), padding(b, layout.usefulRows, layout.rows),
        padding(b, layout.usefulCols, layout.cols));
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
    Value row = physicalIndex(sub(mapped(access.getOrigin()[0]),
                                  mapped(placement.source.getStart()[0])),
                              view.layout.usefulRows, view.layout.rows);
    Value col = c(code, 0);
    if (access.getOrigin().size() == 2)
      col = physicalIndex(sub(mapped(access.getOrigin()[1]), mapped(placement.source.getStart()[1])),
                           view.layout.usefulCols, view.layout.cols);
    view.offset = add(mul(row, view.pitch), mul(col, c(code, aimx::kTypeSize)));
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
    return append(aimx::buildCopy(module, name("copy"), from.layout.rows, from.layout.cols),
                  {from, to});
  }
  LogicalResult binary(aimx::Opcode opcode, const View &a, const View &b, const View &out) {
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
        DenseI64ArrayAttr(), DenseI64ArrayAttr());
    return success();
  }
  FailureOr<View> valueView(Value value) {
    auto found = values.find(value);
    if (found != values.end()) return found->second;
    if (value.getDefiningOp<tensor::EmptyOp>()) {
      View view;
      if (failed(scratch(layouts.lookup(value.getType()), view))) return failure();
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
    Layout layout = layouts.lookup(generic->getResult(0).getType());
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
  LogicalResult lowerMAC(Operation *op) {
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
    auto a = valueView(op->getOperand(0)), b = valueView(op->getOperand(1));
    auto acc = valueView(op->getOperand(2));
    if (failed(a) || failed(b) || failed(acc)) return failure();
    Layout layout = layouts.lookup(op->getResult(0).getType());
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
    if (failed(append(aimx::buildMatmul(module, name(vector ? "matvec" : "matmul"),
                                       batch, layout.usefulRows, a->layout.usefulCols,
                                       layout.usefulCols), {*a, *b, partial}, ValueRange{arel, pmm})) ||
        failed(binary(aimx::Opcode::add, *acc, partial, out))) return failure();
    values[op->getResult(0)] = out;
    return success();
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
      if (failed(initial) || failed(scratch(layouts.lookup(argument.getType()), view)) ||
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
  LogicalResult lowerBlock(Block &body, bool root) {
    for (Operation &op : body) {
      if (isa<pim::PlaceOp, pim::AccessOp, scf::YieldOp>(op)) continue;
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
          if (failed(constant(scalar, layouts.lookup(result.getType()), view))) return failure();
          values[result] = view;
        }
      } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
        if (failed(lowerGeneric(generic))) return failure();
      } else if (isa<linalg::MatmulOp, linalg::MatvecOp>(op)) {
        if (failed(lowerMAC(&op))) return failure();
      } else if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(op)) {
        if (failed(lowerBinary(&op, layouts.lookup(op.getResult(0).getType())))) return failure();
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
        if (failed(source) || failed(destination) ||
            failed(binary(aimx::Opcode::add, *destination, *source, *destination))) return failure();
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
    SmallVector<pim::DeviceOp> devices;
    module.walk([&](pim::DeviceOp device) { devices.push_back(device); });
    unsigned site = 0;
    for (pim::DeviceOp device : devices)
      if (failed(DeviceLowering(module, device, profile, batch, persistent,
                                site++).run()))
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
