#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

// 생성된 op builder / attribute·type printer-parser가 사용하는 헤더들.
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pim;

// Dialect 생성 시점에 attribute와 type과 op를 등록한다.
void PIMDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton-shared/Dialect/PIM/IR/PIMAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton-shared/Dialect/PIM/IR/PIMTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/PIM/IR/PIMOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd attribute / type / op / dialect 정의
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/PIM/IR/PIMAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/PIM/IR/PIMTypes.cpp.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/PIM/IR/PIMOps.cpp.inc"

#include "triton-shared/Dialect/PIM/IR/PIMDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

LogicalResult TargetProfileAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr target,
    int64_t numUnits, int64_t banksPerUnit, int64_t coresPerBank,
    int64_t lanesPerCore, int64_t bankCapacityLimitBytes,
    int64_t localCapacityLimitBytes) {
  if (target.getValue().empty())
    return emitError() << "target must not be empty";
  if (numUnits < 1 || banksPerUnit < 1 || coresPerBank < 1 ||
      lanesPerCore < 1)
    return emitError()
           << "unit, bank, core, and lane topology values must be positive";
  if (bankCapacityLimitBytes < 0 || localCapacityLimitBytes < 0)
    return emitError() << "capacity limits must not be negative";
  return success();
}

LogicalResult DeviceMappingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseI64ArrayAttr partitions) {
  if (partitions.size() != 3)
    return emitError() << "partitions must be [p_m, p_n, p_k]";
  if (llvm::any_of(partitions.asArrayRef(),
                   [](int64_t value) { return value < 1; }))
    return emitError() << "partition counts must be positive";
  return success();
}

LogicalResult AccessOp::verify() {
  int64_t rank = getSizes().size();
  if (static_cast<int64_t>(getStrides().size()) != rank)
    return emitOpError("strides count must equal sizes rank");
  if (!getValid().empty() && static_cast<int64_t>(getValid().size()) != rank)
    return emitOpError("valid count must equal sizes rank");
  // 분해가 안 되면 아예 비운다. 부분 origin은 금지한다.
  if (!getOrigin().empty() && static_cast<int64_t>(getOrigin().size()) != rank)
    return emitOpError("origin count must equal sizes rank");
  // pad(채움값)는 valid(mask)와 함께만 의미가 있다.
  if (getPad() && getValid().empty())
    return emitOpError("pad requires valid");
  return success();
}

LogicalResult DeviceOp::verify() {
  if (getBody().front().getNumArguments() != 3)
    return emitOpError("body must take the [m, n, k] unit coordinates");
  return success();
}

LogicalResult PartitionOp::verify() {
  if (getBody().empty() || getBody().front().getNumArguments() != 3)
    return emitOpError("body must take the [m, n, k] logical coordinates");
  for (BlockArgument coordinate : getBody().front().getArguments())
    if (!coordinate.getType().isIndex())
      return emitOpError("logical coordinates must have index type");

  llvm::DenseMap<int64_t, PlaceOp> placements;
  llvm::DenseSet<int64_t> sent, gathered;
  bool transfersStarted = false;
  bool launched = false;
  for (Operation &op : getBody().front()) {
    if (auto place = dyn_cast<PlaceOp>(&op)) {
      if (transfersStarted)
        return place.emitOpError("placements must precede the transfer plan");
      if (!placements.try_emplace(place.getAccess(), place).second)
        return place.emitOpError("duplicate placement access id");
      continue;
    }
    if (isa<BroadcastOp, ScatterOp>(&op)) {
      transfersStarted = true;
      if (launched)
        return op.emitOpError("input transfer must precede pim.launch");
      int64_t access = isa<BroadcastOp>(&op)
                           ? cast<BroadcastOp>(&op).getAccess()
                           : cast<ScatterOp>(&op).getAccess();
      auto found = placements.find(access);
      if (found == placements.end() || found->second.getDirection() != "in")
        return op.emitOpError("transfer requires an input placement with id ")
               << access;
      if (!sent.insert(access).second)
        return op.emitOpError("input placement is transferred more than once");
      continue;
    }
    if (isa<LaunchOp>(&op)) {
      if (launched)
        return op.emitOpError("partition must contain exactly one pim.launch");
      transfersStarted = true;
      launched = true;
      continue;
    }
    if (auto gather = dyn_cast<GatherOp>(&op)) {
      if (!launched)
        return gather.emitOpError("gather must follow pim.launch");
      auto found = placements.find(gather.getAccess());
      if (found == placements.end() || found->second.getDirection() != "out")
        return gather.emitOpError("gather requires an output placement with id ")
               << gather.getAccess();
      if (!gathered.insert(gather.getAccess()).second)
        return gather.emitOpError("output placement is gathered more than once");
      continue;
    }
    if (transfersStarted || op.getNumRegions() != 0 ||
        !isMemoryEffectFree(&op) ||
        !llvm::all_of(op.getResultTypes(), [](Type type) {
          return type.isIntOrIndex() || isa<FloatType>(type);
        }))
      return op.emitOpError("host plan permits only scalar setup, placements, "
                            "input transfers, one launch, and gathers");
  }
  if (!launched)
    return emitOpError("partition must contain exactly one pim.launch");
  for (auto &entry : placements) {
    bool input = entry.second.getDirection() == "in";
    if (!(input ? sent : gathered).contains(entry.first))
      return entry.second.emitOpError(input ? "input placement has no transfer"
                                           : "output placement has no gather");
  }
  return success();
}

static LogicalResult verifyTransfer(Operation *op, int64_t access,
                                    StringRef direction, bool broadcast) {
  auto partition = dyn_cast_or_null<PartitionOp>(op->getParentOp());
  if (!partition)
    return op->emitOpError("must be directly inside pim.partition");
  for (PlaceOp place : partition.getBody().front().getOps<PlaceOp>()) {
    if (place.getAccess() != access)
      continue;
    if (place.getDirection() != direction)
      return op->emitOpError("placement direction must be ") << direction;
    if (broadcast &&
        (!place.getSharedOverAttr() || place.getSharedOverAttr().empty()))
      return op->emitOpError("broadcast requires nonempty shared_over");
    return success();
  }
  return op->emitOpError("no placement with access id ") << access;
}

LogicalResult BroadcastOp::verify() {
  return verifyTransfer(*this, getAccess(), "in", true);
}

LogicalResult ScatterOp::verify() {
  return verifyTransfer(*this, getAccess(), "in", false);
}

LogicalResult GatherOp::verify() {
  return verifyTransfer(*this, getAccess(), "out", false);
}

LogicalResult LaunchOp::verify() {
  if (!isa_and_nonnull<PartitionOp>((*this)->getParentOp()))
    return emitOpError("must be directly inside pim.partition");
  if (getBody().empty() || getBody().front().getNumArguments() != 0)
    return emitOpError("compute body must have no block arguments");
  return success();
}

LogicalResult BlockAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                DenseI64ArrayAttr shape) {
  if (shape.size() != 3)
    return emitError() << "shape must be [m, n, k]";
  if (llvm::any_of(shape.asArrayRef(), [](int64_t value) { return value < 1; }))
    return emitError() << "shape entries must be positive";
  return success();
}

LogicalResult BlockGroupAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseI64ArrayAttr blocks) {
  if (blocks.size() != 3)
    return emitError() << "blocks must be [m, n, k]";
  if (llvm::any_of(blocks.asArrayRef(), [](int64_t value) { return value < 1; }))
    return emitError() << "block counts must be positive";
  return success();
}

LogicalResult ExecutionTileAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseI64ArrayAttr groups) {
  if (groups.size() != 3)
    return emitError() << "groups must be [m, n, k]";
  if (llvm::any_of(groups.asArrayRef(), [](int64_t value) { return value < 1; }))
    return emitError() << "group counts must be positive";
  return success();
}

LogicalResult AxisAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               StringAttr name) {
  StringRef value = name.getValue();
  if (value != "m" && value != "n" && value != "k")
    return emitError() << "axis must be m, n, or k";
  return success();
}

LogicalResult PlaceOp::verify() {
  size_t rank = getStart().size();
  if (getSize().size() != rank || getAllocationShape().size() != rank ||
      getStrides().size() != rank)
    return emitOpError(
        "start, size, allocation_shape, and strides must have the same rank");
  StringRef dir = getDirection();
  if (dir != "in" && dir != "out")
    return emitOpError("direction must be \"in\" or \"out\"");
  return success();
}

LogicalResult CombineOp::verify() {
  if (getKind() != "add")
    return emitOpError(
        "currently supports only the normalized add combiner");

  auto destination = getDestination().getDefiningOp<AccessOp>();
  if (!destination)
    return emitOpError("destination must be defined by pim.access");

  auto partialType = cast<RankedTensorType>(getPartial().getType());
  ArrayRef<int64_t> destinationShape = destination.getSizes();
  if (partialType.getRank() != static_cast<int64_t>(destinationShape.size()))
    return emitOpError(
        "partial rank must equal the final destination access rank");
  for (auto [partialDim, destinationDim] :
       llvm::zip_equal(partialType.getShape(), destinationShape))
    if (!ShapedType::isDynamic(partialDim) && partialDim != destinationDim)
      return emitOpError(
          "partial shape must equal the final destination access shape");
  return success();
}
