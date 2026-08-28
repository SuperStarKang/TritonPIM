#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

// 생성된 op builder / attribute·type printer-parser가 사용하는 헤더들.
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
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
    int64_t numberOfPimDevices, int64_t numberOfBanksPerPimDevice,
    int64_t numberOfPimCoresPerBank,
    int64_t numberOfParallelLanesPerPimCore,
    StringAttr parallelExecutionKind, IntegerAttr usableBankCapacityBytes,
    IntegerAttr usableLocalCapacityBytes) {
  if (target.getValue().empty())
    return emitError() << "target must not be empty";
  if (numberOfPimDevices < 1 || numberOfBanksPerPimDevice < 1 ||
      numberOfPimCoresPerBank < 1 || numberOfParallelLanesPerPimCore < 1)
    return emitError() << "device, bank, core, and lane topology values must be positive";
  StringRef kind = parallelExecutionKind.getValue();
  if (kind != "multithread" && kind != "simd")
    return emitError() << "parallel_execution_kind must be multithread or simd";
  if (usableBankCapacityBytes && usableBankCapacityBytes.getInt() < 0)
    return emitError() << "usable_bank_capacity_bytes must not be negative";
  if (usableLocalCapacityBytes && usableLocalCapacityBytes.getInt() < 0)
    return emitError() << "usable_local_capacity_bytes must not be negative";
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
  DeviceMappingAttr mapping = getMapping();
  size_t n = mapping.getAxes().size();
  if (mapping.getPartitions().size() != n)
    return emitOpError("mapping must have one partition count per axis");
  if (getAxisExtents().size() != n)
    return emitOpError("axis_extents must have one entry per axis");
  if (getBody().front().getNumArguments() != n)
    return emitOpError("body must take one partition coordinate per axis");
  for (int64_t partition : mapping.getPartitions().asArrayRef())
    if (partition < 1)
      return emitOpError("mapping partition counts must be positive");
  return success();
}

LogicalResult BlockAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                DenseI64ArrayAttr shape) {
  if (shape.size() != 3)
    return emitError() << "shape must be [m, n, k]";
  if (llvm::any_of(shape.asArrayRef(), [](int64_t value) { return value < 0; }))
    return emitError() << "shape entries must not be negative";
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
  if (llvm::any_of(groups.asArrayRef(), [](int64_t value) { return value < 0; }))
    return emitError() << "group counts must not be negative";
  return success();
}

LogicalResult AxisAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               StringAttr name) {
  StringRef value = name.getValue();
  if (value != "m" && value != "n" && value != "k")
    return emitError() << "axis must be m, n, or k";
  return success();
}

LogicalResult IntraDevicePlanAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseI64ArrayAttr bankPartitions, DenseI64ArrayAttr corePartitions,
    int64_t lanes, StringAttr laneAxis) {
  if (bankPartitions.size() != 3 || corePartitions.size() != 3)
    return emitError() << "bank_partitions and core_partitions must be [m, n, k]";
  for (int64_t p : bankPartitions.asArrayRef())
    if (p < 1)
      return emitError() << "bank_partitions entries must be positive";
  for (int64_t p : corePartitions.asArrayRef())
    if (p < 1)
      return emitError() << "core_partitions entries must be positive";
  if (lanes < 1)
    return emitError() << "lanes must be positive";
  StringRef axis = laneAxis.getValue();
  if (axis != "m" && axis != "n" && axis != "k")
    return emitError() << "lane_axis must be m, n, or k";
  return success();
}

LogicalResult PlaceOp::verify() {
  size_t rank = getStart().size();
  if (getSize().size() != rank || getCapacity().size() != rank ||
      getStrides().size() != rank)
    return emitOpError(
        "start, size, capacity, and strides must have the same rank");
  if (getMapping().getAxisSources().size() != rank)
    return emitOpError("mapping must have one axis source per buffer axis");
  StringRef dir = getDirection();
  if (dir != "in" && dir != "out")
    return emitOpError("direction must be \"in\" or \"out\"");
  return success();
}
