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
  auto plan = (*this)->getAttrOfType<DevicePlanAttr>("pim.device_plan");
  if (!plan)
    return success();
  if (plan.getAxes() != mapping.getAxes())
    return emitOpError("device plan axes must equal device mapping axes");
  if (plan.getActiveGrid() != mapping.getPartitions())
    return emitOpError(
        "device plan active_grid must equal device mapping partitions");
  if (plan.getExecutionTile().size() != 3)
    return emitOpError("device plan execution_tile must be [m, n, k]");
  if (plan.getAvailableDevices() < 1)
    return emitOpError("device plan available_devices must be positive");
  if (plan.getSchedulingOrder().size() != n)
    return emitOpError(
        "device plan scheduling_order must contain every axis once");
  for (Attribute axis : mapping.getAxes()) {
    if (llvm::count(plan.getSchedulingOrder(), axis) != 1)
      return emitOpError(
          "device plan scheduling_order must be a permutation of axes");
  }
  int64_t active = 1;
  for (int64_t p : plan.getActiveGrid().asArrayRef()) {
    if (p < 1)
      return emitOpError("device plan active_grid entries must be positive");
    active *= p;
  }
  if (active > plan.getAvailableDevices())
    return emitOpError("device plan uses more devices than are available");
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

LogicalResult ReductionPlanAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr spatialPartials,
    StringAttr temporalAccumulation, StringAttr finalAccumulation) {
  StringRef spatial = spatialPartials.getValue();
  StringRef temporal = temporalAccumulation.getValue();
  StringRef finalKind = finalAccumulation.getValue();
  if (spatial != "none" && spatial != "separate_output_planes")
    return emitError() << "spatial_partials must be none or "
                          "separate_output_planes";
  if (temporal != "none" && temporal != "host")
    return emitError() << "temporal_accumulation must be none or host";
  if (finalKind != "none" && finalKind != "external")
    return emitError() << "final_accumulation must be none or external";
  if ((spatial == "none") != (finalKind == "none"))
    return emitError() << "separate spatial partials require external final "
                          "accumulation";
  return success();
}
