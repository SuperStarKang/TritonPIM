#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.h"

// 생성된 op builder / type-attr printer-parser가 사용하는 헤더들.
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

using namespace mlir;
using namespace mlir::upmem;

// UPMEM DMA 제약. mram_read/mram_write 한 번이 지켜야 하는 값이다.
// 어기면 런타임에 조용히 깨지므로 verifier에서 막는다.
static constexpr int64_t kDMAAlignBytes = 8;
static constexpr int64_t kDMAMaxBytes = 2048;

// Dialect 생성 시점에 attribute / type / op을 등록한다.
void UPMEMDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd enum / attribute / type / op / dialect 정의
//===----------------------------------------------------------------------===//

#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMTypes.cpp.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMOps.cpp.inc"

#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// 공통 helper
//===----------------------------------------------------------------------===//

// memref가 주어진 UPMEM 메모리에 있는가.
static bool inSpace(MemRefType type, MemorySpace space) {
  auto attr = dyn_cast_or_null<MemorySpaceAttr>(type.getMemorySpace());
  return attr && attr.getValue() == space;
}

// DMA 한 번이 옮기는 바이트. shape이 동적이거나 원소가 스칼라가 아니면 실패.
static bool transferBytes(MemRefType type, int64_t &bytes) {
  if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return false;
  int64_t bits =
      type.getNumElements() * type.getElementType().getIntOrFloatBitWidth();
  bytes = (bits + 7) / 8;
  return true;
}

// 정렬과 최대 크기는 하드웨어가 정한다. 큰 타일은 여러 번으로 쪼개 표현해야 한다.
static LogicalResult verifyDMASize(Operation *op, MemRefType type,
                                   StringRef what) {
  int64_t bytes;
  if (!transferBytes(type, bytes))
    return op->emitOpError(what)
           << " must have a static shape and scalar elements";
  if (bytes % kDMAAlignBytes != 0)
    return op->emitOpError("transfer size ")
           << bytes << " must be a multiple of " << kDMAAlignBytes;
  if (bytes > kDMAMaxBytes)
    return op->emitOpError("transfer size ")
           << bytes << " exceeds the " << kDMAMaxBytes << " byte DMA limit";
  return success();
}

// foreach 계열 region의 block argument 개수와 타입을 확인한다.
// leading = 축 좌표 앞에 오는 인자 개수(foreach_unit은 DPU 핸들, foreach_group은 버퍼).
static LogicalResult verifyForeachBody(Operation *op, Region &body,
                                       ArrayAttr axes, ArrayRef<int64_t> counts,
                                       unsigned leading) {
  if (counts.size() != axes.size())
    return op->emitOpError("counts must have one entry per axis");
  Block &block = body.front();
  if (block.getNumArguments() != leading + axes.size())
    return op->emitOpError("body must take ")
           << leading << " leading argument(s) then one coordinate per axis";
  for (unsigned i = leading, e = block.getNumArguments(); i < e; i++)
    if (!isa<IndexType>(block.getArgument(i).getType()))
      return op->emitOpError("unit coordinates must be index");
  return success();
}

// origin/sizes/strides가 같은 rank를 기술하는가.
static LogicalResult verifyRegionShape(Operation *op, size_t origin,
                                       size_t sizes, size_t strides) {
  if (origin != sizes || origin != strides)
    return op->emitOpError("origin, sizes, and strides must have the same rank");
  return success();
}

// fixed slot operand가 정적 memref shape에 그대로 materialize되었는지 확인한다.
static LogicalResult verifyFixedShape(Operation *op, ValueRange extents,
                                      MemRefType type, unsigned leading) {
  if (type.getRank() != static_cast<int64_t>(leading + extents.size()))
    return op->emitOpError("result rank does not match its fixed-slot rank");
  if (!type.hasStaticShape())
    return op->emitOpError("fixed-slot result must have a static shape");
  for (auto [d, extent] : llvm::enumerate(extents)) {
    std::optional<int64_t> value = getConstantIntValue(extent);
    if (!value)
      return op->emitOpError("fixed-slot extent must be a constant");
    if (type.getShape()[leading + d] != *value)
      return op->emitOpError("fixed-slot extent does not match result shape");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

LogicalResult MRAMOp::verify() {
  auto type = dyn_cast<MemRefType>(getType());
  if (!type || !inSpace(type, MemorySpace::MRAM))
    return emitOpError("type must be a memref in the mram space");
  return success();
}

LogicalResult ForeachUnitOp::verify() {
  // DPU 핸들 하나 다음에 축별 좌표.
  if (failed(verifyForeachBody(*this, getBody(), getAxes(), getCounts(), 1)))
    return failure();
  if (!isa<UnitType>(getBody().front().getArgument(0).getType()))
    return emitOpError("first body argument must be !upmem.unit");
  return success();
}

LogicalResult ForeachGroupOp::verify() {
  // 버퍼 하나 다음에 축별 좌표.
  if (failed(verifyForeachBody(*this, getBody(), getAxes(), getCounts(), 1)))
    return failure();
  if (!isa<MemRefType>(getBody().front().getArgument(0).getType()))
    return emitOpError("first body argument must be the device buffer (memref)");
  return success();
}

LogicalResult StageOp::verify() {
  if (failed(verifyRegionShape(*this, getOrigin().size(), getSizes().size(),
                               getStrides().size())))
    return failure();
  // pad_to는 결과 버퍼의 shape이라 같은 rank여야 한다.
  if (getPadTo().size() != getOrigin().size())
    return emitOpError("pad_to must have the same rank as origin");
  if (getTile().size() != getOrigin().size())
    return emitOpError("tile must have the same rank as origin");
  if (auto role = getOperandRoleAttr())
    if (role.getValue() != "lhs" && role.getValue() != "rhs")
      return emitOpError("operand_role must be lhs or rhs");
  auto result = cast<MemRefType>(getResult().getType());
  return verifyFixedShape(*this, getPadTo(), result, /*leading=*/0);
}

LogicalResult GatherOp::verify() {
  auto result = cast<MemRefType>(getResult().getType());
  if (failed(verifyFixedShape(*this, getCapacity(), result, /*leading=*/1)))
    return failure();
  if (auto acquire = getSet().getDefiningOp<AcquireOp>())
    if (result.getShape()[0] != acquire.getNumUnits())
      return emitOpError(
          "leading result extent must equal the acquired DPU count");
  return success();
}

LogicalResult UnstageOp::verify() {
  if (failed(verifyRegionShape(*this, getOrigin().size(), getSizes().size(),
                               getStrides().size())))
    return failure();
  // capacity는 버퍼의 shape이라 같은 rank여야 한다.
  if (getCapacity().size() != getOrigin().size())
    return emitOpError("capacity must have the same rank as origin");
  if (getTile().size() != getOrigin().size())
    return emitOpError("tile must have the same rank as origin");
  return success();
}

LogicalResult MRAMReadOp::verify() {
  auto dst = cast<MemRefType>(getDst().getType());
  if (!inSpace(dst, MemorySpace::WRAM))
    return emitOpError("destination must be a memref in the wram space");
  return verifyDMASize(*this, dst, "destination");
}

LogicalResult MRAMWriteOp::verify() {
  auto src = cast<MemRefType>(getSrc().getType());
  if (!inSpace(src, MemorySpace::WRAM))
    return emitOpError("source must be a memref in the wram space");
  return verifyDMASize(*this, src, "source");
}
