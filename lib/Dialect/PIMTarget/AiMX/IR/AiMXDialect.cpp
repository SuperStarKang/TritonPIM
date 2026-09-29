#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"

// 생성된 op builder / type-attr printer-parser가 사용하는 헤더들.
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
// bit enum 의 stringify 는 llvm::join 으로 flag 이름을 잇는다.
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <limits>

using namespace mlir;
using namespace mlir::aimx;

// Dialect 생성 시점에 attribute / type / op을 등록한다.
void AiMXDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd enum / attribute / type / op / dialect 정의
//===----------------------------------------------------------------------===//

#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXTypes.cpp.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXOps.cpp.inc"

#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// 공통 helper
//===----------------------------------------------------------------------===//

namespace {

// PIM DRAM 안의 배치는 pitch 로 규정된 2차원 배열이다. descriptor 가 행 간격을
// stride 필드 하나로 읽으므로 그보다 높은 rank 를 표현할 방법이 없다.
constexpr unsigned kPitchedRank = 2;

bool isBatch(int64_t b) { return b == 1 || b == 2 || b == 4 || b == 8; }

uint32_t flags(Opcode op) { return static_cast<uint32_t>(op); }

} // namespace

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

LogicalResult ForeachDeviceOp::verify() {
  if (Attribute persistent = (*this)->getAttr("aimx.persistent"))
    if (!isa<UnitAttr>(persistent))
      return emitOpError("aimx.persistent must be a unit attribute");
  ArrayRef<int64_t> counts = getCounts();
  if (counts.size() != getAxes().size())
    return emitOpError("counts must have one entry per axis");
  int64_t devices = 1;
  for (int64_t count : counts) {
    if (count <= 0 || devices > std::numeric_limits<int64_t>::max() / count)
      return emitOpError("counts must be positive and their product must fit i64");
    devices *= count;
  }
  auto acquire = getSet().getDefiningOp<AcquireOp>();
  if (!acquire || acquire.getNumDevices() != devices)
    return emitOpError("counts product must equal the acquired device count");
  Block &block = getBody().front();
  // FPGA 핸들 하나 다음에 축별 좌표.
  if (block.getNumArguments() != 1 + getAxes().size())
    return emitOpError("body must take one !aimx.device then one coordinate "
                       "per axis");
  if (!isa<DeviceType>(block.getArgument(0).getType()))
    return emitOpError("first body argument must be !aimx.device");
  for (unsigned i = 1, e = block.getNumArguments(); i < e; i++)
    if (!isa<IndexType>(block.getArgument(i).getType()))
      return emitOpError("device coordinates must be index");
  return success();
}

LogicalResult AllocOp::verify() {
  if (getBytes() <= 0)
    return emitOpError("bytes must be positive");
  int64_t align = getAlignment();
  if (align <= 0 || (align & (align - 1)) != 0)
    return emitOpError("alignment must be a power of two");
  // PMM 은 크기와 정렬이 하드웨어가 정한 값 하나뿐이다. vACC 가 pmm + 0x10000 에 있고
  // pim_private_malloc 이 256 KB 정렬을 요구한다.
  if (getKind() == AllocKind::PMM) {
    if (getBytes() != kPMMSize)
      return emitOpError("a pmm allocation must be exactly ")
             << kPMMSize << " byte";
    if (align != kPMMAlign)
      return emitOpError("a pmm allocation must be aligned to ")
             << kPMMAlign << " byte";
  }
  // chain 이 자기 물리 주소를 record 에 싣는다. group 하나가 record NUM_CHANNELS 개다.
  if (getKind() == AllocKind::Descriptor && getBytes() % kGroupBytes != 0)
    return emitOpError("a descriptor allocation must be a multiple of ")
           << kGroupBytes << " byte (one descriptor group)";
  return success();
}

// stage / unstage 가 함께 쓰는 검사.
static LogicalResult verifyPitched(Operation *op, size_t origin, size_t sizes,
                                   size_t strides, Type elem) {
  if (origin != sizes || origin != strides)
    return op->emitOpError("origin, sizes, and strides must have the same rank");
  if (origin < kPitchedRank)
    return op->emitOpError(
        "host regions need at least row and column axes; leading axes select "
        "an independent 2-D PIM tile");
  if (!elem.isIntOrFloat())
    return op->emitOpError("element_type must be a scalar type");
  return success();
}

// Optional periodic physical padding; an identity mapping is also valid.
static LogicalResult verifyGroup(Operation *op, DenseI64ArrayAttr g,
                                 StringRef name) {
  if (!g)
    return success();
  ArrayRef<int64_t> v = g.asArrayRef();
  if (v.size() != 2 || v[0] <= 0 || v[0] > v[1])
    return op->emitOpError(name)
           << " must be [useful, phys] with 0 < useful <= phys";
  return success();
}

LogicalResult StageOp::verify() {
  if (failed(verifyPitched(*this, getOrigin().size(), getSizes().size(),
                           getStrides().size(), getElementType())))
    return failure();
  if (getPadTo().size() != kPitchedRank)
    return emitOpError("pad_to describes the 2-D PIM layout and must have rank ")
           << kPitchedRank;
  if (failed(verifyGroup(*this, getGroupRowsAttr(), "group_rows")) ||
      failed(verifyGroup(*this, getGroupColsAttr(), "group_cols")))
    return failure();
  if (auto shared = getSharedOverAttr()) {
    auto foreach = dyn_cast<ForeachDeviceOp>((*this)->getParentOp());
    if (!foreach)
      return emitOpError("shared_over needs a directly enclosing aimx.foreach_device");
    for (Attribute axis : shared)
      if (!llvm::is_contained(foreach.getAxes(), axis))
        return emitOpError("shared_over names an axis outside aimx.foreach_device");
  }
  return success();
}

LogicalResult UnstageOp::verify() {
  if (failed(verifyPitched(*this, getOrigin().size(), getSizes().size(),
                           getStrides().size(), getElementType())))
    return failure();
  if (failed(verifyGroup(*this, getGroupRowsAttr(), "group_rows")))
    return failure();
  return verifyGroup(*this, getGroupColsAttr(), "group_cols");
}

LogicalResult DescriptorProgramOp::verify() {
  if (!isBatch(getBatch()))
    return emitOpError("batch must be 1, 2, 4, or 8 — the descriptor status "
                       "word has no encoding for other values");
  return success();
}

LogicalResult DescOp::verify() {
  if (getSrc() && !getSrcOffset())
    return emitOpError("src needs an offset");
  if (!getSrc() && getSrcOffset())
    return emitOpError("a src offset without a src buffer");
  if (getDst() && !getDstOffset())
    return emitOpError("dst needs an offset");
  if (!getDst() && getDstOffset())
    return emitOpError("a dst offset without a dst buffer");
  if (getSrcPitch() && !getSrc())
    return emitOpError("a src pitch without a src buffer");
  if (getDstPitch() && !getDst())
    return emitOpError("a dst pitch without a dst buffer");
  if (!getSrc() && !getDst())
    return emitOpError("a descriptor must name at least one address");

  auto program = (*this)->getParentOfType<DescriptorProgramOp>();
  if (!program)
    return emitOpError("must be inside an aimx.descriptor_program — batch and "
                       "the length rules come from there");

  int64_t len = getLen();
  if (len <= 0 || len % kBurstSize != 0)
    return emitOpError("len ")
           << len << " must be a multiple of the " << kBurstSize
           << " byte DRAM burst";

  bool hasPitch = getSrcPitch() || getDstPitch();
  bool strideFlag = (flags(getOpcode()) & flags(Opcode::stride)) != 0;
  if (strideFlag != hasPitch)
    return emitOpError(strideFlag
                           ? "the stride flag is set but no pitch is given"
                           : "a pitch is given but the stride flag is not set");

  // Operand/length encoding is checked here, not a whitelist of entire
  // algorithms. Opcode selection belongs to the target lowering rule.
  if (getOpcode() == Opcode::copy && (!getSrc() || !getDst()))
    return emitOpError("copy requires source and destination addresses");
  return success();
}
