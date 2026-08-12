#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"

// 생성된 op builder / type-attr printer-parser가 사용하는 헤더들.
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
// bit enum 의 stringify 는 llvm::join 으로 flag 이름을 잇는다.
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

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
  ArrayRef<int64_t> counts = getCounts();
  if (counts.size() != getAxes().size())
    return emitOpError("counts must have one entry per axis");
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
  if (getBytes() == 0)
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
  if (origin != kPitchedRank)
    return op->emitOpError("the PIM DRAM layout is a pitched 2-D array, so "
                           "origin/sizes/strides must have rank ")
           << kPitchedRank;
  if (!elem.isIntOrFloat())
    return op->emitOpError("element_type must be a scalar type");
  return success();
}

LogicalResult StageOp::verify() {
  if (failed(verifyPitched(*this, getOrigin().size(), getSizes().size(),
                           getStrides().size(), getElementType())))
    return failure();
  if (getPadTo().size() != getOrigin().size())
    return emitOpError("pad_to must have the same rank as origin");
  return success();
}

LogicalResult UnstageOp::verify() {
  return verifyPitched(*this, getOrigin().size(), getSizes().size(),
                       getStrides().size(), getElementType());
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
  int64_t batch = program.getBatch();

  int64_t len = getLen();
  if (len % kBurstSize != 0)
    return emitOpError("len ")
           << len << " must be a multiple of the " << kBurstSize
           << " byte DRAM burst";

  bool hasPitch = getSrcPitch() || getDstPitch();
  bool strideFlag = (flags(getOpcode()) & flags(Opcode::stride)) != 0;
  if (strideFlag != hasPitch)
    return emitOpError(strideFlag
                           ? "the stride flag is set but no pitch is given"
                           : "a pitch is given but the stride flag is not set");

  // 확인된 descriptor 형태만 허용한다. 여기서 벗어나면 하드웨어에서 조용히 틀린다.
  // 각 행은 (status word, len, src pitch 필요, dst pitch 필요) 다.
  struct Form {
    uint32_t status;
    int64_t len;
    bool srcPitch;
    bool dstPitch;
    const char *role;
  };
  const uint32_t A = flags(Opcode::a_mask), B = flags(Opcode::b_mask),
                 C = flags(Opcode::c_mask), MM = flags(Opcode::matmul),
                 BC = flags(Opcode::bc), TP = flags(Opcode::tp),
                 PM = flags(Opcode::pim_mode), ST = flags(Opcode::stride);
  const Form forms[] = {
      {PM | ST, kPIMWidth, true, false, "A gather"},
      {BC | TP, kDeviceWidth * batch, false, false, "Arel write"},
      {A | MM | PM, kDeviceWidth, false, false, "A operand"},
      {B | MM | PM | ST, kDeviceWidth / batch, true, false, "B operand"},
      {C | MM | ST, kDeviceWidth / batch, true, true, "C drain"},
      {BC | TP | ST, kDeviceWidth * batch, true, true, "merged relayout"},
  };
  for (const Form &f : forms) {
    if (f.status != flags(getOpcode()))
      continue;
    if (f.len != len)
      return emitOpError("a ")
             << f.role << " descriptor must have len " << f.len << ", not "
             << len;
    if (f.srcPitch != static_cast<bool>(getSrcPitch()))
      return emitOpError("a ") << f.role << " descriptor "
                               << (f.srcPitch ? "needs" : "must not have")
                               << " a src pitch";
    if (f.dstPitch != static_cast<bool>(getDstPitch()))
      return emitOpError("a ") << f.role << " descriptor "
                               << (f.dstPitch ? "needs" : "must not have")
                               << " a dst pitch";
    return success();
  }
  return emitOpError("this flag combination is not one of the descriptor forms "
                     "confirmed on hardware");
}
