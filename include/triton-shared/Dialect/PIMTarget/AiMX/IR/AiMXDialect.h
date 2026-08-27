#ifndef TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_DIALECT_H_
#define TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"  // 생성된 op이 ::mlir::BytecodeOpInterface::Trait를 상속
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h.inc"

#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXTypes.h.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXOps.h.inc"

namespace mlir {
namespace aimx {

// pim_math.h 에서 그대로 온 기계 상수. IR 의 verifier 와 codegen 이 함께 쓴다.
constexpr int64_t kNumChannels = 16;
constexpr int64_t kNumBanks = 16;
constexpr int64_t kALUSize = 16;
constexpr int64_t kTypeSize = 2;      // sizeof(short) — bfloat16
constexpr int64_t kBurstSize = 32;    // DRAM burst
constexpr int64_t kRegSize = 256;     // vecA / vACC 한 channel 몫 (element)
constexpr int64_t kPIMWidth = 512;    // ALU_SIZE * NUM_BANKS * TYPE_SIZE (byte)
constexpr int64_t kChunk = 4096;      // (BURST_SIZE/TYPE_SIZE) * ALU_SIZE * NUM_CHANNELS
constexpr int64_t kStrideShift = 33;  // pitch 를 주소에 실을 때 미는 비트 수
constexpr int64_t kRecordBytes = 0x20;
constexpr int64_t kGroupBytes = kRecordBytes * kNumChannels;
constexpr int64_t kPMMSize = 0x30000;
constexpr int64_t kVACCOffset = 0x10000;
constexpr int64_t kPMMAlign = 256 * 1024;

// pim_sw의 multi_mmap_drv는 allocation을 2 MiB pool 단위로 확장하면서 크기를
// signed int로 받는다. 따라서 2 GiB(0x80000000)부터 음수로 overflow한다.
// runtime을 바꾸지 않는 동안 한 allocation이 요청할 수 있는 최대 안전 크기는
// INT32_MAX 이하의 가장 큰 2 MiB 배수다.
constexpr int64_t kRuntimePoolBytes = 2 * 1024 * 1024;
constexpr int64_t kMaxRuntimeAllocationBytes = 1023 * kRuntimePoolBytes;

// PIM Device 하나가 한 step 에 내는 출력 byte 수 = lane 수 * TYPE_SIZE.
constexpr int64_t kDeviceWidth = kNumBanks * kNumChannels * kALUSize * kTypeSize;
constexpr int64_t kLanes = kDeviceWidth / kTypeSize;  // 4096

} // namespace aimx
} // namespace mlir

#endif // TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_DIALECT_H_
