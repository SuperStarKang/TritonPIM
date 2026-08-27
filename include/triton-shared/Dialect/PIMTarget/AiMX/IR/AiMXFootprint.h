#ifndef TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_FOOTPRINT_H_
#define TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_FOOTPRINT_H_

#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"

#include <cstdint>

namespace mlir {
namespace aimx {

// execution tile 하나가 PIM DRAM 에 요구하는 바이트. 설계 문서 §6.2 의 표 그대로다.
// --convert-pim-to-aimx(예산 재검증)와 공통 --pim-form-execution-tiles(k 상한 탐색)가
// 같은 식을 써야 하므로 여기 둔다. mT/kT/nT 는 target 실행 눈금에 맞춰 padding한
// 물리 원소수다.
struct Footprint {
  int64_t paddedK, paddedN;
  int64_t a, arel, b, c, chain;
  int64_t groups; // descriptor group 수

  int64_t total() const { return a + arel + b + c + chain + kPMMSize; }

  // AiMX의 한 bank access granule은 kBurstSize(32 B)이고, 연속 주소
  // kDeviceWidth(8192 B)는 256개 bank에 burst 하나씩을 놓는다. pim_malloc은 각
  // buffer를 별도 allocation으로 만들며 시작 bank를 compiler에 알려 주지 않는다.
  // 따라서 각 allocation의 tail이 같은 bank에 놓일 수 있다고 보고 독립적으로
  // 올림한 합이 compile-time에 증명 가능한 안전한 per-bank 상한이다.
  int64_t maxBankBytes() const {
    auto oneAllocation = [](int64_t bytes) {
      return ((bytes + kDeviceWidth - 1) / kDeviceWidth) * kBurstSize;
    };
    return oneAllocation(kPMMSize) + oneAllocation(a) + oneAllocation(arel) +
           oneAllocation(b) + oneAllocation(c) + oneAllocation(chain);
  }

  // 공통 execution-tile 탐색도 target verifier와 같은 보수적 경계를 쓰기 위한 값.
  int64_t bankAccountedDeviceBytes() const {
    return maxBankBytes() * kNumBanks * kNumChannels;
  }
};

inline Footprint footprintOf(int64_t batch, int64_t mT, int64_t kT,
                             int64_t nT) {
  auto ceilDiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  Footprint f;
  f.paddedK = kChunk * ceilDiv(kT, kChunk);
  f.paddedN = kChunk * ceilDiv(nT, kChunk);
  f.a = mT * f.paddedK * kTypeSize;
  // Arel 은 A 를 16 배로 복제한다 — vecA 가 reduction 방향으로 늘어선 A 를 원한다.
  f.arel = mT * kT * kTypeSize * kNumChannels;
  f.b = kT * f.paddedN * kTypeSize;
  f.c = mT * f.paddedN * kTypeSize;
  int64_t rows = mT / batch;                 // output block 행
  int64_t cols = f.paddedN * batch / kLanes; // output block 열
  int64_t steps = kT * batch / kRegSize;     // reduction step
  f.groups = 2 * (kT / kRegSize) * rows + rows * cols * (2 * steps + 1);
  f.chain = f.groups * kGroupBytes;
  return f;
}

} // namespace aimx
} // namespace mlir

#endif // TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_FOOTPRINT_H_
