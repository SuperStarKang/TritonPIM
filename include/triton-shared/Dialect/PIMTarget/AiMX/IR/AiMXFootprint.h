#ifndef TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_FOOTPRINT_H_
#define TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_FOOTPRINT_H_

#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"

#include <cstdint>

namespace mlir {
namespace aimx {

// execution tile 하나가 PIM DRAM 에 요구하는 바이트. 설계 문서 §6.2 의 표 그대로다.
// --convert-pim-to-aimx(예산 재검증)와 공통 --pim-form-execution-tiles(k 상한 탐색)가
// 같은 식을 써야 하므로 여기 둔다. mT/kT/nT 는 물리 원소(block group phys 배수)다.
struct Footprint {
  int64_t paddedK, paddedN;
  int64_t a, arel, b, c, chain;
  int64_t groups; // descriptor group 수

  int64_t total() const { return a + arel + b + c + chain + kPMMSize; }
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
