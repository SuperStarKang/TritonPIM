#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXTargetSpec.h"

#include "triton-shared/Dialect/PIM/IR/TargetSpec.h"
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXFootprint.h"

#include <memory>
#include <numeric>

using namespace mlir;

namespace {

struct AiMXTargetSpec : public pim::TargetSpec {
  // buffer 목록은 쓰지 않는다 — AiMX 의 footprint 는 matmul 역할(A/Arel/B/C)과
  // 기계 상수로 닫힌 식이다.
  int64_t deviceFootprintBytes(pim::TargetProfileAttr profile,
                               ArrayRef<pim::BufferShape> /*buffers*/,
                               int64_t mT, int64_t nT,
                               int64_t kT) const override {
    int64_t batch = profile.getExecutionUnit().asArrayRef()[0];
    return aimx::footprintOf(batch, mT, kT, nT).total();
  }

  // n 상한이 없으면 한 device 폭 — pitch 눈금 kChunk 를 group 물리 폭으로 올린 것.
  int64_t preferredNGroups(pim::TargetProfileAttr /*profile*/,
                           int64_t /*nGroupUseful*/,
                           int64_t nGroupPhys) const override {
    int64_t physWidth =
        (aimx::kChunk + nGroupPhys - 1) / nGroupPhys * nGroupPhys;
    return physWidth / nGroupPhys;
  }

  // k tile 은 block group 과 vecA 적재 단위(kRegSize)를 동시에 채워야 한다. 물리
  // 눈금은 그래서 둘의 lcm 이고, padding 이 없으면 kRegSize 그대로다.
  std::pair<int64_t, int64_t> kGrain(pim::TargetProfileAttr /*profile*/,
                                     int64_t kGroupUseful,
                                     int64_t kGroupPhys) const override {
    int64_t kUnit = std::lcm(kGroupPhys, aimx::kRegSize);
    return {kUnit / kGroupPhys * kGroupUseful, kUnit};
  }
};

} // namespace

void mlir::aimx::registerAiMXTargetSpec() {
  pim::registerTargetSpec("aimx", std::make_unique<AiMXTargetSpec>());
}
