#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMTargetSpec.h"

#include "triton-shared/Dialect/PIM/IR/TargetSpec.h"

#include <memory>

using namespace mlir;

namespace {

// UPMEM 의 execution tile 은 MRAM 에 buffer 별 연속 배열로 상주한다. footprint 는
// buffer 모양의 곱을 그대로 더한 것이고, DMA 정렬을 위해 buffer 마다 8 byte 로 올린다.
struct UPMEMTargetSpec : public pim::TargetSpec {
  int64_t deviceFootprintBytes(pim::TargetProfileAttr /*profile*/,
                               ArrayRef<pim::BufferShape> buffers,
                               int64_t mT, int64_t nT,
                               int64_t kT) const override {
    int64_t total = 0;
    for (const pim::BufferShape &b : buffers) {
      int64_t elems = 1;
      for (const pim::BufferDim &d : b.dims) {
        switch (d.axis) {
        case pim::BufferDim::M:
          elems *= mT;
          break;
        case pim::BufferDim::N:
          elems *= nT;
          break;
        case pim::BufferDim::K:
          elems *= kT;
          break;
        case pim::BufferDim::Static:
          elems *= d.staticSize;
          break;
        }
      }
      total += (elems * b.elemBytes + 7) / 8 * 8;
    }
    return total;
  }
};

} // namespace

void mlir::upmem::registerUPMEMTargetSpec() {
  pim::registerTargetSpec("upmem", std::make_unique<UPMEMTargetSpec>());
}
