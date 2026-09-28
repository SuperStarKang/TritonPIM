#ifndef TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_FOOTPRINT_H_
#define TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_FOOTPRINT_H_

#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"

#include <cstdint>
#include <initializer_list>
#include <limits>

namespace mlir {
namespace aimx {

// Checked physical extents shared by target allocations and descriptor builders.
inline bool checkedProduct(std::initializer_list<int64_t> factors,
                           int64_t &result) {
  result = 1;
  for (int64_t factor : factors) {
    if (factor <= 0 || result > std::numeric_limits<int64_t>::max() / factor)
      return false;
    result *= factor;
  }
  return true;
}

inline bool checkedRoundUp(int64_t value, int64_t unit, int64_t &result) {
  if (value <= 0 || unit <= 0)
    return false;
  return checkedProduct({value / unit + (value % unit != 0), unit}, result);
}

} // namespace aimx
} // namespace mlir

#endif // TRITON_SHARED_DIALECT_PIMTARGET_AIMX_IR_AIMX_FOOTPRINT_H_
