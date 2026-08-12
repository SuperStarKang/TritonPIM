#ifndef PIM_TO_AIMX_CONVERSION_PASSES_H
#define PIM_TO_AIMX_CONVERSION_PASSES_H

#include "triton-shared/Conversion/PIMToAiMX/PIMToAiMX.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/PIMToAiMX/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
