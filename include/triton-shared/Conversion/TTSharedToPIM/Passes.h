#ifndef TTSHARED_TO_PIM_CONVERSION_PASSES_H
#define TTSHARED_TO_PIM_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
