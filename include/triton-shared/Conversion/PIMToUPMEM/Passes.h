#ifndef PIM_TO_UPMEM_CONVERSION_PASSES_H
#define PIM_TO_UPMEM_CONVERSION_PASSES_H

#include "triton-shared/Conversion/PIMToUPMEM/PIMToUPMEM.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/PIMToUPMEM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
