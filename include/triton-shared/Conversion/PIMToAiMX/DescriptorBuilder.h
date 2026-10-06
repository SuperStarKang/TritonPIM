#ifndef TRITON_SHARED_CONVERSION_PIMTOAIMX_DESCRIPTORBUILDER_H
#define TRITON_SHARED_CONVERSION_PIMTOAIMX_DESCRIPTORBUILDER_H

#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::aimx {

// A program is one chain segment, not a classification of a Common device.
// View arguments are (buffer, byte offset, byte row pitch). Their addresses
// are bound when the chain is materialized, after device allocation.
struct DescriptorSequence {
  DescriptorProgramOp program;
  int64_t groups;
};

// Copy/elementwise views are contiguous panels with rows*columns divisible by
// kLanes. MAC views retain independent row pitches.
DescriptorSequence buildCopy(ModuleOp module, StringRef name, int64_t rows,
                             int64_t columns);
DescriptorSequence buildElementwise(ModuleOp module, StringRef name,
                                    Opcode opcode, int64_t rows,
                                    int64_t columns);
DescriptorSequence buildMatmul(ModuleOp module, StringRef name, int64_t batch,
                               int64_t rows, int64_t reduction,
                               int64_t columns);
// Arel stays resident for every K group until all output microtiles are drained.
DescriptorSequence buildMatmulRelayout(ModuleOp module, StringRef name,
                                       int64_t batch, int64_t rows,
                                       int64_t reduction);
DescriptorSequence buildMatmulAccumulate(ModuleOp module, StringRef name,
                                         int64_t batch, int64_t reduction);
DescriptorSequence buildMatmulDrain(ModuleOp module, StringRef name,
                                    int64_t batch);

} // namespace mlir::aimx
#endif
