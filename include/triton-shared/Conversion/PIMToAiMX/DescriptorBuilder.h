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

DescriptorSequence buildCopy(ModuleOp module, StringRef name, int64_t rows,
                             int64_t columns);
DescriptorSequence buildElementwise(ModuleOp module, StringRef name,
                                    Opcode opcode, int64_t rows,
                                    int64_t columns);
DescriptorSequence buildMatmul(ModuleOp module, StringRef name, int64_t batch,
                               int64_t rows, int64_t reduction,
                               int64_t columns);

} // namespace mlir::aimx
#endif
