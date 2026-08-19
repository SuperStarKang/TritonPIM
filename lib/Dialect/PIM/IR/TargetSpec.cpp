#include "triton-shared/Dialect/PIM/IR/TargetSpec.h"

#include "llvm/ADT/StringMap.h"

using namespace mlir;

static llvm::StringMap<std::unique_ptr<pim::TargetSpec>> &registry() {
  static llvm::StringMap<std::unique_ptr<pim::TargetSpec>> specs;
  return specs;
}

void mlir::pim::registerTargetSpec(llvm::StringRef target,
                                   std::unique_ptr<TargetSpec> spec) {
  registry()[target] = std::move(spec);
}

const mlir::pim::TargetSpec *
mlir::pim::lookupTargetSpec(llvm::StringRef target) {
  auto it = registry().find(target);
  return it == registry().end() ? nullptr : it->second.get();
}
