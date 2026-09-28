#ifndef TRITON_SHARED_TARGET_AIMX_DESCRIPTOR_REUSE_H_
#define TRITON_SHARED_TARGET_AIMX_DESCRIPTOR_REUSE_H_

#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <utility>

namespace mlir::aimx_codegen {

// Compiler-local code generation choices, not a Common IR annotation or a
// runtime policy. Both translators must agree on which patch functions exist.
enum class DescriptorReuse { Unchanged, Addresses, Rebuild };

struct DescriptorReusePlan {
  DescriptorReuse kind = DescriptorReuse::Rebuild;
  // Existing SSA values and their full-tile values, not runtime cache keys.
  SmallVector<std::pair<Value, int64_t>> fullBounds;
};

inline bool descriptorValueInvariant(Value value, scf::ExecuteRegionOp region,
                                     aimx::BuildDescriptorsOp build = {},
                                     DescriptorReusePlan *conditional = nullptr) {
  SmallVector<Value> pending{value};
  llvm::DenseSet<Value> visited;
  auto localLoop = [&](scf::ForOp loop) {
    return region->isProperAncestor(loop) ||
           (build && loop->getParentOfType<aimx::DescriptorProgramOp>());
  };
  auto loopBounds = [&](scf::ForOp loop) {
    pending.append({loop.getLowerBound(), loop.getUpperBound(), loop.getStep()});
  };
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!visited.insert(current).second) continue;
    // Resident allocations and per-device coordinates retain their identity
    // until the compiled plan is replaced. Host data pointers are not PIM PAs.
    if (isa<aimx::BufferType>(current.getType()) || getConstantIntValue(current)) continue;
    if (auto argument = dyn_cast<BlockArgument>(current)) {
      Operation *owner = argument.getOwner()->getParentOp();
      if (owner == region->getParentOfType<aimx::ForeachDeviceOp>().getOperation())
        continue;
      if (build && isa<aimx::DescriptorProgramOp>(owner)) {
        pending.push_back(build.getArgs()[argument.getArgNumber()]);
        continue;
      }
      auto loop = dyn_cast<scf::ForOp>(owner);
      if (!loop || !localLoop(loop)) return false;
      loopBounds(loop);
      if (argument.getArgNumber() != 0) {
        unsigned position = argument.getArgNumber() - 1;
        pending.push_back(loop.getInitArgs()[position]);
        pending.push_back(loop.getBody()->getTerminator()->getOperand(position));
      }
      continue;
    }
    Operation *definition = current.getDefiningOp();
    if (conditional && definition && !region->isAncestor(definition) &&
        !definition->getParentOfType<aimx::DescriptorProgramOp>()) {
      // Common scheduling emits min(tileGroups, remainingGroups) before the
      // construction boundary. Equality to the constant makes this value fixed
      // for full tiles without changing the loop or rounding up a tail.
      if (auto bound = dyn_cast<arith::MinSIOp>(definition)) {
        auto full = getConstantIntValue(bound.getLhs());
        if (!full) full = getConstantIntValue(bound.getRhs());
        if (full) {
          if (llvm::none_of(conditional->fullBounds, [&](const auto &entry) {
                return entry.first == current;
              }))
            conditional->fullBounds.emplace_back(current, *full);
          continue;
        }
      }
    }
    if (definition && definition->getName().getDialectNamespace() == "arith") {
      llvm::append_range(pending, definition->getOperands());
      continue;
    }
    auto loop = dyn_cast_or_null<scf::ForOp>(definition);
    if (!loop || !localLoop(loop)) return false;
    unsigned position = cast<OpResult>(current).getResultNumber();
    loopBounds(loop);
    pending.push_back(loop.getInitArgs()[position]);
    pending.push_back(loop.getBody()->getTerminator()->getOperand(position));
  }
  return true;
}

template <typename RecordOp>
inline bool descriptorLoopInvariant(scf::ForOp loop, scf::ExecuteRegionOp region,
                                    aimx::BuildDescriptorsOp build = {},
                                    DescriptorReusePlan *conditional = nullptr) {
  bool emitsRecords = false;
  loop.walk([&](RecordOp) { emitsRecords = true; });
  // A scalar-only loop may update an address without changing the record count.
  if (!emitsRecords) return true;
  return descriptorValueInvariant(loop.getLowerBound(), region, build, conditional) &&
         descriptorValueInvariant(loop.getUpperBound(), region, build, conditional) &&
         descriptorValueInvariant(loop.getStep(), region, build, conditional);
}

inline FailureOr<DescriptorReuse> classifyDescriptorReuseImpl(
    scf::ExecuteRegionOp region, DescriptorReusePlan *conditional) {
  bool fixedAddresses = true, fixedTopology = true;
  WalkResult result = region.walk([&](aimx::BuildDescriptorsOp build) {
    auto program = SymbolTable::lookupNearestSymbolFrom<aimx::DescriptorProgramOp>(
        build, build.getProgramAttr());
    if (!program || program.getBody().front().getNumArguments() != build.getArgs().size()) {
      build.emitError("AiMX descriptor reuse requires a resolved program with matching arguments");
      return WalkResult::interrupt();
    }
    // Only address fields are SSA operands. Length, opcode and batch are fixed
    // attributes; next links depend on the preserved descriptor traversal.
    program.walk([&](aimx::DescOp desc) {
      for (Value operand : desc->getOperands())
        fixedAddresses &= descriptorValueInvariant(operand, region, build, conditional);
    });
    program.walk([&](scf::ForOp loop) {
      fixedTopology &= descriptorLoopInvariant<aimx::DescOp>(loop, region, build, conditional);
    });
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return failure();
  region.walk([&](scf::ForOp loop) {
    fixedTopology &= descriptorLoopInvariant<aimx::BuildDescriptorsOp>(loop, region, {}, conditional);
  });
  if (!fixedTopology) return DescriptorReuse::Rebuild;
  return fixedAddresses ? DescriptorReuse::Unchanged : DescriptorReuse::Addresses;
}

inline FailureOr<DescriptorReusePlan> classifyDescriptorReuse(scf::ExecuteRegionOp region) {
  DescriptorReusePlan plan;
  auto kind = classifyDescriptorReuseImpl(region, nullptr);
  if (failed(kind)) return failure();
  // Do not add conditions when unconditional reuse or address patching already
  // works. Conditional reuse only handles the previously dynamic topology.
  if (*kind == DescriptorReuse::Rebuild) {
    kind = classifyDescriptorReuseImpl(region, &plan);
    if (failed(kind)) return failure();
    if (*kind == DescriptorReuse::Rebuild) plan.fullBounds.clear();
  }
  plan.kind = *kind;
  return plan;
}

} // namespace mlir::aimx_codegen

#endif // TRITON_SHARED_TARGET_AIMX_DESCRIPTOR_REUSE_H_
