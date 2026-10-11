#include "triton-shared/Conversion/PIMToAiMX/DescriptorBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace {

// Count the descriptors at the same sites that emit them. There is no second
// kernel-specific chain-size formula to keep in sync with these programs.
struct ProgramBuilder {
  OpBuilder b;
  Location loc;
  aimx::DescriptorProgramOp program;
  int64_t groups = 0;
  int64_t repetitions = 1;
  SmallVector<int64_t> repeatStack;

  ProgramBuilder(ModuleOp module, StringRef name, int64_t batch,
                 unsigned views, unsigned buffers = 0)
      : b(module.getContext()), loc(module.getLoc()) {
    b.setInsertionPointToStart(module.getBody());
    program = b.create<aimx::DescriptorProgramOp>(
        loc, b.getStringAttr(name), b.getI64IntegerAttr(batch));
    SmallVector<Type> types;
    for (unsigned i = 0; i < views; ++i) {
      types.push_back(aimx::BufferType::get(b.getContext()));
      types.push_back(b.getIndexType());
      types.push_back(b.getIndexType());
    }
    types.append(buffers, aimx::BufferType::get(b.getContext()));
    b.createBlock(&program.getBody(), {}, types,
                  SmallVector<Location>(types.size(), loc));
  }

  Value arg(unsigned i) { return program.getBody().front().getArgument(i); }
  Value c(int64_t value) {
    return b.create<arith::ConstantIndexOp>(loc, value);
  }
  Value add(Value lhs, Value rhs) {
    return b.create<arith::AddIOp>(loc, lhs, rhs);
  }
  Value mul(Value lhs, Value rhs) {
    return b.create<arith::MulIOp>(loc, lhs, rhs);
  }
  Value mul(Value lhs, int64_t rhs) { return mul(lhs, c(rhs)); }
  Value div(Value lhs, int64_t rhs) {
    return b.create<arith::DivSIOp>(loc, lhs, c(rhs));
  }
  Value at(unsigned view, Value row, Value column) {
    return add(arg(view * 3 + 1),
               add(mul(row, arg(view * 3 + 2)), mul(column, aimx::kTypeSize)));
  }
  scf::ForOp loop(int64_t end, int64_t step = 1) {
    auto loop = b.create<scf::ForOp>(loc, c(0), c(end), c(step));
    repeatStack.push_back(repetitions);
    repetitions *= (end + step - 1) / step;
    b.setInsertionPointToStart(loop.getBody());
    return loop;
  }
  void leave(scf::ForOp loop) {
    b.setInsertionPointAfter(loop);
    repetitions = repeatStack.pop_back_val();
  }
  void desc(Value src, Value srcOffset, Value srcPitch, Value dst,
            Value dstOffset, Value dstPitch, int64_t bytes, aimx::Opcode opcode) {
    b.create<aimx::DescOp>(
        loc, src, srcOffset, srcPitch, dst, dstOffset, dstPitch,
        b.getI64IntegerAttr(bytes), aimx::OpcodeAttr::get(b.getContext(), opcode));
    groups += repetitions;
  }
  aimx::DescriptorSequence finish() { return {program, groups}; }
};

} // namespace

aimx::DescriptorSequence aimx::buildCopy(ModuleOp module, StringRef name,
                                        int64_t rows, int64_t columns) {
  ProgramBuilder p(module, name, 1, 2);
  auto chunk = p.loop(rows * columns, kLanes);
  p.desc(p.arg(0), p.at(0, p.c(0), chunk.getInductionVar()), {},
         p.arg(3), p.at(1, p.c(0), chunk.getInductionVar()), {},
         kDeviceWidth, Opcode::copy);
  return p.finish();
}

aimx::DescriptorSequence aimx::buildElementwise(
    ModuleOp module, StringRef name, Opcode opcode, int64_t rows,
    int64_t columns) {
  ProgramBuilder p(module, name, 1, 3, 1);
  Value pmm = p.arg(9), vacc = p.c(kVACCOffset);
  auto chunk = p.loop(rows * columns, kLanes);
  Value a = p.at(0, p.c(0), chunk.getInductionVar());
  Value b = p.at(1, p.c(0), chunk.getInductionVar());
  Value c = p.at(2, p.c(0), chunk.getInductionVar());
  bool multiply = opcode == Opcode::mul;
  p.desc(p.arg(0), a, {}, multiply ? Value() : pmm,
         multiply ? Value() : vacc, {}, kDeviceWidth,
         Opcode::a_mask | opcode |
             (multiply ? Opcode::pim_mode : Opcode::descr_dst_mask));
  p.desc(p.arg(3), b, {}, {}, {}, {}, kDeviceWidth,
         Opcode::b_mask | Opcode::pim_mode | opcode);
  p.desc(pmm, vacc, {}, p.arg(6), c, {}, kDeviceWidth,
         Opcode::c_mask | opcode);
  return p.finish();
}

aimx::DescriptorSequence aimx::buildMatmul(
    ModuleOp module, StringRef name, int64_t batch, int64_t rows,
    int64_t reduction, int64_t columns) {
  // Inputs and output have independent pitches. Arel is private to this MAC
  // site; subsequent Common operations receive C, not the relayout buffer.
  ProgramBuilder p(module, name, batch, 3, 2);
  Value arel = p.arg(9), pmm = p.arg(10);
  int64_t m = ((rows + batch - 1) / batch) * batch;
  int64_t k = ((reduction + kRegSize - 1) / kRegSize) * kRegSize;
  int64_t width = kLanes / batch;
  int64_t n = ((columns + width - 1) / width) * width;
  int64_t blockBytes = kDeviceWidth / batch;
  int64_t stepK = kRegSize / batch;

  auto gatherM = p.loop(m, batch);
  auto gatherK = p.loop(k, kRegSize);
  Value mg = gatherM.getInductionVar(), kg = gatherK.getInductionVar();
  p.desc(p.arg(0), p.at(0, mg, kg), p.arg(2), {}, {}, {}, kPIMWidth,
         Opcode::pim_mode | Opcode::stride);
  Value relayoutOffset = p.add(p.mul(mg, k * kTypeSize * kNumChannels),
                               p.mul(p.div(kg, kRegSize), kDeviceWidth * batch));
  p.desc({}, {}, {}, arel, relayoutOffset, {}, kDeviceWidth * batch,
         Opcode::bc | Opcode::tp);
  p.leave(gatherK);
  p.leave(gatherM);

  auto outputM = p.loop(m, batch);
  auto outputN = p.loop(n / width);
  Value row = outputM.getInductionVar();
  Value col = p.mul(outputN.getInductionVar(), width);
  auto macK = p.loop(k, stepK);
  Value s = macK.getInductionVar();
  Value aOffset = p.add(p.mul(row, k * kTypeSize * kNumChannels),
                        p.mul(p.div(s, stepK), kDeviceWidth));
  p.desc(arel, aOffset, {}, {}, {}, {}, kDeviceWidth,
         Opcode::a_mask | Opcode::matmul | Opcode::pim_mode);
  p.desc(p.arg(3), p.at(1, s, col), p.arg(5), {}, {}, {}, blockBytes,
         Opcode::b_mask | Opcode::matmul | Opcode::pim_mode | Opcode::stride);
  p.leave(macK);
  p.desc(pmm, p.c(kVACCOffset), p.c(blockBytes), p.arg(6), p.at(2, row, col),
         p.arg(8), blockBytes, Opcode::c_mask | Opcode::matmul | Opcode::stride);
  return p.finish();
}

aimx::DescriptorSequence aimx::buildMatmulRelayout(
    ModuleOp module, StringRef name, int64_t batch, int64_t rows,
    int64_t reduction) {
  ProgramBuilder p(module, name, batch, 2);
  int64_t m = ((rows + batch - 1) / batch) * batch;
  int64_t k = ((reduction + kRegSize - 1) / kRegSize) * kRegSize;
  auto gatherM = p.loop(m, batch);
  auto gatherK = p.loop(k, kRegSize);
  Value row = gatherM.getInductionVar(), col = gatherK.getInductionVar();
  p.desc(p.arg(0), p.at(0, row, col), p.arg(2), {}, {}, {}, kPIMWidth,
         Opcode::pim_mode | Opcode::stride);
  Value offset = p.add(p.arg(4),
      p.add(p.mul(row, k * kTypeSize * kNumChannels),
            p.mul(p.div(col, kRegSize), kDeviceWidth * batch)));
  p.desc({}, {}, {}, p.arg(3), offset, {}, kDeviceWidth * batch,
         Opcode::bc | Opcode::tp);
  return p.finish();
}

aimx::DescriptorSequence aimx::buildMatmulAccumulate(
    ModuleOp module, StringRef name, int64_t batch, int64_t reduction) {
  ProgramBuilder p(module, name, batch, 2);
  int64_t stepK = kRegSize / batch;
  auto macK = p.loop(reduction, stepK);
  Value k = macK.getInductionVar();
  Value offset = p.add(p.arg(1), p.mul(p.div(k, stepK), kDeviceWidth));
  p.desc(p.arg(0), offset, {}, {}, {}, {}, kDeviceWidth,
         Opcode::a_mask | Opcode::matmul | Opcode::pim_mode);
  p.desc(p.arg(3), p.at(1, k, p.c(0)), p.arg(5), {}, {}, {},
         kDeviceWidth / batch,
         Opcode::b_mask | Opcode::matmul | Opcode::pim_mode | Opcode::stride);
  return p.finish();
}

aimx::DescriptorSequence aimx::buildMatmulDrain(
    ModuleOp module, StringRef name, int64_t batch) {
  ProgramBuilder p(module, name, batch, 1, 1);
  int64_t bytes = kDeviceWidth / batch;
  p.desc(p.arg(3), p.c(kVACCOffset), p.c(bytes),
         p.arg(0), p.arg(1), p.arg(2), bytes,
         Opcode::c_mask | Opcode::matmul | Opcode::stride);
  return p.finish();
}
