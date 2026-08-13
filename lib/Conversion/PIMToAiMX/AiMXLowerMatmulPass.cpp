#include "triton-shared/Conversion/PIMToAiMX/PIMToAiMX.h"
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_AIMXLOWERMATMUL
#include "triton-shared/Conversion/PIMToAiMX/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

using aimx::Opcode;

// descriptor program 하나를 짓는 동안 쓰는 도구. 주소식이 전부 byte 단위 정수 산술이라
// 이 몇 줄이면 §9.4 의 식을 그대로 옮길 수 있다.
struct ProgramBuilder {
  OpBuilder b;
  Location loc;

  ProgramBuilder(Block *body, Location loc)
      : b(OpBuilder::atBlockBegin(body)), loc(loc) {}

  Value cst(int64_t v) {
    return b.create<arith::ConstantIndexOp>(loc, v).getResult();
  }
  Value mul(Value x, Value y) {
    return b.create<arith::MulIOp>(loc, x, y).getResult();
  }
  Value mul(Value x, int64_t y) { return mul(x, cst(y)); }
  Value add(Value x, Value y) {
    return b.create<arith::AddIOp>(loc, x, y).getResult();
  }
  Value divBy(Value x, int64_t y) {
    return b.create<arith::DivSIOp>(loc, x, cst(y)).getResult();
  }

  // scf.for 를 만들고 몸체 안으로 들어간다. 안쪽에서 만드는 값이 바깥 값을 그대로
  // 쓰므로 삽입 지점만 옮기면 된다.
  scf::ForOp loop(Value ub, int64_t step) {
    auto f = b.create<scf::ForOp>(loc, cst(0), ub, cst(step));
    b = OpBuilder::atBlockBegin(f.getBody());
    return f;
  }
  // loop 몸체에서 나와 그 loop 바로 뒤로 옮긴다.
  void leave(scf::ForOp f) { b.setInsertionPointAfter(f); }

  aimx::OpcodeAttr op(Opcode flags) {
    return aimx::OpcodeAttr::get(b.getContext(), flags);
  }
  // src / dst 가 없는 쪽은 널이다. 참조 구현이 0x0 을 넣는 자리이고, 그 방향은 PIM
  // logic 내부(vecA / vACC)라 주소가 없다.
  void desc(Value src, Value srcOff, Value srcPitch, Value dst, Value dstOff,
            Value dstPitch, int64_t len, Opcode flags) {
    b.create<aimx::DescOp>(loc, src, srcOff, srcPitch, dst, dstOff, dstPitch,
                           b.getI64IntegerAttr(len), op(flags));
  }
};

struct AiMXLowerMatmulPass
    : public mlir::triton::impl::AiMXLowerMatmulBase<AiMXLowerMatmulPass> {
  using AiMXLowerMatmulBase::AiMXLowerMatmulBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> funcs;
    module.walk([&](func::FuncOp f) {
      if (f->hasAttr("aimx.tiling"))
        funcs.push_back(f);
    });
    // program 은 module 앞머리에 func 순서대로 쌓는다. 한 func 안에서는 재배치가 먼저이고
    // 계산이 다음인데, 그 순서가 chain 안의 순서이기도 하다.
    OpBuilder mb(module.getBody(), module.getBody()->begin());
    for (func::FuncOp f : funcs) {
      auto tiling = f->getAttrOfType<aimx::TilingAttr>("aimx.tiling");
      if (!tiling) {
        f->emitError("aimx-lower-matmul: aimx.tiling is not an #aimx.tiling");
        return signalPassFailure();
      }
      // #aimx.tile_group 은 --convert-pim-to-aimx 가 적어 둔 하드웨어 눈금이다. 아래
      // 두 program 이 같은 값을 자기 손으로 다시 계산하므로 여기서 대조한다 — 안 그러면
      // 한쪽만 고쳐도 조용히 어긋난다.
      auto group = f->getAttrOfType<aimx::TileGroupAttr>("aimx.tile_group");
      if (!group) {
        f->emitError("aimx-lower-matmul: the function has no #aimx.tile_group");
        return signalPassFailure();
      }
      int64_t batch = tiling.getBatch();
      ArrayRef<int64_t> exec = group.getExecution().asArrayRef();
      if (exec.size() != 3 || exec[0] != tiling.getMT() ||
          exec[1] != aimx::kLanes / batch || exec[2] != aimx::kRegSize / batch) {
        f->emitError("aimx-lower-matmul: #aimx.tile_group execution disagrees "
                     "with #aimx.tiling");
        return signalPassFailure();
      }
      buildRelayout(mb, f.getName(), tiling);
      buildCompute(mb, f.getName(), tiling);
    }
  }

  // program 껍데기. block argument 는 aimx.build_descriptors 의 args 와 순서대로 대응한다.
  // createBlock 이 삽입 지점을 새 block 안으로 옮기므로 되돌려 놓는다 — 그래야 다음
  // program 이 이 program 뒤에 온다.
  aimx::DescriptorProgramOp shell(OpBuilder &mb, StringRef name, int64_t batch,
                                  ArrayRef<Type> args, Location loc) {
    auto prog = mb.create<aimx::DescriptorProgramOp>(
        loc, mb.getStringAttr(name), mb.getI64IntegerAttr(batch));
    SmallVector<Location> locs(args.size(), loc);
    OpBuilder::InsertionGuard guard(mb);
    mb.createBlock(&prog.getBody(), {}, args, locs);
    return prog;
  }

  //===--------------------------------------------------------------------===//
  // A 재배치
  //===--------------------------------------------------------------------===//
  //
  // vecA 는 "행 방향으로 늘어선 A" 가 아니라 "reduction 방향으로 늘어선 A" 를 원한다.
  // 그래서 A 를 256 개씩 읽어(OPCODE_STRIDE 로 행 간격을 padded_k 로 주고) BC|TP 로
  // 다시 쓴다. 두 descriptor 가 한 쌍이다.
  void buildRelayout(OpBuilder &mb, StringRef base, aimx::TilingAttr t) {
    MLIRContext *ctx = &getContext();
    Location loc = mb.getInsertionBlock()->getParentOp()->getLoc();
    auto idxTy = IndexType::get(ctx);
    auto bufTy = aimx::BufferType::get(ctx);
    int64_t batch = t.getBatch();

    auto prog = shell(mb, (base + "_relayout").str(), batch,
                      {idxTy, idxTy, idxTy, bufTy, bufTy}, loc);
    Block *body = &prog.getBody().front();
    Value mT = body->getArgument(0), kT = body->getArgument(1),
          paddedK = body->getArgument(2), a = body->getArgument(3),
          arel = body->getArgument(4);

    ProgramBuilder p(body, loc);
    Value rowPitch = p.mul(paddedK, aimx::kTypeSize);
    Value arelRow = p.mul(kT, aimx::kTypeSize * aimx::kNumChannels);

    auto mLoop = p.loop(mT, batch);
    Value mg = mLoop.getInductionVar();
    Value aRow = p.mul(mg, rowPitch);
    Value arelBase = p.mul(mg, arelRow);

    auto sLoop = p.loop(kT, aimx::kRegSize);
    Value s = sLoop.getInductionVar();

    // A 를 읽어 온다. dst 는 PIM logic 안(vecA)이라 주소가 없다.
    p.desc(a, p.add(p.mul(s, aimx::kTypeSize), aRow), rowPitch,
           /*dst=*/Value(), Value(), Value(), aimx::kPIMWidth,
           Opcode::pim_mode | Opcode::stride);
    // 32 byte 를 512 byte 덩어리 안에서 16 번 되풀이해 쓴다. src 는 vecA 다.
    Value block = p.mul(p.divBy(s, aimx::kRegSize), aimx::kDeviceWidth * batch);
    p.desc(/*src=*/Value(), Value(), Value(), arel, p.add(block, arelBase),
           Value(), aimx::kDeviceWidth * batch, Opcode::bc | Opcode::tp);
  }

  //===--------------------------------------------------------------------===//
  // 계산
  //===--------------------------------------------------------------------===//
  //
  // reduction step 하나가 A operand descriptor 와 B operand descriptor 한 쌍이고,
  // output block 하나가 끝나면 vACC 를 C 로 흘려보낸다.
  void buildCompute(OpBuilder &mb, StringRef base, aimx::TilingAttr t) {
    MLIRContext *ctx = &getContext();
    Location loc = mb.getInsertionBlock()->getParentOp()->getLoc();
    auto idxTy = IndexType::get(ctx);
    auto bufTy = aimx::BufferType::get(ctx);
    int64_t batch = t.getBatch();
    int64_t blockBytes = aimx::kDeviceWidth / batch;   // output block 한 개분
    int64_t stepK = aimx::kRegSize / batch;            // vecA 한 번이 지탱하는 행 수

    auto prog = shell(mb, (base + "_compute").str(), batch,
                      {idxTy, idxTy, idxTy, bufTy, bufTy, bufTy, bufTy}, loc);
    Block *body = &prog.getBody().front();
    Value mT = body->getArgument(0), kT = body->getArgument(1),
          paddedN = body->getArgument(2), arel = body->getArgument(3),
          bBuf = body->getArgument(4), cBuf = body->getArgument(5),
          pmm = body->getArgument(6);

    ProgramBuilder p(body, loc);
    Value rowPitch = p.mul(paddedN, aimx::kTypeSize);
    Value arelRow = p.mul(kT, aimx::kTypeSize * aimx::kNumChannels);
    // 출력 4096 칸을 한 번에 내므로 n 방향 조각 수가 이것이다.
    Value cols = p.divBy(p.mul(paddedN, batch), aimx::kLanes);
    Value vacc = p.cst(aimx::kVACCOffset);
    Value vaccPitch = p.cst(blockBytes);

    auto mLoop = p.loop(mT, batch);
    Value mg = mLoop.getInductionVar();
    Value arelBase = p.mul(mg, arelRow);
    Value cRow = p.mul(mg, rowPitch);

    auto rLoop = p.loop(cols, 1);
    Value rOff = p.mul(rLoop.getInductionVar(), blockBytes);

    auto sLoop = p.loop(kT, stepK);
    Value s = sLoop.getInductionVar();

    // vecA 를 읽는다. 재배치가 이미 순서를 맞춰 두었으므로 pitch 가 없다.
    p.desc(arel, p.add(p.mul(p.divBy(s, stepK), aimx::kDeviceWidth), arelBase),
           Value(), Value(), Value(), Value(), aimx::kDeviceWidth,
           Opcode::a_mask | Opcode::matmul | Opcode::pim_mode);
    // B 는 행 간격이 padded_n 인 2 차원 배열에서 온다.
    p.desc(bBuf, p.add(rOff, p.mul(s, rowPitch)), rowPitch, Value(), Value(),
           Value(), blockBytes,
           Opcode::b_mask | Opcode::matmul | Opcode::pim_mode | Opcode::stride);
    p.leave(sLoop);

    // vACC 를 C 로 흘려보낸다. 여기서 output block 하나가 확정된다.
    p.desc(pmm, vacc, vaccPitch, cBuf, p.add(rOff, cRow), rowPitch, blockBytes,
           Opcode::c_mask | Opcode::matmul | Opcode::stride);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createAiMXLowerMatmulPass() {
  return std::make_unique<AiMXLowerMatmulPass>();
}
