//===----------------------------------------------------------------------===//
//
// Lowers the tptr ops produced by TritonToPtr / ReconcilePtrCasts and the
// !ptr.ptr type to LLVM.
//
// !ptr.ptr is also used as a tensor/memref element type (a tensor of pointers
// becomes memref<Nx!ptr.ptr> after bufferization). The standard memref-to-LLVM
// type converter does not know !ptr.ptr, so the tptr patterns cannot run in a
// separate pass before or after it. This pass runs the same standard patterns
// that the CPU backend previously ran as separate passes
// (arith/math/complex/vector/index/finalize-memref/func/cf to LLVM) in one
// conversion whose type converter additionally maps !ptr.ptr -> !llvm.ptr.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TPtrToLLVM/TPtrToLLVM.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TPtrToLLVM/Passes.h.inc"

namespace {

// TritonToPtr multiplies element offsets by this value to get byte offsets.
struct TypeOffsetConverter : public ConvertOpToLLVMPattern<tptr::TypeOffsetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tptr::TypeOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type baseType = getTypeConverter()->convertType(op.getBaseType());
    if (!baseType)
      return rewriter.notifyMatchFailure(op, "base type has no LLVM type");
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    uint64_t size =
        DataLayout::closest(op).getTypeSize(baseType).getFixedValue();
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, resultType, rewriter.getIntegerAttr(resultType, size));
    return success();
  }
};

// The offset is already in bytes, so step over i8.
struct PtrAddConverter : public ConvertOpToLLVMPattern<tptr::PtrAddOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tptr::PtrAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()),
        rewriter.getI8Type(), adaptor.getBase(),
        ValueRange{adaptor.getOffset()});
    return success();
  }
};

// The pointer becomes both the allocated and aligned pointer of a descriptor
// with offset 0 and the static sizes/strides of the result type.
struct ToMemrefConverter : public ConvertOpToLLVMPattern<tptr::ToMemrefOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tptr::ToMemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value descriptor = MemRefDescriptor::fromStaticShape(
        rewriter, op.getLoc(), *getTypeConverter(), op.getRes().getType(),
        adaptor.getArg());
    rewriter.replaceOp(op, descriptor);
    return success();
  }
};

// ReconcilePtrCasts casts unranked memrefs to memref<1xT> before from_memref,
// so only ranked inputs are expected.
struct FromMemrefConverter : public ConvertOpToLLVMPattern<tptr::FromMemrefOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tptr::FromMemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefType = dyn_cast<MemRefType>(op.getInput().getType());
    if (!memrefType)
      return rewriter.notifyMatchFailure(op, "unranked memref input");
    Type elementType =
        getTypeConverter()->convertType(memrefType.getElementType());
    MemRefDescriptor descriptor(adaptor.getInput());
    Value alignedPtr = descriptor.alignedPtr(rewriter, op.getLoc());
    Value offset = descriptor.offset(rewriter, op.getLoc());
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()),
        elementType, alignedPtr, ValueRange{offset});
    return success();
  }
};

struct IntToPtrConverter : public ConvertOpToLLVMPattern<tptr::IntToPtrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tptr::IntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(
        op, getTypeConverter()->convertType(op.getRes().getType()),
        adaptor.getArg());
    return success();
  }
};

struct PtrToIntConverter : public ConvertOpToLLVMPattern<tptr::PtrToIntOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tptr::PtrToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(
        op, getTypeConverter()->convertType(op.getRes().getType()),
        adaptor.getArg());
    return success();
  }
};

class TPtrToLLVMPass : public TPtrToLLVMBase<TPtrToLLVMPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    LLVMTypeConverter typeConverter(context);
    typeConverter.addConversion([](ptr::PtrType type) -> Type {
      return LLVM::LLVMPointerType::get(type.getContext());
    });

    RewritePatternSet patterns(context);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    populateComplexToLLVMConversionPatterns(typeConverter, patterns);
    populateVectorToLLVMConversionPatterns(typeConverter, patterns);
    index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateAssertToLLVMConversionPattern(typeConverter, patterns);
    patterns.add<TypeOffsetConverter, PtrAddConverter, ToMemrefConverter,
                 FromMemrefConverter, IntToPtrConverter, PtrToIntConverter>(
        typeConverter);

    LLVMConversionTarget target(*context);
    target.addIllegalDialect<tptr::TPtrDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTPtrToLLVMPass() {
  return std::make_unique<TPtrToLLVMPass>();
}
