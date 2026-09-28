#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h"
#include "triton-shared/Conversion/PIMToAiMX/Passes.h"
#include "triton-shared/Conversion/PIMToUPMEM/Passes.h"
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h"
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Transform/AddLLVMDebugInfo/Passes.h"

#include "mlir/InitAllPasses.h"

inline void registerTritonSharedDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerLinalgPasses();
  mlir::triton::registerTritonPasses();
  mlir::triton::registerTritonToLinalgPass();
  mlir::triton::registerTritonToLinalgExperimentalPasses();
  mlir::triton::registerTritonToStructuredPass();
  mlir::triton::registerTritonPtrToMemref();
  mlir::triton::registerUnstructuredToMemref();
  mlir::triton::registerTritonToUnstructuredPasses();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerStructuredToMemrefPasses();
  mlir::triton::registerPIMNormalizeAddressPass();
  mlir::triton::registerConvertTTSharedToPIMPass();
  mlir::triton::registerPIMBuildLogicalGridPass();
  mlir::triton::registerPIMRecoverAccessOriginPass();
  mlir::triton::registerPIMFormComputePass();
  mlir::triton::registerPIMFormBlockGroupsPass();
  mlir::triton::registerPIMSetTargetProfilePass();
  mlir::triton::registerPIMMapDevicesPass();
  mlir::triton::registerPIMFormExecutionTilesPass();
  mlir::triton::registerPIMPlaceDataPass();
  mlir::triton::registerPIMVerifyPass();
  mlir::triton::registerConvertPIMToUPMEMPass();
  mlir::triton::registerConvertPIMToAiMXPass();
  mlir::triton::registerAddLLVMDebugInfoPass();

  // TODO: register Triton & TritonGPU passes
  registry.insert<
      mlir::pim::PIMDialect, mlir::upmem::UPMEMDialect,
      mlir::aimx::AiMXDialect,
      mlir::tptr::TPtrDialect, mlir::ptr::PtrDialect,
      mlir::ttx::TritonTilingExtDialect, mlir::tts::TritonStructuredDialect,
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::linalg::LinalgDialect, mlir::func::FuncDialect,
      mlir::tensor::TensorDialect, mlir::memref::MemRefDialect,
      mlir::affine::AffineDialect, mlir::bufferization::BufferizationDialect>();
}
