#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TPtrToLLVM/Passes.h"
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
  mlir::triton::registerTPtrToLLVMPasses();
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
  mlir::triton::registerPIMReuseExecutionTileInputsPass();
  mlir::triton::registerPIMFormHostPlanPass();
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
      mlir::affine::AffineDialect, mlir::bufferization::BufferizationDialect,
      mlir::LLVM::LLVMDialect>();

  // CPU backend의 llir 단계(one-shot-bufferize)를 mlir-opt 대신 이 도구로 실행하므로
  // mlir-opt가 등록하던 bufferization 구현을 같이 등록한다. 다른 pipeline은 bufferize를 쓰지 않는다.
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  // buffer-deallocation-pipeline(같은 llir 단계)이 memref.dealloc을 넣는 데 필요한 구현
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
}
