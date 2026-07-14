add_triton_library(TTSharedToPIM
  TTSharedToPIMPass.cpp
  PIMRemoveBuffersPass.cpp
  PIMClassifyComputePass.cpp

  DEPENDS
  PIMTableGen
  TTSharedToPIMConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRFuncDialect
  PIMIR
)
