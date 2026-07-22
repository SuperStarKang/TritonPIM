add_triton_library(TTSharedToPIM
  TTSharedToPIMPass.cpp
  PIMBuildGridPass.cpp
  PIMRemoveBuffersPass.cpp
  PIMAnalyzeLogicalWorkPass.cpp
  PIMPartitionPass.cpp
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
