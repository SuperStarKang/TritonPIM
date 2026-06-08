add_triton_library(TTSharedToPIM
  TTSharedToPIMPass.cpp

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
  MLIRFuncDialect
  PIMIR
)
