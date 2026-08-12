add_triton_library(PIMToAiMX
  PIMToAiMXPass.cpp
  AiMXLowerMatmulPass.cpp

  DEPENDS
  PIMTableGen
  AiMXTableGen
  PIMToAiMXConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  PIMIR
  AiMXIR
)
