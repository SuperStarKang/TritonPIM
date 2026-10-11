add_triton_library(PIMToAiMX
  PIMToAiMXPass.cpp
  DescriptorBuilder.cpp

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
