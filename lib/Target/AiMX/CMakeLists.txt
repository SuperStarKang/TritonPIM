add_triton_library(AiMXTranslation
  AiMXToRuntimeStub.cpp
  AiMXToDescriptors.cpp

  DEPENDS
  AiMXTableGen
  PIMTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  PIMIR
  AiMXIR
)
