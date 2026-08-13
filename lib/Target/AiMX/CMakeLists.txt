add_triton_library(AiMXTranslation
  AiMXToRuntimeStub.cpp
  AiMXToDescriptors.cpp

  DEPENDS
  AiMXTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  AiMXIR
)
