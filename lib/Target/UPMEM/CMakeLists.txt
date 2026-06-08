add_triton_library(UPMEMTranslation
  UPMEMToHost.cpp
  UPMEMToDevice.cpp

  DEPENDS
  UPMEMTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  UPMEMIR
)
