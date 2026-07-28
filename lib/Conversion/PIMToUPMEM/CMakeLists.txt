add_triton_library(PIMToUPMEM
  PIMToUPMEMPass.cpp

  DEPENDS
  PIMTableGen
  UPMEMTableGen
  PIMToUPMEMConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  PIMIR
  UPMEMIR
)
