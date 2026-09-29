add_triton_library(PIMIR
  PIMDialect.cpp
  PIMUtils.cpp

  DEPENDS
  PIMTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  )
