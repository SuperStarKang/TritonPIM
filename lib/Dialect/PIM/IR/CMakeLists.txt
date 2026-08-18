add_triton_library(PIMIR
  PIMDialect.cpp
  PIMUtils.cpp
  TargetSpec.cpp

  DEPENDS
  PIMTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRSCFDialect
  )
