add_triton_library(UPMEMIR
  UPMEMDialect.cpp

  DEPENDS
  UPMEMTableGen
  PIMTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  PIMIR
  )
