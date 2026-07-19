add_triton_library(UPMEMIR
  UPMEMDialect.cpp

  DEPENDS
  UPMEMTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
