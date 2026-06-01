add_triton_library(PIMIR
  PIMDialect.cpp

  DEPENDS
  PIMTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
