add_triton_library(AiMXIR
  AiMXDialect.cpp

  DEPENDS
  AiMXTableGen
  PIMTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  PIMIR
  )
