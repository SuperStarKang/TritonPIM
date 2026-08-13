add_triton_library(AiMXIR
  AiMXDialect.cpp

  DEPENDS
  AiMXTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
