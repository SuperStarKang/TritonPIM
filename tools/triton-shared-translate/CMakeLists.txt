get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_llvm_executable(triton-shared-translate triton-shared-translate.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(triton-shared-translate)
target_link_libraries(triton-shared-translate PRIVATE
  ${dialect_libs}
  ${triton_libs}
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
mlir_check_all_link_libraries(triton-shared-translate)
