add_llvm_executable(triton-shared-translate triton-shared-translate.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(triton-shared-translate)
target_link_libraries(triton-shared-translate PRIVATE
  UPMEMTranslation
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTranslateLib
  MLIRFuncDialect
  MLIRMemRefDialect
)

mlir_check_all_link_libraries(triton-shared-translate)
