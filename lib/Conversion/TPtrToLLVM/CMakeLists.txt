add_triton_library(TPtrToLLVM
  TPtrToLLVMPass.cpp

  DEPENDS
  TPtrToLLVMConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRComplexToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRPass
  MLIRTransforms
  MLIRVectorToLLVM
  TPtrIR
)
