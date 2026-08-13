add_triton_library(TTSharedToPIM
  PIMNormalizeAddressPass.cpp
  TTSharedToPIMPass.cpp
  PIMBuildLogicalGridPass.cpp
  PIMRecoverAccessOriginPass.cpp
  PIMFormComputePass.cpp
  PIMFormBlockGroupsPass.cpp
  PIMSetTargetProfilePass.cpp
  PIMPlanGroupExecutionPass.cpp
  PIMMapDevicesPass.cpp
  PIMPlaceDataPass.cpp
  PIMVerifyPass.cpp

  DEPENDS
  PIMTableGen
  TTSharedToPIMConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRFuncDialect
  PIMIR
)
