add_triton_library(TTSharedToPIM
  PIMNormalizeAddressPass.cpp
  TTSharedToPIMPass.cpp
  PIMBuildLogicalGridPass.cpp
  PIMRecoverAccessOriginPass.cpp
  PIMFormComputePass.cpp
  PIMFormBlockGroupsPass.cpp
  PIMSetTargetProfilePass.cpp
  PIMMapDevicesPass.cpp
  PIMFormExecutionTilesPass.cpp
  PIMPlaceDataPass.cpp
  PIMReuseExecutionTileInputsPass.cpp
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
