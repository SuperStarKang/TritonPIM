#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "triton-shared/Target/UPMEM/UPMEMTranslations.h"

using namespace mlir;

// UPMEM IR을 host C++ / DPU C 소스로 옮기는 도구.
// 사용: triton-shared-translate --upmem-to-host  in.mlir -o host.cpp
//       triton-shared-translate --upmem-to-device in.mlir -o device.c
int main(int argc, char **argv) {
  triton::registerUPMEMToHostTranslation();
  triton::registerUPMEMToDeviceTranslation();
  return failed(mlirTranslateMain(argc, argv, "Triton-Shared Translation Tool"));
}
