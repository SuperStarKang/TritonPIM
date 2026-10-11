#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "triton-shared/Target/AiMX/AiMXTranslations.h"
#include "triton-shared/Target/UPMEM/UPMEMTranslations.h"

using namespace mlir;

// Target IR을 소스로 옮기는 도구.
// 사용: triton-shared-translate --upmem-to-host       in.mlir -o host.cpp
//       triton-shared-translate --upmem-to-device     in.mlir -o device.c
//       triton-shared-translate --aimx-to-runtime-stub in.mlir -o stub.cpp
//       triton-shared-translate --aimx-to-model-stub   in.mlir -o model.cpp
//       triton-shared-translate --aimx-to-descriptors  in.mlir -o descriptors.h
int main(int argc, char **argv) {
  triton::registerUPMEMToHostTranslation();
  triton::registerUPMEMToDeviceTranslation();
  triton::registerAiMXToRuntimeStubTranslation();
  triton::registerAiMXToModelStubTranslation();
  triton::registerAiMXToDescriptorsTranslation();
  return failed(mlirTranslateMain(argc, argv, "Triton-Shared Translation Tool"));
}
