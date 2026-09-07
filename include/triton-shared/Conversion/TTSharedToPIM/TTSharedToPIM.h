#ifndef TRITON_CONVERSION_TTSHAREDTOPIM_TTSHAREDTOPIM_H
#define TRITON_CONVERSION_TTSHAREDTOPIM_TTSHAREDTOPIM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL_PIMNORMALIZEADDRESS
#define GEN_PASS_DECL_CONVERTTTSHAREDTOPIM
#define GEN_PASS_DECL_PIMBUILDLOGICALGRID
#define GEN_PASS_DECL_PIMRECOVERACCESSORIGIN
#define GEN_PASS_DECL_PIMFORMCOMPUTE
#define GEN_PASS_DECL_PIMFORMBLOCKGROUPS
#define GEN_PASS_DECL_PIMSETTARGETPROFILE
#define GEN_PASS_DECL_PIMMAPDEVICES
#define GEN_PASS_DECL_PIMFORMEXECUTIONTILES
#define GEN_PASS_DECL_PIMPLACEDATA
#define GEN_PASS_DECL_PIMVERIFY
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createPIMNormalizeAddressPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertTTSharedToPIMPass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMBuildLogicalGridPass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMRecoverAccessOriginPass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMFormComputePass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMFormBlockGroupsPass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMSetTargetProfilePass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMMapDevicesPass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMFormExecutionTilesPass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMPlaceDataPass();
std::unique_ptr<OperationPass<ModuleOp>> createPIMVerifyPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TTSHAREDTOPIM_TTSHAREDTOPIM_H
