#ifndef PIM_TO_UPMEM_CONVERSION_PASSES
#define PIM_TO_UPMEM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LegalizePIMToUPMEM : Pass<"legalize-pim-to-upmem", "mlir::ModuleOp"> {
  let summary = "Lower PIM Common IR to UPMEM IR";
  let description = [{
    pim.compute @gemm / pim.transfer / pim.tile_plan / pim.memory_region을 가진 함수를
    UPMEM host 시퀀스(alloc_units / load_program / copy_to_global / launch / copy_from_global)와
    device_func(local_gemm)로 내린다. no-split 단일 tile만 다룬다 (DPU 1개).
  }];
  let constructor = "mlir::triton::createLegalizePIMToUPMEMPass()";
  let dependentDialects = ["mlir::upmem::UPMEMDialect", "mlir::func::FuncDialect"];
}

#endif
