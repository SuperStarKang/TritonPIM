#ifndef PIM_TO_UPMEM_CONVERSION_PASSES
#define PIM_TO_UPMEM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LegalizePIMToUPMEM : Pass<"legalize-pim-to-upmem", "mlir::ModuleOp"> {
  let summary = "Lower PIM Common IR to UPMEM IR";
  let description = [{
    pim.compute / pim.transfer / pim.tile_plan / pim.memory_region을 가진 함수를
    UPMEM host 시퀀스(alloc_units / load_program / copy_to_global / launch / copy_from_global)와
    device_func로 내린다. faithful-body(O1.4): pim.compute 본문을 device_func region으로 옮긴다.
    SPLIT은 tile_plan에서 읽어 alloc_units(num_units/split_m/split_n)에 반영한다.
  }];
  let constructor = "mlir::triton::createLegalizePIMToUPMEMPass()";
  let dependentDialects = ["mlir::upmem::UPMEMDialect", "mlir::func::FuncDialect"];
}

#endif
