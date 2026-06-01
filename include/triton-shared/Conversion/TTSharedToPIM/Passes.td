#ifndef TTSHARED_TO_PIM_CONVERSION_PASSES
#define TTSHARED_TO_PIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTTSharedToPIM : Pass<"convert-ttshared-to-pim", "mlir::ModuleOp"> {
  let summary = "Convert TTSharedIR (linalg.matmul) to vendor-neutral PIM Common IR";
  let description = [{
    triton.dot_origin이 붙은 linalg.matmul(=tt.dot에서 내려온 gemm)을 찾아, 그 함수를
    PIM Common IR(memory_region / tile_plan / transfer / compute / launch)로 교체한다.
    no-split 단일 tile만 다룬다 (split=1 고정). granular plan pass 분해와 다중 전략은 이후 단계.
  }];
  let constructor = "mlir::triton::createConvertTTSharedToPIMPass()";
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::func::FuncDialect"];
  let options = [
    Option<"vendor", "vendor", "std::string", /*default*/"\"upmem\"",
           "pim.launch에 기록할 target PIM vendor 이름">,
  ];
}

#endif
