#ifndef TTSHARED_TO_PIM_CONVERSION_PASSES
#define TTSHARED_TO_PIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTTSharedToPIM : Pass<"convert-ttshared-to-pim", "mlir::ModuleOp"> {
  let summary = "Convert TTSharedIR (linalg.matmul) to vendor-neutral PIM Common IR";
  let description = [{
    tt.dot에서 내려온 linalg.matmul을 가진 함수를 PIM Common IR
    (memory_region / tile_plan / transfer / compute / launch)로 교체한다.
    BLOCK은 matmul shape에서, 전체 M/N/K는 함수 인자에서 가져온다. split_n은 옵션으로 받아
    tile_plan에 기록한다(>1이면 vendor lowering이 여러 유닛에 N축 분배).
  }];
  let constructor = "mlir::triton::createConvertTTSharedToPIMPass()";
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::func::FuncDialect"];
  let options = [
    Option<"vendor", "vendor", "std::string", /*default*/"\"upmem\"",
           "pim.launch에 기록할 target PIM vendor 이름">,
    Option<"splitN", "split-n", "unsigned", /*default*/"1",
           "N축 inter-unit SPLIT 수. >1이면 출력 N을 여러 유닛에 분배(output_n_partitioned).">,
  ];
}

#endif
