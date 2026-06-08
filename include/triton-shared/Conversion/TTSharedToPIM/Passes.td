#ifndef TTSHARED_TO_PIM_CONVERSION_PASSES
#define TTSHARED_TO_PIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def BuildPIMOffload : Pass<"build-pim-offload", "mlir::ModuleOp"> {
  let summary = "Build vendor-neutral PIM Common IR around a preserved compute body (faithful-body)";
  let description = [{
    tt.dot에서 내려온 linalg.matmul을 가진 함수에 PIM 오프로드 분배 구조를 합성한다(DESIGN 11.1).
    계산 본문은 치환하지 않고 pim.compute region에 유닛-로컬 nest로 보존하며(faithful-body),
    그 주위에 memory_region / transfer / tile_plan(grid·SPLIT) / execution_group(mapping_policy·
    num_units·local_lanes)을 합성한다. BLOCK은 matmul shape에서, SPLIT/LOCAL_LANES는 option에서
    읽는다(DESIGN 16). 전체 M/N/K는 IR에 박지 않고 본문 block arg의 런타임 dim으로 흐른다.
  }];
  let constructor = "mlir::triton::createBuildPIMOffloadPass()";
  // faithful-body 본문은 scf/memref/arith/linalg op을 생성하므로, 입력 IR이 이들을 참조하지
  // 않더라도 context에 로드돼 있도록 의존 dialect로 선언한다(미선언 시 build 시점 크래시).
  let dependentDialects = ["mlir::pim::PIMDialect", "mlir::func::FuncDialect",
                           "mlir::scf::SCFDialect", "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect", "mlir::linalg::LinalgDialect"];
  let options = [
    Option<"vendor", "vendor", "std::string", /*default*/"\"upmem\"",
           "pim.launch에 기록할 target PIM vendor 이름">,
    Option<"splitM", "split-m", "unsigned", /*default*/"1",
           "M축 inter-unit SPLIT 수. >1이면 출력 M(행)을 여러 유닛에 분배(output_m_partitioned).">,
    Option<"splitN", "split-n", "unsigned", /*default*/"1",
           "N축 inter-unit SPLIT 수. >1이면 출력 N(열)을 여러 유닛에 분배(output_n_partitioned).">,
    Option<"localLanes", "local-lanes", "unsigned", /*default*/"1",
           "유닛 내부 병렬 lane 수(=NR_TASKLETS). execution_group.local_lanes에 기록(DESIGN 16).">,
  ];
}

#endif
