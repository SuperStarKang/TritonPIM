#ifndef PIM_DIALECT
#define PIM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// PIM dialect — vendor-neutral PIM 공통 실행 모델 IR.
// UPMEM/AiMX 같은 vendor 용어(DPU/tasklet/MRAM 등)는 여기 등장하지 않는다.
//===----------------------------------------------------------------------===//

def PIM_Dialect : Dialect {
  let name = "pim";
  let cppNamespace = "::mlir::pim";

  let summary = "Vendor-neutral PIM common dialect";
  let description = [{
    PIM Common IR. TTSharedIR에서 lowering되어 PIM 공통 실행 계획(메모리 영역,
    타일 계획, 전송, 연산, 실행)을 vendor 중립적으로 표현한다.
  }];

  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

class PIM_Op<string mnemonic, list<Trait> traits = []>
    : Op<PIM_Dialect, mnemonic, traits>;

class PIM_Type<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<PIM_Dialect, name, traits> {
  // 파라미터 이름이 TypeDef의 `mnemonic` 필드와 같으면 자기 대입으로 막히므로 _mnemonic 사용.
  let mnemonic = _mnemonic;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// 메모리 영역 핸들. pim.memory_region이 buffer를 감싸 만든 결과로, 이후 transfer/
// compute가 이 핸들을 참조한다. 파라미터 없는 opaque 핸들로 시작한다.
def PIM_RegionType : PIM_Type<"Region", "region"> {
  let summary = "Handle to a PIM memory region";
}

// 타일 계획 핸들. pim.tile_plan이 만든 결과로 compute가 참조한다.
def PIM_TilePlanType : PIM_Type<"TilePlan", "tile_plan"> {
  let summary = "Handle to a PIM tile plan";
}

// 실행 그룹 핸들. pim.execution_group이 만든 유닛 분배 정책을 compute가 참조한다.
def PIM_ExecGroupType : PIM_Type<"ExecGroup", "exec_group"> {
  let summary = "Handle to a PIM execution group (unit assignment policy)";
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

// buffer를 PIM 메모리 영역으로 감싼다. space/layout/role은 vendor 중립 메타데이터로
// 지금은 문자열 attribute로 둔다 (검증이 필요한 enum은 해당 pass 도입 시 교체).
def PIM_MemoryRegionOp : PIM_Op<"memory_region", [Pure]> {
  let summary = "Wrap a buffer as a vendor-neutral PIM memory region";
  let arguments = (ins AnyType:$buffer,
                       StrAttr:$space,
                       StrAttr:$layout,
                       StrAttr:$role);
  let results = (outs PIM_RegionType:$result);
  let assemblyFormat = "$buffer attr-dict `:` functional-type($buffer, $result)";
}

// 타일 계획. block_*는 유닛 내부 타일 크기, split_*는 유닛 간 분배 수(13.0/11.5.1 참고).
def PIM_TilePlanOp : PIM_Op<"tile_plan", [Pure]> {
  let summary = "Tile plan: BLOCK (intra-unit) + SPLIT (inter-unit)";
  let arguments = (ins I64Attr:$block_m, I64Attr:$block_n, I64Attr:$block_k,
                       I64Attr:$split_m, I64Attr:$split_n, I64Attr:$split_k);
  let results = (outs PIM_TilePlanType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

// 유닛 분배 정책 (DESIGN 10.2). mapping_policy는 split로부터 결정한 vendor-neutral enum 문자열
// (none / output_n_partitioned / output_m_partitioned / output_2d_partitioned /
//  reduction_k_partitioned). num_units/local_lanes는 option·autotune 값(O2에서 capability_ref로
// 대체 가능). stationary_data는 상주 입력(gemm은 weight "B"). 실제 유닛 배정·reduce 실현은 vendor
// lowering이 하고, 이 공통 op은 intent만 기록한다(DESIGN 11.0 경계).
def PIM_ExecutionGroupOp : PIM_Op<"execution_group", [Pure]> {
  let summary = "Unit assignment policy (mapping_policy / num_units / local_lanes)";
  let arguments = (ins StrAttr:$mapping_policy, I64Attr:$num_units,
                       I64Attr:$local_lanes, StrAttr:$stationary_data);
  let results = (outs PIM_ExecGroupType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

// host↔device 명시적 전송. src/dst는 memory space 이름 문자열.
def PIM_TransferOp : PIM_Op<"transfer"> {
  let summary = "Explicit transfer of a region between memory spaces";
  let arguments = (ins PIM_RegionType:$region, StrAttr:$src, StrAttr:$dst);
  let assemblyFormat = "$region attr-dict `:` type($region)";
}

// PIM 연산 (faithful-body). inputs/outputs는 memory_region 핸들(전체 buffer), tile_plan은 grid·SPLIT.
// 본문(scf/linalg/memref)은 치환하지 않고 region에 그대로 보존한다(DESIGN 4.1/10.2/10.6).
// ins/outs는 variadic이라 gemm(2 in/1 out)뿐 아니라 임의 arity 커널(gemv/relu/fused)도 표현한다.
// region block arg는 본문이 참조하는 buffer(전체 또는 로컬 슬라이스) memref다 — operand의 region
// 핸들과는 별개다. primitive("gemm" 등)는 전략 힌트 속성일 뿐 본문을 대체하지 않으며, 본문 각 op의
// 세부 class는 pim.class 태그로 단다(DESIGN 10.6). 전체 M/N/K는 IR에 박지 않는다: 본문은 block arg의
// 런타임 dim을 쓰고, host codegen은 launcher가 넘기는 런타임 M/N/K를 쓴다. NoTerminator: 본문 끝에
// 별도 종결 op를 요구하지 않는다. AttrSizedOperandSegments: ins/outs 두 variadic 경계를 기록.
def PIM_ComputeOp : PIM_Op<"compute", [NoTerminator, AttrSizedOperandSegments]> {
  let summary = "PIM compute: per-unit body in a region (faithful-body)";
  let arguments = (ins Variadic<PIM_RegionType>:$inputs,
                       Variadic<PIM_RegionType>:$outputs,
                       PIM_ExecGroupType:$execution_group,
                       PIM_TilePlanType:$tile_plan,
                       StrAttr:$primitive);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    `ins` `(` $inputs `)` `outs` `(` $outputs `)` `using` $execution_group `,` $tile_plan attr-dict
    `:` `(` type($inputs) `)` `(` type($outputs) `)` `(` type($execution_group) `,` type($tile_plan) `)` $body
  }];
}

// 실행 진입점. callee는 실행할 함수, vendor/device_id는 launch 대상.
def PIM_LaunchOp : PIM_Op<"launch"> {
  let summary = "Launch a PIM function on a vendor device";
  let arguments = (ins FlatSymbolRefAttr:$callee, StrAttr:$vendor, I64Attr:$device_id);
  let assemblyFormat = "$callee attr-dict";
}

#endif // PIM_DIALECT
