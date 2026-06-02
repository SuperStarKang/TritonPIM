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

// host↔device 명시적 전송. src/dst는 memory space 이름 문자열.
def PIM_TransferOp : PIM_Op<"transfer"> {
  let summary = "Explicit transfer of a region between memory spaces";
  let arguments = (ins PIM_RegionType:$region, StrAttr:$src, StrAttr:$dst);
  let assemblyFormat = "$region attr-dict `:` type($region)";
}

// PIM 연산. op="gemm"부터. a,b,c 영역과 tile_plan을 받아 결과 영역(partial C)을 낸다.
// full_m/n/k는 전체 문제 크기(dynamic). Triton 커널이 SPMD(타일 1개)라 IR의 BLOCK은
// tile_plan에 들어가지만 전체 M/N/K는 런타임 인자다. codegen이 전체를 돌려면 이 값이 필요하다.
def PIM_ComputeOp : PIM_Op<"compute"> {
  let summary = "PIM compute primitive (gemm, ...)";
  let arguments = (ins PIM_RegionType:$a, PIM_RegionType:$b, PIM_RegionType:$c,
                       PIM_TilePlanType:$tile_plan,
                       I32:$full_m, I32:$full_n, I32:$full_k,
                       StrAttr:$op);
  let results = (outs PIM_RegionType:$result);
  let assemblyFormat = [{
    `(` $a `,` $b `,` $c `)` `using` $tile_plan
    `extent` `(` $full_m `,` $full_n `,` $full_k `)` attr-dict
    `:` functional-type(operands, results)
  }];
}

// 실행 진입점. callee는 실행할 함수, vendor/device_id는 launch 대상.
def PIM_LaunchOp : PIM_Op<"launch"> {
  let summary = "Launch a PIM function on a vendor device";
  let arguments = (ins FlatSymbolRefAttr:$callee, StrAttr:$vendor, I64Attr:$device_id);
  let assemblyFormat = "$callee attr-dict";
}

#endif // PIM_DIALECT
