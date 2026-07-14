#ifndef PIM_DIALECT
#define PIM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// PIM dialect — target 무관 PIM 공통 IR.
// UPMEM/AiMX 같은 target 용어(DPU/tasklet/MRAM 등)는 여기 등장하지 않는다.
//===----------------------------------------------------------------------===//

def PIM_Dialect : Dialect {
  let name = "pim";
  let cppNamespace = "::mlir::pim";

  let summary = "Vendor-neutral PIM common dialect";
  let description = [{
    PIM 공통 IR. ttshared IR(--triton-to-linalg-experimental 출력)에서 올라와,
    메모리 접근을 주소 명세(pim.access)와 전송(pim.load/pim.store)으로 표현한다.
    계산은 linalg를 그대로 쓰고, 물리 배치(target)는 담지 않는다.
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

// 논리 주소 명세 핸들. pim.access가 만든 결과로, pim.load/pim.store가 참조한다.
// shape/element type은 pim.load 결과 tensor 타입이 들고 있어, 지금은 파라미터 없는
// 핸들로 둔다(상세 타입/verifier는 추후).
def PIM_AccessType : PIM_Type<"Access", "access"> {
  let summary = "Handle to a PIM logical memory access (address spec)";
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

// 논리 주소 명세 (memref.reinterpret_cast 대응). 데이터 이동 없음(Pure).
// base = 원본 host 버퍼(memref<*xT>), offset = 선형 시작 위치, sizes = 타일 shape(static),
// strides = 축별 간격. broadcast 축은 stride 0.
def PIM_AccessOp : PIM_Op<"access", [Pure]> {
  let summary = "Logical strided access spec (no data movement)";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base, Index:$offset,
                       Variadic<Index>:$strides, DenseI64ArrayAttr:$sizes);
  let results = (outs PIM_AccessType:$result);
  let assemblyFormat = [{
    $base `offset` $offset `sizes` $sizes `strides` `[` $strides `]` attr-dict
    `:` type($base) `->` type($result)
  }];
}

// host→local 전송. 값(tensor)을 직접 만들어 staging 버퍼를 암묵화한다. Read 효과.
def PIM_LoadOp : PIM_Op<"load", [MemoryEffects<[MemRead]>]> {
  let summary = "Load a tile through a PIM access (buffer implicit)";
  let arguments = (ins PIM_AccessType:$access);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$access attr-dict `:` type($access) `->` type($result)";
}

// local→host 전송. Write 효과.
def PIM_StoreOp : PIM_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store a tile through a PIM access";
  let arguments = (ins AnyRankedTensor:$value, PIM_AccessType:$access);
  let assemblyFormat = "$value `,` $access attr-dict `:` type($value) `,` type($access)";
}

#endif // PIM_DIALECT
