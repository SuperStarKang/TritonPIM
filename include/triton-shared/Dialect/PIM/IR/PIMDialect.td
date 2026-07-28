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

  let summary = "Target-neutral PIM common dialect";
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
// origin = 축별 타일 시작 위치(원소 단위). offset을 축으로 되돌린 것이라 항상
//   offset = Σ_d origin[d]·strides[d]. 유닛이 맡을 데이터 영역은 origin에 유닛의 loop
//   범위를 대입해서 나오므로, 합쳐진 offset만으로는 유도할 수 없다. 분해가 안 되는
//   형태(주소를 loop iter_arg로 나르는 커널)에서는 비워 둔다.
// 선택 clause: valid+pad(⑤ mask 경계+채움값), gather_dim+index(⑥ 1축 row-gather).
def PIM_AccessOp : PIM_Op<"access", [Pure, AttrSizedOperandSegments]> {
  let summary = "Logical strided access spec (no data movement)";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base, Index:$offset,
                       Variadic<Index>:$origin,
                       Variadic<Index>:$strides, DenseI64ArrayAttr:$sizes,
                       Variadic<Index>:$valid, Optional<AnyType>:$pad,
                       Optional<AnyRankedTensor>:$index,
                       OptionalAttr<I64Attr>:$gather_dim);
  let results = (outs PIM_AccessType:$result);
  let assemblyFormat = [{
    $base `offset` $offset `sizes` $sizes `strides` `[` $strides `]`
    ( `origin` `[` $origin^ `]` )?
    ( `valid` `[` $valid^ `]` )?
    ( `pad` $pad^ `:` type($pad) )?
    ( `gather` $gather_dim `index` $index^ `:` type($index) )?
    attr-dict `:` type($base) `->` type($result)
  }];
  let hasVerifier = 1;
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

// access→access 직접 전송. passthrough 최적화 산출물(로컬 버퍼 없이 src→dst).
// src를 읽고 dst에 쓴다(Read+Write). 두 access가 각자의 clause(valid 등)를 그대로 가진다.
def PIM_CopyOp : PIM_Op<"copy", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Direct descriptor-to-descriptor transfer (no local buffer)";
  let arguments = (ins PIM_AccessType:$src, PIM_AccessType:$dst);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

// ⑦ 원소별 random gather. index 텐서의 각 원소를 base 버퍼의 임의 위치에서 읽는다. Read 효과.
// 선택 clause: mask 있으면 masked-out 원소는 default 값으로 채운다(predicated load).
def PIM_GatherOp : PIM_Op<"gather", [MemoryEffects<[MemRead]>, AttrSizedOperandSegments]> {
  let summary = "Per-element random gather from a base buffer";
  // default는 C++ 예약어라 operand 이름으로 못 쓴다. Triton 용어인 other를 쓴다.
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base, AnyRankedTensor:$index,
                       Optional<AnyRankedTensor>:$mask, Optional<AnyType>:$other);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $base `index` $index
    ( `mask` $mask^ `:` type($mask) )?
    ( `other` $other^ `:` type($other) )?
    attr-dict `:` type($base) `,` type($index) `->` type($result)
  }];
  let hasVerifier = 1;
}

// ⑦ 원소별 random scatter. value의 각 원소를 base 버퍼의 임의 위치에 쓴다. Write 효과.
// 선택 clause: mask 있으면 masked-out 원소는 쓰지 않는다(predicated store).
// optional operand가 mask 하나뿐이라 AttrSizedOperandSegments는 불필요(operand 개수로 판별).
def PIM_ScatterOp : PIM_Op<"scatter", [MemoryEffects<[MemWrite]>]> {
  let summary = "Per-element random scatter into a base buffer";
  let arguments = (ins AnyRankedTensor:$value, AnyRankedOrUnrankedMemRef:$base,
                       AnyRankedTensor:$index, Optional<AnyRankedTensor>:$mask);
  let assemblyFormat = [{
    $value `,` $base `index` $index
    ( `mask` $mask^ `:` type($mask) )?
    attr-dict `:` type($value) `,` type($base) `,` type($index)
  }];
}

//===----------------------------------------------------------------------===//
// 유닛 형태 — --pim-partition이 만든다.
//===----------------------------------------------------------------------===//

// 유닛 하나가 하는 일을 담는 region. 반복 공간을 유닛 수만큼 나눈 결과다.
// region 밖 = host가 하는 일(데이터 준비·배포·회수), region 안 = 유닛 하나가 하는 일.
// block argument = 이 유닛의 좌표(축마다 하나, axes 순서). counts = 축별 유닛 수.
// axes 이름은 pim.split_axes가 낸 것과 같다(grid 축이면 "x"/"y"/"z", reduction이면 "k").
def PIM_UnitsOp : PIM_Op<"units", [NoTerminator]> {
  let summary = "Per-unit region: the body is what one unit executes";
  let arguments = (ins StrArrayAttr:$axes, DenseI64ArrayAttr:$counts);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "`axes` $axes `counts` $counts attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

// 유닛 하나가 자기 메모리에 갖고 있어야 하는 원본 버퍼의 영역.
// pim.access가 인스턴스 하나의 타일이라면 이쪽은 유닛 하나가 전체 실행 동안 쓰는 영역이다.
// origin/sizes는 pim.access의 origin에 이 유닛의 loop 범위를 대입해서 나온다.
// strides는 원본 버퍼 안 배치라 pim.access의 것을 그대로 쓴다.
// direction: "in"(host→유닛) | "out"(유닛→host).
// access: 이 영역을 유도한 pim.access의 pim.access_id. 이 영역 안에서 타일이 어디인지는
//   그 access의 origin에서 이 영역의 origin을 빼면 나온다. 같은 버퍼를 여러 access가
//   쓰면 base만으로는 어느 것인지 못 정하므로 id로 가리킨다.
// replicated_over: 이 축만 다른 유닛들은 같은 데이터를 받는다 — host가 한 번만 준비하면 된다.
// combine/combine_over: 이 축만 다른 유닛들이 같은 영역에 부분 결과를 쓴다 — host가 합쳐야 한다.
def PIM_UnitDataOp : PIM_Op<"unit_data", [AttrSizedOperandSegments]> {
  let summary = "Region of a buffer that one unit needs in its own memory";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base,
                       Variadic<Index>:$origin, Variadic<Index>:$sizes,
                       Variadic<Index>:$strides, StrAttr:$direction,
                       I64Attr:$access,
                       OptionalAttr<StrArrayAttr>:$replicated_over,
                       OptionalAttr<StrAttr>:$combine,
                       OptionalAttr<StrArrayAttr>:$combine_over);
  let assemblyFormat = [{
    $base `origin` `[` $origin `]` `sizes` `[` $sizes `]` `strides` `[` $strides `]`
    attr-dict `:` type($base)
  }];
  let hasVerifier = 1;
}

#endif // PIM_DIALECT
