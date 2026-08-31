#ifndef PIM_DIALECT
#define PIM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// PIM dialect — target 무관 PIM 공통 IR.
// 공통 구조는 데이터 3계층이다: block(Triton tile 눈금) ⊂ block group(device-local
// 계산 단위) ⊂ execution tile(PIM Device 하나의 compute+transfer/residency 단위).
// 기계는 각 계층의 소비자로 대응된다 — execution tile 하나는 PIM Device 하나가,
// block group 하나는 그 Device 안의 계산 단위다. bank/core/lane으로 실제 실행하는
// 방식은 target lowering이 정하며 target 용어(DPU/tasklet/MRAM/channel 등)는 여기
// 등장하지 않는다.
//===----------------------------------------------------------------------===//

def PIM_Dialect : Dialect {
  let name = "pim";
  let cppNamespace = "::mlir::pim";

  let summary = "Target-neutral PIM common dialect";
  let description = [{
    ttshared IR(--triton-to-linalg-experimental 출력)에서 올라와, 메모리 접근을
    주소 명세(pim.access)와 전송(pim.load/pim.store)으로 표현하고, 배치를
    pim.device + pim.place 로 표현한다. 계산은 linalg 를 그대로 쓴다.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

class PIM_Op<string mnemonic, list<Trait> traits = []>
    : Op<PIM_Dialect, mnemonic, traits>;

class PIM_Type<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<PIM_Dialect, name, traits> {
  // 파라미터 이름이 TypeDef의 `mnemonic` 필드와 같으면 자기 대입으로 막히므로 _mnemonic 사용.
  let mnemonic = _mnemonic;
}

class PIM_Attr<string name, string _mnemonic>
    : AttrDef<PIM_Dialect, name> {
  let mnemonic = _mnemonic;
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// 기계 profile. --pim-set-target-profile 이 module 에 붙이고, 그 뒤의 pass 들이 읽는다.
// 데이터 3계층이 하드웨어에서 필요로 하는 공통 topology와 byte 상한만 담는다.
// target별 실행 방식(tasklet/SIMD), 실제 활성 lane 수, runtime/toolchain 설정은
// target lowering/config의 책임이다. capacity limit은 항상 존재하며 0은 해당 저장소가
// 없음을 뜻한다.
def PIM_TargetProfileAttr : PIM_Attr<"TargetProfile", "target_profile"> {
  let summary = "Common machine profile shared by every PIM target";
  let parameters = (ins
      "::mlir::StringAttr":$target,
      "int64_t":$num_units,
      "int64_t":$banks_per_unit,
      "int64_t":$cores_per_bank,
      "int64_t":$lanes_per_core,
      "int64_t":$bank_capacity_limit_bytes,
      "int64_t":$local_capacity_limit_bytes);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

// PIM Common IR의 canonical 좌표 순서는 [m, n, k]로 고정한다. m/n의 독립 output
// loop는 pim.axis로 표시한다. K slot은 hierarchy와 device mapping에 그대로 존재하지만
// compute domain은 별도 source plane이 아니라 pim.reduction loop 하나로 표현한다.
// scheduling 뒤 그 loop의 local window는 pim.temporal_reduction으로 명시한다.
def PIM_AxisAttr : PIM_Attr<"Axis", "axis"> {
  let summary = "Canonical m, n, or k axis identity";
  let parameters = (ins "::mlir::StringAttr":$name);
  let assemblyFormat = "`<` $name `>`";
  let genVerifyDecl = 1;
}

// target 확장 attribute 는 두지 않는다. 공통 pass 가 target 이름의 attribute 를 읽으면
// 이 dialect 가 target 무관하다는 전제가 깨지고, 공통 pass 가 읽지 않는 값이면 target IR 에
// 두면 된다. 공통 pass 가 정말로 필요로 하는 기계 값은 위 target_profile 의 필드가 된다.

// pim.device 의 필수 파라미터. 고정 축 순서 [m,n,semantic-k]의 partition 수다.
// m/n axis 또는 K reduction이 없으면 해당 값은 1이다.
def PIM_DeviceMappingAttr : PIM_Attr<"DeviceMapping", "device_mapping"> {
  let summary = "PIM Device partitions in canonical [m,n,k] order";
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$partitions);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

// block — 데이터 3계층의 첫째 층. Triton block 한 개가 각 canonical 축에서
// 계산하는 유효 원소 수다. 축 순서는 [m, n, k]이고 없는 축은 1이다.
def PIM_BlockAttr : PIM_Attr<"Block", "block"> {
  let summary = "The useful element shape of one Triton block";
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$shape);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

// block group — 데이터 3계층의 둘째 층. 각 축에서 인접한 Block을 몇 개 묶는지만
// 저장한다. Group 원소 shape는 #pim.block.shape * blocks로 파생한다.
def PIM_BlockGroupAttr : PIM_Attr<"BlockGroup", "block_group"> {
  let summary = "An integer grid of adjacent Triton blocks";
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$blocks);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

// execution tile — 데이터 3계층의 셋째 층. 각 축에서 Group을 몇 개 상주시킬지만
// 저장한다. 원소 shape는 Block과 Group에서 파생한다. 없는 축의
// group 수도 곱의 항등원인 1이다. PIM DRAM 용량은 이 값을 늘리는 목표가 아니라
// legality 상한이다.
// groups[k]는 device 하나가 K round 한 번에 처리할 semantic reduction Group 수다.
// p_k와 K round는 이 하나의 domain을 spatial/temporal 차원으로 함께 나눈다.
// Common scheduling이 각 Tile의 Group 범위와 반복 순서를 명시한다. target lowering은
// 그 경계를 바꾸지 않고 고정된 Tile 내부 실행을 target op로 옮긴다.
def PIM_ExecutionTileAttr : PIM_Attr<"ExecutionTile", "execution_tile"> {
  let summary = "Resident group grid of one execution tile";
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$groups);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// 논리 주소 명세 핸들. pim.access가 만든 결과로, pim.load/pim.store가 참조한다.
def PIM_AccessType : PIM_Type<"Access", "access"> {
  let summary = "Handle to a PIM logical memory access (address spec)";
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

// 논리 주소 명세 (memref.reinterpret_cast 대응). 데이터 이동 없음(Pure).
// base = 원본 host 버퍼, offset = 선형 시작 위치, sizes = 타일 shape(static),
// strides = 축별 간격, origin = 축별 타일 시작 위치(원소 단위).
// 언제나 offset = Σ_d origin[d]·strides[d] 다.
// valid/pad = mask 경계와 채움값. id = 이 access 를 가리키는 이름으로, pim.place 가
// 이 값으로 대응 access 를 가리킨다.
def PIM_AccessOp : PIM_Op<"access", [Pure, AttrSizedOperandSegments]> {
  let summary = "Logical strided access spec (no data movement)";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$base, Index:$offset,
                       Variadic<Index>:$origin,
                       Variadic<Index>:$strides, DenseI64ArrayAttr:$sizes,
                       Variadic<Index>:$valid, Optional<AnyType>:$pad,
                       I64Attr:$id);
  let results = (outs PIM_AccessType:$result);
  let assemblyFormat = [{
    $base `offset` $offset `sizes` $sizes `strides` `[` $strides `]`
    ( `origin` `[` $origin^ `]` )?
    ( `valid` `[` $valid^ `]` )?
    ( `pad` $pad^ `:` type($pad) )?
    attr-dict `:` type($base) `->` type($result)
  }];
  let hasVerifier = 1;
}

// host→local 전송. 값(tensor)을 직접 만들어 staging 버퍼를 암묵화한다.
def PIM_LoadOp : PIM_Op<"load", [MemoryEffects<[MemRead]>]> {
  let summary = "Load a tile through a PIM access (buffer implicit)";
  let arguments = (ins PIM_AccessType:$access);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$access attr-dict `:` type($access) `->` type($result)";
}

// local→host 직접 전송. 서로 독립적인 K contribution을 최종 output에 합치는 write는
// 아래 pim.combine으로 표현한다. 따라서 이 op 자체에는 partial/final 의미가 없다.
def PIM_StoreOp : PIM_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store a tile through a PIM access";
  let arguments = (ins AnyRankedTensor:$value, PIM_AccessType:$access);
  let assemblyFormat = "$value `,` $access attr-dict `:` type($value) `,` type($access)";
}

// 서로 독립적으로 계산된 K contribution을 원래 operation의 최종 output에
// 결합한다. partial은 Execution Tile/device가 계산한 내부 값이고 destination은
// reduction 축이 제거된 최종 논리 pim.access다. spatial/temporal/device K의 범위는
// Common scheduling에 이미 있으므로 이 op에는 별도 partition/policy를 저장하지
// 않는다. Target은 결합 여부나 목적지를 재해석하지 않고, 이 연산을 host reduction,
// device accumulator 등 target에 맞는 동일 의미의 구현으로 내린다.
def PIM_CombineOp : PIM_Op<"combine", [MemoryEffects<[MemWrite]>]> {
  let summary = "Combine one internal partial into the final logical output";
  let arguments = (ins AnyRankedTensor:$partial,
                       PIM_AccessType:$destination,
                       StrAttr:$kind);
  let assemblyFormat = [{
    $partial `into` $destination `kind` $kind attr-dict
    `:` type($partial) `,` type($destination)
  }];
  let hasVerifier = 1;
}

// PIM Device 하나가 하는 일을 담는 region. region 밖 = host(데이터 준비·배포·회수),
// region 안 = device 하나.
// block argument는 고정 순서 [m,n,k]의 device 좌표다. 없는 축은 partition 1과 좌표 0을
// 사용한다. 전체 logical extent는 scheduling loop에 이미 materialize되므로 복사해 두지 않는다.
def PIM_DeviceOp : PIM_Op<"device", [NoTerminator]> {
  let summary = "Per-device region: the body is what one PIM Device executes";
  let arguments = (ins PIM_DeviceMappingAttr:$mapping);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $mapping attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

// 버퍼 하나가 이 PIM Device 에 상주해야 하는 영역. 최종 값을 직접 싣는다.
//   start    = 이 device 의 첫 원소 (balanced block 이 낸 값 그대로)
//   size     = 실제로 맡는 원소 수. 경계 tile 의 유효 범위까지 반영된 값이라
//              언제나 start + size 가 원본 버퍼 안이다.
//   allocation_shape = Block×Group×Execution Tile에서 정적으로 파생된 device의
//                      fixed-slot 크기. scheduling round가 바뀌어도 변하지 않는다.
//   strides  = 원본 버퍼 안의 축별 간격
// direction: "in"(host→device) | "out"(device→host).
// access: 이 영역을 유도한 pim.access 의 id.
// replicated_over: 이 축만 다른 device 들은 같은 데이터를 받는다.
def PIM_PlaceOp : PIM_Op<"place", [AttrSizedOperandSegments]> {
  let summary = "Final placement of one buffer on one PIM Device";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$buffer,
                       Variadic<Index>:$start, Variadic<Index>:$size,
                       Variadic<Index>:$allocation_shape, Variadic<Index>:$strides,
                       StrAttr:$direction, I64Attr:$access,
                       OptionalAttr<StrArrayAttr>:$replicated_over);
  let assemblyFormat = [{
    $buffer `start` `[` $start `]` `size` `[` $size `]`
    `allocation_shape` `[` $allocation_shape `]` `strides` `[` $strides `]`
    attr-dict `:` type($buffer)
  }];
  let hasVerifier = 1;
}

#endif // PIM_DIALECT
