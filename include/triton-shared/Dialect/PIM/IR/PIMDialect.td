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
// block group 하나는 그 Device 안의 bank -> core -> lane 계층이 함께 분해해
// 소비한다(#pim.intra_device_plan). target 용어(DPU/tasklet/MRAM/channel 등)는
// 여기 등장하지 않는다.
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
// 데이터 3계층이 하드웨어에서 필요로 하는 것만 담는다 — device/bank/core/lane
// topology와 capacity(execution tile 의 예산).
// usable_*_capacity_bytes 는 세 상태를 구분한다: 값이 있으면 알려진 용량, 0 이면 그
// 저장소가 존재하지 않아 검사가 성립하지 않음, 필드가 없으면 미측정이라 그 값을
// 필요로 하는 검사에 도달하는 순간 진단을 낸다.
def PIM_TargetProfileAttr : PIM_Attr<"TargetProfile", "target_profile"> {
  let summary = "Common machine profile shared by every PIM target";
  let parameters = (ins
      "::mlir::StringAttr":$target,
      "int64_t":$number_of_pim_devices,
      "int64_t":$number_of_banks_per_pim_device,
      "int64_t":$number_of_pim_cores_per_bank,
      "int64_t":$number_of_parallel_lanes_per_pim_core,
      "::mlir::StringAttr":$parallel_execution_kind,
      OptionalParameter<"::mlir::IntegerAttr">:$usable_bank_capacity_bytes,
      OptionalParameter<"::mlir::IntegerAttr">:$usable_local_capacity_bytes);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

// PIM Common IR의 축은 [m, n, k]로 고정한다. m/n은 병렬 출력 축이고 k는
// reduction 축이다. 축의 이름 자체가 의미를 가지므로 별도의 scope/kind 문자열은
// 두지 않는다.
def PIM_AxisAttr : PIM_Attr<"Axis", "axis"> {
  let summary = "Canonical m, n, or k axis identity";
  let parameters = (ins "::mlir::StringAttr":$name);
  let assemblyFormat = "`<` $name `>`";
  let genVerifyDecl = 1;
}

// target 확장 attribute 는 두지 않는다. 공통 pass 가 target 이름의 attribute 를 읽으면
// 이 dialect 가 target 무관하다는 전제가 깨지고, 공통 pass 가 읽지 않는 값이면 target IR 에
// 두면 된다. 공통 pass 가 정말로 필요로 하는 기계 값은 위 target_profile 의 필드가 된다.

// pim.device 의 필수 파라미터. logical 축마다 device partition 수 하나다.
def PIM_DeviceMappingAttr : PIM_Attr<"DeviceMapping", "device_mapping"> {
  let summary = "Logical axes and the number of PIM Device partitions on each";
  let parameters = (ins "::mlir::ArrayAttr":$axes,
                        "::mlir::DenseI64ArrayAttr":$partitions);
  let assemblyFormat = "`<` struct(params) `>`";
}

// 하나의 block group 자체를 device 안에서 bank -> core -> lane 순서로 분할한다.
// partition vector의 고정 축 순서는 [m, n, k]다. 여러 block group을 서로 다른
// bank/core에 통째로 배정하는 attribute가 아니다.
def PIM_IntraDevicePlanAttr
    : PIM_Attr<"IntraDevicePlan", "intra_device_plan"> {
  let summary = "Bank, core, and lane decomposition of one block group";
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$bank_partitions,
                        "::mlir::DenseI64ArrayAttr":$core_partitions,
                        "int64_t":$lanes,
                        "::mlir::StringAttr":$lane_axis);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

// block — 데이터 3계층의 첫째 층. Triton block 한 개가 각 canonical 축에서
// 계산하는 유효 원소 수다. 축 순서는 [m, n, k]이고 없는 축은 0이다.
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
// group 수는 0이다. PIM DRAM 용량은 이 값을 늘리는 목표가 아니라 legality 상한이다.
// spatial split-K가 있으면 pim.axis="k" 좌표마다 별도 Execution Tile instance가
// 생기고, groups[k]는 각 spatial plane 내부 temporal reduction residency를 뜻한다.
// Group 순회 순서는 target 실행 방식에 속하므로 target lowering이 정한다.
def PIM_ExecutionTileAttr : PIM_Attr<"ExecutionTile", "execution_tile"> {
  let summary = "Resident group grid of one execution tile";
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$groups);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

// pim.place 가 어떤 logical 축을 따라 나뉘었는지 적는 정적 metadata. 버퍼 축마다
// 하나이고, 나뉘지 않은 축은 "" 다. 검증과 진단에만 쓰이고 target lowering 은 읽지 않는다.
def PIM_PlaceMappingAttr : PIM_Attr<"PlaceMapping", "place_mapping"> {
  let summary = "Which logical axis partitions each buffer axis";
  let parameters = (ins "::mlir::ArrayAttr":$axis_sources);
  let assemblyFormat = "`<` struct(params) `>`";
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

// local→host 전송.
def PIM_StoreOp : PIM_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store a tile through a PIM access";
  let arguments = (ins AnyRankedTensor:$value, PIM_AccessType:$access);
  let assemblyFormat = "$value `,` $access attr-dict `:` type($value) `,` type($access)";
}

// PIM Device 하나가 하는 일을 담는 region. region 밖 = host(데이터 준비·배포·회수),
// region 안 = device 하나. block argument 는 이 device 의 partition 좌표다(axes 순서).
// axis_extents 는 나누기 전의 축별 logical tile 개수로, 배치 pass 가 여기서 device 하나의
// 배정량을 다시 계산한다.
def PIM_DeviceOp : PIM_Op<"device", [NoTerminator]> {
  let summary = "Per-device region: the body is what one PIM Device executes";
  let arguments = (ins PIM_DeviceMappingAttr:$mapping,
                       Variadic<Index>:$axis_extents);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $mapping `extents` `[` $axis_extents `]` attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

// 버퍼 하나가 이 PIM Device 에 상주해야 하는 영역. 최종 값을 직접 싣는다.
//   start    = 이 device 의 첫 원소 (balanced block 이 낸 값 그대로)
//   size     = 실제로 맡는 원소 수. 경계 tile 의 유효 범위까지 반영된 값이라
//              언제나 start + size 가 원본 버퍼 안이다.
//   capacity = device 에 배정된 크기. device 불변이라 버퍼 크기가 여기서 나온다.
//   strides  = 원본 버퍼 안의 축별 간격
// direction: "in"(host→device) | "out"(device→host).
// access: 이 영역을 유도한 pim.access 의 id.
// replicated_over: 이 축만 다른 device 들은 같은 데이터를 받는다.
def PIM_PlaceOp : PIM_Op<"place", [AttrSizedOperandSegments]> {
  let summary = "Final placement of one buffer on one PIM Device";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$buffer,
                       Variadic<Index>:$start, Variadic<Index>:$size,
                       Variadic<Index>:$capacity, Variadic<Index>:$strides,
                       StrAttr:$direction, I64Attr:$access,
                       PIM_PlaceMappingAttr:$mapping,
                       OptionalAttr<StrArrayAttr>:$replicated_over);
  let assemblyFormat = [{
    $buffer `start` `[` $start `]` `size` `[` $size `]`
    `capacity` `[` $capacity `]` `strides` `[` $strides `]`
    attr-dict `:` type($buffer)
  }];
  let hasVerifier = 1;
}

#endif // PIM_DIALECT
