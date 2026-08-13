#ifndef PIM_DIALECT
#define PIM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// PIM dialect — target 무관 PIM 공통 IR.
// 공통 구조는 데이터 3계층이다: block(Triton tile 눈금) ⊂ block group(local 상주·
// 계산 단위) ⊂ execution tile(PIM Device 하나가 한 번에 맡는 몫). 기계는 각 계층의
// 소비자로 대응된다 — execution tile 은 PIM Device 가(#pim.device_mapping),
// block group 은 PIM Core 가(현재 두 target 모두 core 1개), group 내부는 lane 이
// 소비한다(#pim.group_execution). target 용어(DPU/tasklet/MRAM/channel/bank 등)는
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
// usable_*_capacity_bytes 는 세 상태를 구분한다: 값이 있으면 알려진 용량, 0 이면 그
// 저장소가 존재하지 않아 검사가 성립하지 않음, 필드가 없으면 미측정이라 그 값을
// 필요로 하는 검사에 도달하는 순간 진단을 낸다.
def PIM_TargetProfileAttr : PIM_Attr<"TargetProfile", "target_profile"> {
  let summary = "Common machine profile shared by every PIM target";
  let parameters = (ins
      "::mlir::StringAttr":$target,
      "int64_t":$number_of_pim_devices,
      "::mlir::DenseI64ArrayAttr":$pim_core_shape,
      "int64_t":$number_of_parallel_lanes_per_pim_core,
      "::mlir::StringAttr":$parallel_execution_kind,
      "::mlir::BoolAttr":$requires_uniform_instruction_sequence,
      OptionalParameter<"::mlir::IntegerAttr">:$usable_bank_capacity_bytes,
      OptionalParameter<"::mlir::IntegerAttr">:$usable_local_capacity_bytes);
  let assemblyFormat = "`<` struct(params) `>`";
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

// block group 하나를 PIM Core 의 lane 들이 어떻게 실행하는가 — lane 수와, lane 이
// group 의 무엇을 나눠 갖는가(whole_tile: 서로 다른 group 을 통째로 / output_row:
// 한 group 을 출력 행으로 협력). 다중 core 로의 group 분배는 그런 target 이 실제로
// 올 때 group 계층의 확장으로 재설계한다 — 그때까지는 core 1개가 전제다.
def PIM_GroupExecutionAttr : PIM_Attr<"GroupExecution", "group_execution"> {
  let summary = "How the lanes of a PIM Core execute one block group";
  let parameters = (ins "int64_t":$lanes,
                        "::mlir::StringAttr":$lane_axis_role);
  let assemblyFormat = "`<` struct(params) `>`";
}

// execution tile 의 상한 (원소 단위) — 데이터 3계층의 셋째 층. device 하나가 한 번에
// 상주시키는 몫의 상한이고, 0 은 그 축을 자르지 않는다는 뜻이다. 공간 분할은
// #pim.device_mapping 이, 시간(창) 분할은 이 상한이 정한다 — 배정이 상한 안이면 창은
// 하나로 퇴화한다. target lowering 이 자기 실행 모델(UPMEM 의 round / AiMX 의
// pim_exec 창)로 옮긴다. --pim-form-execution-tiles 가 pim.device 에 부착한다.
def PIM_ExecutionTileAttr : PIM_Attr<"ExecutionTile", "execution_tile"> {
  let summary = "Element-count caps of one device's resident share";
  let parameters = (ins "int64_t":$m, "int64_t":$n, "int64_t":$k);
  let assemblyFormat = "`<` struct(params) `>`";
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
