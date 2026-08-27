#ifndef AIMX_DIALECT
#define AIMX_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AiMX dialect — silent-PIM FPGA 실행 모델.
//
// PIM Common IR(pim.device + pim.place)을 AiMX 하드웨어에 맞춘 것이다. PIM IR이
// 답하지 않는 것을 여기서 정한다:
//   - 메모리 계층 (host DRAM → PIM DRAM 2단. core 안의 빠른 메모리가 없다)
//   - PIM DRAM 안의 layout (행 pitch 로 규정된 2차원 배열. tile-major 가 아니다)
//   - execution tile 과 padding
//   - A 재배치 (vecA 가 reduction 방향으로 늘어선 A 를 원한다)
//
// UPMEM 과 구조적으로 다른 점: AiMX 에는 device program 이 없다. instruction fetch 를
// 하는 processor 가 없고 descriptor chain 자체가 프로그램이다. 그래서
// upmem.device_func 에 대응하는 것이 없고, 대신 descriptor 를 IR 안에서 직접 표현하는
// 두 번째 층(aimx.descriptor_program + aimx.desc)이 생긴다.
//
// codegen은 이 IR을 그대로 옮긴다. op의 파라미터가 나올 코드를 유일하게 정해야 하며,
// codegen이 스스로 정하는 것이 있으면 그 결정은 여기 없는 것이다.
//===----------------------------------------------------------------------===//

def AiMX_Dialect : Dialect {
  let name = "aimx";
  let cppNamespace = "::mlir::aimx";

  let summary = "AiMX silent-PIM execution dialect";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

class AiMX_Op<string mnemonic, list<Trait> traits = []>
    : Op<AiMX_Dialect, mnemonic, traits>;

class AiMX_Type<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<AiMX_Dialect, name, traits> {
  // 파라미터 이름이 TypeDef의 mnemonic 필드와 충돌하지 않도록 _mnemonic 사용.
  let mnemonic = _mnemonic;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// 확보한 FPGA 집합. host 쪽 op들이 이것을 주고받는다.
def AiMX_DeviceSetType : AiMX_Type<"DeviceSet", "device_set"> {
  let summary = "Handle to an allocated set of AiMX FPGAs";
}

// 집합 안의 FPGA 하나. aimx.foreach_device 몸체에서만 생긴다.
def AiMX_DeviceType : AiMX_Type<"Device", "device"> {
  let summary = "Handle to one FPGA inside a foreach_device body";
}

// PIM DRAM 할당 하나. descriptor 주소가 전부 byte 단위라 원소 타입이 없다 —
// 원소 타입은 그 buffer 를 채우는 aimx.stage / aimx.unstage 가 들고 있다.
def AiMX_BufferType : AiMX_Type<"Buffer", "buffer"> {
  let summary = "Handle to one PIM DRAM allocation (raw bytes)";
}

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

// 할당의 성격. 셋을 구분하는 이유는 runtime 호출이 서로 다르기 때문이다:
// pmm 은 pim_private_malloc 이고 program 안에서 가장 먼저 나와야 하며,
// descriptor 는 chain 이 자기 물리 주소를 싣기 때문에 launch 내내 free 하면 안 된다.
def AiMX_KindData       : I32EnumAttrCase<"Data", 0, "data">;
def AiMX_KindDescriptor : I32EnumAttrCase<"Descriptor", 1, "descriptor">;
def AiMX_KindPMM        : I32EnumAttrCase<"PMM", 2, "pmm">;
def AiMX_AllocKind
    : I32EnumAttr<"AllocKind", "AiMX PIM DRAM allocation kind",
                  [AiMX_KindData, AiMX_KindDescriptor, AiMX_KindPMM]> {
  let cppNamespace = "::mlir::aimx";
  let genSpecializedAttr = 0;
}
def AiMX_AllocKindAttr : EnumAttr<AiMX_Dialect, AiMX_AllocKind, "kind">;

// descriptor 의 status word(CDMA reserved, CDMA_DESC_INFO 0x24)에 실리는 flag.
// memory controller 는 평범한 DMA descriptor 로 보고 PIM logic 이 이 flag 를 보고
// 동작한다 — 이것이 silent 의 뜻이다. 비트 위치는 pim_math.h 의 OPCODE_* 그대로다.
//
// batch 비트(15/16/17)는 여기 없다. batch 는 aimx.descriptor_program 의 attribute 이고
// 그 비트는 codegen 이 유일하게 유도한다. 두 곳에 두면 서로 어긋날 수 있다.
def AiMX_OpDescrDstMask : I32BitEnumAttrCaseBit<"descr_dst_mask", 0>;
def AiMX_OpAMask        : I32BitEnumAttrCaseBit<"a_mask", 1>;
def AiMX_OpBMask        : I32BitEnumAttrCaseBit<"b_mask", 2>;
def AiMX_OpCMask        : I32BitEnumAttrCaseBit<"c_mask", 3>;
def AiMX_OpMatmul       : I32BitEnumAttrCaseBit<"matmul", 4>;
def AiMX_OpAdd          : I32BitEnumAttrCaseBit<"add", 5>;
def AiMX_OpSub          : I32BitEnumAttrCaseBit<"sub", 6>;
def AiMX_OpMul          : I32BitEnumAttrCaseBit<"mul", 7>;
def AiMX_OpDecoupled    : I32BitEnumAttrCaseBit<"decoupled", 8>;
def AiMX_OpFusion       : I32BitEnumAttrCaseBit<"fusion", 9>;
def AiMX_OpBC           : I32BitEnumAttrCaseBit<"bc", 10>;
def AiMX_OpCL           : I32BitEnumAttrCaseBit<"cl", 11>;
def AiMX_OpPimMode      : I32BitEnumAttrCaseBit<"pim_mode", 12>;
def AiMX_OpTP           : I32BitEnumAttrCaseBit<"tp", 13>;
def AiMX_OpHold         : I32BitEnumAttrCaseBit<"hold", 14>;
def AiMX_OpStride       : I32BitEnumAttrCaseBit<"stride", 18>;
def AiMX_OpLUT          : I32BitEnumAttrCaseBit<"lut", 19>;

def AiMX_Opcode
    : I32BitEnumAttr<"Opcode", "AiMX descriptor status flags",
                     [AiMX_OpDescrDstMask, AiMX_OpAMask, AiMX_OpBMask,
                      AiMX_OpCMask, AiMX_OpMatmul, AiMX_OpAdd, AiMX_OpSub,
                      AiMX_OpMul, AiMX_OpDecoupled, AiMX_OpFusion, AiMX_OpBC,
                      AiMX_OpCL, AiMX_OpPimMode, AiMX_OpTP, AiMX_OpHold,
                      AiMX_OpStride, AiMX_OpLUT]> {
  let cppNamespace = "::mlir::aimx";
  let genSpecializedAttr = 0;
}
def AiMX_OpcodeAttr : EnumAttr<AiMX_Dialect, AiMX_Opcode, "opcode">;

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// --convert-pim-to-aimx 가 고른 execution tile. --aimx-lower-matmul 이 이 값만 보고
// descriptor program 을 펼치므로, tile 선택이라는 결정이 IR 에 한 번만 적힌다.
//   batch     output block 을 batch x (4096/batch) 로 자르는 값. {1,2,4,8}
//   t_m/t_n/t_k   padding이 반영된 물리 execution tile 크기
//   padded_k  A 의 행 pitch (element). CHUNK = 4096 의 배수
//   padded_n  B 와 C 의 행 pitch (element). 둘이 같아야 한다
def AiMX_TilingAttr : AttrDef<AiMX_Dialect, "Tiling"> {
  let mnemonic = "tiling";
  let summary = "Execution tile chosen by --convert-pim-to-aimx";
  let parameters = (ins "int64_t":$batch, "int64_t":$t_m, "int64_t":$t_n,
                        "int64_t":$t_k, "int64_t":$padded_k,
                        "int64_t":$padded_n, "bool":$transposed);
  let assemblyFormat = "`<` struct(params) `>`";
}

def AiMX_TileGroupAttr : AttrDef<AiMX_Dialect, "TileGroup"> {
  let mnemonic = "tile_group";
  let summary = "How Triton blocks fill one block group of the hardware grid";
  let description = [{
    `#aimx.tiling` 은 **잡은 buffer 의 모양**이고, 이 attribute 는 **source 의 Triton
    block 과 하드웨어 실행 눈금의 관계**다. 둘을 나누는 이유는 소비자가 다르기
    때문이다 — buffer 모양은 runtime stub 이 쓰고, 이 관계는 legality 와 device 분할이
    쓴다.

    계층은 **Triton block → block group → execution tile** 이다. block group 은
    Triton block 정수 개를 하드웨어 실행 단위(output block 높이 `batch`, output block
    폭 `4096/batch`, reduction step `256/batch`)의 정수 배에 채운 것이다. 딱 맞지
    않으면 남는 자리가 padding 이다 — stage 가 0 을 채우고 unstage 가 읽지 않으므로
    정확성에는 무해하고, 그 몫의 lane 과 reduction step 이 노는 것이 비용이다.

    `block` 은 `linalg.matmul` 의 operand 모양에서 읽은 `[BLOCK_M, BLOCK_N, BLOCK_K]` 다.

    `group` 은 block group 하나가 담는 Triton block 수 `[m, n, k]` 다.

    `execution` 은 block group 의 **물리 크기**다. 세 축 모두 같은 규칙이고 값은
    공통 `#pim.block_group`의 useful 크기를 AiMX 실행 눈금에 맞춰 padding한 값이다:
    `execution[i] = roundUp(group[i] * block[i], unit[i])`, 단위는
    `[batch, 4096/batch, 256/batch]`. padding 은
    `execution[i] - block[i] * group[i]` 로 유도된다 — 따로 싣지 않는다.

    execution tile은 이 block group의 정수 배이고 `#aimx.tiling`의 t_m/t_n/t_k에
    compile time 고정된다. tail round의 실제 유효 크기는 stage/unstage operand가
    표현하며 descriptor chain 모양은 바뀌지 않는다.
  }];
  // DenseI64ArrayAttr 이어야 `[a, b, c]` 로 출력되고 다시 읽힌다. ArrayRefParameter 는
  // struct(params) 안에서 대괄호 없이 나가 round-trip 이 깨진다 — 도구 사이를 MLIR
  // 텍스트로 잇고 있으므로 그러면 파이프라인이 통째로 멈춘다. #pim.target_profile 의
  // target lowering에서 계산한 고정 tiling 값도 같은 이유로 이 타입이다.
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$block,
                        "::mlir::DenseI64ArrayAttr":$group,
                        "::mlir::DenseI64ArrayAttr":$execution);
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Level 1 — host op
//===----------------------------------------------------------------------===//

// PIM Common의 target-neutral compute를 L1에서 잃지 않기 위한 compiler-only
// template이다. body의 block argument는 원래 linalg op의 operand와 같은
// 순서·type이고, body에는 linalg.matmul/matvec/generic 같은 의미 연산이
// 하나 든다. L1은 이 template을 보존하고 L2 descriptor lowering이 소비한
// 뒤 지운다. 따라서 최종 translator가 linalg를 host 연산으로 실행하지
// 않으면서도, operation kind·operand 순서·scalar body를 attribute로 다시
// 인코딩하지 않고 L2까지 유지한다.
def AiMX_ComputeTemplateOp
    : AiMX_Op<"compute_template", [NoTerminator, Symbol, IsolatedFromAbove]> {
  let summary = "Target compute semantics retained between AiMX L1 and L2";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kind);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name `kind` $kind attr-dict-with-keyword $body";
}

// FPGA 를 확보한다. UPMEM 과 달리 적재할 device program 이 없으므로 커널 심볼을 받지
// 않는다. 짝이 되는 반납 op 이 없는 것은 upmem.acquire 와 같은 이유다 — 다음 호출이
// 다시 쓰므로 놓을 자리가 프로세스 끝뿐이고 그것은 런타임이 한다.
def AiMX_AcquireOp : AiMX_Op<"acquire"> {
  let summary = "Reserve a set of AiMX FPGAs";
  let arguments = (ins I64Attr:$num_devices);
  let results = (outs AiMX_DeviceSetType:$set);
  let assemblyFormat = "$num_devices attr-dict `:` type($set)";
}

// 집합의 FPGA 위를 돈다. 몸체가 FPGA 하나에 대한 host 작업이다.
// block argument 는 그 FPGA 핸들 다음 축별 좌표다(axes 순서).
// counts 는 #pim.device_mapping 의 partition 수를 그대로 옮긴 값이고, 곱이
// acquire 의 num_devices 와 같아야 한다.
def AiMX_ForeachDeviceOp : AiMX_Op<"foreach_device", [NoTerminator]> {
  let summary = "Host loop over the FPGAs of a set";
  let arguments = (ins AiMX_DeviceSetType:$set, StrArrayAttr:$axes,
                       DenseI64ArrayAttr:$counts);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $set `:` type($set) `axes` $axes `counts` $counts
    attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

// PIM DRAM 할당 하나. 크기가 compile time 상수인 이유는 pim_malloc 이 실패해도 NULL 을
// 돌려주지 않고 프로세스를 죽이기 때문이다 — 할당해 보고 물러서는 설계가 성립하지 않는다.
def AiMX_AllocOp : AiMX_Op<"alloc"> {
  let summary = "Allocate one PIM DRAM region on a device";
  let arguments = (ins AiMX_DeviceType:$device, I64Attr:$bytes,
                       I64Attr:$alignment, AiMX_AllocKindAttr:$kind);
  let results = (outs AiMX_BufferType:$buffer);
  let assemblyFormat = [{
    $device `bytes` $bytes `align` $alignment `kind` $kind
    attr-dict `:` type($buffer)
  }];
  let hasVerifier = 1;
}

// 원본 버퍼의 strided 영역 마지막 두 축을 PIM DRAM의 2차원 배열로 모은다. 앞축은
// split-K output plane처럼 독립된 2-D slice를 선택하며 PIM layout에는 들어가지 않는다.
// upmem.stage 와 세 가지가 다르다.
//   1. pitch 를 받는다. descriptor 의 stride 필드가 행 간격을 읽으므로 PIM DRAM 안의
//      layout 이 pitch 로 규정된 2차원 배열이다 (UPMEM 은 tile-major 연속 배열).
//      A 는 padded_k, B 와 C 는 padded_n 이고 B 와 C 는 서로 같아야 한다.
//   2. element_type 으로 원소 타입을 바꾼다. host 가 f32 이고 PIM 이 bf16 이다.
//   3. transpose 를 받는다. C = A*B 를 C^T = B^T*A^T 로 도는 선택이 여기서 이루어진다.
// pad_to 는 결과의 행/열 수다. sizes 보다 크면 남는 자리를 0 으로 채운다 — 그 0 이
// 누산에 그대로 더해지므로 padding 은 성능 최적화가 아니라 정확성 요구사항이다.
//
// group_rows / group_cols 는 block group 의 주기 배치다 (`[useful, phys]`,
// useful < phys). 논리 index i 가 물리 칸 `(i/useful)*phys + i%useful` 로 가고,
// 각 phys 구간의 나머지 `phys - useful` 칸은 0 이다 — Triton block 이 하드웨어
// 단위를 딱 채우지 못할 때 그 어긋남이 이 0 이 된다 (#aimx.tile_group). padding 이
// 없는 축은 attr 를 아예 싣지 않는다. sizes 와 origin 은 논리(유효) 단위이고
// pad_to 와 pitch 는 물리 단위다.
def AiMX_StageOp : AiMX_Op<"stage", [AttrSizedOperandSegments]> {
  let summary = "Gather a strided host region into a pitched PIM DRAM array";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$src,
                       Variadic<Index>:$origin, Variadic<Index>:$sizes,
                       Variadic<Index>:$strides, Variadic<Index>:$pad_to,
                       Index:$pitch, AiMX_BufferType:$dst,
                       TypeAttr:$element_type, UnitAttr:$transpose,
                       OptionalAttr<DenseI64ArrayAttr>:$group_rows,
                       OptionalAttr<DenseI64ArrayAttr>:$group_cols);
  let assemblyFormat = [{
    $src `origin` `[` $origin `]` `sizes` `[` $sizes `]`
    `strides` `[` $strides `]` `pad_to` `[` $pad_to `]` `pitch` $pitch
    (`group_rows` $group_rows^)? (`group_cols` $group_cols^)?
    (`transpose` $transpose^)? `into` $dst
    attr-dict `:` type($src) `->` type($dst)
  }];
  let hasVerifier = 1;
}

// aimx.stage의 반대. PIM DRAM의 2차원 배열을 원본 버퍼 마지막 두 축으로 되돌린다.
// 앞축이 있으면 origin/stride가 선택한 독립 slice에 쓴다.
// 읽는 범위는 sizes 이므로 padding 자리는 읽지 않는다.
//
// accumulate 는 런타임 값이고 0 이 아니면 대입 대신 덧셈이다. k 를 여러 execution tile 로
// 나눌 때만 0 이 아니다 — m 이나 n 을 나누면 tile 마다 C 의 다른 영역을 쓰지만, k tile 은
// **전부 같은 C 영역**에 쓰기 때문이다. 첫 tile 이 대입인 이유는 host 의 C 가 0 이라는
// 보장이 없기 때문이다(Triton kernel 의 출력 포인터다).
//
// attribute 가 아니라 operand 인 이유는 같은 unstage op 하나가 k loop 안에 있고 첫 반복과
// 나머지가 달라야 해서다. compile time 에는 어느 쪽인지 알 수 없다.
//
// 이 경로는 k 가 buffer 상한을 넘을 때만 쓰인다. 넘지 않으면 k 전체가 한 chain 에 들어가고
// reduction 은 vACC 안에서 끝난다 — vACC 는 출력 block 하나 깊이라 invoke 를 건너 살아남지
// 못하므로, k 를 나누는 순간 부분합은 반드시 PIM 밖으로 나온다.
//
// group_rows / group_cols 는 aimx.stage 와 같은 block group 주기 배치다. C 의
// padding 자리는 계산 결과 자체가 0 이고(0 인 A 행·0 인 B 열의 곱) 여기서 읽지
// 않는다 — 행은 m group, 열은 n group 의 어긋남이다.
def AiMX_UnstageOp : AiMX_Op<"unstage", [AttrSizedOperandSegments]> {
  let summary = "Scatter a pitched PIM DRAM array back into a strided host region";
  let arguments = (ins AiMX_BufferType:$src,
                       AnyRankedOrUnrankedMemRef:$dst,
                       Variadic<Index>:$origin, Variadic<Index>:$sizes,
                       Variadic<Index>:$strides, Index:$pitch,
                       Index:$accumulate,
                       TypeAttr:$element_type, UnitAttr:$transpose,
                       OptionalAttr<DenseI64ArrayAttr>:$group_rows,
                       OptionalAttr<DenseI64ArrayAttr>:$group_cols);
  let assemblyFormat = [{
    $src `into` $dst `origin` `[` $origin `]` `sizes` `[` $sizes `]`
    `strides` `[` $strides `]` `pitch` $pitch `accumulate` $accumulate
    (`group_rows` $group_rows^)? (`group_cols` $group_cols^)?
    (`transpose` $transpose^)?
    attr-dict `:` type($src) `,` type($dst)
  }];
  let hasVerifier = 1;
}

// descriptor program 하나를 chain 에 펼친다. tile loop **밖**에 두는 것이 재사용을 IR 에
// 드러내는 방식이다 — tile 모양과 base 주소가 모두 같으므로 chain 이 모든 tile 에서
// byte 단위로 동일하다.
// 같은 chain 에 여러 번 펼칠 수 있고, 나오는 순서가 곧 chain 안의 순서다
// (재배치 program 다음에 계산 program).
def AiMX_BuildDescriptorsOp : AiMX_Op<"build_descriptors"> {
  let summary = "Expand a descriptor program into a descriptor chain";
  let arguments = (ins FlatSymbolRefAttr:$program, Variadic<AnyType>:$args,
                       AiMX_BufferType:$chain);
  let assemblyFormat = [{
    $program `(` $args `)` `into` $chain attr-dict (`:` type($args)^)?
  }];
}

// chain 을 실행하고 끝날 때까지 기다린다 (pim_exec).
def AiMX_InvokeOp : AiMX_Op<"invoke"> {
  let summary = "Run a descriptor chain on one FPGA and wait";
  let arguments = (ins AiMX_DeviceType:$device, AiMX_BufferType:$chain);
  let assemblyFormat = "$device `,` $chain attr-dict";
}

//===----------------------------------------------------------------------===//
// Level 2 — descriptor op
//===----------------------------------------------------------------------===//

// descriptor 하나를 내는 loop nest. 몸체는 scf.for + arith + aimx.desc 다.
// block argument 는 aimx.build_descriptors 의 args 와 순서대로 대응한다.
//
// batch 가 program attribute 인 이유가 둘이다. (1) program 하나 안에서 batch 가 하나여야
// descriptor 의 len 규칙이 성립한다. (2) chain 이 byte 단위로 같아야 tile 마다 다시
// 만들지 않는다. 둘 다 batch 가 loop 안에서 변하면 깨진다.
def AiMX_DescriptorProgramOp
    : AiMX_Op<"descriptor_program", [NoTerminator, Symbol, IsolatedFromAbove]> {
  let summary = "Loop nest that emits a descriptor chain segment";
  let arguments = (ins SymbolNameAttr:$sym_name, I64Attr:$batch);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name `batch` $batch attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

// descriptor group 하나 = record NUM_CHANNELS 개. 1:1 대응이므로 codegen 이 정할 것이
// 남지 않는다.
//
//   (src, src_pitch) --[opcode, batch]--> (dst, dst_pitch),  len byte
//
// src / dst 는 각각 없을 수 있다. 참조 구현이 쓰지 않는 쪽에 0x0 을 넣는 자리이고,
// 그 방향은 PIM logic 내부(vecA / vACC)라 주소가 없다.
// offset 과 pitch 는 byte 단위다. pitch 는 IR 에서 따로 두지만 codegen 이
// addr | (pitch << 33) 으로 합친다 — 물리 주소가 33 bit 라서 그 바로 위에 놓인다.
// len 은 **출력** byte 수다. 입력 크기를 주면 그만큼만 쓰이고 나머지는 이전 내용이 남는다.
def AiMX_DescOp : AiMX_Op<"desc", [AttrSizedOperandSegments]> {
  let summary = "One descriptor group (NUM_CHANNELS identical records)";
  let arguments = (ins Optional<AiMX_BufferType>:$src, Optional<Index>:$src_offset,
                       Optional<Index>:$src_pitch,
                       Optional<AiMX_BufferType>:$dst, Optional<Index>:$dst_offset,
                       Optional<Index>:$dst_pitch,
                       I64Attr:$len, AiMX_OpcodeAttr:$opcode);
  let assemblyFormat = [{
    (`src` `(` $src^ `,` $src_offset `)`)? (`pitch` $src_pitch^)?
    (`dst` `(` $dst^ `,` $dst_offset `)`)? (`pitch` $dst_pitch^)?
    `len` $len `opcode` $opcode attr-dict
  }];
  let hasVerifier = 1;
}

#endif // AIMX_DIALECT
