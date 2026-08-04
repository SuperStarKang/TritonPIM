#ifndef UPMEM_DIALECT
#define UPMEM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// UPMEM dialect — UPMEM DPU 실행 모델.
//
// Placed PIM IR(pim.device + pim.place)을 UPMEM 하드웨어에 맞춘 것이다.
// PIM IR이 답하지 않는 것을 여기서 정한다:
//   - 메모리 계층 (host DRAM → MRAM → WRAM 3단)
//   - 계층 사이 전송 단위 (DMA: 8 byte 정렬, 한 번에 2048 byte 이하, 연속 영역만)
//   - 유닛 내부 병렬 (tasklet: 독립 스레드, 서로 겹치지 않는 타일을 맡는다)
//   - 원본 레이아웃 → 유닛 레이아웃 재배치 (stage/unstage)
//
// codegen은 이 IR을 그대로 옮긴다. op의 파라미터가 나올 코드를 유일하게 정해야 하며,
// codegen이 스스로 정하는 것이 있으면 그 결정은 여기 없는 것이다.
//===----------------------------------------------------------------------===//

def UPMEM_Dialect : Dialect {
  let name = "upmem";
  let cppNamespace = "::mlir::upmem";

  let summary = "UPMEM DPU execution dialect";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

class UPMEM_Op<string mnemonic, list<Trait> traits = []>
    : Op<UPMEM_Dialect, mnemonic, traits>;

class UPMEM_Type<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<UPMEM_Dialect, name, traits> {
  // 파라미터 이름이 TypeDef의 mnemonic 필드와 충돌하지 않도록 _mnemonic 사용.
  let mnemonic = _mnemonic;
}

//===----------------------------------------------------------------------===//
// Memory space — memref에 붙여 어느 메모리인지 나타낸다.
//===----------------------------------------------------------------------===//

// MRAM: 유닛당 64 MB DRAM. DMA로만 접근한다.
// WRAM: 유닛당 64 KB SRAM. 직접 주소 지정이 되고 계산은 여기서만 한다.
def UPMEM_SpaceMRAM : I32EnumAttrCase<"MRAM", 0, "mram">;
def UPMEM_SpaceWRAM : I32EnumAttrCase<"WRAM", 1, "wram">;
def UPMEM_MemorySpace
    : I32EnumAttr<"MemorySpace", "UPMEM unit memory space",
                  [UPMEM_SpaceMRAM, UPMEM_SpaceWRAM]> {
  let cppNamespace = "::mlir::upmem";
  let genSpecializedAttr = 0;
}
def UPMEM_MemorySpaceAttr
    : EnumAttr<UPMEM_Dialect, UPMEM_MemorySpace, "space">;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// 확보한 DPU 집합. host 쪽 op들이 이것을 주고받는다.
def UPMEM_SetType : UPMEM_Type<"Set", "set"> {
  let summary = "Handle to an allocated set of DPUs";
}

// 집합 안의 DPU 하나. upmem.foreach_unit 몸체에서만 생긴다.
def UPMEM_UnitType : UPMEM_Type<"Unit", "unit"> {
  let summary = "Handle to one DPU inside a foreach_unit body";
}

//===----------------------------------------------------------------------===//
// 모듈 수준 선언 — host와 device가 이름으로 공유하는 것들.
//===----------------------------------------------------------------------===//

// device의 MRAM 버퍼. device 코드에서는 전역 배열(__mram_noinit)이고 host에서는 전송 대상
// 이름이다. 크기는 유닛 하나가 받는 최대 영역이며, push_xfer가 모든 DPU에 같은 바이트
// 수를 요구하므로 덜 맡는 유닛도 이 크기로 잡는다.
def UPMEM_MRAMOp : UPMEM_Op<"mram", [Symbol]> {
  let summary = "MRAM buffer shared by host (by name) and device (as a global)";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyMemRef>:$type);
  let assemblyFormat = "$sym_name attr-dict `:` $type";
  let hasVerifier = 1;
}

// host가 값을 넣어 주는 device 스칼라(__host). device가 실제로 도는 횟수, MRAM 안의 축
// 간격, 커널 인자에서 나오는 loop 상한 등 device가 알아야 하는 수가 여기로 온다.
def UPMEM_HostVarOp : UPMEM_Op<"host_var", [Symbol]> {
  let summary = "Device scalar the host writes before launch";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyType>:$type);
  let assemblyFormat = "$sym_name attr-dict `:` $type";
}

//===----------------------------------------------------------------------===//
// Host 쪽
//===----------------------------------------------------------------------===//

// DPU를 확보하고 device 프로그램을 적재한다. num_units = device partition 수의 곱이고
// kernel은 upmem.device_func 심볼이다.
//
// 확보와 적재를 한 op으로 두는 이유: 둘 다 커널을 부를 때마다 다시 할 일이 아니다.
// dpu_alloc은 수백 ms, dpu_load는 수십 ms가 드는데 둘 다 데이터와 무관하다. 그래서
// 런타임이 같은 유닛 수·같은 프로그램이면 쥐고 있던 것을 그대로 준다. 나눠 두면
// "이번에는 확보만 다시" 같은 것을 표현할 수 있는데, 그런 경우가 없다.
//
// 짝이 되는 반납 op이 없는 것도 같은 이유다. 다음 호출이 다시 쓰므로 놓을 자리가
// 프로세스 끝뿐이고, 그것은 런타임이 한다.
def UPMEM_AcquireOp : UPMEM_Op<"acquire"> {
  let summary = "Reserve a set of DPUs with the device program loaded";
  let arguments = (ins I64Attr:$num_units, FlatSymbolRefAttr:$kernel);
  let results = (outs UPMEM_SetType:$set);
  let assemblyFormat = "$num_units `,` $kernel attr-dict `:` type($set)";
}

// DPU_FOREACH. 몸체가 유닛 하나에 대한 host 작업(영역 모으기 + MRAM 전송)이다.
// block argument는 그 DPU 핸들 다음 축별 좌표다(axes 순서).
// counts는 #pim.device_mapping의 partition 수를 그대로 옮긴 값이고, 곱이 alloc의 num_units와 같아야 한다.
def UPMEM_ForeachUnitOp : UPMEM_Op<"foreach_unit", [NoTerminator]> {
  let summary = "Host loop over the DPUs of a set";
  let arguments = (ins UPMEM_SetType:$set, StrArrayAttr:$axes,
                       DenseI64ArrayAttr:$counts);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $set `:` type($set) `axes` $axes `counts` $counts
    attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

// 원본 버퍼의 strided 영역을 연속 버퍼로 모은다. MRAM DMA가 연속 영역만 다루고
// push_xfer가 모든 DPU에 같은 바이트 수를 요구하므로, 이 재배치가 반드시 필요하다.
//   origin/sizes/strides = 원본 안의 영역 (pim.place에서 그대로 온다)
//   pad_to               = 결과 버퍼 shape. sizes보다 크면 남는 자리를 0으로 채운다.
//                          덜 맡는 마지막 유닛이 여기서 균일 크기로 맞춰진다.
//   tile                 = 유닛 버퍼를 타일 단위로 쓴다. 타일 하나가 연속이 되므로
//                          device가 타일을 DMA 한 번으로 옮긴다. 행 단위로 두면 타일의
//                          행 수만큼 DMA가 필요하고, 한 행이 몇십 바이트라 준비 비용이
//                          옮기는 양보다 커진다.
//   dedup_over           = 이 축만 다른 유닛은 같은 결과를 받는다. 그 축을 뺀 조합
//                          개수만큼만 실제로 모으면 된다(pim.place replicated_over).
def UPMEM_StageOp : UPMEM_Op<"stage", [AttrSizedOperandSegments]> {
  let summary = "Gather a strided region of a host buffer into a padded tile-major buffer";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$src,
                       Variadic<Index>:$origin, Variadic<Index>:$sizes,
                       Variadic<Index>:$strides, Variadic<Index>:$pad_to,
                       DenseI64ArrayAttr:$tile,
                       OptionalAttr<StrArrayAttr>:$dedup_over);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $src `origin` `[` $origin `]` `sizes` `[` $sizes `]`
    `strides` `[` $strides `]` `pad_to` `[` $pad_to `]` `tile` $tile
    attr-dict `:` type($src) `->` type($result)
  }];
  let hasVerifier = 1;
}

// dpu_prepare_xfer + dpu_push_xfer(TO_DPU). foreach_unit 몸체 안에서만 쓴다.
def UPMEM_ToMRAMOp : UPMEM_Op<"to_mram"> {
  let summary = "Send a staged buffer to one DPU's MRAM";
  let arguments = (ins UPMEM_UnitType:$unit, AnyMemRef:$buffer,
                       FlatSymbolRefAttr:$dst);
  let assemblyFormat = "$unit `,` $buffer `,` $dst attr-dict `:` type($buffer)";
}

// 유닛마다 다른 스칼라를 그 DPU의 host 변수에 넣는다. foreach_unit 몸체에서만 쓴다.
// 유닛이 실제로 도는 횟수는 맡은 타일 수만큼이라 유닛마다 다르므로 집합 단위 전송으로는
// 보낼 수 없다.
def UPMEM_ToHostVarOp : UPMEM_Op<"to_host_var"> {
  let summary = "Write a per-DPU scalar into that DPU's host variable";
  let arguments = (ins UPMEM_UnitType:$unit, AnyType:$value,
                       FlatSymbolRefAttr:$dst);
  let assemblyFormat = "$unit `,` $value `,` $dst attr-dict `:` type($value)";
}

// dpu_launch(SYNCHRONOUS).
def UPMEM_LaunchOp : UPMEM_Op<"launch"> {
  let summary = "Run the loaded program on every DPU and wait";
  let arguments = (ins UPMEM_SetType:$set);
  let assemblyFormat = "$set attr-dict `:` type($set)";
}

// dpu_push_xfer(FROM_DPU). 집합 전체에서 한 번에 회수하므로 foreach가 필요 없다.
// 결과는 유닛 하나 몫의 shape 앞에 유닛 축이 하나 더 붙은 memref다.
// capacity는 유닛 하나 몫의 축별 크기다. 모든 DPU에서 같은 바이트 수를 읽어야 하므로
// 실제 크기가 아니라 배정량이며, MRAM 심볼의 선언 크기(넉넉히 잡은 상한)와도 다르다.
def UPMEM_GatherOp : UPMEM_Op<"gather"> {
  let summary = "Read one MRAM buffer back from every DPU at once";
  let arguments = (ins UPMEM_SetType:$set, FlatSymbolRefAttr:$src,
                       Variadic<Index>:$capacity);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $set `,` $src `capacity` `[` $capacity `]` attr-dict
    `:` type($set) `->` type($result)
  }];
}

// 출력 device 위를 돈다. 몸체가 device 하나의 버퍼를 원본으로 되돌리는 일이다.
// block argument는 그 버퍼 다음 축별 좌표다.
def UPMEM_ForeachGroupOp : UPMEM_Op<"foreach_group", [NoTerminator]> {
  let summary = "Host loop over the output buffer of each DPU";
  let arguments = (ins AnyMemRef:$parts, StrArrayAttr:$axes,
                       DenseI64ArrayAttr:$counts);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $parts `:` type($parts) `axes` $axes `counts` $counts
    attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

// upmem.stage의 반대. 연속 버퍼를 원본 버퍼의 strided 영역으로 되돌린다.
//   sizes    = 실제로 쓴 영역. 채운 자리는 이 밖이라 읽지 않는다.
//   capacity = 버퍼 자체의 축별 크기(upmem.gather의 것과 같다). 축을 하나 건너뛰는
//              간격이 여기서 나온다. 유닛이 배정량보다 적게 맡으면 sizes와 달라지므로
//              sizes로 간격을 재면 엉뚱한 자리를 읽는다.
//   tile     = upmem.stage와 같은 뜻. 유닛 버퍼가 타일 단위로 쓰여 있으므로 되돌릴 때도
//              같은 순서로 읽어야 한다.
def UPMEM_UnstageOp : UPMEM_Op<"unstage", [AttrSizedOperandSegments]> {
  let summary = "Scatter a tile-major buffer back into a strided region";
  let arguments = (ins AnyMemRef:$buffer, AnyRankedOrUnrankedMemRef:$dst,
                       Variadic<Index>:$origin, Variadic<Index>:$sizes,
                       Variadic<Index>:$strides, Variadic<Index>:$capacity,
                       DenseI64ArrayAttr:$tile);
  let assemblyFormat = [{
    $buffer `into` $dst `origin` `[` $origin `]` `sizes` `[` $sizes `]`
    `strides` `[` $strides `]` `capacity` `[` $capacity `]` `tile` $tile
    attr-dict `:` type($buffer) `,` type($dst)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Device 쪽
//===----------------------------------------------------------------------===//

// DPU 하나가 실행하는 코드. 몸체는 scf/memref/linalg + 아래 op들이다.
// tasklet마다 이 몸체가 처음부터 돈다(C의 main과 같다).
def UPMEM_DeviceFuncOp : UPMEM_Op<"device_func", [NoTerminator, Symbol]> {
  let summary = "Body one DPU executes";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

// me(). 이 tasklet의 번호.
def UPMEM_TaskletIdOp : UPMEM_Op<"tasklet_id", [Pure]> {
  let summary = "Index of the running tasklet";
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

// NR_TASKLETS. 빌드 시점에 -D로 주입되므로 device 코드에서는 상수다.
def UPMEM_NumTaskletsOp : UPMEM_Op<"num_tasklets", [Pure]> {
  let summary = "Number of tasklets on this DPU";
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

// upmem.host_var 읽기. device 코드에서는 전역 변수 참조 하나다.
// 유닛이 실제로 도는 횟수가 이 통로로 온다 — 그래서 device는 자기 좌표를 몰라도
// [0, tiles)만 돌면 되고, 마지막 유닛이 덜 맡는 것도 값 하나로 표현된다.
def UPMEM_GetHostVarOp : UPMEM_Op<"get_host_var", [Pure]> {
  let summary = "Read a scalar the host wrote before launch";
  let arguments = (ins FlatSymbolRefAttr:$var);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$var attr-dict `:` type($result)";
}

// mram_read. MRAM의 연속 구간을 WRAM 버퍼로 옮긴다.
// offset은 원소 단위다. 옮기는 바이트 수는 dst 타입에서 나온다.
// 하드웨어 제약(정렬 8, 최대 2048)을 verifier가 확인한다 — 어기면 런타임에 조용히
// 깨지므로 여기서 막는다. 큰 타일은 여러 번으로 쪼개서 표현해야 한다.
def UPMEM_MRAMReadOp : UPMEM_Op<"mram_read"> {
  let summary = "DMA a contiguous MRAM range into a WRAM buffer";
  let arguments = (ins FlatSymbolRefAttr:$src, Index:$offset, AnyMemRef:$dst);
  let assemblyFormat = "$src `[` $offset `]` `,` $dst attr-dict `:` type($dst)";
  let hasVerifier = 1;
}

// mram_write. 위의 반대 방향.
def UPMEM_MRAMWriteOp : UPMEM_Op<"mram_write"> {
  let summary = "DMA a WRAM buffer into a contiguous MRAM range";
  let arguments = (ins AnyMemRef:$src, FlatSymbolRefAttr:$dst, Index:$offset);
  let assemblyFormat = "$src `,` $dst `[` $offset `]` attr-dict `:` type($src)";
  let hasVerifier = 1;
}

#endif // UPMEM_DIALECT
