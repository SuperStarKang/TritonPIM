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
// PIM IR의 유닛 형태(pim.units + pim.unit_data)를 UPMEM 하드웨어에 맞춘 것이다.
// PIM IR이 답하지 않는 것을 여기서 정한다:
//   - 메모리 계층 (host DRAM → MRAM → WRAM 3단)
//   - 계층 사이 전송 단위 (DMA: 8 byte 정렬, 한 번에 2048 byte 이하, 연속 영역만)
//   - 유닛 내부 병렬 (tasklet: 독립 스레드, barrier로 동기화)
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

// 유닛의 MRAM 버퍼. device에서는 전역 배열(__mram_noinit)이고 host에서는 전송 대상
// 이름이다. 크기는 유닛 하나가 받는 최대 영역이며, push_xfer가 모든 DPU에 같은 바이트
// 수를 요구하므로 덜 맡는 유닛도 이 크기로 잡는다.
def UPMEM_MRAMOp : UPMEM_Op<"mram", [Symbol]> {
  let summary = "MRAM buffer shared by host (by name) and device (as a global)";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyMemRef>:$type);
  let assemblyFormat = "$sym_name attr-dict `:` $type";
  let hasVerifier = 1;
}

// host가 값을 넣어 주는 device 스칼라(__host). 유닛 좌표, 유닛이 실제로 도는 횟수,
// 문제 크기 등 device가 알아야 하는 수가 여기로 온다.
def UPMEM_HostVarOp : UPMEM_Op<"host_var", [Symbol]> {
  let summary = "Device scalar the host writes before launch";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyType>:$type);
  let assemblyFormat = "$sym_name attr-dict `:` $type";
}

//===----------------------------------------------------------------------===//
// Host 쪽
//===----------------------------------------------------------------------===//

// dpu_alloc. num_units = pim.units counts의 곱.
def UPMEM_AllocOp : UPMEM_Op<"alloc"> {
  let summary = "Allocate a set of DPUs";
  let arguments = (ins I64Attr:$num_units);
  let results = (outs UPMEM_SetType:$set);
  let assemblyFormat = "$num_units attr-dict `:` type($set)";
}

// dpu_load. kernel은 upmem.device_func 심볼.
def UPMEM_LoadProgramOp : UPMEM_Op<"load_program"> {
  let summary = "Load the device program into every DPU of the set";
  let arguments = (ins UPMEM_SetType:$set, FlatSymbolRefAttr:$kernel);
  let assemblyFormat = "$set `,` $kernel attr-dict `:` type($set)";
}

// dpu_broadcast_to. 모든 DPU에 같은 값을 넣는다.
def UPMEM_BroadcastOp : UPMEM_Op<"broadcast"> {
  let summary = "Write the same scalar into every DPU";
  let arguments = (ins UPMEM_SetType:$set, FlatSymbolRefAttr:$var, AnyType:$value);
  let assemblyFormat =
      "$set `,` $var `,` $value attr-dict `:` type($set) `,` type($value)";
}

// DPU_FOREACH. 몸체가 유닛 하나에 대한 host 작업(영역 모으기 + MRAM 전송)이다.
// block argument는 그 DPU 핸들 다음 축별 좌표다(axes 순서).
// counts는 pim.units의 것을 그대로 옮긴 값이고, 곱이 alloc의 num_units와 같아야 한다.
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
//   origin/sizes/strides = 원본 안의 영역 (pim.unit_data에서 그대로 온다)
//   pad_to               = 결과 버퍼 shape. sizes보다 크면 남는 자리를 0으로 채운다.
//                          덜 맡는 마지막 유닛이 여기서 균일 크기로 맞춰진다.
//   dedup_over           = 이 축만 다른 유닛은 같은 결과를 받는다. 그 축을 뺀 조합
//                          개수만큼만 실제로 모으면 된다(pim.unit_data replicated_over).
def UPMEM_StageOp : UPMEM_Op<"stage", [AttrSizedOperandSegments]> {
  let summary = "Gather a strided region of a host buffer into a padded contiguous buffer";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$src,
                       Variadic<Index>:$origin, Variadic<Index>:$sizes,
                       Variadic<Index>:$strides, Variadic<Index>:$pad_to,
                       OptionalAttr<StrArrayAttr>:$dedup_over);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $src `origin` `[` $origin `]` `sizes` `[` $sizes `]`
    `strides` `[` $strides `]` `pad_to` `[` $pad_to `]`
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
// broadcast와 나누는 이유: 유닛이 실제로 도는 횟수는 마지막 유닛에서 작아지므로
// 모든 DPU에 같은 값을 보낼 수 없다.
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

// dpu_free. 확보한 유닛을 놓는다. alloc과 짝이며, 없으면 부를 때마다 유닛이 샌다.
def UPMEM_FreeOp : UPMEM_Op<"free"> {
  let summary = "Release the allocated set of DPUs";
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

// 출력 그룹 위를 돈다. 그룹 하나는 결합 축만 다른 유닛들의 묶음이다.
// reduce_axis/combine/reduce_count가 있으면 그 묶음을 먼저 합친 결과를 몸체에 준다.
// 없으면(축약 축을 안 쪼갠 경우) 유닛 하나의 버퍼가 그대로 온다.
// block argument는 합쳐진 버퍼 다음 축별 좌표다.
def UPMEM_ForeachGroupOp : UPMEM_Op<"foreach_group", [NoTerminator]> {
  let summary = "Host loop over output groups, reducing partial results first";
  let arguments = (ins AnyMemRef:$parts, StrArrayAttr:$axes,
                       DenseI64ArrayAttr:$counts,
                       OptionalAttr<StrAttr>:$reduce_axis,
                       DefaultValuedAttr<I64Attr, "1">:$reduce_count,
                       OptionalAttr<StrAttr>:$combine);
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
def UPMEM_UnstageOp : UPMEM_Op<"unstage", [AttrSizedOperandSegments]> {
  let summary = "Scatter a contiguous buffer back into a strided region";
  let arguments = (ins AnyMemRef:$buffer, AnyRankedOrUnrankedMemRef:$dst,
                       Variadic<Index>:$origin, Variadic<Index>:$sizes,
                       Variadic<Index>:$strides, Variadic<Index>:$capacity);
  let assemblyFormat = [{
    $buffer `into` $dst `origin` `[` $origin `]` `sizes` `[` $sizes `]`
    `strides` `[` $strides `]` `capacity` `[` $capacity `]`
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

// barrier_wait. tasklet들이 공유 WRAM 버퍼를 함께 채우거나 읽을 때 필요하다.
def UPMEM_BarrierOp : UPMEM_Op<"barrier"> {
  let summary = "Synchronize all tasklets on this DPU";
  let assemblyFormat = "attr-dict";
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
