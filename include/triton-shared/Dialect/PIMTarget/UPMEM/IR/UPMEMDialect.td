#ifndef UPMEM_DIALECT
#define UPMEM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// UPMEM dialect — UPMEM DPU 실행 모델.
// host_func / device_func로 host-device를 나누고, host는 DPU alloc/load/copy/launch,
// device_func는 본문(scf/linalg/memref, faithful-body)을 담는다. host C++ / DPU C codegen의 입력이다.
//===----------------------------------------------------------------------===//

def UPMEM_Dialect : Dialect {
  let name = "upmem";
  let cppNamespace = "::mlir::upmem";

  let summary = "UPMEM DPU execution dialect";

  let useDefaultTypePrinterParser = 1;
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
// Types
//===----------------------------------------------------------------------===//

// 할당된 DPU 집합 핸들. alloc_units가 만들고 copy/launch가 참조한다.
def UPMEM_DPUSetType : UPMEM_Type<"DPUSet", "dpu_set"> {
  let summary = "Handle to an allocated set of DPUs";
}

//===----------------------------------------------------------------------===//
// Region ops
//===----------------------------------------------------------------------===//

// host 측 실행 시퀀스(alloc/load/copy/launch)를 담는 region. host C++로 codegen된다.
def UPMEM_HostFuncOp : UPMEM_Op<"host_func", [NoTerminator]> {
  let summary = "Host-side launch sequence";
  let arguments = (ins StrAttr:$sym_name);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

// device 측 커널 본문(scf/linalg/memref, faithful-body)을 담는 region. region block arg는
// 유닛이 보는 로컬 MRAM 슬라이스(A/B/C). UPMEMToDevice가 이 본문에서 BLOCK·구조를 읽어 DPU C로 번역한다.
def UPMEM_DeviceFuncOp : UPMEM_Op<"device_func", [NoTerminator]> {
  let summary = "Device-side DPU kernel body (faithful-body region)";
  let arguments = (ins StrAttr:$sym_name);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

//===----------------------------------------------------------------------===//
// Host-side leaf ops
//===----------------------------------------------------------------------===//

// dpu_alloc(num_units). 결과 dpu_set 핸들을 반환.
// num_units = split_m * split_n * split_k (= 총 DPU 수). split_m/n은 출력 2D 분배, split_k는 K분담.
// host가 DPU index를 (pm,pn,kj)로 매핑해 A/B 슬라이스·C 위치·K-reduce 그룹을 정하는 데 쓴다.
def UPMEM_AllocUnitsOp : UPMEM_Op<"alloc_units"> {
  let summary = "Allocate a set of DPUs";
  let arguments = (ins I64Attr:$num_units, I64Attr:$split_m, I64Attr:$split_n, I64Attr:$split_k);
  let results = (outs UPMEM_DPUSetType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

// dpu_load(set, kernel). kernel은 device_func 심볼 이름.
def UPMEM_LoadProgramOp : UPMEM_Op<"load_program"> {
  let summary = "Load a DPU program";
  let arguments = (ins UPMEM_DPUSetType:$dpu_set, FlatSymbolRefAttr:$kernel);
  let assemblyFormat = "$dpu_set `,` $kernel attr-dict `:` type($dpu_set)";
}

// host buffer를 MRAM으로 복사. dst는 MRAM 심볼 이름.
def UPMEM_CopyToGlobalOp : UPMEM_Op<"copy_to_global"> {
  let summary = "Copy a host buffer into MRAM";
  let arguments = (ins UPMEM_DPUSetType:$dpu_set, AnyType:$buffer, StrAttr:$dst);
  let assemblyFormat = "$dpu_set `,` $buffer attr-dict `:` type($dpu_set) `,` type($buffer)";
}

// MRAM에서 host buffer로 복사. src는 MRAM 심볼 이름.
def UPMEM_CopyFromGlobalOp : UPMEM_Op<"copy_from_global"> {
  let summary = "Copy MRAM back to a host buffer";
  let arguments = (ins UPMEM_DPUSetType:$dpu_set, AnyType:$buffer, StrAttr:$src);
  let assemblyFormat = "$dpu_set `,` $buffer attr-dict `:` type($dpu_set) `,` type($buffer)";
}

// dpu_launch.
def UPMEM_LaunchOp : UPMEM_Op<"launch"> {
  let summary = "Launch the loaded DPU program";
  let arguments = (ins UPMEM_DPUSetType:$dpu_set);
  let assemblyFormat = "$dpu_set attr-dict `:` type($dpu_set)";
}

#endif // UPMEM_DIALECT
