# Triton PIMBackend Architecture Design

## 1. Design Goal

Triton PIMBackend는 Triton 내부에서 PIM을 정식 backend로 인식하고, `device="pim"`을 통해 PIM device를 대상으로 kernel을 compile·실행하는 backend이다.

PIMBackend는 `triton-shared` repository 안에 **별도 backend 패키지(`pim_backend/`)** 로 추가된다. 이는 Triton의 표준 backend 등록 방식이다.

Triton은 모든 backend를 `triton/python/triton/backends/<name>/` 디렉토리(symlink)로 등록한다. `nvidia/`, `amd/`, `triton_shared/`(=CPU)가 모두 각자의 `compiler.py` + `driver.py` + `name.conf`를 가진 독립 디렉토리이다. `backends/compiler.py`와 `backends/driver.py`는 **공유 구현이 아니라 추상 base class**(`BaseBackend`, `DriverBase`)이며, 각 backend는 이를 상속한 자신만의 구현을 자기 디렉토리에 둔다.

backend discovery(`backends/__init__.py`의 `_find_concrete_subclasses`)는 각 backend 디렉토리의 `compiler.py`에 `BaseBackend` 서브클래스가 **정확히 하나**만 있을 것을 요구한다. 따라서 PIMBackend도 nvidia/amd/cpu와 똑같이 자신만의 `pim_backend/` 디렉토리를 갖는다. CPUBackend(`triton-shared/backend/`)에 PIMBackend를 넣는 것은 이 규약 위반이므로 불가능하다.

Python측 `pim_backend/compiler.py`는 `triton-shared-opt`/`triton-shared-translate` 도구를 subprocess로 호출하는 thin dispatcher이며, 모든 dialect/conversion/transform 코드는 `triton-shared/include/triton-shared/`, `triton-shared/lib/`에 C++로 구현된다.

PIMBackend는 Triton DSL에서 생성된 TTIR을 TTSharedIR로 lowering한 뒤, hardware-specific execution hierarchy가 제거된 TTSharedIR을 기반으로 PIM Common IR을 생성한다. 이후 PIM Common IR에서 공통 최적화를 수행하고, 최종적으로 UPMEM IR 또는 AiMX IR로 vendor-specific lowering을 진행한다.

최종 사용 형태는 다음과 같다.

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A, B, C,
                  M: tl.constexpr,
                  N: tl.constexpr,
                  K: tl.constexpr,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    ...

matmul_kernel[grid](
    A, B, C,
    M, N, K,
    BLOCK_M=32,
    BLOCK_N=64,
    BLOCK_K=256,
    device="pim",
    vendor="aimx",
)
```

`device="pim"`은 PIMBackend를 선택하는 frontend-level trigger이고, `vendor="upmem"`/`vendor="aimx"`는 `PIMOptions.vendor`로 전달되어 vendor-specific lowering pipeline을 선택한다.

backend 객체를 직접 사용할 수도 있다 (`GPUTarget`은 Triton의 표준 target 타입; backend 이름이 `"pim"`이면 PIMBackend가 dispatch된다).

```python
from triton.backends.compiler import GPUTarget
from triton_shared.pim_backend.compiler import PIMBackend

target = GPUTarget(backend="pim", arch="dpu_v1", warp_size=0)
backend = PIMBackend(target)
```

---

## 2. Top-Level Compilation Flow

```text
Triton Python DSL
    ↓
TTIR
    ↓
TTSharedIR
    ↓
PIM Common IR
    ↓
PIM Common Optimization Passes
    ↓
Vendor Extension IR
    ├── UPMEM IR
    └── AiMX IR
    ↓
Vendor-specific Codegen
    ├── UPMEM Host Code + DPU Kernel
    └── AiMX Descriptor Stream + Runtime Stub
    ↓
PIM Runtime Launch
```

---

## 3. IR Layering Model

Cinnamon 스타일의 계층적 lowering 구조를 사용한다.

```text
TTIR
  ↓
TTSharedIR
  ↓
PIM Common IR
  ↓
Vendor Extension IR
    ├── UPMEM IR
    └── AiMX IR
```

| Layer         | 역할                                                              |
| ------------- | ---------------------------------------------------------------- |
| TTIR          | Triton frontend에서 생성되는 기본 IR                                |
| TTSharedIR    | hardware-specific execution hierarchy를 제거한 tile/block-level IR |
| PIM Common IR | vendor-neutral PIM 공통 실행 모델 (본문 보존 + 분배 구조)            |
| UPMEM IR      | UPMEM DPU/MRAM/WRAM/host-device split 표현                        |
| AiMX IR       | AiMX descriptor/DMA/channel/bank command 표현                     |

---

## 4. Design Principle

PIMBackend는 세 계층으로 구성된다.

```text
1. Triton Shared Layer
   - TTIR
   - TTSharedIR

2. PIM Common Layer
   - PIM Common Dialect / IR
   - PIM Common Optimization Passes

3. Vendor Extension Layer
   - UPMEM Dialect / UPMEM IR / UPMEM Passes / UPMEM Codegen
   - AiMX Dialect / AiMX IR / AiMX Passes / AiMX Codegen
```

PIM Common Layer는 vendor-neutral하다. UPMEM, AiMX, 그리고 미래의 PIM vendor는 PIM Common IR 이후의 lowering만 구현한다.

### 4.1 핵심 원칙 — faithful-body (본문은 보존, 구조는 추가)

TTSharedIR → PIM Common IR 변환은 **계산 본문을 치환하지 않는다.** 두 가지만 한다.

1. **계산 본문을 손실 없이 옮긴다.** TTSharedIR의 `scf`/`memref`/`linalg`/`arith`를 **그대로** PIM Common IR에 둔다. `linalg.matmul`은 `linalg.matmul`로 남는다 — `pim.matmul`로 바꾸지 않는다. 이미 vendor-neutral 표준 dialect이므로 재발명할 이유가 없고, vendor가 충실히 lowering한다.
2. **TTSharedIR에 없던 PIM 분배 구조를 추가로 합성한다.** host/device 구분(`memory_region`)·전송(`transfer`)·분배(`tile_plan`/`execution_group`)는 단일 SPMD 커널엔 없는 정보다. 이것만 `pim.*`로 새로 만들어 본문 **주위에** 붙인다.

```text
TTSharedIR (단일 SPMD)                       PIM Common IR (faithful-body)
  func.func                          →   pim.func {primitive="gemm"}              (속성 부착)
  scf.for / linalg.matmul / memref   →   (그대로, pim.compute region 안)          ① 충실 보존
  %row = pid_m*BLOCK_M (전역 오프셋)  →   tile_plan grid + host scatter            ② 분배층으로 이동
  (없음)                             →   memory_region/transfer/tile_plan/execution_group   ② 추가
```

- **인식 = 속성, 치환 아님**: "이건 gemm이다"는 `primitive="gemm"` 속성으로만 기록한다. 전략 선택에 쓰고 본문은 건드리지 않는다.
- **`pim.compute` = region-carrying**: 본문을 담는 껍데기다. `@gemm` 심볼로 본문을 대체하는 coarse op이 아니다.
- **파생 정보는 본문에서 읽는다**: `BLOCK_*`는 본문 subview 모양, K-loop은 본문 `scf.for`에서 나온다 → K-chunk를 별도 op로 두지 않는다.

**공통 vs vendor 경계**: 위 ② 구조(`memory_region`/`transfer`/`tile_plan`/`execution_group`)는 **UPMEM·AiMX가 공유하는 사실**(host/device 분리 + 명시적 전송 + 출력 분배)이라 vendor-neutral 공통이다. 아키텍처가 갈리는 곳은 **본문 ①의 실현 한 군데뿐**이다.

- **UPMEM**(프로그래머블 코어): 본문을 loops로 **충실 번역**(12장). 임의/fused 커널 가능.
- **AiMX**(고정기능): `primitive`로 **descriptor 선택**(13장). 디스크립터 없는 본문은 fail-fast.

본문을 coarse primitive로 치환하면 gemm 외 연산(gemv/relu/fused)을 표현할 수 없고, "커널을 어떻게 짜느냐가 성능을 좌우한다"는 Triton의 성질(15장)이 깨진다. 본문 보존이 그 성질을 PIM에서 성립시킨다.

---

## 5. Backend Structure

```text
triton.backends
    ├── nvidia
    ├── amd
    ├── cpu
    └── pim
          ├── PIMBackend
          ├── PIMDriver
          ├── PIMRuntime
          ├── PIMCapability
          ├── PIMConstraints
          └── VendorExtensionRegistry
                ├── UPMEMVendorExtension
                └── AiMXVendorExtension
```

새로운 PIM vendor가 추가될 경우 generic 우회 경로 없이, 다음 interface를 구현한 새로운 `VendorExtension`을 등록한다.

```text
VendorExtension
    ├── registerDialect()
    ├── registerPasses()
    ├── buildLoweringPipeline()
    ├── buildOptimizationPipeline()
    ├── buildCodegenPipeline()
    ├── queryCapability()
    ├── queryConstraints()
    └── createRuntimeLauncher()
```

---

## 6. Directory Structure

PIMBackend는 `triton-shared` repository 안에 추가된다. dialect/conversion 코드의 root namespace는 `triton-shared/Dialect/...`이다.

Python backend 코드는 `triton-shared/pim_backend/`에 독립 패키지로 둔다. nvidia/amd/cpu가 각자 별도 디렉토리인 것과 동일한 방식이며, CPUBackend(`triton-shared/backend/`)와 물리적으로 분리되어 "디렉토리당 BaseBackend 서브클래스 1개" 규약을 만족한다. 빌드 시 `TRITON_PLUGIN_DIRS`에 두 경로를 모두 등록한다.

```bash
export TRITON_PLUGIN_DIRS="/path/to/triton-shared;/path/to/triton-shared/pim_backend"
```

```text
triton-shared/
  backend/                          # 기존 CPUBackend 패키지 (변경 없음)
    compiler.py                     #   CPUBackend / CPUOptions — PIM이 _ttir_to_ttsharedir 등을 import해 재사용
    driver.py                       #   CPUDriver / CPULauncher
    name.conf                       #   "triton_shared"

  pim_backend/                      # PIMBackend 전용 독립 패키지 (nvidia/amd와 동일 구조)
    backend/
      compiler.py                   #   PIMOptions + PIMBackend (add_stages로 pim 파이프라인 정의)
      driver.py                     #   PIMDriver (GPUTarget("pim",...) 반환) + PIMLauncher
      name.conf                     #   "pim" — device="pim" 라우팅 키
    include/
      ExecutionEngine/              #   launcher가 쓰는 C 런타임 헤더 (CPU와 공유, symlink)

  include/
    triton-shared/
      Dialect/
        TPtr/                       # 기존 dialect
        TritonStructured/           # 기존 dialect (TTSharedIR의 tts.* op 정의)
        TritonTilingExt/            # 기존 dialect
        PIM/                        # ── PIM Common Dialect (vendor-neutral) ──
          IR/
            PIMDialect.h
            PIMOps.h                #   pim.compute / tile_plan / transfer / execution_group 등
            PIMTypes.h              #   MemoryRegionType 등
            PIMAttrs.h              #   Layout/Residency/Role/MappingPolicy/Space/CapabilityRef enum attr
          Transforms/
            Passes.h
          Interfaces/
            PIMVendorExtension.h    #   vendor 공통 인터페이스 (lowering/codegen/capability 훅)
            PIMTargetInterface.h
            PIMMemoryInterface.h
            PIMExecutionInterface.h

        PIMVendor/                  # ── vendor-specific dialect ──
          UPMEM/
            IR/        UPMEMDialect.h, UPMEMOps.h, UPMEMTypes.h, UPMEMAttrs.h
            Transforms/UPMEMPasses.h
            Runtime/   UPMEMRuntime.h
          AiMX/
            IR/        AiMXDialect.h, AiMXOps.h, AiMXTypes.h, AiMXAttrs.h
            Transforms/AiMXPasses.h
            Runtime/   AiMXRuntime.h

      Conversion/
        TTSharedToPIM/Passes.h      #   TTSharedIR → PIM Common IR
        PIMToUPMEM/Passes.h         #   PIM Common IR → UPMEM IR
        PIMToAiMX/Passes.h          #   PIM Common IR → AiMX IR

  lib/                              # ── 위 선언들의 C++ 구현 ──
    Dialect/
      PIM/
        IR/         PIMDialect.cpp, PIMOps.cpp, PIMTypes.cpp, PIMAttrs.cpp
        Transforms/
          BuildPIMOffload.cpp           #   [핵심] 본문 보존 + memory_region/transfer/
                                        #     tile_plan(grid·SPLIT)/execution_group 합성, primitive 기록
          SelectPIMStrategy.cpp         #   split + capability로 적법 전략 표시
          PIMLayoutNormalize.cpp        #   layout 표현 정규화                 (opt-in)
          PIMDataPlacement.cpp          #   placement 의도 갱신                 (opt-in)
          PIMTransferHoisting.cpp       #   반복 transfer를 loop 밖으로          (opt-in)
          PIMPersistentData.cpp         #   weight/KV 상주 표시                 (opt-in)
          PIMPartialResultPlanning.cpp  #   partial/reduce 구조 (split_k>1)      (opt-in)
          PIMPipelineOverlap.cpp        #   transfer/compute/store overlap       (opt-in)

      PIMVendor/
        UPMEM/
          IR/         UPMEMDialect.cpp, UPMEMOps.cpp, UPMEMTypes.cpp, UPMEMAttrs.cpp
          Transforms/
            LegalizePIMToUPMEM.cpp      #   pim.* → upmem.* (execution_unit→DPU 등)
            UPMEMMemoryPlacement.cpp    #   device_global→MRAM, device_local→WRAM
            UPMEMExecutionMapping.cpp   #   tile → DPU/tasklet 배정
            UPMEMTransferLowering.cpp   #   pim.transfer → host↔MRAM copy
            UPMEMHostDeviceSplit.cpp    #   host_func / device_func 분리
        AiMX/
          IR/         AiMXDialect.cpp, AiMXOps.cpp, AiMXTypes.cpp, AiMXAttrs.cpp
          Transforms/
            LegalizePIMToAiMX.cpp
            AiMXMemoryBankMapping.cpp
            AiMXExecutionMapping.cpp
            AiMXDescriptorSelection.cpp
            AiMXDMAScheduling.cpp

    Target/                          # ── IR → 소스/바이너리 텍스트 생성 (translate) ──
      UPMEM/
        UPMEMToHost.cpp              #   UPMEM IR → host.cpp (multi-DPU scatter/gather)
        UPMEMToDevice.cpp            #   UPMEM IR 본문 → device.c (loops + MRAM↔WRAM DMA)
      AiMX/
        AiMXCommandStreamCodegen.cpp #   descriptor binary stream
        AiMXRuntimeStub.cpp

    Conversion/
      TTSharedToPIM/ ...
      PIMToUPMEM/ ...
      PIMToAiMX/ ...

  tools/
    triton-shared-opt/
      RegisterTritonSharedDialects.h  # 모든 PIM/UPMEM/AiMX dialect·pass 등록
    triton-shared-translate/          # IR → host.cpp / device.c / descriptor stream

  runtime/
    pim/
      include/  pim_runtime.h, pim_device.h, pim_launch.h, pim_executable.h, pim_memory.h
      lib/      PIMRuntime.cpp, PIMDevice.cpp, PIMExecutable.cpp, PIMLaunch.cpp, PIMMemory.cpp
      vendors/
        upmem/  UPMEMRuntime.cpp, UPMEMLauncher.cpp   # dpu_alloc/load/copy/launch
        aimx/   AiMXRuntime.cpp, AiMXLauncher.cpp      # descriptor stream submit + wait
```

CPUBackend와 동일하게 IR 변환은 Python이 아니라 `triton-shared-opt`/`triton-shared-translate` 실행 파일을 subprocess로 호출해 수행한다. 새 PIM dialect/pass는 모두 `RegisterTritonSharedDialects.h`에 등록한다. `pim_backend/backend/compiler.py`는 `triton-shared/backend/compiler.py`의 공유 유틸(`_ttir_to_ttsharedir`, `_get_triton_shared_opt_path` 등)을 import해 재사용한다.

---

## 7. Python Backend API

PIMBackend는 Triton의 `BaseBackend` 규약을 따른다. `BaseBackend.__init__`과 `supports_target`은 `GPUTarget`을 인자로 받는다 (Triton legacy 네이밍 — 실제로는 모든 backend의 공통 Target 타입). PIM 고유 정보(`vendor`, `arch`, `device_id`)는 `PIMOptions`에 둔다.

### 7.1 PIMOptions

```python
from dataclasses import dataclass
from typing import Tuple
import hashlib

@dataclass(frozen=True)
class PIMOptions:
    # Triton 표준 옵션 (CPUOptions와 동일)
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    sanitize_overflow: bool = True

    # PIM 고유 옵션
    vendor: str = None           # "upmem" | "aimx" — 누락 시 명시적 에러
    device_id: int = 0
    # PIM 분배 노브 (커널 본문이 참조하지 않는 노브; option/autotune으로 전달)
    local_lanes: int = 1
    # SPLIT_M/N/K, NUM_UNITS 등은 autotune config 또는 driver default로 전달

    def __post_init__(self):
        if self.vendor is None:
            raise ValueError(
                "PIMBackend requires an explicit `vendor` option. "
                "Supported values: 'upmem', 'aimx'."
            )

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
```

### 7.2 PIMBackend

```python
from triton.backends.compiler import BaseBackend, GPUTarget

class PIMBackend(BaseBackend):
    binary_ext = 'pimobj'   # vendor extension에서 최종 결정

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'pim'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        # vendor extension은 parse_options() 이후 options.vendor를 보고 결정한다.
        # IR 변환은 모두 triton-shared-opt subprocess가 수행하므로
        # backend 객체는 가벼운 dispatcher 역할만 한다.

    def parse_options(self, opts) -> PIMOptions:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in PIMOptions.__dataclass_fields__.keys() if k in opts})
        return PIMOptions(**args)

    def load_dialects(self, ctx):
        return  # IR 변환은 triton-shared-opt subprocess가 수행 (Python에 dialect 로드 안 함)

    def add_stages(self, stages, options, language):
        stages["ttir"]       = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _ttir_to_ttsharedir(src)
        stages["pim_common"] = lambda src, metadata: _ttsharedir_to_pim_common(src, options)
        stages["vendor_ir"]  = lambda src, metadata: _pim_common_to_vendor(src, options)
        stages["artifact"]   = lambda src, metadata: _vendor_codegen(src, metadata, options)

    # pack_metadata / get_codegen_implementation / hash / get_module_map 는 CPUBackend와 동일.
```

`_ttir_to_ttsharedir`는 CPUBackend에서 import해 재사용하고, 나머지 `_*` helper는 `triton-shared-opt`/`triton-shared-translate`를 호출하는 thin wrapper다. 호출하는 pass 목록은 11장·12장·13장에 정의한다.

---

## 8. Vendor Extension Registry & Interface

```python
class VendorExtensionRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, extension):
        cls._registry[name] = extension

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise ValueError(
                f"Unsupported PIM vendor '{name}'. "
                f"Registered vendors: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

VendorExtensionRegistry.register("upmem", UPMEMVendorExtension())
VendorExtensionRegistry.register("aimx", AiMXVendorExtension())
```

지원하지 않는 vendor는 명확하게 error를 발생시킨다.

```cpp
class PIMVendorExtension {
public:
  virtual llvm::StringRef getVendorName() const = 0;
  virtual void registerDialect(mlir::DialectRegistry &registry) = 0;
  virtual void registerPasses() = 0;
  virtual void buildLoweringPipeline(mlir::OpPassManager &pm) = 0;
  virtual void buildOptimizationPipeline(mlir::OpPassManager &pm) = 0;
  virtual void buildCodegenPipeline(mlir::OpPassManager &pm) = 0;
  virtual void buildRuntimeStubPipeline(mlir::OpPassManager &pm) = 0;
  virtual PIMCapability queryCapability(const PIMTarget &target) = 0;
  virtual PIMConstraints queryConstraints(const PIMTarget &target) = 0;
};
```

---

## 9. TTSharedIR

TTSharedIR은 다음 정보를 유지한다.

```text
program_id, tile index, block pointer, tile load/store,
dot, reduce, mask, layout hint, block shape, memory access pattern
```

예시:

```mlir
ttshared.func @matmul(%A, %B, %C, %M, %N, %K) {
  %pid_m = ttshared.program_id x
  %pid_n = ttshared.program_id y
  %a_tile = ttshared.load_tile %A { shape = [BLOCK_M, BLOCK_K], layout = "row_major" }
  %b_tile = ttshared.load_tile %B { shape = [BLOCK_K, BLOCK_N], layout = "row_major" }
  %acc = ttshared.dot %a_tile, %b_tile { out_shape = [BLOCK_M, BLOCK_N] }
  ttshared.store_tile %C, %acc
}
```

---

## 10. PIM Common IR

PIM Common IR은 특정 vendor에 종속되지 않는 PIM 공통 실행 모델이다. 본문(`scf`/`linalg`/`memref`/`arith`)을 그대로 보존하고(4.1), 그 주위에 vendor-neutral 분배 구조를 합성한다.

PIM Common IR에서 **쓰지 않는** vendor 표현: `tasklet`, `DPU`, `FPGA`, `MRAM`, `WRAM`, `GBuf`, `OutReg`, channel-specific descriptor.

PIM Common IR이 쓰는 **중립 표현**: `execution_unit`, `execution_group`, `local_lane`, `memory_space`, `memory_region`, `placement`, `transfer`, `compute`, `data_residency`, `tile_plan`, `partial_result`, `reduce`, `pipeline`, `launch`.

### 10.1 Operations

```text
pim.func            # primitive/dtype/accumulation 속성을 단 함수
pim.memory_region   # buffer + space/layout/role/residency
pim.tile_plan       # grid(BLOCK 파생) + SPLIT 분배도
pim.execution_group # 유닛 분배 정책 (capability_ref)
pim.placement       # 분산 의도 enum
pim.data_residency  # 상주 정책 enum
pim.transfer        # host ↔ device 전송 (골격)
pim.compute         # 본문(scf/linalg/memref)을 담는 region
pim.partial_result  # split_k>1일 때 부분 결과
pim.reduce          # 부분 결과 합산
pim.pipeline        # transfer/compute/store overlap (opt-in)
pim.launch          # device/vendor/device_id
```

memory space는 enum attribute로 표현하며, `pim.memory_region`의 `space`와 본문 memref의 memory-space에 쓰인다.

```text
host             # CPU DRAM
device_global    # 유닛이 접근하는 대용량 device 메모리
device_local     # 유닛 내부 scratchpad
device_scratch   # 임시 scratchpad
device_register  # 누산 레지스터
```

| PIM Common        | UPMEM          | AiMX               |
| ----------------- | -------------- | ------------------ |
| `host`            | CPU DRAM       | CPU DRAM           |
| `device_global`   | MRAM           | device memory      |
| `device_local`    | WRAM           | GBuf/local buffer  |
| `device_scratch`  | WRAM temporary | scratch buffer     |
| `device_register` | DPU register   | MAC/OutReg         |

### 10.2 Operation 정의

`pim.func` — primitive/dtype 등 속성을 단 함수.

```mlir
pim.func @matmul(%A, %B, %C, %M, %N, %K) attributes {
  primitive = "gemm", dtype = "bf16", accumulation = "fp32"
} { ... }
```

`pim.memory_region` — buffer를 space/layout/role/residency로 감싼다.

```mlir
%A_region = pim.memory_region %A { space = "host", layout = "row_major", role = "input_activation" }
%B_region = pim.memory_region %B { space = "device_global", layout = "packed_by_output_tile",
                                   role = "weight", residency = "persistent" }
%C_region = pim.memory_region %C { space = "host", layout = "row_major", role = "output" }
```

`pim.tile_plan` — grid(BLOCK에서 파생)와 SPLIT 분배도. BLOCK은 본문 subview에서 파생하므로 여기엔 grid·SPLIT만 둔다.

```mlir
%tp = pim.tile_plan {
  grid_m = ceildiv(%M, BLOCK_M),
  grid_n = ceildiv(%N, BLOCK_N),
  split_m = #pim.from_option<"SPLIT_M">,   // driver default / autotune config (16장)
  split_n = #pim.from_option<"SPLIT_N">,
  split_k = #pim.from_option<"SPLIT_K">
}
```

`pim.execution_group` — 유닛 분배 정책. vendor-specific 상수는 IR에 박지 않고 `capability_ref`로 표현하며, lowering 시 capability에서 값을 읽어 구체화한다.

```mlir
%eg = pim.execution_group {
  num_units = #pim.capability_ref<"max_num_units">,
  local_lanes_per_unit = #pim.capability_ref<"max_local_lanes_per_unit">,
  mapping_policy = "output_2d_partitioned",
  stationary_data = "B"
}
```

| Neutral Term      | 의미                            | UPMEM            | AiMX                          |
| ----------------- | ----------------------------- | ---------------- | ----------------------------- |
| `execution_unit`  | tile을 수행하는 독립 실행 단위    | DPU              | PU / compute lane group       |
| `execution_group` | 여러 execution unit의 집합     | DPU set          | channel group / compute group |
| `local_lane`      | 한 unit 내부 병렬 lane         | tasklet          | internal parallel slot        |
| `num_units`       | 활성 유닛 수                   | number of DPUs   | number of compute units       |

`pim.transfer` — host↔device 전송 골격. 실제 copy 방식(DMA/dpu_copy)은 vendor가 정한다.

```mlir
pim.transfer %A_region { src = "host", dst = "device_global", async = true, transform = "none" }
pim.transfer %B_region { src = "host", dst = "device_global", async = true,
                         transform = "pack_by_output_tile", residency = "persistent" }
```

`pim.placement` / `pim.data_residency` — 분산·상주 **의도**를 enum으로 표현 (구체 산술은 vendor).

```mlir
pim.placement %B_region { axis = "N", policy = "interleave", granularity = 256, target = "device_global" }
pim.data_residency %B_region { residency = "persistent", lifetime = "multi_launch", update_policy = "explicit" }
```

`pim.compute` (region-carrying) — 한 execution_unit이 수행하는 본문을 region으로 담는다. 본문은 TTSharedIR에서 온 `scf`/`linalg`/`memref` 그대로다.

```mlir
pim.compute ins(%A_region, %B_region) outs(%C_region)
  using %eg, %tp
  attributes { primitive = "gemm" }      // primitive는 전략 힌트일 뿐, 본문을 대체하지 않음
{
^unit(%a, %b, %c):                         // 한 unit이 맡은 로컬 슬라이스에 대한 계산
  %acc = memref.alloc() : memref<?x?xf32>  // BLOCK_M x BLOCK_N 누산기 (device_local)
  linalg.fill ins(%f0) outs(%acc)
  scf.for %k0 = %c0 to %K step %BLOCK_K {  // K-loop 그대로 (= K-chunk 스트리밍)
    %at = memref.subview %a[0, %k0]  [%BLOCK_M, %BLOCK_K] [1, 1]
    %bt = memref.subview %b[%k0, 0]  [%BLOCK_K, %BLOCK_N] [1, 1]
    linalg.matmul ins(%at, %bt) outs(%acc)   // 본문 그대로 → vendor가 충실 번역, in-place 누적
  }
  memref.copy %acc, %c
}
```

`primitive` 힌트(전략 선택·placement에만 사용, 본문 lowering과 무관): `gemm gemv batched_gemm attention_qk attention_av reduction elementwise`. 본문이 보존되므로 목록에 없는 연산·fused 본문도 표현된다. UPMEM은 본문을 loops로 번역하고(12장), AiMX는 `primitive`로 descriptor를 고른다(13장).

`pim.partial_result` / `pim.reduce` — `split_k>1`일 때 유닛 간 부분 결과와 합산.

```mlir
%partial = pim.partial_result { shape = [BLOCK_M, BLOCK_N], accumulation_dtype = "fp32", owner = "execution_unit" }
%C_final = pim.reduce %partial { axis = "K", method = "device_reduce" }   // none|host_reduce|device_reduce|tree_reduce
```

`pim.pipeline` (opt-in) / `pim.launch`.

```mlir
pim.pipeline { stage0 = "transfer_next_chunk", stage1 = "compute_current_chunk", stage2 = "store_previous_result" }
pim.launch @matmul { device = "pim", vendor = "upmem", device_id = 0 }
```

### 10.3 실행 모델 — grid / SPLIT / BLOCK 3계층

GPU Triton은 `grid`(HW가 SM에 분배) + `BLOCK`(유닛 내부 타일) 2계층이다. PIM에는 하드웨어 블록 스케줄러가 없고 유닛이 고정 개수의 물리 자원이므로, **grid의 타일 격자를 고정 유닛 수에 명시적으로 얹는 SPLIT 계층**이 추가된다.

규약: `A(M,K) · B(K,N) → C(M,N)`, row-major, dtype 크기 `s`(bf16=2, fp32=4), 누산 dtype `s_acc`.

**(1) grid — BLOCK에서 파생 (Triton 그대로)**

```text
Gm = ceil(M / BLOCK_M)        # 출력 타일 행 수
Gn = ceil(N / BLOCK_N)        # 출력 타일 열 수
T  = Gm * Gn                  # 전체 출력 BLOCK 타일 수
Gk = ceil(K / BLOCK_K)        # 타일당 K-loop 청크 수 (intra-tile)
```

`BLOCK_M/N`이 정해지면 grid가 결정된다. grid는 출력 타일 격자(intent)다.

**(2) SPLIT — 유닛 간 분배 (PIM 추가, host-driven)**

```text
U = split_m * split_n * split_k     # 활성 유닛 수
U <= NUM_UNITS                      # capability 상한

각 유닛 (ri, ci, kj) 가 맡는 부분:
Mu = M / split_m      # 유닛의 출력 행
Nu = N / split_n      # 유닛의 출력 열
Ku = K / split_k      # 유닛의 reduction 길이
```

grid와 SPLIT의 다리:

```text
T = Gm * Gn = (유닛당 타일 수) * split_m * split_n
유닛당 타일 수 = (Gm/split_m) * (Gn/split_n) = (Mu/BLOCK_M) * (Nu/BLOCK_N)
```

즉 SPLIT은 grid의 T개 타일을 `split_m*split_n`개 유닛에 묶어 배정한다.

**(3) BLOCK — 유닛 내부 타일 + K-청크 (Triton 그대로)**

각 유닛은 자기 `Mu x Nu` 부분행렬을 `BLOCK_M x BLOCK_N` 타일로 순회하고, 타일마다 K-loop이 `BLOCK_K` 청크를 streaming하며 누적한다. `local_lanes`(intra-unit 병렬)가 `BLOCK_M` 행을 나눠 메모리 지연을 숨긴다.

**정렬 조건** (앞 절들의 "SPLIT 영역 ≥ BLOCK"):

```text
split_m | Gm  (and split_m <= Gm)   <=>  Mu가 BLOCK_M의 배수
split_n | Gn  (and split_n <= Gn)   <=>  Nu가 BLOCK_N의 배수
split_k | Gk  (split_k > 1일 때)      <=>  Ku가 BLOCK_K의 배수
```

SPLIT 경계는 항상 BLOCK 타일 경계 위에 떨어진다(유닛 = 정수개 BLOCK 타일). `split > grid`로 가면 한 BLOCK 타일을 여러 유닛이 sub-분할 — 유닛당 영역이 BLOCK보다 작아지고 재사용이 줄어드는 tradeoff가 생긴다. 나눠떨어지지 않는 가장자리는 tail policy(pad/mask)로 처리한다.

**일의 총량** — BLOCK은 총량을 바꾸지 않고 SPLIT만 분배한다:

```text
work/unit = (Gm/split_m)(Gn/split_n) * (Ku/BLOCK_K) * (BLOCK_M*BLOCK_N*BLOCK_K)
          = Mu * Nu * Ku
          = (M*N*K) / U
```

SPLIT은 전체 일 `M*N*K`를 U개 유닛으로 나누는 분모이고, BLOCK은 각 유닛 몫을 scratchpad에 맞게 재청크할 뿐이다(곱에서 상쇄).

**유닛별 입력/출력과 offset** (split_k 무시한 기본형):

```text
유닛 (ri,ci) 담당 출력:  C[ri*Mu:(ri+1)*Mu, ci*Nu:(ci+1)*Nu]
유닛 (ri,ci) 받는 입력:  A 행밴드 A[ri*Mu:.., 0:K]   (BLOCK_K로 스트리밍)
                         B 열밴드 B[0:K, ci*Nu:..]    (전치 시 연속)

A_offset = ri*Mu*K            # A(M,K) row-major
B_offset = ci*Nu*K            # Bt(N,K) (host가 전치) — 열타일 연속
C_offset = ri*Mu*N + ci*Nu    # 최종 C 위치
```

A는 M(행)으로, B는 N(열)으로 split된다. 같은 행의 유닛들은 동일한 A 행밴드를 공유(N 방향 복제), 같은 열의 유닛들은 동일한 B 열밴드를 공유(M 방향 복제)한다.

두 regime:

| | 타일 ≥ 유닛 (큰 행렬) | 타일 < 유닛 (작은 행렬 + 많은 유닛) |
| --- | --- | --- |
| 분배 | `SPLIT_M/N`로 출력 분할, 유닛당 여러 BLOCK 타일 | M/N을 더 잘게(sub-BLOCK, reduce 없음) 또는 `SPLIT_K`로 K 분담 |
| reduce | 없음 | `SPLIT_K`>1이면 있음 (`reduction_k_partitioned`) |

### 10.4 메모리 계층 대응 — SPLIT↔MRAM, BLOCK_K↔WRAM

데이터는 메모리 계층을 두 번 내려가고, 각 내려감이 한 단계의 타일링이다.

```text
host DRAM  (전체 A, B, C)
   │  ① SPLIT  — host가 각 유닛 몫을 scatter
   ▼
per-unit device_global (MRAM)  (그 유닛의 Au, Bu, Cu 밴드 상주)
   │  ② BLOCK_K  — K-loop이 청크를 DMA
   ▼
device_local (WRAM)  (현재 타일의 acc + A_sub + B_sub)
   │  MAC
   ▼
device_register (누산)
```

용량식 — BLOCK의 WRAM 예산식과 SPLIT의 MRAM 예산식이 대응한다.

```text
WRAM (BLOCK): (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N)*s + BLOCK_M*BLOCK_N*s_acc  <=  WRAM_per_unit
MRAM (SPLIT): (Mu*Ku + Ku*Nu)*s + Mu*Nu*s_acc                                <=  MRAM_per_unit
```

- BLOCK: scratchpad(WRAM, 작음 — UPMEM 기준 ~64KB)에 항상 binding된다. 그래서 BLOCK의 주목적은 "WRAM에 맞춤 + 재사용 극대화". `BLOCK_M/N`은 WRAM 상주 `acc` 타일 크기, `BLOCK_K`는 MRAM→WRAM 스트림 청크 크기다.
- SPLIT: 주목적은 병렬 분배(고정 유닛 수를 채움). MRAM(~64MB)은 상한일 뿐 보통 binding되지 않는다.

컴파일러 관점에서 이는 **메모리 계층별 multi-level loop tiling**이다 — SPLIT = 바깥 타일링(per-unit scratchpad 목표), BLOCK = 안쪽 타일링(WRAM 목표). 단 투명 캐시가 아니라 명시적 scratchpad라 컴파일러가 단계마다 DMA를 직접 emit한다.

### 10.5 PIM Common IR 예시

```mlir
pim.func @matmul(%A, %B, %C, %M, %N, %K) attributes {
  primitive = "gemm", dtype = "fp32", accumulation = "fp32"
} {
  // ── ② TTSharedIR에 없던 PIM 구조 (새로 합성) ──
  %ra = pim.memory_region %A { space = "host", role = "input"  }
  %rb = pim.memory_region %B { space = "host", role = "weight" }
  %rc = pim.memory_region %C { space = "host", role = "output" }

  %tp = pim.tile_plan {
    grid_m = ceildiv(%M, BLOCK_M),
    grid_n = ceildiv(%N, BLOCK_N),
    split_m = #pim.from_option<"SPLIT_M">,
    split_n = #pim.from_option<"SPLIT_N">,
    split_k = #pim.from_option<"SPLIT_K">
  }
  %eg = pim.execution_group { mapping_policy = "output_2d_partitioned" }

  pim.transfer %ra { src = "host", dst = "device_global" }
  pim.transfer %rb { src = "host", dst = "device_global" }

  // ── ① TTSharedIR 본문을 그대로 보존 (한 unit의 로컬 슬라이스 기준) ──
  pim.compute ins(%ra, %rb) outs(%rc) using %eg, %tp
    attributes { primitive = "gemm" }
  {
  ^unit(%a, %b, %c):
    %acc = memref.alloc() : memref<?x?xf32>          // device_local 누산기
    linalg.fill ins(%f0) outs(%acc)
    scf.for %k0 = %c0 to %K step %BLOCK_K {
      %at = memref.subview %a[0, %k0] [%Mu, %BLOCK_K] [1, 1]
      %bt = memref.subview %b[%k0, 0] [%BLOCK_K, %Nu] [1, 1]
      linalg.matmul ins(%at, %bt) outs(%acc)         // ← 그대로, pim.matmul 아님, in-place 누적
    }
    memref.copy %acc, %c
  }

  pim.transfer %rc { src = "device_global", dst = "host" }
  pim.launch @matmul { device = "pim", vendor = "upmem" }
}
```

---

## 11. PIM Common Pass Pipeline

### 11.0 공통 pass의 경계 (핵심)

공통(PIM Common) pass는 **값을 최적화하지 않는다.** 다음 세 가지만 한다.

1. **구조 생성** — 주어진 값(constexpr/option)을 IR에 기록만 한다. "무엇이 최적인가"를 결정하지 않는다. 예: `build-pim-offload`는 block/split을 "고르지" 않고 주어진 값을 적는다.
2. **intent 표현** — 분산/매핑/잔류를 vendor-neutral enum으로 표현한다. 구체 산술은 하지 않는다. 예: `placement="interleave"`는 의도일 뿐, 실제 bank 오프셋은 vendor가 정한다.
3. **capability-gated 결정** — vendor 사실이 필요하면 그 vendor의 `PIMCapability`를 query한다. 컴파일 시점에 vendor는 이미 하나로 정해져 있으므로 vendor-aware하게 동작할 수 있다.

따라서 다음은 공통 pass의 일이 **아니다**: 최적값 선택(어느 block/split이 빠른가 → autotune이 실측, 16장), 구체 실현(실제 bank/channel/register 산술 → vendor lowering, 12·13장), 적법성 판단(block_k가 WRAM에 들어가나 → PIMConstraints 기반 적법성 필터, 11.4).

한 줄 요약: **공통 = 구조 + 의도 + (필요시) capability gate**, **autotune = 값 선택**, **vendor = 구체 실현**.

### 11.1 핵심 변환 (TTSharedIR → PIM Common IR)

faithful-body 모델에서 변환의 핵심은 본문을 보존한 채 분배 구조만 합성하는 것이라 패스는 둘이면 충분하다.

```text
build-pim-offload      # PIM 오프로드 region 인식 → memory_region/transfer/
                       #   tile_plan(grid·SPLIT)/execution_group을 본문 주위에 합성.
                       #   본문(scf/linalg/memref)은 그대로 두고, program_id 전역 오프셋만
                       #   분배층으로 이동. primitive는 속성으로 기록.
select-pim-strategy    # split factor + capability로 적법 전략 표시 (11.4).
                       #   단일 vendor·no-split이면 생략 가능.
canonicalize
cse
```

`SPLIT_*`/`NUM_UNITS`/`LOCAL_LANES`는 `build-pim-offload`가 option에서 직접 읽어 `tile_plan`/`execution_group`에 반영한다(16장). K-chunk는 본문 `scf.for`에 있으므로 별도 op로 두지 않는다.

### 11.2 PIM Common Optimization (opt-in, 기본 OFF)

아래는 기본 파이프라인에 넣지 않는 선택적 최적화다. toggle로 하나씩 켜며, OFF 시 11.1 결과가 그대로 통과해야 한다(11.0 경계 준수).

```text
pim-layout-normalize        # 중복 transpose 제거 등 layout 표현 정규화
pim-data-placement          # placement intent enum 갱신
pim-transfer-hoisting       # 반복 transfer를 상위 scope로
pim-persistent-data         # weight/KV persistent 표시
pim-partial-result-planning # split_k>1일 때 partial/reduce 구조
pim-pipeline-overlap        # supports_pipeline_overlap일 때 pim.pipeline 삽입
```

### 11.3 Pass Responsibilities

꼬리표: **[구조]** 기록·표현만, **[intent]** vendor-neutral policy enum만, **[gate]** vendor capability query, **[vendor가 실현]** 이 공통 pass가 만든 intent를 실제 산술로 바꾸는 책임은 vendor lowering에 있음.

| Pass | 분류 | 공통 layer 역할 | vendor 실현 |
| --- | --- | --- | --- |
| `build-pim-offload` | [구조] | region 인식 + `BLOCK_*`/grid/dtype 추출 + `primitive` 분류 + `memory_region`/`transfer`(골격)/`tile_plan`(grid·SPLIT)/`execution_group`(mapping enum) 합성. 본문 보존, program_id 오프셋만 분배층으로 이동. | copy 방식·유닛/채널 배정·타일 micro-fit·본문 lowering(12장) |
| `select-pim-strategy` | [gate] | split factor + capability로 적법 전략 표시(11.4) | 전략 attribute legalize (`reduction_k_partitioned`→host_reduce 등) |
| `pim-layout-normalize` | [구조] | layout 표현 정규화만 | 물리 layout 변환(`pack_by_*`) |
| `pim-data-placement` | [intent] | reuse 고려 placement 의도 갱신 | bank/channel 오프셋 |
| `pim-transfer-hoisting` | [구조] | 반복 transfer를 상위 scope로 | — |
| `pim-persistent-data` | [intent] | weight/KV persistent 표시 | 상주 버퍼 관리 |
| `pim-partial-result-planning` | [구조] | partial/reduce 구조 생성 (split_k>1) | reduce 구현 |
| `pim-pipeline-overlap` | [gate] | `supports_pipeline_overlap`일 때 `pim.pipeline` 삽입 | async/double-buffer |

적법성("BLOCK_K가 WRAM에 들어가나" 등)은 위 어느 pass도 판단하지 않는다. `PIMConstraints` 기반 적법성 필터(11.4)가 담당한다.

### 11.4 Execution Strategy Selection

#### 동기 — 분배 패턴의 분기

UPMEM용 손수 작성 라이브러리(PGEMM)는 `(splitM, splitK, splitN)` 조합에 따라 서로 다른 커널로 분기한다: no-split / N-split / K-split(→partial reduce) / M-split(=N-split + 전치). 각 분기는 데이터 layout·유닛 매핑·reduction이 다른 별개 실행 전략이다. 즉 맞는 전략은 split factor와 hardware에 따라 달라진다.

#### 전략의 IR 표현

이 분기를 코드에 if-else로 박지 않고, 전략을 선택해 vendor-neutral attribute로 표현한다. 한 전략 = 다음 3개의 묶음이다.

```text
strategy = {
  mapping_policy : output_n_partitioned | output_m_partitioned | reduction_k_partitioned | none
  reduction      : none | host_reduce | device_reduce | tree_reduce
  placement      : blocked | interleave | replicated | output_tile_interleaved
}
```

| PGEMM 분기 | PIM 전략 (mapping_policy / reduction) |
| --- | --- |
| `basic` (no split) | `none` / `none` |
| `basic_n` (N-split) | `output_n_partitioned` / `none` |
| `basic_k` (K-split) | `reduction_k_partitioned` / `device_reduce` 또는 `host_reduce` |
| M-split (전치 후 N-split) | `output_m_partitioned` / `none` |

#### 선택은 Common, 실현은 Vendor

| 단계 | 위치 | 하는 일 |
| --- | --- | --- |
| 전략 선택 + 표현 | PIM Common IR | "K-split + device_reduce로 간다"를 결정하고 `mapping_policy`/`reduction`/`placement` attribute로 박음 |
| 전략 구체화(legalize) | vendor IR | 실제 유닛 개수·MRAM 오프셋·reduce 구현 (UPMEM) / channel·OutReg 누적·descriptor (AiMX) |

선택을 Common에 두는 이유: ① `mapping_policy`/`reduction`/`placement`는 이미 vendor-neutral 개념이다. ② 같은 TTSharedIR이 어느 vendor로도 갈 수 있으므로 선택 로직을 vendor마다 중복하면 안 된다. ③ 공통 최적화 pass가 단일 표현 위에서 동작해야 한다.

#### vendor-aware 선택

컴파일 시점에 vendor는 이미 하나로 정해져 있다(`options.vendor`). 따라서 `select-pim-strategy`는 Common 레벨 pass이지만 그 vendor의 `PIMCapability`/`PIMConstraints`를 query해 vendor-aware하게 고른다. 로직은 공통 한 곳, 그 로직을 움직이는 데이터는 vendor별 — vendor 차이가 코드 중복이 아니라 capability 데이터로 들어온다.

#### 선택 방법 — 적법성 필터 + autotune (cost model 없음)

1. **적법성 필터 (deterministic, vendor-aware)**: split factor + capability/constraints로 적용 가능한 전략만 추린다.
   - `SPLIT_K==1` → K-split 전략 제외
   - `block_k`가 WRAM 초과 → 해당 BLOCK 조합 제외
   - `supports_device_reduce=false` → `device_reduce` 제외 (단 `host_reduce` K-split은 적법)
2. **적법한 전략 중 best**: `@triton.autotune` 사용 시 적법 전략들을 각각 별도 config로 컴파일·실행·벤치마크해 empirical best를 고른다(16장). 미사용 시 단순 기본 규칙(예: no-split이 적법하면 no-split, 아니면 N-split)으로 하나를 확정한다.

전략 간 우열은 정적 비용 추정이 아니라 autotune 실측으로 가린다. 정적으로는 "적법한가"만 판단한다 — 그래서 cost model이 필요 없다.

#### 확장성

새 전략 추가 = 전략 레지스트리에 항목 하나 + 그 전략의 lowering 한 벌. 기존 전략 코드를 건드리지 않는다.

```text
PIMExecutionStrategy (interface, vendor-neutral)
  ├── isApplicable(tile_plan, capability, constraints) -> bool   // 적법성만 (cost 추정 없음)
  └── applyToIR(builder)                                         // execution_group/reduce/placement 확정
```

전략 목록은 vendor-neutral하고, vendor IR은 확정된 전략 attribute를 자기 방식으로 legalize만 한다 (예: `reduction_k_partitioned` → UPMEM은 host_reduce, AiMX는 descriptor accumulate).

---

## 12. UPMEM Vendor Extension

UPMEM은 프로그래머블 코어(DPU)이므로 본문을 loops로 충실 번역한다. 임의/fused 커널이 가능하다.

### 12.1 Mapping

| PIM Common IR     | UPMEM IR                 |
| ----------------- | ------------------------ |
| `execution_unit`  | DPU                      |
| `execution_group` | DPU set                  |
| `local_lane`      | tasklet                  |
| `device_global`   | MRAM                     |
| `device_local`    | WRAM                     |
| `device_register` | DPU register             |
| `pim.transfer`    | host ↔ MRAM copy         |
| `pim.compute`     | DPU kernel               |
| `pim.launch`      | `dpu_launch()`           |
| `pim.reduce`      | host reduce / DPU reduce |

### 12.2 UPMEM Dialect Operations

```text
upmem.module  upmem.host_func  upmem.device_func
upmem.alloc_units  upmem.load_program  upmem.copy_to_global  upmem.copy_from_global  upmem.launch
upmem.unit_id  upmem.local_lane_id
upmem.global_load  upmem.global_store  upmem.local_alloc  upmem.local_load  upmem.local_store
upmem.local_gemm  upmem.local_gemv  upmem.barrier  upmem.host_reduce  upmem.device_reduce
```

### 12.3 Lowering & Codegen Pipeline

```text
PIM Common IR
  ↓ legalize-pim-to-upmem
UPMEM IR
  ↓ upmem-memory-placement / upmem-execution-mapping / upmem-transfer-lowering / upmem-host-device-split
Optimized UPMEM IR
  ↓ triton-shared-translate --upmem-to-host / --upmem-to-device
host.cpp + device.c
  ↓ g++ + libdpu / dpu-upmem-dpurte-clang
host.so + device.elf
```

translate는 성격이 다른 두 산출물을 만든다: **device 커널은 본문 IR(`scf`/`memref`/`linalg`)을 충실히 번역**해 생성하고(12.8), **host는 grid 기반 multi-DPU 오케스트레이션**을 생성한다(12.6). 전자는 op마다 다르므로 생성이 필수이고, 후자는 데이터 이동이라 plan-구동 골격을 재사용한다.

### 12.4 UPMEM IR Example

```mlir
upmem.host_func @matmul_host(%A, %B, %C) {
  %units = upmem.alloc_units { num_units = #num_units, split_m = #sm, split_n = #sn }
  upmem.load_program %units, @matmul_device_kernel
  upmem.copy_to_global %units, %A { dst = "global_a", layout = "row_major" }
  upmem.copy_to_global %units, %B { dst = "global_b", layout = "packed_by_output_tile" }
  upmem.launch %units, @matmul_device_kernel { async = false }
  upmem.copy_from_global %units, %C { src = "global_c" }
}

upmem.device_func @matmul_device_kernel(%global_a, %global_b, %global_c) {
  %unit = upmem.unit_id
  %lane = upmem.local_lane_id
  // 본문(scf/linalg/memref)을 충실 번역한 loops + MRAM↔WRAM DMA (12.8)
  ...
}
```

### 12.5 Codegen Artifacts

```text
generated/upmem/
  matmul_host.cpp   matmul_device_kernel.c   matmul_device_kernel.elf
  matmul_metadata.json   pim_runtime_stub.cpp
```

### 12.6 Host 분배 모델 (grid → DPU)

PGEMM `basic_n`의 multi-DPU 오케스트레이션 구조를 차용하되, device 커널은 12.7의 충실 번역으로 생성하고 host는 plan(`tile_plan`/`execution_group`)으로 구동되는 생성 코드다.

host가 받는 정보:

| 출처 | 정보 |
| --- | --- |
| 런타임 인자 | `M, N, K`, host 포인터 `A,B,C` |
| Common IR (`tile_plan`) | `BLOCK_M/N/K`, grid `Gm, Gn` |
| `execution_group` / option | `split_m, split_n` (inter-DPU), `num_dpus` (capability) |

2D output partition (`num_units = split_m × split_n`, DPU `d=(ri,ci)`, `ri=d/split_n`, `ci=d%split_n`), offset은 10.3과 동일. 분배 알고리즘(basic_n 구조):

```text
1. dpu_alloc(split_m*split_n) + dpu_load(device.elf)
2. 각 DPU offset/dim 계산 → input_args[]  (DPU_INPUT_ARGUMENTS)
3. push input_args
4. scatter A: DPU_FOREACH{prepare_xfer(A+A_off)} + push_xfer   (A 행밴드)
   scatter B: DPU_FOREACH{prepare_xfer(Bt+B_off)} + push_xfer  (B 열밴드, 전치)
5. dpu_launch(DPU_SYNCHRONOUS)
6. gather C: DPU_FOREACH{prepare_xfer(Cbuf)} + push_xfer(FROM); LayoutTransform → C[C_off]
```

device(12.8)는 `DPU_INPUT_ARGUMENTS`의 `Mu/Nu/K/BLOCK_K/offset`을 루프 경계로 쓴다 — 그래서 device.c는 상수가 아니라 런타임 차원으로 동작한다. `split_m×split_n > Gm×Gn`이면 한 BLOCK 타일을 여러 DPU가 행/열로 나눈다(적은 타일로 많은 DPU 채움, 활용률↑).

### 12.7 SPLIT / LOCAL_LANES 실현 (inter-DPU vs intra-DPU)

두 병렬 축은 "값이 어디에 박히는가"가 다르다.

- **`NUM_UNITS`(=SPLIT) — host.cpp 소스에 박힘**: `alloc_units{num_units=U}` → `UPMEMToHost`가 `U`를 C 상수로 host.cpp에 박는다(`dpu_alloc(U)`, A는 행밴드 scatter, B/C는 per-DPU prepare_xfer + push_xfer). host는 g++로 컴파일되는 일반 코드라 상수면 충분 → 소스 번들만으로 완결.
- **`LOCAL_LANES`(=NR_TASKLETS) — device 컴파일 플래그로 전달**: DPU 런타임은 tasklet 개수를 `-DNR_TASKLETS` 컴파일 플래그로만 안다. `UPMEMToDevice`는 device.c를 `NR_TASKLETS` 매크로 기반으로 generic하게 emit하고(`BARRIER_INIT`, tasklet마다 자기 WRAM 캐시, `for(i=me(); i<Mu; i+=NR_TASKLETS)` 행 round-robin), launcher가 `metadata.local_lanes`를 읽어 `dpu-upmem-dpurte-clang -DNR_TASKLETS=L`로 컴파일한다. IR 구조를 바꾸지 않는 device 빌드 파라미터라 metadata 채널로 흐른다.

이득: DPU의 fine-grained multithreading이 한 tasklet의 `mram_read` 지연을 다른 tasklet 연산으로 숨긴다. 제약: per-tasklet WRAM × NR_TASKLETS ≤ WRAM 예산 → `BLOCK`·`local_lanes` 균형(autotune).

### 12.8 Device 충실 번역 (scf/memref/linalg → DPU C)

device 커널은 고정 gemm 템플릿이 아니라 본문(`scf`/`memref`/`linalg`)을 충실히 lowering해 생성한다. 그래야 커널 의도가 보존되고 gemm 외 연산(gemv/relu/fused)도 같은 경로로 처리된다. (`upmem.local_gemm`은 gemm primitive를 인식했을 때의 단축 경로이고, 충실 번역이 일반 경로다.)

```text
linalg + scf + memref
   ↓  (MLIR 재사용: --convert-linalg-to-loops 등 — CPU 백엔드와 동일)
scf/affine loops + memref.load/store + arith
   ↓  (DPU 전용 번역: UPMEMToDevice)
device.c   (MRAM↔WRAM DMA 삽입)
   ↓  dpu-upmem-dpurte-clang
device.elf
```

CPU lowering은 통합 주소공간(memref=포인터)을 가정하지만, DPU는 데이터가 MRAM에 있고 계산하려면 WRAM으로 명시적 DMA(`mram_read`/`mram_write`)해야 한다. 그래서 `linalg→loops`까지는 MLIR을 재사용하되 **memref→실제 메모리 접근**만 DPU 전용으로 번역한다.

| TTSharedIR 구문 | device.c |
| --- | --- |
| `scf.for %k0 = 0 to K step BLOCK_K` | `for(k0=0;k0<K;k0+=BLOCK_K)` (K 스트리밍) |
| `memref.subview` (MRAM 상주) | `mram_read(base+off, wram, len)` |
| `linalg.matmul` → loops | `for i,j,kk: acc += a_w[..]*b_w[..]` |
| 바깥 parallel 차원(행) | `for(i=me(); i<Mu; i+=NR_TASKLETS)` (tasklet 분배) |
| `memref.copy → %C` | `mram_write(wram, C_base+off, len)` |
| 루프 경계 `Mu,Nu,K` | `DPU_INPUT_ARGUMENTS`에서 수신 |

본문이 무엇이든 번역하므로 `linalg.matvec`→gemv, `linalg.generic`→relu, gemm+epilogue→fused가 별도 prebuilt 커널 없이 같은 번역기로 처리된다. `scf.for`(K-loop)를 그대로 옮겨 큰 K도 WRAM 초과 없이 스트리밍한다.

---

## 13. AiMX Vendor Extension

AiMX는 고정기능 하드웨어이므로 `primitive`로 descriptor를 선택한다. 매칭되는 descriptor가 없는 본문은 fail-fast한다.

### 13.1 Mapping

| PIM Common IR     | AiMX IR                        |
| ----------------- | ------------------------------ |
| `execution_unit`  | compute unit / PU              |
| `execution_group` | channel group / compute group  |
| `local_lane`      | internal execution lane        |
| `device_global`   | device memory                  |
| `device_local`    | GBuf / local buffer            |
| `device_register` | MAC / output register          |
| `pim.transfer`    | DMA descriptor                 |
| `pim.compute`     | compute descriptor             |
| `pim.placement`   | bank/channel mapping           |
| `pim.launch`      | command queue submit           |

### 13.2 AiMX Dialect Operations

```text
aimx.module  aimx.func  aimx.device  aimx.buffer  aimx.channel_group  aimx.compute_group
aimx.bank_map  aimx.layout_pack  aimx.dma_h2d  aimx.dma_d2h
aimx.compute_gemm  aimx.compute_gemv  aimx.compute_attention_qk
aimx.descriptor  aimx.descriptor_chain  aimx.submit  aimx.wait
```

### 13.3 Lowering & Codegen Pipeline

```text
PIM Common IR
  ↓ legalize-pim-to-aimx
AiMX IR
  ↓ aimx-memory-bank-mapping / aimx-execution-mapping / aimx-descriptor-selection / aimx-dma-scheduling
Optimized AiMX IR
  ↓ aimx-command-stream-codegen / aimx-runtime-stub
Descriptor Stream + Runtime Stub
```

### 13.4 AiMX IR Example

```mlir
aimx.func @matmul_command_stream(%A, %B, %C) {
  %dev = aimx.device { id = 0 }
  %cg  = aimx.channel_group { count = 16 }
  %A_dev = aimx.buffer { space = "device", layout = "row_major" }
  %B_dev = aimx.buffer { space = "device", layout = "packed_by_bank" }
  %C_dev = aimx.buffer { space = "device", layout = "row_major" }
  %d0 = aimx.dma_h2d %A, %A_dev { async = true }
  %d1 = aimx.dma_h2d %B, %B_dev { layout_pack = "pack_by_output_tile", async = true }
  %d2 = aimx.compute_gemm %A_dev, %B_dev, %C_dev {
    M=1024, N=4096, K=4096, block_m=32, block_n=64, block_k=256,
    channels=16, banks=16, dtype="bf16", accumulation="fp32"
  }
  %d3 = aimx.dma_d2h %C_dev, %C { async = false }
  %chain = aimx.descriptor_chain %d0, %d1, %d2, %d3
  aimx.submit %chain { device_id = 0 }
  aimx.wait %chain
}
```

### 13.5 Codegen Artifacts

```text
generated/aimx/
  matmul_runtime_stub.cpp   matmul_descriptors.bin
  matmul_plan.json   matmul_metadata.json   layout_pack.cpp
```

---

## 14. 지원 커널 스코프 (affine-tileable)

host가 각 유닛의 입력 footprint를 컴파일/런칭 시점에 정해 scatter해야 하므로, PIMBackend는 **affine·tileable 접근 패턴** 커널을 대상으로 한다.

- 지원: GEMM, GEMV, batched GEMM, attention tile(QK/AV), 그리고 인덱스가 affine한 elementwise/reduction과 그 fused 조합. 각 유닛의 입력 슬라이스가 연속 밴드/스트라이드로 정적 결정된다.
- 비대상(또는 host 전체 복제로 fallback): data-dependent gather/scatter, 런타임 값에 의존하는 임의 인덱싱 등 각 유닛 footprint를 미리 못 정하는 패턴.

이는 PIM이 노리는 dense-LA 워크로드와 일치하는 스코프다. faithful-body 덕에 이 스코프 안에서는 gemv/relu/fused 등 임의 본문이 같은 경로로 처리된다(12.8).

---

## 15. Triton 실행 모델 대응

PIMBackend는 Triton의 SPMD 실행 모델을 그대로 반영한다. "커널을 어떻게 짜느냐"와 "config 값"이 성능을 좌우하는 Triton의 성질이 PIM에서도 성립하며, 그것을 성립시키는 장치가 faithful-body 번역(4.1/12.8)이다.

### 15.1 GPU ↔ PIM 대응

| Triton GPU | Triton PIM | 비고 |
| --- | --- | --- |
| 커널 = SPMD (한 program instance) | 커널 = SPMD (한 유닛의 일) | 동일 |
| SM | execution_unit (DPU) | |
| warp / thread | local_lane (tasklet) | |
| shared memory | device_local (WRAM) | |
| global memory (통합 주소) | device_global (유닛별 private) + host DRAM | 메모리 모델 다름 |
| `num_warps` | `local_lanes`(=NR_TASKLETS) | 동일 성격 (컴파일 플래그) |
| `BLOCK_M/N/K` (constexpr) | `BLOCK_M/N/K` (constexpr) | 동일 |
| grid (런타임 launch) | grid (런타임 launch) | 동일 |
| 하드웨어 블록 스케줄러가 CTA→SM 분배 | host scatter/gather가 tile→유닛 분배 (SPLIT) | 메커니즘 다름 |

### 15.2 충실한 부분 vs 아키텍처가 가르는 부분

**충실 반영**: SPMD 커널 작성, 런타임 grid launch, "커널 코드·config가 성능을 좌우", `@triton.autotune`. fast-memory 관리만 GPU는 하드웨어(shared mem 자동 스테이징, `num_stages` 파이프라인)가 암묵 처리하고, PIM은 vendor 패스가 MRAM↔WRAM DMA를 명시 emit한다(12.8).

**아키텍처가 강제한 차이**:

1. **분배가 암묵(하드웨어) → 명시(host)**. GPU는 통합 메모리라 블록을 SM에 자동 배치하지만, PIM은 유닛별 private 메모리라 host가 명시적으로 scatter/gather한다. 같은 개념(grid→유닛 분배), 다른 위치(하드웨어 vs host 코드).
2. **주소 모델**. GPU 커널은 통합 global을 직접 주소. PIM은 `program_id` 전역 오프셋을 분배층(`tile_plan` + host scatter)으로 끌어올려 본문은 로컬 슬라이스 기준으로 단순화한다. 사용자에겐 투명.
3. **`SPLIT` 추가 knob**. GPU 사용자는 "몇 SM 쓸지" 정하지 않는다. PIM은 `split_m/split_n`이 있지만, driver가 grid로 자동 기본값을 주고(15.4) autotune이 튜닝하므로 "안 건드려도 되지만 튜닝하면 빨라지는" Triton의 `num_warps`/grid 성질로 맞춘다. PIM은 grid보다 잘게 쪼개 더 많은 유닛을 쓸 수 있다는 점이 추가 자유도다.

### 15.3 분배 = 컴파일타임(구조) + 런타임(숫자)

host 코드는 한 번 생성·컴파일되어 모든 shape·grid에 재사용된다. "무엇을 나눌지"(구조)와 "몇 개/어디로"(숫자)를 분리한다.

- 컴파일타임: IR의 `execution_group`/`tile_plan`이 scatter/gather 코드 모양을 결정 → translate가 `(M,N,K,split_m,split_n)`를 인자로 받는 제네릭 host.cpp를 생성. `local_lanes`(=NR_TASKLETS)만 device 빌드 플래그라 컴파일타임에 고정.
- 런타임: driver가 grid와 `num_units`로 `split`을 정해 인자로 넘긴다.

```python
def decide_split(Gm, Gn, num_units):
    split_m, split_n = Gm, Gn          # 기본: grid 타일 하나당 유닛 하나
    while split_m * split_n * 2 <= num_units:   # 여유 유닛이 있으면 M축을 더 잘게
        split_m *= 2
    return split_m, split_n
```

한 번 컴파일한 `host.so`가 모든 shape·grid·유닛 수에 적응한다. `M`이 바뀌어 grid가 달라져도 재컴파일 없이 driver가 `split`만 다시 정한다.

### 15.4 런타임 default vs autotune config

`split`은 기본적으로 `decide_split`(런타임)이 정하지만, `@triton.autotune`이 `SPLIT_M/N`을 config로 sweep할 때는 config 값이 driver 자동 결정보다 우선한다.

```text
SPLIT_*가 autotune config에 있으면  → 그 값을 host.cpp 인자로 전달 (decide_split 우회)
없으면                              → driver.decide_split(grid, num_units)
```

Triton에서 grid(런타임)와 `BLOCK_*`(autotune config)가 공존하는 것과 같은 구조다 — default는 자동, autotune이면 config가 이긴다.

---

## 16. Autotune Integration

`@triton.autotune`은 별도의 PIM 전용 메커니즘이 아니다. Triton 표준 autotuner는 config마다 독립적으로 컴파일→실행→측정하고 가장 빠른 것을 고른다.

```text
configs = [Config(BLOCK_M=32, ...), Config(BLOCK_M=64, ...), ...]
for config in configs:
    timing = bench(kernel.run(*args, **config.all_kwargs()))   # config마다 별도 IR/artifact
best_config = argmin(timing)
```

하나의 IR에 모든 config를 박는 것이 아니라, config마다 별도 IR/artifact를 만들어 각각 실측한다. PIM에서 적용 지점:

1. `tl.constexpr`로 들어오는 `BLOCK_M/N/K`: TTIR → TTSharedIR → PIM Common IR 각 단계에서 자연스럽게 서로 다른 IR로 specialize된다. `build-pim-offload`가 본문 subview에서 BLOCK을 읽어 `tile_plan`에 반영한다.
2. 커널 본문이 참조하지 않는 PIM 전용 노브(`SPLIT_M/N/K`, `NUM_UNITS`, `LOCAL_LANES`): `tl.constexpr`이 아니라 compile option/metadata로 전달되며, `build-pim-offload`가 option에서 직접 읽어 `tile_plan`(SPLIT)·`execution_group`(NUM_UNITS·LOCAL_LANES)에 반영한다.
3. best 탐색 루프는 Python autotuner가 담당하고, PIMBackend는 매 config를 평범한 단일 컴파일로 처리한다.

별도 annotation pass는 두지 않는다 — autotune 값을 IR에 반영하는 일은 `build-pim-offload`가 이미 수행한다.

PIM autotune parameter: `BLOCK_M/N/K`, `NUM_UNITS`, `LOCAL_LANES`, `SPLIT_M/N/K`, `LAYOUT_POLICY`, `PLACEMENT_POLICY`, `TRANSFER_POLICY`, `PIPELINE_POLICY`, `RESIDENCY_POLICY`, `REDUCTION_POLICY`.

| PIM Autotune Param | UPMEM | AiMX |
| --- | --- | --- |
| `NUM_UNITS` | number of DPUs | number of compute units |
| `LOCAL_LANES` | tasklets per DPU | lanes per unit |
| `BLOCK_K` | local reduction chunk | hardware K chunk |
| `PLACEMENT_POLICY` | DPU/MRAM partition | bank/channel mapping |

---

## 17. Capability & Constraint Model

```python
@dataclass
class PIMCapability:
    vendor: str
    arch: str
    max_num_units: int
    max_local_lanes_per_unit: int
    supports_gemm: bool
    supports_gemv: bool
    supports_batched_gemm: bool
    supports_attention_qk: bool
    supports_attention_av: bool
    supports_reduction: bool
    supports_fp32: bool
    supports_fp16: bool
    supports_bf16: bool
    supports_int8: bool
    supports_async_transfer: bool
    supports_pipeline_overlap: bool
    supports_persistent_data: bool
    supports_device_reduce: bool
    memory_spaces: dict

@dataclass
class PIMConstraints:
    block_m_alignment: int
    block_n_alignment: int
    block_k_alignment: int
    min_block_m: int; max_block_m: int
    min_block_n: int; max_block_n: int
    min_block_k: int; max_block_k: int
    max_num_units: int
    max_local_lanes_per_unit: int
    transfer_alignment: int
    placement_granularity: int
    requires_m_multiple: int | None
    requires_n_multiple: int | None
    requires_k_multiple: int | None
```

`#pim.capability_ref<...>` attribute는 공통 표현이고, lowering 시 vendor가 capability 값으로 구체화한다. 적법성 필터(11.4)와 opt-in gate(11.3)가 이 값을 query한다.

---

## 18. Runtime API

```text
PIMRuntime
    ├── PIMDevice
    ├── PIMExecutable
    ├── PIMMemory
    ├── PIMLaunchParams
    ├── PIMStream
    └── VendorRuntime
          ├── UPMEMRuntime
          └── AiMXRuntime
```

```c
typedef struct { const char* vendor; const char* arch; int device_id; } PimDeviceConfig;

typedef struct {
    int B, M, N, K;
    int BLOCK_M, BLOCK_N, BLOCK_K;
    int NUM_UNITS, LOCAL_LANES;
    int SPLIT_M, SPLIT_N, SPLIT_K;
    const char* dtype;
    const char* layout_policy;  const char* placement_policy; const char* transfer_policy;
    const char* pipeline_policy; const char* residency_policy; const char* reduction_policy;
} PimLaunchParams;

int pim_device_open(PimDeviceConfig* config, void** device_out);
int pim_device_query_capability(void* device, void* capability_out);
int pim_device_query_constraints(void* device, void* constraints_out);
int pim_executable_load(void* device, const char* artifact_path, void** executable_out);
int pim_memory_import(void* device, void* host_ptr, size_t size, void** memory_out);
int pim_memory_release(void* memory);
int pim_launch(void* device, void* executable, PimLaunchParams* params, void** buffers);
int pim_synchronize(void* device);
int pim_device_close(void* device);
```

UPMEM의 경우 생성된 host.cpp가 dpu_alloc/load/copy/launch를 직접 수행하는 런타임 glue이므로, launcher는 이 host.so를 컴파일·dlopen하여 `<kernel>_host(...)`를 호출한다.

---

## 19. Device Selection

PIMBackend는 환경변수로 enable하지 않는다. 사용자는 명시적으로 `device="pim"`을 사용한다.

```text
device == "pim"   → PIMBackend 선택
vendor == "upmem" → UPMEMVendorExtension
vendor == "aimx"  → AiMXVendorExtension
그 외             → UnsupportedPIMVendorError
```

---

## 20. Debug & Artifact Dump

PIM 전용 dump 코드를 두지 않고 Triton 내장 dump를 그대로 쓴다. `compile()`의 stage 루프가 등록된 모든 stage 결과를 자동으로 떨구므로, PIM이 추가하는 stage(`pim_common`, `vendor_ir`, `artifact`)도 별도 코드 없이 같이 dump된다.

```bash
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=/tmp/triton_pim
python my_kernel.py
```

```text
$TRITON_DUMP_DIR/<kernel_hash>/
  matmul.ttir   matmul.ttsharedir   matmul.pim_common   matmul.vendor_ir   matmul.pimobj
```

---

## 21. End-to-End Example

```python
@triton.jit
def matmul_kernel(A, B, C,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
    acc = tl.dot(a, b)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)
```

```python
matmul_kernel[grid](A, B, C, M, N, K,
                    BLOCK_M=32, BLOCK_N=64, BLOCK_K=256,
                    device="pim", vendor="upmem")   # 또는 vendor="aimx"
```

```text
TTIR → TTSharedIR → PIM Common IR → Optimized PIM Common IR → Vendor IR → Codegen → Runtime Launch
```

동일한 커널이 `vendor`만 바꿔 UPMEM(host.cpp + DPU kernel) 또는 AiMX(descriptor stream)로 내려간다.

---

## 22. CPU+PIM Hybrid Execution

TTSharedIR은 `program_id`(tile 좌표) 기반으로 tile을 표현하며, GEMM의 경우 타일 간 data dependency가 없다. 따라서 동일한 TTSharedIR에서 CPU binary와 PIM binary를 각각 컴파일한 뒤, 런타임에 tile grid를 분할해 두 하드웨어에서 동시에 실행할 수 있다. (Cinnamon의 공통 IR 계층 + dual-program runtime scheduler 방식의 결합.)

### 22.1 컴파일타임

```text
Triton Python DSL → TTIR → TTSharedIR  ←── heterogeneous dispatch 경계선
                                          ├── [CPUBackend] → CPU binary
                                          └── [PIMBackend] → PIM binary
```

두 backend는 동일한 TTSharedIR 텍스트를 입력으로 독립 lowering한다. `_ttir_to_ttsharedir`는 한 번만 수행하고 결과를 두 pipeline에 전달한다.

### 22.2 런타임

```text
HybridDriver.launch(grid=(M_tiles, N_tiles), ...)
    ├── partition_policy(grid, cpu_cap, pim_cap)
    │     → cpu_range: pid_m ∈ [0, split_m)
    │     → pim_range: pid_m ∈ [split_m, M_tiles)
    ├── Thread 1: cpu_launcher(cpu_range, ...)
    └── Thread 2: pim_launcher(pim_range, ...)   → DPU kernel에 tile_offset, tile_count 전달
```

출력 buffer `C`는 `pid_m` 기준으로 disjoint하게 분할되므로 동기화 없이 안전하다. (`split_k>1` PIM 경로를 섞을 경우엔 partial reduce가 필요하므로, hybrid 분할은 M/N 출력 분할에 한정한다.) HybridDriver를 지원하려면 DPU kernel이 `tile_offset`/`tile_count`를 받아야 하며, `UPMEMToDevice`/host codegen이 이 파라미터를 생성한다.

### 22.3 Python API

```python
matmul_kernel[grid](A, B, C, M, N, K,
                    BLOCK_M=32, BLOCK_N=64, BLOCK_K=256,
                    device="hybrid", pim_vendor="upmem",
                    split_policy="half")   # "half" | "pim_ratio:0.75" | "auto"
```

```text
triton-shared/hybrid/
  HybridDriver.py           # cpu + pim 동시 실행
  HybridPartitionPolicy.py  # tile grid 분할 정책
  HybridBackend.py          # BaseBackend wrapper (optional)
```

---

## 23. Future Vendor Extension Model

새로운 PIM vendor는 PIM Common IR 이후의 lowering만 구현한다.

```text
PIM Common IR → NewVendor IR → NewVendor Passes → NewVendor Codegen → NewVendor Runtime Stub
```

파일 구조:

```text
include/triton-shared/Dialect/PIMVendor/NewVendor/{IR,Transforms,Runtime}/...
lib/Dialect/PIMVendor/NewVendor/{IR,Transforms}/...
lib/Target/NewVendor/...
runtime/pim/vendors/newvendor/...
```

반드시 구현할 pass: `legalize-pim-to-newvendor`, `newvendor-memory-mapping`, `newvendor-execution-mapping`, `newvendor-transfer-lowering`, codegen, runtime-stub.

반드시 제공할 정보: memory space mapping, execution unit mapping, local lane mapping, supported primitives, supported dtype, transfer mechanism, layout/alignment constraints, launch mechanism, generated artifact format.

---

## 24. Final Architecture Summary

```text
Triton PIMBackend
    ↓ TTIR ↓ TTSharedIR ↓
PIM Common IR
    ├── Common PIM Passes
    │     - build-pim-offload (본문 보존 + memory_region/transfer/tile_plan/execution_group 합성)
    │     - select-pim-strategy (적법 전략 표시)
    │     - opt-in: layout-normalize / data-placement / transfer-hoisting /
    │               persistent-data / partial-result-planning / pipeline-overlap
    ↓
Vendor Extension IR
    ├── UPMEM IR
    │     - execution_unit→DPU, local_lane→tasklet
    │     - device_global→MRAM, device_local→WRAM
    │     - host/device split, 본문 충실 번역 → DPU kernel codegen
    └── AiMX IR
          - execution_unit→compute unit, local_lane→internal lane
          - device_global→device memory, device_local→GBuf
          - primitive→descriptor 선택, command stream codegen
```

## 24.1 Final Definition

PIMBackend는 `device="pim"`으로 선택되는 Triton backend이다. TTSharedIR을 입력으로 받아 본문(`scf`/`linalg`/`memref`)을 보존한 vendor-neutral PIM Common IR을 생성하고, 그 주위에 memory_region·transfer·tile_plan(grid·SPLIT)·execution_group을 합성한다. PIM에 필요한 memory/transfer/tile/placement/residency/execution 최적화를 opt-in으로 수행한 뒤, 등록된 Vendor Extension(UPMEM/AiMX)으로 전달한다. UPMEM은 본문을 loops + MRAM↔WRAM DMA로 충실 번역하고, AiMX는 primitive로 descriptor를 선택하여, 각각 실행 가능한 artifact와 runtime launch stub을 생성한다. grid·BLOCK은 Triton 의미 그대로(출력 타일 격자 / 유닛 내부 타일·재사용) 활용되고, SPLIT만 PIM이 추가해 고정 유닛 수에 대한 명시적 분배를 담당한다.
