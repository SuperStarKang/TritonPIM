# Triton PIMBackend Final Architecture Design

## 1. Design Goal

Triton PIMBackend는 Triton 내부에서 PIM을 정식 backend로 인식하고, `device="pim"`을 통해 PIM device를 대상으로 kernel을 compile하고 실행할 수 있도록 하는 backend이다.

PIMBackend는 `triton-shared` repository 안에 **별도 backend 패키지(`pim_backend/`)**로 추가된다. 이는 Triton의 표준 backend 등록 방식이다.

Triton은 모든 backend를 `triton/python/triton/backends/<name>/` 디렉토리(symlink)로 등록한다. 실제로 `nvidia/`, `amd/`, `triton_shared/`(=CPU)가 모두 각자의 `compiler.py` + `driver.py` + `name.conf`를 가진 독립 디렉토리이다. `backends/compiler.py`와 `backends/driver.py`는 **공유 구현이 아니라 추상 base class**(`BaseBackend`, `DriverBase`)이며, 각 backend는 이를 상속한 자신만의 구현을 자기 디렉토리에 둔다.

backend discovery(`backends/__init__.py`의 `_find_concrete_subclasses`)는 각 backend 디렉토리의 `compiler.py`에 `BaseBackend` 서브클래스가 **정확히 하나**만 있을 것을 요구한다. 따라서 PIMBackend도 nvidia/amd/cpu와 똑같이 자신만의 `pim_backend/` 디렉토리를 갖는다. CPUBackend(`triton-shared/backend/`)에 PIMBackend를 추가로 넣는 것은 이 규약 위반이므로 불가능하다.

Python측 `pim_backend/compiler.py`는 `triton-shared-opt` 도구를 subprocess로 호출하는 thin dispatcher 역할만 수행하며, 모든 dialect/conversion/transform 코드는 `triton-shared/include/triton-shared/`, `triton-shared/lib/`에 C++로 구현된다.

PIMBackend는 Triton Python DSL에서 생성된 TTIR을 TTSharedIR로 lowering한 뒤, hardware-specific 정보가 제거된 TTSharedIR을 기반으로 PIM Common IR을 생성한다. 이후 PIM Common IR에서 공통 최적화를 수행하고, 최종적으로 UPMEM IR 또는 AiMX IR로 vendor-specific lowering을 진행한다.

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

또는 backend 객체를 직접 사용할 수 있다 (`GPUTarget`은 Triton의 표준 target 타입; backend 이름이 "pim"이면 PIMBackend가 dispatch된다).

```python
from triton.backends.compiler import GPUTarget
from triton_shared.pim_backend.compiler import PIMBackend

target = GPUTarget(backend="pim", arch="dpu_v1", warp_size=0)
backend = PIMBackend(target)
```

`device="pim"`은 PIMBackend를 선택하는 frontend-level trigger이다.
`vendor="upmem"` 또는 `vendor="aimx"`는 `PIMOptions.vendor` 로 전달되어 PIMBackend 내부에서 vendor-specific lowering pipeline을 선택한다.

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

## 3. Backend Structure

```text
triton.backends
    ├── nvidia
    ├── amd
    ├── cpu
    └── pim
          ├── PIMBackend
          ├── PIMTarget
          ├── PIMDriver
          ├── PIMRuntime
          ├── PIMCapability
          ├── PIMConstraints
          ├── PIMAutotune
          └── VendorExtensionRegistry
                ├── UPMEMVendorExtension
                └── AiMXVendorExtension
```

`pim_generic`은 사용하지 않는다.

새로운 PIM vendor가 추가될 경우 `pim_generic`으로 우회하지 않고, 다음 interface를 구현한 새로운 `VendorExtension`을 등록한다.

```text
VendorExtension
    ├── registerDialect()
    ├── registerPasses()
    ├── buildLoweringPipeline()
    ├── buildCodegenPipeline()
    ├── queryCapability()
    ├── queryConstraints()
    └── createRuntimeLauncher()
```

---

## 4. Design Principle

PIMBackend는 다음 세 계층으로 구성된다.

```text
1. Triton Shared Layer
   - TTIR
   - TTSharedIR

2. PIM Common Layer
   - PIM Common Dialect
   - PIM Common IR
   - PIM Common Optimization Passes

3. Vendor Extension Layer
   - UPMEM Dialect / UPMEM IR / UPMEM Passes / UPMEM Codegen
   - AiMX Dialect / AiMX IR / AiMX Passes / AiMX Codegen
```

PIM Common Layer는 vendor-neutral하다.

UPMEM, AiMX, future PIM vendor들은 PIM Common IR 이후의 lowering만 구현한다.

---

## 5. Directory Structure

PIMBackend는 `triton-shared` repository 안에 추가된다. dialect/conversion 코드의 root namespace는 `triton/Dialect/...`가 아니라 `triton-shared/Dialect/...`이다.

Python backend 코드는 `triton-shared/pim_backend/`에 독립 패키지로 구성한다. nvidia/amd/cpu가 각자 별도 디렉토리인 것과 동일한 표준 방식이며, CPUBackend(`triton-shared/backend/`)와 물리적으로 분리되어 "디렉토리당 BaseBackend 서브클래스 1개" 규약을 만족시킨다. 빌드 시 `TRITON_PLUGIN_DIRS`에 두 경로를 모두 등록한다:

```bash
export TRITON_PLUGIN_DIRS="/path/to/triton-shared;/path/to/triton-shared/pim_backend"
```

각 파일의 역할은 다음과 같다.

```text
triton-shared/
  backend/                          # 기존 CPUBackend 패키지 (변경 없음)
    compiler.py                     #   CPUBackend / CPUOptions — PIM이 _ttir_to_ttsharedir 등을 import해서 재사용
    driver.py                       #   CPUDriver / CPULauncher
    name.conf                       #   "triton_shared" — 이 디렉토리가 등록될 backend 이름

  pim_backend/                      # PIMBackend 전용 독립 패키지 (nvidia/amd와 동일 구조)
    backend/
      compiler.py                   #   PIMOptions(vendor 등 옵션) + PIMBackend(add_stages로 pim pass 파이프라인 정의)
      driver.py                     #   PIMDriver(GPUTarget("pim",...) 반환) + PIMLauncher(pim runtime 호출)
      name.conf                     #   "pim" — device="pim" 라우팅 키
    include/
      ExecutionEngine/              #   launcher가 쓰는 C 런타임 헤더 (CPU와 공유, symlink)

  include/
    triton-shared/
      Dialect/
        TPtr/                       # 기존 dialect
        TritonStructured/           # 기존 dialect (TTSharedIR의 tts.* op 정의)
        TritonTilingExt/            # 기존 dialect
        PIM/                        # ── PIM Common Dialect (vendor-neutral) 선언부 ──
          IR/
            PIMDialect.h            #   "pim" dialect 등록 선언
            PIMOps.h                #   pim.compute / pim.tile_plan / pim.transfer 등 op 선언
            PIMTypes.h              #   MemoryRegionType, MemorySpaceType 등 타입
            PIMAttrs.h              #   Layout/Residency/Role/MappingPolicy/CapabilityRef enum attr
          Transforms/
            Passes.h                #   PIM Common pass(plan 생성/최적화) 선언
          Interfaces/
            PIMVendorExtension.h    #   vendor 공통 인터페이스 (lowering/codegen/capability 훅)
            PIMTargetInterface.h    #   target(arch/device) 추상 인터페이스
            PIMMemoryInterface.h    #   memory space 매핑 인터페이스
            PIMExecutionInterface.h #   execution unit 매핑 인터페이스

        PIMVendor/                  # ── vendor-specific dialect 선언부 ──
          UPMEM/
            IR/
              UPMEMDialect.h        #   "upmem" dialect 등록
              UPMEMOps.h            #   upmem.host_func / device_func / copy_to_global 등
              UPMEMTypes.h          #   MRAM/WRAM 버퍼 타입
              UPMEMAttrs.h          #   DPU/tasklet 관련 attr
            Transforms/
              UPMEMPasses.h         #   legalize/codegen pass 선언
            Runtime/
              UPMEMRuntime.h        #   UPMEM SDK 런타임 wrapper 선언

          AiMX/
            IR/
              AiMXDialect.h         #   "aimx" dialect 등록
              AiMXOps.h             #   aimx.dma_h2d / compute_gemm / descriptor_chain 등
              AiMXTypes.h           #   device buffer / channel 타입
              AiMXAttrs.h           #   bank/channel mapping attr
            Transforms/
              AiMXPasses.h          #   legalize/descriptor codegen pass 선언
            Runtime/
              AiMXRuntime.h         #   AiMX 런타임 wrapper 선언

      Conversion/                   # ── IR 계층 간 변환 pass 선언 ──
        TTSharedToPIR/Passes.h      #   TTSharedIR → PIM Common IR
        PIMToUPMEM/Passes.h         #   PIM Common IR → UPMEM IR
        PIMToAiMX/Passes.h          #   PIM Common IR → AiMX IR

  lib/                              # ── 위 선언들의 실제 C++ 구현 ──
    Dialect/
      PIM/
        IR/
          PIMDialect.cpp            #   dialect 초기화
          PIMOps.cpp                #   op verifier + custom assembly format
          PIMTypes.cpp              #   타입 parse/print
          PIMAttrs.cpp              #   attr parse/print

        Transforms/                 #   TTSharedIR을 분석해 PIM Common IR을 "채워가는" plan pass들
          DetectPIMCandidate.cpp        #     PIM으로 내릴 수 있는 region 탐지
          ExtractTileMetadata.cpp       #     BLOCK_M/N/K, grid, shape, dtype 추출
          RecognizePIMPrimitive.cpp     #     gemm/gemv/attention 패턴 인식
          BuildPIMMemoryPlan.cpp        #     host/device memory_region 생성
          BuildPIMTilePlan.cpp          #     tile_plan(BLOCK·SPLIT) 생성
          BuildPIMTransferPlan.cpp      #     host↔device transfer 삽입
          BuildPIMExecutionPlan.cpp     #     execution_group 매핑
          BuildPIMPlacementPlan.cpp     #     interleave/replicate placement
          BuildPIMResidencyPlan.cpp     #     persistent/ephemeral 결정
          PIMLayoutNormalize.cpp        #     layout을 vendor-neutral form으로 정규화
          PIMDataPlacement.cpp          #     reuse 고려 placement 최적화
          PIMChunkPlanning.cpp          #     K축 chunk 분할 계획
          PIMTransferHoisting.cpp       #     반복 transfer를 loop 밖으로
          PIMPersistentData.cpp         #     weight/KV 등 상주 데이터 표시
          PIMPartialResultPlanning.cpp  #     partial result 저장/reduce 계획
          PIMPipelineOverlap.cpp        #     transfer/compute/store overlap
          # autotune 값(SPLIT_*/NUM_UNITS 등)은 BuildPIMTilePlan/BuildPIMExecutionPlan이
          # option에서 직접 읽으므로 별도 annotation pass를 두지 않는다 (17.0 참고)

      PIMVendor/
        UPMEM/
          IR/
            UPMEMDialect.cpp        #   dialect 구현
            UPMEMOps.cpp            #   op 구현
            UPMEMTypes.cpp
            UPMEMAttrs.cpp
          Transforms/
            LegalizePIMToUPMEM.cpp      #   pim.* → upmem.* 변환 (execution_unit→DPU 등)
            UPMEMMemoryPlacement.cpp    #   device_global→MRAM, device_local→WRAM 배치
            UPMEMExecutionMapping.cpp   #   tile → DPU/tasklet 배정
            UPMEMTransferLowering.cpp   #   pim.transfer → host↔MRAM copy
            UPMEMHostDeviceSplit.cpp    #   host_func / device_func 분리
            UPMEMCodegen.cpp            #   IR → host.cpp + device.c 텍스트 생성
            UPMEMRuntimeStub.cpp        #   pim runtime이 호출할 host wrapper 생성

        AiMX/
          IR/
            AiMXDialect.cpp
            AiMXOps.cpp
            AiMXTypes.cpp
            AiMXAttrs.cpp
          Transforms/
            LegalizePIMToAiMX.cpp       #   pim.* → aimx.* 변환
            AiMXMemoryBankMapping.cpp   #   region → bank/channel 매핑
            AiMXExecutionMapping.cpp    #   tile → compute unit 배정
            AiMXDescriptorSelection.cpp #   compute/dma descriptor 선택
            AiMXDMAScheduling.cpp       #   DMA 순서/overlap 스케줄
            AiMXCommandStreamCodegen.cpp#   descriptor binary stream 생성
            AiMXRuntimeStub.cpp         #   command queue submit wrapper 생성

    Conversion/                     # 변환 pass 구현체 (위 Conversion 헤더 대응)
      TTSharedToPIR/ ...
      PIMToUPMEM/ ...
      PIMToAiMX/ ...

  tools/
    triton-shared-opt/
      RegisterTritonSharedDialects.h  # 모든 PIM/UPMEM/AiMX dialect·pass를 triton-shared-opt에 등록

  runtime/                          # ── 생성된 artifact를 실제 장치에서 실행하는 런타임 ──
    pim/
      include/
        pim_runtime.h               #   vendor-neutral C API (pim_device_open/launch/...)
        pim_device.h                #   device 핸들
        pim_launch.h                #   launch 파라미터(grid, BLOCK, SPLIT 등)
        pim_executable.h            #   load된 artifact 핸들
        pim_memory.h                #   host buffer import/release
      lib/
        PIMRuntime.cpp              #   vendor 이름 보고 적절한 VendorRuntime으로 dispatch
        PIMDevice.cpp
        PIMExecutable.cpp
        PIMLaunch.cpp
        PIMMemory.cpp
      vendors/
        upmem/
          UPMEMRuntime.cpp          #   dpu_alloc/dpu_load 등 SDK 호출
          UPMEMLauncher.cpp         #   dpu_copy_to/dpu_launch/dpu_copy_from 시퀀스
        aimx/
          AiMXRuntime.cpp
          AiMXLauncher.cpp          #   descriptor stream submit + wait
```

CPUBackend와 동일하게 IR 변환은 Python이 아니라 `triton-shared-opt` 실행 파일을 subprocess로 호출하여 수행한다. 따라서 새 PIM dialect/pass는 모두 `tools/triton-shared-opt/RegisterTritonSharedDialects.h`에 등록되어야 한다.

`pim_backend/backend/compiler.py`는 `triton-shared/backend/compiler.py`의 공유 유틸리티(`_ttir_to_ttsharedir`, `_get_triton_shared_opt_path` 등)를 import해서 재사용한다.

---

## 6. Python Backend API

PIMBackend는 Triton의 `BaseBackend` 규약을 따른다. `BaseBackend.__init__`과 `supports_target`은 모두 `GPUTarget`을 인자로 받는다 (Triton legacy 네이밍 — 실제로는 모든 backend의 공통 Target 타입). PIM 고유 정보(`vendor`, `arch`, `device_id`)는 `PIMOptions`에 둔다.

### 6.1 PIMOptions

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
    vendor: str = None           # "upmem" | "aimx" — 누락 시 PIMBackend가 명시적 에러
    device_id: int = 0
    dump_ir: bool = False
    dump_artifacts: bool = False
    dump_dir: str = ""

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

지원 vendor:

```text
upmem
aimx
```

새로운 vendor는 `VendorExtensionRegistry`에 등록한다.

---

### 6.2 PIMBackend

```python
from triton.backends.compiler import BaseBackend, GPUTarget

class PIMBackend(BaseBackend):
    # vendor마다 binary_ext가 달라질 수 있으므로 vendor extension에서 결정
    binary_ext = 'pimobj'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'pim'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        # vendor extension은 parse_options() 이후 options.vendor를 보고 lazy하게 결정한다.
        # (CPUBackend와 마찬가지로 IR 변환은 모두 triton-shared-opt subprocess가 수행하므로
        # backend 객체는 가벼운 dispatcher 역할만 한다.)

    def parse_options(self, opts) -> PIMOptions:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in PIMOptions.__dataclass_fields__.keys() if k in opts})
        return PIMOptions(**args)

    # CPUBackend와 동일: IR 변환은 triton-shared-opt subprocess가 수행하므로
    # Python MLIRContext에 별도 dialect를 로드하지 않는다.
    def load_dialects(self, ctx):
        return

    # CPUBackend.add_stages 시그니처와 동일 (language 인자 포함).
    def add_stages(self, stages, options, language):
        stages["ttir"]       = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _ttir_to_ttsharedir(src)
        stages["pim_common"] = lambda src, metadata: _ttsharedir_to_pim_common(src, options)
        stages["vendor_ir"]  = lambda src, metadata: _pim_common_to_vendor(src, options)
        stages["artifact"]   = lambda src, metadata: _vendor_codegen(src, metadata, options)

    # `pack_metadata`, `get_codegen_implementation`, `hash`, `get_module_map`는 CPUBackend와
    # 동일한 형태로 구현한다 (생략).
```

`_ttir_to_ttsharedir`는 `triton-shared/backend/compiler.py`에서 import해서 재사용한다. 나머지 `_*` helper는 CPUBackend의 `_ttsharedir_to_llir`과 동일하게 `triton-shared-opt` 또는 `mlir-opt`를 subprocess로 호출하는 thin wrapper이다. Pass 목록은 다음과 같다.

**TTShared → PIM Common** (`_ttsharedir_to_pim_common`):

```text
detect-pim-candidate
extract-tile-metadata
recognize-pim-primitive
build-pim-memory-plan
build-pim-tile-plan
build-pim-transfer-plan
build-pim-execution-plan
build-pim-placement-plan
build-pim-residency-plan
convert-ttshared-to-pim
canonicalize
cse
```

**PIM Common optimization** (`_ttsharedir_to_pim_common` 후반 또는 별도 stage):

```text
pim-layout-normalize
pim-data-placement
pim-chunk-planning
pim-transfer-hoisting
pim-persistent-data
pim-partial-result-planning
pim-pipeline-overlap
canonicalize
cse
```

**PIM Common → Vendor IR** (`_pim_common_to_vendor`): `options.vendor` 값에 따라 `legalize-pim-to-upmem` 또는 `legalize-pim-to-aimx` 등을 호출.

---

## 7. Vendor Extension Registry

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
```

등록:

```python
VendorExtensionRegistry.register("upmem", UPMEMVendorExtension())
VendorExtensionRegistry.register("aimx", AiMXVendorExtension())
```

새 vendor 추가 예시:

```python
VendorExtensionRegistry.register("new_vendor", NewVendorExtension())
```

`pim_generic`은 존재하지 않는다.
지원하지 않는 vendor는 명확하게 error를 발생시킨다.

---

## 8. Vendor Extension Interface

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

# 9. IR Layering Model

Cinnamon 스타일의 계층적 lowering 구조를 참고하여 다음 IR hierarchy를 사용한다.

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

각 계층의 책임은 다음과 같다.

| Layer         | 역할                                                             |
| ------------- | -------------------------------------------------------------- |
| TTIR          | Triton frontend에서 생성되는 기본 IR                                   |
| TTSharedIR    | hardware-specific execution hierarchy를 제거한 tile/block-level IR |
| PIM Common IR | PIM 공통 실행 모델 표현                                                |
| UPMEM IR      | UPMEM DPU/MRAM/WRAM/host-device split 표현                       |
| AiMX IR       | AiMX descriptor/DMA/channel/bank command 표현                    |

---

# 10. TTSharedIR

TTSharedIR은 다음 정보를 유지한다.

```text
program_id
tile index
block pointer
tile load
tile store
dot
reduce
mask
layout hint
block shape
memory access pattern
```

예시:

```mlir
ttshared.func @matmul(%A, %B, %C, %M, %N, %K) {
  %pid_m = ttshared.program_id x
  %pid_n = ttshared.program_id y

  %a_tile = ttshared.load_tile %A {
    shape = [BLOCK_M, BLOCK_K],
    layout = "row_major"
  }

  %b_tile = ttshared.load_tile %B {
    shape = [BLOCK_K, BLOCK_N],
    layout = "row_major"
  }

  %acc = ttshared.dot %a_tile, %b_tile {
    out_shape = [BLOCK_M, BLOCK_N]
  }

  ttshared.store_tile %C, %acc
}
```

---

# 11. PIM Common IR

PIM Common IR은 특정 vendor에 종속되지 않는 PIM 공통 실행 모델이다.

PIM Common IR에서 사용하지 않는 표현:

```text
tasklet
DPU
FPGA
channel-specific descriptor
MRAM
WRAM
GBuf
OutReg
```

PIM Common IR에서 사용하는 중립 표현:

```text
execution_unit
execution_group
local_lane
memory_space
memory_region
placement
transfer
compute
data_residency
tile_plan
chunk_plan
partial_result
pipeline
launch
```

---

## 11.1 PIM Common Operations

```text
pim.func
pim.memory_space
pim.memory_region
pim.tile_plan
pim.chunk_plan
pim.execution_group
pim.placement
pim.data_residency
pim.transfer
pim.compute
pim.partial_result
pim.reduce
pim.pipeline
pim.launch
```

---

## 11.2 `pim.func`

```mlir
pim.func @matmul(%A, %B, %C, %M, %N, %K) attributes {
  op = "gemm",
  dtype = "bf16",
  accumulation = "fp32"
} {
  ...
}
```

속성:

```text
op
dtype
accumulation
target_device_class
requires_explicit_transfer
allows_persistent_data
allows_partial_result
```

---

## 11.3 `pim.memory_space`

```mlir
%host = pim.memory_space {
  name = "host"
}

%global = pim.memory_space {
  name = "device_global"
}

%local = pim.memory_space {
  name = "device_local"
}

%scratch = pim.memory_space {
  name = "device_scratch"
}
```

공통 memory space:

```text
host
device_global
device_local
device_scratch
device_register
```

vendor mapping:

| PIM Common        | UPMEM          | AiMX               |
| ----------------- | -------------- | ------------------ |
| `host`            | CPU DRAM       | CPU DRAM           |
| `device_global`   | MRAM           | FPGA/device memory |
| `device_local`    | WRAM           | GBuf/local buffer  |
| `device_scratch`  | WRAM temporary | scratch buffer     |
| `device_register` | DPU register   | MAC/OutReg         |

---

## 11.4 `pim.memory_region`

```mlir
%A_region = pim.memory_region %A {
  space = "host",
  layout = "row_major",
  role = "input_activation"
}

%B_region = pim.memory_region %B {
  space = "device_global",
  layout = "packed_by_output_tile",
  role = "weight",
  residency = "persistent"
}

%C_region = pim.memory_region %C {
  space = "host",
  layout = "row_major",
  role = "output"
}
```

속성:

```text
space
layout
role
residency
alignment
capacity_hint
```

---

## 11.5 `pim.tile_plan`

```mlir
%tile_plan = pim.tile_plan {
  M = %M,
  N = %N,
  K = %K,

  block_m = 32,
  block_n = 64,
  block_k = 256,

  output_tile = [32, 64],
  reduction_tile = 256,

  split_m = 1,
  split_n = 64,
  split_k = 1
}
```

### 11.5.1 BLOCK vs SPLIT — 헷갈리지 않게

**행렬 규약**: 이 문서 전체에서 GEMM은 `A(M,K) × B(K,N) → C(M,N)` 이다. row-major 기준.

`tile_plan`에는 성격이 전혀 다른 두 종류의 정보가 들어간다. 절대 섞어 쓰면 안 된다.

```text
전체 (M,N,K) 문제
   │
   ① SPLIT  ── 유닛 "간" 분배 (inter-unit, 바깥 레벨)
   │           전체 출력 grid를 여러 execution_unit(DPU)에 어떻게 나눠줄까
   ▼
각 유닛이 맡은 부분 행렬
   │
   ② BLOCK  ── 유닛 "내부" 처리 (intra-unit, 안쪽 레벨)
   │           그 부분을 local memory(WRAM/GBuf)에 맞는 타일로 끊어 계산
   ▼
BLOCK_M × BLOCK_N 출력 타일 한 조각
```

| 파라미터 | 레벨 | 의미 | 무엇을 결정하나 | 값의 출처 |
| --- | --- | --- | --- | --- |
| `BLOCK_M` | intra-unit (안쪽) | 출력 타일 행 수 (= A에서 읽는 행 수) | 한 유닛이 한 번에 다루는 타일 크기 | 커널 `tl.constexpr` → IR |
| `BLOCK_N` | intra-unit (안쪽) | 출력 타일 열 수 (= B에서 읽는 열 수) | 〃 | 커널 `tl.constexpr` → IR |
| `BLOCK_K` | intra-unit (안쪽) | reduction chunk 크기 | local memory에 한 번에 올릴 K 조각 | 커널 `tl.constexpr` → IR |
| `SPLIT_M` | inter-unit (바깥) | M축을 몇 partition으로 나눌까 | 출력 grid → 유닛 분배 | compile option / autotune |
| `SPLIT_N` | inter-unit (바깥) | N축을 몇 partition으로 나눌까 | 〃 | compile option / autotune |
| `SPLIT_K` | inter-unit (바깥) | K축을 몇 partition으로 나눌까 (>1이면 partial result + reduce 필요) | 〃 | compile option / autotune |

핵심 구분:

- **BLOCK = "각 bank(WRAM/GBuf/register)를 최대한 채우는 지역 타일 크기"** — 한 유닛 내부 작업 단위.
- **SPLIT = "그 바깥에서, 일을 여러 유닛에 어떻게 쪼개 분배할까"** — 병렬도/매핑 결정.

즉 **SPLIT이 먼저 바깥에서 유닛에 분배하고, BLOCK이 그 안에서 local memory에 맞춰 타일링**한다. (SPLIT은 "BLOCK 안에서 쪼개는 것"이 아니다 — 방향이 반대다.)

메모리 계층과의 대응 (UPMEM / AiMX):

| 정보 | 대응 memory_space | UPMEM | AiMX |
| --- | --- | --- | --- |
| `BLOCK_K` load tile | `device_local` | WRAM | GBuf |
| `BLOCK_M×BLOCK_N` accumulator | `device_register` / `device_local` | DPU register / WRAM | OutReg / local buffer |
| 전체 A,B,C 상주 | `device_global` | MRAM | device memory |
| `SPLIT_*` 분배 | `execution_group` | DPU 개수·배정 | compute unit 배정 |

`BuildPIMTilePlan`은 IR에서 `BLOCK_*`를, option에서 `SPLIT_*`를 읽어 위 `tile_plan`을 채운다. `BuildPIMExecutionPlan`은 `SPLIT_*`와 capability(NUM_UNITS 등)를 보고 실제 유닛 매핑(`execution_group`)을 만든다.

---

## 11.6 `pim.chunk_plan`

```mlir
%chunk_plan = pim.chunk_plan {
  axis = "K",
  chunk_size = 256,
  num_chunks = ceildiv(%K, 256),
  tail_policy = "pad"
}
```

지원 tail policy:

```text
pad
mask
reject
vendor_defined
```

---

## 11.7 `pim.execution_group`

```mlir
%eg = pim.execution_group {
  num_units = #pim.capability_ref<"max_num_units">,
  local_lanes_per_unit = #pim.capability_ref<"max_local_lanes_per_unit">,
  mapping_policy = "output_n_partitioned",
  stationary_data = "B"
}
```

`num_units`, `local_lanes_per_unit` 같은 vendor-specific 상수는 IR에 직접 박지 않고 `PIMCapability`를 reference하는 attribute로 표현한다. 실제 값은 lowering 시점에 capability에서 읽어 구체화된다. (UPMEM 예: `max_num_units = 1856`, AiMX 예: vendor 별로 다름.)

중립 표현:

| Neutral Term      | 의미                            |
| ----------------- | ----------------------------- |
| `execution_unit`  | 독립적으로 tile을 수행하는 PIM 실행 단위    |
| `execution_group` | 여러 execution unit의 집합         |
| `local_lane`      | 하나의 execution unit 내부 병렬 lane |
| `mapping_policy`  | tile을 execution unit에 배정하는 정책 |

vendor mapping:

| PIM Common             | UPMEM            | AiMX                               |
| ---------------------- | ---------------- | ---------------------------------- |
| `execution_unit`       | DPU              | PU / compute lane group            |
| `execution_group`      | DPU set          | channel group / compute unit group |
| `local_lane`           | tasklet          | lane / internal parallel slot      |
| `num_units`            | number of DPUs   | number of active compute units     |
| `local_lanes_per_unit` | tasklets per DPU | lanes per unit                     |

---

## 11.8 `pim.placement`

```mlir
pim.placement %B_region {
  axis = "N",
  policy = "interleave",
  granularity = 256,
  target = "device_global"
}
```

지원 placement policy:

```text
interleave
blocked
replicated
output_tile_interleaved
reduction_chunk_interleaved
vendor_preferred
```

---

## 11.9 `pim.data_residency`

```mlir
pim.data_residency %B_region {
  residency = "persistent",
  lifetime = "multi_launch",
  update_policy = "explicit"
}
```

지원 residency:

```text
ephemeral
persistent
streaming
cached
```

---

## 11.10 `pim.transfer`

```mlir
pim.transfer %A_region {
  src = "host",
  dst = "device_global",
  async = true,
  transform = "none"
}

pim.transfer %B_region {
  src = "host",
  dst = "device_global",
  async = true,
  transform = "pack_by_output_tile",
  residency = "persistent"
}
```

속성:

```text
src
dst
async
transform
granularity
overlap_group
```

---

## 11.11 `pim.compute`

```mlir
%partial = pim.compute @gemm(%A_region, %B_region, %C_region)
  using %eg, %tile_plan, %chunk_plan {
    input_dtype = "bf16",
    accumulation_dtype = "fp32",
    output_dtype = "bf16"
}
```

지원 primitive:

```text
gemm
gemv
batched_gemm
attention_qk
attention_av
reduction
elementwise
```

---

## 11.12 `pim.partial_result`

```mlir
%partial_c = pim.partial_result {
  shape = [BLOCK_M, BLOCK_N],
  accumulation_dtype = "fp32",
  owner = "execution_unit"
}
```

---

## 11.13 `pim.reduce`

```mlir
%C_final = pim.reduce %partial_c {
  axis = "K",
  method = "device_reduce"
}
```

지원 method:

```text
none
host_reduce
device_reduce
tree_reduce
accumulate_in_place
```

---

## 11.14 `pim.pipeline`

```mlir
pim.pipeline {
  stage0 = "transfer_next_chunk",
  stage1 = "compute_current_chunk",
  stage2 = "store_previous_result"
}
```

---

## 11.15 `pim.launch`

```mlir
pim.launch @matmul {
  device = "pim",
  vendor = "aimx",
  device_id = 0
}
```

---

# 12. PIM Common IR Example

```mlir
pim.func @matmul(%A, %B, %C, %M, %N, %K) attributes {
  op = "gemm",
  dtype = "bf16",
  accumulation = "fp32"
} {
  %eg = pim.execution_group {
    num_units = #pim.capability_ref<"max_num_units">,
    local_lanes_per_unit = #pim.capability_ref<"max_local_lanes_per_unit">,
    mapping_policy = "output_n_partitioned",
    stationary_data = "B"
  }

  %A_region = pim.memory_region %A {
    space = "host",
    layout = "row_major",
    role = "input_activation",
    residency = "ephemeral"
  }

  %B_region = pim.memory_region %B {
    space = "device_global",
    layout = "packed_by_output_tile",
    role = "weight",
    residency = "persistent"
  }

  %C_region = pim.memory_region %C {
    space = "host",
    layout = "row_major",
    role = "output",
    residency = "ephemeral"
  }

  %tile_plan = pim.tile_plan {
    M = %M,
    N = %N,
    K = %K,
    block_m = 32,
    block_n = 64,
    block_k = 256,
    output_tile = [32, 64],
    reduction_tile = 256,
    split_m = 1,
    split_n = 64,
    split_k = 1
  }

  %chunk_plan = pim.chunk_plan {
    axis = "K",
    chunk_size = 256,
    num_chunks = ceildiv(%K, 256),
    tail_policy = "pad"
  }

  pim.placement %B_region {
    axis = "N",
    policy = "output_tile_interleaved",
    granularity = 256,
    target = "device_global"
  }

  pim.data_residency %B_region {
    residency = "persistent",
    lifetime = "multi_launch",
    update_policy = "explicit"
  }

  pim.transfer %A_region {
    src = "host",
    dst = "device_global",
    async = true,
    transform = "none"
  }

  pim.transfer %B_region {
    src = "host",
    dst = "device_global",
    async = true,
    transform = "pack_by_output_tile",
    residency = "persistent"
  }

  pim.pipeline {
    stage0 = "transfer_next_chunk",
    stage1 = "compute_current_chunk",
    stage2 = "store_previous_result"
  }

  %partial = pim.compute @gemm(%A_region, %B_region, %C_region)
    using %eg, %tile_plan, %chunk_plan {
      input_dtype = "bf16",
      accumulation_dtype = "fp32",
      output_dtype = "bf16"
    }

  %result = pim.reduce %partial {
    axis = "K",
    method = "none"
  }

  pim.transfer %C_region {
    src = "device_global",
    dst = "host",
    async = false,
    transform = "none"
  }

  pim.launch @matmul {
    device = "pim",
    vendor = "aimx",
    device_id = 0
  }
}
```

---

# 13. PIM Common Pass Pipeline

## 13.0 공통 pass의 경계 (가장 중요)

공통(PIM Common) pass는 **값을 최적화하지 않는다.** 오직 다음 세 가지만 한다.

```text
1. 구조 생성 (neutral structure)
     주어진 값(constexpr/option)을 IR에 기록만 한다. "무엇이 최적인가"를 결정하지 않는다.
     예: build-pim-tile-plan은 block/split을 "고르지" 않고, 주어진 값을 적는다.

2. intent 표현 (policy enum)
     분산/매핑/잔류를 vendor-neutral enum으로 표현한다. 구체 산술은 하지 않는다.
     예: placement="interleave"는 "의도"일 뿐, 실제 bank 오프셋은 vendor가 정한다.

3. capability-gated 결정
     vendor 사실이 필요하면 그 vendor의 PIMCapability를 query한다.
     (컴파일 시점에 vendor는 이미 하나로 정해져 있으므로 vendor-aware하게 동작 가능.)
     예: pim-pipeline-overlap은 supports_pipeline_overlap일 때만 동작한다.
```

따라서 다음은 공통 pass의 일이 **아니다.**

```text
- 최적값 선택 (어느 block/split/placement가 빠른가)  → autotune이 실측으로 가린다 (17장)
- 구체 실현 (실제 bank/channel/register 산술)         → vendor lowering이 한다 (14·15장)
- 적법성 판단 (block_k가 WRAM에 들어가나)              → PIMConstraints 기반 적법성 필터 (13.4)
```

한 줄 요약: **공통 = "구조 + 의도 + (필요시) capability gate"**, **autotune = "값 선택"**, **vendor = "구체 실현"**.

## 13.1 TTSharedIR to PIM Common IR

```text
detect-pim-candidate
extract-tile-metadata
recognize-pim-primitive
build-pim-memory-plan
build-pim-tile-plan
build-pim-transfer-plan
build-pim-execution-plan
build-pim-placement-plan
build-pim-residency-plan
convert-ttshared-to-pim
canonicalize
cse
```

## 13.2 PIM Common Optimization

```text
pim-layout-normalize
pim-data-placement
pim-chunk-planning
pim-transfer-hoisting
pim-persistent-data
pim-partial-result-planning
pim-pipeline-overlap
canonicalize
cse
```

## 13.3 Pass Responsibilities

꼬리표 의미 (13.0 기준):
- **[구조]** — 주어진 값/패턴을 기록·표현만. 결정/최적화 없음.
- **[intent]** — vendor-neutral policy enum만 표현. 구체 산술은 vendor가.
- **[gate]** — vendor capability를 query해 동작 여부/형태를 결정 (vendor-aware).
- **[vendor가 실현]** — 이 공통 pass가 만든 intent를 실제 산술로 바꾸는 책임은 vendor lowering에 있음.

| Pass                          | 분류 | 역할 (공통 layer 한정) | vendor가 실현하는 부분 |
| ----------------------------- | --- | ------------------- | ------------------- |
| `detect-pim-candidate`        | [구조] | PIM lowering 가능 region 탐지·표시 | — |
| `extract-tile-metadata`       | [구조] | `BLOCK_M/N/K`, grid, shape, dtype 추출(기록) | — |
| `recognize-pim-primitive`     | [구조] | gemm/gemv/attention/reduction 패턴 분류 | — |
| `build-pim-memory-plan`       | [구조] | host/device memory_region 생성 | — |
| `build-pim-tile-plan`         | [구조] | 주어진 block/split을 tile_plan에 **기록** (최적화 X) | 실제 타일 micro-fit |
| `build-pim-transfer-plan`     | [구조] | transfer **골격** 삽입 | 실제 copy 방식(DMA/dpu_copy) |
| `build-pim-execution-plan`    | [intent] | mapping_policy **enum**만 표현 | 실제 DPU/channel 배정 |
| `build-pim-placement-plan`    | [intent] | placement policy **enum**만 표현 | bank/channel 오프셋 산술 (`UPMEMMemoryPlacement` / `AiMXMemoryBankMapping`) |
| `build-pim-residency-plan`    | [intent] | residency enum 표현 | 실제 상주 버퍼 관리 |
| `pim-layout-normalize`        | [구조] | layout **표현 정규화만** (중복 transpose 제거, 단일 canonical form) | 물리 layout 변환(`pack_by_*`) |
| `pim-data-placement`          | [intent] | reuse 고려 placement **의도** 갱신 | 구체 분산 |
| `pim-chunk-planning`          | [구조] | `num_chunks=ceildiv(K,BLOCK_K)` **산술만** | register/WRAM fitting |
| `pim-transfer-hoisting`       | [구조] | 반복 transfer를 상위 scope로 이동 | — |
| `pim-persistent-data`         | [intent] | weight/KV를 persistent로 **표시** | 실제 상주 구현 |
| `pim-partial-result-planning` | [구조] | partial/reduce **구조** 생성 (split_k>1일 때) | 실제 reduce 구현 |
| `pim-pipeline-overlap`        | [gate] | `supports_pipeline_overlap`일 때만 `pim.pipeline` 삽입 | 실제 async/double-buffer |

> 적법성("BLOCK_K가 WRAM에 들어가나" 등)은 이 표의 어느 pass도 판단하지 않는다. `PIMConstraints` 기반 적법성 필터(13.4.5)가 담당한다.

---

## 13.4 Execution Strategy Selection (plan dispatch)

> **전제**: Phase 2~5에서는 단일 전략(no-split)만 구현하고, 이 절의 다중 전략 선택은 Phase 8 이후 확장으로 도입한다. 처음부터 일반화하지 않는다 (premature abstraction 회피).

### 13.4.1 동기 — PGEMM의 분기 패턴

UPMEM용 손수 작성 라이브러리(PGEMM `pgemm_float.c`)는 `(splitM, splitK, splitN)` 조합에 따라 서로 다른 커널 구현으로 분기한다.

```c
if      (splitK==1 && splitM==1 && splitN==1)  basic(...);     // 분할 없음
else if (splitK>1  && splitM==1)               basic_k(...);   // K-split → partial reduce
else if (splitN>1)                             basic_n(...);   // N-split
else if (splitM>1) { SWAP(M,N); SWAP(splitM,splitN);           // M-split = N-split + 전치
                     basic_n(B,A,...); transpose_C(...); }
```

각 분기는 **데이터 layout, DPU 매핑, reduction 방식이 다른 별개의 실행 전략**이다. 즉 "맞는 plan"은 split factor와 hardware에 따라 달라진다.

### 13.4.2 전략의 IR 표현

이 분기 로직을 if-else로 코드에 박지 않고, **전략을 선택해 vendor-neutral attribute로 표현**한다. 한 전략 = 다음 3개의 묶음이다.

```text
strategy = {
  mapping_policy   :  output_n_partitioned | output_m_partitioned | reduction_k_partitioned | none
  reduction        :  none | host_reduce | device_reduce | tree_reduce
  placement        :  blocked | interleave | replicated | output_tile_interleaved
}
```

| PGEMM 분기 | PIM 전략 (mapping_policy / reduction) |
| --- | --- |
| `basic` (no split) | `none` / `none` |
| `basic_n` (N-split) | `output_n_partitioned` / `none` |
| `basic_k` (K-split) | `reduction_k_partitioned` / `device_reduce` 또는 `host_reduce` |
| `M-split` (전치 후 N-split) | `output_m_partitioned` / `none` (layout 전치 포함) |

### 13.4.3 어디서 결정하고 어디서 구체화하나 — 선택은 Common, 실현은 Vendor

전략은 **선택(WHAT)** 과 **실현(HOW)** 으로 나뉜다.

| 단계 | 위치 | 하는 일 |
| --- | --- | --- |
| 전략 **선택 + 표현** | **PIM Common IR** | "K-split + device_reduce, N-partitioned로 간다"를 결정하고 `mapping_policy`/`reduction`/`placement` attribute로 박음 |
| 전략 **구체화(legalize)** | **vendor IR (UPMEM/AiMX)** | 그 전략을 실제로 실현: DPU 개수·MRAM 오프셋·reduce 구현 (UPMEM) / channel·OutReg 누적·descriptor (AiMX) |

```text
[Common IR]  pim.execution_group { mapping_policy="reduction_k_partitioned" }
             pim.reduce { method="device_reduce" }
                  │
     ┌────────────┴────────────┐
     ▼ UPMEM lowering           ▼ AiMX lowering
 "K를 4 DPU에 나누고            "K를 4 channel에 나누고
  partial을 MRAM에,             partial을 OutReg에 누적,
  host_reduce로 합산"           descriptor_chain accumulate로 합산"
```

**선택을 Common에 두는 이유**: ① `mapping_policy`/`reduction`/`placement`는 이미 vendor-neutral 개념이다(11.7/11.13/11.8). ② 같은 TTSharedIR이 어느 vendor로도 갈 수 있으므로 선택 로직을 vendor마다 중복하면 안 된다. ③ Common 최적화 pass(transfer-hoisting, pipeline-overlap)가 단일 표현 위에서 동작해야 한다.

### 13.4.4 "일찍 고르면 어떤 vendor엔 손해" 문제 — vendor-aware 선택으로 해결

vendor마다 유리/가능한 전략이 다르다. 그렇다고 Common에서 고르는 것이 손해는 아니다. **결정의 위치(Common)와 결정을 좌우하는 정보(vendor)는 별개**이기 때문이다.

핵심 사실: **컴파일 시점에 vendor는 이미 하나로 정해져 있다.** `options.vendor`를 `_ttsharedir_to_pim_common`이 알고 있으므로, 선택 pass가 그 vendor의 정보를 보고 고른다. 한 번의 컴파일은 항상 한 vendor만 대상으로 한다.

따라서 `SelectPIMExecutionStrategy`는 **Common 레벨 pass이지만 vendor-aware**하다. 입력으로 그 vendor의 `PIMCapability`/`PIMConstraints`를 query한다.

```text
PIMVendorExtension
  └── queryCapability() / queryConstraints()   # supports_device_reduce, max_num_units, WRAM 크기, alignment 등
```

→ 로직은 공통 한 곳, 그 로직을 움직이는 데이터는 vendor별. vendor 차이가 **코드 중복이 아니라 capability 데이터로** 들어온다.

### 13.4.5 선택 방법 — 적법성 필터 + autotune (cost model 없음)

**(1) 적법성 필터 (deterministic, vendor-aware)** — PGEMM의 if-else에 해당.
`SelectPIMExecutionStrategy`가 `tile_plan`의 split factor + vendor의 `PIMCapability`/`PIMConstraints`를 보고 **적용 가능한 전략만** 추린다.

- `SPLIT_K==1` → K-split 전략 제외
- `block_k`가 WRAM 초과 → 해당 BLOCK 조합 제외
- `supports_device_reduce=false` (UPMEM) → `device_reduce` 전략 제외 (단, `host_reduce` K-split은 여전히 적법)

**(2) 적법한 전략 중 best 결정**

- **`@triton.autotune` 사용 시**: 적법한 전략들을 각각 별도 config로 컴파일·실행·**벤치마크하여 empirical best**를 고른다(17.0 표준 autotuner 경로). **미리 추정하지 않고 측정**한다 — 그래서 cost model이 필요 없다.
  - 예: UPMEM에서 N-split과 host_reduce K-split이 둘 다 적법하면, autotuner가 둘 다 돌려보고 빠른 쪽을 택한다.
- **autotune 미사용 시**: 단순 기본 규칙으로 적법 전략 하나를 확정한다 (예: "no-split이 적법하면 no-split, 아니면 N-split"). 이는 휴리스틱 default일 뿐이며, 실제 최적화는 autotune이 담당한다.

> **cost model은 두지 않는다.** 전략 간 우열은 정적 비용 추정이 아니라 autotune 실측으로 가린다. 정적으로는 "적법한가"만 판단한다.

초기 구현은 (1)의 필터 + 단일 default 전략만으로 시작하고, autotune 다중 전략은 나중에 얹는다.

### 13.4.6 확장성

새 전략 추가 = "전략 레지스트리에 항목 하나 + 그 전략의 lowering 한 벌" 추가이며, 기존 전략 코드를 건드리지 않는다 (PGEMM처럼 if-else를 늘려가는 대신).

```text
PIMExecutionStrategy (interface, vendor-neutral)
  ├── isApplicable(tile_plan, capability, constraints) -> bool   // 적법성만 판단 (cost 추정 없음)
  └── applyToIR(builder)                                         // execution_group/reduce/placement attribute 확정
```

전략 목록은 vendor-neutral하다. vendor IR은 확정된 전략 attribute를 자기 방식으로 legalize만 한다 (예: `reduction_k_partitioned` → UPMEM은 host_reduce, AiMX는 descriptor accumulate). "WHAT"만 Common에서 vendor-aware하게 고르고, "HOW"는 vendor lowering이 맡는다.

---

# 14. UPMEM Vendor Extension

## 14.1 UPMEM Mapping

| PIM Common IR     | UPMEM IR                 |
| ----------------- | ------------------------ |
| `execution_unit`  | DPU                      |
| `execution_group` | DPU set                  |
| `local_lane`      | tasklet                  |
| `device_global`   | MRAM                     |
| `device_local`    | WRAM                     |
| `device_scratch`  | WRAM temporary           |
| `device_register` | DPU register             |
| `pim.transfer`    | host ↔ MRAM copy         |
| `pim.compute`     | DPU kernel               |
| `pim.launch`      | `dpu_launch()`           |
| `pim.reduce`      | host reduce / DPU reduce |

---

## 14.2 UPMEM Dialect Operations

```text
upmem.module
upmem.host_func
upmem.device_func
upmem.alloc_units
upmem.load_program
upmem.copy_to_global
upmem.copy_from_global
upmem.launch
upmem.unit_id
upmem.local_lane_id
upmem.global_load
upmem.global_store
upmem.local_alloc
upmem.local_load
upmem.local_store
upmem.local_gemm
upmem.local_gemv
upmem.barrier
upmem.host_reduce
upmem.device_reduce
```

---

## 14.3 UPMEM Lowering Pipeline

```text
PIM Common IR
  ↓
legalize-pim-to-upmem
  ↓
UPMEM IR
  ↓
upmem-memory-placement
upmem-execution-mapping
upmem-transfer-lowering
upmem-host-device-split
  ↓
Optimized UPMEM IR
  ↓
upmem-codegen
upmem-runtime-stub
  ↓
Host C/C++ Code + DPU Kernel Code
```

---

## 14.4 UPMEM IR Example

```mlir
upmem.host_func @matmul_host(%A, %B, %C) {
  %units = upmem.alloc_units {
    num_units = 2048
  }

  upmem.load_program %units, @matmul_device_kernel

  upmem.copy_to_global %units, %A {
    dst = "global_a",
    layout = "row_major"
  }

  upmem.copy_to_global %units, %B {
    dst = "global_b",
    layout = "packed_by_output_tile"
  }

  upmem.launch %units, @matmul_device_kernel {
    async = false
  }

  upmem.copy_from_global %units, %C {
    src = "global_c"
  }
}
```

```mlir
upmem.device_func @matmul_device_kernel(%global_a, %global_b, %global_c) {
  %unit = upmem.unit_id
  %lane = upmem.local_lane_id

  %a_tile = upmem.global_load %global_a {
    shape = [32, 256],
    dst = "local"
  }

  %b_tile = upmem.global_load %global_b {
    shape = [256, 64],
    dst = "local"
  }

  %acc = upmem.local_gemm %a_tile, %b_tile {
    accumulation = "fp32"
  }

  upmem.global_store %acc, %global_c
}
```

---

## 14.5 UPMEM Codegen Artifacts

```text
generated/
  upmem/
    matmul_host.cpp
    matmul_device_kernel.c
    matmul_device_kernel.elf
    matmul_metadata.json
    pim_runtime_stub.cpp
```

## 14.6 SPLIT / LOCAL_LANES 실현 (inter-DPU vs intra-DPU)

공통 IR이 정한 두 병렬 축을 UPMEM이 실제 코드로 푸는 방식. **둘은 "값이 어디에 박히는가"가 다르다.**

### `NUM_UNITS`(=SPLIT) — host.cpp 소스에 박힘
`alloc_units{num_units=U}` → `UPMEMToHost`가 `U`를 **C 상수로** host.cpp에 박는다: `dpu_alloc(U)`, A는 `dpu_broadcast_to`(모든 유닛), B/C는 `DPU_FOREACH`로 per-DPU `prepare_xfer`+`push_xfer`(scatter/gather). host는 g++로 컴파일되는 일반 코드라 상수면 충분 → **소스 번들만으로 완결**.

### `LOCAL_LANES`(=NR_TASKLETS) — device 컴파일 플래그로 전달
DPU 런타임은 tasklet 개수를 **`-DNR_TASKLETS` 컴파일 플래그로만** 안다 (device.c에 `#define`을 박아도 런타임 startup이 못 봄). 따라서:
- `UPMEMToDevice`는 device.c를 **`NR_TASKLETS` 매크로로 generic하게** emit한다 (소스 텍스트는 lane 수와 무관):
  - `BARRIER_INIT(b, NR_TASKLETS)` + 초기화 시 `me()==0`이 `mem_reset()` 후 `barrier_wait`.
  - tasklet마다 자기 WRAM 캐시(`mem_alloc`, 8B 정렬).
  - **일 분배는 SPMD로 명시**: `for (i = me(); i < M; i += NR_TASKLETS)` — 각 tasklet이 출력 **행**을 round-robin으로 가져감(행 disjoint → 끝 barrier 불필요). 행 단위라 Triton BLOCK 2D 기하를 보존(추후 2D lane 분배·WRAM 재사용으로 확장).
- 실제 값은 launcher가 `metadata.local_lanes`를 읽어 `dpu-upmem-dpurte-clang -DNR_TASKLETS=L`로 컴파일한다. **IR 구조를 바꾸지 않는 device 빌드 파라미터라 IR/소스가 아닌 metadata 채널로 흐른다.**

> 이득: DPU의 fine-grained multithreading이 한 tasklet의 `mram_read` 지연을 다른 tasklet 연산으로 숨긴다 → lane 수가 늘수록 throughput 증가(보통 11+ tasklet 권장). 제약: per-tasklet WRAM × NR_TASKLETS ≤ 64KB (`NR_TASKLETS·(2K+N)·4B`). 초과 시 `BLOCK_K` chunk(O3.3) 필요.

---

# 15. AiMX Vendor Extension

## 15.1 AiMX Mapping

| PIM Common IR     | AiMX IR                        |
| ----------------- | ------------------------------ |
| `execution_unit`  | compute unit / PU              |
| `execution_group` | channel group / compute group  |
| `local_lane`      | internal execution lane        |
| `device_global`   | device memory                  |
| `device_local`    | GBuf / local buffer            |
| `device_scratch`  | scratch buffer                 |
| `device_register` | MAC register / output register |
| `pim.transfer`    | DMA descriptor                 |
| `pim.compute`     | compute descriptor             |
| `pim.placement`   | bank/channel mapping           |
| `pim.launch`      | command queue submit           |

---

## 15.2 AiMX Dialect Operations

```text
aimx.module
aimx.func
aimx.device
aimx.buffer
aimx.channel_group
aimx.compute_group
aimx.bank_map
aimx.layout_pack
aimx.dma_h2d
aimx.dma_d2h
aimx.compute_gemm
aimx.compute_gemv
aimx.compute_attention_qk
aimx.descriptor
aimx.descriptor_chain
aimx.submit
aimx.wait
```

---

## 15.3 AiMX Lowering Pipeline

```text
PIM Common IR
  ↓
legalize-pim-to-aimx
  ↓
AiMX IR
  ↓
aimx-memory-bank-mapping
aimx-execution-mapping
aimx-descriptor-selection
aimx-dma-scheduling
  ↓
Optimized AiMX IR
  ↓
aimx-command-stream-codegen
aimx-runtime-stub
  ↓
Descriptor Stream + Runtime Stub
```

---

## 15.4 AiMX IR Example

```mlir
aimx.func @matmul_command_stream(%A, %B, %C) {
  %dev = aimx.device {
    id = 0
  }

  %cg = aimx.channel_group {
    count = 16
  }

  %A_dev = aimx.buffer {
    space = "device",
    layout = "row_major"
  }

  %B_dev = aimx.buffer {
    space = "device",
    layout = "packed_by_bank"
  }

  %C_dev = aimx.buffer {
    space = "device",
    layout = "row_major"
  }

  %d0 = aimx.dma_h2d %A, %A_dev {
    async = true
  }

  %d1 = aimx.dma_h2d %B, %B_dev {
    layout_pack = "pack_by_output_tile",
    async = true
  }

  %d2 = aimx.compute_gemm %A_dev, %B_dev, %C_dev {
    M = 1024,
    N = 4096,
    K = 4096,
    block_m = 32,
    block_n = 64,
    block_k = 256,
    channels = 16,
    banks = 16,
    dtype = "bf16",
    accumulation = "fp32"
  }

  %d3 = aimx.dma_d2h %C_dev, %C {
    async = false
  }

  %chain = aimx.descriptor_chain %d0, %d1, %d2, %d3

  aimx.submit %chain {
    device_id = 0
  }

  aimx.wait %chain
}
```

---

## 15.5 AiMX Codegen Artifacts

```text
generated/
  aimx/
    matmul_runtime_stub.cpp
    matmul_descriptors.bin
    matmul_plan.json
    matmul_metadata.json
    layout_pack.cpp
```

---

# 16. Future Vendor Extension Model

새로운 PIM vendor가 추가될 경우 다음 구조를 따른다.

```text
PIM Common IR
    ↓
NewVendor IR
    ↓
NewVendor Optimization Passes
    ↓
NewVendor Codegen
    ↓
NewVendor Runtime Stub
```

새로운 vendor는 다음 파일 구조로 추가한다.

```text
include/triton/Dialect/PIMVendor/NewVendor/
  IR/
    NewVendorDialect.h
    NewVendorOps.h
    NewVendorTypes.h
    NewVendorAttrs.h
  Transforms/
    NewVendorPasses.h
  Runtime/
    NewVendorRuntime.h

lib/Dialect/PIMVendor/NewVendor/
  IR/
    NewVendorDialect.cpp
    NewVendorOps.cpp
    NewVendorTypes.cpp
    NewVendorAttrs.cpp
  Transforms/
    LegalizePIMToNewVendor.cpp
    NewVendorMemoryMapping.cpp
    NewVendorExecutionMapping.cpp
    NewVendorTransferLowering.cpp
    NewVendorCodegen.cpp
    NewVendorRuntimeStub.cpp
```

새 vendor가 반드시 구현해야 하는 pass:

```text
legalize-pim-to-newvendor
newvendor-memory-mapping
newvendor-execution-mapping
newvendor-transfer-lowering
newvendor-codegen
newvendor-runtime-stub
```

새 vendor가 반드시 제공해야 하는 정보:

```text
1. memory space mapping
2. execution unit mapping
3. local lane mapping
4. supported primitives
5. supported dtype
6. transfer mechanism
7. layout constraints
8. alignment constraints
9. launch mechanism
10. generated artifact format
```

---

# 17. Autotune Integration

## 17.0 동작 모델 (중요)

`@triton.autotune`은 별도의 PIM 전용 메커니즘이 아니다. Triton 표준 autotuner(`runtime/autotuner.py`)는 다음과 같이 동작한다.

```text
configs = [Config(BLOCK_M=32, ...), Config(BLOCK_M=64, ...), ...]
for config in configs:
    # config.kwargs(BLOCK_M 등)를 constexpr meta-param으로 주입하여
    # 각 config마다 독립적으로 컴파일 → 실행 → 시간 측정
    timing = bench(kernel.run(*args, **config.all_kwargs()))
best_config = argmin(timing)   # 가장 빠른 config 선택
# 이후 호출은 best_config로 한 번만 컴파일된 커널을 사용
```

즉 **하나의 IR에 모든 config를 metadata로 박는 것이 아니라, config마다 별도의 IR/artifact를 만들어 각각 실행·벤치마크한 뒤 가장 빠른 것을 고른다.** `BLOCK_M`/`BLOCK_N`/`BLOCK_K` 등이 `tl.constexpr`이므로, 값이 바뀌면 TTIR 단계부터 서로 다른 IR이 생성된다.

PIM에서 autotune이 적용되는 지점은 다음과 같다.

1. config 값이 `tl.constexpr`로 들어오는 경우(`BLOCK_M/N/K`): **TTIR → TTSharedIR → PIM Common IR** 각 단계에서 자연스럽게 서로 다른 IR로 specialize된다. `BuildPIMTilePlan`이 이 값을 읽어 `pim.tile_plan.block_*`에 박는다.
2. config 값이 커널 본문에서 참조되지 않는 PIM 전용 노브인 경우(`SPLIT_M/N/K`, `NUM_UNITS`, `LOCAL_LANES`): `tl.constexpr`이 아니라 compile option/metadata로 전달되며, `BuildPIMTilePlan`(SPLIT_*) / `BuildPIMExecutionPlan`(NUM_UNITS·LOCAL_LANES)이 option에서 직접 읽어 IR attribute로 확정한다.
3. 여러 config 중 best를 고르는 탐색 루프는 전적으로 Python autotuner가 담당하며, PIMBackend는 매 config를 평범한 단일 컴파일로 처리할 뿐이다.

> **별도 `pim-autotune-annotation` pass는 두지 않는다.** autotune 값을 IR에 반영하는 일은 이미 `BuildPIMTilePlan` / `BuildPIMExecutionPlan`이 수행하므로, 같은 option을 다시 읽어 attribute를 덮어쓰는 전용 pass는 불필요한 중복이다.

`@triton.autotune`이 존재하면 PIMBackend가 자동으로 이 경로를 타며, 별도 환경변수로 켜지 않는다.

```python
@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 32,
            "BLOCK_N": 64,
            "BLOCK_K": 256,
            "NUM_UNITS": 1856,
            "LOCAL_LANES": 16,
            "SPLIT_M": 1,
            "SPLIT_N": 64,
            "SPLIT_K": 1,
        }),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 256,
            "NUM_UNITS": 1856,
            "LOCAL_LANES": 16,
            "SPLIT_M": 1,
            "SPLIT_N": 32,
            "SPLIT_K": 1,
        }),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(...):
    ...
```

PIM autotune parameter:

```text
BLOCK_M
BLOCK_N
BLOCK_K
NUM_UNITS
LOCAL_LANES
SPLIT_M
SPLIT_N
SPLIT_K
LAYOUT_POLICY
PLACEMENT_POLICY
TRANSFER_POLICY
PIPELINE_POLICY
RESIDENCY_POLICY
REDUCTION_POLICY
```

`NUM_UNITS`와 `LOCAL_LANES`는 PIM Common IR의 중립 표현이다.

vendor mapping:

| PIM Autotune Param | UPMEM                 | AiMX                    |
| ------------------ | --------------------- | ----------------------- |
| `NUM_UNITS`        | number of DPUs        | number of compute units |
| `LOCAL_LANES`      | tasklets per DPU      | local lanes per unit    |
| `BLOCK_K`          | local reduction chunk | hardware K chunk        |
| `PLACEMENT_POLICY` | DPU/MRAM partition    | bank/channel mapping    |

---

# 18. Runtime API

## 18.1 PIM Runtime Object Model

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

---

## 18.2 Runtime C API

```c
typedef struct {
    const char* vendor;
    const char* arch;
    int device_id;
} PimDeviceConfig;

typedef struct {
    int B;
    int M;
    int N;
    int K;

    int BLOCK_M;
    int BLOCK_N;
    int BLOCK_K;

    int NUM_UNITS;
    int LOCAL_LANES;

    int SPLIT_M;
    int SPLIT_N;
    int SPLIT_K;

    const char* dtype;
    const char* layout_policy;
    const char* placement_policy;
    const char* transfer_policy;
    const char* pipeline_policy;
    const char* residency_policy;
    const char* reduction_policy;
} PimLaunchParams;

int pim_device_open(PimDeviceConfig* config, void** device_out);
int pim_device_query_capability(void* device, void* capability_out);
int pim_device_query_constraints(void* device, void* constraints_out);

int pim_executable_load(void* device, const char* artifact_path, void** executable_out);

int pim_memory_import(void* device, void* host_ptr, size_t size, void** memory_out);
int pim_memory_release(void* memory);

int pim_launch(void* device,
               void* executable,
               PimLaunchParams* params,
               void** buffers);

int pim_synchronize(void* device);
int pim_device_close(void* device);
```

---

# 19. Capability and Constraint Model

## 19.1 PIMCapability

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
```

---

## 19.2 PIMConstraints

```python
@dataclass
class PIMConstraints:
    block_m_alignment: int
    block_n_alignment: int
    block_k_alignment: int

    min_block_m: int
    max_block_m: int

    min_block_n: int
    max_block_n: int

    min_block_k: int
    max_block_k: int

    max_num_units: int
    max_local_lanes_per_unit: int

    transfer_alignment: int
    placement_granularity: int

    requires_m_multiple: int | None
    requires_n_multiple: int | None
    requires_k_multiple: int | None
```

---

# 20. Device Selection

PIMBackend는 환경변수로 enable하지 않는다.

사용자는 명시적으로 `device="pim"`을 사용한다.

```python
kernel[grid](
    ...,
    device="pim",
    vendor="upmem",
)
```

또는:

```python
kernel[grid](
    ...,
    device="pim",
    vendor="aimx",
)
```

Backend selection rule:

```text
if device == "pim":
    select PIMBackend

if vendor == "upmem":
    use UPMEMVendorExtension

if vendor == "aimx":
    use AiMXVendorExtension

else:
    raise UnsupportedPIMVendorError
```

---

# 21. Debug and Artifact Dump

PIM 전용 dump 코드를 두지 않고 **Triton 내장 dump**를 그대로 사용한다. `compile()`의 stage 루프가 등록된 모든 stage 결과를 자동으로 떨구므로, PIM이 추가하는 stage(`pim_common`, `vendor_ir`, `pimobj` 등)도 별도 코드 없이 같이 dump된다.

```bash
export TRITON_KERNEL_DUMP=1                 # knobs.compilation.dump_ir
export TRITON_DUMP_DIR=/tmp/triton_pim       # knobs.cache.dump_dir (기본: ~/.triton/dump)
python my_kernel.py
```

(환경변수 대신 Python에서 `knobs.compilation.dump_ir = True` / `knobs.cache.dump_dir = ...`로도 켤 수 있다.)

생성 파일: 커널 hash별 하위 폴더에 `{kernel}.{stage_ext}` 형식으로 stage마다 하나씩.

```text
$TRITON_DUMP_DIR/<kernel_hash>/
  matmul.ttir
  matmul.ttsharedir
  matmul.pim_common      # M3 이후 stage 추가 시 자동 포함
  matmul.vendor_ir       # M4 이후
  matmul.pimobj          # 최종 artifact
```

> `PIMOptions`에는 dump 관련 필드를 두지 않는다 (내장 knobs가 전역으로 제어).

---

# 22. End-to-End Example

## 22.1 Triton Source

```python
@triton.jit
def matmul_kernel(A, B, C,
                  M: tl.constexpr,
                  N: tl.constexpr,
                  K: tl.constexpr,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):

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

---

## 22.2 Run on AiMX

```python
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

Compilation flow:

```text
TTIR
  ↓
TTSharedIR
  ↓
PIM Common IR
  ↓
Optimized PIM Common IR
  ↓
AiMX IR
  ↓
AiMX Descriptor Stream
  ↓
AiMX Runtime Launch
```

---

## 22.3 Run on UPMEM

```python
matmul_kernel[grid](
    A, B, C,
    M, N, K,
    BLOCK_M=32,
    BLOCK_N=64,
    BLOCK_K=256,
    device="pim",
    vendor="upmem",
)
```

Compilation flow:

```text
TTIR
  ↓
TTSharedIR
  ↓
PIM Common IR
  ↓
Optimized PIM Common IR
  ↓
UPMEM IR
  ↓
UPMEM Host Code + DPU Kernel
  ↓
UPMEM Runtime Launch
```

---

# 23. Final Architecture Summary

```text
Triton PIMBackend
    |
    v
TTIR
    |
    v
TTSharedIR
    |
    v
PIM Common IR
    |
    +-- Common PIM Passes
    |     - candidate detection
    |     - tile metadata extraction
    |     - memory plan construction
    |     - tile plan construction
    |     - transfer plan construction
    |     - execution plan construction
    |     - placement plan construction
    |     - residency planning
    |     - chunk planning
    |     - pipeline overlap
    |
    v
Vendor Extension IR
    |
    +-- UPMEM IR
    |     - execution_unit → DPU
    |     - local_lane → tasklet
    |     - device_global → MRAM
    |     - device_local → WRAM
    |     - host/device split
    |     - DPU kernel codegen
    |
    +-- AiMX IR
          - execution_unit → compute unit
          - local_lane → internal lane
          - device_global → device memory
          - device_local → GBuf/local buffer
          - DMA descriptor generation
          - compute descriptor generation
          - command stream codegen
```

---

# 24. Final Definition

```text
PIMBackend is a Triton backend selected by device="pim". It lowers TTSharedIR into a vendor-neutral PIM Common IR, applies PIM-aware memory, transfer, tile, chunk, placement, residency, and execution mapping optimizations, and then dispatches the optimized PIM Common IR into a registered Vendor Extension such as UPMEM or AiMX. Each Vendor Extension lowers the PIM Common IR into its own vendor-specific IR and generates executable artifacts and runtime launch stubs for the target PIM device.
```

한국어 정의:

```text
PIMBackend는 device="pim"으로 선택되는 Triton backend이다. TTSharedIR을 입력으로 받아 vendor-neutral PIM Common IR을 생성하고, PIM에 필요한 memory plan, transfer plan, tile plan, chunk plan, data placement, residency, execution mapping 최적화를 수행한 뒤, 등록된 Vendor Extension으로 전달한다. UPMEM과 AiMX는 각각 PIM Common IR을 자신들의 vendor-specific IR로 lowering하고, 실제 장치에서 실행 가능한 artifact와 runtime launch stub을 생성한다.
```

---

# 25. CPU+PIM 동시 실행 (HybridDriver)

> **전제**: Phase 5 (UPMEM E2E) 완료 후 추가한다.

## 25.1 설계 원칙

TTSharedIR은 `program_id`(tile 좌표) 기반으로 tile을 표현하며, GEMM의 경우 타일 간 data dependency가 없다. 따라서 동일한 TTSharedIR에서 CPU binary와 PIM binary를 각각 컴파일한 뒤, 런타임에 tile grid를 분할하여 두 하드웨어에 동시에 실행할 수 있다.

이 구조는 Cinnamon의 공통 IR 계층 + NeuPIMs의 runtime scheduler dual-program 방식을 결합한 것이다.

## 25.2 컴파일 타임

```text
Triton Python DSL
    ↓
TTIR
    ↓
TTSharedIR  ←── heterogeneous dispatch 경계선
    ├── [CPUBackend]  → CPU binary  (grid 전체를 처리할 수 있는 코드)
    └── [PIMBackend]  → PIM binary  (grid 전체를 처리할 수 있는 코드)
```

두 backend는 동일한 TTSharedIR 텍스트를 입력으로 받아 독립적으로 lowering한다. TTSharedIR 생성(`_ttir_to_ttsharedir`)은 한 번만 수행하고 결과를 두 pipeline에 전달한다.

## 25.3 런타임

```text
HybridDriver.launch(grid=(M_tiles, N_tiles), ...)
    │
    ├── partition_policy(grid, cpu_cap, pim_cap)
    │     → cpu_range: pid_m ∈ [0, split_m)
    │     → pim_range: pid_m ∈ [split_m, M_tiles)
    │
    ├── Thread 1: cpu_launcher(cpu_range, ...)
    └── Thread 2: pim_launcher(pim_range, ...)
                  → DPU kernel에 tile_offset, tile_count 파라미터 전달
```

출력 buffer `C`는 pid_m 기준으로 disjoint하게 분할되므로 동기화 없이 안전하다.

## 25.4 PIM kernel 요건

HybridDriver를 지원하려면 DPU kernel이 `tile_offset`, `tile_count` 파라미터를 받아야 한다. `UPMEMCodegen` pass에서 이 파라미터를 생성한다.

## 25.5 Python API

```python
matmul_kernel[grid](
    A, B, C,
    M, N, K,
    BLOCK_M=32, BLOCK_N=64, BLOCK_K=256,
    device="hybrid",          # HybridDriver 선택
    pim_vendor="upmem",
    split_policy="half",      # "half" | "pim_ratio:0.75" | "auto"
)
```

`device="hybrid"`는 별도 `HybridBackend`로 구현하거나, 런타임 wrapper로 구현한다.

## 25.6 구현 파일

```text
triton-shared/
  hybrid/
    HybridDriver.py           # Python launcher: cpu + pim 동시 실행
    HybridPartitionPolicy.py  # tile grid 분할 정책
    HybridBackend.py          # Triton BaseBackend wrapper (optional)
```
