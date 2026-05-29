# Triton PIMBackend Final Architecture Design

## 1. Design Goal

Triton PIMBackend는 Triton 내부에서 PIM을 정식 backend로 인식하고, `device="pim"`을 통해 PIM device를 대상으로 kernel을 compile하고 실행할 수 있도록 하는 backend이다.

PIMBackend는 `triton-shared` repository 안에 `CPUBackend`와 동일한 방식으로 추가된다. 즉 Python측 backend(`triton-shared/backend/compiler.py`)는 `triton-shared-opt` 도구를 subprocess로 호출하는 thin dispatcher 역할만 수행하며, 모든 dialect/conversion/transform 코드는 `triton-shared/include/triton-shared/`, `triton-shared/lib/`에 C++로 구현된다.

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
from triton_shared.backend.compiler import PIMBackend

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

PIMBackend는 `triton-shared` repository 안에 추가된다. 따라서 dialect/conversion 코드의 root namespace는 `triton/Dialect/...`가 아니라 `triton-shared/Dialect/...`이며, Python backend 코드는 `triton-shared/backend/` 아래 단일 파일들로 구성한다 (CPUBackend와 동일한 구조).

```text
triton-shared/
  backend/
    compiler.py              # PIMBackend, PIMOptions 추가
    driver.py                # PIMDriver, PIMLauncher 추가
    name.conf                # backend 이름 등록
    include/
      ExecutionEngine/       # 기존 (CPU와 공유)

  include/
    triton-shared/
      Dialect/
        TPtr/                # 기존
        TritonStructured/    # 기존
        TritonTilingExt/     # 기존
        PIM/
          IR/
            PIMDialect.h
            PIMOps.h
            PIMTypes.h
            PIMAttrs.h
          Transforms/
            Passes.h
          Interfaces/
            PIMVendorExtension.h
            PIMTargetInterface.h
            PIMMemoryInterface.h
            PIMExecutionInterface.h

        PIMVendor/
          UPMEM/
            IR/
              UPMEMDialect.h
              UPMEMOps.h
              UPMEMTypes.h
              UPMEMAttrs.h
            Transforms/
              UPMEMPasses.h
            Runtime/
              UPMEMRuntime.h

          AiMX/
            IR/
              AiMXDialect.h
              AiMXOps.h
              AiMXTypes.h
              AiMXAttrs.h
            Transforms/
              AiMXPasses.h
            Runtime/
              AiMXRuntime.h

      Conversion/
        TTSharedToPIM/
          Passes.h
        PIMToUPMEM/
          Passes.h
        PIMToAiMX/
          Passes.h

  lib/
    Dialect/
      PIM/
        IR/
          PIMDialect.cpp
          PIMOps.cpp
          PIMTypes.cpp
          PIMAttrs.cpp

        Transforms/
          DetectPIMCandidate.cpp
          ExtractTileMetadata.cpp
          RecognizePIMPrimitive.cpp
          BuildPIMMemoryPlan.cpp
          BuildPIMTilePlan.cpp
          BuildPIMTransferPlan.cpp
          BuildPIMExecutionPlan.cpp
          BuildPIMPlacementPlan.cpp
          BuildPIMResidencyPlan.cpp
          PIMLayoutNormalize.cpp
          PIMDataPlacement.cpp
          PIMChunkPlanning.cpp
          PIMTransferHoisting.cpp
          PIMPersistentData.cpp
          PIMPartialResultPlanning.cpp
          PIMPipelineOverlap.cpp
          PIMAutotuneAnnotation.cpp

      PIMVendor/
        UPMEM/
          IR/
            UPMEMDialect.cpp
            UPMEMOps.cpp
            UPMEMTypes.cpp
            UPMEMAttrs.cpp
          Transforms/
            LegalizePIMToUPMEM.cpp
            UPMEMMemoryPlacement.cpp
            UPMEMExecutionMapping.cpp
            UPMEMTransferLowering.cpp
            UPMEMHostDeviceSplit.cpp
            UPMEMCodegen.cpp
            UPMEMRuntimeStub.cpp

        AiMX/
          IR/
            AiMXDialect.cpp
            AiMXOps.cpp
            AiMXTypes.cpp
            AiMXAttrs.cpp
          Transforms/
            LegalizePIMToAiMX.cpp
            AiMXMemoryBankMapping.cpp
            AiMXExecutionMapping.cpp
            AiMXDescriptorSelection.cpp
            AiMXDMAScheduling.cpp
            AiMXCommandStreamCodegen.cpp
            AiMXRuntimeStub.cpp

    Conversion/
      TTSharedToPIM/
        ...
      PIMToUPMEM/
        ...
      PIMToAiMX/
        ...

  tools/
    triton-shared-opt/
      RegisterTritonSharedDialects.h   # PIM dialect/pass 등록 추가

  runtime/
    pim/
      include/
        pim_runtime.h
        pim_device.h
        pim_launch.h
        pim_executable.h
        pim_memory.h
      lib/
        PIMRuntime.cpp
        PIMDevice.cpp
        PIMExecutable.cpp
        PIMLaunch.cpp
        PIMMemory.cpp
      vendors/
        upmem/
          UPMEMRuntime.cpp
          UPMEMLauncher.cpp
        aimx/
          AiMXRuntime.cpp
          AiMXLauncher.cpp
```

CPUBackend와 동일하게 IR 변환은 Python이 아니라 `triton-shared-opt` 실행 파일을 subprocess로 호출하여 수행한다. 따라서 새 PIM dialect/pass는 모두 `tools/triton-shared-opt/RegisterTritonSharedDialects.h`에 등록되어야 한다.

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

각 `_*` helper는 CPUBackend의 `_ttir_to_ttsharedir` / `_ttsharedir_to_llir`과 동일하게 `triton-shared-opt` 또는 `mlir-opt`를 subprocess로 호출하는 thin wrapper이다. Pass 목록은 다음과 같다.

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
pim-autotune-annotation
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

의미:

```text
BLOCK_M → output tile row size
BLOCK_N → output tile column size
BLOCK_K → reduction chunk size
```

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
pim-autotune-annotation
canonicalize
cse
```

## 13.3 Pass Responsibilities

| Pass                          | 역할                                                       |
| ----------------------------- | -------------------------------------------------------- |
| `detect-pim-candidate`        | TTSharedIR에서 PIM으로 lowering 가능한 region 탐지                |
| `extract-tile-metadata`       | `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, grid, shape, dtype 추출   |
| `recognize-pim-primitive`     | GEMM, GEMV, attention-like pattern, reduction pattern 인식 |
| `build-pim-memory-plan`       | host/device memory region 생성                             |
| `build-pim-tile-plan`         | Triton block 정보를 PIM tile plan으로 변환                      |
| `build-pim-transfer-plan`     | explicit transfer 계획 생성                                  |
| `build-pim-execution-plan`    | execution unit/group mapping 계획 생성                       |
| `build-pim-placement-plan`    | data placement, interleaving, replication 계획 생성          |
| `build-pim-residency-plan`    | persistent/ephemeral/streaming data residency 결정         |
| `pim-layout-normalize`        | layout 표현을 vendor-neutral layout으로 정규화                   |
| `pim-data-placement`          | data locality와 reuse를 고려한 placement 최적화                  |
| `pim-chunk-planning`          | reduction axis chunking 계획 생성                            |
| `pim-transfer-hoisting`       | 반복 transfer를 상위 scope로 이동                                |
| `pim-persistent-data`         | weight/KV cache 등 persistent data 지정                     |
| `pim-partial-result-planning` | partial result 저장 및 reduce 계획                            |
| `pim-pipeline-overlap`        | transfer/compute/store overlap 구성                        |
| `pim-autotune-annotation`     | `@triton.autotune` config를 PIM IR metadata로 연결           |

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

PIMBackend는 별도 환경변수로 autotune을 켜지 않는다.

`@triton.autotune`이 존재하면 PIMBackend는 자동으로 PIM autotune path를 사용한다.

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

환경변수 대신 Python compile option으로 제어한다.

```python
matmul_kernel[grid](
    A, B, C,
    M, N, K,
    BLOCK_M=32,
    BLOCK_N=64,
    BLOCK_K=256,
    device="pim",
    vendor="aimx",
    debug_options={
        "dump_ir": True,
        "dump_artifacts": True,
        "dump_dir": "/tmp/triton_pim/matmul",
    },
)
```

생성 파일:

```text
/tmp/triton_pim/matmul/
  00_ttir.mlir
  01_ttshared.mlir
  02_pim_common.mlir
  03_pim_common_opt.mlir
  04_vendor.mlir
  05_vendor_opt.mlir
  06_codegen/
    artifact.bin
    runtime_stub.cpp
    metadata.json
```

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
    |     - autotune annotation
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
