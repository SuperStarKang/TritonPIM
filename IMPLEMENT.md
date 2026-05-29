# Triton PIMBackend Implementation Plan

이 문서는 [DESIGN.md](DESIGN.md)를 실제 코드로 구현하기 위한 단계별 계획이다.
각 phase는 독립적으로 빌드/테스트 가능한 단위이며, 체크박스로 진행 상황을 추적한다.

## 진행 상태 범례

- `[ ]` 미착수
- `[~]` 진행 중
- `[x]` 완료
- `[!]` 차단 (blocker 있음 — 코멘트로 사유 기재)
- `[-]` 보류 (skip / 후순위로 이동)

## 전체 진행 요약

| Phase | 제목 | 상태 | 진행률 |
|------:|------|:----:|:------:|
| 0 | Skeleton: PIMBackend / PIMDriver 골격 | `[ ]` | 0 / 7 |
| 1 | PIM Common Dialect 최소 셋 (6 ops) | `[ ]` | 0 / 8 |
| 2 | TTShared → PIM Conversion (GEMM 패턴 하나) | `[ ]` | 0 / 5 |
| 3 | UPMEM Vendor Extension (단일 vendor 우선) | `[ ]` | 0 / 7 |
| 4 | Runtime & Launcher | `[ ]` | 0 / 5 |
| 5 | End-to-end: MatMul kernel 실행 | `[ ]` | 0 / 4 |
| 6 | AiMX Vendor Extension 추가 | `[ ]` | 0 / 6 |
| 7 | Vendor Extension Registry 리팩토링 | `[ ]` | 0 / 3 |
| 8 | Common Optimization Passes | `[ ]` | 0 / 9 |
| 9 | Capability / Constraints / Autotune | `[ ]` | 0 / 5 |
| 10 | Debug / Artifact Dump | `[ ]` | 0 / 3 |

총 진행률: **0 / 62**

---

## Phase 0 — Skeleton: PIMBackend / PIMDriver 골격

**목표**: `device="pim"`으로 호출 시 Triton runtime이 PIMBackend를 dispatch하고, CPU와 동일한 lowering으로 일단 실행되게 한다. IR/dialect/codegen은 아직 PIM 고유 동작 없음. 빌드/runtime 연결 검증만.

**산출물**:
- `triton-shared/backend/compiler.py` 에 `PIMOptions`, `PIMBackend` 추가
- `triton-shared/backend/driver.py` 에 `PIMDriver`, `PIMLauncher` 추가
- backend 등록 (`name.conf` 또는 setup 변경)

**완료 조건**: `pytest`로 다음이 통과
```python
matmul_kernel[grid](A, B, C, M, N, K, ..., device="pim", vendor="upmem")
```
(이 시점 실제 실행은 CPU lowering 결과로 수행되며, vendor="upmem"은 단순 attribute로만 보존된다.)

**작업 단위**:

- [ ] **0.1** `PIMOptions` dataclass 추가 (CPUOptions 기반, `vendor`/`device_id`/`dump_*` 필드 추가, `__post_init__`에서 `vendor` 누락 시 ValueError)
- [ ] **0.2** `PIMBackend(BaseBackend)` 클래스 추가
  - `binary_ext = 'pimobj'`
  - `supports_target`: `target.backend == 'pim'`
  - `parse_options` → `PIMOptions`
  - `load_dialects(ctx)` 빈 함수
  - `add_stages(stages, options, language)` — Phase 0에서는 CPU와 동일한 4-stage 그대로 차용
  - `pack_metadata`, `get_codegen_implementation`, `hash`, `get_module_map` 구현
- [ ] **0.3** `PIMDriver(DriverBase)` 추가 (CPUDriver 복제, `get_current_target`에서 `GPUTarget("pim", 0, 0)` 반환, `is_active()` False 유지)
- [ ] **0.4** `PIMLauncher` / `PIMUtils` 추가 (Phase 0에서는 CPULauncher의 launcher 코드 그대로 사용 가능)
- [ ] **0.5** backend 등록 경로 확인: `triton-shared/backend/name.conf`가 단일 backend만 받는지 검사 → 필요시 `triton-pim/backend/`로 별도 디렉토리 분리하거나, `name.conf`에서 두 backend 등록하는 방법 결정
- [ ] **0.6** 빌드 확인: `pip install -e python/` (또는 기존 빌드 절차) 후 `python -c "import triton; triton.runtime.driver.set_active(...)"` 로 import 가능 검증
- [ ] **0.7** smoke test: `python/examples/`에 `test_pim_dispatch.py` 추가 — `device="pim"` 호출이 PIMBackend로 라우팅되는지만 확인 (실행 결과는 CPU와 동일)

---

## Phase 1 — PIM Common Dialect 최소 셋 (6 ops)

**목표**: vendor-neutral PIM dialect를 정의한다. DESIGN.md 11장의 15+ ops를 전부가 아니라 GEMM 1개를 lowering하는 데 필요한 최소 6 ops만.

**산출물**:
- `triton-shared/include/triton-shared/Dialect/PIM/IR/PIMDialect.h`, `PIMOps.h`, `PIMTypes.h`, `PIMAttrs.h`
- `triton-shared/lib/Dialect/PIM/IR/*.cpp`
- TableGen 정의: `PIMOps.td`, `PIMTypes.td`, `PIMAttrs.td`
- `triton-shared-opt`에 PIM dialect 등록

**완료 조건**: `triton-shared-opt --verify-roundtrip` 으로 PIM Common IR MLIR 파일을 parse/print 가능.

**대상 6 ops** (나머지는 attribute로 표현):

```
pim.func              (or func.func + attrs)
pim.memory_region
pim.tile_plan
pim.transfer
pim.compute           (op="gemm"부터)
pim.launch
```

**작업 단위**:

- [ ] **1.1** `PIMDialect.td` 작성 (dialect name "pim", namespace `mlir::triton::pim`)
- [ ] **1.2** `PIMTypes.td` — `MemorySpaceType`, `MemoryRegionType` 정의
- [ ] **1.3** `PIMAttrs.td` — `LayoutAttr`, `ResidencyAttr`, `RoleAttr`, `MappingPolicyAttr`, `CapabilityRefAttr` 등 enum/string attr 정의
- [ ] **1.4** `PIMOps.td` — 위 6 ops 정의 (operand/result/attr/verifier)
- [ ] **1.5** `lib/Dialect/PIM/IR/PIMDialect.cpp` 및 `PIMOps.cpp` 구현 (verifier, custom assembly format)
- [ ] **1.6** `tools/triton-shared-opt/RegisterTritonSharedDialects.h`에 `mlir::triton::pim::PIMDialect` 등록 추가
- [ ] **1.7** `CMakeLists.txt` 업데이트 (`include/`, `lib/Dialect/PIM/CMakeLists.txt` 추가)
- [ ] **1.8** roundtrip test: `test/Dialect/PIM/ops.mlir` 추가, `triton-shared-opt -verify-diagnostics` 통과

---

## Phase 2 — TTShared → PIM Conversion (GEMM 패턴 하나)

**목표**: TTSharedIR의 `linalg.matmul` (또는 `tt.dot`이 lowering된 형태)을 `pim.compute @gemm`으로 변환하는 conversion pass 작성.

**산출물**:
- `include/triton-shared/Conversion/TTSharedToPIM/Passes.h`, `Passes.td`
- `lib/Conversion/TTSharedToPIM/*.cpp`

**완료 조건**: 간단한 matmul TTSharedIR를 입력으로 `triton-shared-opt --convert-ttshared-to-pim` 실행 시 `pim.compute`가 포함된 IR 생성.

**작업 단위**:

- [ ] **2.1** `RecognizeGemmPattern` pass — `linalg.matmul` → `pim.compute @gemm` 변환 패턴 작성
- [ ] **2.2** Tile plan 추출: matmul operand shape에서 BLOCK_M/N/K를 읽어 `pim.tile_plan` 생성
- [ ] **2.3** Memory region 생성: input/output buffer를 `pim.memory_region`으로 감싸기
- [ ] **2.4** Transfer 삽입: host→device, device→host transfer를 `pim.transfer`로 명시
- [ ] **2.5** `convert-ttshared-to-pim` pass를 `triton-shared-opt`에 등록 + lit test

---

## Phase 3 — UPMEM Vendor Extension (단일 vendor 우선)

**목표**: PIM Common IR을 UPMEM IR로 lowering하고, host C++ + DPU kernel C 코드를 생성한다. **AiMX는 Phase 6에서 추가.**

**산출물**:
- `include/triton-shared/Dialect/PIMVendor/UPMEM/IR/*.h`
- `lib/Dialect/PIMVendor/UPMEM/{IR,Transforms}/*.cpp`
- UPMEM SDK 연동 codegen 스크립트

**완료 조건**: `triton-shared-opt --legalize-pim-to-upmem --upmem-codegen` 으로 host/device 소스코드 파일 생성.

**작업 단위**:

- [ ] **3.1** UPMEM dialect 정의 (Phase 1과 동일한 패턴: `.td` + `.cpp`)
  - 최소 ops: `upmem.host_func`, `upmem.device_func`, `upmem.alloc_units`, `upmem.copy_to_global`, `upmem.copy_from_global`, `upmem.launch`, `upmem.local_gemm`
- [ ] **3.2** `LegalizePIMToUPMEM` conversion pass
- [ ] **3.3** `UPMEMHostDeviceSplit` pass (host 코드와 device kernel 분리)
- [ ] **3.4** `UPMEMCodegen` pass — IR → C/C++ 텍스트 생성 (단순 string emission으로 시작)
- [ ] **3.5** `UPMEMRuntimeStub` pass — host wrapper 생성
- [ ] **3.6** `triton-shared-opt`에 UPMEM dialect/passes 등록
- [ ] **3.7** test: 작은 matmul로 host.cpp + device.c 생성 확인 (실제 컴파일은 Phase 4)

---

## Phase 4 — Runtime & Launcher

**목표**: 생성된 UPMEM artifact를 실제 PIM device(또는 UPMEM functional simulator)에서 실행한다.

**산출물**:
- `triton-shared/runtime/pim/` 디렉토리 신설
- `PIMRuntime`, `PIMDevice`, `PIMExecutable`, `PIMLaunch` C++ 구현
- `UPMEMLauncher` (UPMEM SDK 호출)
- `backend/driver.py`의 `PIMLauncher`가 위 runtime을 호출하도록 연결

**완료 조건**: `pim_executable_load` → `pim_launch` → `pim_synchronize` 시퀀스로 host에서 device 실행 호출 성공.

**작업 단위**:

- [ ] **4.1** `runtime/pim/include/pim_runtime.h` C API 정의 (DESIGN.md 18.2)
- [ ] **4.2** `runtime/pim/lib/PIMRuntime.cpp` 등 기본 구현 (UPMEM SDK 의존)
- [ ] **4.3** `runtime/pim/vendors/upmem/UPMEMLauncher.cpp` — `dpu_alloc`, `dpu_load`, `dpu_launch` 등 wrapping
- [ ] **4.4** Phase 0의 `PIMLauncher`를 갱신: CPU launcher 대신 PIM runtime을 dlopen해서 호출
- [ ] **4.5** `CMakeLists.txt` 업데이트 (UPMEM SDK include/link)

---

## Phase 5 — End-to-end: MatMul kernel 실행

**목표**: Triton Python source → ... → UPMEM device 실행까지 한 번에 통과시키고 정답 검증.

**완료 조건**: `numpy.allclose(C_pim, A @ B)` 통과.

**작업 단위**:

- [ ] **5.1** `python/examples/test_pim_matmul.py` 작성 (DESIGN.md 22장 예시 기준)
- [ ] **5.2** Phase 0의 `add_stages`를 실제 PIM pipeline으로 교체 (CPU lowering 제거, Phase 2~3 pass 사용)
- [ ] **5.3** small shape (M=32, N=64, K=256) 실행 + 검증
- [ ] **5.4** medium shape (M=256, N=512, K=512) 실행 + 검증

---

## Phase 6 — AiMX Vendor Extension 추가

**목표**: 두 번째 vendor를 추가하여 vendor-neutral 추상화가 실제로 작동함을 검증한다.

**작업 단위**:

- [ ] **6.1** AiMX dialect 정의 (`aimx.dma_h2d`, `aimx.dma_d2h`, `aimx.compute_gemm`, `aimx.descriptor_chain`, `aimx.submit`, `aimx.wait` 최소셋)
- [ ] **6.2** `LegalizePIMToAiMX` conversion
- [ ] **6.3** `AiMXMemoryBankMapping` + `AiMXExecutionMapping`
- [ ] **6.4** `AiMXCommandStreamCodegen` (descriptor binary emission)
- [ ] **6.5** `runtime/pim/vendors/aimx/AiMXLauncher.cpp`
- [ ] **6.6** `python/examples/test_pim_matmul_aimx.py` 실행 + 검증

---

## Phase 7 — Vendor Extension Registry 리팩토링

**목표**: 지금까지 if-else로 분기되던 vendor dispatch를 DESIGN.md 7장의 `VendorExtensionRegistry`로 통합한다. 두 vendor가 모두 동작한 뒤 진행 (premature abstraction 회피).

**작업 단위**:

- [ ] **7.1** Python측 `VendorExtensionRegistry` 구현 (DESIGN.md 7장)
- [ ] **7.2** C++측 `PIMVendorExtension` interface (DESIGN.md 8장) + UPMEM/AiMX 등록
- [ ] **7.3** `PIMBackend.add_stages`를 `self.vendor_extension.build_*_pipeline(options)` 형태로 리팩토링

---

## Phase 8 — Common Optimization Passes

**목표**: DESIGN.md 13.2의 common optimization pass들을 단계별로 추가. 각 pass는 기능 ON/OFF가 가능해야 함.

**작업 단위** (각 pass별로 lit test 포함):

- [ ] **8.1** `pim-layout-normalize`
- [ ] **8.2** `pim-data-placement`
- [ ] **8.3** `pim-chunk-planning`
- [ ] **8.4** `pim-transfer-hoisting`
- [ ] **8.5** `pim-persistent-data`
- [ ] **8.6** `pim-partial-result-planning`
- [ ] **8.7** `pim-pipeline-overlap`
- [ ] **8.8** `pim-autotune-annotation`
- [ ] **8.9** end-to-end matmul로 baseline 대비 성능 개선 측정

---

## Phase 9 — Capability / Constraints / Autotune

**목표**: vendor별 capability/constraint를 IR과 autotune에 연결.

**작업 단위**:

- [ ] **9.1** `PIMCapability`, `PIMConstraints` dataclass 구현 (DESIGN.md 19장)
- [ ] **9.2** UPMEM/AiMX vendor에서 `queryCapability`, `queryConstraints` C++ 구현
- [ ] **9.3** `#pim.capability_ref<...>` attribute가 lowering 시 capability 값으로 구체화되도록 처리
- [ ] **9.4** `@triton.autotune` config을 PIM Common IR metadata로 연결 (`pim-autotune-annotation`과 통합)
- [ ] **9.5** autotune key (BLOCK_M/N/K, NUM_UNITS, LOCAL_LANES 등) 가 실제 IR에 반영되는지 검증

---

## Phase 10 — Debug / Artifact Dump

**목표**: DESIGN.md 21장의 debug option을 지원.

**작업 단위**:

- [ ] **10.1** `PIMOptions.dump_ir`, `dump_artifacts`, `dump_dir` 지원
- [ ] **10.2** 각 stage의 IR을 `00_ttir.mlir`, `01_ttshared.mlir`, ... 형식으로 저장
- [ ] **10.3** codegen artifact (host.cpp, device.c, descriptors.bin, metadata.json) 저장

---

## 의존성 그래프

```text
Phase 0 (skeleton)
  ↓
Phase 1 (PIM dialect)
  ↓
Phase 2 (TTShared → PIM)
  ↓
Phase 3 (UPMEM lowering) ──┐
  ↓                        │
Phase 4 (runtime)          │
  ↓                        │
Phase 5 (E2E UPMEM) ───────┘
  ↓
Phase 6 (AiMX) ──→ Phase 7 (registry refactor)
  ↓
Phase 8 (optimizations)
  ↓
Phase 9 (capability/autotune)
  ↓
Phase 10 (debug/dump)
```

Phase 8/9/10은 Phase 5 완료 후 순서를 바꿔 진행해도 무방.

---

## 다음 액션

현재 다음 작업을 시작하면 됩니다:

- [ ] **Phase 0.1** — `PIMOptions` dataclass를 [triton-shared/backend/compiler.py](triton-shared/backend/compiler.py)에 추가
