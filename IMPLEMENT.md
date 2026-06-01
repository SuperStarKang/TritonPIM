# Triton PIMBackend Implementation Plan

이 문서는 [DESIGN.md](DESIGN.md)를 실제 코드로 구현하기 위한 단계별 계획이다.
각 단계는 독립적으로 빌드/테스트 가능한 단위이며, 체크박스로 진행 상황을 추적한다.

## 구현 철학 (중요)

**먼저 끝까지 돌리고(MVP), 그다음 하나씩 최적화한다.**

- **Part I (MVP)**: `device="pim", vendor="upmem"`로 **no-split GEMM** 하나를 실제 DPU에서 정답까지 통과시킨다. 이 구간에서는 granular plan pass·placement·execution mapping·residency·chunk·reduce·strategy를 만들지 않는다. 최단 경로로 "돈다"에 도달한다. 단, **Debug/IR Dump는 MVP 초반에 먼저 넣어** 각 단계 변환이 제대로 되는지 눈으로 확인하며 개발한다.
- **Part II (최적화)**: MVP가 도는 상태를 baseline으로, 최적화를 **additive하게 하나씩** 얹는다. 각 pass는 **독립 toggle 가능**해야 하고, **OFF 시 MVP(M6)가 그대로 통과**해야 한다. (DESIGN.md 13.0 경계 원칙 준수: 공통 pass는 구조/intent/gate만, 값 최적화는 autotune, 구체 실현은 vendor.)
- **Part III (확장)**: 두 번째 vendor(AiMX), registry 리팩터, CPU+PIM Hybrid.

## 진행 상태 범례

- `[ ]` 미착수
- `[~]` 진행 중
- `[x]` 완료
- `[!]` 차단 (blocker 있음 — 코멘트로 사유 기재)
- `[-]` 보류 (skip / 후순위로 이동)

## 전체 진행 요약

| 단계 | 제목 | 상태 | 진행률 |
|------|------|:----:|:------:|
| **Part I — Minimal Executable (MVP)** | | | |
| M0 | Skeleton: pim_backend 디렉토리 + PIMBackend / PIMDriver | `[x]` | 8 / 8 |
| M1 | 최소 PIM Common Dialect (6 ops) | `[x]` | 8 / 8 |
| M2 | Debug / IR Dump (변환 확인용, 먼저 구축) | `[~]` | 2 / 3 |
| M3 | TTShared→PIM 단일 변환 (no-split GEMM) | `[x]` | 4 / 4 |
| M4 | 최소 UPMEM lowering + codegen (no-split) | `[ ]` | 0 / 7 |
| M5 | Runtime & Launcher | `[ ]` | 0 / 5 |
| M6 | End-to-end: MatMul 실행 + 검증 | `[ ]` | 0 / 4 |
| **Part II — Incremental Optimization** | | | |
| O1 | 단일 변환을 granular plan pass로 분해 | `[ ]` | 0 / 7 |
| O2 | Capability / Constraints | `[ ]` | 0 / 3 |
| O3 | Common Optimization Passes (하나씩) | `[ ]` | 0 / 8 |
| O4 | Execution Strategy Selection | `[ ]` | 0 / 5 |
| O5 | Autotune Integration | `[ ]` | 0 / 3 |
| **Part III — Breadth** | | | |
| B1 | AiMX Vendor Extension | `[ ]` | 0 / 6 |
| B2 | Vendor Extension Registry 리팩토링 | `[ ]` | 0 / 3 |
| B3 | CPU+PIM HybridDriver | `[ ]` | 0 / 5 |

총 진행률: **0 / 79**

---

# Part I — Minimal Executable (MVP)

> 목표: `device="pim", vendor="upmem"`, **no-split GEMM**, 실제 DPU에서 `numpy.allclose` 통과. 최적화·다중 전략·placement 정교화는 일절 하지 않는다.

## M0 — Skeleton: pim_backend 디렉토리 + PIMBackend / PIMDriver

**목표**: `device="pim"` 호출 시 Triton runtime이 PIMBackend를 dispatch. IR/codegen은 아직 PIM 고유 동작 없이 CPU lowering으로 일단 실행. 빌드/runtime 연결만 검증.

**배경**: Triton의 `_find_concrete_subclasses`는 하나의 `compiler.py`에 `BaseBackend` 서브클래스가 정확히 하나만 있어야 한다. nvidia/amd/cpu처럼 PIMBackend도 자신만의 `triton-shared/pim_backend/backend/` 디렉토리를 갖는다.

**산출물**:
- `triton-shared/pim_backend/backend/compiler.py` — `PIMOptions`, `PIMBackend`
- `triton-shared/pim_backend/backend/driver.py` — `PIMDriver`, `PIMLauncher`
- `triton-shared/pim_backend/backend/name.conf` — `"pim"`
- `triton-shared/pim_backend/backend/include/ExecutionEngine/` — CPU와 공유 (symlink)

**완료 조건**: `matmul_kernel[grid](..., device="pim", vendor="upmem")`가 PIMBackend로 라우팅(실행 결과는 CPU lowering과 동일).

**작업 단위**:

- [x] **M0.1** `pim_backend/backend/` 디렉토리 생성, `name.conf`에 `pim` 기재
- [x] **M0.2** `PIMOptions` dataclass (CPUOptions 기반, `vendor`/`device_id`/`dump_*` 추가, `__post_init__`에서 `vendor` 누락 시 `ValueError`)
- [x] **M0.3** `PIMBackend(BaseBackend)`: `binary_ext='pimobj'`, `supports_target`=`backend=='pim'`, `parse_options`, 빈 `load_dialects`, `add_stages`(M0에서는 `triton-shared/backend/compiler.py`의 CPU 4-stage import해 차용), `pack_metadata`/`get_codegen_implementation`/`hash`/`get_module_map`
- [x] **M0.4** `PIMDriver(DriverBase)` (CPUDriver 복제, `get_current_target`→`GPUTarget("pim",0,0)`, `is_active()` False)
- [x] **M0.5** `PIMLauncher`/`PIMUtils` (M0에서는 CPULauncher 코드 복사/import)
- [x] **M0.6** 빌드 등록: `TRITON_PLUGIN_DIRS`에 `pim_backend` 경로 추가 후 `pip install -e .`, entry_points에 `pim` 확인
- [x] **M0.7** `include/ExecutionEngine/` symlink (→ `backend/include/ExecutionEngine`)
- [x] **M0.8** smoke test: `pim_backend/examples/test_pim_dispatch.py` — `device="pim"` 라우팅 확인

---

## M1 — 최소 PIM Common Dialect (6 ops)

**목표**: vendor-neutral PIM dialect 최소셋 정의. DESIGN.md 11장의 전체가 아니라 no-split GEMM 1개에 필요한 6 ops만.

**대상 6 ops** (나머지는 attribute로):

```
pim.func              (or func.func + attrs)
pim.memory_region
pim.tile_plan
pim.transfer
pim.compute           (op="gemm")
pim.launch
```

**완료 조건**: `triton-shared-opt --verify-roundtrip`로 PIM Common IR parse/print 가능.

**작업 단위**:

- [x] **M1.1** `PIMDialect.td` (name "pim", namespace `mlir::triton::pim`)
- [x] **M1.2** `PIMTypes.td` — `MemorySpaceType`, `MemoryRegionType`
- [x] **M1.3** `PIMAttrs.td` — `LayoutAttr`, `ResidencyAttr`, `RoleAttr`, `MappingPolicyAttr`, `CapabilityRefAttr` 등
- [x] **M1.4** `PIMOps.td` — 위 6 ops (operand/result/attr/verifier)
- [x] **M1.5** `lib/Dialect/PIM/IR/PIMDialect.cpp`, `PIMOps.cpp` (verifier, custom assembly)
- [x] **M1.6** `RegisterTritonSharedDialects.h`에 `mlir::triton::pim::PIMDialect` 등록
- [x] **M1.7** `CMakeLists.txt` 업데이트 (`include/`, `lib/Dialect/PIM/`)
- [x] **M1.8** roundtrip test: `test/Dialect/PIM/ops.mlir`, `-verify-diagnostics` 통과

---

## M2 — Debug / IR Dump (변환 확인용, 먼저 확인)

**목표**: 변환 파이프라인을 만들기 **전에** 각 단계 IR을 파일로 볼 수 있는 수단을 확보한다. 이후 M3(변환)·M4(codegen) 개발 시 **실제로 잘 변환되는지 눈으로 확인**하며 진행한다.

**핵심**: Triton에 **내장 dump가 이미 있다.** `compile()`의 stage 루프(`compiler/compiler.py`)가 등록된 모든 stage 결과를 `fn_dump_manager`로 자동 저장한다. 따라서 backend에 dump 코드를 새로 짜지 않고 내장 메커니즘을 그대로 쓴다. M3/M4에서 stage(`pim_common`, `vendor_ir`)가 늘면 자동으로 같이 dump된다. (DESIGN.md 21장)

**완료 조건**: `TRITON_KERNEL_DUMP=1`로 실행 시 현재 stage들(`ttir`/`ttsharedir`/`llir`/`pimobj`)이 `TRITON_DUMP_DIR`에 저장됨을 확인.

**작업 단위**:

- [x] **M2.1** 내장 dump 동작 확인: `TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton_pim`로 smoke test 실행 → 커널 hash 폴더에 `*.ttir`, `*.ttsharedir`, `*.llir`, `*.pimobj` 생성 확인
- [x] **M2.2** `PIMOptions`에서 미사용 dump 필드 제거(내장 knobs로 대체), DESIGN 21장을 내장 dump 기준으로 정정
- [ ] **M2.3** (M3/M4 이후) `pim_common`/`vendor_ir`/codegen artifact가 같은 메커니즘으로 자동 dump되는지 확인

---

## M3 — TTShared→PIM 단일 변환 (no-split GEMM)

**목표**: `linalg.matmul`을 `pim.compute @gemm`으로 바꾸는 **단일 변환 pass**. no-split 고정 — `split=1` 하드코딩, placement/execution_group/residency/chunk/reduce **만들지 않음**. (granular plan pass 분해는 O1에서.)

**완료 조건**: matmul TTSharedIR → `triton-shared-opt --convert-ttshared-to-pim` → `pim.compute @gemm` 포함 IR 생성. (M2 dump로 `02_pim_common.mlir` 눈으로 확인.)

**작업 단위**:

- [x] **M3.1** `convert-ttshared-to-pim` 단일 pass: `linalg.matmul` 인식 → `pim.compute @gemm`
- [x] **M3.2** BLOCK_M/N/K를 읽어 `pim.tile_plan {block_*, split_m=1, split_n=1, split_k=1}` 하드코딩 생성
- [x] **M3.3** A/B/C를 `pim.memory_region`(host/device_global)으로 감싸고 `pim.transfer`(host↔device) 삽입 — placement/residency/chunk/reduce 없음
- [x] **M3.4** pass를 `triton-shared-opt`에 등록 + lit test (`pim.compute` 존재 확인)

---

## M4 — 최소 UPMEM lowering + codegen (no-split)

**목표**: PIM Common IR → UPMEM IR → host.cpp + device.c 생성. **no-split만** (모든 DPU에 동일 매핑, K-split/N-split 없음).

**완료 조건**: `triton-shared-opt --legalize-pim-to-upmem --upmem-codegen`으로 host/device 소스 생성. (M2 dump로 `03_vendor.mlir`·`host.cpp`·`device.c` 확인.)

**작업 단위**:

- [ ] **M4.1** UPMEM dialect 최소 ops (`upmem.host_func`, `device_func`, `alloc_units`, `copy_to_global`, `copy_from_global`, `launch`, `local_gemm`)
- [ ] **M4.2** `LegalizePIMToUPMEM` (no-split: execution_unit→DPU 1:1, device_global→MRAM, device_local→WRAM)
- [ ] **M4.3** `UPMEMHostDeviceSplit` (host_func / device_func 분리)
- [ ] **M4.4** `UPMEMCodegen` — IR → host.cpp + device.c 텍스트 (string emission)
- [ ] **M4.5** `UPMEMRuntimeStub` — host wrapper
- [ ] **M4.6** `triton-shared-opt`에 UPMEM dialect/passes 등록
- [ ] **M4.7** lit test: 작은 matmul로 host.cpp + device.c 생성 확인 (실제 컴파일은 M5)

---

## M5 — Runtime & Launcher

**목표**: 생성된 UPMEM artifact를 실제 PIM device(또는 functional simulator)에서 실행.

**완료 조건**: `pim_executable_load` → `pim_launch` → `pim_synchronize` 시퀀스로 device 실행 호출 성공.

**작업 단위**:

- [ ] **M5.1** `runtime/pim/include/pim_runtime.h` C API (DESIGN.md 18.2)
- [ ] **M5.2** `runtime/pim/lib/PIMRuntime.cpp` 등 기본 구현 (UPMEM SDK)
- [ ] **M5.3** `runtime/pim/vendors/upmem/UPMEMLauncher.cpp` — `dpu_alloc`/`dpu_load`/`dpu_copy_to`/`dpu_launch`/`dpu_copy_from` wrapping
- [ ] **M5.4** M0의 `PIMLauncher`를 갱신: CPU launcher 대신 PIM runtime을 dlopen해 호출
- [ ] **M5.5** `CMakeLists.txt` 업데이트 (UPMEM SDK include/link)

---

## M6 — End-to-end: MatMul 실행 + 검증

**목표**: Triton Python → ... → UPMEM device 실행까지 한 번에 통과, 정답 검증. **이 지점이 MVP 완료 = "돈다".**

**완료 조건**: `numpy.allclose(C_pim, A @ B)` 통과.

**작업 단위**:

- [ ] **M6.1** `pim_backend/examples/test_pim_matmul.py` (DESIGN.md 22장 기준, `A(M,K)×B(K,N)`)
- [ ] **M6.2** M0의 `add_stages`를 실제 PIM 파이프라인으로 교체 (CPU lowering 제거, M3~M4 pass 사용)
- [ ] **M6.3** small shape (M=32, N=64, K=256) 실행 + 검증 (실패 시 M2 dump로 단계별 IR 추적)
- [ ] **M6.4** medium shape (M=256, N=512, K=512) 실행 + 검증

> ✅ **여기까지가 MVP.** 이후 Part II는 전부 이 상태를 baseline으로 유지하며 additive하게 진행한다.

---

# Part II — Incremental Optimization

> 규칙: 각 항목은 **독립 toggle 가능**, **OFF 시 M6가 그대로 통과**, **각 pass는 lit test + e2e 재검증**. DESIGN.md 13.0 경계 원칙(구조/intent/gate만) 준수. (Debug/dump는 M2에서 이미 구축됨.)

## O1 — 단일 변환을 granular plan pass로 분해

**목표**: M3의 단일 `convert-ttshared-to-pim`을 DESIGN.md 13.1의 granular plan pass들로 분해한다. **behavior-preserving 리팩터** — no-split 입력에 대해 분해 전후 출력 IR(및 최종 vendor artifact)이 동일해야 한다. 이 분해가 이후 per-pass 최적화의 토대가 된다.

**작업 단위**:

- [ ] **O1.1** `build-pim-memory-plan` (host/device memory_region 생성) 분리
- [ ] **O1.2** `build-pim-tile-plan` (주어진 block/split **기록만**, 최적화 X)
- [ ] **O1.3** `build-pim-transfer-plan` (transfer 골격 삽입)
- [ ] **O1.4** `build-pim-execution-plan` (mapping_policy **enum**만)
- [ ] **O1.5** `build-pim-placement-plan` (placement policy **enum**만; 구체 산술은 vendor)
- [ ] **O1.6** `build-pim-residency-plan` (residency enum)
- [ ] **O1.7** 동치 검증: no-split에 대해 분해 파이프라인 출력 == M3 출력(M2 dump로 비교), M6 재통과

---

## O2 — Capability / Constraints

**목표**: vendor capability/constraint를 도입한다. O3의 gate와 O4의 적법성 필터가 이걸 query한다. (그래서 최적화 pass보다 먼저.)

**작업 단위**:

- [ ] **O2.1** `PIMCapability`, `PIMConstraints` dataclass (DESIGN.md 19장)
- [ ] **O2.2** UPMEM `queryCapability`/`queryConstraints` C++ 구현 (실제 DPU 수치)
- [ ] **O2.3** `#pim.capability_ref<...>` attribute가 lowering 시 capability 값으로 구체화

---

## O3 — Common Optimization Passes (하나씩)

**목표**: DESIGN.md 13.2 최적화 pass를 **하나씩** 추가. 각 pass는 13.0 경계 준수(intent/gate/구조만), toggle 가능, lit test + e2e 재검증.

**작업 단위** (각 pass별 lit test 포함):

- [ ] **O3.1** `pim-layout-normalize` — 표현 정규화만 (중복 transpose 제거·단일 canonical form). 물리 layout은 건드리지 않음
- [ ] **O3.2** `pim-data-placement` — placement **intent enum**만 갱신
- [ ] **O3.3** `pim-chunk-planning` — `num_chunks=ceildiv(K,BLOCK_K)` **산술만**
- [ ] **O3.4** `pim-transfer-hoisting` — 반복 transfer를 상위 scope로
- [ ] **O3.5** `pim-persistent-data` — weight/KV를 persistent로 **표시**
- [ ] **O3.6** `pim-partial-result-planning` — partial/reduce **구조**만 (split_k>1일 때)
- [ ] **O3.7** `pim-pipeline-overlap` — **capability-gated**: `supports_pipeline_overlap`일 때만 `pim.pipeline` 삽입
- [ ] **O3.8** end-to-end matmul로 baseline(MVP) 대비 성능 개선 측정

---

## O4 — Execution Strategy Selection (plan dispatch)

**목표**: DESIGN.md 13.4 구현. `(SPLIT_M/K/N)`에 따라 no-split / N-split / K-split / M-split 전략을 선택. 선택은 Common IR에서 vendor-aware하게, 실현은 vendor lowering에서. **cost model 없음** — 정적으로는 적법성만, 우열은 autotune 실측 (13.4.5).

**전제**: O2(capability) 완료. 적법성 필터가 `PIMCapability`/`PIMConstraints`를 query.

**작업 단위**:

- [ ] **O4.1** `PIMExecutionStrategy` interface (vendor-neutral): `isApplicable(tile_plan, capability, constraints)` + `applyToIR(builder)` (cost 메서드 없음)
- [ ] **O4.2** 전략 구현 + 레지스트리 등록: `no-split`, `n-split`(output_n_partitioned), `k-split`(reduction_k_partitioned), `m-split`(output_m_partitioned, 전치 포함)
- [ ] **O4.3** `SelectPIMExecutionStrategy` Common pass — split factor + vendor capability/constraints로 적법 전략 필터, autotune 미사용 시 단순 기본 규칙(no-split 우선)
- [ ] **O4.4** vendor lowering이 전략 attribute legalize (`reduction_k_partitioned` → UPMEM host_reduce) + lit test
- [ ] **O4.5** e2e: SPLIT에 따라 다른 `mapping_policy`/`reduction`이 IR에 박히는지 검증

---

## O5 — Autotune Integration

**목표**: `@triton.autotune`이 O3/O4의 후보들을 실측으로 선택하도록 연결. PIM 전용 노브(SPLIT_*/NUM_UNITS/LOCAL_LANES)는 별도 annotation pass 없이 plan pass가 option에서 읽음 (DESIGN.md 17.0).

**작업 단위**:

- [ ] **O5.1** PIM 전용 노브를 compile option으로 받아 `build-pim-tile-plan`(SPLIT_*) / `build-pim-execution-plan`(NUM_UNITS·LOCAL_LANES)이 IR attr로 반영
- [ ] **O5.2** autotune config별 독립 IR/artifact 생성·실측이 동작하는지 검증 (config N개 → N벌 컴파일)
- [ ] **O5.3** O4 전략 후보를 autotune이 실측 선택하는 통합 (vendor별 best가 달라지는지 확인)

---

# Part III — Breadth

## B1 — AiMX Vendor Extension

**목표**: 두 번째 vendor를 추가해 vendor-neutral 추상화가 실제로 작동함을 검증.

**작업 단위**:

- [ ] **B1.1** AiMX dialect 최소셋 (`aimx.dma_h2d`, `dma_d2h`, `compute_gemm`, `descriptor_chain`, `submit`, `wait`)
- [ ] **B1.2** `LegalizePIMToAiMX` conversion
- [ ] **B1.3** `AiMXMemoryBankMapping` + `AiMXExecutionMapping`
- [ ] **B1.4** `AiMXCommandStreamCodegen` (descriptor binary emission)
- [ ] **B1.5** `runtime/pim/vendors/aimx/AiMXLauncher.cpp`
- [ ] **B1.6** `pim_backend/examples/test_pim_matmul_aimx.py` 실행 + 검증

---

## B2 — Vendor Extension Registry 리팩토링

**목표**: if-else vendor dispatch를 DESIGN.md 7장 `VendorExtensionRegistry`로 통합. 두 vendor가 모두 동작한 뒤 진행 (premature abstraction 회피).

**작업 단위**:

- [ ] **B2.1** Python측 `VendorExtensionRegistry` (DESIGN.md 7장)
- [ ] **B2.2** C++측 `PIMVendorExtension` interface (DESIGN.md 8장) + UPMEM/AiMX 등록
- [ ] **B2.3** `PIMBackend.add_stages`를 `self.vendor_extension.build_*_pipeline(options)` 형태로 리팩토링

---

## B3 — CPU+PIM HybridDriver (동시 실행)

**목표**: 동일 TTSharedIR에서 컴파일된 CPU binary와 PIM binary를 런타임에 tile grid 분할해 동시 실행 (DESIGN.md 25장).

**완료 조건**: `numpy.allclose(C_hybrid, A @ B)` 통과 + CPU 단독 대비 시간 단축.

**작업 단위**:

- [ ] **B3.1** `UPMEMCodegen`에 `tile_offset`/`tile_count` 파라미터 추가 (DPU kernel이 grid 일부만 처리)
- [ ] **B3.2** `HybridPartitionPolicy` — grid를 cpu_range/pim_range로 분할 (half/pim_ratio/auto)
- [ ] **B3.3** `HybridDriver` — cpu_launcher + pim_launcher를 Thread로 동시 실행, 동기화
- [ ] **B3.4** `device="hybrid"` 라우팅 — CPUBackend + PIMBackend 각각 컴파일 후 HybridDriver에 전달
- [ ] **B3.5** `pim_backend/examples/test_hybrid_matmul.py` — small shape 정답 + 시간 출력

---

## 의존성 그래프

```text
── Part I (MVP) ──────────────────────────
M0 skeleton
  ↓
M1 dialect
  ↓
M2 debug/dump   ← 변환 확인 도구 (M3~M6에서 계속 사용)
  ↓
M3 single conversion (no-split)
  ↓
M4 UPMEM lowering (no-split)
  ↓
M5 runtime
  ↓
M6 E2E 검증  ← ✅ "돈다"
  │
── Part II (최적화, additive) ─────────────
  ├─ O1 granular 분해 ──┐
  │                     ↓
  │                  O2 capability ──┐
  │                                  ↓
  │                               O3 opt passes (하나씩)
  │                                  ↓
  │                               O4 strategy selection
  │                                  ↓
  │                               O5 autotune
  │
── Part III (확장) ────────────────────────
  ├─ B1 AiMX ──→ B2 registry refactor
  └─ B3 Hybrid
```

원칙:
- M2(dump)는 M0(add_stages) 직후면 동작. M3/M4 stage가 늘면 dump가 자동 캡처 → 변환 즉시 검증.
- Part II는 전부 M6(MVP) 완료 후. 각 항목은 OFF 시 M6가 그대로 통과해야 함.
- O2(capability)는 O3의 gate·O4의 적법성 필터가 의존하므로 그 앞.
- O4는 O2 완료 후. O5는 O3/O4 후보를 실측 선택.
- B1(AiMX)은 M6 후 언제든. B2는 두 vendor 동작 후. B3는 M6 후 언제든 병렬 가능.

---

## 다음 액션

현재 다음 작업을 시작하면 됩니다:

- [x] **M0.1** — `triton-shared/pim_backend/backend/` 디렉토리 생성 및 `name.conf` 작성
