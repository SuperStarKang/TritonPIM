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
| M2 | Debug / IR Dump (변환 확인용, 먼저 구축) | `[x]` | 3 / 3 |
| M3 | TTShared→PIM 단일 변환 (no-split GEMM) | `[x]` | 4 / 4 |
| M4 | 최소 UPMEM lowering + codegen (no-split) | `[x]` | 5 / 5 |
| M5 | Runtime & Launcher | `[x]` | 4 / 4 |
| M6 | End-to-end: MatMul 실행 + 검증 | `[x]` | 4 / 4 |
| **Part II — Incremental Optimization** | | | |
| O1 | 단일 변환을 granular plan pass로 분해 + multi-tile 일반화 | `[~]` | 1 / 8 |
| O2 | Capability / Constraints | `[ ]` | 0 / 3 |
| O3 | Common Optimization Passes (하나씩) | `[ ]` | 0 / 8 |
| O4 | Execution Strategy Selection | `[~]` | 2 / 7 |
| O5 | Autotune Integration | `[ ]` | 0 / 3 |
| **Part III — Breadth** | | | |
| B1 | AiMX Vendor Extension | `[ ]` | 0 / 6 |
| B2 | Vendor Extension Registry 리팩토링 | `[ ]` | 0 / 3 |
| B3 | CPU+PIM HybridDriver | `[ ]` | 0 / 5 |

총 진행률: **39 / 79** (Part I MVP 완료, O1.8 multi-tile 1-DPU, O4 n-split(split_n) + multi-tasklet(local_lanes))

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
- [x] **M2.3** `pim_common`/`vendor_ir`/codegen artifact가 같은 메커니즘으로 자동 dump됨 확인 (dump 디렉토리에 `matmul_kernel.pim_common`·`.vendor_ir` 생성 확인)

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

**목표**: PIM Common IR → UPMEM IR → host.cpp + device.c **생성**. **no-split만** (DPU 1개가 단일 tile 계산). device 커널은 prebuilt를 링크하지 않고 **IR에서 생성**한다 (block 정보를 활용·확장 가능하게). PGEMM은 SDK 관용구 참고용.

**codegen 메커니즘**: MLIR에서 IR→소스 텍스트는 pass가 아니라 **translate**의 일이다. 새 도구 `triton-shared-translate`를 만들고, `--upmem-to-host` / `--upmem-to-device` 번역을 등록해 UPMEM IR을 walk하며 host.cpp / device.c를 emit한다.

**완료 조건**: `triton-shared-opt --convert-ttshared-to-pim --legalize-pim-to-upmem` → `triton-shared-translate --upmem-to-host`(host.cpp) / `--upmem-to-device`(device.c) 생성. (실제 컴파일·실행은 M5.)

**작업 단위**:

- [x] **M4.1** UPMEM dialect 최소 ops (`upmem.host_func`, `device_func`, `alloc_units`, `copy_to_global`, `copy_from_global`, `launch`, `local_gemm`)
- [x] **M4.2** `LegalizePIMToUPMEM` (no-split: func(host 시퀀스) + device_func(local_gemm)로 분리)
- [x] **M4.3** `triton-shared-translate` 도구 + `--upmem-to-host`/`--upmem-to-device` 번역: UPMEM IR을 walk해 host.cpp + device.c 생성. MVP device.c = 최소 gemm(single tasklet, host가 B 전치)
- [x] **M4.4** `triton-shared-opt`에 UPMEM dialect/passes 등록
- [x] **M4.5** test: `test/Target/UPMEM/matmul.mlir` — `triton-shared-translate`가 host.cpp/device.c 생성 확인 (실제 컴파일은 M5)

---

## M5 — Runtime & Launcher

**목표**: 생성된 host.cpp/device.c를 컴파일해 실제 DPU에서 실행한다. **생성된 host.cpp가 dpu_alloc/load/copy/launch를 직접 수행하므로(=런타임 glue), 별도 `pim_runtime.h` C API 계층은 두지 않는다** (DESIGN 18은 MVP 미사용). CPULauncher가 launcher .so를 컴파일·dlopen하는 패턴을 그대로 따른다.

**환경**: DPU 컴파일러 `dpu-upmem-dpurte-clang`(/usr/bin), host 헤더 `/usr/include/dpu`, host는 `libdpu` 링크.

**완료 조건**: `matmul_kernel_host(A,B,C)` 호출로 DPU에서 gemm 실행 성공.

**작업 단위**:

- [x] **M5.1** device.c → device.elf 컴파일 (`dpu-upmem-dpurte-clang -DNR_TASKLETS=1`) — runtime_test에서 검증
- [x] **M5.2** host.cpp → 컴파일·실행 (`g++ + libdpu`, `dpu_broadcast_to`/`dpu_push_xfer` 패턴, WRAM `__dma_aligned`) — runtime_test `max_diff=0` 통과
- [x] **M5.3** `PIMLauncher` 갱신: artifact(생성 소스)를 device.elf+host.so로 컴파일·캐시하고 dlopen, `<kernel>_host(A,B,C)`를 ctypes 호출 — `test_pim_matmul.py`로 검증
- [x] **M5.4** UPMEM SDK 경로/플래그 확인 (`dpu-upmem-dpurte-clang`, `/usr/include/dpu`, `-ldpu`, `UPMEM_PROFILE=chipId=0x42`)

---

## M6 — End-to-end: MatMul 실행 + 검증

**목표**: Triton Python → ... → UPMEM device 실행까지 한 번에 통과, 정답 검증. **이 지점이 MVP 완료 = "돈다".**

**완료 조건**: `numpy.allclose(C_pim, A @ B)` 통과.

**작업 단위**:

- [x] **M6.1** `pim_backend/examples/test_pim_matmul.py` (`A(M,K)×B(K,N)`, 단일 tile)
- [x] **M6.2** `add_stages`를 실제 PIM 파이프라인으로 교체 (ttir→ttsharedir→pim_common→vendor_ir→pimobj; CPU lowering 제거)
- [x] **M6.3** small shape (M=32, N=64, K=256) 실행 + 검증 통과 (`assert_close(C, A@B)`)
- [x] **M6.4** medium shape (M=256, N=512, K=512) 실행 + 검증 통과

> ✅ **여기까지가 MVP.** 이후 Part II는 전부 이 상태를 baseline으로 유지하며 additive하게 진행한다.

---

# Part II — Incremental Optimization

> 규칙: 각 항목은 **독립 toggle 가능**, **OFF 시 M6가 그대로 통과**, **각 pass는 lit test + e2e 재검증**. DESIGN.md 13.0 경계 원칙(구조/intent/gate만) 준수. (Debug/dump는 M2에서 이미 구축됨.)

> **공통 vs vendor 구분 태그** (DESIGN 13.0/13.3): 각 항목 앞에 계층을 표시한다.
> - `[공통·구조]` — PIM Common IR에서 값/패턴을 **기록·표현만** (결정/최적화 없음, vendor-neutral)
> - `[공통·intent]` — vendor-neutral **policy enum**만 표현 (구체 산술은 vendor)
> - `[공통·gate]` — vendor **capability를 query**해 동작 여부 결정 (vendor-aware하지만 pass 자체는 공통)
> - `[vendor]` — 위 intent/전략을 특정 하드웨어 산술·코드로 **실현(HOW)**. UPMEM/AiMX별 구현
> - `[검증]` — lit/e2e 테스트 (벤더 중립 도구)
>
> 원칙: **WHAT(선택·표현)은 공통, HOW(실현·legalize·codegen)는 vendor.** 같은 O-그룹 안에서도 항목마다 계층이 다를 수 있으니 구현 시 디렉토리(`lib/Conversion/*` vs `lib/Conversion/PIMToUPMEM`·`lib/Target/UPMEM`)도 그에 맞춰 둔다.

> **전제 ① — 전체 문제(full-problem) 정보 복원**: MVP의 M3 변환은 single-tile만 가정해 **grid·program_id·K-loop을 버리고** "타일 하나 = 전체 gemm"으로 취급한다. 실제 multi-tile matmul(grid>1, K-loop)에서 tiling/SPLIT 최적화를 하려면, **전체 M/N/K·grid·program_id→tile 매핑을 복원**해 `tile_plan`(BLOCK/SPLIT)과 `chunk_plan`(BLOCK_K)으로 표현해야 한다. 이 복원이 O3.3(chunk-planning)·O4(strategy)의 전제이며, **O1 변환 일반화 시 함께 수행**한다 (MVP는 BLOCK=전체라 이 정보가 비어 있어도 동작).

> **전제 ② — codegen이 IR 값을 받아 생성**: plan/strategy pass는 IR에 intent/구조(`num_units`·tile offset·`chunk_plan`·tasklet 수 등)를 표현한다 (DESIGN 13.0 경계). 그러나 현재 translate(`UPMEMToHost`/`UPMEMToDevice`)는 이 값들을 **읽지 않고 고정 템플릿**을 emit한다: `dpu_alloc(1,...)`(항상 DPU 1개), `local_gemm{m,n,k}`를 상수로 박음, device는 `if(me()!=0) return`(tasklet 1개)로 전체를 혼자 계산. 따라서 SPLIT(여러 DPU)·chunk(큰 K)·multi-tasklet은 IR 표현만으론 실행되지 않고, **translate·launcher가 그 IR 값을 읽어 parameterized code를 생성**하도록 일반화해야 한다 (O4.6). "IR에 박힘"과 "여러 DPU에서 실제로 돔"은 다른 단계임에 주의.

## O1 — 단일 변환을 granular plan pass로 분해

**목표**: M3의 단일 `convert-ttshared-to-pim`을 DESIGN.md 13.1의 granular plan pass들로 분해한다. **behavior-preserving 리팩터** — no-split 입력에 대해 분해 전후 출력 IR(및 최종 vendor artifact)이 동일해야 한다. 이 분해가 이후 per-pass 최적화의 토대가 된다.

**작업 단위**:

- [ ] **O1.1** `[공통·구조]` `build-pim-memory-plan` (host/device memory_region 생성) 분리
- [ ] **O1.2** `[공통·구조]` `build-pim-tile-plan` (주어진 block/split **기록만**, 최적화 X) — 실제 타일 micro-fit은 vendor
- [ ] **O1.3** `[공통·구조]` `build-pim-transfer-plan` (transfer 골격 삽입) — 실제 copy 방식(DMA/dpu_copy)은 vendor
- [ ] **O1.4** `[공통·intent]` `build-pim-execution-plan` (mapping_policy **enum**만) — 실제 DPU/channel 배정은 vendor
- [ ] **O1.5** `[공통·intent]` `build-pim-placement-plan` (placement policy **enum**만) — bank/channel 오프셋 산술은 vendor(`UPMEMMemoryPlacement`)
- [ ] **O1.6** `[공통·intent]` `build-pim-residency-plan` (residency enum) — 실제 상주 버퍼 관리는 vendor
- [ ] **O1.7** `[검증]` 동치 검증: no-split에 대해 분해 파이프라인 출력 == M3 출력(M2 dump로 비교), M6 재통과
- [x] **O1.8** `[공통·구조]` 전체 M/N/K 복원: 변환이 tiled-gemm(scf.for+matmul+accumulate) 인식, 전체 M/N/K(`%arg3/4/5`)를 `pim.compute extent`로 운반(BLOCK은 tile_plan에 기록). codegen(UPMEMToHost·launcher)이 baked 상수 대신 런타임 M/N/K로 동작 → **multi-tile이 1 DPU에서 정답**(128×128×256, grid=(2,2) 검증). `chunk_plan`(BLOCK_K streaming)은 O3.3, grid→유닛 SPLIT은 O4에서.

---

## O2 — Capability / Constraints

**목표**: vendor capability/constraint를 도입한다. O3의 gate와 O4의 적법성 필터가 이걸 query한다. (그래서 최적화 pass보다 먼저.)

**작업 단위**:

- [ ] **O2.1** `[공통]` `PIMCapability`, `PIMConstraints` dataclass — vendor-neutral 스키마/인터페이스 (DESIGN.md 19장)
- [ ] **O2.2** `[vendor]` UPMEM `queryCapability`/`queryConstraints` C++ 구현 (실제 DPU 수치)
- [ ] **O2.3** `[공통→vendor]` `#pim.capability_ref<...>` attribute는 공통 표현, **lowering 시 vendor가** capability 값으로 구체화

---

## O3 — Common Optimization Passes (하나씩)

**목표**: DESIGN.md 13.2 최적화 pass를 **하나씩** 추가. 각 pass는 13.0 경계 준수(intent/gate/구조만), toggle 가능, lit test + e2e 재검증.

**작업 단위** (각 pass별 lit test 포함):

- [ ] **O3.1** `[공통·구조]` `pim-layout-normalize` — 표현 정규화만 (중복 transpose 제거·단일 canonical form). 물리 layout(`pack_by_*`)은 vendor
- [ ] **O3.2** `[공통·intent]` `pim-data-placement` — placement **intent enum**만 갱신. 구체 분산은 vendor
- [ ] **O3.3** `[공통·구조]` `pim-chunk-planning` — `num_chunks=ceildiv(K,BLOCK_K)` **산술만**. register/WRAM fitting은 vendor(O4.6 codegen)
- [ ] **O3.4** `[공통·구조]` `pim-transfer-hoisting` — 반복 transfer를 상위 scope로
- [ ] **O3.5** `[공통·intent]` `pim-persistent-data` — weight/KV를 persistent로 **표시**. 실제 상주는 vendor
- [ ] **O3.6** `[공통·구조]` `pim-partial-result-planning` — partial/reduce **구조**만 (split_k>1일 때). 실제 reduce는 vendor
- [ ] **O3.7** `[공통·gate]` `pim-pipeline-overlap` — `supports_pipeline_overlap`일 때만 `pim.pipeline` 삽입. 실제 async/double-buffer는 vendor
- [ ] **O3.8** `[검증]` end-to-end matmul로 baseline(MVP) 대비 성능 개선 측정

---

## O4 — Execution Strategy Selection (plan dispatch)

**목표**: DESIGN.md 13.4 구현. `(SPLIT_M/K/N)`에 따라 no-split / N-split / K-split / M-split 전략을 선택. 선택은 Common IR에서 vendor-aware하게, 실현은 vendor lowering에서. **cost model 없음** — 정적으로는 적법성만, 우열은 autotune 실측 (13.4.5).

**전제**: O2(capability) 완료. 적법성 필터가 `PIMCapability`/`PIMConstraints`를 query.

**작업 단위**:

- [ ] **O4.1** `[공통]` `PIMExecutionStrategy` interface (vendor-neutral): `isApplicable(tile_plan, capability, constraints)` + `applyToIR(builder)` (cost 메서드 없음)
- [ ] **O4.2** `[공통]` 전략 구현 + 레지스트리 등록 (전략 목록은 vendor-neutral): `no-split`, `n-split`(output_n_partitioned), `k-split`(reduction_k_partitioned), `m-split`(output_m_partitioned, 전치 포함)
- [ ] **O4.3** `[공통·gate]` `SelectPIMExecutionStrategy` Common pass — split factor + vendor capability/constraints로 적법 전략 필터, autotune 미사용 시 단순 기본 규칙(no-split 우선)
- [ ] **O4.4** `[vendor]` UPMEM lowering이 전략 attribute legalize (`reduction_k_partitioned` → UPMEM `host_reduce`) + lit test
- [~] **O4.5** `[검증]` e2e: SPLIT에 따라 동작 검증 — **n-split `split_n=2` 정답 통과**(`test_pim_matmul_tiled.py`). k/m-split·reduce 검증은 이후.
- [~] **O4.6** `[vendor]` translate·launcher가 IR 값을 받아 생성(전제 ②). **완료**: (a) n-split host codegen(`alloc_units{num_units}`→`dpu_alloc(U)`, A broadcast/B 열scatter/C gather, split_n=2 검증). (b) **device BLOCK 타일링** — `UPMEMToDevice`가 `local_gemm`의 BLOCK_M/N/K를 bake, 부분행렬을 `BLOCK_M×BLOCK_N` 타일로 순회하며 `BLOCK_K`를 WRAM(aW/bW/acc)에 올려 누적·재사용. **①a 단일 tasklet** + **①b cooperative 다중 tasklet**(WRAM 타일 공유 + BLOCK_M 행 tasklet 분할 + barrier) 모두 HW에서 검증 (split_n=2 × local_lanes=8/16). **남음**: k-split `host_reduce`. **제약**: WRAM = aW+bW+acc+tasklet 스택 ≤ 64KB → BLOCK/local_lanes 균형(autotune), per-call `dpu_alloc/load`는 persistent set으로 개선 여지.
- [x] **O4.7** `[공통 knob + vendor]` multi-tasklet (intra-DPU 병렬, `local_lanes`=`NR_TASKLETS`): `PIMOptions.local_lanes`(공통 knob, default 1) → `metadata` → launcher가 `-DNR_TASKLETS=L`로 device.c 컴파일. `UPMEMToDevice`가 `NR_TASKLETS` 매크로 기반 device.c 생성 — `BARRIER_INIT` + `me()==0`이 `mem_reset` 후 barrier, tasklet별 `mem_alloc` WRAM, `for(i=me();i<M;i+=NR_TASKLETS)` 행 round-robin(행 disjoint→끝 barrier 불필요). **검증: `split_n=2, local_lanes=16` (2 DPU × 16 tasklet) 정답 통과.** (DESIGN 14.6)

> **수직 슬라이스 (완료)**: 전략 프레임워크(O4.1~O4.4)를 건너뛰고 옵션으로 끝까지 관통 —
> - `split_m`/`split_n` (inter-DPU 2D 분배): `tile_plan.split_m/n` → `alloc_units{num_units,split_m,split_n}` → `UPMEMToHost` 2D scatter/gather (A 행밴드·B 열밴드·C 2D). **HW 선형 스케일링 검증**(1→4 DPU ~3.8×).
> - `BLOCK_M/N/K` (intra-DPU 타일): `UPMEMToDevice`가 BLOCK 타일 순회 + `BLOCK_K` WRAM 청크 재사용 (cooperative 다중 tasklet).
> - `local_lanes` (intra-DPU 병렬): `metadata`→launcher `-DNR_TASKLETS`→ tasklet이 BLOCK_M 행 분할.
>
> **다음 후보**: ③ SPLIT_K + `host_reduce`(타일<DPU), ④ persistent DPU set(per-call alloc/load 제거), O4.1~O4.4 전략 프레임워크.

---

## O5 — Autotune Integration

**목표**: `@triton.autotune`이 O3/O4의 후보들을 실측으로 선택하도록 연결. PIM 전용 노브(SPLIT_*/NUM_UNITS/LOCAL_LANES)는 별도 annotation pass 없이 plan pass가 option에서 읽음 (DESIGN.md 17.0).

**작업 단위**:

- [ ] **O5.1** `[공통]` PIM 전용 노브를 compile option으로 받아 `build-pim-tile-plan`(SPLIT_*) / `build-pim-execution-plan`(NUM_UNITS·LOCAL_LANES)이 IR attr로 반영
- [ ] **O5.2** `[검증]` autotune config별 독립 IR/artifact 생성·실측이 동작하는지 검증 (config N개 → N벌 컴파일)
- [ ] **O5.3** `[공통+vendor]` O4 전략 후보를 autotune이 실측 선택하는 통합 — 후보 선택은 공통, 각 후보의 실측 대상 artifact는 vendor codegen(O4.6) (vendor별 best가 달라지는지 확인)

---

# Part III — Breadth

## B1 — AiMX Vendor Extension

**목표**: 두 번째 vendor를 추가해 vendor-neutral 추상화가 실제로 작동함을 검증.

**작업 단위**:

- [ ] **B1.1** `[vendor]` AiMX dialect 최소셋 (`aimx.dma_h2d`, `dma_d2h`, `compute_gemm`, `descriptor_chain`, `submit`, `wait`)
- [ ] **B1.2** `[vendor]` `LegalizePIMToAiMX` conversion (공통 PIM IR → AiMX IR)
- [ ] **B1.3** `[vendor]` `AiMXMemoryBankMapping` + `AiMXExecutionMapping`
- [ ] **B1.4** `[vendor]` `AiMXCommandStreamCodegen` (descriptor binary emission)
- [ ] **B1.5** `[vendor]` `runtime/pim/vendors/aimx/AiMXLauncher.cpp`
- [ ] **B1.6** `[검증]` `pim_backend/examples/test_pim_matmul_aimx.py` 실행 + 검증

---

## B2 — Vendor Extension Registry 리팩토링

**목표**: if-else vendor dispatch를 DESIGN.md 7장 `VendorExtensionRegistry`로 통합. 두 vendor가 모두 동작한 뒤 진행 (premature abstraction 회피).

**작업 단위**:

- [ ] **B2.1** `[공통]` Python측 `VendorExtensionRegistry` (DESIGN.md 7장)
- [ ] **B2.2** `[공통 interface + vendor 등록]` C++측 `PIMVendorExtension` interface (DESIGN.md 8장) + UPMEM/AiMX 어댑터 등록
- [ ] **B2.3** `[공통]` `PIMBackend.add_stages`를 `self.vendor_extension.build_*_pipeline(options)` 형태로 리팩토링

---

## B3 — CPU+PIM HybridDriver (동시 실행)

**목표**: 동일 TTSharedIR에서 컴파일된 CPU binary와 PIM binary를 런타임에 tile grid 분할해 동시 실행 (DESIGN.md 25장).

**완료 조건**: `numpy.allclose(C_hybrid, A @ B)` 통과 + CPU 단독 대비 시간 단축.

**작업 단위**:

- [ ] **B3.1** `[vendor]` `UPMEMCodegen`에 `tile_offset`/`tile_count` 파라미터 추가 (DPU kernel이 grid 일부만 처리) — O4.6의 parameterization 재사용
- [ ] **B3.2** `[공통]` `HybridPartitionPolicy` — grid를 cpu_range/pim_range로 분할 (half/pim_ratio/auto)
- [ ] **B3.3** `[공통]` `HybridDriver` — cpu_launcher + pim_launcher를 Thread로 동시 실행, 동기화
- [ ] **B3.4** `[공통]` `device="hybrid"` 라우팅 — CPUBackend + PIMBackend 각각 컴파일 후 HybridDriver에 전달
- [ ] **B3.5** `[검증]` `pim_backend/examples/test_hybrid_matmul.py` — small shape 정답 + 시간 출력

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

O1.8(multi-tile 1-DPU) + O4 n-split(`split_n`, 여러 DPU) + multi-tasklet(`local_lanes`, DPU 내부 병렬) 완료. 현재 `split_n=2, local_lanes=16` (2 DPU × 16 tasklet) 정답.

**실행 모델 확정 (DESIGN 11.5.2): SPLIT(굵게, 유닛 간) → BLOCK(잘게, 유닛 내부 타일) → BLOCK_K(WRAM 청크, A/B 재사용).** "1 BLOCK = 1 DPU"는 폐기 — BLOCK은 DPU 분배 단위가 아니라 DPU 내부 타일이다. PGEMM `basic_n`과 분배(2D 출력분배)는 동일하나, **BLOCK 타일링으로 WRAM 재사용**하는 게 차이·이점.

① **device BLOCK 타일 루프** (intra-DPU, O4.6) — **완료**: ①a 단일 + ①b cooperative 다중 tasklet (BLOCK_M/N/K 타일 + BLOCK_K WRAM 재사용 + tasklet 행분할, HW 검증). 환경: 시뮬레이터→HW(`UPMEM_BACKEND=hw`+upmem-override lib).
② **2D SPLIT (`split_m` 추가)** — **완료**: `SPLIT_M×SPLIT_N` DPU에 출력 2D 분배, **A 행밴드 scatter(전체 복제 해소)** + B 열밴드 + C 2D gather. `tile_plan.split_m` → `alloc_units{num_units,split_m,split_n}` → `UPMEMToHost` 2D. **HW 검증 + 선형 스케일링**(1→2→4 DPU = 3833→1949→1008 ms, ~3.8×). split_n은 SPLIT_M=1 특수형.
③ **SPLIT_K + reduce** — 타일 < DPU일 때 한 타일을 여러 DPU가 K분담(`host_reduce`). (O4.1~O4.4 전략 프레임워크와 함께.) ← 다음 후보
④ **persistent DPU set** — 매 호출 `dpu_alloc/load/free` 대신 1회 alloc/load 재사용 (per-call 오버헤드 제거; 작은 행렬 절대 시간 개선). ← 다음 후보

계층: SPLIT/BLOCK/LOCAL_LANES **knob·intent는 공통**(option/`tile_plan`/metadata), **실현(dpu_alloc·DPU_FOREACH·NR_TASKLETS·me()·BLOCK_K WRAM)은 UPMEM(`lib/Target/UPMEM`)**.
