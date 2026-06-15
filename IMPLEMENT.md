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
| M1 | 최소 PIM Common Dialect (coarse 5 ops · StrAttr) | `[x]` | 7 / 8 |
| M2 | Debug / IR Dump (변환 확인용, 먼저 구축) | `[x]` | 3 / 3 |
| M3 | TTShared→PIM 단일 변환 (no-split GEMM) | `[x]` | 4 / 4 |
| M4 | 최소 UPMEM lowering + codegen (no-split) | `[x]` | 5 / 5 |
| M5 | Runtime & Launcher | `[x]` | 4 / 4 |
| M6 | End-to-end: MatMul 실행 + 검증 | `[x]` | 4 / 4 |
| **Part II — Incremental Optimization** | | | |
| O1 | faithful-body 전환 (본문 보존 + build-pim-offload 통합) + multi-tile | `[~]` | 5 / 6 |
| O2 | Capability / Constraints | `[ ]` | 0 / 3 |
| O3 | Common Optimization Passes (하나씩, opt-in) | `[ ]` | 0 / 7 |
| O4 | Execution Strategy Selection | `[~]` | 3 / 7 |
| O5 | Autotune Integration | `[ ]` | 0 / 3 |
| **Part III — Breadth** | | | |
| B1 | AiMX Vendor Extension | `[~]` | 1 / 7 |
| B2 | Vendor Extension Registry 리팩토링 | `[ ]` | 0 / 3 |
| B3 | CPU+PIM HybridDriver | `[ ]` | 0 / 5 |

총 진행률: **44 / 77** (Part I MVP 완료. O1.1/O1.2/O1.3/O1.5 **faithful-body IR 변환 + build-pim-offload·execution_group 완료**(`pim.compute` region-carrying, 본문 보존 + program_id 로컬화, `build-pim-offload`가 `execution_group` 합성, e2e 동치 검증). O1.4a(device가 본문 gemm nest에서 BLOCK 읽어 device.c 구동, `local_gemm` 제거, e2e 동치). O1.8 multi-tile, O4 2D SPLIT + **SPLIT_K(host tree-reduce)** + BLOCK 타일 + multi-tasklet HW 검증(2→4→8 DPU 선형 스케일). **host staging 중복 제거(2048-DPU 8029→4165ms) + dtype-aware codegen(int8 입력/int32 누적 검증 — f32 대비 ~5.9× 4535→772ms, 정답 정확).** **O1 남음: O1.4b(비-gemm op-by-op 번역)**)

> **B1.0 (AiMX vendor-dispatch seam) 완료**: `compiler.py`가 `options.vendor`로 `vendor_ir`/`pimobj` stage를 분기(upmem|aimx, 미지원 값은 fail-fast)하고 `_pim_common_to_aimx`/`_aimx_to_artifact`를 upmem helper와 평행 추가. `driver.py` PIMLauncher는 `metadata.vendor`로 빌드 단계를 분기하고 `_build_aimx_so`(seam, `NotImplementedError`)를 둔다. **AiMX legalize/translate/toolchain은 미구현** — `vendor="aimx"` 실행 시 그 자리에서 fail(의도). 실제 구현은 별도 서버/브랜치에서 B1.1~B1.6으로 채운다.

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

## M1 — 최소 PIM Common Dialect (coarse, 5 ops · StrAttr)

**목표**: vendor-neutral PIM dialect 최소셋 정의. DESIGN.md 10장 전체가 아니라 no-split GEMM 1개에 필요한 최소 op만. **실제 구현은 5 ops + 2 types를 단일 `PIMDialect.td`에 담았고, 모든 attribute는 StrAttr/I64/I32다** (DESIGN의 enum attr·region-carrying compute·`pim.func`는 미도입 — O1/O2에서 확장).

**구현된 5 ops** (`pim.func`는 미도입 — `func.func` 사용):

```
pim.memory_region     (buffer + space/layout/role — 모두 StrAttr)
pim.tile_plan         (block_m/n/k, split_m/n/k — I64Attr)
pim.transfer          (region, src/dst — StrAttr)
pim.compute           (coarse: a,b,c + tile_plan + full_m/n/k:I32 + op="gemm":StrAttr, region 없음)
pim.launch            (callee, vendor, device_id)
```

> coarse `pim.compute`는 본문(scf/linalg/memref)을 버리고 `op="gemm"`으로 치환한다. DESIGN 4.1/10.2의 region-carrying(본문 보존) compute로의 전환은 **O1.1**에서 한다.

**완료 조건**: `triton-shared-opt`로 PIM Common IR parse/print(roundtrip) 가능.

**작업 단위**:

- [x] **M1.1** `PIMDialect.td` (name "pim", namespace `::mlir::pim`) — types·ops를 이 한 파일에 통합 (계획의 분리 파일 `PIMTypes.td`/`PIMAttrs.td`/`PIMOps.td`는 만들지 않음)
- [x] **M1.2** types: `RegionType`(region), `TilePlanType`(tile_plan) — 파라미터 없는 opaque 핸들 (계획의 `MemorySpaceType`는 미구현 — space는 StrAttr로 표현)
- [-] **M1.3** enum attrs(`LayoutAttr`/`ResidencyAttr`/`RoleAttr`/`MappingPolicyAttr`/`CapabilityRefAttr`) **미구현** — 현재 전부 StrAttr. faithful-body/capability 도입 시 추가 (O1/O2로 이동)
- [x] **M1.4** ops: 위 5 ops를 `PIMDialect.td`에 정의 (operand/result/attr + declarative `assemblyFormat`; 별도 verifier 없음). `pim.func` 미도입
- [x] **M1.5** `lib/Dialect/PIM/IR/PIMDialect.cpp`만 (op은 tablegen 자동 생성 + declarative assembly라 `PIMOps.cpp`·custom verifier 없음)
- [x] **M1.6** `RegisterTritonSharedDialects.h`에 `mlir::pim::PIMDialect` 등록
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

**목표**: `linalg.matmul`을 `pim.compute @gemm`으로 바꾸는 **단일 변환 pass** (interim coarse 모델). no-split 고정 — `split=1` 하드코딩, placement/execution_group/residency/chunk/reduce **만들지 않음**. (faithful-body 전환 = 본문 보존은 O1에서.)

**완료 조건**: matmul TTSharedIR → `triton-shared-opt --convert-ttshared-to-pim` → `pim.compute @gemm` 포함 IR 생성. (M2 dump로 `02_pim_common.mlir` 눈으로 확인.)

**작업 단위**:

- [x] **M3.1** `convert-ttshared-to-pim` 단일 pass: `linalg.matmul` 인식 → `pim.compute @gemm`
- [x] **M3.2** BLOCK_M/N/K를 읽어 `pim.tile_plan {block_*, split_*}` 생성 (M3 당시 split_*=1 하드코딩 → **O4에서 split_m/n은 compile option으로 승격**; split_k는 여전히 1 고정)
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

> **전제 ① — 전체 문제(full-problem) 정보 복원**: MVP의 M3 변환은 single-tile만 가정해 **grid·program_id·K-loop을 버리고** "타일 하나 = 전체 gemm"으로 취급한다. 실제 multi-tile matmul(grid>1, K-loop)에서 tiling/SPLIT 최적화를 하려면, **전체 M/N/K·grid·program_id→tile 매핑을 복원**해 `tile_plan`(grid·SPLIT)으로 표현하고 **K-loop은 본문 `scf.for`에 보존**해야 한다 (별도 `chunk_plan` 없음 — DESIGN 11.6). 이 복원이 O4(strategy)의 전제이며, **O1 faithful-body 전환 시 함께 수행**한다 (MVP는 BLOCK=전체라 이 정보가 비어 있어도 동작). **현황**: O1.8이 전체 M/N/K를 런타임 인자로 복원해 multi-tile은 이미 동작한다. 남은 건 `program_id` 전역 오프셋을 분배층으로 옮겨 본문을 로컬 슬라이스 기준으로 단순화(O1.3)하고 K-loop을 본문 `scf.for`로 보존(O1.1)하는 부분이다.

> **전제 ② — codegen이 IR 값을 받아 생성 (O4.6에서 해소됨)**: plan/strategy pass는 IR에 intent/구조(`num_units`·split·BLOCK·tasklet 수 등)를 표현한다 (DESIGN 13.0 경계). 초기 translate는 이 값을 읽지 않고 고정 템플릿(`dpu_alloc(1)`·single tasklet)을 emit했으나, **O4.6에서 일반화 완료**: `UPMEMToHost`가 `alloc_units{num_units,split_m,split_n}`를 읽어 `dpu_alloc(NUM_UNITS)` + 2D scatter/gather를, `UPMEMToDevice`가 `local_gemm`의 BLOCK_*를 bake + `NR_TASKLETS` round-robin을 생성한다. launcher는 `metadata.local_lanes`로 `-DNR_TASKLETS`를 넘긴다. **남은 일반화**: split_k(여러 DPU가 한 타일을 K분담) + `host_reduce` (O4.4/O3.5). "IR에 박힘"과 "여러 DPU에서 실제로 돔"은 다른 단계임에 주의.

## O1 — faithful-body 전환 (본문 보존 + build-pim-offload 통합)

**목표**: M3의 coarse `pim.compute @gemm`(본문 치환)을 DESIGN 11.0의 **faithful-body**로 전환한다. 본문(`scf`/`linalg`/`memref`)을 보존하고, 변환은 granular 분해 대신 **단일 `build-pim-offload`**로 통합한다. (이전 계획의 "10개 granular 패스 분해"는 과설계로 폐기 — DESIGN 13.1.)

> **현재 구현과의 격차**: M3/M4/O4는 interim coarse 모델(`pim.compute @gemm` + `local_gemm` 템플릿)로 빌드돼 gemm은 HW 검증까지 통과했다. O1은 이를 faithful-body로 옮겨 **임의 커널(gemv/relu/fused)** 까지 같은 경로로 처리하게 만든다. gemm 결과는 전환 전후 동일해야 한다.

**작업 단위**:

- [x] **O1.1** `[공통·구조]` faithful-body 변환: 본문을 `pim.compute` **region에 보존**(gemm 패턴 인식 후 유닛-로컬 nest로 재구성), `primitive="gemm"`은 속성으로만 기록. dialect `pim.compute`를 **region-carrying**으로 변경(`op`→`primitive`, `full_m/n/k`·result 제거, `[NoTerminator]`). **검증: ops.mlir roundtrip + tiled 커널 변환 출력 확인.**
- [x] **O1.2** `[공통·구조]` 패스 `convert-ttshared-to-pim` → **`build-pim-offload`** 개명 + `execution_group` 합성: split→`mapping_policy`, `num_units=split_m*split_n`, `local_lanes`(option, DESIGN 16), `stationary_data="B"`. `pim.compute`가 `using %eg, %tp`로 참조. **검증: roundtrip + build-pim-offload 출력 + legalize 무영향 + e2e(split 2D) 정답.** (파일 위치는 `Conversion/TTSharedToPIM/` 유지 — DESIGN 6의 `Dialect/PIM/Transforms/` 이동은 조직적 후속. execution_group은 **기록만** — 소비는 O4, `num_units`의 `#pim.capability_ref` 형태는 O2.)
- [x] **O1.3** `[공통·구조]` `program_id` 전역 오프셋을 분배층으로 이동 — region 본문은 유닛-로컬 슬라이스 기준(`tm/tn` 루프 = 복원된 grid, `Mu/Nu/K`는 block-arg `memref.dim` 런타임). **O1.1과 같은 패스에서 함께 수행.**
- [~] **O1.4** `[vendor]` `UPMEMToDevice` 충실 번역. **O1.4a 완료**: legalize가 `pim.compute` 본문을 `device_func` region으로 옮기고(`takeBody`), `UPMEMToDevice`/`UPMEMToHost`가 본문 gemm nest(`linalg.matmul` + 3중 `scf.for`)에서 BLOCK을 읽어 device.c/host.cpp 생성. `upmem.local_gemm` 제거. **검증: translate 출력(BLOCK 32/64/64) + e2e(split 2D) 동치.** **O1.4b 남음**: device.c emission을 gemm 템플릿 → op-by-op 충실 번역으로 일반화(`linalg.matvec`→gemv, `linalg.generic`→relu/fused). 현재는 BLOCK·gemm 구조만 body에서 읽는다.
- [x] **O1.5** `[검증]` 동치: faithful-body 전환 후 e2e 재통과 — `test_pim_matmul_tiled.py` 3 config(`split_m/n=1` no-split 포함) 정답, `pim_common` dump가 faithful-body 확인. device.c는 gemm shortcut(`local_gemm`) 유지라 결과 동일(M6 green).
- [x] **O1.8** `[공통·구조]` 전체 M/N/K 복원: 변환이 tiled-gemm(scf.for+matmul+accumulate) 인식, 전체 M/N/K(`%arg3/4/5`)를 운반(BLOCK은 tile_plan에 기록). codegen(UPMEMToHost·launcher)이 런타임 M/N/K로 동작 → **multi-tile 정답**(128×128×256, grid=(2,2)). faithful-body에선 이 복원이 O1.3(분배층 이동)과 합쳐진다.

> O1.6/O1.7(옛 granular 분해·동치 항목)은 build-pim-offload 통합으로 폐기.

---

## O2 — Capability / Constraints

**목표**: vendor capability/constraint를 도입한다. O3의 gate와 O4의 적법성 필터가 이걸 query한다. (그래서 최적화 pass보다 먼저.)

**작업 단위**:

- [ ] **O2.1** `[공통]` `PIMCapability`, `PIMConstraints` dataclass — vendor-neutral 스키마/인터페이스 (DESIGN.md 19장)
- [ ] **O2.2** `[vendor]` UPMEM `queryCapability`/`queryConstraints` C++ 구현 (실제 DPU 수치)
- [ ] **O2.3** `[공통→vendor]` `#pim.capability_ref<...>` attribute는 공통 표현, **lowering 시 vendor가** capability 값으로 구체화

---

## O3 — Common Optimization Passes (하나씩, 전부 opt-in)

**목표**: DESIGN.md 13.2 최적화 pass를 **하나씩** 추가. 각 pass는 13.0 경계 준수(intent/gate/구조만), toggle 가능, **OFF 시 O1 결과가 그대로 통과**, lit test + e2e 재검증.

**작업 단위** (각 pass별 lit test 포함):

- [ ] **O3.1** `[공통·구조]` `pim-layout-normalize` — 표현 정규화만 (중복 transpose 제거·단일 canonical form). 물리 layout(`pack_by_*`)은 vendor
- [ ] **O3.2** `[공통·intent]` `pim-data-placement` — placement **intent enum**만 갱신. 구체 분산은 vendor
- [ ] **O3.3** `[공통·구조]` `pim-transfer-hoisting` — 반복 transfer를 상위 scope로
- [ ] **O3.4** `[공통·intent]` `pim-persistent-data` — weight/KV를 persistent로 **표시**. 실제 상주는 vendor
- [ ] **O3.5** `[공통·구조]` `pim-partial-result-planning` — partial/reduce **구조**만 (split_k>1일 때). 실제 reduce는 vendor
- [ ] **O3.6** `[공통·gate]` `pim-pipeline-overlap` — `supports_pipeline_overlap`일 때만 `pim.pipeline` 삽입. 실제 async/double-buffer는 vendor
- [ ] **O3.7** `[검증]` end-to-end matmul로 baseline(MVP) 대비 성능 개선 측정

> `pim-chunk-planning`은 제거 — K-loop이 본문 `scf.for`에 보존되므로 chunk는 파생(DESIGN 11.6). WRAM fitting은 vendor device codegen(O1.4/O4.6)이 담당.

---

## O4 — Execution Strategy Selection (plan dispatch)

**목표**: DESIGN.md 13.4 구현. `(SPLIT_M/K/N)`에 따라 no-split / N-split / K-split / M-split 전략을 선택. 선택은 Common IR에서 vendor-aware하게, 실현은 vendor lowering에서. **cost model 없음** — 정적으로는 적법성만, 우열은 autotune 실측 (13.4.5).

**전제**: O2(capability) 완료. 적법성 필터가 `PIMCapability`/`PIMConstraints`를 query.

**작업 단위**:

- [ ] **O4.1** `[공통]` `PIMExecutionStrategy` interface (vendor-neutral): `isApplicable(tile_plan, capability, constraints)` + `applyToIR(builder)` (cost 메서드 없음)
- [ ] **O4.2** `[공통]` 전략 구현 + 레지스트리 등록 (전략 목록은 vendor-neutral): `no-split`, `n-split`(output_n_partitioned), `k-split`(reduction_k_partitioned), `m-split`(output_m_partitioned, 전치 포함)
- [ ] **O4.3** `[공통·gate]` `SelectPIMExecutionStrategy` Common pass — split factor + vendor capability/constraints로 적법 전략 필터, autotune 미사용 시 단순 기본 규칙(no-split 우선)
- [ ] **O4.4** `[vendor]` UPMEM lowering이 전략 attribute legalize (`reduction_k_partitioned` → UPMEM `host_reduce`) + lit test
- [x] **O4.5** `[검증]` e2e: SPLIT에 따라 동작 검증 — `test_pim_matmul_tiled.py`에서 **m-split(`split_m=2`)·n-split(`split_n=2`)·k-split(`split_k=2`)·2D+K(8 DPU) 모두 정답**. DPU 2→4→8에서 시간 ≈2×씩 감소(선형 스케일). k-split은 host tree-reduce(OpenMP) 동작 확인.
- [x] **O4.6** `[vendor]` translate·launcher가 IR 값을 받아 생성(전제 ②). **완료**: (a) n-split host codegen(`alloc_units{num_units}`→`dpu_alloc(U)`, A broadcast/B 열scatter/C gather, split_n=2 검증). (b) **device BLOCK 타일링** — `UPMEMToDevice`가 `local_gemm`의 BLOCK_M/N/K를 bake, 부분행렬을 `BLOCK_M×BLOCK_N` 타일로 순회하며 `BLOCK_K`를 WRAM(aW/bW/acc)에 올려 누적·재사용. **①a 단일 tasklet** + **①b cooperative 다중 tasklet**(WRAM 타일 공유 + BLOCK_M 행 tasklet 분할 + barrier) 모두 HW에서 검증 (split_n=2 × local_lanes=8/16). (c) **split_k host_reduce 완료**: `alloc_units{...,split_k}` → `UPMEMToHost`가 idx→(pm,pn,kj) 3D 분배(A strided staging·B Ku-슬라이스 전치·K=Ku broadcast) + 그룹별 host **tree-reduce(OpenMP)**, device는 K=Ku만 받아 부분곱(불변). 8 DPU(2D+K) HW 정답·선형 스케일. (d) **host staging 중복 제거**: A/B 전치·복사를 per-DPU(NUM_UNITS) → distinct(A: split_m·split_k, B: split_n·split_k)만 1회 빌드하고 idx→버퍼 공유 (같은 N열/M행 DPU가 동일 B/A 공유) → 2048-DPU big gemm **8029→4165ms**. (e) **dtype-aware codegen**: IR matmul에서 입력(Tin)·누적(Tacc) element type을 읽어 host/device를 `typedef Tin/Tacc`로 생성(f32/f64/i8/i16/i32/i64). **int8 입력+int32 누적 HW 검증: 2048-DPU에서 f32 4535→int8 772ms (~5.9×), 정수라 정답 정확** — fp32 소프트 에뮬이 주병목이었음을 확인. **제약**: WRAM ≤ 64KB(BLOCK/local_lanes 균형, autotune); int8은 MRAM 8-byte 정렬 위해 BLOCK_K%8==0 + K/N 짝수 필요(PGEMM식 shift 보정 미구현); persistent set은 시도했으나 config sweep 시 DPU 누적 고갈로 **매 호출 alloc/load/free로 환원**.
- [x] **O4.7** `[공통 knob + vendor]` multi-tasklet (intra-DPU 병렬, `local_lanes`=`NR_TASKLETS`): `PIMOptions.local_lanes`(공통 knob, default 1) → `metadata` → launcher가 `-DNR_TASKLETS=L`로 device.c 컴파일. `UPMEMToDevice`가 `NR_TASKLETS` 매크로 기반 device.c 생성 — `BARRIER_INIT` + `me()==0`이 `mem_reset` 후 barrier, tasklet별 `mem_alloc` WRAM, `for(i=me();i<M;i+=NR_TASKLETS)` 행 round-robin(행 disjoint→끝 barrier 불필요). **검증: `split_n=2, local_lanes=16` (2 DPU × 16 tasklet) 정답 통과.** (DESIGN 14.6)

> **수직 슬라이스 (완료)**: 전략 프레임워크(O4.1~O4.4)를 건너뛰고 옵션으로 끝까지 관통 —
> - `split_m`/`split_n` (inter-DPU 2D 분배): `tile_plan.split_m/n` → `alloc_units{num_units,split_m,split_n}` → `UPMEMToHost` 2D scatter/gather (A 행밴드·B 열밴드·C 2D). **HW 선형 스케일링 검증**(1→4 DPU ~3.8×).
> - `BLOCK_M/N/K` (intra-DPU 타일): `UPMEMToDevice`가 BLOCK 타일 순회 + `BLOCK_K` WRAM 청크 재사용 (cooperative 다중 tasklet).
> - `local_lanes` (intra-DPU 병렬): `metadata`→launcher `-DNR_TASKLETS`→ tasklet이 BLOCK_M 행 분할.
>
> **다음 후보**: ✅ ③ SPLIT_K + `host_reduce` **완료**. ✅ host staging 중복 제거 + ✅ dtype-aware codegen(int8) **완료**(int8 f32 대비 ~5.9×). ④ persistent DPU set은 sweep DPU 누적으로 **보류**(매 호출 free로 환원). **⑦ device 커널 효율: 원인은 dpu 컴파일 `-O0` → `-O2` 추가(driver.py)로 해결, kernel 101→? 검증 대기**(PGEMM식 구조 재설계는 재사용 손실로 환원, 옛 cooperative-tile 유지). 현실적 남은 레버: **④ persistent(wall-clock alloc 300ms 제거)**, ⑧ split_k 작게(D2H/pack 폭증 회피). 이후 O1.4b(비-gemm), O4.1~O4.4 전략 프레임워크.

---

## O5 — Autotune Integration

**목표**: `@triton.autotune`이 O3/O4의 후보들을 실측으로 선택하도록 연결. PIM 전용 노브(SPLIT_*/NUM_UNITS/LOCAL_LANES)는 별도 annotation pass 없이 plan pass가 option에서 읽음 (DESIGN.md 17.0).

**작업 단위**:

- [ ] **O5.1** `[공통]` PIM 전용 노브를 compile option으로 받아 `build-pim-offload`가 `tile_plan`(SPLIT_*)·`execution_group`(NUM_UNITS·LOCAL_LANES) attr로 반영 (별도 annotation pass 없음, DESIGN 17.0). split의 런타임 default vs autotune config 우선순위는 DESIGN 26.4(config 우선)
- [ ] **O5.2** `[검증]` autotune config별 독립 IR/artifact 생성·실측이 동작하는지 검증 (config N개 → N벌 컴파일)
- [ ] **O5.3** `[공통+vendor]` O4 전략 후보를 autotune이 실측 선택하는 통합 — 후보 선택은 공통, 각 후보의 실측 대상 artifact는 vendor codegen(O4.6) (vendor별 best가 달라지는지 확인)

---

# Part III — Breadth

## B1 — AiMX Vendor Extension

**목표**: 두 번째 vendor를 추가해 vendor-neutral 추상화가 실제로 작동함을 검증.

**작업 단위**:

- [x] **B1.0** `[seam]` vendor-dispatch 경로: `compiler.py` `add_stages`가 `options.vendor`로 분기(`vendor_ir`/`pimobj`) + `_pim_common_to_aimx`/`_aimx_to_artifact` 평행 helper(배관 `_run_opt`/`_run_translate`는 공유). `driver.py` PIMLauncher가 `metadata.vendor`로 빌드 분기 + `_build_aimx_so` seam(`NotImplementedError`). dlopen→ctypes 호출은 같은 C ABI 전제로 공유. **실제 AiMX codegen/toolchain은 미구현(fail-fast, fallback 없음)** — 아래 B1.1~B1.6에서 채운다.
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
  ├─ O1 faithful-body 전환 ──┐
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

**⓪ DESIGN faithful-body 전환 (O1) — 구조 우선순위**: **IR 변환·구조 합성 + device가 본문 구동까지 완료**(O1.1/O1.2/O1.3/O1.4a/O1.5). `build-pim-offload`가 본문을 `pim.compute` region에 보존하고(program_id는 `tm/tn` 복원된 grid로 흡수), legalize가 그 본문을 `device_func`로 옮기며, `UPMEMToDevice`가 본문 gemm nest에서 BLOCK을 읽어 device.c를 만든다(`local_gemm` 제거). **O1 남은 일: O1.4b** — device.c emission을 gemm 템플릿 → op-by-op 충실 번역으로 일반화하면 gemv/relu/fused가 같은 경로로 열린다(현재는 gemm 본문만). `execution_group`은 여전히 기록만(O4 소비). 성능 작업 ①~④와 독립.

**실행 모델 확정 (DESIGN 11.5.2): SPLIT(굵게, 유닛 간) → BLOCK(잘게, 유닛 내부 타일) → BLOCK_K(WRAM 청크, A/B 재사용).** "1 BLOCK = 1 DPU"는 폐기 — BLOCK은 DPU 분배 단위가 아니라 DPU 내부 타일이다. PGEMM `basic_n`과 분배(2D 출력분배)는 동일하나, **BLOCK 타일링으로 WRAM 재사용**하는 게 차이·이점.

① **device BLOCK 타일 루프** (intra-DPU, O4.6) — **완료**: ①a 단일 + ①b cooperative 다중 tasklet (BLOCK_M/N/K 타일 + BLOCK_K WRAM 재사용 + tasklet 행분할, HW 검증). 환경: 시뮬레이터→HW(`UPMEM_BACKEND=hw`+upmem-override lib).
② **2D SPLIT (`split_m` 추가)** — **완료**: `SPLIT_M×SPLIT_N` DPU에 출력 2D 분배, **A 행밴드 scatter(전체 복제 해소)** + B 열밴드 + C 2D gather. `tile_plan.split_m` → `alloc_units{num_units,split_m,split_n}` → `UPMEMToHost` 2D. **HW 검증 + 선형 스케일링**(1→2→4 DPU = 3833→1949→1008 ms, ~3.8×). split_n은 SPLIT_M=1 특수형.
③ **SPLIT_K + reduce** — **완료**: 한 출력을 여러 DPU가 K분담 → `UPMEMToHost`가 idx→(pm,pn,kj) 3D 분배(A strided staging·B Ku-슬라이스 전치·K=Ku broadcast) + 그룹별 host **tree-reduce(OpenMP, `-fopenmp`)**. device는 K=Ku만 받아 부분곱(불변). 공통은 `tile_plan.split_k`+`mapping_policy=reduction_k_partitioned` intent만(명시 `pim.reduce` op은 O3.5로 보류). HW: 8 DPU(2D+K) 정답, 4→8 DPU ≈2× 가속.
④ **persistent DPU set** — **구현 + HW 검증(2026-06-09)**. 분해 측정으로 병목 확정: **`dpu_alloc`(2048 예약) ≈ 280ms ≫ `dpu_load`(elf) ≈ 25ms** (total\* 124, free 7). 즉 wall(~447ms)의 ~63%가 alloc. PGEMM은 `main`서 alloc 1회(측정 제외)라 0 → **persistent alloc이 유일한 큰 레버**. **구현 = 공유 런타임 `pim_runtime.{h,c}`**(config 무관 1회 빌드 → `libpim_runtime.so`, 모든 host.so가 full-path 링크+rpath → globals 1개 공유): `pim_acquire(num_units, elf)`. host.cpp는 alloc/load/free 대신 이걸로 set 받아 transfer/launch/gather만(`dpu_free` 없음, runtime이 atexit free). **set은 1개만 캐시(단일 set)**: 같은 num_units이면 재사용(acquire ≈ 0), elf 다르면 reload(25ms), num_units 다르면 옛 set free 후 새로 alloc. (set이 보관하는 건 예약+elf뿐이고 데이터 A/B/C는 매 호출 재전송 → 1개면 충분.) **검증: 같은 config 반복 시 acquire 110/170→0.00/0.00/0.00 (재사용 확인), sweep도 고갈 없음.** (초기 num_units별 multi-set 캐시는 서로 다른 카운트(512+1024+2048)가 누적돼 고갈 → 단일 set으로 변경.) (`pim_backend/backend/pim_runtime.*` + `UPMEMToHost.cpp` + `driver.py`.) **후속(미구현, DESIGN 18 참고)**: 작은 config 공존 시 multi-set+LRU, weight-stationary(weight MRAM 상주 → set이 데이터 보유, 재전송 제거).
⑤ **host staging 중복 제거** — **완료**: A/B 전치·복사를 per-DPU → distinct(A: split_m·split_k, B: split_n·split_k)만 1회 빌드 + idx→버퍼 공유 (2048-DPU 8029→4165ms). transfer 자체의 복제(B를 M열밴드 DPU에 32배 전송)는 미해결 — sub-set broadcast가 후속 과제.
⑥ **dtype-aware codegen** — **완료**: `build-pim-offload`가 입력(Tin)·누적(Tacc) type 분리 보존, `UPMEMToHost/Device`가 IR에서 읽어 `typedef`로 생성(f32/i8/i16/i32/i64). **int8+int32 HW 검증: f32 대비 ~5.9×(4535→772ms)** — fp32 소프트 에뮬이 주병목 확인. 제약: int8 MRAM 8B 정렬(BLOCK_K%8, K/N 짝수; shift 보정 미구현).

⑦ **device 커널 효율 (intra-DPU)** — **원인 규명: dpu 컴파일이 `-O0`였음 → `-O2` 추가(2026-06-09, HW 검증 대기)**. 경과: ⓐ PGEMM식 disjoint-tasklet 재설계는 **bW 재사용 손실(B를 행마다 재로드, mram_read 32×↑)로 더 느려져 환원**(교훈: 데이터 WRAM 상주 시 "DMA 은닉"보다 "재사용" 우선). ⓑ PGEMM Makefile과 **같은 분배(split_n=128/split_k=1, per-DPU [64,32]×K1024)로 재측정해도 kernel 101ms 동일** → 작업 shape 탓 아님. ⓒ **진짜 원인 = 컴파일 최적화**: `driver.py`의 `dpu-upmem-dpurte-clang`에 `-O` 플래그가 없어 **clang 기본 `-O0`**(MAC 루프 언옵티마이즈드)로 빌드됨. PGEMM은 `-O2`(libPGEMM.sh). **수정**: `driver.py`에 `dpu_opt="-O2"` 추가 + **캐시 키에 포함**(옛 -O0 elf 재사용 방지). 곱셈·구조 동일(PGEMM gemv도 `(T2)a*(T2)b`)인데 101 vs 54였던 ~1.9×가 이걸로 설명됨. int8 vs int8 확정(PGEMM `gemm_dpu_b`도 -DPREC 없이 빌드 → PRECISION 4 기본). **HW 검증 완료(2026-06-09): kernel 101→42.9ms — PGEMM 54ms보다 빠름. ⑦ 해결.** 옛 cooperative-tile 커널 유지, `-O2`만 추가. (`pim_backend/backend/driver.py`.)
⑧ **D2H / split_k 팽창** — D2H가 split_k 부분합을 전부 host로 gather(D2H ∝ split_k): config2(k=4) 19ms, config1(k=32) **95ms** vs PGEMM 6ms. split_k를 작게 유지하거나 부분합 축약(rank-level/on-device 결합) 개선 필요. (보조 레버.)

> **PGEMM 공정비교 (2026-06-09, int8 1024×1024×4096 2048 DPU)**: `-O2` 적용 후 PGEMM 분배 매칭(split_m=16/split_n=128/split_k=1, BLOCK 32×32×256)에서 **total\* 118ms vs PGEMM 114ms(동등)**. 분해 — H2D 34 vs 29, **kernel 42.9 vs 54(내가 빠름)**, D2H 8 vs 6, pack 34 vs ~19~47(동등권). **kernel 격차(옛 101 vs 54)의 정체는 dpu 컴파일 `-O0` → `-O2`로 해결(⑦).** **남은 진짜 병목 = alloc/load 304ms**(total\* 전부의 2.6배, wall-clock의 ~70%). PGEMM은 main서 1회만 하므로 측정 제외 → **④ persistent가 유일한 큰 레버**(반복 호출 wall ~429→~125ms 예상). 보조: split_k 작게(⑧). [PROF] 계측은 `UPMEMToHost.cpp`의 `PIM_PROFILE` 마커(임시).

계층: SPLIT/BLOCK/LOCAL_LANES **knob·intent는 공통**(option/`tile_plan`/metadata), **실현(dpu_alloc·DPU_FOREACH·NR_TASKLETS·me()·BLOCK_K WRAM·dtype)은 UPMEM(`lib/Target/UPMEM`)**.
