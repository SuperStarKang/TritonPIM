# PIM target 추가 방법

새 target은 PIM Common IR의 의미를 다시 정의하지 않고, 동일한
Block–Group–Execution Tile과 device mapping을 target IR로 옮긴다. 현재 target은
`upmem`, `aimx` 두 개이며 아래 각 단계의 target 이름 분기(dict 또는 `if`)에 새 이름을
추가한다. Common 계약은 [PIM Common IR 설계](PIMCommonIR_Design.md)를 따른다.

## 1. 공유 target spec

`pim_backend/backend/target.py`에 runtime config와 builder를 추가한다.

```text
PIMTargetSpec
  common_profile: PIMTargetProfile      (#pim.target_profile과 같은 일곱 필드)
    target
    num_units
    banks_per_unit
    cores_per_bank
    lanes_per_core
    bank_capacity_limit_bytes
    local_capacity_limit_bytes
  runtime_config: <Target>RuntimeConfig (PIMTargetRuntimeConfig 상속)

<target>_target_spec(**options) -> PIMTargetSpec   # 0으로 받은 profile 값을 target 기본값으로 채운다
make_target_spec(target, **options)                # builders dict에 등록
```

Common profile 필드는 추가하지 않는다. 새 target만 요구하는 값은
`<Target>RuntimeConfig`에 둔다(UPMEM `nr_tasklets`/`host_threads`, AiMX
`batch`/`base_fpga`/`matmul_k_alignment`). compiler(`compiler.py`의 `_TARGET_SPEC`)와
autotuner(`autotuner/targets.py`의 `target_spec` → `make_target_spec`)가 같은 builder를
쓰므로 profile 값을 두 군데에 복사하지 않는다.

`local_capacity_limit_bytes=0`은 compiler가 배치할 local memory가 없다는 뜻이다(AiMX).
`bank_capacity_limit_bytes`는 bank 하나의 byte 상한이며, unit 전체 상한은
`banks_per_unit * bank_capacity_limit_bytes`로 파생한다(`TargetModel.unit_capacity`,
adapter `resources()`).

## 2. Compiler pipeline 연결

`pim_backend/backend/compiler.py`에서 다음을 추가한다.

| 위치 | 추가할 것 |
|---|---|
| `_<target>_spec(o)`, `_TARGET_SPEC` | `PIMOptions` → `PIMTargetSpec` 변환 |
| `PIMOptions` | target runtime option 필드, `__post_init__`/`add_stages`의 지원 target 메시지 |
| `_ttsharedir_to_pim_common` | target별 Common pass 목록 조정이 필요할 때만 (AiMX는 `--pim-reuse-execution-tile-inputs` 제거, UPMEM은 끝에 `-canonicalize` 추가) |
| `add_stages` | `stages["target_ir"]`, `stages["pimobj"]` 분기 |

```text
ttir → ttsharedir → pim_common      (공유, 수정하지 않음)
pim_common → target_ir              triton-shared-opt --convert-pim-to-<target>=...
target_ir  → pimobj                 triton-shared-translate --<target>-to-* → JSON 번들
```

`pimobj` 단계는 생성 소스에서 `extern "C" void <name>(`를 찾아 `metadata["name"]`을
채우고, launcher가 빌드에 쓸 소스와 값을 JSON 번들로 돌려준다(`_upmem_to_artifact`,
`_aimx_to_artifact`).

새 target lowering은 Common IR의 다음 값만 읽는다.

- `#pim.target_profile`
- `#pim.block`, `#pim.block_group`, `#pim.execution_tile`
- `#pim.device_mapping<partitions=[p_m,p_n,p_k]>`
- `pim.access/load/store/combine/place`
- `pim.partition`의 `pim.broadcast/scatter → pim.launch → pim.gather` 계획
- loop 표식(`pim.scheduling_axis`, `pim.temporal_reduction`, `pim.device_origin` 등)
- linalg compute

Common pipeline 마지막의 `--pim-form-host-plan`이 `pim.device`를 `pim.partition`으로
바꾼다. 최종 IR에 `pim.device`가 남아 있으면 target pass가 진단한다. 전송 action의 access
ID는 같은 partition의 `pim.place`를 가리킨다. `broadcast`는 `shared_over` 축으로 같은
입력을 받는 그룹의 배포이며 특정 SDK 호출을 강제하지 않는다.

Target의 물리 padding, instruction/descriptor 생성, K partial(`pim.combine`) 결합
방식, 실제 lane 활성화 방식은 target lowering에서 결정한다.

## 3. IR, pass, translation

```text
include/triton-shared/Dialect/PIMTarget/<Target>/IR/<Target>Dialect.td
lib/Dialect/PIMTarget/<Target>/IR/<Target>Dialect.cpp
include/triton-shared/Conversion/PIMTo<Target>/Passes.td
lib/Conversion/PIMTo<Target>/PIMTo<Target>Pass.cpp
include/triton-shared/Target/<Target>/<Target>Translations.h
lib/Target/<Target>/<Target>To*.cpp
```

등록은 두 도구에 한다.

| 도구 | 파일 | 등록 내용 |
|---|---|---|
| `triton-shared-opt` | `tools/triton-shared-opt/RegisterTritonSharedDialects.h` | dialect, `registerConvertPIMTo<Target>Pass()` |
| `triton-shared-translate` | `tools/triton-shared-translate/triton-shared-translate.cpp` | `register<Target>To*Translation()` |

Target IR에는 Common IR에 이미 있는 hierarchy나 mapping을 다른 이름으로 복제하지
않는다. Target operation은 codegen에 필요한 물리 정보만 추가한다.

## 4. Launcher

`pim_backend/backend/driver.py`의 `PIMLauncher._launch`가 `pim_target`으로 분기해
번들을 공유 라이브러리로 빌드한다(`_build_upmem_so`, `_build_aimx_so`). 생성된 진입
함수의 C ABI는 커널의 비-constexpr 인자(pointer는 `void*`, scalar는 `int32`) 뒤에
grid 크기 3개다. 단계별 시간은 생성 코드의 `pim_timing_ms` 전역에서 읽으며 그 이름을
`_STAGE_NAMES[<target>]`에 추가한다.

## 5. Autotuner 연결

공통 `PIMConfig`(`autotuner/types.py`)는 target을 추가해도 바뀌지 않는다.

```text
block_m/n/k  group_m/n/k  tile_m/n/k  p_m/n/k
```

Target 물리 규칙은 `autotuner/model.py`에 `TargetModel` 하위 클래스로 두고
`target_model()`의 dict에 등록한다. `ConstraintEngine`(`constraints.py`), search space,
feature, model compile 경로는 이 메서드만 부른다. 검사 결과나 target policy를
`PIMConfig` 필드로 추가하지 않는다.

| 멤버 | 기본 동작 | 재정의 이유 (현재 target) |
|---|---|---|
| `launch_timing_name` | 필수 | 동기 device 실행 시간 이름 (UPMEM `kernel`, AiMX `invoke`) |
| `gemm_profile_spec`, `gemv_profile_spec` | 필수 | profiling 시작 Block/Group과 dtype (`PIMGemmProfileSpec`, `PIMGemvProfileSpec`) |
| `stage_target(problem)` | spec 그대로 | GEMM 하나를 tuning·compile할 spec (AiMX: GEMM batch를 M에 맞춤) |
| `block(problem)` | `TRITON_BLOCK` | Triton Block (AiMX matmul: Block m ≤ batch) |
| `problem_reason(problem)` | None | 지원 연산·dtype 제한 (AiMX: matmul/matvec, BF16/F16/F32) |
| `group_reason(problem, extent)` | None | Block Group 하나의 계산 가능 여부 (UPMEM: tasklet 분할, AiMX: 연산 단위) |
| `mapping_reason(problem, partitions)` | None | `[p_m,p_n,p_k]` lowering 불가 사유. model compile은 `(1,1,2)` 결과로 split-K seed 여부를 정한다 |
| `allocation_reason(problem, config)` | None | unit 용량 외 할당 제한 (AiMX: 단일 할당 상한, bank footprint) |
| `resident_operand_reason(problem, config)` | None | 상주 B(KV cache)가 extent보다 큰 padding으로 저장되는 사유 |
| `resident_operand_config(problem)` | None | 상주 B에 추가 padding이 없는 1 device·1 round config |
| `access_allocations`, `overhead_allocations` | 논리 footprint, `{}` | Execution Tile 하나의 access별·access 외 상주 byte |
| `scheduled_compute(problem, config)` | Tile 원소 곱 | padding 포함 계산점 |
| `minimum_compute(problem)` | 문제 원소 곱 | target 계산 폭이 만드는 최소 padding 포함 계산점 |
| `physical_pressure(plans)` | unit 용량 대비 최대 합 | lowering이 보고한 할당의 메모리 압력 (AiMX: bank 기준) |

GEMV profiling 경로 `autotuner/gemv.py`의 `run_gemv`는 target별 입력·출력 dtype을
분기하므로 새 target 분기를 추가한다.

## 6. Backend option

공통 option은 다음으로 고정한다(`PIMOptions`).

```text
pim_target
num_units  banks_per_unit  cores_per_bank  lanes_per_core
bank_capacity_limit_bytes  local_capacity_limit_bytes
group_m/n/k  tile_m/n/k  p_m/n/k
```

Target runtime option만 `PIMOptions`와 `<Target>RuntimeConfig`에 추가한다. UPMEM은
`nr_tasklets`, `host_threads`, AiMX는 `batch`, `base_fpga`, `aimx_persistent`,
`aimx_matmul_row_alignment`, `aimx_matmul_useful_n_only`, `aimx_matmul_k_alignment`가
있다. 옵션은 `parse_options`가 받은 것만 쓰며 환경 변수 통로는 없다.

## 7. Model runtime adapter

`PIMSession` 모델 경로를 쓰려면 `pim_backend/backend/runtime.py`의
`create_runtime(spec)`에 adapter를 등록한다. 등록이 없으면 `NotImplementedError`다.
Adapter는 `RuntimeBackend` protocol을 구현한다.

| 메서드 | 시점 | 역할 |
|---|---|---|
| `begin_model()` / `end_model()` | compile 전 / release | model runtime(`adapter.model`) 생성·해제, unit은 유지 |
| `resources()` | compile time | `UnitResources(unit 수, unit 용량)` |
| `requirements(plans)` | compile time | phase 하나의 operand별 byte (공통 memory table 입력) |
| `prepare_model(plans, memory)` | compile time | memory table offset에 모든 phase의 layer 배치, native artifact 연결 |
| `open()`, `num_units` | session 진입 | unit 할당 |
| `bind_model(plans)` | session 진입 | slot별 device handle 생성 |
| `weight_table(plans)`, `memory_table(plans)` | 진입 후 | weight 배치, device 주소가 붙은 table 행 |
| `stats()`, `close()` | 실행 후 | 누적 전송 byte·launch 수, 자원 해제 |

`adapter.model`(현재 `UPMEMModelRuntime`, `AiMXModelRuntime`)은 `PIMSession`과
torch backend가 직접 부르는 `compile_invocation(stage, physical_inputs)`,
`enter_layer`/`exit_layer`, `run(stage, matrices, output)`,
`write_kv(stage, cache, start, stop)`를 구현한다. `compile_invocation`이 돌려주는
객체는 `current_invocation`에 걸려 launcher 대신 `launch(launcher, bundle, entry, grid,
arguments)`로 stage를 만든다. 세부는 [ModelRuntime 설계](ModelRuntime_Design.md)를
따른다.

## 8. 확인 항목

사용자가 빌드한 뒤 다음 경계를 각각 확인한다.

1. target profile이 일곱 필드로 PIM Common IR에 기록되는가
2. Block–Group–Execution Tile attribute가 target lowering까지 보존되는가
3. `[p_m,p_n,p_k]` mapping과 scheduling round가 target IR에 반영되는가
4. `pim.place.allocation_shape`와 `shared_over`를 target transfer가 사용하는가
5. target 물리 capacity/padding 오류가 target lowering에서 명확히 진단되는가
6. generated artifact가 target runtime config를 그대로 사용하는가
7. `TargetModel`의 할당·계산점이 lowering이 보고한 값과 맞는가

## 근거/경로

| 항목 | 경로 |
|---|---|
| target spec, builder | `pim_backend/backend/target.py` |
| compiler stage, `PIMOptions` | `pim_backend/backend/compiler.py` |
| launcher, timing 이름 | `pim_backend/backend/driver.py` |
| autotuner target model | `pim_backend/autotuner/model.py`, `constraints.py`, `targets.py` |
| GEMV profiling 분기 | `pim_backend/autotuner/gemv.py` |
| model runtime adapter | `pim_backend/backend/runtime.py`, `upmem_adapter.py`, `aimx_adapter.py` |
| pass·translation 등록 | `tools/triton-shared-opt/RegisterTritonSharedDialects.h`, `tools/triton-shared-translate/triton-shared-translate.cpp` |
