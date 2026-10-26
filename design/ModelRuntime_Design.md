# 모델 단위 PIMSession과 LayerPlan 실행 설계

## 역할과 수명

**내용**
PIMSession은 adapter(장치), 요청 단계(phase)별 LayerPlan, 메모리 table을 소유한다.
LayerPlan은 Torch block 하나를 CPU partition과 PIM stage로 나눈 실행 계획이다.
장치는 Session 진입(`__enter__`)에서 한 번 확보하고 `close()`에서 한 번 해제한다.

**근거**
- [runtime.py](../pim_backend/backend/runtime.py) `PIMSession`: `plans`(phase → LayerPlan tuple), `memory`(MemoryTable), `weight_table`, `kv_table`, `loaded_layer`.
- [model.py](../pim_backend/torch_backend/model.py) `LayerPlan`: `index`, `phase`, `graph`(분할된 FX), `stages`(`_LayerGemm`), `exported_graph`.
- Session은 중첩할 수 없고 연 thread와 context에서만 쓸 수 있다(`_check_owner`).

**예시**

| PIMSession | LayerPlan |
|---|---|
| adapter와 model runtime (`backend`, `backend.model`) | 분할된 FX graph (CPU partition + PIM stage) |
| `plans[phase]`: phase별 layer 실행 순서 | stage별 config, `p_b`, batch round, M/N/K round |
| `memory`: 모든 phase와 KV 영역의 offset table | `phase`, `index`, `backend_plan`(slot, group, bytes) |
| `weight_table`: `(phase, layer, stage, operand)` → weight 배치 | weight operand(frozen tensor, view, `ConcatenatedWeight`) |
| `run_model(phase, ...)`, `write_kv(...)` | `run_model(*inputs)`: 분할 graph 실행 |

## 준비 과정

**내용**
`compile(phases, kv_inputs=)`는 장치를 확보하기 전에 모든 phase를 준비하고, Session 진입이 장치 확보와 bind를 한다.
`prepare(phases, kv_inputs=)`는 이미 열린 Session 안에서 준비와 bind를 함께 한다.
두 경로 모두 `_build`에서 export, config, native 컴파일, 메모리 배치를 하며 장치에 접근하지 않는다.

**근거**
- `phases`는 실행 순서의 `{phase: (layers, example_inputs, configure)}`이다. `configure(gemms, spec)`는 원래 FX GEMM 이름 → `PIMTorchConfig`를 돌려준다.
- `kv_inputs[phase](index, inputs)`는 그 layer 입력 중 host key/value cache를 `{KEY/VALUE: tensor}`로 알려준다.
- `__enter__`는 compile된 Session이면 실제 unit 수가 spec과 다를 때 `ValueError`를 낸다.

**예시**

```text
PIMSession.compile(phases)                       (장치 없음)
  _build
    adapter.begin_model                          model runtime 생성
    phase마다, layer마다
      torch.export → analyze_fx_gemms → configure
      build_layer_plan(kv=...)                   freeze, packing, stage 묶음, split_module
      plan.plan_model                            stage마다 select_layer_config + PIM native 컴파일
      plan.compile_cpu                           CPU partition Inductor 컴파일, 출력 → 다음 layer 예제 입력
    plan_memory(resources, {phase: requirements}) 메모리 table (TorchIntegration §3)
    adapter.prepare_model(plans, memory)         table offset 배치 (UPMEM: program link)
    │
with session:                                    open → unit 수 확인 → bind_model → weight_table
    session.write_kv(...)                        host cache 위치 → 장치 cache 영역
    session.run_model(phase, *inputs)
    │
close                                            model runtime, 장치 해제
```

`prepare()` 경로는 `release()`로 model runtime만 해제하고 장치를 유지한 채 다시 `prepare()`할 수 있다
(stage 측정이 이 경로를 쓴다, [model_compile.py](../pim_backend/torch_backend/model_compile.py) `measure_stage`).

## LayerPlan: CPU partition과 PIM stage

**내용**
`build_layer_plan`은 export graph를 정규화하고 weight를 상수로 freeze한 뒤 GEMM packing을 적용한다.
weight GEMM과 `bmm`은 각자 stage 하나이고, 같은 depth·op·shape·dtype·config·cache 읽기 방식의 독립 activation GEMM은 instance로 묶어 stage 하나가 된다.
stage 앞뒤에서 graph를 잘라(`split_module`) CPU partition과 PIM stage를 번갈아 둔다.

**근거**
- `_freeze_graph`: parameter, buffer와 weight만 읽는 view/copy(`t`, `transpose`, `permute`, `view`, `reshape`, `contiguous`, `clone`, `detach`, `_to_copy`)를 graph 상수로 바꾼다.
- `CompileModelCPU`: PIM stage가 없는 partition은 Inductor(`backend="inductor"`)로 컴파일하고, stage는 CPU reference로 계산해 다음 layer의 예제 입력을 만든다.
- weight를 가진 `bmm`은 `_LayerGemm.prepare`가 거부한다("resident batched weights require expert placement support").

**예시**

| stage 종류 | 묶음 | operand 전달 |
|---|---|---|
| weight GEMM | 단독, instance 1 | weight는 stage가 가짐, 실행 인자는 placeholder |
| activation GEMM | 같은 key의 독립 GEMM → instance | 매 실행 연속 사본 전달 |
| cache GEMM | 같은 cache 종류·operand·전치 여부끼리만 | host cache 자체 전달 (pointer만 쓰임, 장치 영역을 읽음) |
| `bmm` | batch 차원이 instance | instance별 연속 사본 |

## Execution Tile과 batch round

**내용**
같은 프로그램이 Execution Tile을 처리하고, instance(독립 GEMM)는 unit 집합을 나눠 동시에 실행한다.
`p_b`는 instance 병렬성이며 Common IR 축이 아니다.

**근거**
- [execution.py](../pim_backend/torch_backend/execution.py) `select_layer_config`: `P = p_m·p_n·p_k`, `p_b = min(instances, ⌊units / P⌋)`, batch round = `⌈instances / p_b⌉`.
- weight stage는 instance 1개와 M/N/K round 1이어야 한다. cache stage는 M/N/K round 1과 batch round 1이어야 한다.
- UPMEM은 마지막 batch에서 배정이 없는 DPU를 비활성으로 표시하고 전송에서 뺀다([upmem_model.c](../pim_backend/backend/upmem_model.c) `model_active`, `_pim_model_active`).

**예시**

32개 instance, Execution Tile 하나에 partition `(8,8,1)`(P = 64):

| units | p_b | batch round |
|---:|---:|---:|
| 2048 | 32 | 1 |
| 512 | 8 | 4 |

## 모델 실행

**내용**
`run_model(phase, *inputs, after_layer=None)`은 그 phase의 layer를 순서대로 실행하고 tuple 출력을 다음 layer의 positional 입력으로 넘긴다.
layer마다 `enter_layer` → `plan.run_model` → `exit_layer` 순서이며, `enter_layer`는 `(phase, group)`이 바뀔 때만 그 group의 weight를 올린다.
group 업로드 순서와 background pack은 target 중립인 `GroupLoader`가 맡는다.

**근거**
- `group_order(memory)`: 한 요청이 들어가는 `(phase, group)` 순서.
- `GroupLoader.enter`: 이미 올라간 key면 아무것도 하지 않는다. 아니면 pack 결과(background 결과 또는 즉시 pack) → push → host 사본 해제 → 다음 key의 pack을 thread로 시작한다.
- pack은 host 메모리만 쓰므로 앞 group 실행과 겹친다. push(장치 전송)는 같은 장치의 PIM 연산과 겹칠 수 없다.

**예시**

```text
order = [(prefill, 0), (prefill, 1), ..., (decode, 0), ...]

enter(key)
  key == loaded            → 업로드 없음
  else                     → pack(대개 background에서 끝남) → push → release
                             다음 key pack 시작 (thread "pim-pack-<phase>-group<g>")
마지막 phase의 마지막 group 다음 → 그 phase의 첫 group (decode step 반복)
```

| 경우 | 업로드 |
|---|---|
| phase 하나, groups = 1 | 첫 요청에서 한 번, 이후 없음 |
| groups > 1 | phase를 실행할 때마다 모든 group 재업로드 |
| 여러 phase | phase가 바뀔 때 다음 phase의 첫 group이 앞 phase 영역을 덮어씀 |

`exit_layer`: AiMX는 아무것도 해제하지 않는다. UPMEM은 그 layer stage의 host pool을 놓는다
([aimx_model.py](../pim_backend/backend/aimx_model.py), [upmem_model.py](../pim_backend/backend/upmem_model.py)).

## 메모리 table

**내용**
무엇을 어디에 둘지는 target 중립인 `plan_memory`([residency.py](../pim_backend/backend/residency.py))가 정한다.
adapter는 unit 수, unit당 용량, stage operand별 unit당 bytes와 종류(weight, state, input, output, key, value)만 보고한다.
상세 규칙은 [TorchIntegration.md](TorchIntegration.md) §3이다.

**근거**
- `RuntimeBackend.resources()`: `UnitResources(units, banks_per_unit × bank_capacity_limit_bytes)`.
- `RuntimeBackend.requirements(plans)`: `StageRequirement(layer, stage, operands)`.
- 모든 unit이 같은 배치이므로 offset table 하나가 모든 unit에 쓰인다.

**예시**

```text
offset 0                                              kv_base                capacity
│ phase 영역 (phase마다 offset 0부터, 서로 덮어씀)     │ key/value cache 영역        │
│ slot weight들 │ slot state들 │ input │ output │      │ … │ L1 V │ L1 K │ L0 V │ L0 K │
```

| target | state | 주소 |
|---|---|---|
| UPMEM | 0 (program은 IRAM) | `MRAM_BASE + offset`으로 stage program을 link (`place`) |
| AiMX | descriptor chain, PMM | table을 region(≤ 1023 × 2 MiB)으로 잘라 FPGA마다 할당, region 주소 + offset |

## Key/value cache

**내용**
장치 cache는 layer마다 KEY, VALUE 영역에 있고, cache를 읽는 stage operand는 그 영역을 그대로 읽는다.
`write_kv(layer, kind, cache, start, stop)`는 host cache `[kv_heads, positions, head_dim]`의 위치 범위를 그 layer에서 해당 cache를 읽는 모든 stage의 adapter에 넘긴다.
adapter는 위치를 대기열에 넣고, cache를 읽는 stage가 다음에 실행되기 직전에 보낸다.

**근거**
- `PIMSession.write_kv`: `stage.kv[1] == kind`인 stage가 없으면 `ValueError`.
- AiMX `_push_kv`: 위치를 region host shadow에 stub 배치(`pim_model_slot_layout`)로 쓰고 항목마다 바뀐 byte 구간만 `aimx_model_push_ranges`로 보낸다.
- UPMEM `_push_kv`: DPU별 host image에 반영한 뒤 `[kv_base, capacity)` 전체를 한 번 보낸다(`upmem_model_kv_push_region`).

**예시**

| API | 저장소 | 그래프 연결 |
|---|---|---|
| `write_kv` | 장치 KEY/VALUE 영역 | cache operand가 장치 영역을 읽음 (`kv_inputs`) |
| `reserve_kv` / `append_kv` / `kv_view` | Session의 CPU `kv_table` (sequence 축은 끝에서 두 번째) | 없음, 호출자가 직접 사용 |

## Config 선택

**내용**
stage config는 `select_layer_config`가 실제 unit 수로 고른다.
명시 config가 있으면 그것을, 없으면 autotune cache의 측정 기록 중 제약을 만족하는 것에서 `batch round × total_ms`가 최소인 것을 쓴다.
`compile_model`은 phase별로 고른 config를 명시 config로 넘긴다.

**근거**
- 명시 config도 cache 경로도 없으면 `ValueError("model layers require an explicit config or measured autotune records")`.
- 측정 기록은 `target_model(spec).stage_target(problem)`(AiMX GEMM batch)으로 조회한다.
- 모든 후보가 거부되면 후보별 거부 이유를 모아 `ValueError`를 낸다.

**예시**

```python
def configure(gemms, spec):
    return {gemm.node: PIMTorchConfig(spec=spec, config=configs[gemm.node]) for gemm in gemms}

session = PIMSession(target_spec("upmem", num_units=2048, nr_tasklets=16))
session.compile({"prefill": (layers, example_inputs, configure)})
with session:
    output = session.run_model("prefill", *inputs)
```

## 통계

**내용**
`session.stats()`는 adapter의 누적 통계를 돌려준다. group 업로드 통계 이름은 두 target이 같다.
`group0_*`는 첫 key의 pack과 push로, 가릴 앞 실행이 없으므로 따로 누적한다.

**근거**
- `GROUP_LOAD_STATS`: `weight_uploads`, `pack_wall_ms`, `background_pack_ms`, `upload_wall_ms`, `group0_pack_ms`, `group0_upload_ms`.
- [upmem_adapter.py](../pim_backend/backend/upmem_adapter.py), [aimx_adapter.py](../pim_backend/backend/aimx_adapter.py) `stats`.

**예시**

| target | 통계 |
|---|---|
| 공통 | `GROUP_LOAD_STATS` |
| UPMEM | `TransferStats`(`weight_bytes`, `activation_bytes`, `output_bytes`, `metadata_bytes`, `launches`, `activation_hits`, `kv_bytes`), `program_loads`, `device_compiles`, `weight_pushes`, `program_load_ms`, `invoke_ms`, `kv_push_ms`, `kv_layout_check_ms`, `resource_opens`, `resource_closes` |
| AiMX | `weight_bytes`, `invocations`, `run_wall_ms`, `chain_build_ms`, `kv_bytes`, `kv_push_ms`, `kv_layout_check_ms`, `breakdown`(stage 이름별 호출 수·wall·native 시간), `live_bytes`, `visible_fpgas`, `units` |
