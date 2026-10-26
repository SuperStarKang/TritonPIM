# 모델 단위 PIMSession과 LayerPlan 실행 설계

## 역할과 수명

**내용**
PIMSession은 모델 전체의 Device, Weight, KV, staging memory를 소유한다.
LayerPlan은 Transformer Block 또는 MoE Block 하나의 CPU/PIM 실행 계획이다.
Device는 모델 시작에 한 번 확보하고 Session 종료에 한 번 해제한다.

**근거**
- Weight 교체와 KV 수명은 Layer 사이에 걸쳐 있다.
- CPU/PIM 분할과 GEMM 배정은 각 Layer의 shape와 의존성에 따라 결정된다.

**예시**

| PIMSession | LayerPlan |
|---|---|
| 실제 자원과 topology | Layer 연산 그래프 |
| 모델 전체 Weight/KV table | CPU/PIM partition |
| 현재 Layer weight와 프로그램 | GEMM별 config와 artifact |
| 메모리 주소와 staging buffer | instance 대응, p_b, round, unit mapping |
| 모델의 Layer 실행 순서 | 컴파일된 Layer 실행 |

## 준비 과정

**내용**
Session이 먼저 Device를 확보한 뒤 Torch block을 FX로 분석하고 LayerPlan을 생성한다.
LayerPlan은 전달받은 실제 자원으로 config를 선택하고 CPU/PIM 구간을 준비한다.
전체 Layer의 메모리 요구량을 모은 뒤 Session의 adapter가 주소와 최종 device artifact를 확정한다.
컴파일과 고정 buffer 준비는 반복 실행 경로에서 수행하지 않는다.

**근거**
- Execution Tile 크기가 local layout과 MRAM 사용량을 정한다.
- 모델 전체 저장 공간의 소유자는 Session이다.
- Device 확보와 tensor 주소 배정은 별도 단계다. 공용 주소를 정하려면 전체 Layer의 요구량이 필요하다.

**예시**

```text
PIMSession 진입: Device alloc / 실제 자원 조회
    │
session.prepare(torch_blocks, example_inputs, configure=...)
    │
    ├─ adapter.begin_model: 타깃 실행 상태 생성
    │
    ├─ Layer별 준비
    │    ├─ FX 분석 → LayerPlan 생성
    │    ├─ CPU/PIM 분할 / config / Execution Tile / p_b / round
    │    ├─ PIM 컴파일 준비: buffer 요구량 확보
    │    └─ CPU 컴파일 준비 출력 → 다음 Layer의 예제 입력
    │
    └─ adapter.prepare_model
         └─ 전체 Layer의 Weight/scratch 배치 / 최종 ELF / native binding
    │
session.run_model: 준비한 Layer들을 순서대로 실행
    │
Session 종료: Device free
```

## Execution Tile과 Batch GEMM

**내용**
GEMM 코드를 head별로 나누지 않는다. 같은 프로그램이 처리할 Execution Tile의 크기와 배정 수를 선택한다.
Block–Block Group–Execution Tile은 instance 내부 계산을 표현한다.
p_b는 Torch runtime의 instance 병렬성이며 Common IR의 축으로 추가하지 않는다.

**근거**
- E_d = block_d × group_d × tile_d.
- P = p_m × p_n × p_k, p_b = min(H, floor(R / P)), R_b = ceil(H / p_b).
- 자원 한도는 p_b × P ≤ R이다.

**예시**

32개 head의 Q[1024,128] × Kᵀ[128,1024], Block=(16,16,32), Group=(1,1,1):

| R | Tile | Execution Tile | partition | p_b | Batch round |
|---:|---|---|---|---:|---:|
| 2048 | (8,8,4) | (128,128,128) | (8,8,1) | 32 | 1 |
| 512 | (8,8,4) | (128,128,128) | (8,8,1) | 8 | 4 |
| 512 | (16,16,4) | (256,256,128) | (4,4,1) | 32 | 1 |

```text
batch round
  └─ 기존 M/N/K round
       ├─ instance별 입력 packing
       ├─ DPU마다 다른 데이터를 같은 local slot에 H2D
       ├─ 전체 집합 synchronous launch
       ├─ D2H
       └─ instance/tile 위치로 복원, 필요하면 K 부분합 누적
```

첫 BMM은 p_k=1이다. p_k=1이어도 K가 Execution Tile보다 크면 K round가 필요하다.
Physical DPU 번호를 head 내부 좌표로 직접 해석하지 않고 배정표를 사용한다.
마지막 batch의 빈 배정은 device에서 계산하지 않으며 출력 복원에서 제외한다.
DPU 하나는 해당 M/N/K round에 배정된 Execution Tile을 처리한다. Round가 여러 개이면
같은 DPU가 다음 tile의 데이터를 받아 동일 프로그램으로 계산한다.

## Layer와 전체 모델 실행

**내용**
LayerPlan은 CPU/PIM stage의 의존성을 보존하며 동일 형태의 독립 GEMM을 묶는다.
Session은 Layer 진입 시 weight를 올리고 해당 Layer 내부에서는 유지한다.
QK와 PV는 별도 stage/config이며 같은 stage의 instance들이 프로그램을 공유한다.

**근거**
- QK와 PV의 shape가 다르므로 같은 config나 ELF를 강제하지 않는다.
- Q/K/V activation과 모델 weight WQ/WK/WV는 다른 tensor이다.

**예시**

```text
Session: Layer weight 업로드
                  │
CPU Norm → PIM Q/K/V projection → CPU head/position 처리
                  │
              Session KV 접근
                  │
PIM batched QK → CPU softmax → PIM batched PV
                  │
CPU head 결합 → PIM output projection → CPU residual/norm
                  │
          PIM/CPU FFN → CPU residual
                  │
             다음 Layer 입력
```

```text
모델 입력 처리
    │
Session: Layer 0 weight → Compiled LayerPlan 0 ↔ KV[0]
    │
Session: Layer 1 weight → Compiled LayerPlan 1 ↔ KV[1]
    │
   ...
    │
Session: Layer L-1 weight → Compiled LayerPlan L-1 ↔ KV[L-1]
    │
모델 출력 처리
```

## 최소 상태와 메모리 배치

**내용**
Activation은 매 호출 전송하고 scratch는 출력 회수 후 덮어쓴다.
Activation residency, ready, generation, completion token, overwrite 추적은 새 모델 실행 경로에 넣지 않는다.
Weight/KV의 저장 위치는 준비 단계에 정하며 실행 중 자동 migration하지 않는다.

**근거**
- 동기식 H2D/launch/D2H와 고정 배정으로 scratch 수명을 결정할 수 있다.
- 모델 전체 weight를 table에 등록하는 것과 전부 device에 상주시키는 것은 다르다.

**예시**

| 상태 | 내용 |
|---|---|
| weight_table | Layer/operand별 host tensor, shard, device slot |
| kv_table | Layer별 K/V 저장소, capacity, 현재 길이 |
| loaded_layer | 현재 device weight 영역의 Layer |
| loaded_program | 현재 적재된 device 프로그램 |
| scratch | 준비된 고정 layout과 재사용 staging buffer |

```text
보존되는 KV 저장소
현재 Layer의 weight 영역
공용 activation/output scratch
```

KV는 준비 시 정한 CPU 또는 PIM 위치를 사용한다. 첫 구현에서 지원하는 저장 위치를 API와 예제에 명시한다.
한 Layer weight 상주 정책에서는 다음 모델 순회 때 weight를 다시 전송한다.

## 공통 상주 정책 (P-1 / P-2)

**내용**
무엇을 device에 남길지는 target과 무관한 `ResidencyPlanner`(`backend/residency.py`)가 결정한다.
Adapter는 실제 자원(unit 수, unit당 용량)과 stage별 요구량(weight, state, 공용 scratch의
unit당 byte)만 보고한다. Planner는 아래 순서로 처음 만족하는 정책을 고르고, `PIMSession.run_model`은
그 정책에 따라 layer 진입과 종료 때 adapter의 `enter_layer`/`exit_layer`를 호출한다.

**근거**
- Weight는 요청 사이에 바뀌지 않는다. 용량이 허락하면 다시 보낼 이유가 없다.
- State는 weight가 아니면서 device에 상주하는 것이다. AiMX에서는 descriptor chain, PMM,
  Arel, activation buffer이고, UPMEM에서는 MRAM을 쓰지 않으므로 0이다.
  prefill에서는 state가 weight보다 커질 수 있어서(AiMX chain), weight와 분리해서 판단한다.
- 공용 scratch는 모든 stage가 같은 영역을 쓰므로 최댓값만 더한다. stage마다 자기 activation
  buffer를 갖는 target은 그 크기를 state로 보고하고 scratch는 0으로 보고한다.
- Stage가 일부 unit만 쓰더라도 첫 unit은 모든 stage가 쓴다. 그래서 stage 요구량을 모두 더한
  값을 unit 하나의 상한으로 사용한다(보수적 판단).

**예시**

```text
W_l = layer l의 weight 합, S_l = layer l의 state 합, X = 공용 input + output scratch 최댓값
cap = unit당 용량 (adapter 보고)

P-1 model    Σ_l W_l + Σ_l S_l + X ≤ cap   모든 layer의 weight와 state를 상주
P-2 layer    max_l (W_l + S_l) + X ≤ cap      layer 하나만 상주, layer마다 weight와 state를 교체
그 외        준비 단계에서 오류 (필요량과 용량을 보고)
```

| 정책 | `enter_layer` | `exit_layer` | 요청당 weight 전송 |
|---|---|---|---|
| P-1 model | 첫 진입에서만 weight upload + state 생성 | 없음 | session 동안 1회 |
| P-2 layer | weight upload | weight와 state 해제 | layer마다 |

모든 adapter가 이 정책을 따른다. Target별 차이는 보고하는 숫자와 배치 방법뿐이다.

| target | state | 공용 scratch | 정책별 배치 |
|---|---|---|---|
| UPMEM | 0 (프로그램은 IRAM) | 입력·출력 MRAM scratch 1벌 | P-1: layer마다 다른 weight 주소(누적), P-2: 모든 layer가 같은 weight 영역을 재사용. stage 프로그램은 배치된 주소로 link |
| AiMX | descriptor chain, PMM, Arel, activation buffer | 0 (stage마다 자기 buffer) | (stage, FPGA)별 context. P-2는 layer 퇴장 시 모두 해제 |

작은 모델은 P-1이 되어 두 번째 요청부터 weight를 보내지 않고, Llama-2-7B INT8처럼
전체 weight가 MRAM을 넘으면 P-2가 되어 layer마다 weight를 보낸다.

## Config와 cache

**내용**
기존 config 후보와 측정 비용을 사용한다. 예상 BMM 비용은 우선 R_b × 기존 GEMM 비용으로 비교한다.
이 값은 후보 순위를 위한 추정이며 실제 BMM latency는 별도로 측정한다.
프로그램 key에서 head 번호와 입력 주소는 제외하되 실제 codegen/layout 의존성은 유지한다.

**근거**
- 작은 tile의 여러 batch round와 큰 tile의 한 round는 서로 다른 전송·계산 비용을 갖는다.
- 같은 Execution Tile 크기만으로 dtype, tasklet, symbol 주소까지 같다고 보장할 수 없다.

**예시**

```text
compile count: 새 artifact 생성 횟수
program load count: device 프로그램 적재 횟수
launch count: 실제 SDK launch 횟수
```

## 구현 범위와 검증

**내용**
UPMEM, 정적 shape, 동기 실행, host 출력 회수를 먼저 구현한다.
기존 resident API는 유지하고 모델용 API를 추가한다.
사용자가 빌드와 검증을 실행하며, 코드 작성 과정에서는 실행 결과를 주장하지 않는다.

**근거**
- 기존 resident API의 전체 DPU/단일 round 가정과 새 BMM 실행을 분리해야 한다.
- 모델 수명과 BMM 정확성을 함께 검증해야 한다.

**예시**

| 검증 | 기대 결과 |
|---|---|
| 여러 Layer 실행 | Device alloc/free 각각 1회 |
| Layer 내부 여러 GEMM | Layer load 이후 weight H2D 없음 |
| 서로 다른 head 입력 | 각 reference와 일치 |
| Physical mapping 변경 | 동일한 논리 출력 |
| 여러 batch/M/N/K round | 출력 복원 및 K 누적 정확 |
| 마지막 불완전 batch | 빈 instance 실행/출력 사용 없음 |
| 입력 내용 변경 | 항상 새 입력 계산 |
| Layer 교체 | KV 내용 유지 |
| 반복 실행 | 추가 compile 없음 |

MoE는 고정 expert placement, capacity 기반 token packing과 출력 복원으로 확장한다.
동적 expert 이동, weight 복제, 비동기 overlap, 전체 모델 weight 동시 상주는 후속 범위다.

## 추가한 API와 실행 규칙

**내용**
`PIMSession.prepare()`는 열린 Session 안에서 Torch block들을 받아 FX 분석과 LayerPlan 생성을 수행한다.
내부 분석 함수 `build_layer_plan()`은 device를 확보하거나 컴파일하지 않는다.
LayerPlan은 Session이 전달한 target spec과 runtime으로 config 및 batch 배정을 준비한다.
`run_model()`은 준비된 Layer 목록을 순서대로 실행하며 tuple 출력은 다음 Layer의 positional 입력으로 전달한다.

**근거**
- [model.py](../pim_backend/torch_backend/model.py)는 독립 GEMM을 묶고 CPU/PIM partition을 구성한다.
- [runtime.py](../pim_backend/backend/runtime.py)는 자원 수명, Weight/KV table과 모델 실행 순서를 소유한다.
- 모델 입력 embedding은 호출자가 처리한다. GEMM이 있는 최종 head는 LayerPlan으로 포함할 수 있다.

**예시**

```python
from torch_backend import PIMSession, PIMTorchConfig, target_spec


def configure(gemms, spec):
    return {
        gemm.node: PIMTorchConfig(spec=spec, config=gemm_configs[gemm.mnk])
        for gemm in gemms
    }


with PIMSession(target_spec("upmem", num_units=512, nr_tasklets=16)) as session:
    plans = session.prepare(blocks, example_inputs, configure=configure)
    output = session.run_model(*inputs)
```

`blocks`는 실행 순서의 Torch module 목록이고 `example_inputs`는 첫 block의 입력 tuple이다.
`configure`는 실제 확보한 spec을 받아 GEMM별 명시적 config 또는 autotune cache 설정을 제공한다.
위 예제의 `gemm_configs`는 `(M, N, K)`별 config 사전이다. 후보 선택과 batch 배정은 LayerPlan에서 수행한다.
반환값 `plans`는 `session.layers`와 같은 tuple이며 index는 입력 block 순서대로 부여한다.

| 준비 또는 실행 단계 | 실제 동작 |
|---|---|
| Session 진입 | DPU alloc 1회, SDK의 실제 확보 수 조회 |
| Layer 분석 | Session 안에서 FX export 후 독립적인 동일 shape/dtype/config GEMM을 같은 의존성 깊이에서 묶음 |
| Config 선택 | 명시한 config 또는 기존 cache records 중 `batch_rounds × total_ms` 최소값 |
| 다음 Layer 입력 | CPU 컴파일 준비에서 계산한 출력을 다음 block의 예제 입력으로 전달 |
| 메모리 배치 | 모델에서 가장 큰 Layer weight 영역 + 공용 입력/output scratch |
| 컴파일 | 기존 GEMM template/packing 사용, 모델용 host entry와 주소가 확정된 ELF 생성 |
| Layer 진입 | 해당 Layer의 모든 PIM weight slot을 한 번씩 업로드 |
| Layer 내부 | CPU Inductor 구간과 native GEMM 구간 실행, weight H2D 생략 |
| Batch round | 동일 프로그램 유지, instance별 입력 전송과 결과 회수 |
| Session 종료 | Host staging과 DPU 해제 |

Session이 공통 준비 순서를 제어하고 adapter는 타깃 실행 상태와 물리 배치만 관리한다.
LayerPlan은 Session 객체 내부를 조회하지 않고 전달받은 spec과 runtime을 사용한다.
`LayerPlan.exported_graph`는 benchmark의 원본 FX 그림 출력에 사용하며, 실행에는 분할된 `graph`를 사용한다.

## 현재 구현 범위와 후속 작업

**내용**
이번 변경은 모델 수명 관리와 정적 BMM 실행 경로의 첫 구현이다.
현재 자원 정보 중 DPU 수·topology는 SDK에서 얻고, 메모리 용량·tasklet 설정은 기존 target spec을 사용한다.
사용자는 준비한 shape/dtype의 CPU 입력과 변경하지 않는 모델 weight를 사용한다.

**근거**
- 자원 spec과 config는 prepare에서 확정하며, 실행 중 재탐색하거나 자동 재준비하지 않는다.
- Scratch는 매 round 전송·회수하고, 프로그램 교체는 기존 MRAM 보존용 ELF 연결 방식을 사용한다.
- Cache 조회는 기존 key를 유지하므로 실제 자원 수와 일치하는 측정 records가 필요하다.

**예시**

| 항목 | 현재 구현 |
|---|---|
| Activation GEMM | FP32 mm/bmm, INT8 `_int_mm` 묶음; 여러 batch/M/N/K round |
| Resident weight GEMM | 2차원 operand, instance 1개, M/N/K round 1회 |
| Programs | Stage 내 head별 ELF 분리 없음; code/layout이 같은 artifact는 기존 cache로 공유 |
| Physical mapping | Topology 순서와 instance 내부 logical unit 번호 분리 |
| 마지막 batch | 비활성 DPU는 device 진입 시 반환, 전송·출력 복원 제외 |
| KV | `reserve_kv`, `append_kv`, `kv_view`; CPU 저장소, sequence 축은 끝에서 두 번째 |
| 준비된 plan | Session에 연결됨; 다른 Session에는 새 plan을 만들어 준비 |
| 기존 API | `compile_layer/start/load_weights/run_layer` 경로는 제거됨 (model 경로만 유지) |
| 남은 구현 | Attention 그래프의 KV 연결, PIM KV 저장, MoE expert placement/routing |

KV API는 exported graph에 append/read 연산을 자동 삽입하지 않는다.
실제 KV를 사용하는 attention 연결은 후속 구현이며, 현재 예제는 Layer 교체 중 보존 여부만 검증한다.
전체 모델 weight 동시 상주와 weight GEMM의 여러 scheduling round도 후속 범위다.

## 검증 방법과 측정값

**내용**
아래 명령은 사용자가 기존 TritonPIM 환경에서 실행한다. 이 변경 작성 과정에서는 빌드·테스트를 실행하지 않았다.
장비 예제에는 설정된 SDK와 8개 이상의 사용 가능한 DPU가 필요하다.

**근거**
- Device compile/load/launch 횟수는 다른 사건이므로 `session.stats()`에서 구분한다.
- 여러 M/N/K round의 GEMM 정확성과 2048-DPU 대규모 성능 측정은 별도 장비 검증이 남아 있다.

**예시**

프로젝트 루트 `/home/dlrkdals/TritonPIM` 기준:

```bash
cmake --build triton/build/cmake.linux-x86_64-cpython-3.12 \
  --target triton-shared-opt triton-shared-translate

python triton_shared/pim_backend/torch_backend/examples/upmem/torch_model_session.py
```

| 검증 | 기대 결과 |
|---|---|
| Python planner | 2048/512 DPU 예제 배정, 후보 비용 비교, QK/PV 의존성 유지 |
| UPMEM Session | 실제 자원 수 사용, alloc/free 각 1회, 여러 Layer와 KV 보존 |
| UPMEM Session | 준비 중 launch 없음, CPU tuple 출력의 다음 Layer 입력 연결 |
| 장비 예제 | Layer마다 stage `(H,p_b,R_b) = (1,1,1), (5,2,3)` |
| 장비 예제 | 모델 호출마다 `exact_match=True launches=8`, 동일 입력 buffer의 내용 변경 반영 |
| 장비 예제 | 두 Layer의 weight H2D 합계 4096 bytes/call, Layer 내부 재전송 없음 |

```text
device_compiles: 이번 Session 준비 중 실제 생성한 새 DPU ELF 수 (cache hit 제외)
program_loads: SDK dpu_load 호출 수
launches: SDK synchronous launch 호출 수
weight_bytes / activation_bytes / output_bytes: 실제 전송 payload
metadata_bytes: tile metadata와 DPU 활성 값의 전송 payload
resource_opens / resource_closes: DPU 집합 확보/해제 횟수
```

`device_compiles`는 Torch/MLIR/host compiler 호출 전체를 세는 값이 아니다.
장비 예제의 `prepare_ms`, `latency_ms`는 host wall time이다.
Session-first 경로의 `prepare_ms`에는 FX export와 LayerPlan 분석도 포함된다.
새 준비 순서의 성능 영향은 별도 장비 측정으로 확인한다.
