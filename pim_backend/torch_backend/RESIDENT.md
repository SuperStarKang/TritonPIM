## Block 단위 weight-stationary

**내용**
`prepare_block()`은 exported FX graph의 고정 weight를 분석하고, 각 weight의 조각을
동일한 전체 DPU 집합에 미리 전송한다. `run()`은 공용 입력·출력 scratch를 사용한다.
대상은 정적 shape의 CPU FP32 `linear`/`addmm`/`mm`과 INT8 `_int_mm`이다.
`linear`/`addmm`를 `mm`과 주변 연산으로 정규화하고, 각 GEMM은 PIM, 주변 연산은 CPU Inductor로 실행한다.

**근거**
- 기존 FP32/INT8 Torch template을 사용하며, Common/UPMEM pass와 host/device 생성기는 변경하지 않는다.
- FX 입력 출처로 weight를 판정하며, 전치형에서는 weight가 왼쪽 입력으로 이동한다.
- DPU 집합은 block이 `close()`될 때까지 한 block이 소유한다. 그동안 다른 PIM 실행을 섞지 않는다.

**예시**
```text
각 DPU                     동일한 모든 DPU가 모든 GEMM에 참여
┌─────────────────────────┐
│ Wq shard / window들      │
│ Wk shard / window들      │
│ Wv shard / window들 ...  │
├─────────────────────────┤
│ 공용 INPUT scratch      │ ← A 또는 Aᵀ, 필요하면 두 동적 입력
│ 공용 OUTPUT scratch     │ → C 또는 Cᵀ / K partial
└─────────────────────────┘
```

## 명시적 FX 실행 설정

**내용**
정규화 전 원본 `linear`/`addmm`/`mm`/`_int_mm` node 이름에 `ResidentGemmConfig`를 지정한다.
`padded_shape`은 전치하기 전 논리 축 순서 `(M,N,K)`이며, padding이 필요 없으면 원래 shape를 쓴다.
전치형은 `B.T @ A.T`를 계산한 뒤 출력의 전치와 slice를 복원한다.

**근거**
- 컴파일 실패를 잡아 다른 방향·padding·config로 바꾸지 않는다. 예외는 호출자에게 전달한다.
- 모든 config의 `p_m*p_n*p_k`와 hardware profile의 `num_units`가 `dpu_count`와 같아야 한다.
- `Block Group M % nr_tasklets == 0`과 기존 `ConstraintEngine` 조건을 확인한다.
- 모든 GEMM의 weight window 합과 최대 공용 scratch가 MRAM 예산 안에 있어야 한다.
- FX 분석의 원본→정규화 GEMM 대응으로 설정을 전달하고, weight 출처와 실제 고정 operand가 일치하는지 확인한다.
- 메모리 제약과 MRAM slot 크기는 실제 dtype의 바이트 수를 사용한다. FP32는 원소당 4바이트다.

**예시**
```python
from autotuner import PIMConfig
from torch_backend import PIMTorchConfig, ResidentGemmConfig, prepare_block, target_spec
from torch_backend.fx_analysis import analyze_fx_gemms

# 논리 (M,N,K)=(1,32,32); 물리 전치 GEMM은 (32,4,32).
config = ResidentGemmConfig(
    torch_config=PIMTorchConfig(
        spec=target_spec("upmem", num_units=2, nr_tasklets=16),
        config=PIMConfig(16, 2, 16, 1, 1, 1, 1, 2, 2, 2, 1, 1),
    ),
    orientation="dual",
    padded_shape=(4, 32, 32),
)

configs = {
    gemm.node: config
    for gemm in analyze_fx_gemms(exported)
}
with prepare_block(exported, (activation,), configs=configs, dpu_count=2) as block:
    output = block.run(activation)
```

## 준비와 실행의 수명

**내용**
준비 단계는 각 GEMM을 별도로 컴파일하고 기존 host staging을 실행해 전송 window를 수집한다.
이때 DPU 계산은 생략하고 임시 출력은 0으로 채운다. 주소 배치와 ELF 링크 후 모든 weight를 전송한다.
실행 단계는 기록된 round별 ELF를 선택하고 weight H2D를 생략한다.

**근거**
- `target_ir`의 stage → source argument와 to_mram/gather → symbol 관계를 읽는다.
- Weight window는 같은 binding에서 모든 DPU의 packed byte가 같을 때만 하나로 저장한다. Hash 일치 후 byte 비교도 수행한다.
- 각 round의 symbol을 `--defsym`으로 배치하고, 전송은 별도의 `{address,size}`로 수행한다.
- ELF의 symbol 주소를 확인하고, loadable MRAM segment와 별도로 남은 MRAM section 할당을 거부한다.
- Input scratch는 같은 block 호출 안에서 동일한 packed byte일 때 재사용한다. 다음 `run()`은 입력 재사용 상태를 비운다.

**예시**
```text
prepare_block
  FX 분석 + linear/addmm 정규화 (원본 GEMM 이름과 weight 출처 유지)
  FX 고정 weight/view 평가
  각 mm/_int_mm을 개별 PIM 구간으로 분리, 나머지는 CPU 구간
  각 GEMM 컴파일 + host 전송 capture (DPU launch 없음)
  모든 weight window와 공용 scratch 주소 결정
  round별 device ELF 링크
  전체 DPU에 모든 weight window H2D

run
  frozen state와 입력 signature 확인
  CPU 구간 → layout/padding → resident GEMM → 출력 복원
  weight H2D: 0
  다음 GEMM: 동일 INPUT/OUTPUT scratch 사용
```

## 측정과 후보 비교

**내용**
`benchmark_resident_plans()`에 비교할 완성된 block 설정들을 명시적으로 전달한다.
각 후보는 전체 block을 준비하고 한 번 실행하여 CPU의 지연 컴파일과 정확성 확인을 마친 뒤 측정한다.
후보 생성이나 실패 후보를 건너뛰는 처리는 하지 않는다.

**근거**
- `weight_prepare_ms`는 고정 입력 처리, 준비용 입력 layout, host capture, 최초 프로그램 적재와 weight H2D를 포함한다.
- `prepare_ms`는 컴파일·링크·DPU 확보까지 포함한 준비 전체 시간이다. 두 시간의 차이를 전송 성능으로 해석하지 않는다.
- `run_ms`는 activation layout, host staging, 전송, 프로그램 교체, DPU 실행, combine, CPU 구간을 포함한다.
- 점수는 `weight_prepare_ms + reuse_count * median(run_ms)`이며, 반복 사용에 대한 비교 지표다.
- 반복 host weight packing과 activation byte 비교 비용도 남아 있으므로 측정에 포함된다.

**예시**
```python
from torch_backend import benchmark_resident_plans

measurements = benchmark_resident_plans(
    exported,
    (activation,),
    plans={"normal": normal_configs, "dual": dual_configs},
    dpu_count=2,
    repetitions=3,
    reuse_count=8,
)
print(measurements[0])
```

## 사용자가 수행할 검증

**내용**
아래 예제는 두 projection의 weight를 모든 DPU에 분산하고 두 번 실행한다.
CPU 결과와 일치하는지, 준비 이후 weight H2D가 0인지 확인한다.
구현 과정에서는 빌드·실행·검증을 수행하지 않았다.

**근거**
- 예제는 같은 activation을 사용하는 두 GEMM으로 공용 scratch 재사용도 관찰한다.
- `stats()`는 누적 전송량, `last_transfers`는 마지막 `run()`의 전송량을 반환한다.
- 기존 compiler 산출물에 `target_ir`가 없으면 사용할 수 없으며, 새 산출물 생성이 필요하다.

**예시**
```bash
python pim_backend/torch_backend/examples/torch_resident_block.py --orientation normal
python pim_backend/torch_backend/examples/torch_resident_block.py --orientation dual
```

기대하는 핵심 출력:

```text
prepare_weight_bytes=2048
step=0 correctness=PASS weight_h2d=0 ...
step=1 correctness=PASS weight_h2d=0 ...
```

## FP32 FX 연결 검증

**내용**
`torch_resident_fp32.py`는 `nn.Linear`, `torch.addmm`, 동적 입력 두 개의 `torch.mm`을 한 graph에서 실행한다.
각 GEMM은 `[512,64] × [64,1024]`이고, 모든 연산이 같은 2048 DPU·16 tasklet 설정을 사용한다.
고정된 두 weight는 준비 때 전송하고, 서로 다른 입력으로 두 번 실행해 정확도와 전송량을 확인한다.

**근거**
- `linear` bias와 `addmm`의 `alpha=0.5`, `beta=2.0` 처리는 CPU 구간에 남는다.
- 원본 node 이름으로 설정하고, weight 영역에는 두 고정 weight의 GEMM만 있는지 확인한다.
- 준비 시 DPU launch는 0, 각 실행은 3회이며, 반복 실행 중 weight H2D는 0이어야 한다.
- normal/dual 모두 K block 두 개를 한 Execution Tile에서 처리한다. 성능 비교용 attention 실험은 아니다.

**예시**
```bash
python pim_backend/torch_backend/examples/torch_resident_fp32.py --orientation normal
python pim_backend/torch_backend/examples/torch_resident_fp32.py --orientation dual
```

성공 시 각 방향에서 다음 두 줄을 출력한다. 중간에는 activation 전송량과 재사용 횟수도 표시한다.

```text
step=0 correctness=PASS weight_h2d=0 ... launches=3
step=1 correctness=PASS weight_h2d=0 ... launches=3
```

## Single-head FP32 attention 성능 실험

**내용**
`torch_resident_attention.py`는 sequence/KV length 1024, input/head dimension 4096의 single-head attention을 실행한다.
Q/K/V는 bias 없는 별도 `nn.Linear`이며, `softmax(QKᵀ / 64) @ V`를 계산한다.
Mask, dropout, output projection, residual, LayerNorm, FFN은 포함하지 않는다.
GEMM 다섯 개는 모두 같은 2048 DPU·16 tasklet을 사용하며, scale과 softmax는 CPU에서 실행한다.

**근거**
- FX 분석에서 `linear` 3개, `mm` 2개와 Wq/Wk/Wv 출처를 확인한다. QKV fusion은 하지 않는다.
- 서로 다른 FP32 입력 두 개를 번갈아 사용하고, 모든 실행 결과를 CPU reference와 비교한다.
- 준비 시 weight 영역은 세 개, 계산 launch는 0이어야 한다. 실행마다 weight H2D 0, launch 5, activation hit 2 이상을 확인한다.
- 최적화된 설정을 탐색한 결과가 아니라, 각 GEMM을 한 scheduling round에 배치한 고정 기준 설정이다.

**예시**
공통 block은 `(16,16,32)`, group은 `(1,1,1)`이다. Normal의 partition은 `(32,64,1)`이다.
Dual은 M/N tile과 partition을 교환한다. Padding과 split-K는 사용하지 않는다.

| 연산 | 논리 `(M,N,K)` | Normal `(tile_m,tile_n,tile_k)` | Normal Execution Tile |
|---|---|---|---|
| Q/K/V projection 각각 | `(1024,4096,4096)` | `(2,4,128)` | `(32,64,4096)` |
| QKᵀ | `(1024,1024,4096)` | `(2,1,128)` | `(32,16,4096)` |
| PV | `(1024,4096,1024)` | `(2,4,32)` | `(32,64,1024)` |

이 설정에서 각 DPU의 weight는 3 MiB, 입력 scratch는 768 KiB, 출력 scratch는 8 KiB다.
전체 DPU의 최초 weight 전송량은 6 GiB이며, 원본 weight 192 MiB의 M축 복제를 포함한다.
현재 capture 구현은 upload 전까지 이 6 GiB의 packed weight를 host에도 보관한다. 다른 host buffer는 별도다.

```bash
python pim_backend/torch_backend/examples/torch_resident_attention.py --orientation normal > resident_attention_normal.json
python pim_backend/torch_backend/examples/torch_resident_attention.py --orientation dual > resident_attention_dual.json
```

진행 로그는 stderr, 최종 JSON은 stdout으로 출력한다. 사용자가 직접 실행하며, 작성 과정에서는 실행하지 않았다.
각 명령은 준비 후 warm-up 3회, 일반 측정 3회, 별도 구간 계측 1회를 수행한다.
성공 시 진행 로그의 핵심 내용은 다음과 같다. 시간과 전송량은 실제 측정값으로 표시된다.

```text
sample=0 correctness=PASS wall_ms=... weight_h2d=0 ... launches=5
sample=1 correctness=PASS wall_ms=... weight_h2d=0 ... launches=5
sample=2 correctness=PASS wall_ms=... weight_h2d=0 ... launches=5
partition_profile correctness=PASS PIM_partitions=5 (excluded from median)
```

## Attention 시간의 측정 범위

**내용**
비교의 주 지표는 `median_wall_ms`다. 추가 계측 wrapper 없이 `block.run()` 호출 전체를 재며,
입력 검사, layout, packing, 전송, 프로그램 교체, DPU 계산, CPU 구간, 출력 복원을 포함한다.
입력 생성, CPU reference, FX export, 준비, warm-up, 정확도 검사, 출력/JSON 저장은 이 값에서 제외한다.

**근거**
- CPU partition의 지연 컴파일은 첫 warm-up에서 수행된다.
- 구간 계측은 warm-up과 일반 측정이 끝난 뒤 별도 실행에서만 `forward`를 감싼다.
- 각 PIM partition 직후 native timing과 전송 counter를 복사하므로 마지막 GEMM의 시간만 기록하는 문제를 피한다.
- CPU partition의 `operators`는 포함된 FX 연산 목록이다. 하나의 CPU 구간에 속한 연산별 시간을 나누어 추정하지 않는다.

**예시**
| JSON 항목 | 범위 |
|---|---|
| `preparation.export_ms` | `torch.export.export()` |
| `preparation.prepare_ms` | FX 정규화, 컴파일, DPU 확보, capture, 링크, weight upload |
| `preparation.weight_prepare_ms` | 기존 resident API의 weight/layout/capture/upload 묶음; 순수 H2D 시간이 아님 |
| `preparation.resident_upload_ms` | 최초 프로그램 적재, weight 전송, capture buffer 정리 |
| `samples[*].wall_ms` / `median_wall_ms` | 계측 wrapper 없는 `block.run()` 전체 |
| `samples[*].graph_ms` / `median_graph_ms` | 입력·state 검사 및 출력 pytree 복원 등을 제외한 내부 graph 실행 |
| `profile.partitions[*].wall_ms` | CPU 또는 PIM partition 한 번의 호출 |
| `profile.partitions[*].native_ms` | PIM host의 layout/H2D/kernel/D2H 등 기존 13개 단계 |
| `profile.partitions[*].outside_native_stages_ms` | partition 시간에서 native 단계 합을 뺀 값; Torch layout과 wrapper 등 포함 |
| `profile.outside_partitions_ms` | partition 밖 작업과 계측 overhead 등 |

`profile`의 시간은 별도의 한 호출에서 얻은 값이며 `median_wall_ms`와 합산하지 않는다.
Resident 경로의 native H2D 구간에는 전송 생략 판정과 round 시작 시 프로그램 적재 비용도 들어갈 수 있다.
Weight tile packing은 반복 실행에도 남아 있으며, weight H2D 0을 weight 관련 CPU 비용 0으로 해석하지 않는다.

## PIMLLM 비교 조건

**내용**
로컬 `PGEMMlib/PIMLLM_artifact/scripts/profile_attention_tvm.py`는 INT8 GEMM과 재양자화를 포함한다.
이번 FP32 attention 결과는 dtype과 연산 범위를 맞추기 전에는 동일 조건의 speedup으로 계산하지 않는다.
JSON에 shape, dtype, fusion 여부, target 설정, thread 수, 준비/반복 실행 시간을 남긴다.

**근거**
- 해당 TVM script의 `attention_module()`은 INT8 입력, INT32 GEMM 출력, projection/probability 재양자화를 사용한다.
- 해당 script는 QKV 포함 여부에 따라 native fusion 경로를 사용할 수 있으므로 실제 graph 범위를 확인해야 한다.
- 같은 warm-up 횟수라도 weight 전송 포함 여부와 전체 호출 경계가 다르면 다른 지표다.

**예시**
| 비교 조건 | 이번 실험 |
|---|---|
| 연산 범위 | 별도 Q/K/V projection + scale + softmax + PV |
| 수치 경로 | FP32, 재양자화 없음 |
| Attention mask | 없음 |
| 준비 비용 | 별도 보고 |
| 반복 실행의 weight H2D | 0 검사 |
| 전체 시간 | `block.run()` wall time, 3회 median |
| 반복 실행 | 동일한 weight, 서로 다른 두 activation 입력 교대 |

## 적용 범위와 제한

**내용**
입력은 export 당시의 positional pytree 구조를 유지해야 한다. Shape·stride·dtype·device가 바뀌면 오류를 낸다.
고정 state의 Tensor 교체, version, data pointer, shape/stride 변경을 감지하면 새 block 준비를 요구한다.
각 GEMM은 두 입력과 한 출력의 현재 UPMEM artifact 형태를 사용한다.

**근거**
- Buffers나 user input을 수정하는 exported graph는 weight 불변 조건을 만족하지 않는다.
- Weight는 immutable이어야 한다. Tensor version을 갱신하지 않는 외부 메모리 쓰기는 사용하지 않는다.
- 자동 eviction, 실행 중 자동 재준비, CPU fallback, CPU/PIM 비동기 파이프라이닝은 포함하지 않는다.
- 원형/전치형의 성능 우위는 실제 장비의 전체 block 측정으로 판단한다.

**예시**
| 상황 | 동작 |
|---|---|
| 고정 weight를 여러 round에서 사용 | 같은 resident window 참조 |
| QK/PV처럼 두 입력이 모두 동적 | INPUT 안의 겹치지 않는 두 슬롯 사용 |
| 같은 activation, 같은 packed byte | 같은 호출 안에서 activation H2D 생략 |
| Weight/config 변경 | 기존 block을 닫고 명시적으로 다시 준비 |
| 전체 weight와 scratch가 MRAM 초과 | 준비 단계에서 오류 |
| 컴파일 또는 SDK 실행 실패 | 실패를 숨기지 않고 전달 |
