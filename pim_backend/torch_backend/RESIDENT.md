## Block 단위 weight-stationary

**내용**
`prepare_block()`은 exported FX graph의 고정 weight를 분석하고, 각 weight의 조각을
동일한 전체 DPU 집합에 미리 전송한다. `run()`은 공용 입력·출력 scratch를 사용한다.
대상은 정적 shape의 CPU int8 `aten._int_mm`과 int32 출력이며, 다른 연산은 CPU Inductor로 실행한다.

**근거**
- Torch lowering, Common/UPMEM pass, host/device 생성기는 변경하지 않는다.
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
각 `_int_mm` node 이름에 `ResidentGemmConfig`를 지정한다.
`padded_shape`은 전치하기 전 논리 축 순서 `(M,N,K)`이며, padding이 필요 없으면 원래 shape를 쓴다.
전치형은 `B.T @ A.T`를 계산한 뒤 출력의 전치와 slice를 복원한다.

**근거**
- 컴파일 실패를 잡아 다른 방향·padding·config로 바꾸지 않는다. 예외는 호출자에게 전달한다.
- 모든 config의 `p_m*p_n*p_k`와 hardware profile의 `num_units`가 `dpu_count`와 같아야 한다.
- `Block Group M % nr_tasklets == 0`과 기존 `ConstraintEngine` 조건을 확인한다.
- 모든 GEMM의 weight window 합과 최대 공용 scratch가 MRAM 예산 안에 있어야 한다.

**예시**
```python
from autotuner import PIMConfig
from torch_backend import PIMTorchConfig, ResidentGemmConfig, prepare_block, target_spec

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
    node.name: config
    for node in exported.graph_module.graph.nodes
    if node.op == "call_function" and node.target == torch.ops.aten._int_mm.default
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
  FX 고정 weight/view 평가
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
