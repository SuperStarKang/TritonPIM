# PIM Common IR AutoTuner

이 autotuner는 PIM Common IR에 실제로 기록되는 계층과 mapping만 탐색한다.

```text
Triton Block       [block_m, block_n, block_k]
Block Group        [group_m, group_n, group_k]
Execution Tile     [tile_m, tile_n, tile_k]
Device mapping     [p_m, p_n, p_k]
```

`block_*`는 Triton kernel constexpr이며 Common IR의 `#pim.block`으로 복원된다.
`group_*`, `tile_*`, `p_*`는 compiler option으로 전달되어 각각
`#pim.block_group`, `#pim.execution_tile`, `#pim.device_mapping`이 된다.

## Target 정보

compiler와 autotuner는 `backend/target.py`의 동일한 타입과 builder를 사용한다.

```text
PIMTargetSpec
  common_profile: PIMTargetProfile
  runtime_config: UPMEMRuntimeConfig | AiMXRuntimeConfig
```

`PIMTargetProfile`은 Common IR의 일곱 필드와 정확히 같다. `nr_tasklets`,
`host_threads`, `batch`, `base_fpga`는 target runtime config이며 Common IR 계층
변수가 아니다.

초기 후보의 첫 `P`는 problem grid가 허용하는 범위에서
`p_m*p_n*p_k`가 `num_units`에 가장 가까운 값이다. 이후에는 축별 seed와
random/LFBO 후보를 측정하므로 최대 device 사용이 최종
선택으로 고정되지 않는다.
`(1,1,1)`은 강제하지 않고 problem grid가 unit 하나만 허용할 때만 첫
seed로 나온다.

UPMEM은 split-K를 충분히 비교할 수 있도록 유효한 각 `p_k` 단계마다
`p_m*p_n*p_k`가 최대가 되는 `p_m/p_n` 조합을 초기 seed에 포함한다. 따라서
적은 DPU를 쓰는 K-only 후보가 아니라, 비슷한 전체 device parallelism을
유지한 상태에서 `p_k` 변화의 실제 실행 시간을 비교한다. LFBO가 partition을
변경할 때는 같은 축의 Execution Tile만 새 device plane 안에 들어오도록
줄인 뒤 constraint를 검사한다.

## 탐색 흐름

```text
PIMProblem
  -> Common IR config seed/random candidate
  -> static constraint
  -> Common IR 값에서 feature 파생
  -> Random Forest LFBO scoring
  -> 실제 target lowering 및 실행
  -> end-to-end 시간 측정
  -> cache
```

최종 objective는 실제 end-to-end 실행시간이다. utilization, transfer,
replication, reduction, padding, memory pressure는 후보를 설명하기 위해 Common IR
값에서 계산하는 내부 feature일 뿐 compiler option이나 IR attribute로 저장하지 않는다.

## 사용 예

```python
from autotuner import (
    PIMBenchmarkResult, PIMProblem, SearchBudget,
    build_config_space, target_spec, tune,
)

problem = PIMProblem(M=8192, N=8192, K=128)
target = target_spec("upmem")
space = build_config_space(problem, target)

def benchmark(problem, target, config):
    elapsed_ms = run_kernel(config)
    return PIMBenchmarkResult(total_ms=elapsed_ms)

result = tune(problem, target, benchmark, space=space,
              budget=SearchBudget(max_benchmarks=30))
print(result.best.config)
```

UPMEM 직접 예제는 실제 장비 크기를 명시할 수 있다.

```bash
python examples/autotune_gemm.py --m 256 --n 256 --k 256 \
  --num-units 256 --lanes-per-core 16 --nr-tasklets 16
```

`num_units`는 profile 상한이고, 각 후보의 실제 DPU 수는
`p_m*p_n*p_k`다. `nr_tasklets=0`은 `lanes_per_core`를 모두 사용한다.

현재 `PIMProblem`과 byte 계산은 GEMM을 지원한다. 다른 연산을 추가할 때도 별도
distribution/reduction/residency policy를 Config에 넣지 않고, 해당 연산의
Block–Group–Tile과 access dependency에서 feature를 계산한다.
