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

모든 후보는 Common IR과 같은 식으로 해석한다.

```text
block_group_extent[d] = block[d] * group[d]
total_groups[d]       = ceildiv(problem[d], block_group_extent[d])
execution_tiles[d]    = ceildiv(total_groups[d], tile[d])
rounds[d]             = ceildiv(execution_tiles[d], p[d])
```

K도 예외가 아니다. `p_k`는 같은 round에서 semantic K Block Group을 나눌
device 수이고, `tile_k`는 device 하나가 round 한 번에 처리할 K Block Group
수다. Source K plane이나 `ceildiv(K, p_k)` domain은 만들지 않는다.

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

Reduction 연산은 semantic K 병렬성을 충분히 비교하도록 유효한 각 `p_k` 단계마다
`p_m*p_n*p_k`가 최대가 되는 `p_m/p_n` 조합을 초기 seed에 포함한다. 따라서
적은 DPU를 쓰는 K-only 후보가 아니라, 비슷한 전체 device parallelism을
유지한 상태에서 `p_k` 변화의 실제 실행 시간을 비교한다. LFBO는 `p_k`와
`tile_k`를 독립적으로 변경하며, 변경된 조합의 round 수를 위 식으로 다시 계산한다.

## 탐색 흐름

```text
PIMProblem
  -> compute op와 pim.place access dependency 분석
  -> Common IR config seed/random candidate
  -> static constraint
  -> Common IR 값에서 feature 파생
  -> Random Forest LFBO scoring
  -> 실제 target lowering 및 실행
  -> end-to-end 시간 측정
  -> cache
```

최종 objective는 실제 end-to-end 실행시간이다. 내부 feature의 역할은 계층별
trade-off를 Random Forest에 설명하는 것이다.

- Block Group: compute-domain 크기 대비 local buffer byte와 local-memory pressure
- Execution Tile: compute-domain 크기 대비 resident byte와 stage/H2D/D2H byte
- Multi-device scheduling: active unit 수, useful/scheduled tile 비율,
  M/N/K round 수, K contribution 결합 byte

이 값들은 후보 선택에만 쓰며 compiler option이나 IR attribute를 새로 만들지 않는다.
실행 시간은 세 번 측정한 median이 기본이고 UPMEM에서는
`stage/h2d/kernel/d2h/combine` median도 후보 로그와 cache에 기록한다.
기본 cache는 `TRITON_CACHE_DIR/pim_autotune_cache.json`에 저장한다. 저장할 때
같은 디렉터리의 임시 파일을 완성한 뒤 교체하므로 `/tmp`를 사용하지 않으며,
중단되더라도 기존 cache를 먼저 비우지 않는다. 0-byte 파일은 빈 cache로 취급하고
내용이 있는 잘못된 JSON은 해당 경로를 포함한 오류로 보고한다.

`PIMProblem`은 연산별 policy를 갖지 않는다. `compute_kind`는 PIM compute op의
종류를 식별하는 의미 값이고, 각 `PIMAccess`는 unique buffer의 direction, dtype,
logical axis dependency를 나타낸다. 입력 access가 사용하지 않는 축은 Common IR의
`replicated_over`와 같은 의미다. Cache와 constraint/feature는 이 access 구성을
포함하므로 서로 다른 연산 결과를 섞지 않는다.

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

같은 tuner에 다른 연산을 전달할 때도 `PIMConfig` 필드는 바뀌지 않는다.

```python
elementwise = PIMProblem.elementwise(
    M=1 << 20, input_dtypes=("float32", "float32"))
matvec = PIMProblem.matvec(M=8192, K=4096)
reduction = PIMProblem.reduction(M=8192, K=4096)
convolution = PIMProblem.convolution(
    spatial=56 * 56, output_channels=256,
    reduction_extent=3 * 3 * 128)
```

Fused operator와 gather/scatter는 PIM Common IR에서 unique `pim.place`마다
`PIMAccess`를 하나씩 만들면 된다. Irregular access는 `irregular=True`로 cache에서
구분한다. Feature의 byte 수는 placement allocation 범위를 사용하고, 최종 선택은
항상 실제 target 실행시간으로 결정한다.

UPMEM 직접 예제는 실제 장비 크기를 명시할 수 있다.

```bash
python examples/autotune_gemm.py --m 256 --n 256 --k 256 \
  --num-units 256 --lanes-per-core 16 --nr-tasklets 16
```

`num_units`는 profile 상한이고, 각 후보의 실제 DPU 수는
`p_m*p_n*p_k`다. `nr_tasklets=0`은 `lanes_per_core`를 모두 사용한다.

현재 공통 problem/footprint/transfer/reduction 분석은 elementwise, GEMV/GEMM,
reduction, flattened convolution, fused placement와 irregular access를 표현한다.
실제로 실행하려면 해당 연산을 PIM Common IR로 만드는 frontend lowering과 target
compute lowering이 존재해야 한다. Autotuner는 없는 target 연산 구현을 대신 만들지
않는다.
