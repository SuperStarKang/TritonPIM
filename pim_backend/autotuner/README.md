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

Heuristic seed는 `PIMPerformanceProfile`에 저장한 **단일 unit 실측 기록**으로
고른다. 이 자료는 compiler의 자원 한도나 autotune 결과 cache와 별개다.
GEMM은 target의 Block 후보마다 Group의 K 방향을 측정하고, 처리량이 가장 좋은
Group을 고정해 Execution Tile의 K 방향을 측정한다. 실제 문제에 맞는 실측
Block/Group부터 타일 후보를 만들고 target 제약과 호출자의 추가 제약을 검사한다.

| 순서 | 선택 기준 |
|---|---|
| 중심 | 한 타일의 평균 유효 작업량이 포화점 이상인 후보 중 가장 많은 unit을 사용하는 후보 |
| 병렬성 증가 | 중심보다 unit 수가 바로 많은 후보 |
| 병렬성 감소 | 중심보다 unit 수가 바로 적은 후보 |
| 최대 병렬 | 유효 후보 중 가장 많은 unit을 사용하는 후보. 이미 선택했다면 추가하지 않음 |

Tile 측정의 안정 구간 처리량으로 Block 후보의 순서를 정한다. 여러 Block/Group
조합의 중심을 먼저 배치하고 그다음 주변 후보를 배치한다.
`max_ladder_steps`는 조합당 후보 수 상한이며 기본값은 4다. 용량이 가장 큰 타일이나
1-unit 후보를 의무적으로 넣지 않는다. 모든 후보가 포화점 아래이면 가장 많은
유효 작업을 처리하는 후보를 중심으로 삼고 최대 병렬 후보도 비교한다. 합법적인
배치가 하나의 unit만 허용하면 seed도 하나의 unit을 사용한다.

실제 문제가 작아 profiling에서 선택한 Group이 들어가지 않으면, 그 문제에 맞는
측정 Group 중 처리량이 가장 좋은 것을 선택한다. 이때 작업량 기준은 같은 Block에서
실측한 Tile 포화점이다. 더 작은 Group의 시간을 만들어 내지 않는다.
포화점은 seed 선택용 근사 기준이다. 동일한
Block/Group에서도 타일의 축 비율과 데이터 재사용이 달라지면 처리량이 달라질 수 있다.
최종 선택은 기존 end-to-end 실측으로 결정한다. LFBO는 전체 타일 범위를 계속
탐색하며, 외부 `initial`, 유사 cache 후보, structured 후보에도 각각의 규칙이 적용된다.
따라서 이 경로에서 1-unit 후보가 나오는 것은 허용된다. 로그의 `source=saturation`,
`source=initial`, `source=warm-cache`, `source=structured`, `source=LFBO`로 출처를 구분한다.

초기 exploration 후보는 Block/Group/Tile/P를 서로 독립적으로 random sampling하지
않는다. Block과 Group을 고른 뒤 유효한 `P`와 목표 scheduling round를 먼저 정하고,
그에 맞는 Tile을 역산한다. Tile은 해당 축의 total Block Group 수를 넘을 수 없고,
`p_m/p_n/p_k`도 각 축의 execution tile 수를 넘을 수 없다. 이 exploration 후보는
schedule utilization 50% 이상, padding 50% 이하, 전체 scheduling round 4 이하인
경우만 실제 측정한다. Reduction 초기 후보는 output contribution이 64개를 넘지
않도록 제한하며, 이 경계 밖은 LFBO가 측정 결과를 바탕으로 확장한다. 이 초기 안전
경계들은 `PIMConfigSpace` 또는 `space_overrides`로 바꿀 수 있다.

장치가 실제로 지원하는 분할 축만 타일 축소에 사용한다. 예를 들어 AiMX의 K 분할은
제외되며 UPMEM의 K 분할은 mapping 제약 안에서 선택한다. LFBO는 `p_k`와
`tile_k`를 변경하며, 변경된 조합의 round 수를 Common IR 계층에서 다시 계산한다.

## 단일 unit 자료 생성

`autotuner/gemm.py`의 GEMM 소스를 UPMEM·AiMX 예제와 profiler가 함께 사용한다.
`profile_gemm(target, output_dir=...)`는 target 정보만으로 기본 dtype과 측정 후보를
구성한다. 일반 autotuning에서는 `ensure_gemm_profile` callback이 기존 기록을
검사하고, 기록이 없으면 먼저 profiling한다. 완료된 tuning cache가 있으면 profiling도
생략한다. 명시적으로 제공한 자료가 없거나 호환되지 않으면 해당 오류를 전달한다.

### Block 후보의 근거

`TargetModel.gemm_profile_spec`은 `PIMGemmProfileSpec`으로 기본 Block, 기본 Group,
입출력 dtype을 제공한다. 공통 코드는 기본 행 크기의 1·2·4배와 열 크기의 1·2배를
조합한 여섯 후보를 검사한다. 누산 방향의 Block 크기는 유지한다. 이는 측정 후보이며
최적 성능을 미리 가정한 값이 아니다. 유효하지 않은 후보는 실행 전에 제약 검사로 제외한다.

| target | 출발점 |
|---|---|
| UPMEM | 활성 tasklet마다 출력 한 행, SingleUtil의 열 32·누산 16. 16 tasklet이면 기본 Block `(16,32,16)` |
| AiMX | 현재 lowering의 batch 행·4096을 batch로 나눈 열·256을 batch로 나눈 누산 눈금. batch 8이면 `(8,512,32)` |

Triton Block은 2의 거듭제곱이어야 한다. UPMEM tasklet 수에 홀수 인자가 있으면
그 인자를 기본 Group의 행 방향에 둔다. 예를 들어 11 tasklet은 Block 행 1과
Group 행 11에서 시작한다. 각 측정 곡선의 실제 출력 행·열 크기는 고정한다.

### Group 선택과 Tile 포화 측정

1. device와 device partition을 모두 하나로 제한한다.
2. 각 Block 후보에서 Tile `(1,1,1)`을 고정하고 Group의 K 개수를 두 배씩 늘린다.
3. Group 처리량이 안정되거나, 다음 크기가 local·bank·할당 한도를 넘으면 Group 측정을 끝낸다.
4. 측정 Group 중 동기 실행시간 기준 처리량이 가장 높은 것을 선택한다.
5. 그 Group을 고정하고 Tile의 K 개수를 두 배씩 늘린다. local 계산 단위는 유지되고 상주 데이터가 증가한다.
6. Tile 처리량이 안정되면 포화점을 기록한다. 용량이나 지정한 상한 때문에 먼저 끝나면 오류를 전달한다.

안정성은 연속 세 크기의 처리량이 중앙값의 5% 범위인지 보고, 더 큰 크기 한 점으로
확인한다. 확인 후 마지막 세 점도 같은 조건을 만족해야 한다. 앞선 처리량보다 크게
낮아진 구간을 포화 구간으로 인정하지 않는다. 안정된 처리량의 95% 이상을 이후에도
유지하는 최소 측정 작업량을 포화점으로 저장한다. 용량 한도를 포화점으로 대체하지 않는다.

실행마다 컴파일을 포함하는 준비 호출을 제외하고 이후 반복의 중앙값을 사용한다.
ones 입력의 결과도 검사한다. 실제 device 호출 시간은 UPMEM `kernel`, AiMX `invoke`이며
host 입력·출력 전송은 제외한다. 최종 tuning objective는 전송과 결합을 포함한 전체 시간이다.

| 기록 | 의미 |
|---|---|
| `single_unit.csv` | Group·Tile의 모든 측정점, Block/Group/Tile, 실제 문제 크기, 동기 시간과 처리량 |
| `group_curves` | Block마다 측정한 Group과 종료 이유. 작은 실제 문제에 맞는 Group을 다시 고를 때도 사용 |
| `curves` | 선택한 Group을 고정한 Tile 포화 곡선 |
| `local_bytes`, `resident_bytes` | 정적 모델의 Group 논리 버퍼 합과 padding·부가 할당을 포함한 상주량 추정 |
| `performance.json` | target·runtime·dtype·access·profiling 정책·커널 버전과 위 측정 기록 |

CSV는 각 측정 직후 저장한다. 포화 검증에 실패해도 이미 수집한 CSV는 남는다.
JSON schema는 `pim-single-unit-saturation-v2`이며 이전 schema는 재측정해야 한다.
unit 수만 달라지면 단일-device 자료를 공유할 수 있다. tasklet·batch·dtype 등이 다르면
별도의 결과 폴더로 측정한다. SingleUtil의 반복 횟수를 Group이나 Tile 값으로 복사하지 않는다.

### 실행

기존 하드웨어 실행 환경에서 사용자가 실행한다. 아래 기준 경로는
`/home/dlrkdals/TritonPIM`이다. 첫 명령 하나로 필요한 profiling과 autotuning을 이어서 수행한다.

```bash
cd /home/dlrkdals/TritonPIM

python triton_shared/pim_backend/examples/upmem/autotune_gemm.py \
  --m 1024 --n 1024 --k 1024 --num-units 2048 --nr-tasklets 16 \
  --reps 5 --max-benchmarks 30 \
  --output-dir workspace/Experiment/Autotuner/upmem_gemm

python triton_shared/pim_backend/examples/silent_pim/autotune_gemm.py \
  --m 1024 --n 4096 --k 4096 --num-units 1 --batch 8 \
  --reps 5 --max-benchmarks 30 \
  --output-dir workspace/Experiment/Autotuner/aimx_gemm
```

profiling만 새로 수행하려면 같은 명령에 `--profile-only`를 붙인다. `--retune`은
저장된 성능 자료를 사용해 tuning을 다시 한다. `--performance`로 기록을 직접 지정할 수도 있다.
기존 `--calibrate-performance M N K`는 선택적으로 측정 상한을 지정하고 종료하는 모드다.
상한이 작아 포화점을 확인할 수 없으면 오류를 낸다. Profiling 시간·호출 수는
`--max-seconds`·`--max-benchmarks`가 제한하는 tuning 탐색 예산에 포함하지 않는다.

```text
workspace/Experiment/Autotuner/upmem_gemm/
  single_unit.csv
  performance.json
  autotune_cache.json
  temporary/
  triton_cache/
  triton_dump/
  triton_override/
```

예제는 Python과 자식 compiler의 임시 파일, Triton cache·dump·override 경로를
위 폴더 아래로 지정한다. 결과·성능 자료·명시적 cache 경로는 프로젝트 내부만 허용한다.
이는 실행 프로세스에만 적용하며 셸 설정 파일이나 시스템 환경을 변경하지 않는다.

GEMV 예제도 자료가 없으면 실제 GEMV 커널을 먼저 측정하며 `--profile-only`를 지원한다.
GEMV·elementwise의 측정은 해당 연산의 기존 측정 범위를 사용한다. GEMM 기준 자료를
다른 연산이나 dtype의 실측값처럼 사용하지 않는다. Torch 경로의 calibration callback은
같은 Group 선택·Tile 측정 절차에서 Torch template을 실행하며, 커널 버전이 달라 직접
GEMM 예제 자료와 섞지 않는다.

새 PIM은 target adapter에 `gemm_profile_spec`, 자원·mapping 제약,
`launch_timing_name`을 제공하고 공통 GEMM을 lowering하면 같은 profiler를 사용한다.

회귀 테스트는 사용자가 저장소 루트에서 실행한다. 하드웨어를 사용하지 않고 합성 자료로
검사하며, 테스트 임시 파일도 프로젝트 결과 디렉터리 안에 만든다.

```bash
cd /home/dlrkdals/TritonPIM/triton_shared
python -m unittest pim_backend.autotuner.test_saturation_seeds \
  pim_backend.autotuner.test_seed_frontier pim_backend.autotuner.test_aimx_capacity
```

## 탐색 흐름

```text
PIMProblem
  -> compute op와 pim.place access dependency 분석
  -> Common IR config seed/structured candidate
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
`TIMING_STAGES`의 행동별 median도 후보 로그와 cache에 기록한다.
UPMEM timing v3는 입력 layout, 버퍼 할당·초기화, 출력 layout과 `reduction`을 분리한다.
라이브러리 기본 cache는 프로젝트의 `workspace/Experiment/Autotuner/autotune_cache.json`이다.
예제는 각 `--output-dir`의 `autotune_cache.json`을 사용한다. 저장할 때
같은 디렉터리의 임시 파일을 완성한 뒤 교체하며,
중단되더라도 기존 cache를 먼저 비우지 않는다. 0-byte 파일은 빈 cache로 취급하고
내용이 있는 잘못된 JSON은 해당 경로를 포함한 오류로 보고한다.

AiMX의 컴파일 전 용량 검사는 Group마다 적용되는 물리 padding을 계산한다.
`batch=b`일 때 M은 `b`, N은 `4096/b`, K는 `256`의 배수로 올림하며,
A의 입력 pitch만 `4096` 원소 정렬을 유지한다. Execution Tile의 resident
A/B/C는 Group 개수의 곱으로 계산한다. 순수 GEMM의 Arel은 현재 M/N Group에
속한 모든 K-Group을 유지하므로 `tile_k`배가 필요하다. MAC은 K-Group 경계에서
누적을 끊지 않고, 출력 microtile마다 마지막에 한 번만 저장한다.
descriptor 추정은 Group별 Arel 준비, `256/batch` 단위 MAC, 마지막 K padding,
출력 microtile별 drain을 각각 센다. 저장용 K padding과 MAC padding은 다르다.

이 값은 GEMM의 **하한 추정**이다. Common IR의 scratch, 상수, copy/add,
fusion에 필요한 전체 descriptor는 lowering에서 모두 합산해 최종 용량을 검사한다.
따라서 static constraint를 통과해도 lowering이 용량 초과 오류를 낼 수 있으며,
그 오류를 무시하거나 다른 config로 대체하지 않는다.
측정 결과의 `physical_allocations`에는 lowering이 생성한 device별 전체 할당 목록을
기록하고, 측정된 후보의 `unit_memory_pressure`에는 이 목록을 사용한다.
배치가 달라졌으므로 이전 버전의 컴파일·autotuning cache는 새 키로 구분한다.

관련 Python 회귀 테스트는 저장소 루트에서 다음과 같이 실행한다.

```sh
python -m unittest pim_backend.autotuner.test_aimx_capacity
```

`PIMProblem`은 연산별 policy를 갖지 않는다. `compute_kind`는 PIM compute op의
종류를 식별하는 의미 값이고, 각 `PIMAccess`는 unique buffer의 direction, dtype,
logical axis dependency를 나타낸다. 입력 access가 사용하지 않는 축은 Common IR의
`shared_over`와 같은 의미다. Cache와 constraint/feature는 이 access 구성을
포함하므로 서로 다른 연산 결과를 섞지 않는다.

## 사용 예

```python
from autotuner import (
    PIMBenchmarkResult, PIMPerformanceProfile, PIMProblem, SearchBudget,
    build_config_space, target_spec, tune,
)

problem = PIMProblem(M=8192, N=8192, K=128)
target = target_spec("upmem")
space = build_config_space(problem, target)
performance = PIMPerformanceProfile.load("performance.json", kernel_version="my-kernel-v1")

def benchmark(problem, target, config):
    elapsed_ms = run_kernel(config)
    return PIMBenchmarkResult(total_ms=elapsed_ms)

result = tune(problem, target, benchmark, space=space,
              performance=performance, kernel_version="my-kernel-v1",
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
python pim_backend/examples/upmem/autotune_gemm.py --m 256 --n 256 --k 256 \
  --num-units 256 --lanes-per-core 16 --nr-tasklets 16 --performance performance.json
```

`num_units`는 profile 상한이고, 각 후보의 실제 DPU 수는
`p_m*p_n*p_k`다. `nr_tasklets=0`은 `lanes_per_core`를 모두 사용한다.

현재 공통 problem/footprint/transfer/reduction 분석은 elementwise, GEMV/GEMM,
reduction, flattened convolution, fused placement와 irregular access를 표현한다.
실제로 실행하려면 해당 연산을 PIM Common IR로 만드는 frontend lowering과 target
compute lowering이 존재해야 한다. Autotuner는 없는 target 연산 구현을 대신 만들지
않는다.
