# torch.compile → PIM 연동

`torch.compile`된 그래프에서 `aten._int_mm`만 PIM split-K template로 실행하고,
나머지 연산은 기존 CPU Inductor(C++/OpenMP)로 실행하는 계층의 설계다.

기준 트리는 `triton_shared`이고, 경로는 전부 그 아래의 상대 경로다.

---

## 0. 한눈에

| 계층 | 무엇을 하는가 | 어디 |
| --- | --- | --- |
| Triton backend | TTIR → TTShared IR → PIM IR → artifact, launch | `pim_backend/backend/` |
| AutoTuner | problem → config. torch를 모른다 | `pim_backend/autotuner/` |
| torch 통합 | `aten._int_mm` lowering, scheduling, runtime patch | `pim_backend/torch_backend/` |

의존 방향은 한쪽이다.

```
torch_backend  →  autotuner  →  backend
```

`autotuner`는 torch를 모르고, `backend`는 둘 다 모른다. `torch_backend`는
`autotuner`를 **읽기만** 한다 — 고치지 않는다.

tensor device는 처음부터 끝까지 CPU다. **PIM은 tensor device가 아니라 template의
compile / execution backend다.**

---

## 1. 세 단계

```
① 튜닝    오프라인, 하드웨어 필요   problem → config, cache에 저장
② 컴파일  torch.compile 1회        FX → Inductor IR → 커널들 → PIM artifact
③ 실행    호출마다                 cpp → PIM launch → cpp
```

### 1.1 튜닝 — `examples/autotune_torch.py`

```
PIMProblem(M,N,K,int8→int32)              shape에서
target_profile("upmem")                   autotuner/targets.py (2048 units, 16 lanes)
build_config_space(problem, hw)           P_K 전략은 target이 정한다
        │
tune(problem, hw, benchmark_fn, cache=cache, budget=...)
        ├─ heuristic_seeds                초기 population
        ├─ ConstraintEngine.check         무효 config는 재기 전에 버린다
        ├─ PIMPlanner.build               config → plan
        │
        └─ benchmark_fn(problem, hw, config, plan)      ← torch_backend가 준다
              torch._dynamo.reset()
              torch.compile(fn, backend=PIMInductorBackend(
                  PIMTorchConfig(config=config)))       ← config 주입
              out = compiled(a, b)                      # compile + PIM 빌드
              median(reps)                              # PIM launch + CPU 합
              표본 정답 검증
        │
        ├─ RandomForest로 good/bad 학습 → semantic_mutate로 다음 세대
        └─ cache 저장 (~/.pim_autotune_cache.json)
```

**튜닝은 lowering 안에서 일어나지 않는다.** Inductor lowering은 fake tensor를
다루므로 그 안에는 잴 입력이 없다. 하지만 튜닝이 거기서 일어날 이유도 없다 —
바깥에는 실제 텐서가 있다. config 하나마다 `torch.compile`을 새로 만들어 재므로
측정 대상이 실제 실행 경로 그대로이고, PIM launch와 CPU partial 합이 모두 들어간다.

compile/실행 실패는 예외를 삼켜 `compile_status="error"`로 돌린다. AutoTuner
설계 §35의 BAD sample이 되어 탐색이 그것도 학습한다.

`examples/autotune_gemm.py`와 다른 점: 그쪽은 `examples/split_k.py`의 커널을
직접 launch한 것을 잰다. torch 경로가 도는 것은 Inductor template이라 mask가 없고
(§4.3), shape/stride를 런타임 인자로 받고(§4.2), partial 합이 별도 cpp 커널이다.
**재는 대상이 다르다.**

### 1.2 컴파일

```
torch.compile(f, backend=PIMInductorBackend(cfg), fullgraph=True, dynamic=False)
        │
   ① TorchDynamo → FX graph                aten._int_mm이 그대로 남는다
        │
   ② PIMInductorBackend.__call__           torch_backend/backend.py
        set_driver_to_backend("pim")        §3.1 — 여기서 골라야 한다
        runtime_compat.install()            §3
        with PIMIntMMLowering(cfg),         lowerings[aten._int_mm] 교체
             PIMCombinedSchedulingInstalled(),   cpu scheduling 교체
             inductor_config.patch(...):    compile_threads=1, fx_graph_cache=False
                 compile_fx(gm, inputs)
        │
   ③ lowering — _int_mm 하나마다            torch_backend/lowering.py
        mm_args()                           실현된 mat1/mat2, M/N/K
        validate_pim_int_mm()               → PIMProblem (2-D, int8, CPU, static)
        resolve_execution_plan()            §2
        _zeroed_partial()                   zeros[P_K,M,N] int32 (CPU Cpp)
        template.generate(...)              §4
        lowerings[aten.sum](partial, 0)     CPU Cpp
        │
   ④ scheduling                             torch_backend/scheduling.py
        PIM template → TritonScheduling
        나머지        → CppScheduling
        │
   ⑤ wrapper codegen                        cpp_fused_* + triton_tem_*
        │
   ⑥ Triton 커널 컴파일                     §3.2
        compile_backend == "pim" 이면
          target = GPUTarget("pim",0,0)     tensor device(cpu)가 아니다
          options += backend_options
          triton.compile → TTIR → … → {host.cpp, device.c}
```

### 1.3 실행

```
cpp_fused_0(...)                       transpose + partial zeros
triton_tem_1.run(A, B, PARTIAL, ks0..ksN, gx, gy, gz, stream=0)
      └→ PIMLauncher.__call__          pim_backend/backend/driver.py
           번들(JSON: host.cpp/device.c/lanes) 해제
           dpu-upmem-dpurte-clang → device.elf     첫 회만, 캐시된다
           g++ + libdpu           → host.so
           dlopen → ctypes로 <kernel>_host(A,B,C, gx,gy,gz)
cpp_fused_2(...)                       partial 합 + 그다음 연산 + 다음 zeros
```

---

## 2. config를 고르는 세 경로

`torch_backend/plan.py::resolve_execution_plan`

```
PIMTorchConfig(config=PIMConfig(...))    ① 그대로 쓴다      튜닝 중
PIMTorchConfig(autotune_cache_path=...)  ② 실측된 best      운영. miss는 fail-fast
PIMTorchConfig(split_k=4)                ③ heuristic seed   정답 확인용
```

①②는 **보정하지 않는다.** 실측 결과를 lowering이 몰래 바꾸면 잰 것과 기록한 것이
달라진다. 셋 다 `ConstraintEngine.check()`와 `check_template_support()`는 거친다.

③에만 적용되는 보정이 둘 있다.

- **`_fit_k_axis`** — seed의 execution 유도는 device 활용 단위에서 나오므로 K가
  작은 shape에서 `tile_k`가 K 몫(설계 §16의 spatial K plane)을 넘긴다. 예: K=128,
  P_K=4면 plane이 32인데 seed는 tile_k=256을 낸다. 줄이는 방향으로만 맞춘다.
  성능이 아니라 **유효성** 보정이다 — 안 하면 Constraint 4가 거부한다.
- **`pin_row_partitions`** — P_M=1 고정. 처음에는 P_M>1이 미검증이라 두었으나
  탐색이 P=(32,32,2)와 (16,32,4)를 정답까지 통과시켰으므로 근거가 약해졌다.

③은 성능 숫자를 내는 용도가 아니다. seed는 shape를 다 보지 않는다 — `targets.py`의
`utilization_tile_m = 131072`이 어떤 M보다도 커서 `tile_m`이 항상 M까지 자라고,
그러면 `e_m = ceil(M/tile_m) = 1`이라 M축 device 병렬이 0이 된다. 그 축을 흔드는
것은 탐색의 일이다.

### 2.1 값이 한 군데에만 있게 한다

| 무엇 | 어디 |
| --- | --- |
| `PIMProblem` (M,N,K,dtype,layout) | `lowering.validate_pim_int_mm` |
| `PIMHardwareProfile` | `autotuner/targets.py` |
| `PIMConfig` (block/group/execution, p_m/p_n/p_k, policy) | seed 또는 cache |
| `PIMPlan` (active_device_grid, rounds, transfer/reduction) | `PIMPlanner.build` |
| config가 이 shape에서 성립하는가 | `ConstraintEngine.check` |
| `backend_options` dict | `autotuner.decorator.pim_launch_options` |

`torch_backend`가 이것들을 복제하지 않는 것이 중요하다. 복제하면 (a) `tile =
block × group × execution` 관계를 손으로 검사해야 하고, (b) `tile_m` 하나에서
`(block, group, execution)`을 복원할 수 없어 config를 `ConstraintEngine`이나
`PIMPlanner`에 다시 넣을 수 없고, (c) shape에 의존하는 조건(`p_n ≤ ceil(N/tile_n)`)을
config 단독으로 검사할 수 없다.

`backend_options`의 본체가 `pim_launch_options`인 것도 같은 이유다 — 손으로 띄우는
경로와 인덕터 경로가 같은 plan에서 같은 문자열을 만들어야 두 경로가 갈라지지 않는다.
그 helper가 다루지 않는 `lane_mode` / `host_threads`만 torch 쪽에서 얹는다.

---

## 3. torch runtime 수정 두 곳

`torch_backend/runtime_compat.py`. site-packages를 고치지 않고 monkeypatch한다 —
torch 업데이트마다 patch 충돌을 관리하지 않아도 되고, 이 패키지를 import하지 않은
프로세스는 stock torch 그대로다.

대신 조건이 붙는다. **async compile worker(subprocess)에는 이 patch가 없다.**
그래서 `PIMInductorBackend`가 `compile_threads=1`로 묶는다.

### 3.1 `set_driver_to_backend(name)`

backend를 이름으로 명시 선택한다. stock `set_driver_to_gpu()`는 backends를 훑어
`is_active()`가 True인 것을 스스로 고르는데, 그 방식이면 PIM이 "자동 탐지되는
backend"가 되어 PIM을 쓰지 않는 torch.compile까지 끌고 간다. 그래서
`PIMDriver.is_active()`는 항상 `False`다.

**호출 시점이 compile_fx 진입 직전이어야 한다.** driver를 처음 요구하는 곳은
컴파일 단계가 아니라 **lowering 중 template 렌더**다.

```
jit_lines() → inductor_meta_common() → triton_hash_with_backend()
            → driver.active.get_current_target()
```

`triton_hash_with_backend`는 `functools.cache`라 첫 호출의 backend가 프로세스
전체의 해시를 결정한다. PIM이 active인 상태에서 불려야 캐시 키가 PIM backend를
반영한다.

이 함수 자신은 `driver.active`(property)가 아니라 슬롯 `_active`를 본다. property
접근이 자동 탐지를 트리거하는데, GPU가 없고 CPU/PIM 둘 다 명시 선택 전용인
환경에서는 "0 active drivers"로 터진다.

### 3.2 `CachingAutotuner._precompile_config`

stock 코드는 tensor의 device type만 보고 driver를 고른다(cpu면 CPU driver). PIM은
tensor device가 CPU인데 compile target은 PIM이라 그 기준이 성립하지 않는다. 판단
기준을 **"이 template이 무엇으로 컴파일되기를 요구하는가"**로 바꾼다 —
`triton_meta["compile_backend"]`.

`_make_launchers`도 함께 patch한다. `CompiledKernel._init_handles`가
`driver.active.launcher_cls`로 launcher를 만들기 때문이다.

---

## 4. split-K template

`torch_backend/template.py`. kernel body는 `examples/split_k.py`의
`matmul_split_k_kernel`과 같은 연산이다. 같아야 하는 이유: 그 커널은 PIM
파이프라인을 통과하는 것이 확인된 유일한 GEMM이다. Inductor의 stock mm template를
가져다 쓰면 `tl.assume` / tensor descriptor / GPU 전용 heuristic이 섞여 들어와,
실패했을 때 그것이 PIM 연동 문제인지 IR 변환 문제인지 갈라내지 못한다.

stock template와 다른 점 넷은 전부 Inductor 규약이나 PIM pass 구조를 맞추기 위한
것이다.

### 4.1 PARTIAL을 mutated input으로

`examples/split_k.py`는 partial을 `torch.zeros`로 만든다. Inductor의 template
output buffer는 초기화되지 않은 버퍼라 그 의미가 유지되지 않는다. 그래서 zero로
채운 버퍼를 lowering이 따로 만들어 template의 mutated input으로 넘긴다. template의
선언상 출력은 쓰이지 않는 `[1]` int32 dummy다.

PIM generated host code가 partial의 모든 원소를 반드시 덮어쓴다는 것이 확인되면
`store_output`을 쓰는 보통의 template output으로 바꿀 수 있다.

### 4.2 M/N/K와 stride를 런타임 인자로 — `{{rt_size}}` / `{{rt_stride}}`

**이것이 없으면 컴파일되지 않는다.**

Inductor의 `{{size}}` / `{{stride}}` hook은 static shape에서 정수 리터럴을 낸다.
그러면 `PIMRecoverAccessOriginPass`가 split-K 출력면을 별도 축으로 승격하지 못한다.
그 pass의 `canonicalStride`는 stride 식을 `(기준 SSA 값, 상수 배수)`로 가른다.

```
stride_ck 가 상수 M*N   → base = null, coeff = M*N
tile 축 0 stride 도 상수 N → base = null, coeff = N
→ base 가 둘 다 null 로 일치하고 (M*N) % N == 0 이라 pid_k 항이 행 축에 흡수된다
```

출력 access가 rank-2가 되고, 버퍼 축 0을 `pid_k`와 `pid_m`이 함께 몰게 되어
`PIMPlaceDataPass`의 `checkBlockedComposite`가 진단한다.

stride가 런타임 인자면 `base`가 SSA 값이라 어느 tile 축의 `base`(null)와도 일치하지
않는다. 그때 pass는 `promote()`로 크기 1짜리 축을 앞에 새로 만든다 — 그 함수 주석
그대로 "split-K의 출력면처럼 원본 버퍼에는 있는데 tile에는 없는 축"이다. 축마다
loop가 하나씩만 남으므로 composite 판정 자체가 일어나지 않는다.

구현은 `PIMTemplateKernel`이 `_register_extra_template_env_fns`(Inductor의 공식
확장점)로 hook을 등록하고, 값을 `args.sizevars`에 넣는 것이다. 그러면
`python_argdefs`가 `SizeArg`를 만들고 `call_kernel`이 값을 호출부에 싣는다. 값이
같은 것끼리는 한 인자로 모이고, 값이 1인 인자는 Triton이 constexpr로 특수화한다 —
둘 다 직접 경로와 같다.

### 4.3 mask를 만들지 않는다 — `EVEN_TILES`

mask의 extent가 상수면 `PIMPlaceDataPass`의 `isExtentCandidate`가 상수를 extent
후보에서 배제하므로 복원이 실패한다. tile이 나누어떨어지는 shape에서는 mask가 항상
참이므로 아예 만들지 않는다.

M/N/K가 런타임 인자가 된 뒤로는 extent가 SSA 값이라 이 제약이 풀렸을 수 있으나
확인하지 않았다. `allow_masked_tiles=True`로 확인할 수 있다.

### 4.4 blocked K 분할

각 k 분할이 K의 **연속** 구간을 맡는다(`K_PER = K // SPLIT_K`,
`offs_k = pid_k*K_PER + k*BLOCK_K`). cyclic으로 가르면 한 device의 K 영역이
띄엄띄엄이 되어 `pim-place-data`가 진단한다. 커널의 `SPLIT_K`와 device K 분할 수
`p_k`는 같은 값이다 — `PIMConfig`에 `p_k` 하나뿐이고 `examples/autotune_gemm.py`도
`SPLIT_K=config.p_k`로 넘긴다.

---

## 5. scheduling

`torch_backend/scheduling.py::PIMCombinedScheduling`. 한 CPU 그래프 안에서 두
codegen backend를 섞는다.

```
PIM split-K template  → TritonScheduling
그 밖의 모든 노드      → CppScheduling
```

`CUDACombinedScheduling`과 같은 구조지만 "기본"이 반대다 — 여기서는 Cpp가 기본이고
PIM template만 Triton으로 빠진다.

### 5.1 `group_fn`이 Cpp 것이어서 생기는 일

`SIMDScheduling.codegen_template`은 이렇게 시작한다.

```python
_, (_numel, rnumel) = template_node.group
assert rnumel == 1
```

`node.group`은 device backend의 `group_fn`이 만든다. Triton은 축마다 크기를 곱해
정수를 내고(`rnumel == 1`), Cpp는 크기 tuple을 그대로 남긴다(`rnumel == ()`).
노드의 절대다수가 Cpp로 가므로 `group_fn`을 Cpp 것으로 두어야 하고, 그러면 저
assert가 성립하지 않는다. group은 template codegen 자체에 쓰이지 않으므로
(`TritonTemplateKernel`은 tiling을 output layout에서 직접 만든다) 그 분기를
`_codegen_pim_template`이 옮겨 놓았다.

### 5.2 flush 순서

`CppScheduling`은 노드를 `kernel_group`에 모아 두었다가 flush 때 커널 정의와 호출을
한꺼번에 낸다. Scheduler는 template 노드를 만나면 codegen 직전에 flush를 부르므로,
`PIMCombinedScheduling.flush()`가 **Cpp를 먼저** 비워야 wrapper에 찍히는 호출
순서가 프로그램 순서와 같아진다. `TritonScheduling.flush()`는 아무것도 모으지
않으므로(no-op) 뒤에 둔다.

### 5.3 fusion

PIM template 주변은 전부 막는다. epilogue/prologue가 붙으면 생성되는 Triton 소스가
split-K kernel과 달라져 PIM 파이프라인이 실패했을 때 원인이 갈라지지 않는다.
lowering이 `allow_epilogue_fusion` / `allow_prologue_fusion`을 False로 두고,
scheduling이 수평 fusion까지 한 번 더 막는다.

CPU 쪽 fusion은 그대로 둔다 — 실제로 Inductor가 다음 partial의 zero 채우기를 앞선
cpp 커널에 합친다.

---

## 6. 지원 조건

`strict` 모드에서 하나라도 어기면 즉시 멈춘다. CPU fallback은 하지 않는다 —
fallback이 있으면 PIM template compile이 실패했는데도 결과가 맞아 테스트가 통과한
것처럼 보인다.

그래프 모양 (`lowering.validate_pim_int_mm`):

- 입력 2개, 각각 2-D
- A·B dtype `torch.int8`, 출력 `torch.int32`
- tensor device CPU
- shape static (`dynamic=False`)
- `K_A == K_B`

template 고유 조건 (`plan.check_template_support`):

- `K % (P_K × BLOCK_K) == 0` — Constraint 5도 같은 것을 보지만 그쪽은
  `target_constraints["k_split_block_multiple"]`가 있는 target에서만 본다. 이
  template은 target과 무관하게 blocked 분할을 전제하므로 여기서도 본다.
- `M % BLOCK_M == 0`, `N % BLOCK_N == 0` (§4.3)
- `reduction_policy ∈ {"host", "none"}` — 이 lowering은 항상 CPU 합을 붙인다.
  `TARGET_REDUCTION_POLICY`가 `"device"`인 target(aimx)에서는 device가 이미 합친
  결과에 또 합치게 되므로 거부한다.

나머지(memory 예산, 축별 분할 수, alignment, tile grid)는 `ConstraintEngine`이
본다. 검사를 복제하지 않는다.

---

## 7. 아직 안 되는 것

| | 상태 |
| --- | --- |
| `P_M > 1` ∧ `P_K = 1` | 컴파일 실패. 원인 미확정 (§7.1) |
| batch (`b > 1`) | template에 없다 |
| dynamic shape | `dynamic=False` 전제 |
| aimx target | profile 없음. `reduction_policy="device"`라 lowering이 거부 |
| backward 그래프 | lowering 교체 범위 밖. `_int_mm`은 미분 대상이 아니라 지금은 무관 |
| epilogue fusion | 막아 두었다 (§5.3) |
| CPU–PIM partition, TTShared IR fan-out, 비동기 실행 | 다음 단계 |

### 7.1 `P_M > 1` ∧ `P_K = 1`

탐색 중 일부 config가 이 진단으로 실패한다.

```
pim-place-data: buffer axis 0 is driven by the partitioned axis output_row_tile
                together with another loop, ...
%189 = "pim.access"(%arg0, ...) sizes = array<i64: 64, 64> : (memref<*xi8>, ...)
```

`memref<*xi8>`에 `id = 0`이므로 **A(입력) 접근**이다. §4.2의 출력 접근 문제와는
다른 자리다.

관측된 패턴:

| B | G | P=(p_m,p_n,p_k) | |
| --- | --- | --- | --- |
| (32,32,64) | (2,1,1) | (64, 32, **1**) | 실패 |
| (16,16,32) | (1,4,2) | (8, 4, **1**) | 실패 |
| (16,16,32) | (1,1,2) | (64, 32, **1**) | 실패 |
| (32,128,32) | (1,1,1) | (16, 32, **4**) | 성공 |
| (32,128,32) | (1,1,1) | (32, 32, **2**) | 성공 |

`P_M > 1` 자체는 문제가 아니다. `P_K = 1`과 겹칠 때만 난다.

A 접근의 축 0(M축) origin은 `pid_m × BLOCK_M` 하나만 몰아야 한다. `group_m = 1`인
경우에도 나므로 block group loop는 아니다. `lane_mode="output_row"`가 tile을 출력
행으로 쪼개니 그 lane loop가 후보인데, 그렇다면 `P_K > 1`인 성공 사례에서도 났어야
한다. **소스만으로는 확정하지 못했다.**

막힌 것은 아니다 — 설계 §35대로 BAD sample로 기록되고 탐색이 계속된다. 다만
`max_benchmarks`를 소모한다.

다음 단계는 IR을 보는 것이다.

```
TRITON_KERNEL_DUMP=1 로 .ttsharedir 보존
triton-shared-opt <dump> <실패 로그의 pass 목록> --mlir-print-ir-before=pim-place-data
```

`--pim-map-devices`까지의 IR에서 `pim.device` 안의 loop 목록과 `pim.access` id=0의
`origin[0]`을 보면 두 번째 driver가 무엇인지 나온다.

---

## 8. 알려진 함정

**두 튜닝 경로가 같은 cache 파일을 쓴다.** `cache._key`에 template 정보가 없으므로
같은 `(M,N,K)`를 `autotune_gemm.py`와 `autotune_torch.py`에서 튜닝하면 나중 것이
이긴다. 재는 대상이 다른데 key가 같다. `PIMConfigCache(path, template_version=...)`
통로가 있으므로 필요하면 그 자리에 넣는다.

**partial 버퍼가 크다.** `[P_K, M, N]` int32다. attention QK를 seq=8192로 돌리면
`P_K=4`에서 1 GiB, `P_K=16`에서 4 GiB다. 탐색이 `P_K`를 16까지 열어 둔다.

**CPU partial 합이 launch 밖에 있다.** 별도 cpp 커널이라 `pim_driver.timing`에
잡히지 않는다. 총 시간을 볼 때는 반드시 포함해야 한다.

**중단된 탐색은 아무것도 남기지 않는다.** `search.py:493`의 `cache.store`는
`tune()` 맨 끝에서 한 번만 불린다. 반대로 `--max-benchmarks 1` 같은 smoke test는
record 1개짜리 항목을 남기는데, 그것을 "튜닝된 config"로 오해하기 쉽다.
`_cached_config`가 측정 수를 로그로 남기고 `attention_compare.py`가 그것을
출력하는 이유다.

**`pim_driver.timing`은 마지막 launch만 담는다.** attention처럼 한 그래프에 PIM
커널이 둘이면 뒤엣것만 남는다. 단계별 breakdown은 GEMM 하나짜리 그래프에서 봐야
한다.
