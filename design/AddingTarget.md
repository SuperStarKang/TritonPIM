# 새 PIM target 추가하기

UPMEM 하나만 도는 상태에서 두 번째 target을 붙일 때 무엇을 만들고 어디를 건드리는지
적는다. 예시는 계획된 `aimx`를 쓰지만 절차는 target과 무관하다.

기준 트리는 `triton_shared`이고, 경로는 전부 그 아래의 상대 경로다.

---

## 0. 한눈에

| 계층 | 새로 만드는 것 | 기존 파일에서 고치는 곳 |
| --- | --- | --- |
| PIM 공통 IR | 없음 | 없음 (§1) |
| 공통 pass | 없음 | topology 게이트 두 곳 — **실제 임계 경로** (§2) |
| target dialect | `Dialect/PIMTarget/<T>/` | 등록 (§4.4) |
| 변환 | `Conversion/PIMTo<T>/` | 등록 (§4.4) |
| codegen | `Target/<T>/` | 등록 (§4.4) |
| compiler.py | 없음 | 3곳 (§5.1) |
| driver.py | 없음 | 2곳 (§5.2) |
| 런타임 C shim | 선택 (§7) | — |

가장 오래 걸리는 것은 dialect·변환·codegen이고, 가장 먼저 막히는 것은 §2다.

---

## 1. 경계 원칙

> 공통 계층이 **읽고**, 그 읽기가 **다른 모양의 기계에서도 의미를 갖는** 것만 PIM 공통에
> 남는다. 벤더 어휘를 이름에 담거나, 벤더 도구를 호출하거나, `pim_target == "<name>"`으로
> 갈리는 것은 target 아래로 간다.

따라서 다음은 하지 않는다.

- `#pim.<target>_ext` 같은 target 확장 attribute를 PIM dialect에 추가하지 않는다.
  공통 pass가 target 이름의 attribute를 읽으면 이 dialect가 target 무관하다는 전제가
  깨지고, 공통 pass가 읽지 않는 값이면 애초에 target IR에 두면 된다. 과거에 있던
  `#pim.upmem_ext`가 이 경우였고 제거했다 — `PIMDialect.td`의 해당 자리 주석 참고.
- 공통 계층에 `pim_<target>_runtime` 같은 중립 이름으로 target 코드를 두지 않는다.
  공통 pass가 정말로 필요로 하는 기계 값이라면 `#pim.target_profile`의 필드가 된다.

---

## 2. 먼저 확인 — 공통 pass가 지금 막는 것

아래 두 진단은 target 누수가 아니라 **공통 계층의 미구현 표시**다. Python 쪽 분기를
아무리 늘려도 여기서 멈춘다.

```
lib/Conversion/TTSharedToPIM/PIMMapCoresPass.cpp:48
    cores = pim_core_shape 원소의 곱;  cores != 1 이면 진단
    "PIM Group with N PIM Cores is not mapped yet"

lib/Conversion/TTSharedToPIM/PIMFormExecutionWavesPass.cpp:59
    wave 하나가 tile 여럿이면 진단
    "lockstep waves are not built yet"
```

`_TARGET_PROFILE["aimx"]`가 선언한 `pim_core_shape = "8,16"`은 곱이 128이라
`PIM_COMMON_PASSES`의 `--pim-map-cores` 단계에서 실패한다. `--convert-pim-to-aimx`는
호출조차 되지 않는다.

**PIM Core가 하나뿐인 topology(`pim_core_shape = "1"`)로 시작하면** 이 게이트를 지나
나머지를 먼저 검증할 수 있다. 실제 topology를 쓰려면 multi-core 매핑을 공통 계층에
구현해야 하고, 그것이 이 작업의 임계 경로다.

반대로 다음은 이미 준비돼 있으므로 손대지 않아도 된다.

- `parallel_execution_kind = "simd"` 처리 — `PIMMapCoresPass.cpp:62`가 lane 예산 계산을
  건너뛰고(SIMD는 lane마다 버퍼를 복제하지 않는다), `PIMVerifyPass.cpp:81`이
  `need = simd ? bytes : bytes * lanes`로 센다.
- `usable_local_capacity_bytes = 0` — "그 저장소가 존재하지 않음"이라
  `PIMVerifyPass.cpp:78`이 예산 검사를 건너뛴다.
- `usable_bank_capacity_bytes` — 공통 pass 중 아무도 읽지 않는다. `PIMToUPMEMPass`만
  읽으므로 미측정(음수)이어도 된다.

---

## 3. 만들 것 — 파이프라인 관점

```
TTIR → TTSharedIR → PIM Common IR → <T> IR → 소스 번들 → .so → 실행
                    └─ 공통, 그대로 ─┘  └──── 새로 만드는 구간 ────┘
```

`PIM_COMMON_PASSES`(`compiler.py:137`)는 target을 가리지 않는다. 그 뒤 두 단계
(`target_ir`, `pimobj`)만 target마다 다르다.

---

## 4. MLIR 쪽

### 4.1 target dialect — `Dialect/PIMTarget/<T>/`

`Dialect/PIMTarget/UPMEM/`을 본으로 삼는다. 벤더 어휘(DPU/tasklet/MRAM, descriptor/
opcode 등)는 전부 여기서부터 등장한다.

```
include/triton-shared/Dialect/PIMTarget/<T>/IR/<T>Dialect.td
include/triton-shared/Dialect/PIMTarget/<T>/IR/CMakeLists.txt   → <T>TableGen
lib/Dialect/PIMTarget/<T>/IR/<T>Dialect.cpp
lib/Dialect/PIMTarget/<T>/IR/CMakeLists.txt                     → add_triton_library(<T>IR)
```

각 상위 `CMakeLists.txt`에 `add_subdirectory` 한 줄씩 추가한다.

### 4.2 변환 pass — `Conversion/PIMTo<T>/`

```
include/triton-shared/Conversion/PIMTo<T>/Passes.td
include/triton-shared/Conversion/PIMTo<T>/PIMTo<T>.h    → GEN_PASS_DECL_ + create...Pass()
include/triton-shared/Conversion/PIMTo<T>/CMakeLists.txt
lib/Conversion/PIMTo<T>/PIMTo<T>Pass.cpp
lib/Conversion/PIMTo<T>/CMakeLists.txt
```

`PIMToUPMEMPass.cpp`의 마무리 부분을 따라, 소비한 `pim.target_profile`과
`pim.core_mapping`을 여기서 제거한다. 남겨 두면 target IR이 여전히 PIM을 가리키게 되고
뒤 단계가 기계 값을 다시 읽을 수 있다는 오해를 만든다.

### 4.3 codegen — `Target/<T>/`

```
include/triton-shared/Target/<T>/<T>Translations.h
lib/Target/<T>/<T>ToHost.cpp        (또는 stub)
lib/Target/<T>/<T>ToDevice.cpp      (또는 descriptors)
lib/Target/<T>/CMakeLists.txt       → add_triton_library(<T>Translation)
```

`lib/Target/CMakeLists.txt`에 `add_subdirectory(<T>)`를 넣는다.
host 소스는 §6의 C ABI를 반드시 지켜야 한다.

### 4.4 등록 — 여섯 곳

하나라도 빠지면 unknown-pass 진단이나 **파싱 단계 실패**로 나타난다. 특히 마지막 항목이
잘 잊힌다.

| # | 파일 | 넣을 것 |
| --- | --- | --- |
| 1 | `tools/triton-shared-opt/RegisterTritonSharedDialects.h:21` | `#include ".../Conversion/PIMTo<T>/Passes.h"` |
| 2 | `tools/triton-shared-opt/RegisterTritonSharedDialects.h:25` | `#include ".../Dialect/PIMTarget/<T>/IR/<T>Dialect.h"` |
| 3 | `tools/triton-shared-opt/RegisterTritonSharedDialects.h:56` | `mlir::triton::registerConvertPIMTo<T>Pass();` |
| 4 | `tools/triton-shared-opt/RegisterTritonSharedDialects.h:60` | `registry.insert<..., mlir::<t>::<T>Dialect>` |
| 5 | `tools/triton-shared-translate/triton-shared-translate.cpp:12` | `triton::register<T>To*Translation();` |
| 6 | `lib/Target/<T>/<T>To*.cpp` 꼬리 | 각 translation이 **자기 `DialectRegistry` 람다**를 갖는다 |

6번은 `UPMEMToHost.cpp:518`과 `UPMEMToDevice.cpp:245`를 보면 된다.
`triton-shared-translate.cpp`에는 `DialectRegistry`가 없고, 각 translation이 필요한
dialect를 스스로 등록한다. 빠뜨리면 입력 IR을 파싱하는 순간 실패한다.

라이브러리 링크 자체는 일반적이다 — `add_triton_library`가 `TRITON_LIBS`에 밀어 넣고
두 도구의 `CMakeLists.txt`가 그것을 펼치므로, 도구 쪽 CMake는 건드리지 않는다.

---

## 5. Python 쪽

### 5.1 `pim_backend/backend/compiler.py` — 3곳

**(1) `_TARGET_PROFILE`(117행)에 행 하나.** target 고유 기계 값은 전부 여기 있고,
다른 코드에는 target 이름 분기가 없다.

```python
    "<T>": {
        "pim_core_shape": "1",              # §2 — 처음에는 1로 시작하는 편이 낫다
        "parallel_execution_kind": "simd",  # "simd" | "multithread"
        "uniform_instruction_sequence": "true",
        # (usable_bank_capacity_bytes, usable_local_capacity_bytes)
        #   음수 = 미측정 → profile에 필드가 생기지 않고, 그 값을 필요로 하는 검사에
        #                   도달하는 순간 진단이 된다
        #   0     = 그 저장소가 존재하지 않음 → 해당 검사를 건너뛴다
        "capacity": lambda o: (-1, 0),
    },
```

**(2) 변환·codegen 함수 한 쌍.** `_pim_common_to_upmem`(176행)과
`_upmem_to_artifact`(182행)의 쌍둥이다. 배관(`_run_opt`, `_run_translate`)은 중립이라
그대로 쓴다.

```python
def _pim_common_to_<T>(pim_common: str, options) -> str:
    return _run_opt(pim_common, ["--convert-pim-to-<T>"])

def _<T>_to_artifact(<T>_ir: str, metadata) -> bytes:
    stub_src = _run_translate(<T>_ir, "--<T>-to-runtime-stub")
    ...
    # launcher가 호출할 진입 함수 이름을 stub에서 파싱해 metadata["name"]에 채운다.
    # Triton은 name을 기본 설정하지 않으므로 backend가 채워야 한다.
    match = re.search(r'extern "C" void (\w+)\(', stub_src)
    if not match:
        raise RuntimeError("generated <T> stub에서 진입 함수를 찾지 못함")
    metadata["name"] = match.group(1)
    return json.dumps({...}).encode("utf-8")
```

**(3) `add_stages`(318행)에 분기 하나.** 마지막 stage 이름은 `binary_ext`(`pimobj`)와
같아야 `CompiledKernel`이 kernel을 찾는다.

```python
        elif options.pim_target == "<T>":
            stages["target_ir"] = lambda src, metadata: _pim_common_to_<T>(src, options)
            stages["pimobj"] = lambda src, metadata: _<T>_to_artifact(src, metadata)
```

`PIMOptions.__post_init__`의 지원 target 문구도 함께 갱신한다.

### 5.2 `pim_backend/backend/driver.py` — 2곳

**(1) `_build_<T>_so`.** UPMEM 전용 코드는 전부 `_upmem_` 접두사 안에 있으므로 새 함수도
`_<T>_` 접두사 안에 자기 툴체인·환경변수·캐시 키를 전부 담는다.

- 캐시 키에는 **소스뿐 아니라 target 전용 빌드 입력을 전부** 넣는다. UPMEM은 최적화
  레벨(`dpu_opt`)과 런타임 `.so` 경로를 키에 넣는다 — 빠뜨리면 최적화 레벨을 바꿔도
  옛 산출물이 재사용된다.
- 벤더 환경변수는 모듈 수준이 아니라 이 함수 안, **캐시 조회보다 위**에 찍는다. 캐시
  적중이면 컴파일을 건너뛰지만 장치 예약은 그 뒤 launch에서 처음 돈다.

**(2) `__call__`(161행)에 분기 하나.**

```python
        elif self.pim_target == "<T>":
            so_path = _build_<T>_so(bundle["stub"], bundle["descriptors"])
```

`dlopen → getattr → ctypes 호출`은 공통이다. stub이 §6의 ABI를 지키면 그대로 돈다.

timing(191행)은 이미 target 분기 안에 있다. `pim_timing_ms`를 내보내지 않는 target은
아무것도 하지 않아도 되고, 내보내겠다면 그 분기를 늘린다.

---

## 6. host C ABI 계약

생성된 host 소스는 다음을 지켜야 공통 launcher가 그대로 호출한다.

1. **진입 함수는 `extern "C" void <name>(` 형태가 정확히 하나**, 그리고 그것이 파일에서
   첫 출현이어야 한다. `compiler.py`의 정규식이 이 철자로 이름을 회수한다.
2. **매개변수는 커널의 non-constexpr 인자를 원래 순서대로**, 그 뒤에 num_programs
   세 개(`gridX, gridY, gridZ`)다. `--pim-build-logical-grid`가 program_id 인자를 지우고
   num_programs만 남기기 때문이다. grid는 컴파일 캐시 키에 없으므로 반드시 런타임
   인자로 받아야 한다.
3. **버퍼 인자는 순수 데이터 포인터**다. rank·shape·stride 서술자를 받지 않는다.
   주소 계산에 필요한 값은 별도 스칼라 인자로 온다.
4. **반환형은 `void`.**
5. **timing은 선택**이다. 내겠다면 `extern "C" double <name>[5]`를 export하고
   `driver.py:191`의 분기를 늘린다.

### 지금의 제약 — 스칼라 인자는 32비트 정수만 안전하다

`driver.py:183`이 모든 비포인터 인자를 `ctypes.c_int32(int(...))`로 넘긴다. 반면
emitter는 `CEmitter::scalarType`(`lib/Target/UPMEM/CEmitter.h:36`)이 정하는 대로
`float`/`double`/`int64_t`/`int16_t`/`int8_t`/`uint8_t`도 낸다. **비-constexpr fp32나
i64 인자를 받는 커널은 지금 값이 잘린다.** target과 무관한 공통 launcher의 결함이다.

새 target이 i32 아닌 스칼라를 받아야 하면 이것을 먼저 고쳐야 한다. 고치는 방법은
emitter가 프로토타입을 쓰는 그 루프에서 선언 시그니처를 한 줄로 함께 내보내고
(`// pim.lanes = N` 관례와 같은 방식), launcher가 그 C 타입 이름대로 `argtypes`를
세우는 것이다. 두 목록을 겹쳐 보는 방식은 쓰지 않는다 — emitter는 lowering 후 MLIR
타입을, launcher는 lowering 전 Triton 시그니처를 보고 있어 대응이 위치 관례일 뿐이다.

---

## 7. 런타임 C shim은 선택이다

`upmem_runtime.c/h`가 존재하는 이유는 하나다 — `dpu_alloc` 약 280ms와 `dpu_load`
약 25ms를 세션당 1회로 줄이려고 장치 예약을 프로세스 전역에 붙잡아 두는 것이다.
**새 target에 그만큼 비싼 예약 단계가 없으면 만들 이유가 없고**, 생성된 stub이 벤더
런타임을 직접 링크하면 된다.

만든다면 두 가지를 지킨다.

- **파일과 심볼에 target 이름을 붙인다.** `<T>_runtime.c/h`, `<T>_acquire()`.
  공통 이름을 쓰면 이 문서의 §1이 다시 깨진다.
- **`-Wl,-soname,lib<T>_runtime.so`를 반드시 준다.** 없으면 링커가 `DT_NEEDED`에
  명령줄 경로(캐시 절대경로)를 그대로 적어, 캐시 디렉터리가 다른 런타임이 별개 객체로
  적재된다. 그러면 전역 캐시가 둘이 되어 장치 예약이 두 벌 잡힌다. target마다 soname이
  달라야 로더가 둘을 구분한다.

---

## 8. 검증 순서

빌드·실행은 전부 사람이 한다. 아래 순서로 하면 실패 지점이 좁혀진다.

1. **IR까지** — `triton-shared-opt`를 새로 빌드하고 `--convert-pim-to-<T>`가 등록됐는지
   본다(`triton-shared-opt --help | grep <T>`). lit 테스트를 `test/Conversion/PIMTo<T>/`에
   먼저 하나 둔다.
2. **소스까지** — `triton-shared-translate --<T>-to-* in.mlir`. 여기서 파싱이 실패하면
   §4.4의 6번(translation 자체의 `DialectRegistry` 람다)을 확인한다.
3. **공통 pass 통과 여부** — `pim_core_shape = "1"`로 두고 예제를 돌려 §2의 두 진단이
   안 나는지 본다. 나면 그 topology는 아직 지원되지 않는 것이다.
4. **적재까지** — `.so`가 만들어지고 `dlopen`되는지. 진입 함수 이름이 안 맞으면
   `getattr`에서, timing 심볼이 없는데 분기를 잘못 늘렸으면 `in_dll`에서 실패한다.
5. **수치** — 같은 커널을 CPU backend로 돌린 결과와 비교한다.
6. **자원 수명** — 한 프로세스에서 유닛 수를 A, B, A로 바꿔 가며 세 번 launch하고
   매번 정확성을 확인한다. 예제 한 번 실행으로는 예약 재사용·재할당 경로를 지나지
   않아 문제가 "느리지만 맞음"으로만 보인다.

---

## 9. 남아 있는 공통 계층의 제약

새 target을 붙일 때 부딪힐 수 있는, target 탓이 아닌 것들이다.

| 제약 | 위치 | 영향 |
| --- | --- | --- |
| 스칼라 인자가 i32만 안전 | `driver.py:183` | i32 아닌 스칼라를 받는 커널은 값이 잘린다 (§6) |
| `PIMOptions` 기계 기본값이 UPMEM 숫자 | `compiler.py:255-266` | `num_units = 2048`, `local_lanes = 1` 등. 새 target은 명시적으로 넘겨야 한다 |
| multi-core topology 미구현 | `PIMMapCoresPass.cpp:48`, `PIMFormExecutionWavesPass.cpp:59` | §2 |
| `_run_translate`가 utf-8 디코드 | `compiler.py:109` | descriptor를 바이너리로 내보내야 하면 바이너리용 헬퍼가 필요하다 |
| `--pim-map-devices` / `--pim-set-target-profile`에 lit 커버리지 없음 | `test/Conversion/TTSharedToPIM/` | 공통 배치 pass를 고칠 때 자동 검증 수단이 없다 |
| `triton-shared-translate`가 lit에 연결돼 있지 않음 | `test/CMakeLists.txt`, `test/lit.cfg.py` | emitter 테스트가 빌드 순서에 의존한다 |

---

## 부록 — Triton backend discovery 제약

`compiler.py`에는 `BaseBackend` 서브클래스가, `driver.py`에는 `DriverBase` 서브클래스가
**정확히 하나씩만** 있어야 한다. Triton의 `_find_concrete_subclasses`가 `dir(module)`을
훑어 개수를 세기 때문이다. 그래서 CPU 쪽 코드를 클래스가 아닌 모듈로 참조한다
(`compiler.py:34`, `driver.py:31`).

새 target을 위해 `class <T>Backend(BaseBackend)` 같은 것을 만들어 이 두 모듈의
top-level 이름으로 가져오면 **PIM뿐 아니라 모든 Triton backend의 discovery가 실패한다.**
target별 코드는 일반 함수로 둔다.
