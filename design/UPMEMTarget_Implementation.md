# UPMEM Target IR 구현

PIM IR의 유닛 형태를 UPMEM DPU에서 실제로 돌아가는 host C++ / device C로 옮기는
계층의 현재 구현을 기술한다. `PIMIR_Implementation.md`가 "어느 데이터가 어느 유닛에
몇 번 가는가"까지를 다룬다면, 이 문서는 "그 데이터가 유닛 안에서 어떤 계층을 어떤
단위로 오가는가"부터를 다룬다.

용어는 `PIMIR_Implementation.md`와 같게 쓴다. UPMEM에서 유닛은 DPU, 유닛의 큰
메모리는 MRAM, 빠른 메모리는 WRAM, 유닛 내부 병렬 단위는 tasklet이다.

---

## 1. 이 계층이 존재하는 이유

PIM IR은 유닛 하나가 어떤 데이터를 맡는지까지 답한다. 그것만으로는 코드가 안 나온다.
UPMEM 하드웨어가 답을 요구하는 것이 네 가지 더 있다.

| 답해야 하는 것 | 하드웨어가 강제하는 것 |
| --- | --- |
| 메모리 계층 배치 | host DRAM → MRAM(64 MB) → WRAM(64 KB) 3단. 계산은 WRAM에서만 |
| 계층 사이 전송 단위 | MRAM DMA는 8 byte 정렬, 한 번에 2048 byte 이하, 연속 영역만 |
| 유닛 내부 병렬 | tasklet은 독립 스레드. 11단 pipeline이라 11개까지는 늘려도 시간이 그대로 |
| 원본 → 유닛 레이아웃 | `dpu_push_xfer`가 모든 DPU에 같은 바이트 수를 요구한다 |

이 넷은 전부 UPMEM 고유다. 그래서 target 중립인 PIM IR 뒤에 한 계층을 더 둔다.

경계선은 이렇게 긋는다.

> **PIM IR은 "어느 데이터가 어느 유닛에 몇 번 가는가"까지, UPMEM IR은 "그 데이터가
> 유닛 안에서 어떤 계층을 어떤 단위로 오가는가"부터.**

---

## 2. 전체 흐름

### 2.1 컴파일

```
PIM Common IR (pim.units + pim.unit_data)
   │
   ├ --convert-pim-to-upmem                     결정: 메모리 계층, 전송 단위,
   │     unit-memory-bytes / local-lanes              tasklet 분배, 레이아웃
   │     local-memory-bytes / host-threads
   ▼
UPMEM IR (upmem.* + scf/memref/linalg)
   │
   ├ triton-shared-translate --upmem-to-host    결정: 없음 (IR → C++)
   ├ triton-shared-translate --upmem-to-device  결정: 없음 (IR → C)
   ▼
{"host": host.cpp, "device": device.c}          JSON 번들 (binary_ext = pimobj)
   │
   ├ PIMLauncher (driver.py)                    빌드는 실행 시점에 한 번
   ▼
device.elf + host.so
```

변환은 `--convert-pim-to-upmem` 하나가 다 한다. codegen 두 단계는 IR을 그대로 옮기며
아무것도 정하지 않는다 — op의 파라미터가 나올 코드를 유일하게 정해야 하고, codegen이
스스로 정하는 것이 있으면 그 결정은 IR에 없는 것이다.

### 2.2 실행

한 번의 커널 호출이 하는 일이다. 괄호 안은 `PIM_TIMING` 계측의 칸 이름이다.

```
pim_acquire(units, elf)          런타임이 쥐고 있던 집합을 준다 (계측 밖)
   │
유닛 좌표 loop                    원본에서 유닛 몫을 모아 연속 버퍼로            (stage)
   │
DPU_FOREACH + dpu_push_xfer      버퍼마다 집합 전체로 밀어 넣는다               (h2d)
   │
dpu_launch(SYNCHRONOUS)          모든 DPU가 동시에 돈다                        (kernel)
   │
dpu_push_xfer(FROM_DPU)          출력 버퍼를 집합 전체에서 회수                 (d2h)
   │
그룹 loop                         부분합을 더하고 원본으로 되돌린다              (combine)
```

---

## 3. Dialect — `upmem`

`include/triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.td`.

### 3.1 타입과 메모리 공간

| | 뜻 |
| --- | --- |
| `!upmem.set` | 확보한 DPU 집합. host op들이 주고받는다 |
| `!upmem.unit` | 집합 안의 DPU 하나. `foreach_unit` 몸체에서만 생긴다 |
| `#upmem.space<mram>` | memref에 붙여 MRAM임을 나타낸다 |
| `#upmem.space<wram>` | 같은 방식으로 WRAM |

### 3.2 모듈 수준 — host와 device가 이름으로 공유한다

| op | device C | host C++ |
| --- | --- | --- |
| `upmem.mram` | `__mram_noinit T sym[N];` | `dpu_push_xfer`의 심볼 이름 |
| `upmem.host_var` | `__host T sym;` | `dpu_broadcast_to` / `dpu_push_xfer`의 심볼 이름 |

`upmem.mram`의 크기는 유닛이 받는 최대 영역이다. 실제 배정량(capacity)이 런타임
값이고 `__mram_noinit`은 상수 크기를 요구하므로, **유닛 메모리를 버퍼 수로 나눈
상한**을 쓴다. 초기화하지 않는 주소 예약이라 넉넉히 잡아도 적재 비용이 없다.

### 3.3 host 쪽

| op | C 대응 |
| --- | --- |
| `upmem.acquire` | `pim_acquire(n, DPU_BINARY)` — 확보 + 적재 |
| `upmem.foreach_unit` | 유닛 좌표 loop (준비) + `DPU_FOREACH` (전송) |
| `upmem.stage` | 원본의 strided 영역 → 타일 단위 연속 버퍼 |
| `upmem.to_mram` | `dpu_prepare_xfer` + `dpu_push_xfer(TO_DPU)` |
| `upmem.to_host_var` | 유닛마다 다른 스칼라를 그 DPU에 |
| `upmem.broadcast` | `dpu_broadcast_to` — 모든 DPU에 같은 값 |
| `upmem.launch` | `dpu_launch(SYNCHRONOUS)` |
| `upmem.gather` | `dpu_push_xfer(FROM_DPU)` — 집합 전체에서 한 번에 |
| `upmem.foreach_group` | 결합 축만 다른 유닛들을 묶어 먼저 합친다 |
| `upmem.unstage` | `stage`의 반대 |

**`acquire`가 확보와 적재를 함께 하는 이유.** 둘 다 커널을 부를 때마다 다시 할 일이
아니다. `dpu_alloc`은 수백 ms, `dpu_load`는 수십 ms가 드는데 둘 다 데이터와 무관하다.
나눠 두면 "이번에는 확보만 다시" 같은 것을 표현할 수 있는데 그런 경우가 없다. 짝이
되는 반납 op이 없는 것도 같은 이유다 — 다음 호출이 다시 쓰므로 놓을 자리가 프로세스
끝뿐이고, 그것은 런타임이 한다.

**`stage`의 필드.**

```mlir
upmem.stage %A origin[%r, %c] sizes[%m, %n] strides[%lda, %c1]
      pad_to[%pm, %pn] tile array<i64: 8, 16> {dedup_over = ["k"]}
```

| 필드 | 뜻 |
| --- | --- |
| `origin`/`sizes`/`strides` | 원본 안의 영역. `pim.unit_data`에서 그대로 온다 |
| `pad_to` | 결과 버퍼 shape = 배정량. 남는 자리는 0 |
| `tile` | 유닛 버퍼를 이 단위로 쓴다 |
| `dedup_over` | 이 축만 다른 유닛은 같은 결과를 받는다 |

`pad_to`가 `sizes`와 갈리는 이유는 `dpu_push_xfer`가 모든 DPU에 같은 바이트 수를
요구하기 때문이다. 마지막 유닛이 덜 맡아도 배정량 크기로 보낸다.

`unstage`는 여기에 `capacity`가 더 있다. 되돌릴 때 **읽는 범위**는 실제로 쓴
영역(`sizes`)이고 **축을 하나 건너뛰는 간격**은 버퍼 자체의 크기(`capacity`)에서
나온다. 유닛이 배정량보다 적게 맡으면 둘이 다르므로 나누어 써야 한다.

### 3.4 device 쪽

| op | C 대응 |
| --- | --- |
| `upmem.device_func` | `int main()` |
| `upmem.tasklet_id` | `me()` |
| `upmem.num_tasklets` | `NR_TASKLETS` |
| `upmem.get_host_var` | 전역 변수 참조 |
| `upmem.barrier` | `barrier_wait(&_barrier)` |
| `upmem.mram_read` | `mram_read(...)` |
| `upmem.mram_write` | `mram_write(...)` |

`mram_read`/`mram_write`는 verifier가 하드웨어 제약(정렬 8, 최대 2048)을 확인한다.
어기면 런타임에 조용히 깨지므로 IR 수준에서 막는다.

---

## 4. `--convert-pim-to-upmem`

`lib/Conversion/PIMToUPMEM/PIMToUPMEMPass.cpp`. `pim.units` 하나를 module 수준 선언 +
host 몸체 + `upmem.device_func`로 나눈다.

### 4.1 결정 1 — MRAM 버퍼는 유닛 메모리를 버퍼 수로 나눈다

```cpp
int64_t bufferBytes = unitMemoryBytes / (inputs.size() + outputs.size());
```

GEMM이면 A/B/C 셋이라 32 MB / 3 ≈ 10.6 MB씩이다. 유닛이 받는 배정량이 이보다 크면
링커가 `.mram.noinit` 구역이 넘쳤다고 실패한다. 이것이 **큰 shape에서 split을 더 잘게
잡아야 하는 이유**다.

축을 하나 건너뛰는 간격(pitch)은 컴파일 시점에 모르므로 `upmem.host_var`로 축마다
하나씩 만들어 host가 넣어 준다.

### 4.2 결정 2 — 타일 단위 MRAM 레이아웃

`upmem.stage`가 유닛 버퍼를 **타일 단위로** 채운다. 타일 하나가 MRAM에서 연속이 되어
device가 DMA 한 번으로 옮긴다.

```
자리 = (타일 번호를 축별 타일 수로 편 값) × 타일 원소 수 + (타일 안의 자리)
번호[d] = origin[d] / tile[d]        축별 타일 수[d] = capacity[d] / tile[d]
```

행 단위로 두면 타일 하나를 옮기는 데 **그 타일의 행 수만큼** DMA가 필요하다.
`BLOCK_M=32, BLOCK_K=16`이면 A 타일 하나에 32회이고 한 행이 16 byte다 — 준비 비용이
옮기는 양보다 크다. 타일 단위로 두면 타일 하나당 1회이고, 2048 byte를 넘을 때만
그만큼 쪼갠다(32×32 i32인 C 타일이 4096 byte라 2회).

`buildDMA`가 타일이 2048 byte를 넘을 때만 행 묶음으로 쪼갠다.

```cpp
int64_t perDMA = shape[0];
while (perDMA > 1 && (perDMA * rowElems * elemBytes > 2048 || shape[0] % perDMA))
  perDMA--;
```

### 4.3 결정 3 — tasklet 분배

`local_lanes > 1`이면 유닛이 맡은 타일을 tasklet에 나눈다.

```
병렬 grid loop들을 하나로 합친다 → tasklet 번호부터 tasklet 수만큼 건너뛰며 돈다
선형 번호를 안쪽 축부터 나머지·몫으로 벗겨 축별 좌표를 복원한다
```

**합치는 이유:** loop 하나만 나누면 그 loop의 반복이 tasklet 수보다 적을 때 나머지
tasklet이 논다. 유닛이 맡은 타일 수는 축을 합쳐야 제대로 나온다.

**barrier가 없는 이유:** 타일마다 쓰는 자리가 겹치지 않는다. 대신 tile 버퍼를
tasklet마다 따로 잡는다 — 앞에 tasklet 축을 하나 더 두고 자기 번호로 잘라 쓴다.

```cpp
shape.insert(shape.begin(), localLanes);       // 앞에 tasklet 축
offsets[0] = taskletId;  sizes[0] = 1;         // 자기 몫 subview
```

### 4.4 결정 4 — WRAM 예산을 IR 수준에서 막는다

tile 버퍼가 tasklet 수만큼 곱해지므로 쉽게 넘친다. 넘치면 `dpu-clang`이 링크 단계에서
수십 줄을 쏟아 내고 원인이 안 보인다.

```cpp
int64_t lanes = localLanes, budget = localMemoryBytes;
if (wramBytes > budget)
  return dev->emitError("tile buffers need ") << wramBytes
         << " byte of unit fast memory (" << lanes << " tasklet(s)) but the budget is "
         << budget << " byte; lower local_lanes or the BLOCK sizes";
```

IR을 고치기 **전에** 본다 — 반쯤 바뀐 IR이 오류에 딸려 나오면 읽기 어렵다.
`llvm::cl::opt`를 그대로 스트림에 넣으면 아무것도 안 찍히므로 값을 꺼내 쓴다.

i8 GEMM에서 tile 버퍼 합계는 `BLOCK_M·BLOCK_K + BLOCK_K·BLOCK_N + 4·BLOCK_M·BLOCK_N`
byte다. `32/32/16`이면 5,120 byte라 48 KB 예산에서 lanes가 9까지다.

### 4.5 device 몸체를 유닛 로컬로 만들기

| 하는 일 | 왜 |
| --- | --- |
| loop 상한을 `upmem.get_host_var`로 | device가 자기 좌표를 몰라도 되고, 마지막 유닛이 덜 맡는 것이 값 하나로 표현된다 |
| `pim.unit_origin`의 유닛 시작을 0으로 | 그러면 `pim.access`의 origin이 유닛 안의 위치가 된다 |
| tile → WRAM 버퍼, 계산 → memref 형태 linalg | tensor iter_arg가 없어진다. 버퍼가 loop를 관통하므로 값을 나를 필요가 없다 |
| `materializeExternals` | 옮겨 온 loop가 host func 안의 상수를 가리킨다. `device_func`은 그 func의 형제라 안 보이므로 다시 만든다 |

유닛 시작을 0으로 놓는 것은 Triton이 내는 `pid * BLOCK` 형태를 전제한다. 그 형태라야
유닛 시작에서 0이 되어 뺄셈 없이 성립한다.

**op을 지우는 순서**가 중요하다. 소비하는 쪽부터 지우지 않으면 아직 쓰이는 값을
지우게 되고, 남은 사용처가 해제된 op을 가리킨다.

```cpp
for (Operation *op : oldStores) op->erase();          // loop 결과를 놓아 준다
for (scf::ForOp l : loops) if (...) dropIterArgs(l);  // yield와 init을 놓아 준다
for (Operation *op : llvm::reverse(oldCompute)) op->erase();  // 쓰는 쪽이 먼저
for (Operation *op : oldLoads) op->erase();           // pim.access를 놓아 준다
```

---

## 5. Codegen

### 5.1 공통 — `CEmitter`

**MLIR 값 하나가 C 변수 하나, op 하나가 문장 하나.** 그래야 생성된 코드의 모든 줄을
IR의 op으로 되짚을 수 있다. 다룰 줄 모르는 op은 추측하지 않고 에러를 낸다.

`index`는 host에서 `int64_t`, device에서 `int32_t`다. DPU가 32비트 기계다.

### 5.2 host (`--upmem-to-host`)

`DPU_FOREACH`와 준비 loop를 **나누는** 것이 이 emitter의 뼈대다. `dpu_prepare_xfer`는
DPU마다이고 `dpu_push_xfer`는 집합 전체라, 버퍼 하나에 두 자리가 필요하다.

```c
// 1. 준비: 좌표를 풀고 유닛별 결과를 배열에 담는다
for (int64_t _u = 0; _u < UNITS; ++_u) {
  int64_t v3 = (_u / 16) % 16;            // 유닛 좌표 (자리값 분해)
  ...
  int64_t _s0_key = ...;                  // dedup 축을 뺀 조합 번호
  if (!_s0[_s0_key]) { ... }              // 그 조합에 대해 한 번만 모은다
  _p0[_u] = v9;  _p0_n[_u] = v9_bytes;
}
// 2. 전송: 버퍼마다 FOREACH + push
_idx = 0;
DPU_FOREACH(_set, _dpu, _idx) DPU_ASSERT(dpu_prepare_xfer(_dpu, _p0[_idx]));
DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "..._mram_0", 0, _p0_n[0], ...));
```

**staging의 중복 제거.** `dedup_over` 축만 다른 유닛은 같은 결과를 받으므로, 그 축을
뺀 조합마다 한 번씩만 모은다. GEMM에서 A는 n축에 불변이라 `split_n`배만큼 준비가
줄어든다.

**host 스레드.** 타일 loop 바깥을 묶어 나눈다. 스레드 팀을 만드는 비용이 고정이라
타일이 스레드 수보다 적으면 나누지 않는다.

```c
#pragma omp parallel for collapse(2) num_threads(8) if(v5 / 32 * v6 / 16 > 8)
```

전체 코어를 쓰면 작은 경우가 오히려 느려진다. 그래서 `host-threads`를 옵션으로 두고
기본을 8로 잡았다.

### 5.3 device (`--upmem-to-device`)

```c
__mram_noinit int8_t kernel_mram_0[11184810];   // host가 이름으로 채운다
__host int32_t kernel_tiles_x;
__dma_aligned int8_t w0[8192];                  // WRAM 타일 버퍼는 전역
BARRIER_INIT(_barrier, NR_TASKLETS);
int main() { ... }
```

WRAM 버퍼를 전역으로 빼는 이유는 둘이다. 유닛 메모리가 작아 스택에 둘 수 없고,
반복마다 새로 잡는 것이 아니라 하나를 돌려쓴다(`pim.co_resident`의 streaming).

**곱셈은 C의 `*` 그대로 둔다.** DPU에 32비트 곱셈기가 없어 `built_ins.h`의
`__builtin_mul_sl_sl_rrr`(아래 16비트끼리 곱하는 한 명령)로 바꿔 봤는데 커널이 두 배
느려졌다. 그 매크로가 인라인 어셈이라 최적화기가 안을 못 보고, 그 자리에서 loop
펼치기와 레지스터 재사용이 막힌다. i8 피연산자면 clang이 이미 좁은 곱셈을 낸다.

---

## 6. 런타임과 빌드

### 6.1 `pim_runtime.c` — 집합 하나를 쥐고 있는다

```c
struct dpu_set_t pim_acquire(uint32_t num_units, const char *binary_path);
```

| 상황 | 하는 일 |
| --- | --- |
| 같은 `num_units`, 같은 elf | 쥐고 있던 집합을 그대로 준다 |
| 같은 `num_units`, 다른 elf | `dpu_load`만 다시 (config 전환) |
| 다른 `num_units` | 기존 집합을 놓고 새로 확보 |
| 프로세스 끝 | `atexit`으로 한 번 놓는다 |

집합 하나만 쥔다. 집합이 보관하는 것은 예약과 적재된 elf뿐이고 데이터는 매 호출
다시 보내므로 여러 개를 둘 이유가 없다. 하나만 쥐므로 서로 다른 유닛 수가 쌓여 DPU가
고갈되는 일도 없다(sweep 안전).

### 6.2 빌드 (`driver.py`)

```
device.c → device.elf   dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=<local_lanes>
host.cpp → host.so      g++ -O2 -std=c++17 -fPIC -shared -fopenmp
                        -DDPU_BINARY="<elf 절대경로>" libpim_runtime.so -Wl,-rpath,...
```

`-O2`가 device에 필요하다. clang 기본인 `-O0`이면 MAC loop가 최적화되지 않아 약 두 배
느리다.

`libpim_runtime.so`는 내용 해시를 키로 **한 번만** 빌드한다. 모든 `host.so`가 같은
경로의 `.so`를 링크해야 동적 로더가 하나만 적재하고, 그래야 집합 캐시 전역이
프로세스에서 공유된다. 이것이 지속 집합의 핵심이다.

캐시 키는 `host_src + device_src + local_lanes + dpu_opt + runtime_so`다.
`local_lanes`는 소스에 안 박히고 `-D`로만 들어가므로 따로 넣어야 한다.

---

## 7. 하드웨어 제약이 IR에 남긴 자국

각 제약이 어느 결정을 강제했는지의 대응표다.

| 제약 | IR에 남은 것 |
| --- | --- |
| DMA가 연속 영역만 | `upmem.stage` / `unstage` 자체 |
| DMA 8 byte 정렬, ≤2048 byte | `mram_read`/`write` verifier, `buildDMA`의 행 묶음 쪼개기 |
| `push_xfer`가 균일 바이트 요구 | `stage`의 `pad_to`, `unstage`의 `capacity`, `unit_data`의 capacity/sizes 분리 |
| `__mram_noinit`이 상수 크기 요구 | MRAM 버퍼가 유닛 메모리를 나눈 상한, pitch를 `host_var`로 |
| WRAM 64 KB | `local-memory-bytes` 예산 검사 |
| 11단 pipeline | tasklet 11개까지는 늘려도 시간이 그대로 → `local_lanes` 선택 기준 |
| 32비트 곱셈기 없음 | (시도했다가 되돌림 — §5.3) |
| `dpu_alloc` 수백 ms | `upmem.acquire`와 지속 집합 |

---

## 8. 백엔드 옵션

`pim_backend/backend/compiler.py`의 `PIMOptions` 중 UPMEM이 읽는 것들이다.

```python
pim_target: str = None                  # "upmem" | "aimx" — 없으면 즉시 에러
num_units: int = 2048                   # 물리 유닛 수. split의 곱이 넘으면 컴파일 실패
unit_memory_bytes: int = 32*1024*1024   # 유닛당 큰 메모리 몫
local_memory_bytes: int = 48*1024       # 유닛당 tile 버퍼 예산
host_threads: int = 8                   # 영역을 모으고 되돌릴 때 쓰는 스레드 수
split_m/split_n/split_k: int = 1        # 곱 = 유닛 개수
local_lanes: int = 1                    # = NR_TASKLETS
```

`unit_memory_bytes`와 `local_memory_bytes`가 물리 크기(64 MB / 64 KB)보다 작은 이유는
같다 — **커널이 써도 되는 몫**이라는 뜻이다. DPU 런타임도 MRAM에 자리를 잡고, tasklet
스택과 런타임도 WRAM에 산다.

`pim_target`이 `target`이면 안 된다. Triton이 metadata를
`{"target": GPUTarget, **options.__dict__}`로 만들어서 같은 이름의 옵션이 GPUTarget을
덮어쓰고, `CompiledKernel`이 그것을 다시 읽어 `TypeError`가 난다.

---

## 9. 성능 — 지금까지 한 것

측정은 `(1024×512)×(512×1024)` i8 GEMM, `BLOCK 32/32/16`, `split=(16,16,1)`
(256 유닛), `local_lanes=8` 기준이다.

| 한 것 | 전 | 후 |
| --- | ---: | ---: |
| 타일 단위 MRAM 레이아웃 | A 타일 하나에 DMA 32회 | 1회 |
| 지속 집합 (`upmem.acquire`) | 160.23 ms / 6.70 GOPS | 82.01 ms / 13.09 GOPS |
| 〃 (가장 작은 경우) | 13.87 ms | 2.23 ms |

지속 집합이 들어간 뒤에야 `local_lanes` 16이 8보다 빨라졌다(9.09 vs 11.26 ms). 그전에는
launch당 약 11.6 ms의 고정 비용이 유닛 내부 병렬의 차이를 덮고 있었다.

되돌린 것도 적어 둔다. `__builtin_mul_sl_sl_rrr`은 커널을 59.04 → 124.91 ms로
느리게 만들었다(§5.3).

`PIM_TIMING` 계측이 보여 준 것 두 가지.

- **kernel이 거의 전부다.** 작은 경우에서 host 쪽(stage + combine)이 합쳐서 0.04 ms다.
- **`split_k`의 비용이 보인다.** `split=(8,8,8)`에서 kernel 31.31 ms에 d2h 18.74 ms,
  combine 9.86 ms다. 유닛 8개가 각각 온전한 크기의 부분합을 돌려주므로 회수량이 8배가
  되고 host가 그것을 8개씩 더한다. 둘을 합치면 커널보다 길다.

---

## 10. 알려진 한계

### 10.1 유닛 개수가 타일 수로 양자화된다

`PIMPartitionPass`의 `narrowLoop`가 축마다 `per = ceil(전체/유닛수)`로 자르고 상한에서
자른다. 그래서 `split_n`이 `N/BLOCK_N`을 나누지 못하면 남는 유닛이 아무 타일도 못
받는다. `N=12288, BLOCK_N=32 → 384타일`에 `split_n=512`를 주면 384개만 일한다.

균등 배분으로 바꿔도 **가장 바쁜 유닛의 타일 수는 `ceil(전체/유닛수)`로 그대로**라
kernel 시간은 안 줄어든다. 실제로 손해인 경우는 타일 수가 유닛 수보다 **적을** 때다.
그때는 BLOCK을 줄여 타일을 늘려야 하고, 대신 타일당 계산이 줄어 DMA 대비 계산 비중이
떨어진다.

비교 대상인 PGEMM은 출력 **원소** 단위로 나누므로 어떤 DPU 개수로도 나누어떨어진다.
우리는 Triton의 BLOCK 타일 단위라 나누어떨어져야 한다.

### 10.2 아직 안 한 최적화

| | 내용 |
| --- | --- |
| DPU 커널 register blocking | 안쪽 loop가 MAC 하나당 load 2회다. `_m`을 여러 단으로 펼쳐 누산기를 여러 개 두면 `b` load를 나눠 쓴다 |
| DMA double buffering | `mram_read` → 계산 → `mram_read`가 직렬이다. 다음 타일을 미리 읽으면 겹친다. WRAM이 두 배 필요하다 |
| replicated 입력의 broadcast | A가 n축에 불변인데 유닛마다 따로 밀어 넣는다. `dpu_broadcast_to`면 한 번이다 |
| staging 건너뛰기 | 영역이 이미 연속이고 padding이 없으면 원본 포인터를 그대로 넘길 수 있다 |

### 10.3 표현하지 못하는 것

- **결합 연산자는 `add`/`mul`만.** `emitForeachGroup`이 그 둘만 C로 낸다.
- **쪼갠 축약 축들의 결합 연산자가 다르면 실패한다.** `pim.unit_data`가 연산자를
  하나만 나른다.
- **출력의 배정량이 유닛 좌표에 의존하면 실패한다.** 모든 유닛이 같은 바이트 수를
  돌려줘야 `gather`가 성립한다.
- **`pim.access`의 origin이 `pid * BLOCK` 형태가 아니면** 유닛 시작을 0으로 놓는 것이
  성립하지 않는다.

### 10.4 검증

lit 테스트 세 개를 `upmem.acquire`와 `tile`/`capacity`에 맞춰 고쳤지만 **아직 돌린
적이 없다.** `test/Target/UPMEM/codegen.mlir`의 주석 일부는 행 단위 DMA 시절 문구가
남아 있다.

---

## 11. 파일 위치

```
include/triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.td   dialect 정의
lib/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.cpp                    verifier

include/triton-shared/Conversion/PIMToUPMEM/Passes.td              pass 정의와 문서
lib/Conversion/PIMToUPMEM/PIMToUPMEMPass.cpp                       --convert-pim-to-upmem

lib/Target/UPMEM/
  CEmitter.h                  host/device 공통 (이름 짓기, arith → C)
  UPMEMToHost.cpp             --upmem-to-host
  UPMEMToDevice.cpp           --upmem-to-device

pim_backend/backend/
  compiler.py                 파이프라인과 옵션
  driver.py                   빌드와 실행
  pim_runtime.c / .h          지속 집합

test/Dialect/UPMEM/ops.mlir                 dialect 왕복
test/Conversion/PIMToUPMEM/convert.mlir     변환
test/Target/UPMEM/codegen.mlir              codegen
```

실험용 스크립트는 저장소 밖에 있다.

```
/home/dlrkdals/TritonPIM/workspace/GEMM/test_m5_smallest.py    작은 경우부터 단계별로
/home/dlrkdals/TritonPIM/workspace/GEMM/test_llm_shapes.py     LLM shape + 설정 탐색
```

---

## 12. 직접 확인하는 방법

UPMEM IR을 만든다.

```bash
cd /home/dlrkdals/TritonPIM
<build-dir>/bin/triton-shared-opt \
  --convert-pim-to-upmem="unit-memory-bytes=33554432 local-lanes=8 \
                          local-memory-bytes=49152 host-threads=8" \
  dump/<hash>/matmul_kernel_i8.pim_common
```

소스를 본다.

```bash
<build-dir>/bin/triton-shared-translate --upmem-to-host   <target_ir>
<build-dir>/bin/triton-shared-translate --upmem-to-device <target_ir>
```

단계별 IR 덤프.

```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/home/dlrkdals/TritonPIM/dump \
TRITON_ALWAYS_COMPILE=1 python workspace/GEMM/test_m5_smallest.py
```

`dump/<hash>/`에 `ttsharedir`, `pim_common`, `target_ir`이 쌓인다.

테스트.

```bash
cd /home/dlrkdals/TritonPIM/triton_shared
cmake --build <build-dir> --target check-triton-shared-lit-tests
```
