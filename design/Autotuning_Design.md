# Autotuning 설계 — target 무관 config 탐색

> 목적: Triton의 `@triton.autotune`을 PIM backend에서 동작시키고, tile 크기와 유닛 분할을
> **한 탐색 공간**에서 고르게 한다 (`PIMCommon_Design.md` §16.7).
> 원칙: **탐색 층에 target 이름 분기를 두지 않는다.** 컴파일러가 이미 지키는 규칙
> (`pim_backend/backend/compiler.py:116` — "target 고유 값은 전부 여기 한 곳에 있다")을
> Python 층으로 연장한다.
>
> 관련 문서:
> - 분할·비용 모델의 IR 쪽 계획 → `PIMCommon_Partition_Plan.md`
> - `BLOCK`과 분할을 함께 탐색해야 하는 이유 → `PIMCommon_Design.md` §16.7
> - 기계 서술이 IR에 들어가는 경로 → `MemoryLowering_Design.md` §336-338

---

## 0. 현재 상태 (2026-08-07 점검)

### 동작함

- `PIMDriver.get_benchmarker()`가 `_pim_do_bench`를 돌려준다 (`pim_backend/backend/driver.py`).
  launch 전체(staging + h2d + kernel + d2h + combine) 벽시계를 warmup 1회 + rep 3회로 잰다.
  PIM launch는 동기 `ctypes` 호출이라 event도 synchronize도 필요 없다.
- `workspace/GEMM/autotuning_gemm.py`가 tile 크기 4개를 실제로 탐색한다. M=N=K=1024에서
  `BLOCK 32×32×16`이 선택되었다.
- Config kwargs로 `num_units`·`local_lanes`를 넘기는 경로가 이미 열려 있다. 커널 인자가 아닌
  kwargs는 `jit.py`의 `create_function_from_signature`가 만드는 `**options`를 타고
  `PIMOptions`로 흘러간다. `parse_options`가 `PIMOptions.__dataclass_fields__`에 없는 키를
  버리므로 `maxnreg` 같은 GPU 전용 키는 무해하다.

### 동작하지 않음 (이 문서의 대상)

| # | 문제 | 결과 |
|---|---|---|
| 1 | 용량을 넘는 config가 `triton-shared-opt` subprocess 실패로 올라옴 | `Autotuner._bench`가 `OutOfResources`/`CompileTimeAssertionFailure`/`PTXASError`만 잡으므로 **탐색 전체가 중단**. 지금은 사람이 손으로 후보를 골라 회피 중 |
| 2 | 탐색 공간이 tile 크기뿐 | 위 실행에서 활성 device 1024/2048, tasklet 1개. 가장 큰 손실이 탐색 밖에 있다 |
| 3 | 추정 없이 전수 측정 | config 하나당 launch 4회 + `device.elf`/`host.so` 컴파일. 공간을 넓히면 비현실적 |
| 4 | 측정이 총 시간 1종 | `pim_timing_ms` 5단계가 이미 있으나 autotune이 쓰지 않는다 |
| 5 | 컴파일러 → Python 되돌림 채널 없음 | 번들에 `host`/`device`/`lanes`만 실린다. 실제 분할 수와 가동률은 Python이 다시 계산해야 한다 |

### `PIMCommon_Partition_Plan.md`와의 관계

그 문서의 Phase 4가 이 영역의 선행 계획이다. 다만 그 문서가 전제하는 pass 이름
(`--pim-analyze-logical-work`, `--pim-partition`, `--pim-build-grid`, `--pim-remove-buffers`)은
IR 재구성 이후의 현재 파이프라인(`PIM_COMMON_PASSES`)에 없다. 이 문서는 **현재 파이프라인
기준으로 다시 쓴 것**이며, Phase 4의 `perf_model` 계약을 §6에서 다르게 결론짓는다 — 중립 동인
속성을 IR이 emit하기를 기다리지 않고, **측정 기록에서 계수를 학습**한다.

---

## 1. 원칙

### 1.1 target 이름 분기 금지

탐색 층은 `_TARGET_PROFILE`(`compiler.py:117-132`)과 `PIMOptions`가 낸 **값**만 읽는다.
`if target == "upmem"`을 쓰지 않는다. target 추가 비용을 `_TARGET_PROFILE` 한 줄로 유지한다.

### 1.2 3층 구조 — 섞지 않는다

`PIMCommon_Design.md` §16.7이 명시한 분리다.

| 층 | 판단 | 근거 | 틀렸을 때 |
|---|---|---|---|
| **버림** | 실행 가능 여부 | 부등식 (컴파일 전) | 놓쳐도 아래 층이 받는다 |
| **순위** | 추정 | 비용 모델 (컴파일 전) | 최적을 놓친다 |
| **측정** | 사실 | 실행 | — |

전송량만으로 순위를 매기면 항상 유닛 하나짜리 설정이 1등이 되는데, 그것은 버림 층에서 이미
걸러졌거나 연산 시간이 가장 길다 (§16.3). 그래서 **순위는 버리지 않는다.**

### 1.3 실패 방향이 안전한 쪽

인식하지 못한 실패는 "지금과 같은 동작"(탐색 중단)으로 떨어진다. 조용히 틀린 config를 고르는
경로를 만들지 않는다.

---

## 2. 기계 서술 (0단계)

target 상수는 지금도 `_TARGET_PROFILE` 한 곳에 있고 `_profile_pass()`가 그것을
`--pim-set-target-profile` 문자열로 만든다. 탐색 층도 같은 값이 필요하므로 **dict를 먼저 만들고
`_profile_pass`가 그 dict를 문자열로 바꾸도록 뒤집는다.** 두 경로가 어긋날 수 없게 된다.

```python
# pim_backend/backend/tuning.py (신규)
def machine_profile(options) -> dict:
    t = _TARGET_PROFILE[options.pim_target]
    bank, local = t["capacity"](options)
    return {"target":         options.pim_target,
            "devices":        options.num_units,
            "core_shape":     [int(x) for x in t["pim_core_shape"].split(",")],
            "lanes":          options.local_lanes,
            "execution_kind": t["parallel_execution_kind"],
            "bank_bytes":     bank,     # 음수 = 미측정
            "local_bytes":    local}    # 0 = 그 저장소가 없음
```

**이 아래 모든 코드는 이 dict만 읽는다.**

`bank_bytes < 0`(미측정)과 `local_bytes == 0`(없음)의 구분은 `compiler.py:113-115`의 규약을
그대로 따른다. 이 두 값이 aimx를 예외 처리 없이 흡수하는 열쇠다.

---

## 3. 실패를 복구 가능하게 만든다 (1단계)

### 3.1 PIM의 공통 성질

PIM에서는 tile 크기·유닛 수·lane 수가 전부 컴파일 시점 상수이고 기계는 profile로 서술된다.
그래서 **config가 바꿀 수 있는 실패는 언제나 두 상수의 비교로 나온다.** 층이 달라도 문구의
형태가 같다.

| 진단 | 위치 | 층 |
|---|---|---|
| `tile buffers need 69632 byte of local storage but a PIM Core has 49152` | `PIMVerifyPass.cpp:83` | PIM Common |
| `the partition counts need 4096 PIM Devices but the machine has 2048` | `PIMVerifyPass.cpp:99` | PIM Common |
| `this buffer needs 12345 byte of MRAM but the per-buffer share is 10922 byte` | `PIMToUPMEMPass.cpp:366` | target |

반면 config를 바꿔도 사라지지 않는 진단은 이 형태가 아니다.

- `PIM Group with 128 PIM Cores is not mapped yet` (`PIMMapCoresPass.cpp:49`) — 미구현
- `a PIM Group holds N logical tiles per wave; lockstep waves are not built yet`
  (`PIMFormExecutionWavesPass.cpp:59`) — 미구현
- `origin rank does not match sizes rank`, `tile_footprint does not divide pim_core_shape` — IR 불변식

**즉 메시지 형태가 곧 분류다.** 진단 목록을 표로 유지할 필요가 없다.

### 3.2 설계

`_run_opt`(`compiler.py:44`)가 **모든 MLIR pass의 단일 통로**다 — PIM Common(`:173`),
UPMEM 변환(`:179`), AiMX 변환(`:213`)이 전부 지나간다. 여기 한 곳만 고친다.

```python
from triton.runtime.errors import OutOfResources

# PIM은 용량이 전부 컴파일 시점 상수라, config 때문에 나는 실패는 항상 "필요 N, 보유 M"
# 꼴로 나온다. 그 형태만 autotune이 아는 예외로 바꾸고 나머지(미구현·IR 불변식)는 그대로
# 올려 hard fail로 둔다.
_NEED = re.compile(r"needs? (\d+) (.+?) but .+? (?:has|is) (\d+)")

def _run_opt(mlir_text: str, passes: list) -> str:
    opt = cpu_compiler._get_triton_shared_opt_path()
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.mlir"), os.path.join(tmp, "out.mlir")
        Path(src).write_text(mlir_text)
        r = subprocess.run([opt, src, *passes, "-o", dst], capture_output=True, text=True)
        if r.returncode:
            m = _NEED.search(r.stderr)
            if m:
                raise OutOfResources(int(m.group(1)), int(m.group(3)), m.group(2))
            raise RuntimeError(r.stderr)
        if r.stderr:
            sys.stderr.write(r.stderr)   # 경고는 지금처럼 그대로 보인다
        return Path(dst).read_text()
```

`OutOfResources(required, limit, name)`의 세 인자가 정규식의 세 그룹에 1:1로 대응한다.
`Autotuner._bench`가 이 예외를 잡아 해당 config를 `inf`로 두고 탐색을 계속한다.

### 3.3 launcher는 손대지 않는다

- **MRAM**은 `checkCapacity`(`PIMToUPMEMPass.cpp:356-369`)가 컴파일 시점에 잡는다 — 위 표의
  세 번째 진단이다.
- **WRAM**은 `pim-verify`가 `usable_local_capacity_bytes`(기본 48 KB)로 먼저 잡는다. 이 값이
  물리 WRAM(64 KB)보다 작게 잡혀 있어(`compiler.py:256-258`) 링크 단계에 도달하기 전에 걸린다.

링크까지 새는 사례가 관측되면 그때 `_build_<target>_so`에 같은 두 줄을 넣는다. 미리 넣지 않는다.

### 3.4 규약으로 승격

문구에 기대는 취약함은 규약으로 없앤다.

> **PIM 계열 pass는 용량 부족을 `... need(s) <N> <자원> but <보유자> has/is <M>` 형태로 낸다.**
> autotuning이 이 형태를 config 단위 실패로 해석한다.

새 target의 변환 pass가 이 규약만 지키면 탐색 층은 한 줄도 고치지 않는다.

---

## 4. Early pruning

1단계와 **중복이 아니라 계층**이다. early pruning은 싸고 확실한 부등식으로 컴파일을 아끼고,
1단계는 나머지를 안전하게 흡수한다. **그래서 early pruning은 정확할 필요가 없다.**

### 4.1 기호

```
기계 (machine_profile)
  D_phys  물리 유닛 수             upmem 2048
  S       유닛당 local 용량        upmem 49152,  aimx 0 (local 저장소 없음)
  Bnk     유닛당 bank 용량         upmem 32Mi,   aimx -1 (미측정)
  kind    parallel_execution_kind  multithread | simd

config
  bm, bn, bk   BLOCK_M / BLOCK_N / BLOCK_K
  D            num_units      (이 커널이 쓸 유닛 수, D <= D_phys)
  Lreq         local_lanes    (요청 lane 수)

형상 (nargs)
  M, N, K

커널 (target 상수가 들어가지 않는 유일한 커널 지식)
  W(bm,bn,bk) = local tile 바이트 합 = pim::localTileBytes 와 같은 값 (PIMUtils.cpp:56)
  이 GEMM     = bm*bk*1 + bk*bn*1 + bm*bn*4        (A:i8, B:i8, 누산기:i32)
```

### 4.2 규칙 — 전부 PIM Common pass의 사본

**R1. logical tile 개수** (`pim-build-logical-grid`)

```
Tm = ceil(M / bm)      Tn = ceil(N / bn)      Tk = ceil(K / bk)
```

`Tm × Tn`이 grid 축, `Tk`는 tile 안의 K 루프 반복 횟수.

**R2. lane 해석** (`PIMMapCoresPass.cpp:59-72`)

```
        | Lreq                                      kind = simd  (lane마다 버퍼를 복제하지 않음)
L    =  | Lreq                                      S = 0        (local 저장소가 없음)
        | max(1, min(Lreq, floor(S / max(1,W))))    그 외
```

요청값이 아니라 이 `L`이 실제 값이며, `-DNR_TASKLETS`로 들어가는 수다.

**R3. device 분할** (`PIMMapDevicesPass.cpp:23-34, 111-119`)

```
(P1, P2) = argmin (p1 - p2)   subject to  p1 * p2 = D,  p2 <= p1
Pm = P1,  Pn = P2             (tile 축이 2개일 때. 1개면 P = D)

factorize(2048) = (64, 32)    factorize(1024) = (32, 32)    factorize(128) = (16, 8)
```

**R4. device당 일감** (`narrowLoop`, `PIMMapDevicesPass.cpp:40-74`)

몫/나머지로 나누므로 partition `i`는 `floor(Tm/Pm) + [i < Tm mod Pm]`개를 맡는다. 시간은 가장
많이 맡은 device가 정하므로 올림을 쓴다.

```
tm = ceil(Tm / Pm)      tn = ceil(Tn / Pn)      device당 tile = tm * tn
A  = min(Tm, Pm) * min(Tn, Pn)                  실제로 일하는 device 수
u_dev = A / D                                    device 가동률
```

**R5. round와 lane 가동률** (`pim-form-execution-waves`)

```
R      = ceil(tm * tn / L)          round 수
slots  = R * L                      device 하나가 시간상 도는 칸 수
u_lane = (tm * tn) / (R * L)        lane 가동률
```

**R6. 세 층 패딩과 기계 팽창률**

```
Π_tile = (Tm*bm * Tn*bn * Tk*bk) / (M*N*K)      tile 내부 패딩
Π_dev  = (D * tm * tn) / (Tm * Tn)              device 분배 패딩
Π_lane = (R * L) / (tm * tn)                    lane 분배 패딩

Π = Π_tile * Π_dev * Π_lane
  = D * R * L * (bm*bn*bk) * Tk / (M*N*K)
```

곱이 닫힌 형태로 접힌다. 분자는 **기계가 시간상 소화하는 연산량**(모든 device × 모든 round ×
모든 lane × tile 하나의 MAC), 분모는 **실제로 필요한 연산량**이다. 항상 `Π >= 1`이고 `Π = 1`이
낭비 없음이다. `workspace/Tiling/plan.py`의
`machine = num_units * slots * (BM*BN*BK) * k_trips`와 같은 값이다.

### 4.3 버림 — 부등식 4개

```
[1] local 용량    S > 0  이고  W > S                        → 버린다
[2] 유닛 초과     D > D_phys                                → 버린다
[3] lane 무의미   S == 0 또는 kind == simd 이면 lane 축이 없음 → L = Lreq 로 고정
[4] 중복          key = (bm, bn, bk, D, L) 이 이미 나옴       → 버린다
```

- **[1]** R2가 `L`을 낮춰 맞추므로, 실패하는 유일한 경로는 `L = 1`인데도 `W > S`인 경우다.
  한 줄로 충분하다.
- **[2]** `Π P_i = D`이므로 `pim-verify`의 partition 검사와 같은 조건이다.
- **[4]** 실질적인 절약. `Lreq ∈ {1,4,8}`인데 `floor(S/W) = 2`이면 R2가 `L ∈ {1,2,2}`로 접으므로
  **컴파일 결과가 같은 config를 두 번 재게** 된다. 해석된 `L`로 키를 만들면 사라진다.

**MRAM 용량은 일부러 넣지 않는다.** device당 영역 크기는 커널 구조에 더 깊이 의존한다.
1단계가 잡아 주므로 여기서 모델링하지 않는다 — 계층 구조의 이득이다.

### 4.4 순위로 넘기는 것

`Π`, `u_dev`, `u_lane`은 **추정**이므로 순위에만 쓴다. `Π`만으로 자르면 전송 비용이 빠져
"tile을 최대한 크게"로 치우친다 (§16.3). `u_dev`가 낮은 config를 버리고 싶은 유혹도 같은
이유로 순위 쪽이다 — device를 덜 쓰는 대신 전송이 줄어 더 빠를 수 있다.

### 4.5 검산 — 2026-08-07 측정 형상

M = N = K = 1024, upmem (`D_phys = 2048`, `S = 49152`, multithread)

| config | W | L | (Pm,Pn) | tm·tn | A / D | Π_dev | Π_lane | **Π** | 판정 |
|---|---|---|---|---|---|---|---|---|---|
| 32×32×16, D=2048, Lreq=1 | 5120 | 1 | (64,32) | 1 | 1024/2048 | 2.00 | 1.00 | **2.00** | 통과 (실측 선택) |
| 64×64×16, D=2048, Lreq=1 | 18432 | 1 | (64,32) | 1 | 512/2048 | 4.00 | 1.00 | **4.00** | 통과 |
| 128×128×16, D=2048 | 69632 | — | — | — | — | — | — | — | **[1]로 버림** |
| 32×32×16, D=1024, Lreq=1 | 5120 | 1 | (32,32) | 1 | 1024/1024 | 1.00 | 1.00 | **1.00** | 통과 |
| 32×32×16, D=128, Lreq=8 | 5120 | 8 | (16,8) | 8 | 128/128 | 1.00 | 1.00 | **1.00** | 통과 |
| 32×32×16, D=128, Lreq=16 | 5120 | 8 | (16,8) | 8 | 128/128 | 1.00 | 1.00 | **1.00** | **[4]로 버림** |

`Π = 2.00`이 실측의 손실을 설명한다 — 기계가 실제 필요한 일의 두 배를 시간상 소화했고, 그
전부가 `Π_dev`(DPU 절반이 놀았다)다. `D`를 1024로 낮추거나 128 + lane 8로 가면 `Π = 1.00`이 된다.

### 4.6 코드 골격

```python
def make_prune(tile_bytes):        # tile_bytes = 커널이 주는 W(bm,bn,bk)
    def early_config_prune(configs, nargs, **kwargs):
        p = machine_profile_from(kwargs)          # §2. target 이름 분기 없음
        seen, kept = set(), []
        for cfg in configs:
            k = cfg.kwargs
            bm, bn, bk = k["BLOCK_M"], k["BLOCK_N"], k["BLOCK_K"]
            D    = k.get("num_units",   p["devices"])
            Lreq = k.get("local_lanes", p["lanes"])
            W    = tile_bytes(bm, bn, bk)

            if p["local_bytes"] > 0 and W > p["local_bytes"]:      # [1]
                continue
            if D > p["devices"]:                                   # [2]
                continue
            if p["execution_kind"] == "simd" or p["local_bytes"] == 0:
                L = Lreq                                           # [3]
            else:
                L = max(1, min(Lreq, p["local_bytes"] // max(1, W)))
            key = (bm, bn, bk, D, L)                               # [4]
            if key in seen:
                continue
            seen.add(key)
            kept.append(cfg)
        return kept
    return early_config_prune
```

호출 규약은 `autotuner.py`의 `self.early_config_prune(self.configs, self.nargs, **kwargs)`다.
`nargs`에 M·N·K가, `kwargs`에 `pim_target`이 들어 있다.

---

## 5. 탐색 공간 생성 (2단계)

후보를 손으로 쓰지 않고 profile에서 만든다.

```python
def candidates(blocks, options, devices=None, lanes=None):
    p = machine_profile(options)
    devices = devices or _halvings(p["devices"])          # 2048, 1024, 512, 256 ...
    # lane 복제가 없는 target에서는 lane이 탐색 축이 아니다.
    if p["execution_kind"] == "simd" or p["local_bytes"] == 0:
        lanes = [p["lanes"]]
    else:
        lanes = lanes or [1, 4, 8]
    return [triton.Config({"BLOCK_M": b[0], "BLOCK_N": b[1], "BLOCK_K": b[2],
                           "num_units": d, "local_lanes": l})
            for b in blocks for d in devices for l in lanes]
```

**target 무관성의 구체적 형태가 여기다.** lane 축을 넣을지를 target 이름이 아니라
`parallel_execution_kind`로 정한다. aimx는 `simd` + `local_bytes = 0`이므로 lane 축이 저절로
빠지고 후보 수가 1/3로 줄어든다.

`num_units`는 "물리 유닛 개수"이자 "이 커널이 쓸 유닛 수"로 겸용된다(`compiler.py:255`).
후보는 호출부에서 넘긴 값을 상한으로 삼는다.

---

## 6. 측정 기록 기반 비용 모델 (3단계)

`workspace/Tiling/plan.py`를 그대로 옮기면 안 된다. `MAC_PER_MS = 8485`, `PIPELINE = 11`,
`FIXED_MS = 0.23`은 **UPMEM에서 측정해 박아 넣은 상수**라, 옮기면 target 의존을 Python으로
옮겨 심는 셈이 된다.

**구조(target 무관)와 계수(target별 측정)를 분리한다.**

### 6.1 특징 — 컴파일 없이, target 상수 없이

§4.2의 R1~R6이 그대로 특징이다. 전부 PIM Common pass 규칙의 사본이므로 target에 묶이지 않는다.
추가로 커널이 주는 바이트 구조가 필요하다.

```python
def gemm_tile_bytes(bm, bn, bk):
    return {"local": bm*bk + bk*bn + bm*bn*4,   # W — R2/[1]이 쓴다
            "in":    bm*bk + bk*bn,             # 유닛으로 들어가는 바이트
            "out":   bm*bn*4}                   # 유닛에서 나오는 바이트
```

**남는 유일한 의존은 target이 아니라 kernel이다.** §8이 이것도 없앤다.

### 6.2 기록 — target별 JSON 한 파일

```
pim_backend/backend/profiles/<target>.json

{"target": "upmem",
 "samples": [
   {"shape": [1024,1024,1024],
    "config": {"BLOCK_M":32, "BLOCK_N":32, "BLOCK_K":16, "num_units":2048, "local_lanes":1},
    "features": {"macs_per_dev": ..., "bytes_in_per_dev": ..., "rounds": 1, "lanes": 1, ...},
    "ms": 159.44,
    "stages": {"stage":12.00, "h2d":25.12, "kernel":106.49, "d2h":5.65, "combine":10.19}},
   ...]}
```

채우는 방법은 둘이고, **같은 파일을 채우므로 따로 만들지 않는다.**

- **미리** — `workspace/GEMM/profile/`, `workspace/Tiling/`처럼 한 번 크게 sweep해 seed로 넣는다.
  이미 수행된 프로파일링이 그대로 첫 데이터가 된다.
- **쓰면서** — 평소 autotune 실행이 끝날 때 `configs_timings`를 features와 함께 append한다.

**`stages`를 함께 저장하는 것이 중요하다.** 총 시간 하나만으로 회귀하면 전송 계수와 계산
계수가 서로를 흡수한다. 단계별로 저장하면 `h2d`로 전송 계수만, `kernel`로 계산 계수만 따로
적합할 수 있어 표본이 훨씬 적게 든다. `pim_driver.timing`은 이미 있고, 이것을 내지 않는
target에서는 0으로 채워지므로(`driver.py`의 target 분기) "있으면 저장"으로 두면 target 무관하다.

### 6.3 적합과 예측

계수에 대해 선형인 모델로 두어 최소자승으로 푼다.

```
kernel_ms  ≈ a0 + a1 * (R * L * bm*bn*bk * Tk) / L
h2d_ms     ≈ b0 + b1 * bytes_in_per_dev
d2h_ms     ≈ c0 + c1 * bytes_out_per_dev
combine_ms ≈ d0 + d1 * combine_ops

perf_model = kernel_ms + h2d_ms + d2h_ms + combine_ms + stage_ms
```

- `plan.py`의 세 상수는 이 계수에 대응한다 — `MAC_PER_MS`는 `a1`의 역수, `FIXED_MS`는 `a0`,
  `PIPELINE = 11`은 `L < 11`일 때 파이프라인이 안 차는 효과다. **버리는 것이 아니라 upmem
  JSON의 초기값으로 넣는다.**
- `torch.linalg.lstsq`로 푼다 (torch는 이미 의존성이다).
- **cold start가 곧 fallback이다.** 표본이 임계치 미만이면 `perf_model`을 달지 않는다 =
  전수 측정 = 현재 동작. 새 target은 자동으로 여기서 시작한다.
- `top_k`는 신뢰도에 따라 움직인다. leave-one-out으로 예측 순위와 실측 순위의 상관을 재서
  낮으면 `top_k`를 키운다. §16.7의 "3번은 추정이므로 순위만 정하고 최종 판단은 4번의 측정에
  맡긴다"를 그대로 구현한 것이다.

---

## 7. 측정 (4단계)

- `_pim_do_bench`가 `pim_driver.timing`을 단계별로 남겨 §6.2에 넘긴다. Autotuner에는 지금처럼
  총합만 돌려준다.
- `rep = 3` 중앙값은 얕다. autotune은 "이 config가 낼 수 있는 최선"을 비교하는 것이므로
  최솟값이 더 안정적이다.
- `num_units`가 config마다 바뀌면 `dpu_alloc` 재예약(~280 ms)이 끼는데,
  `upmem_runtime.c`의 set 캐시가 같은 `num_units`면 재사용하므로 **warmup 1회가 흡수한다.**
  현재 구조가 이미 맞다.

---

## 8. `pim.metrics` 되돌림 채널 (5단계, 나중)

PIM Common pass가 module attribute로 실측치를 남기고 `compiler.py`가 metadata에 실어 보낸다.

```
pim.metrics = {local_tile_bytes, lanes, partitions, active_devices, macs, bytes_in, bytes_out}
```

얻는 것:

- §6.1의 커널별 `tile_bytes` 함수가 사라진다 (컴파일한 config는 실측치를 쓰고, 미컴파일
  config는 측정된 것들로부터 외삽).
- `find_sweepspot.png` / `autotuning.png`가 말하는 "런타임 모니터링에서 확보한 시스템 가동률"이
  추정이 아니라 실측이 된다.

무거우므로 §6이 실제로 쓸모 있다고 확인된 뒤에 붙인다.

---

## 9. target 추가 비용

| 층 | target 상수를 쓰는가 | 근거 |
|---|---|---|
| `machine_profile()` | **여기만** | `_TARGET_PROFILE` 한 줄 |
| §3 예외 번역 | 아니오 | 진단 형태가 규약 (§3.4) |
| §4 R1 tile 개수 | 아니오 | 형상과 BLOCK만 |
| §4 R2 lane | `S`, `kind` **값만** | `simd` / `S = 0`이면 예산 검사가 빠진다 |
| §4 R3~R6 | 아니오 | PIM Common pass의 사본 |
| §4 버림 [1][2] | `S`, `D_phys` **값만** | `pim-verify`가 검사하는 두 부등식과 같다 |
| §5 후보 생성 | `kind`, `local_bytes` **값만** | lane 축을 넣을지 결정 |
| §6 계수 | **측정으로 흡수** | target별 JSON. 새 target은 빈 파일 → 전수 측정으로 시작 |

**새 target을 추가하는 비용은 `_TARGET_PROFILE`에 한 줄과 `<target>.json` 빈 파일이다.**
`S = 0`(local 저장소 없음)과 `kind = simd`가 aimx를 예외 처리 없이 흡수한다 — 값의 정상적인
범위로 다룬다.

---

## 10. 착수 순서와 검증 계획

검증은 전부 사용자가 직접 실행한다.

| 단계 | 내용 | 확인 방법 |
|---|---|---|
| 1 | §3 예외 번역 | `autotuning_gemm.py`의 configs에 `128×128×16`(W = 69,632 byte) 추가 → **그 config만 건너뛰고** 나머지로 sweep이 끝나면 성공 |
| 2 | §4 early pruning | 같은 목록에서 `128×128×16`이 **컴파일도 되지 않고** 사라지는지. `Lreq ∈ {1,4,8}`을 넣어 [4]가 중복을 접는지 |
| 3 | §5 후보 생성 | `num_units ∈ {2048,1024,512,128}` × `local_lanes ∈ {1,4,8}`로 넓혀, §4.5 표의 `Π` 예측과 실측 순위가 맞는지 |
| 4 | §6.2 기록만 | JSON에 표본이 쌓이는지. 모델 없이 동작이 바뀌지 않아야 한다 |
| 5 | §6.3 적합 | leave-one-out 상관. 낮으면 `top_k`를 키우고 표본을 더 모은다 |

**4단계(기록)를 5단계(적합)보다 먼저 떼어 놓는 것이 요점이다.** 기록은 모델이 없어도 지금
켤 수 있고, 데이터가 쌓여야 모델을 만들 수 있으며, 만들어도 데이터가 없으면 검증할 수 없다.

---

## 11. 미결정 · 위험

- **진단 문구 의존** (§3). 규약(§3.4)으로 완화하지만 컴파일러 쪽 문구가 바뀌면 조용히
  매칭이 풀린다. 다만 그때의 동작은 "탐색 중단"이라 안전한 방향이다. lit 테스트로 문구를
  고정하는 것을 검토한다.
- **커널별 `tile_bytes`가 `pim::localTileBytes`와 어긋날 위험** (§6.1). 어긋나면 [1]과 R2가
  조용히 틀린다. §8이 근본 해결이고, 그 전에는 첫 컴파일 성공 시 번들의 `lanes`와 R2의 `L`을
  대조해 경고하는 정도로 막는다.
- **`num_units`의 이중 의미** (§5). "물리 유닛 개수"와 "이 커널이 쓸 유닛 수"를 한 필드가
  겸한다. 탐색 축으로 쓰면 상한을 별도로 관리해야 한다. `PIMOptions`에 물리 개수를 따로
  두는 편이 나은지 검토한다.
- **aimx는 `pim-map-cores`에서 막힌다** (`PIMMapCoresPass.cpp:48-52`, core가 1개가 아닌 경우
  미구현). 이 문서의 설계는 aimx에서 "예외 없이 동작"하도록 되어 있으나, 실제 검증은 그
  pass가 구현된 뒤에야 가능하다.
- **`Π`가 순위 지표로 충분한지 미검증**. §10의 3단계가 이것을 재는 자리다.
