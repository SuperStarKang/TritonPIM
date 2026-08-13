# PIM Common IR 구조 변경 검토

질문 세 가지에 답한다.

1. PIM Common IR의 구조를 바꾸려면 어떻게 해야 하는가
2. Target IR과 PIM Common IR이 둘 다 필요한가
3. "Triton 구문 = PIM 하드웨어 개념" 해석이 좋은 방향인가, 어떻게 강화하고 어떻게 lowering하는가

방법: 설계안 4개를 서로 독립적으로 만들어(병합 / machine model / 경계 이동 / 커널이 곧 스케줄)
각각을 적대적으로 심사하고, 그 위에 코드·덤프 실측을 얹었다.

---

## 0. 결론

**계층 구조는 지금 병목이 아니다.** 어느 설계안도 현재 실패를 고치지 못한다(심사 4건 모두
coverage 2~4점). 병목은 **주소 표현**이다.

**그런데 계층의 근거는 실제로 무너져 있다.** 다만 무너진 방향이 예상과 반대다 —
메모리 쪽 차이는 전부 숫자로 환원되고, 계층이 정말 필요한 곳은 연산 쪽인데 그 계층이 없다.

**해석의 방향은 맞다.** 단 `num_warps`·`num_stages`·`SPLIT_K` 세 가지에 대한 전제가 틀렸고,
빠진 항목이 셋 있다.

---

## 1. 실측 — 참조 커널 세 개가 지금 어떻게 되는가

`dump/`의 산출물로 확인했다. 디렉터리는 소스 해시로 갈리므로, `llir`/`obj`가 있으면 CPU 실행,
`ttsharedir`만 있으면 **PIM 실행이 그 다음 단계에서 실패**한 것이다.

| 커널 | 디렉터리 | 산출물 | 해석 |
|---|---|---|---|
| `matmul_split_k_kernel` | 3개 | `ttir` `ttsharedir` | **PIM 시도 → `pim_common` 단계에서 실패** |
| | 4개 | `llir` `obj` `ttir` `ttsharedir` | CPU 실행 |
| `matmul_grouped_kernel` | 1개 | `llir` `obj` `ttir` `ttsharedir` | CPU만. PIM 시도 없음 |
| `gemm_kernel` | 1개 | `llir` `obj` **`pim_common`** `ttir` `ttsharedir` | PIM 시도 → **`target_ir` 단계에서 실패** |

**세 커널 중 어느 것도 지금 PIM으로 컴파일되지 않는다.** 그리고 실패 원인이 전부 계층 경계와
무관하다.

| 커널 | 실패 원인 |
|---|---|
| `gemm.py` | `a_ptrs += BLOCK_K * stride_ak` — 주소를 loop `iter_arg`로 나른다. `decomposeOffset`이 축을 못 나눠 `origin`이 틀리고, `--convert-pim-to-upmem`이 `byAccess.at()`에서 죽는다 (`Implementation_Audit.md` §1.1) |
| `split_k.py` | grid 축 0이 `pid // num_pid_n` / `pid % num_pid_n`. 융합된 1차원 grid에서 m·n 축을 복원하지 못한다 |
| `grouped_gemm.py` | 같은 융합 1차원 형태 + `group_size_m = min(num_pid_m - first_pid_m, GROUP_M)`이 런타임 M에 의존 |

덤프 40개 중 16개가 `--convert-ttshared-to-pim` 또는 그 직후에서 죽는다. **분할 계층까지 도달한
커널 중 splits 개수 검사에 걸린 것은 없다** — `pim_common`에 도달한 커널은 전부
`split_axes`가 정확히 3개다. 즉 그 검사를 고쳐도 회수되는 커널은 0이다.

---

## 2. 계층은 필요한가 — 근거는 무너졌지만 방향이 반대다

설계 문서가 주장하는 target 중립성을 하드웨어 속성 10개로 나눠 검사했다.

| 판정 | 개수 | 항목 |
|---|---|---|
| 진짜 파라미터 (숫자로 환원됨) | 4 | 유닛 정의, 전송 granularity/정렬, 결합 위치, 유닛 간 통신 |
| 파라미터인데 **정의가 없음** | 2 | 최소 연산 shape, 결합 연산자 합법성 |
| 변별력 없음 | 1 | 유닛 간 통신 (세 target 모두 없음) |
| **진짜 계층이 필요** | **3** | **유닛 내부 병렬성, 연산 표현 형태(코드 vs descriptor chain), 합법 연산 집합** |

**계층이 필요한 셋은 전부 연산 쪽이다. 메모리 쪽 차이는 전부 숫자다.**

그런데 현재 PIM Common은 거의 전부 메모리다 — `pim.access`/`load`/`store`/`unit_data`/
partition/replication/combine. 연산 쪽 계층인 `--pim-classify-compute`는 **완전히 죽어 있다**
(`pim.op`의 소비자가 0개, 66줄 전체가 dead).

> **존재하는 계층은 없어도 됐던 쪽이고, 필요한 계층은 아예 없다.**

### 중립성이 이미 깨진 자리

| 증거 | 내용 |
|---|---|
| `pim.unit_data`의 `capacity` | `dpu_push_xfer`가 모든 DPU에 같은 바이트 수를 요구해서 생긴 필드다. 벤더 라이브러리 시그니처가 중립 dialect에 들어와 있다. `UPMEMTarget_Implementation.md:377`이 인과를 직접 적어 놓았다 |
| `--pim-partition`의 옵션 | `num-units`, `local-memory-bytes`. `PIMCommon_Design.md:31-34`가 "DPU 개수", "유닛 로컬 메모리 크기", "분할 후보 선택", "Split-K 개수 결정"을 **PIM Common에 넣지 않는다**고 명시한 바로 그 목록이다 |
| `pim.unit_origin` | 생산 `PIMPartitionPass.cpp:129`, 소비 `PIMToUPMEMPass.cpp:308` 단 한 곳. UPMEM이 주소를 유닛 로컬로 되돌리기 위해서만 존재 |
| `pid * BLOCK` 전제 | `PIMToUPMEM/Passes.td:38-39`. target이 **프론트엔드 관용구**가 중립 계층을 그대로 통과하기를 전제한다 |

### 판정

병합이 아니라 **경계를 근거가 있는 곳으로 옮기는 것**이 맞다.

```
[Layer 1] Triton 정규화          하드웨어 숫자 0개
  staging 관용구 → 논리 주소 / grid → loop / 주소 축별 분해 / 축 불변성
  근거: PIM과 무관하게 어떤 backend든 필요하다

[Layer 2] machine 매핑 + 연산 legalization      전부 숫자 + target별
  grid → 유닛 / 메모리 계층 배치 / lane 분배 / 연산 합법화 / codegen
  근거: 위 표의 "계층이 필요한 셋"이 전부 여기 있다
```

`pim.units` / `pim.unit_data` / `capacity` / `unit_origin` / `--pim-partition`은 전부 Layer 2다.

**AiMX는 코드가 0줄이다** (dialect·pass·소스 없음, `--convert-pim-to-aimx`는 stub). 즉 중립
계층을 정당화하는 두 번째 소비자가 아직 없고, 위 판정은 문서가 기술한 AiMX 성질에 근거한다.

---

## 3. 제안한 해석의 검증

### 3.1 유지해도 되는 것

| 제안 | 판정 |
|---|---|
| K loop = 한 PIM 안에서 실행될 연산 | **맞다.** 누산기가 유닛에 상주하고 A/B 타일이 스트리밍된다는 뜻이고, `pim.co_resident`의 streaming/persistent 구분이 이미 그것이다 |
| `tl.load` = 한 번에 올릴 크기 | **맞다.** `pim.access.sizes`가 그 값이다. bank/row 제약은 Layer 2 |
| `acc += tl.dot` = 누산기 설정 | **맞다.** 위치(UPMEM WRAM vs AiMX vACC)는 machine model 필드 |
| grid = 병렬성 | **맞다** |
| grid + split-K = 타일 → 유닛 배치 | **맞다** |

### 3.2 고쳐야 하는 것

**(1) `num_warps` 재사용은 안 된다.**

Triton에서 `num_warps`는 **하나의 tensor 연산을 warp들에 쪼개는 것**이고 layout을 컴파일러가
정한다(`TritonGPUConversion.cpp:29-37`, `AccelerateMatmul.cpp:321`). 모든 warp가 같은 `tl.dot`을
실행하고 누산기 하나를 레지스터 조각으로 나눠 갖는다 — **사용자에게 보이는 결합이 없다.**

제안한 의미("한 타일을 몇이서")는 위상이 반대다. 각 tasklet이 부분 타일에 대해 loop 전체를
돌고, 사적 누산기를 갖고, 명시적 결합이 필요하다.

`PIMOptions.num_warps`는 기본값 **0**이고 어떤 pass도 읽지 않는다. 이미 별도 옵션
`local_lanes`가 있고 그것이 옳은 선택이다.

**추가로 결정해야 할 것:** 현재 구현(`distributeTasklets`)은 tasklet에 **서로 다른 타일**을
준다. 주석이 명시한다 — "타일마다 쓰는 자리가 겹치지 않으므로 barrier가 없다."

| | 현재 구현 | 제안한 해석 |
|---|---|---|
| tasklet이 맡는 것 | 서로 다른 타일 | 한 타일의 일부 |
| 동기화 | 불필요 | barrier + 공유 누산기 |
| 성립 조건 | 유닛당 타일 수 ≥ lanes | 타일이 적어도 됨 |

큰 shape에서는 현재 방식이 낫고, `UPMEMTarget_Implementation.md` §10.1이 적은 "타일 수가 유닛
수보다 적을 때 손해"가 정확히 그 대가다. **작은 shape을 중요하게 본다면 둘 다 필요하다.**

**(2) `num_stages`는 지금 도달 불가능하다.**

TTGIR 전용 software pipeliner이고 NVIDIA backend에서만 호출된다
(`third_party/nvidia/backend/compiler.py:279-292`). `tt.range(num_stages=N)`이 만드는
`tt.num_stages` loop 속성은 `--triton-to-linalg-experimental`이 버린다(덤프 전체에 0건).

PIM에서 살리려면 세 가지가 필요하다.

- 계산 소스와 분리된 staging 메모리 — WRAM 있음
- **완료 토큰이 있는 비차단 전송** — UPMEM `mram_read`는 blocking이고, `pim.load`/`pim.copy`에
  async 형태나 토큰 피연산자가 없다
- 상주량을 `stages-1`배로 세는 residency 모델 변경

즉 옵션을 읽는 문제가 아니라 **dialect op 추가 + 상주량 모델 변경**이다. 매핑 규칙이 아니라
향후 과제로 분류해야 한다.

**(3) `SPLIT_K`라는 이름은 IR에 남지 않는다.**

constexpr은 함수 시그니처에서 **지워진다**(`code_generator.py:252`, `:282-284`).
`split_k.py`의 실측:

| 값 | 복원 가능? | 근거 |
|---|---|---|
| 이름 `SPLIT_K` | **불가** | `.ttir`/`.ttsharedir`에 grep 0건 |
| `BLOCK_M=128`, `BLOCK_N=128` | 가능 | `tensor<128x128xf32>` |
| `SPLIT_K=16` | **비율로만** | `BLOCK_K*SPLIT_K`가 `%c1024_i32` 하나로 접힌다. `BLOCK_K=1024, SPLIT_K=1`과 구별 불가 |
| "K가 grid 축 y로 쪼개졌다" | **가능, 구조로** | `tt.get_program_id y` + `c_ptr + pid_k*stride_ck` |

따라서 "사용자가 쓴 것을 읽는다"는 **구조 인식**이어야 한다 — 출력 주소를 stride하는 grid 축이
있는가. constexpr을 읽는 것이 아니다. 이름을 쓰고 싶다면 backend 옵션으로 축 이름과 함께
넘겨야 한다(`split = {"y": 16}`).

**(4) `tl.dot`의 최소 shape은 두 겹이다.**

Triton에 이미 backend 훅이 있다 — `min_dot_size(lhs, rhs) -> (m,n,k)`
(`semantic.py:1512-1522`). PIM backend는 지금 `(1,1,1)`을 돌려준다
(`compiler.py:270`). NVIDIA는 `(1,1,32)`를 돌려준다.

**단 이 훅은 하한(`>=`)만 표현하고 배수 조건을 표현하지 못한다.** AiMX의 n=4096이
*실행 단위*라면 조건은 `N % 4096 == 0`이고, `N=6000`이 이 훅을 통과한다. 배수 검사는
PIM 쪽 검증 pass에 따로 있어야 한다.

참고로 **저장소 어디에도 n=4096, k=512가 적혀 있지 않다.** 문서에 있는 것은
8ch × 16bank × 16alu와 `r += ALU*BANK*CH`뿐이다. machine model을 만든다면 이 값을 먼저
기록해야 한다.

### 3.3 빠진 것

| # | 항목 | 왜 필요한가 |
|---|---|---|
| 1 | **레이아웃 / 연속성** | `tl.load`의 안쪽 stride가 1인지가 DMA 효율을 정한다. G07(stride 전치)은 통과하고 G06(`tl.trans`)은 실패한다 |
| 2 | **호출 사이 버퍼 상주** | weight는 값이 안 바뀌므로 한 번만 올리면 된다. 커널 하나만 보면 안 보이고 constexpr로는 표현 못 한다. 함수 인자 속성이 필요하다 |
| 3 | **결합 위치** | `split_k.py`는 `c.sum(dim=0)`을 PyTorch에서 한다 → d2h가 SPLIT_K배. 측정치가 있다: `split=(8,8,8)`에서 kernel 31.31 ms, d2h 18.74 ms, combine 9.86 ms. 명시적으로 쓰면 컴파일러가 그것을 유닛/host로 옮길 여지가 줄어든다 |

3번이 이 접근의 **실질 비용**이다. 지금은 생성된 host C가 결합하므로(`emitForeachGroup`)
PyTorch 왕복이 없다. 사용자 명시로 옮기면 그 이점을 잃는다.

---

## 4. 설계안 심사 결과

| 설계안 | A 정확성 | B 커버리지 | C 2nd target | D 최적화 여지 | E 이행 | F 정직성 | 합 |
|---|---|---|---|---|---|---|---|
| 커널이 곧 스케줄 | 7 | 4 | **8** | **3** | 6 | **8** | **36** |
| 경계 이동 | 6 | 3 | 4 | 7 | 5 | 6 | 31 |
| 단일 IR 병합 | 6 | 2 | 5 | 5 | 6 | 7 | 31 |
| machine model | 6 | 3 | 5 | 5 | 4 | 7 | 30 |

제안하신 방향이 가장 높다. **두 번째 target 비용과 개념적 정직성에서 가장 강하고, 최적화
여지에서 가장 약하다.** 컴파일러가 고르지 않으면 비용 모델이 앉을 자리가 없어진다.

**네 심사가 공통으로 지적한 것:** 커버리지 점수가 전부 2~4다. 어느 설계안도 실제 실패를
고치지 못한다.

심사가 각각 꼽은 최고 가치 변경:

| 심사 대상 | 제안 |
|---|---|
| 경계 이동 | 축 이름 집합 대신 **symbolic affine map**으로 origin을 표현 |
| 단일 IR | 계층을 건드리지 말고 **raise 단계에 예산을 전부 쓰라** |
| machine model | 최소 shape을 사용자 BLOCK에 대한 *술어*가 아니라 컴파일러의 **retile 의무**로 |
| 커널이 곧 스케줄 | `grid == units`를 단정하지 말고 **grid→unit 매핑을 선언**하게 |

---

## 5. 권고 — 순서

### Step 0. 주소 표현을 바꾼다 (선행, 최고 가치)

`pim.access`의 `origin`을 축별 `Value` 목록이 아니라 **(grid 축, loop IV)에 대한 `AffineMap` +
런타임 값을 symbol로** 표현한다. 나눗셈/나머지 사실을 함께 기록한다.

이것이 열리는 것:

- 융합 1차원 grid (`pid // n`, `pid % n`) → `split_k.py`, `grouped_gemm.py`
- 3차원 batched grid → G11
- 포인터 전진 형태 → `gemm.py`, G03
- `offset = Σ origin·strides` 불변식이 런타임 stride에서도 검사 가능해진다
- `unit_data` 영역을 패턴 매칭이 아니라 **유도**할 수 있다

지금 구조에서 이것들이 막히는 이유는 계층이 둘이어서가 아니라, origin이 축당 `Value` 하나라
`floordiv`/`mod`를 표현할 수 없기 때문이다.

### Step 1. 검증 pass를 만든다

`Implementation_Audit.md` §6의 3번과 같다. `pim.units`의 region isolation, access ↔ unit_data
완전성, BLOCK ↔ machine 최소 shape 배수, DMA 정렬·최대 크기, 결합 연산자 합법성.

오류 메시지에 **무엇을 어떻게 고치면 되는지**를 넣는다.

```
error: BLOCK_N=1024 is not a multiple of the machine's GEMM width 4096
  note: aimx executes GEMM in units of n=4096, k=512
  note: set BLOCK_N to a multiple of 4096
```

"커널을 어떻게 쓰느냐에 따라 성능이 달라진다"를 제대로 지원하려면, 잘못 쓴 것이 조용히
느려지는 게 아니라 **명확히 실패**해야 한다.

### Step 2. machine model을 1급 입력으로

지금 흩어진 `num_units` / `local_memory_bytes` / `unit_memory_bytes` / `local_lanes` /
`host_threads`를 하나의 구조로 모으고, **빠진 것을 채운다.**

```
PIMMachine {
  units          : 2048
  memory: [ {name, bytes, granularity, max_transfer, contiguous_only} ... ]
  min_compute    : {m, n, k}      // 하한
  compute_granule: {m, n, k}      // 배수 — 지금 어디에도 없다
  lanes          : {max, subdividable, pipeline_depth}
  accumulator    : "local" | "register"
  combine_ops    : ["add", "mul"]
  compute_form   : "code" | "descriptor"
  legal_ops      : [...]          // AiMX legalization의 근거
}
```

`compute_granule`, `subdividable`, `compute_form`, `legal_ops` 넷이 §2에서 "계층이 필요한
셋 + 정의가 없는 둘"에 대응한다.

### Step 3. 계층을 옮긴다

Layer 1에서 `pim.units`·`pim.unit_data`·`capacity`·`unit_origin`·`--pim-partition`을 빼
Layer 2로 내린다. Layer 1에는 하드웨어 숫자가 하나도 남지 않게 한다.

**Step 0 없이 Step 3만 하면 얻는 것이 없다.** 심사 네 건이 일치한 지점이다.

---

## 6. 확인 방법

전부 읽기 전용이다.

```bash
# §1 재현 — 참조 커널의 산출물
cd /home/dlrkdals/TritonPIM/dump
for k in matmul_split_k_kernel matmul_grouped_kernel gemm_kernel; do
  echo "=== $k"
  for d in */; do [ -f "$d$k.ttsharedir" ] && { printf "  %s : " "${d%/}"; \
    ls "$d" | sed 's/.*\.//' | sort -u | tr '\n' ' '; echo; }; done
done

# §3.3 SPLIT_K 소실 확인
grep -c 'SPLIT_K\|BLOCK_M' dump/*/matmul_split_k_kernel.ttir

# §2 죽은 연산 계층 확인
grep -rn "pim.op" /home/dlrkdals/TritonPIM/triton_shared/lib/ | grep -v ClassifyCompute
```
