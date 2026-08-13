# PIM Common — 연산(compute) lowering 종합 정리

> `--triton-to-linalg-experimental` 출력의 **연산**(linalg)을 target 무관한 PIM Common으로 처리하는 작업 정리.
> - 메모리 쪽 종합 정리 → `PIMCommon_Lowering_Summary.md`
> - 설계 근거 → `ComputeLowering_design.md`
> - 연산 패턴 전수 → `Compute_Patterns.md`

---

## 0. 연산은 메모리와 다르다 — linalg를 그대로 둔다

메모리는 삼중구조(`reinterpret_cast+alloc+copy`)를 새 op(`pim.access/load/store`)으로 끌어올렸지만,
연산은 이미 `linalg.matmul/reduce/generic`으로 깔끔하다. 그래서 **새 op을 안 만들고 linalg를 유지**한다.
PIM Common의 연산 처리는 딱 셋:

1. **named op(matmul/reduce/transpose) 유지** — target이 원시연산으로 매핑(scalar 루프로 안 풂).
2. **generic을 종류 태그(`pim.op`)로 분류** — target이 뭘 할지 알게.
3. **DCE + LICM만** 최적화.

연산을 loop로 푸는 것도, **fusion·분해도 target 몫**이다.

---

## 파트 1 — 연산 표현 & 분류

### 원칙: PIM Common = "무슨 종류(WHAT)", target = "어떻게(HOW)"

PIM Common은 **target 무관한 종류**만 태그한다.
- **합법성**(이 HW가 지원?) — legal set이 target 의존(AiMX엔 exp illegal, UPMEM엔 legal) → 태그 안 함.
- **opcode 매핑·분해**(add/sub/mul, exp→다항식) — target이 op body를 읽어서 판단 → 태그 안 함.

즉 PIM Common은 `math.exp`가 있으면 "pointwise(elementwise)"까지만 말하고, 그걸 분해할지는 target이 정한다.

### 종류별 판별 (`--pim-classify-compute`)

| 종류 `pim.op` | 판별 신호 | target 예 |
|---|---|---|
| `gemm` | `linalg.matmul` | AiMX `_matmul` |
| `gemv` | `linalg.matvec` | |
| `transpose` | `linalg.transpose` | 물리 재배치 |
| `reduce` | `linalg.reduce` 또는 reduction iterator | 리덕션 트리/반복 add |
| `fill` | `linalg.fill` | 상수 채움 |
| `index_generation` | generic body에 `linalg.index` | iota |
| `broadcast` | generic **입력 맵이 차원 drop**(결과 수 < 도메인 수) | AiMX `_bias` |
| `elementwise` | generic 전부 parallel + identity map | AiMX `_elewise` add/sub/mul |

- **extern**(exp/sin/erf 등 libdevice): body에 `math.*`가 있는 generic → `elementwise`로 태그. 함수·합법성은 target.
- 판별 순서: reduction → index_generation(linalg.index) → broadcast(입력맵 drop) → elementwise.

### PIM Common에서 안 하는 것 (왜 target인가)

- **fusion** — target-divergent(AiMX 소스 `matmul.c`/`elewise.c` 확인). AiMX elementwise는 **opcode당
  descriptor 하나**(`_elewise`: A읽기·B읽기·C쓰기)라 융합된 body를 되돌려야 하고, matmul+bias는 `_gemm`의
  **누산기(vACC) init**으로 AiMX가 자기 방식으로 함. → **target 또는 별도 late 패스.**
- **레이아웃 접기**(reshape/broadcast/transpose를 소비자로) — 물리 배치가 target마다 다름(인터리빙/padding/
  `srcA_buf` 재배치는 AiMX가) → op을 **그대로 두고** target이 물리화.
- **분해**(exp→다항식, div→Newton) — legal set이 target 의존.

---

## 파트 2 — 구현한 패스

### `--pim-classify-compute` (late 패스, `PIMClassifyComputePass.cpp`)

위 종류를 `pim.op`로 단다(치환 아님). **왜 맨 뒤(late)인가**:

> 분류 태그는 **찍는 순간의 스냅샷**이다. 뒤에 오는 패스가 op을 바꾸거나(canonicalize 접기) 새 op을
> 만들면(fusion이 둘을 합쳐 새 `linalg.generic` 생성) 태그가 낡거나 사라진다. 그래서 **모든 구조 변형 뒤
> 맨 마지막에** 분류를 돌린다 — 앞에서 op이 어떻게 바뀌었든 "지금 최종 op이 무슨 종류냐"만 보면 된다.

→ 분류를 raise 패스(`--convert-ttshared-to-pim`)에서 **떼어내** 독립 late 패스로 뒀다. (그래서 raise 단독으로는
`pim.op`가 안 붙고, 전체 파이프라인에서만 붙는다.)

### DCE + LICM (표준 패스)

- **DCE**: `-canonicalize` — 결과 미사용 연산 제거(값 의미론상 표준).
- **LICM**: `-loop-invariant-code-motion` — 루프 불변 연산을 밖으로. 순수 linalg는 pure라 표준 LICM이 안전.

### 파이프라인 (compiler.py `_ttsharedir_to_pim_common`)

```
--convert-ttshared-to-pim    # 1. 메모리 raise (분류 안 함)
-cse                         # 2. 버퍼 최적화(Stage B) — Pure한 pim.access 합침
--pim-remove-redundant-transfers         #    passthrough→copy / dedup / forwarding
-canonicalize                # 3. DCE
-loop-invariant-code-motion  #    LICM
--pim-classify-compute       # 4. late 분류 (맨 뒤)
```

### 테스트 (`test/Conversion/TTSharedToPIM/`)

| 테스트 | 확인 |
|---|---|
| `classify` | 종류 태그: gemm/fill/elementwise/broadcast/index_generation/reduce |
| `matmul` / `elementwise` | raise + late 분류 후 태그(`--convert-ttshared-to-pim --pim-classify-compute`) |

---

## 3. 손댄 파일

| 파일 | 내용 |
|---|---|
| `lib/.../TTSharedToPIM/TTSharedToPIMPass.cpp` | raise 패스에서 분류 제거 |
| `lib/.../TTSharedToPIM/PIMClassifyComputePass.cpp` | 새 late 분류 패스 |
| `include/.../TTSharedToPIM/Passes.td`, `TTSharedToPIM.h` | 패스 정의·선언 |
| `lib/.../TTSharedToPIM/CMakeLists.txt`, `tools/.../RegisterTritonSharedDialects.h` | 빌드·등록 |
| `pim_backend/backend/compiler.py` | 파이프라인에 LICM + late 분류 추가 |
| `test/.../classify.mlir` (신규), `matmul.mlir`·`elementwise.mlir` | 테스트 |
| `design/ComputeLowering_design.md` | fusion→target·late 분류·LICM 반영 갱신 |

---

## 4. 남은 것 (보류)

- **arg-reduction**(argmax/argmin): `linalg.reduce`인데 입력·출력 2개(값+인덱스)라 "결과>1"로 구분 가능.
  하지만 분류 추론 끝·greedy 디코딩 같은 **경계 연산**이라 드물고, AiMX native 없음(분해/host) → 보류.
  필요하면 "결과>1 → argreduce" 몇 줄로 추가.
- **scan**(cumsum): 순차 누적(`out[i]=out[i-1]⊕in[i]`)이라 linalg 신호가 덜 깔끔하고, SSM(Mamba) 외엔 드묾,
  AiMX native 없음 → 보류. 실제 커널 나올 때 재검토.
- **fusion 패스**: target에서(AiMX: `_gemm`/opcode, UPMEM: tasklet 루프). load-compute 태깅도 fusion과 함께.

> arg-reduction/scan을 태그로 안 나눠도 correctness 문제는 없다 — target이 실제 op 구조(결과 수 등)를 보고
> 구분할 수 있고, AiMX에선 어차피 둘 다 native가 없어 분해/host로 간다. 태그는 편의(자기설명) 힌트일 뿐.
