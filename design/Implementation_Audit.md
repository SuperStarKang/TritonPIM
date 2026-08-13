# PIM Lowering 구현 점검 기록

`PIMIR_Implementation.md`와 `UPMEMTarget_Implementation.md`가 기술한 내용을 실제 구현 코드와
대조한 결과다. 문서가 무엇을 만들었다고 말하는지가 아니라, **코드가 실제로 무엇을 하는지**를
적는다. 둘이 어긋나는 곳과 문서에 없는 동작을 중심으로 정리했다.

점검 대상: `lib/Conversion/TTSharedToPIM/` (8개 pass), `lib/Conversion/PIMToUPMEM/`,
`lib/Target/UPMEM/`, `lib/Dialect/PIM/`, `lib/Dialect/PIMTarget/UPMEM/`,
`pim_backend/backend/`, `test/`, `dump/`의 실제 산출물.

---

## 0. 요약

계층을 나눈 방식은 타당하다. 문제는 설계가 아니라 **계약이 강제되지 않는다**는 데 있다.

1. 정보 전달이 전부 discardable attribute를 통해서만 이루어지고, 어느 pass도 앞 pass가
   돌았는지 확인하지 않는다.
2. 매칭 실패가 30곳 넘게 조용한 `return`이라, 불완전한 IR이 그대로 흘러가 두 계층 뒤에서
   crash로 나타난다.
3. 그 결과 아래 §1의 결함들이 진단 없이 지나간다.

---

## 1. 확인된 결함

### 1.1 정적 stride면 `pim.access`의 `origin`이 빈다 — 실증 확인

**증상.** `reinterpret_cast`의 stride가 컴파일 시점 상수인 커널에서 `origin` 절이 아예
나오지 않는다. 그 access는 `pim.unit_data`가 생기지 않고, target에서
`byAccess.at()`(`PIMToUPMEMPass.cpp:614`)이 죽는다.

**원인.** `decomposeOffset`(`TTSharedToPIMPass.cpp:67-72`)이 stride를 **SSA 동일성**으로
비교하는데, 비교 대상인 `strides`는 `toIndex`(`:28-33`)가 만들어 넘긴 것이다.

```cpp
static Value toIndex(OpBuilder &b, Location loc, OpFoldResult ofr) {
  if (auto v = dyn_cast<Value>(ofr))
    return v;
  int64_t c = cast<IntegerAttr>(cast<Attribute>(ofr)).getInt();
  return b.create<arith::ConstantIndexOp>(loc, c);   // ← 상수마다 새로 만든다
}
```

`OpBuilder::create`는 CSE를 하지 않는다. 그래서 원본 IR의 `muli(%m, %c64)`가 가리키는
`%c64`와 새로 만든 상수는 서로 다른 `Value`다. 매칭이 절대 성립하지 않으므로 모든 항이
fallback(`:73-77`, "stride가 상수 1인 축")으로 떨어지고, 2차원이면 두 항이 같은 축에서
충돌해 `:79-80`이 빈 결과를 낸다.

**근거.** 덤프에서 정적 2차원 stride를 쓰는 커널을 찾아 대조했다.

```mlir
// dump/URKOGDYI.../triton_tem_fused__int_mm_0.ttsharedir:140
memref.reinterpret_cast %arg2 to offset: [%26], sizes: [32, 32], strides: [64, 1]

// 같은 커널의 pim_common:133 — origin 절이 없다
%52 = pim.access %arg2 offset %41 sizes [32, 32] strides[%c64, %c1] valid[%50, %51]
```

통과하는 GEMM은 전부 stride가 런타임 값(`strides[%2, %c1]`, `%2 = index_cast(K)`)이다.
그 `%2`가 offset 계산의 `muli`에도 쓰이므로 매칭이 성립한다. 덤프 통계는 동적 stride
74건 대 정적 2차원 stride 9건이다.

**의미.** 지금까지 `origin` 분실은 "주소를 loop `iter_arg`로 나르는 커널"의 문제로
문서화돼 있었다(`PIMIR_Implementation.md` §10.1). 실제로는 **leading dimension이 상수로
특수화된 모든 커널**이 같은 결과를 낸다. 위 커널은 `torch.compile`이 생성한 것이다.

**고치는 법.** `buildAccess`에서 `rc.getMixedStrides()`를 그대로 넘기거나, 비교를
`getConstantIntValue` 값 비교로 바꾼다. 매칭 실패 시 fallback 대신 **빈 `origin`을 내고
경고**해야 §1.3의 verifier와 짝이 맞는다.

**관련.** loop-carried 주소는 결과가 더 나쁘다. `origin`이 비지 않고 fallback이 offset
전체를 stride-1 축에 넣는다.

```mlir
// g03_pointer_advance.pim_common — 수치는 맞고 축 귀속이 틀렸다
%52 = pim.access %arg0 offset %arg15 sizes [32,16] strides[%2,%c1] origin[%c0, %arg15]
```

`Σ origin[d]·strides[d] = 0·%2 + %arg15·1 = %arg15`로 dialect 불변식이 **성립하므로**
verifier를 통과한다. m축 성분이 k축 자리에 들어간 채로 넘어간다.

---

### 1.2 masked load의 삭제 순서가 살아 있는 사용처를 지운다

**원인.** `TTSharedToPIMPass.cpp:221-229`.

```cpp
dead.append({toTensor, copy, vsub, bsub, alloc.getOperation()});
if (fill) {
  Operation *fillDead = isa<scf::IfOp>(fill->getParentOp()) ? fill->getParentOp()
                                                            : fill.getOperation();
  dead.push_back(fillDead);        // ← alloc 뒤에 들어간다
}
```

`dead`는 "사용처 → 정의" 순으로 쌓는 것이 계약인데(`:459` 주석) 이 한 곳이 반대다.
`findFill`(`:121-125`)이 `alloc->getUsers()`를 훑어 찾으므로 fill의 `outs`가 바로 그
alloc이다. 삭제 loop(`:463-465`)가 순서대로 지우면 `alloc`이 5번째로 지워질 때
`scf.if` 안의 `linalg.fill`이 아직 그것을 쓰고 있다.

**근거.** `test/Conversion/TTSharedToPIM/masked.mlir:16-18`이 정확히 그 모양이다.

```mlir
%buf = memref.alloc() : memref<128xf32>
scf.if %cond { linalg.fill ins(%pad : f32) outs(%buf : memref<128xf32>) }
```

MLIR의 `Operation::~Operation`은 `assert(use_empty() && "operation destroyed but still
has uses")`를 갖는다. assertion을 켠 빌드에서는 abort하고, 끄면 dangling operand다.

**고치는 법.** `dead.push_back(fillDead)`를 `alloc`을 넣기 전으로 옮긴다.

---

### 1.3 `pim.units`의 핵심 계약이 강제되지 않는다

`PIMDialect.td:142`의 trait은 `[NoTerminator]`뿐이고 **`IsolatedFromAbove`가 없다.**
"region 밖 = host, 안 = 유닛 하나"라는 계약이 문서상의 약속일 뿐이라, region 안에서
밖의 값을 쓰는 IR이 verifier를 그대로 통과한다.

`findings_gemm.md`의 C 계열 실패 세 개가 전부 이 한 가지 원인이다.

| case | 새어 나간 값 | 나타나는 곳 |
|---|---|---|
| G06 `tl.trans` | LICM이 밖으로 뺀 `tensor.empty` | `--upmem-to-host`가 죽은 op을 만남 |
| G08 epilogue scale | 커널 인자 `alpha`의 splat | `materializeExternals` 재구성 실패 |
| G09 epilogue bias | `tensor.expand_shape` 결과 | 같은 함수에서 crash |

**고치는 법.** `UnitsOp::verify`에 "region 안에서 쓰는 값 중 밖에서 정의된 것은 함수 인자와
module 수준 상수뿐"이라는 규칙을 넣는다. 세 실패가 전부 올바른 계층의 진단으로 바뀐다.

같은 자리에 "region 안 모든 input/output access에 대응하는 `unit_data`가 있다"를 함께
넣으면 §1.1의 crash도 진단으로 바뀐다. 지금은 반대 방향(`unit_data`인데 access 없음)만
검사한다(`PIMToUPMEMPass.cpp:172-174`).

---

### 1.4 `dropIterArgs`가 loop 속성을 전부 버린다

`PIMToUPMEMPass.cpp:678-692`가 lb/ub/step만으로 새 `scf::ForOp`를 만들고
`loop->getAttrs()`를 복사하지 않는다. `iter_args`를 가졌던 모든 loop가 `pim.grid_dim`과
`pim.reduction_combine`을 잃는다.

`distributeTasklets`(`:542-547`)는 그 뒤에 돌면서 `pim.grid_dim`을 다시 읽는다. 누산기를
나르던 grid loop는 tasklet 분배에서 조용히 빠진다.

---

### 1.5 grid 축의 결합 연산자가 `unit_data`에 반영되지 않는다

`--pim-analyze-reuse`는 grid 축에도 `combine`을 붙인다
(`PIMAnalyzeReusePass.cpp:328-331`, `test/.../analyze_reuse.mlir:70`에 실물이 있다).
그런데 `PIMPartitionPass.cpp:186-198`이 `combineOver`를 `isReduction` 분기 안에서만
채우고, `:327-329`가 `!combineOver.empty()`로 게이트한다.

결과적으로 결합이 필요한 grid 축을 쪼개면 `direction="out"`인 `unit_data`가 결합 없이
나오고, host가 부분 결과를 더하는 대신 덮어쓴다. 진단은 없다.

**단, 도달 조건이 좁다.** `gridCombine`은 reduction loop가 정확히 1개일 때만 정해지고
(`PIMAnalyzeReusePass.cpp:314-317`), 그 loop 자신이 reduction 축으로 emit되므로,
사용자가 `splits[k] == 1`을 주면서 그 grid 축을 쪼개야 한다. 잠복 결함이다.

---

### 1.6 `pim.co_resident`가 누산기를 빠뜨리고도 완전하다고 표시한다

`complete` 플래그는 `add` 안에서만 false가 된다(`PIMAnalyzeTransferPass.cpp:99-101`).
그런데 정적 shape가 아닌 tensor `iter_arg`는 `:123-124`에서 **플래그를 건드리지 않고**
건너뛴다. 누산기가 빠진 목록이 완전함으로 표시되고, `--pim-partition`의 용량 검사
(`:219-229`)가 그 틀린 합계 위에서 통과한다.

---

### 1.7 그 밖

| 항목 | 위치 | 내용 |
|---|---|---|
| `push_xfer` 바이트 수 | `UPMEMToHost.cpp:213` | 모든 DPU에 `_pN_n[0]`, 즉 유닛 0의 크기를 쓴다. 나머지는 계산해 놓고 버린다 |
| `gather` 버퍼 누수 | `UPMEMToHost.cpp:398-476` | `malloc`한 결과 버퍼를 어디서도 해제하지 않는다. 호출마다 `units × Π capacity × sizeof(elem)` |
| zero-trip hoisting | `PIMRemoveRedundantTransfersPass.cpp:119-120` | `scf.if`는 거부하면서 런타임에 0번 도는 `scf.for`는 그대로 뺀다. 범위 밖 host 읽기 |
| `indexing_maps` 무시 | `UPMEMToDevice.cpp:180` | `linalg.matmul`의 전치/broadcast map을 읽지 않고 항상 row-major를 가정 |
| subview stride 무시 | `UPMEMToDevice.cpp:138-140` | stride를 소스 shape에서 완전 밀집으로 재계산. 결과 포인터는 첫 원소에서만 맞다 |
| OpenMP 가드 파싱 | `UPMEMToHost.cpp:330` | `v3 / 8 * v3 / 16`이 C에서 `((v3/8)*v3)/16`으로 파싱된다 |
| `ceildivsi` 음수 | `CEmitter.h:102` | `(A+B-1)/B`는 음수 피제수에서 틀린다. 바로 옆 주석이 "음수를 지나므로 signed로 둔다"고 말한다 |
| tile 꼬리 유실 | `UPMEMToHost.cpp:338` | tile loop 상한이 `sizes[d]/tile[d]`로 절삭. 배수가 아닌 꼬리는 stage에서 채워지지 않고 unstage에서도 되돌려지지 않는다 |
| host var 절삭 | `UPMEMToHost.cpp:216` | host의 `index`는 `int64_t`인데 배열은 `int32_t`로 선언. 진단 없이 절삭 |
| `scalarType` 미검사 | `UPMEMToDevice.cpp:111,142,208`, `CEmitter.h:65,80,114` | 빈 문자열을 검사하지 않아 `" v3 = tiles_x;"` 같은 타입 없는 C가 나온다. f16 실패의 실제 지점 |

---

## 2. 문서에 없는 구현 사실

### 2.1 `pim.unit_data`의 `capacity`는 필수 operand다

`PIMDialect.td:168,175`에서 assembly format의 `capacity [ $capacity ]`가 optional이 아니다.
`PIMIR_Implementation.md` §4.4의 예제 IR과 필드 표에 `capacity`가 없어 **그대로는
파싱되지 않는다.**

의미는 `PIMPartitionPass.cpp:299-309`에서 나온다.

```
sizes[d]    = unitTrips[drv] × tile      // 이 유닛이 실제로 맡은 크기
capacity[d] = unitPer[drv]  × tile       // 배정량. 모든 유닛이 같다
```

반복이 유닛 수로 나누어떨어지지 않으면 마지막 유닛에서 둘이 갈린다. UPMEM의
`dpu_push_xfer`가 모든 DPU에 같은 바이트 수를 요구하므로 `upmem.stage`의 `pad_to`가
`capacity`에서, `unstage`의 축 간격도 `capacity`에서 온다.

### 2.2 `pim.unit_origin`

`PIMPartitionPass.cpp:129`가 유닛 로컬 인덱스를 전역 인덱스로 되돌리는 `arith.addi`에
붙이는 `UnitAttr`. `PIMToUPMEMPass.cpp:207-212`가 그 LHS를 상수 0으로 바꿔 device 주소를
유닛 로컬로 만든다.

`PIMIR_Implementation.md` §7의 속성 목록에 없다. target 관심사가 common 계층에 새어 든
자리이기도 하다.

### 2.3 `--pim-partition`은 host 메타데이터를 device region 안에 만든다

`b.setInsertionPoint(outermost)`(`:250`)가 nest를 region 안으로 옮긴 **뒤에** 실행된다.
그래서 `narrowLoop`의 모든 산술과 모든 `pim.unit_data`가 `pim.units` 안에 만들어진다.

이것은 결함이 아니라 소비자의 요구다. `PIMToUPMEMPass.cpp:117-123`이 region의 **직속
자식**에서 `unit_data`를 모으고, `:216`의 `OpBuilder hb(units)`가 host 코드를 만들면서
밖으로 끌어낸다. 다만 dialect 문서에 그 사실이 없어 IR만 읽으면 계약 위반으로 보인다.

### 2.4 `gridCombine`은 reduction loop가 정확히 1개일 때만 정해진다

`PIMAnalyzeReusePass.cpp:314-317`. 0개거나 2개 이상이면 `needCombine`의 모든 grid 축이
조용히 `split_axes`에서 빠진다. 그러면 `splits` 개수가 안 맞아
`pim-partition: splits 3 but pim.split_axes has N` 오류가 **원인과 다른 pass에서** 난다.

### 2.5 reduction 축 이름과 순서

- 1개면 `"k"`, 여럿이면 `"k0"`, `"k1"`, ... (`:345-347`)
- 인덱스는 `reductionLoops`의 **post-order(안쪽 먼저)** 위치다
- 부수 효과가 있어 건너뛴 loop는 번호에 **구멍**을 남긴다 (`k0`, `k2`)
- `PIMPartitionPass.cpp:159`의 주석은 "바깥→안쪽"이라고 말한다. 코드와 반대다.
  `PIMAnalyzeReusePass.cpp:234`도 같은 post-order를 쓰므로 우연히 짝이 맞을 뿐이다

### 2.6 `--pim-normalize-accumulator`의 접기 조건은 6개다

문서 §5.4의 표는 5개인데 `compute.getNumDpsInits() == 1`
(`PIMNormalizeAccumulatorPass.cpp:110`)이 빠져 있다.

### 2.7 `--pim-classify-compute`의 분류 경계

`(d0,d1)->(d1,d0)` 같은 **치환 map은 `elementwise`로 태그된다.** `getNumResults()`가
`getNumDims()`와 같아 broadcast 검사(`:42-44`)를 통과하고, 출력 map은 아예 보지 않는다.
주석 `:45`의 "전부 parallel + identity map = pointwise"와 코드가 어긋난다.

분류하지 못하는 것에는 태그를 붙이지 않는다. `"unknown"`이 없으므로
`linalg.BroadcastOp`·`BatchMatmulOp`·`linalg.add` 같은 named op이 "계산 op이 아님"과
구별되지 않는다.

### 2.8 `--convert-ttshared-to-pim`은 인식 실패를 남긴다

관용구를 못 알아보면 원본 memref op이 그대로 살아남고, 그 뒤 어디에서도 "staging 관용구가
남지 않았는가"를 검사하지 않는다. 실물이 있다.

```mlir
// dump/URKOGDYI.../triton_tem_fused__int_mm_0.pim_common:109-116
%reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%62], ...
memref.copy %reinterpret_cast, %subview : ...
```

이 커널 하나가 §1.1과 §2.8을 동시에 보여 준다.

---

## 3. 속성 생명주기

13개 전부 discardable attribute다. ODS 선언이 없으므로 op을 다시 만드는 어떤 rewrite도
조용히 버릴 수 있다.

| 속성 | 생산 | 소비 | 없으면 |
|---|---|---|---|
| `pim.grid_dim` | build-grid:92 | reuse, remove-transfers, partition, UPMEM | 조용히 건너뜀 — hoisting이 합법이 되고 축이 사라짐 |
| `pim.grid_dims` | reuse:258 | partition:264,318 | **null 역참조 crash** |
| `pim.access_id` | reuse:226 | transfer:105, partition:280, UPMEM:154,613 | UPMEM 두 곳에서 **crash** |
| `pim.role` | reuse:228 | partition:279 | 조용히 `unit_data` 없음 |
| `pim.invariant_grid_dims` | reuse:285 | partition:316 | 복제 최적화 상실 (느리지만 정확) |
| `pim.reduction_combine` | reuse:244 | reuse, partition:162, UPMEM:287 | 축 탈락. 개수가 밀리면 `reductionLoops[rIdx]` **범위 밖** |
| `pim.split_axes` | reuse:289 | partition:150 | 조용히 no-op return |
| `pim.co_resident` | transfer:169 | partition:220 | **용량 검사 조용히 생략** |
| `pim.unit_origin` | partition:129 | UPMEM:308 | device가 **전역 주소로 MRAM 접근** |
| `pim.execution_model` | reuse:207 | **없음** | write-only |
| `pim.tile_bytes` | transfer:166 | **없음** | write-only |
| `pim.op` | classify:56 | **없음** | write-only |
| `upmem.host_threads` | UPMEM:217 | UPMEMToHost:77 | `threads=1` — 정확, 느림 |

**세 개가 write-only다.** 분류 pass 전체와 tile 바이트 분석의 절반이 아무도 읽지 않는
정보를 만들고 있다.

**낡을 수 있는 것:**

- `pim.access_id`는 쓰는 시점의 walk 순서 dense index다. 그 뒤 `-cse`/`-canonicalize`가
  파이프라인에 있고(`compiler.py:133-134`), `PIMToUPMEMPass.cpp:155`의 `tileOf[id]`는
  충돌 시 마지막이 이긴다
- `pim.grid_dim` / `pim.reduction_combine`은 `dropIterArgs`가 통째로 버린다 (§1.4)
- `pim.unit_origin`은 특정 `arith.addi` 하나에 붙은 `UnitAttr`이라 `-cse`가 그 add를
  접으면 사라진다
- `pim.invariant_grid_dims`는 `pim.grid_dims`에 대한 **위치 인덱스**다. `-canonicalize`가
  1회만 도는 grid loop를 제거하면 인덱스가 다른 축을 가리킨다

**어느 pass도 앞 pass가 돌았는지 확인하지 않는다.** `--pim-partition`만 `pim.split_axes`
부재를 검사하는데 그마저 진단 없는 `return`이다. 이 pass 하나만 raw ttshared IR에 돌리면
exit code 0으로 분할되지 않은 IR이 나온다.

---

## 4. 테스트 상태

lit 테스트 4곳이 현재 구현과 어긋난다. `UPMEMTarget_Implementation.md` §10.4가 적은 것보다
범위가 넓다.

| 파일 | 단언 | 실제 |
|---|---|---|
| `partition.mlir:90-93` | `splits=1,1,1`이면 `pim.units` 없음 | 만든다. `PIMPartitionPass.cpp:166-168` 주석이 "유닛 수가 1인 축도 넣는다"고 명시 |
| `partition.mlir:84-88` | `ceildivsi`/`minsi`/`maxsi`가 `unit_data` 뒤 | `narrowLoop`가 `buildUnitData`보다 먼저 도므로 앞이다 |
| `convert.mlir:8-15` | `mram` 3개 다음 `pitch` 3개 | 버퍼마다 `mram`+`pitch`를 번갈아 낸다 |
| `convert.mlir:26-28` | alloc이 i8/i8/i32 순서 | pre-order가 `linalg.fill`을 먼저 만나 i32/i8/i8 순서다 |

`codegen.mlir:68`의 "행마다 DMA 하나"는 행 단위 DMA 시절 문구가 남은 것이다.

---

## 5. 관문 통과율

덤프 45개 디렉터리 중 빈 것 5개를 제외한 40개 기준이다.

| 산출물 | 개수 | 못 넘은 곳 |
|---|---|---|
| `.ttir` / `.ttsharedir` | 40 / 40 | — |
| `.pim_common` | 25 | **15개** — attention ladder 11개(matmul 없음), `g04`·`g11`(splits 개수), 기타 2 |
| `.target_ir` | 21 | **4개** — `g03`(origin), `g08`·`g09`(units 밖 값), Inductor GEMM 1개 |
| `.pimobj` | 17 | **4개** — `g06`(transpose), `g10`(`linalg.generic`), `g12`(f16), 기타 1 |

---

## 6. 우선순위

앞 두 개는 성격이 다르다. 1번은 정확성(지금 조용히 어긋난다), 2번은 안정성(빌드 설정에
따라 abort한다).

| 순위 | 할 일 | 위치 | 여는 것 |
|---|---|---|---|
| 1 | `decomposeOffset`의 stride 비교를 값 비교로 바꾸고, 실패 시 fallback 대신 빈 `origin` + 경고 | `TTSharedToPIMPass.cpp:28-33, 67-77` | G03, Inductor 커널, 정적 stride 전부 |
| 2 | `dead.push_back(fillDead)`를 `alloc` 앞으로 | `TTSharedToPIMPass.cpp:221-229` | masked load 경로의 abort |
| 3 | `UnitsOp::verify`에 region isolation + `unit_data` 완전성 검사 추가 | `PIMDialect.cpp:69-77` | G06/G08/G09, 그리고 §1.1의 crash가 진단으로 바뀜 |
| 4 | splits를 자리 대신 `source`로 짝짓기 | `PIMPartitionPass.cpp:153` | G04, G11, elementwise, reduction, attention |
| 5 | `dropIterArgs`가 `loop->getAttrs()`를 복사 | `PIMToUPMEMPass.cpp:678-692` | tasklet 분배 누락 |
| 6 | `linalg.generic` device lowering + `CEmitter`의 float 연산 | `UPMEMToDevice.cpp:178-185`, `CEmitter.h:88-120` | G10, attention softmax |
| 7 | grid 축 `combine`을 `unit_data`에 반영 | `PIMPartitionPass.cpp:186-198, 327-329` | 잠복 정확성 |
| 8 | `capacity`와 `pim.unit_origin`을 `PIMIR_Implementation.md`에 반영 | 문서 §4.4, §7 | 문서 예제가 파싱되게 |

**3번을 4번보다 앞에 두는 이유**는 비용이 작고 효과가 넓기 때문이다. verifier 규칙 두
줄이 target에서 나는 crash 네 지점을 "PIM Common IR이 불완전하다"는 정확한 진단으로
바꾼다. 지금은 원인이 `--convert-ttshared-to-pim`에 있는데 증상이 두 pass 뒤 LLVM
assertion으로 나타나, 매번 IR을 손으로 대조해야 원인을 찾을 수 있다.

---

## 7. 확인 방법

전부 읽기 전용이다. 빌드와 실행은 직접 진행한다.

```bash
# §1.1 재현 — 정적 stride 커널의 origin 부재
grep -h "reinterpret_cast" dump/*/*.ttsharedir | grep -oE "strides: \[[0-9]+, [0-9]+\]"
grep -n "pim.access" dump/URKOGDYI*/triton_tem_fused__int_mm_0.pim_common

# §4 테스트
cd /home/dlrkdals/TritonPIM/triton_shared
cmake --build <build-dir> --target check-triton-shared-lit-tests

# 단계별 IR
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/home/dlrkdals/TritonPIM/dump \
TRITON_ALWAYS_COMPILE=1 python workspace/Attention/run_ladder.py --suite gemm
```
