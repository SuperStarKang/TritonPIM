# PIM Common 메모리 lowering — 종합 정리

> `--triton-to-linalg-experimental` 출력(memref + linalg)을 target 무관한 **PIM Common IR**로 끌어올리는
> 작업의 전체 정리. 두 파트로 나눈다.
> - **파트 1**: 어떤 메모리 패턴이 **왜 문제**였고 **어떻게 바꾸는가** (패턴 → 변환).
> - **파트 2**: 그 패턴 매칭을 위해 **어떤 패스를 어떻게 구현**했는가 (패스 구조).
>
> 관련 문서: 패턴 전수(`MemoryAccess_Patterns.md`, `Compute_Patterns.md`), 설계
> (`MemoryLowering_Design.md`, `ComputeLowering_design.md`), 구현 기록(`PIMCommon_Conversion.md`).

---

## 0. 전체 그림

```
Triton 커널
  → ttshared IR              (memref + linalg, 남이 만든 부분)
  → PIM Common IR            (pim.access/load/store/copy/gather/scatter + linalg)  ← 이 문서
  → target IR (UPMEM/AiMX)   (아직)
```

PIM Common으로 올리는 일은 두 단계다.

| 단계 | 패스 | 하는 일 |
|---|---|---|
| **Stage A** | `--convert-ttshared-to-pim` | memref 관용구 → `pim.*`로 **끌어올림**(raising). 버퍼 소멸 |
| **Stage B** | `--pim-remove-buffers` | `pim.*` 사이의 **중복 전송 제거** |

실제 파이프라인: `--convert-ttshared-to-pim -cse --pim-remove-buffers -canonicalize`
(`-cse`가 Pure한 `pim.access`를 합쳐 "같은 descriptor"를 SSA 동일성으로 판별하게 하고, `-canonicalize`가
죽은 `pim.load`를 지운다.)

---

# 파트 1 — 어떤 패턴이 왜 문제였고 어떻게 바꾸나

문제는 두 종류다.
- **(A) 구조적 문제** — 모든 메모리 접근이 `reinterpret_cast + alloc + copy + to_tensor` 같은 **삼중구조 +
  불필요한 staging 버퍼**로 내려온다. 게다가 wrap/mask/gather는 그 위에 루프·subview·fill이 얹혀 복잡하다.
  → Stage A가 **하나의 `pim.*` op으로 접어** 버퍼와 골격을 없앤다.
- **(B) 중복 전송** — 값 의미론으로 바꾸고 나면 같은 데이터를 두 번 나르는 낭비가 드러난다(복사 커널의
  이중 전송, 반복 load 등). → Stage B가 제거한다.

## A. 구조적 문제 — 7개 접근 패턴 + store

핵심: **`reinterpret_cast`가 주소 명세, `alloc`+`copy`가 데이터 이동, 그 위 골격(루프/subview/fill)이
접근 종류**를 뜻한다. 이걸 `pim.access`(주소) + `pim.load/store`(전송)로 접고, 종류는 clause로 단다.

### ① strided (기본형) — 토대

- **입력 (판별: `copy(reinterpret_cast → alloc)`)**
  ```mlir
  %rc  = memref.reinterpret_cast %A to offset:[%o] sizes:[32] strides:[1]
  %buf = memref.alloc() : memref<32xf32>
  memref.copy %rc, %buf
  %t   = bufferization.to_tensor %buf
  ```
- **왜 문제**: `alloc`+`copy`가 매 load마다 생기고 downstream(one-shot-bufferize)이 못 지운다. 삼중구조.
- **바꿈**: 주소/전송 둘로 접고 버퍼 암묵화.
  ```mlir
  %d = pim.access %A offset %o sizes [32] strides[%c1] : memref<*xf32> -> !pim.access
  %v = pim.load %d : !pim.access -> tensor<32xf32>
  ```

### ② loop-carried — ①이 루프 안, offset이 iter_arg

- **왜 문제**: 매 iteration마다 alloc+copy.
- **바꿈**: `scf.for`는 **안 건드림**. 루프 안의 ① 관용구만 접히니 ②는 저절로 따라온다. offset은 iter_arg 그대로.

### ③ block pointer — memref에서 ①과 동일

- **판별/왜**: `tt.make_tensor_ptr` 출처지만 memref 레벨에선 `order`가 stride에 녹아 사라져 **①과 글자 그대로 같은 모양**.
- **바꿈**: ① 규칙이 그대로 처리한다. 새 코드 없음. (`order` clause를 만들어봤자 채울 값이 없어 죽은 코드가 됨.)

### ⑤ mask (경계 타일) — valid + pad

- **입력 (판별: `copy`의 양쪽이 `subview`)**
  ```mlir
  %rc   = reinterpret_cast %A ... sizes:[128]
  %buf  = alloc()
  scf.if %cond { linalg.fill ins(%pad) outs(%buf) }   // 채움값
  %vsub = subview %rc[0][%e][1]                         // 유효분만
  %bsub = subview %buf[0][%e][1]
  copy %vsub, %bsub
  %t    = to_tensor %buf
  ```
- **왜 문제**: 경계에서 유효한 부분만 읽고 나머지는 채워야 해서 fill+subview+scf.if가 붙는다.
- **바꿈**: 유효 길이(`valid`=subview 크기)와 채움값(`pad`=fill 입력)을 clause로.
  ```mlir
  %d = pim.access %A offset %o sizes [128] strides[%c1] valid[%e] pad %pad : f32 : ... -> !pim.access
  %v = pim.load %d
  ```
  **저장**은 대칭이되 `pad` 없음(쓰기라 채울 필요 없음): `materialize(extract_slice, subview)` → `pim.access{valid} + pim.store`.

### ⑥ row-gather (1축만 인덱스, 각 행은 연속) — gather_dim + index

- **입력 (판별: `copy(행별 reinterpret_cast → subview(alloc))` + rc offset이 `index_cast(tensor.extract)`)**
  ```mlir
  %alloc = alloc() : memref<8x8xf32>
  scf.for %i = 0 to 8 {
    %e   = tensor.extract %idx[%i]                 // 이 행의 시작 위치
    %ic  = index_cast %e
    %rc  = reinterpret_cast %A offset:[%ic] sizes:[1,8] strides:[1,1]
    %sub = subview %alloc[%i,0][1,8][1,1]
    copy %rc, %sub
  }
  %t = to_tensor %alloc
  ```
- **왜 문제**: 행 선택만 data-dependent인데 루프+행별 관용구로 펼쳐진다.
- **바꿈**: gather 축(`gather_dim`=subview offset이 동적인 축)과 인덱스 텐서(`index`)를 clause로. **루프째 삭제**.
  ```mlir
  %d = pim.access %A offset %c0 sizes [8,8] strides[%c1,%c1] gather 0 index %idx : tensor<8xi32> : ... -> !pim.access
  %v = pim.load %d -> tensor<8x8xf32>
  ```
  - 다차원도 됨: 3D면 subview offset이 `[0,%i,0]`이라 `gather_dim`이 dim 1로 정확히 잡힌다.

### ⑦ 원소별 gather/scatter (완전 random) — 전용 op

descriptor로는 표현 불가(원소마다 임의 위치) → 전용 op `pim.gather`/`pim.scatter`.

- **입력 (판별: `linalg.generic` body의 `tensor.extract(to_tensor(memref.cast(base)))` / `memref.store(memref.cast)`)**
  ```mlir
  %cast = memref.cast %A : memref<*xf32> to memref<?xf32>
  %src  = to_tensor %cast restrict
  %g = linalg.generic ins(%idx) outs(%empty) {
    ^bb0(%off, %o):
      %ic = index_cast %off
      %e  = tensor.extract %src[%ic]      // 임의 위치 로드
      yield %e
  }
  ```
- **바꿈**: `pim.gather %A index %idx : memref<*xf32>, tensor<64xi32> -> tensor<64xf32>`. scatter는 `memref.store` 버전 → `pim.scatter`.

**⑦ mask 있는 경우** — body가 `scf.if`로 감싸진다:
```mlir
%g = linalg.generic ins(%idx, %mask) outs(%empty) {
  ^bb0(%off, %m, %o):
    %r = scf.if %m -> f32 { ... tensor.extract ... yield %ex }
         else { yield %default }          // masked-out 채움값 (0.0 또는 other)
    yield %r
}
```
- **왜 문제(그리고 한때 버그였던 이유)**: 처음엔 `scf.if` 안의 `extract`만 보고 변환해 **마스크·기본값을 떨어뜨렸다**.
  → `extract/store`가 **body에 직접** 있을 때만(무조건 전송) 변환하는 가드를 넣어 막았고, 그다음 mask 지원을 추가.
- **바꿈**: `mask`(=ins의 i1 텐서)와 `other`(=else 분기 값, masked-out 채움값)를 clause로.
  (`default`는 C++ 예약어라 operand 이름은 Triton 용어인 `other`를 쓴다.)
  ```mlir
  %g = pim.gather %A index %idx mask %m : tensor<64xi1> other %d : f32 : ... -> tensor<64xf32>
       pim.scatter %v, %C index %idx mask %m : tensor<64xi1> : ...   // scatter는 other 없음
  ```

### 지원 안 하는 것 (그대로 둠, no-fallback)

- **④ wrap-around**: 경계 기준 앞/뒤 두 chunk로 쪼갠 `reinterpret_cast×2 + copy×2`. 깨끗한 "경계값"이 memref엔
  안 남고(minsi/divsi/remsi로 분해됨) 그 앞 tts 레벨에만 있어 복원이 지저분. 매칭 안 하고 **그대로 둔다**.

### 요약 표 (A)

| 패턴 | 판별 신호 | 결과 |
|---|---|---|
| ① strided | `copy(reinterpret_cast → alloc)` | `pim.access` + `pim.load` |
| ② loop-carried | ①이 `scf.for` 안(offset=iter_arg) | 루프 유지 + ① |
| ③ block | (①과 동일) | ① 규칙이 처리 |
| ④ wrap | `reinterpret_cast×2 + copy×2` | **미지원, 그대로 둠** |
| ⑤ mask | `copy(subview → subview)` | `pim.access{valid,pad}` + load/store |
| ⑥ row-gather | `copy(rc,subview)` + offset=`index_cast(extract)` | `pim.access{gather_dim,index}` + load, 루프 삭제 |
| ⑦ gather/scatter (무mask) | generic body의 `extract(to_tensor(cast))` / `store(cast)` | `pim.gather` / `pim.scatter` |
| ⑦ gather/scatter (mask) | 위 + body가 `scf.if{...}` | `pim.gather{mask,other}` / `pim.scatter{mask}` |
| store | `materialize(값, reinterpret_cast)` | `pim.access` + `pim.store` |

## B. 중복 전송 — Stage B

Stage A가 버퍼를 없앤 뒤 드러나는 낭비. `pim.load`=Read, `pim.store`=Write, `pim.access`=Pure 효과 위에서
값 의미론 rewrite로 제거한다.

| 문제 | 입력 | 바꿈 |
|---|---|---|
| **passthrough 이중 전송** | `%v = pim.load %ds; pim.store %v, %dd` (v는 store에만 쓰임) | `pim.copy %ds, %dd` (로컬 버퍼 없이 직접) |
| **반복 load** | 같은 `%d`를 또 `pim.load` | 앞 load 결과 재사용 |
| **RAW 왕복** | `pim.store %v,%d; %w = pim.load %d` | `%w`를 `%v`로 대체 |
| **죽은 전송** | 결과 미사용 `pim.load` | `-canonicalize`가 삭제 |

**안전장치 2개**:
1. "같은 descriptor" = 같은 `pim.access` SSA 값 (`-cse`가 Pure한 access를 미리 합쳐줌).
2. "사이에 쓰기 없음" = 두 op 사이를 훑어 **쓰기 효과 op**(store/scatter/copy, 효과 모르는 op)이 있으면 중단.
   읽기는 막지 않음.

---

# 파트 2 — 패턴 매칭을 위해 구현한 패스들

## 1. PIM dialect (`Dialect/PIM/IR/PIMDialect.td`)

target 용어(DPU/FPGA 등) 없는 target-neutral op 집합.

| op | 역할 | 효과 |
|---|---|---|
| `!pim.access` (타입) | 논리 주소 핸들(무비용). clause: `valid`/`pad`(⑤), `gather_dim`/`index`(⑥) | — |
| `pim.access` | 주소 명세 (reinterpret_cast 대응) | Pure |
| `pim.load` | 타일 로드 (버퍼 암묵) | Read |
| `pim.store` | 타일 저장 | Write |
| `pim.copy` | access→access 직접 전송 (passthrough 산출물) | Read+Write |
| `pim.gather` | ⑦ 원소별 로드. clause: `mask`+`other` | Read |
| `pim.scatter` | ⑦ 원소별 저장. clause: `mask` | Write |

- 선택 clause가 있는 op(`pim.access`/`gather`/`scatter`)은 `AttrSizedOperandSegments`로 optional operand를 표현.
  단 optional operand가 1개뿐인 op(`pim.scatter`의 mask)은 이 trait을 쓰지 않는다(개수로 판별, tablegen이 ≥2 요구).
- assembly는 optional group `( `키워드` $operand^ ... )?`으로, 없으면 안 찍힘.
- operand 이름에 C++ 예약어 금지: 채움값은 `default`가 아니라 `other`(Triton 용어)로 명명.

### verifier (`PIMDialect.cpp`)

clause 조합이 말이 되는지 op verify로 확인한다(잘못된 조합을 조용히 통과시키지 않게). 정적으로 확인 가능한 것만:

| op | 규칙 |
|---|---|
| `AccessOp::verify` | R1 `strides` 개수==`sizes` rank · R2 `valid` 쓰면 개수==rank · R3 `pad`⟹`valid`(저장은 valid만 OK) · R4 `gather_dim`⟺`index`+축 범위 |
| `GatherOp::verify` | `mask`⟺`other` (masked gather는 채움값 필요) |

- **안 하는 것**: `valid[i] ≤ sizes[i]` 같은 런타임 값 비교(valid가 SSA라 정적으로 못 봄), wrap 관련(미구현).
- 기존 pass가 만드는 모든 op은 이 규칙을 이미 만족 → `ops.mlir`이 positive, `invalid.mlir`이 negative 테스트.

## 2. Stage A 패스 — `--convert-ttshared-to-pim` (`TTSharedToPIMPass.cpp`)

함수마다 아래 matcher들을 순서대로 `walk`로 적용. 각 matcher는 판별 신호로 자기 패턴만 잡고, 아니면 그냥 넘어감
(예외 처리 없음). matcher들은 판별 신호가 서로 겹치지 않게 설계돼, 같은 op을 여러 matcher가 봐도 안전.

**matcher 목록** (파트 1 A와 1:1):
- ⑤ mask 로드 — `copy(subview,subview)` 앵커
- ⑥ row-gather — `copy(rc, subview(alloc))` + offset=`index_cast(extract)` 앵커
- ① 기본 로드 — `copy(rc, alloc)` 앵커
- store — `materialize`. dest가 reinterpret_cast면 ①③ store, subview면 ⑤ masked store
- ⑦ gather(무mask) — generic body의 `extract`가 **body에 직접** 있을 때
- ⑦ gather(mask) — generic(ins 2개) body가 `scf.if{extract}else{default}`
- ⑦ scatter(무mask/mask) — `store`가 body 직접 / `scf.if` 안
- (계산 분류는 이 raise 패스가 아니라 **별도 late 패스** `--pim-classify-compute`가 맡는다. 아래 §3.5)

**삭제 전략** (dangling 방지):
- `dead`: 반드시 죽는 관용구 op. 사용처→정의 순으로 쌓아 그 순서로 삭제.
- `roots`: `reinterpret_cast`/`memref.cast`처럼 다른 데서도 쓸 수 있는 op. 맨 끝에 **use_empty일 때만** 삭제.
- 이 분리가 "내가 만든 것만 치우고 남의 것은 안 건드림"을 보장.

**⑦ mask 가드의 핵심**: `extract`/`store`가 `scf.if` 안(=조건부)이면 무마스크 matcher는 건드리지 않고, 마스크
matcher가 `mask`+`default`를 뽑아 변환. 이 가드가 없으면 마스크가 유실됨(실제로 있었던 버그).

## 3. Stage B 패스 — `--pim-remove-buffers` (`PIMRemoveBuffersPass.cpp`)

```
1) passthrough: pim.store(pim.load(%ds), %dd) → pim.copy(%ds, %dd)
   조건: load 결과가 store에만 쓰임 + 같은 블록 + 사이에 쓰기 없음
2) load 재사용: pim.load를 뒤에서 앞으로 스캔(earlierValue)
   - 같은 %d의 앞 store 만나면 → 그 값 전달(forwarding)
   - 같은 %d의 앞 load 만나면 → 그 결과 재사용(dedup)
   - 쓰기 op 만나면 → 중단
   대체한 load는 바로 지움(연쇄 dedup이 최초 정의를 찾도록)
```
- **쓰기 판별** `mayWrite(op)`: `isMemoryEffectFree`면 통과, 효과 모르면 보수적으로 장벽, `MemoryEffects::Write`
  있으면 장벽. 읽기는 장벽 아님.

## 3.5 compute 기본 lowering — 별도 문서

연산 쪽(종류 분류 late 패스 `--pim-classify-compute` + DCE/LICM, fusion은 target)은
**`PIMCommon_ComputeLowering_Summary.md`** 에 정리했다. 전체 파이프라인:
`--convert-ttshared-to-pim -cse --pim-remove-buffers -canonicalize -loop-invariant-code-motion --pim-classify-compute`.

## 4. MLIR 패스 배선 (새 패스 추가 시 건드리는 7곳)

TableGen이 반복 코드를 자동 생성해줘, 실제로 쓰는 건 `runOnOperation()` 하나뿐이다.

| # | 파일 | 내용 |
|---|---|---|
| 1 | `Conversion/TTSharedToPIM/Passes.td` | `def X : Pass<"cli-name","OpType"> { constructor; dependentDialects }` |
| 2 | (자동) `Passes.h.inc` | `GEN_PASS_DECL_X` / `GEN_PASS_DEF_X`(→`impl::XBase`) / `GEN_PASS_REGISTRATION`(→`registerXPass`) |
| 3 | `TTSharedToPIM.h` | `#define GEN_PASS_DECL_X` + `createXPass()` 선언 |
| 4 | `XPass.cpp` | `#define GEN_PASS_DEF_X`; `struct XPass : impl::XBase<XPass>` + `runOnOperation()`; `createXPass()` |
| 5 | `lib/.../CMakeLists.txt` | 새 `.cpp`를 라이브러리 소스에 추가 |
| 6 | `tools/triton-shared-opt/RegisterTritonSharedDialects.h` | `registerXPass()` 호출(→ `--cli-name` 인식) |
| 7 | `pim_backend/backend/compiler.py` | 파이프라인에 `--cli-name` 끼움 |

이름 규칙: def `RemovePIMBuffers` → 매크로 `GEN_PASS_DEF_REMOVEPIMBUFFERS`, 부모 `impl::RemovePIMBuffersBase`,
등록 함수 `registerRemovePIMBuffersPass()`. (빌드 전엔 이 `.h.inc`가 없어 IDE가 base 클래스를 못 찾고 오류를 냄 — 정상.)

## 5. 두 패스 성격 비교

| | Stage A (`--convert-ttshared-to-pim`) | Stage B (`--pim-remove-buffers`) |
|---|---|---|
| 하는 일 | memref 관용구 → `pim.*` **raising** | `pim.*` 사이 중복 전송 제거 |
| runOnOperation | cluster를 `walk`로 매칭 → pim op 생성 → `dead`/`roots` 삭제 | pim op을 `walk` → RAUW/erase, 메모리효과로 안전성 판단 |
| 새로 만드는 op | access/load/store/gather/scatter | copy |
| 의존 dialect | PIM + arith | PIM |

**뼈대(배선 1~7)는 완전히 같고, 다른 건 `runOnOperation` 안 로직뿐.**

---

## 6. 테스트 (`test/Conversion/TTSharedToPIM/`, `test/Dialect/PIM/`)

| 테스트 | 확인 |
|---|---|
| `elementwise` / `loop_carried` / `matmul` | ①② + store, 계산 태그 |
| `masked` | ⑤ 로드(valid+pad)/저장(valid) |
| `row_gather` | ⑥ gather_dim+index, 루프 삭제 |
| `element_gather` | ⑦ 무mask gather/scatter |
| `element_gather_masked` | ⑦ mask gather(mask+other)/scatter(mask) |
| `remove_buffers` | Stage B passthrough/dedup/forwarding/장벽 |
| `classify` | 종류 태그 (gemm/splat/elementwise/broadcast/generation/reduce) |
| `Dialect/PIM/ops` | 모든 op·clause parse/print roundtrip (positive) |
| `Dialect/PIM/invalid` | verifier가 잘못된 조합 거부 (negative, `-verify-diagnostics`) |

실행: `lit --filter 'TTSharedToPIM|Dialect/PIM'` (총 10개).

---

## 7. 남은 일

- **④ wrap-around**: 두 조각(reinterpret_cast×2 + copy×2 into 한 alloc의 subview 2개)으로 내려옴.
  깨끗한 경계값이 memref엔 없어 보류. 표현안 후보 W1(축별 순환크기)/W2(wrap_dim+split+wrapped) 논의됨.
- **Stage B의 LICM**: 루프 밖으로 loop-invariant load hoist (블록/region 넘는 분석이라 보류).
- **`!pim.access` 파라미터형 타입**(shape/elemtype를 타입에). verifier는 구현 완료
  (`AccessOp::verify` R1~R4, `GatherOp::verify` mask⟺other; test/Dialect/PIM/invalid.mlir).
- **target 다리**: PIM Common → UPMEM/AiMX.
