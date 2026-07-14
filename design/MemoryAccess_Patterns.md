# `--triton-to-linalg-experimental` 텐서 메모리 접근 패턴 정리

> TritonPIM lowering을 위한 레퍼런스. `microsoft/triton-shared`의 `--triton-to-linalg-experimental`
> 파이프라인이 출력하는 **텐서 메모리 접근 패턴**을 memref 레벨에서 전수 정리한다.

---

## 0. 문서 범위 & 대상 IR

- **대상 IR**: `--triton-to-linalg-experimental`를 **끝까지** 통과한 출력. 즉 `tt.*` / `tts.*`가
  모두 제거되고 vendor-neutral 표준 MLIR dialect만 남은 상태
  (`func` / `memref` / `linalg` / `arith` / `math` / `scf` / `tensor` / `bufferization` / `cf`).
- **제외**: `tptr` dialect. tptr은 **scalar-element 포인터**(`!tt.ptr<scalar>`)의 주소·메모리 연산에서만
  발생하며(scalar load/store, `int_to_ptr`/`ptr_to_int`, scalar ptr bitcast, control-flow로 선택된
  scalar base), 텐서/블록 포인터 접근은 어떤 경우에도 tptr이 아니라 memref로 간다.
  텐서 연산만 다루면 tptr은 신경 쓸 필요 없다.
- **핵심 진입점**: 모든 structured 접근의 패턴 매칭 진입점은 **`memref.reinterpret_cast`**.

### 파이프라인 위치

```
tt dialect
  ├─ (1) triton-to-structured        → tts.make_tptr / make_gather_scatter_tptr / load / store
  ├─ (2) triton-to-unstructured      → tts.gather / tts.scatter
  ├─ (3) triton-arith-to-linalg      → arith/math/dot/reduce → linalg
  ├─ (4) structured-to-memref        → tts.make_tptr/load/store → memref   ← ①~⑥
  ├─ (5) unstructured-to-memref      → tts.gather/scatter → memref          ← ⑦
  ├─ (6) triton-ptr-to-memref / triton-to-ptr → (scalar) tptr
  └─ (7) reconcile casts
```

---

## 1. 통합 원리: `reinterpret_cast` + `copy`

memref 레벨에서 **모든 structured 메모리 접근**은 하나의 primitive로 수렴한다.

```mlir
%view = memref.reinterpret_cast %base to offset: […], sizes: […], strides: […]
        : memref<*xT> to memref<…xT, strided<[…], offset: ?>>
```

`reinterpret_cast`가 access descriptor `{base, offset, sizes[], strides[]}`를 담는다.
- `offset`: static 또는 dynamic (SSA 값 / iter_arg)
- `sizes`: 타일 shape — **structured에서는 항상 static**
- `strides`: 축별 stride — static 또는 dynamic. broadcast 축은 stride `0`.

실제 데이터 이동:

| 방향 | memref 시퀀스 |
|---|---|
| **load** | `memref.alloc`(로컬 버퍼) → `memref.copy`(view→버퍼) → `bufferization.to_tensor` |
| **store** | `bufferization.materialize_in_destination`(tensor→view) |

나머지 6개 패턴은 전부 이 primitive의 변주다 — **reinterpret_cast 개수**, **offset 계산 방식**,
**copy를 감싸는 것**이 다를 뿐이다.

---

## 2. 접근 패턴 전수 (7범주) — 요약 표

규칙성(contiguity) 축 순서: ①=가장 규칙적(벌크 전송) → ⑦=완전 random(인덱스 전송).
**memref op 클러스터** 열이 lowering pass가 매칭할 판별 신호다.

| 범주 | tts 출처 (구조화 조건) | memref op 클러스터 — load (판별 신호) | descriptor 특성 | 의미 |
|---|---|---|---|---|
| **① Contiguous / strided** | `make_tptr` · `isStructuredPtr` (order=[], 모든 shape=0) | `reinterpret_cast`×1 → `alloc` → `copy` → `to_tensor` | sizes static; strides·offset static/dynamic; broadcast 축 stride=**0** | 규칙적 벌크 타일 (연속/스트라이드) |
| **② Loop-carried** | ①이 `scf.for` 내부 (+ `get_structured_state`) | `scf.for`(offset을 **index iter_arg**로 carry) + 루프 내 `reinterpret_cast`(iter_arg offset) → `copy` | sizes·strides static, **offset이 loop-variant** | 타일링 루프 (GEMM/attention K-loop) |
| **③ Block pointer** | `make_tptr` · `isBlockPtr` (order≠[]) / `tt.make_tensor_ptr`·`tt.advance` | `reinterpret_cast`×1 → `copy` (order가 레이아웃 결정), 루프 내 advance | 명시적 block shape/strides/block-shape, **order 지정** | block-ptr API 기반 타일 |
| **④ Wrap-around / modulo** | `make_tptr` · `isSplitPtr` (shape≠0) | **`reinterpret_cast`×2 → `copy`×2 → 한 버퍼의 `subview`×2** + `memref.dim`(런타임 분할점) | `shape`=wrap 경계; 앞/뒤 두 chunk offset | 순환/modular addressing (ring buffer, index wrap) |
| **⑤ Masked (clipped)** | `tts.load`/`store` + `mask_dims` (≠[]) | `arith.minsi`(extent clamp) + `subview`(clip) + `linalg.fill`(=`other`) + `copy` + `scf.if`(guard) | ① + per-dim **valid_extent**(mask_dims) | 경계 타일 (padding) |
| **⑥ Semi-structured gather (1축)** | `make_gather_scatter_tptr` | **`scf.for`(gather 차원)** + `tensor.extract`(행 offset) + `reinterpret_cast`(그 행) + `copy`(destination `subview`) | 1축 **index_tensor**(gather_scatter_offset), 나머지 축 strided | row-gather (embedding lookup, index_select — 각 벡터 연속) |
| **⑦ Unstructured gather (원소별)** | `tts.gather` / `tts.scatter` | **`scf.for` + `linalg.generic`(`parallel`) + 원소별 `tensor.extract`/`memref.store`** (identity affine_map) | per-element **index_tensor**(offset), optional mask+default | 완전 random access (임의 per-element 인덱스) |

---

## 3. 판별 축 (discriminators)

패턴 매칭 시 아래 세 축으로 7범주를 구분한다.

1. **`reinterpret_cast` 개수**
   - 1개 → ① ② ③
   - 2개 (한 버퍼의 subview 2개로 copy) → ④ (wrap-around)

2. **`reinterpret_cast`(+copy)를 감싸는 것**
   - `subview` + `linalg.fill` → ⑤ (mask)
   - `scf.for` + `tensor.extract` + **행별** `reinterpret_cast` → ⑥ (row-gather)
   - `scf.for` + `linalg.generic`(parallel) + **원소별** `tensor.extract`/`memref.store` → ⑦ (element-random)

3. **`offset` 성질**
   - static 상수 / 단일 SSA → ①
   - `scf.for` iter_arg (loop-variant) → ②
   - `tensor.extract`로 뽑은 index → ⑥ ⑦

---

## 4. 범주별 상세

### ① Contiguous / strided (기본형)

나머지 모든 범주의 토대. `tts.make_tptr`의 `isStructuredPtr` 모드(order=[], 모든 static_shape=0).

**tts 입력**
```mlir
%1 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [1, 5], offsets: [%o, 0],
       shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
%2 = "tts.load"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>,
       static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>) -> tensor<4x256xbf16>
```

**memref 출력**
```mlir
%view = memref.reinterpret_cast %arg0 to offset: [%o], sizes: [4, 256], strides: [1, 5]
        : memref<*xbf16> to memref<4x256xbf16, strided<[1, 5], offset: ?>>
%buf  = memref.alloc() : memref<4x256xbf16>
memref.copy %view, %buf
%val  = bufferization.to_tensor %buf restrict writable
```

**descriptor 유도**: tts의 `sizes`/`strides`/`offsets`가 그대로 reinterpret_cast의 `sizes`/`strides`/`offset`으로.
- stride `[1, …]` = inner-contiguous, `[s, …]` = strided
- broadcast(splat) 축은 stride `0` (예: `addptr_scalar_splat`, `addptr_scalar_broadcast`)
- offset이 컴파일타임 상수면 static, `arith.index_cast`된 런타임 값이면 dynamic (`offset: ?`)

**변종**: `addptr_dim1`(inner-dim stride), `addptr_mul_const_const`/`addptr_mul_value_const`(계산된 stride),
`addptr_reshape_broadcast`(reshape+broadcast), `addptr_2d_example`.

---

### ② Loop-carried (advancing window)

①이 `scf.for` 안에 놓여 offset이 매 iteration 전진하는 형태. GEMM/attention의 타일링 루프.

**memref 출력** (누산 tensor + offset index를 iter_args로 carry)
```mlir
%res:2 = scf.for %i = %c0 to %c12 step %c3
           iter_args(%acc = %init, %off = %o0) -> (tensor<4x256xbf16>, index) {
  %view = memref.reinterpret_cast %arg1 to offset: [%off], sizes: [4, 256], strides: [1, 5]
          : memref<*xbf16> to memref<4x256xbf16, strided<[1, 5], offset: ?>>
  %buf  = memref.alloc() : memref<4x256xbf16>
  memref.copy %view, %buf
  %t    = bufferization.to_tensor %buf restrict writable
  // ... compute ...
  %noff = arith.addi %off, %step : index
  scf.yield %nacc, %noff : tensor<4x256xbf16>, index
}
```

**핵심**: `sizes`/`strides`는 static 고정, **`offset`만 iter_arg로 전진**한다. tts 레벨의
`get_structured_state`가 loop-carried offset/stride를 iter_arg로 반출한 것의 최종형.

**변종**: `nested_loops`, `ridiculously_nested_loops`(offset iter_arg 다수),
`addptr_for_used_after_update`/`addptr_for_used_before_update`(update 시점 차이),
`tensor_indices_loop_iterargs_nested`.

---

### ③ Block pointer

`!tt.ptr<tensor<…>>` 타입 (Triton block-pointer API `tt.make_tensor_ptr` 출처).
`tts.make_tptr`의 `isBlockPtr` 모드(order≠[]).

**memref 출력**: 구조는 ①과 동일한 `reinterpret_cast` + `copy`. 차이는 descriptor가 block-pointer
API에서 오므로 **명시적 block shape/strides/block-shape**를 갖고, `order` 필드가 메모리 레이아웃 순서를
지정한다는 점. 루프 안에서는 `tt.advance`로 블록 창을 전진(→ ②처럼 offset iter_arg 전진).

**변종**: `block_ptr_advance`(루프 내 advance), `block_ptr_complex_offset`.

> 참고: memref op 형태만 보면 ①②와 구분이 안 될 수 있다. 구분이 필요하면 tts 레벨에서 `order`
> 필드 유무(또는 원본이 `tt.make_tensor_ptr`인지)로 판별해야 한다.

---

### ④ Wrap-around / modulo (split access)

`%x % N` 류 순환 접근. `tts.make_tptr`의 `isSplitPtr` 모드(`shape`가 non-zero = wrap 경계).

**tts 입력**
```mlir
%1 = tts.make_tptr %arg0 to sizes: [4, 4], strides: [%N, 1], offsets: [0, 0],
       shape: [%0, 0], order: [] : <f32> to tensor<4x4x!tt.ptr<f32>>   // shape[0]=%0 = wrap 경계
```

**memref 출력**: 경계 기준 앞/뒤 **두 contiguous chunk로 분할**, 각각 별도 reinterpret_cast + copy.
```mlir
%d      = memref.dim …                              // 런타임 분할점 (wrap 전 행 수)
%view1  = memref.reinterpret_cast %arg0 to offset: [%o1], sizes: [%d1, %n], strides: [%s, 1] …
%view2  = memref.reinterpret_cast %arg0 to offset: [%o2], sizes: [%d2, %n], strides: [%s, 1] …
%buf    = memref.alloc() …
%sub1   = memref.subview %buf[0,  0][%d1, %n][1, 1]  // 앞 chunk 목적지
%sub2   = memref.subview %buf[%d1,0][%d2, %n][1, 1]  // 뒤 chunk 목적지
memref.copy %view1, %sub1
memref.copy %view2, %sub2
```

**핵심**: reinterpret_cast **2개** + copy **2개** + `memref.dim`(런타임 분할점). 두 view는 같은 base에서
서로 다른 offset을 갖는다.

**변종**: `wraparound_stacked`/`wraparound_stacked_load`(dim0 wrap),
`wraparound_side_by_side`(dim1 wrap), `wraparound_static_shape`.
(`wraparound_unsupported_add_offset`는 구조화 실패 케이스.)

---

### ⑤ Masked (boundary-clipped)

경계 타일(마지막 타일이 block보다 짧음). `tts.load`/`store`의 `mask_dims`가 축별 유효 길이를 담는다
(boolean mask가 아니라 직사각형 sub-region).

**memref 출력 (load)**
```mlir
%e0    = arith.minsi %m0, %c4   : index              // dim0 유효 extent clamp
%e1    = arith.minsi %m1, %c256 : index              // dim1 유효 extent clamp
%view  = memref.reinterpret_cast %arg0 to offset: […], sizes: [4, 256], strides: […] …
%buf   = memref.alloc() : memref<4x256xf32>
linalg.fill ins(%other : f32) outs(%buf : memref<4x256xf32>)   // padding 값으로 선채움
%vsub  = memref.subview %view[0, 0][%e0, %e1][1, 1]  // 유효 영역만
%bsub  = memref.subview %buf [0, 0][%e0, %e1][1, 1]
scf.if %nonempty {
  memref.copy %vsub, %bsub                            // 유효분만 copy
}
```

**핵심**: `arith.minsi`(extent clamp) + `linalg.fill`(=`other` padding) + `subview`(clip) +
`scf.if`(guard). Triton의 predication이 **직사각형 clip + padding**으로 내려간다.

**Store는 비대칭**: masked store는 `linalg.fill`이 **없다**. 유효 영역만 `subview`로 clip해서 쓰면
되고(읽기가 아니라 쓰기라 padding 불필요), 나머지는 건드리지 않는다.

**변종**: `masked_ldst_1d`, `masked_ldst_2d`, `masked_ldst_sitofp_other`(other 값 + 타입 변환).

---

### ⑥ Semi-structured gather/scatter (1축만 non-contiguous)

`tts.make_gather_scatter_tptr` — 정확히 한 축만 data-dependent 인덱싱, 나머지 축은 contiguous.

**tts 입력**
```mlir
%5 = tts.make_gather_scatter_tptr %arg0 to sizes: [8, 8]
       gather_scatter_dim: 0 gather_scatter_offset: %4,   // 축 0의 행별 offset 텐서
       strides: [1, 1], offsets: [0, 0]
       : tensor<8xi32> <f32> to !tt.ptr<tensor<8x8xf32>>
%6 = "tts.load"(%5) … : (!tt.ptr<tensor<8x8xf32>>) -> tensor<8x8xf32>
```

**memref 출력**: gather 차원을 `scf.for`로 돌며, 각 행마다 offset을 뽑아 그 행(contiguous)을 copy.
```mlir
scf.for %i = %c0 to %c8 step %c1 {
  %idx  = tensor.extract %offsets[%i] : tensor<8xi32>       // 이 행의 시작 offset
  %view = memref.reinterpret_cast %arg0 to offset: [%idx…], sizes: [1, 8], strides: [1, 1] …
  %sub  = memref.subview %buf[%i, 0][1, 8][1, 1]            // 목적지 i번째 행
  memref.copy %view, %sub
}
```

**핵심**: 각 행은 **contiguous 벌크 copy**, **행 선택만 data-dependent**. ⑦(원소별)과 달리 행 내부는
규칙적이다.

**의미**: embedding lookup, index_select — 각 임베딩 벡터는 메모리상 연속.

**변종**: `gather_scatter_ptr_simple_to_memref`(→memref), `gather_scatter_ptr_to_linalg`(→linalg),
`gather_scatter_ptr_3d_to_linalg`, `gather_scatter_ptr_in_loop_to_linalg`.

---

### ⑦ Unstructured gather/scatter (원소별 random)

`tts.gather` / `tts.scatter` (구조화 완전 실패 → `triton-to-unstructured` → `unstructured-to-memref`).
offset이 임의의 per-element 정수 인덱스 텐서.

**memref 출력**: `scf.for` + `linalg.generic`(parallel)로 **원소별** 접근.
```mlir
#map = affine_map<(d0) -> (d0)>

%r:2 = scf.for %i = %c0 to %c2 step %c1
         iter_args(%a = %off_t, %b = %off_t) -> (tensor<64xi32>, tensor<64xi32>) : i32 {
  // gather: 원소별 tensor.extract
  %g = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
         ins(%offsets : tensor<64xi32>) outs(%init : tensor<64xf32>) {
    ^bb0(%off: i32, %o: f32):
      %idx = arith.index_cast %off : i32 to index
      %e   = tensor.extract %src[%idx] : tensor<?xf32>       // 임의 위치 로드
      linalg.yield %e : f32
  }
  // scatter variant: linalg.generic 내부에서 memref.store
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%b, %g : tensor<64xi32>, tensor<64xf32>) {
    ^bb0(%off: i32, %v: f32):
      %idx = arith.index_cast %off : i32 to index
      memref.store %v, %dst[%idx] : memref<?xf32>            // 임의 위치 저장
      linalg.yield
  }
  scf.yield … : tensor<64xi32>, tensor<64xi32>
}
```

**핵심**: identity affine_map + `parallel` iterator로 offset 텐서를 순회, 각 원소를 독립적으로
`tensor.extract`(gather) / `memref.store`(scatter). 연속성 가정 없음.

**mask 변종**: `gather_mask_no_other`, `gather_mask_with_other`(predicated + default value),
`gather_scatter_all_mask`.

---

## 5. Store 대칭성 (표는 load 기준)

store는 load의 `alloc`+`copy`+`to_tensor` 자리에 **`bufferization.materialize_in_destination`**
(tensor→view)이 들어간다. 범주별 비대칭:

| 범주 | load | store 차이 |
|---|---|---|
| ①②③ | reinterpret_cast + alloc + copy + to_tensor | reinterpret_cast + **materialize_in_destination** |
| ④ wrap | 2× copy into 2 subviews | 2× materialize into 2 view regions |
| ⑤ mask | subview + **fill** + copy + scf.if | subview + copy + scf.if — **fill 없음** (padding 불필요) |
| ⑥ row-gather | 행별 copy(view→buf subview) | 행별 copy(buf subview→view) 방향 반전 |
| ⑦ element | tensor.extract | **memref.store** |

---

## 6. Materialization & 버퍼링 (버퍼는 어디서 생기고 정리되는가)

> 정확성(value semantics)을 위해 `StructuredToMemref`가 **load마다 private 버퍼를 eager하게 생성**한다.
> 이 버퍼들은 레퍼런스 파이프라인에서 **정리되지 않으므로**, lowering 설계 시 반드시 인지해야 한다.

### 6.1 버퍼가 생기는 세 지점

| 지점 | 생성물 | 비고 |
|---|---|---|
| **load** | `memref.alloc` + `memref.copy` (private 버퍼) | 매 load마다. read-only fast path **없음** |
| **store** | 없음 (`materialize_in_destination`로 view에 직접 write) | 중간 버퍼 없음 — 효율적 |
| **compute 출력** | 각 `linalg` op의 `outs` = `tensor.empty` → `alloc` | op마다 자기 출력 버퍼 |

load 패턴 (예외 없이):
```mlir
%buf = memref.alloc()                                    // 매 load마다 새 버퍼
memref.copy %view, %buf                                  // reinterpret_cast view → 버퍼
%t   = bufferization.to_tensor %buf restrict writable    // restrict/writable = 핵심 의도
```

### 6.2 왜 복사하나 (정확성)

`reinterpret_cast` view는 원본 메모리를 **aliasing**한다. load 이후 겹치는 메모리에 store가 일어나면
aliased view를 읽을 때 변형된 데이터를 보게 된다(aliasing을 통한 WAR/RAW hazard). private 복사가 이를
끊어 **SSA value semantics**(load된 값이 원본과 독립)를 보장한다. `to_tensor`의 `restrict`(다른 것과
alias 안 함) + `writable` 플래그가 그 의도다.

→ 한계: **hazard가 없어도 무조건 복사**한다. 정확성을 위한 보수적 materialization이다.

### 6.3 Downstream은 정리하지 않는다

CPU 백엔드 2단계 파이프라인(`_ttsharedir_to_llir`)에는 buffer 최적화가 사실상 없다:

- **`one-shot-bufferize`는 이 copy를 못 지운다.** StructuredToMemref가 넣은 `memref.copy`/`memref.alloc`은
  이미 memref 레벨이라, tensor를 bufferize하는 one-shot-bufferize의 대상이 아니다. `to_tensor` /
  `materialize` 경계만 처리되고 explicit copy는 그대로 남는다.
- **`--eliminate-empty-tensors`가 비활성화**되어 있다 (compiler.py 주석 처리). compute 출력 버퍼
  (`tensor.empty`)를 소비자에 접어 중복 alloc을 없앨 수 있는 패스인데, early-return 다중 `func.return`
  버그로 꺼둔 상태다. 주석: *"performance on CPU isn't the main focus at the moment."*
- **buffer-hoisting / buffer-loop-hoisting / promote-to-stack / copy-removal / 버퍼 재사용 패스가 전무.**

→ load staging 버퍼와 compute 출력 버퍼 **둘 다 최적화 없이 그대로 남는다**. 이는 대상 IR의 고정된 특성이다.

### 6.4 "불필요한가"는 패턴에 따라 갈린다

핵심: **HW 백엔드 관점에서 load의 copy는 대부분 낭비가 아니라 필요한 데이터 이동(DMA)이다.**

- **load → compute → store (대다수)**: `copy(src→buf)`는 원본에서 로컬 스크래치패드로 가져오는 **필수
  staging 전송**. `reinterpret_cast`=descriptor, `copy`=DMA, `alloc`=로컬 타일 버퍼. 낭비 아님.

**진짜 낭비가 되는 경우:**

| # | 케이스 | 문제 |
|---|---|---|
| 1 | **load→store passthrough** (`tl.store(dst, tl.load(src))`) | `copy(src→buf)`+`copy(buf→dst)` 이중 복사. `copy(src→dst)` 하나면 충분 |
| 2 | **동일 영역 반복 load** | 매번 새 `alloc`+`copy`. `reinterpret_cast`는 CSE돼도 side-effect 있는 `alloc`+`copy`는 dedup 안 됨 |
| 3 | **masked load 과할당** (⑤) | 대부분 마스크아웃돼도 full-tile `alloc` + `linalg.fill` 항상 수행 |
| 4 | **버퍼 재사용/풀링 없음** | 순차 load가 각각 별도 버퍼. 스크래치패드 제한적 HW에서 압박 |
| 5 | **wraparound** (④) | full 타일 alloc 후 2회 copy — full materialization |

### 6.5 Lee 백엔드 설계 함의

이 버퍼들이 입력이고 레퍼런스는 정리해주지 않으니, 효율을 원하면 직접 처리해야 한다:

- **aliasing-aware copy elision**: store와 겹치지 않는 read-only load는 `reinterpret_cast` view를 compute가
  직접 읽게 해 `alloc`+`copy` 제거 (aliasing 분석 필요 — `restrict` 플래그가 힌트 제공)
- **load+compute fusion**: 중간 tensor를 없애 staging 버퍼 자체 제거
- **buffer 재사용/풀링**: 순차 타일이 버퍼 공유
- **또는** MLIR buffer 최적화 패스(`buffer-hoisting`, `eliminate-empty-tensors`,
  `optimize-allocation-liveness` 등)를 Lee 파이프라인에 직접 삽입

---

## 7. Lowering 프레이밍: 파라미터화된 descriptor

7범주는 **하나의 파라미터화된 access descriptor**의 특수 케이스로 접힌다. 공통 IR의 transfer op가
아래 필드를 가지면 7개를 모두 표현한다.

| descriptor 필드 | 활성화하는 범주 |
|---|---|
| `base` + per-dim `{offset, size, stride}` | ① 기본 (모두의 토대) |
| `offset`이 loop-variant (iter_arg) | ② loop-carried |
| descriptor 출처가 block API + `order` | ③ block ptr |
| `wrap_boundary` (per-dim) | ④ modulo |
| `valid_extent` (per-dim, mask) | ⑤ masked |
| `index_tensor` (1축) | ⑥ row-gather |
| `index_tensor` (per-element) | ⑦ element-random |

즉 공통 IR은 규칙성 축을 따라 **6종 transfer primitive**로 환원된다:

```
{ contiguous, strided, split, partial(masked), indexed-block, indexed-element }
```

이 스펙트럼이 규칙성 축과 정확히 대응한다. 하드웨어별 백엔드는 이 primitive들을 각자의 전송
메커니즘(DMA 디스크립터, scatter-gather 엔진 등)으로 매핑한다.

**패턴 매칭 진입점은 언제나 `memref.reinterpret_cast`** — 그 offset/sizes/strides의 static/dynamic
조합, 개수(1개=strided, 2개=wrap), 감싸는 것(`subview`+`fill`=masked, `scf.for`+`extract`=gather)으로
범주를 판별한다.

---

## 8. 소스 위치 (triton-shared)

| 대상 | 파일 |
|---|---|
| `tts` dialect 정의 (make_tptr 3모드, load/store, gather/scatter) | `include/triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.td` |
| 구조화 분석 & 실패 조건 (PtrAnalysis) | `lib/AnalysisStructured/PtrAnalysis.cpp` |
| ①~⑥ structured → memref | `lib/Conversion/StructuredToMemref/StructuredToMemrefPass.cpp` |
| ⑦ 유발 (tensor 구조화 실패 → gather/scatter) | `lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp` |
| ⑦ gather/scatter → memref | `lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp` |
| `tt.dot` → `linalg.matmul` (무조건, 분기 없음) | `include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp` (`MatmulConverter`) |
| 파이프라인 조립 | `lib/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimentalPass.cpp` |
| load/store converter (§6 eager alloc+copy 로직) | `lib/Conversion/StructuredToMemref/StructuredToMemref.cpp` (`LoadConverter` L620, `StoreConverter` L1016) |
| 2단계 파이프라인 & buffer 패스 (§6.3) | `backend/compiler.py` (`_ttsharedir_to_llir`, L83~) |
| 테스트 (범주별 concrete IR) | `test/Conversion/StructuredToMemref/`, `test/Conversion/UnstructuredToMemref/` |

---

## 부록 A. 참고 — `tt.dot`의 lowering

`tt.dot`은 shape/layout에 **관계없이 무조건 `linalg.matmul`** 로 내려간다 (분기 없음).
```mlir
%init   = tensor.empty() : tensor<MxNxf32>
%zeroes = linalg.fill ins(%c0) outs(%init)
%mm     = linalg.matmul ins(%A, %B) outs(%zeroes)   // 항상 이 형태
%res    = arith.addf %C, %mm                        // 누산기 C가 zero가 아니면 (isZeroTensor 검사)
```
- 누산기 `C`는 matmul 밖에서 `arith.addf`(정수면 `addi`)로 더함. `C`가 zero tensor면 생략.
- `linalg.matmul`은 rank-2 전용 — batched/3D 경로 없음. 전치는 앞단 별도 op(`linalg.transpose` 또는
  전치 stride 로드)로 materialize되어, matmul은 항상 canonical `(M,K)×(K,N)→(M,N)`.
