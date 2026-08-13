# `--triton-to-linalg-experimental` 연산(compute) 패턴 정리

> TritonPIM lowering을 위한 레퍼런스. `microsoft/triton-shared`의 `--triton-to-linalg-experimental`
> 파이프라인이 출력하는 **연산 패턴**을 linalg/tensor/arith/math 레벨에서 전수 정리한다.
> (자매 문서: 메모리 접근 패턴 정리 → `triton-shared-memory-access-patterns.md`)

---

## 0. 문서 범위 & 대상 IR

- **대상 IR**: `--triton-to-linalg-experimental`를 끝까지 통과한 출력. `tt.*` / `tts.*`가 모두 제거되고
  표준 MLIR dialect만 남은 상태 (`linalg` / `tensor` / `arith` / `math` / `scf` / `func` / `cf`).
- **담당 패스**: 연산 lowering은 대부분 **`--triton-arith-to-linalg`** (`TritonArithToLinalg`)가 수행.
  등록된 conversion pattern들이 ground truth다.
- **핵심 진입점**: 대부분의 pointwise 연산은 **`linalg.generic`**. 구조화 연산은 named op
  (`linalg.matmul` / `linalg.reduce` / `linalg.transpose`), 순수 레이아웃은 `tensor.*`.

### 담당 패스 & 패턴 등록 (`TritonArithToLinalg.cpp`)

```
populateTritonArithToLinalgConversionPatterns:
  Broadcast / Transpose / MakeRange / ExpandDims / Bitcast / Call / MulHiUI
  PreciseSqrt / PreciseDiv / Cat / Split / Join / FpToFp / Clamp / Matmul
  Splat / Unsplat / DenseConstant / CumSum / Reshape
  populateExternElementwiseOpToMLIROps        # libdevice → math dialect
  ArgMin / ArgMax / Reduce
  linalg::populateElementwiseToLinalgConversionPatterns   # ← catch-all, 마지막에 시도
```

마지막 줄이 핵심: upstream MLIR 패턴이 **남은 모든 arith/math 텐서 연산을 `linalg.generic`으로** 변환한다
(마지막에 등록되어 Triton 전용 converter가 우선).

---

## 1. 통합 원리: `linalg.generic`가 pointwise의 primitive

메모리에서 `reinterpret_cast`가 진입점이었듯, 연산에서는 `linalg.generic`이 pointwise 연산의 보편
진입점이다. **scalar body(arith/math)만 바뀌고 골격은 동일**하다.

```mlir
#id = affine_map<(d0) -> (d0)>
%init = tensor.empty() : tensor<128xf32>
%r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
       ins(%a, %b : tensor<128xf32>, tensor<128xf32>) outs(%init : tensor<128xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
} -> tensor<128xf32>
```

- **구조화 연산**(contraction/reduction/permutation)은 named op으로 노출 → HW 연산 유닛에 직접 매핑 용이
- **순수 레이아웃**(reshape/slice/concat)은 `tensor.*` → 데이터 이동 없는 metadata

---

## 2. 연산 패턴 전수 (연산 구조 축)

| 범주 | Triton 출처 | 출력 op 클러스터 (판별 신호) | body / 특성 |
|---|---|---|---|
| **A. Pointwise** | 모든 arith/math 텐서 연산; `mulhiui`, `precise_sqrt/divf`, `clampf`, `fp_to_fp`, `bitcast` | `linalg.generic`(**전부 parallel**) + `tensor.empty` + `linalg.yield` | body = `arith.*`(add/mul/sub/div/cmp/select, cast) + `math.*`(exp/sqrt/log/trig/erf…) |
| **A′. Extern (libdevice)** | `tt.extern_elementwise` | `linalg.generic` + `math.*` | libdevice → MLIR `math` dialect 매핑 (대규모 함수셋) |
| **B. Contraction** | `tt.dot` | **`linalg.matmul`** + `linalg.fill`(0-init) + `tensor.empty` + (누산기≠0) `arith.addf` | 무조건 matmul, rank-2 전용, 분기 없음 |
| **C. Reduction** | `tt.reduce` | **`linalg.reduce`** + `linalg.fill`(init) + `tensor.empty` | tt.reduce body를 linalg.reduce에 **복제** |
| **C′. Arg-reduction** | `tt.reduce`(argmax/argmin) | `linalg.reduce` + `linalg.generic` + **`linalg.index`** + `tensor.extract`/`insert` | 값+인덱스 동시 추적 (가장 복잡) |
| **D. Scan** | `tt.scan` / cumsum | **`linalg.generic`**(index-carried) + `tensor.empty` | reduction 아님 — 모든 부분합 유지 |
| **E. Permutation** | `tt.trans` | **`linalg.transpose`** (permutation 배열) | 축 재배열 |
| **F. Broadcast** | `tt.broadcast` | `linalg.generic`(broadcast indexing map) + `tensor.empty` | 차원 투영으로 확장 |
| **F′. Splat** | `tt.splat` | **`linalg.fill`** + `tensor.empty` | scalar → tensor 채움 |
| **G. Shape (metadata)** | `tt.expand_dims` / `tt.reshape` | `tensor.expand_shape` / `tensor.collapse_shape` | 순수 view — 연산 없음 |
| **G′. Concat / slice** | `tt.cat` / `tt.join` / `tt.split` | `tensor.concat`+`extract_slice` / `insert_slice` / `extract_slice` | 순수 레이아웃 |
| **H. Generation** | `tt.make_range` | `linalg.generic` + **`linalg.index`** (iota) | index → value |
| **H′. Constant** | `arith.constant`(dense) | `tensor.empty`(+`fill`) | dense 상수 materialize |
| **I. ABI / control** | `program_id`/`num_programs`, `tt.assert`, `tt.call` | func args / `cf.assert` / `func.call` | 연산 아님 |

---

## 3. 범주별 상세

### A. Pointwise (elementwise)

모든 텐서 element-wise 연산. body에 실제 scalar 연산이 들어간 `linalg.generic`.

**출력** (binary 예)
```mlir
%r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
       ins(%a, %b) outs(%init) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32          // body가 곧 연산
    linalg.yield %s : f32
}
```

**body에 나타나는 scalar 연산**
- `arith.*`: `addf/mulf/subf/divf/cmpf/select`(binary·ternary), `addi/muli/subi/…`(정수),
  타입 변환 `extf/truncf/sitofp/fptosi/uitofp/bitcast`
- `math.*`: `exp/sqrt/log/sin/cos/tanh/erf/rsqrt/floor/ceil/…`

**여기로 접히는 Triton 전용 연산**
- `tt.clampf` → `arith.maximumf`/`minimumf` (또는 maxnumf/minnumf)
- `tt.fp_to_fp` → `arith.extf` / `arith.truncf`
- `tt.mulhiui` → 정수 상위비트 곱 (arith 조합)
- `tt.precise_sqrt` → `math.sqrt`, `tt.precise_divf` → `arith.divf`
- `tt.bitcast`(텐서) → `arith.bitcast`

**변종**: `convert_1d/2d_elemwise_arith_binary/unary/ternary`.

---

### A′. Extern elementwise (libdevice)

`tt.extern_elementwise`(NVIDIA libdevice 호출)는 `linalg.generic` body 안의 **MLIR `math` dialect**로 매핑.

**출력**
```mlir
%r = linalg.generic {…parallel…} ins(%x) outs(%init) {
  ^bb0(%v: f32, %o: f32):
    %e = math.erf %v : f32                // __nv_erff → math.erf
    linalg.yield %e : f32
}
```

**매핑되는 함수셋** (대규모): `absf, acos, acosh, asin, asinh, atan, atanh, ceil, cos, cosh, erf,
exp, floor, log, powf, rsqrt, sin, sinh, sqrt, tan, tanh, trunc, …`

**변종**: `convert_extern_elementwise`, `unsupported_extern_elementwise`(미지원 함수).

---

### B. Contraction (matmul)

`tt.dot` — **shape/layout 무관하게 무조건 `linalg.matmul`** (분기 없음).

**출력**
```mlir
%init   = tensor.empty() : tensor<128x64xf32>
%zeroes = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x64xf32>)
%mm     = linalg.matmul ins(%A, %B : tensor<128x256xbf16>, tensor<256x64xbf16>)
                        outs(%zeroes : tensor<128x64xf32>) -> tensor<128x64xf32>
%res    = arith.addf %C, %mm             // 누산기 C가 zero tensor가 아니면 (isZeroTensor 검사)
```

- 누산기 `C`는 matmul **밖에서** `arith.addf`(정수면 `addi`)로 더함. `C`가 zero면 생략.
- `linalg.matmul`은 **rank-2 전용** — batched/3D 경로 없음. 전치는 앞단 별도 op(E 참조)로 materialize되어
  matmul은 항상 canonical `(M,K)×(K,N)→(M,N)`.
- 혼합정밀도(fp16 입력→fp32 누산)는 출력 타입 기준으로 `linalg.matmul`이 처리.

**변종**: `dot`, `use_dot_opc`, `kernel-03-matrix-multiplication`.

---

### C. Reduction

`tt.reduce` — body를 `linalg.reduce`에 **복제**한다 (두 op이 구조적으로 동일). 단, 초기값을 올바로
설정하기 위한 패턴 매칭이 필요(sum→0, max→-inf, min→+inf 등).

**출력** (sum, axis=1)
```mlir
%init   = tensor.empty() : tensor<512xf32>
%zeroes = linalg.fill ins(%c0 : f32) outs(%init : tensor<512xf32>)
%r = linalg.reduce ins(%src : tensor<512x256xf32>) outs(%zeroes : tensor<512xf32>) dimensions = [1]
  (%in: f32, %acc: f32) {
    %s = arith.addf %in, %acc : f32
    linalg.yield %s : f32
  }
```

- sum → `arith.addf`; min/max → `arith.minsi/maxsi/minui/maxui`/`minimumf/maximumf`
  (`cmpf/cmpi`+`select` 형태는 `MinMaxConverter`가 먼저 정규화)
- 스칼라 결과(rank-0)는 `tensor.extract`/`tensor.insert`로 처리

**변종**: `reducesum_*`(axis 0/1/middle/scalar), `reducemax_*`, `convert_addi_reduce`,
`convert_minmax_reduce`, `convert_minmax_fp_reduce`.

---

### C′. Arg-reduction (argmax / argmin)

값뿐 아니라 **인덱스까지 추적**해야 해서 가장 복잡. `linalg.reduce` + `linalg.generic` + `linalg.index`
조합에, body에서 비교·선택·타이브레이크를 수행.

**출력 op**: `linalg.reduce`, `linalg.generic`, `linalg.index`, `arith.cmpf`/`cmpi`, `arith.select`,
`arith.andi`/`ori`(타이브레이크), `arith.muli`, `linalg.fill`, `tensor.empty`, `tensor.extract`.

**변종**: `convert_argmin_argmax`, `convert_argmin_argmax_2d`.

---

### D. Scan (cumsum)

`tt.scan`/cumsum — **reduction이 아님**. 모든 부분합을 유지하므로 축을 접지 않고 `linalg.generic`으로
index 기반 누적.

**출력 op**: `linalg.generic`, `linalg.yield`, `tensor.empty`, `arith.index_cast`, `arith.muli`,
(타입에 따라) `arith.fptosi` 등.

**변종**: `cumsum`.

---

### E. Permutation (transpose)

`tt.trans` → **`linalg.transpose`** (permutation 배열).

**출력**
```mlir
%init = tensor.empty() : tensor<64x256xbf16>
%t = linalg.transpose ins(%src : tensor<256x64xbf16>) outs(%init : tensor<64x256xbf16>)
       permutation = [1, 0]
```

> GEMM에서 `tt.trans` 후 `tt.dot`이면, transpose는 이렇게 별도 `linalg.transpose`로 materialize된 뒤
> `linalg.matmul`이 canonical 형태를 받는다. (전치 stride 로드로 처리되는 경우는 메모리 측 ①/③ 참조.)

---

### F. Broadcast

`tt.broadcast` → `linalg.generic`의 **broadcast indexing map**(차원 투영)으로 확장.

**출력**
```mlir
#proj = affine_map<(d0, d1) -> (d0)>       // d1 축을 drop = broadcast
#id2  = affine_map<(d0, d1) -> (d0, d1)>
%b = linalg.generic {indexing_maps = [#proj, #id2], iterator_types = ["parallel", "parallel"]}
       ins(%v : tensor<128xf32>) outs(%init : tensor<128x64xf32>) {
  ^bb0(%x: f32, %o: f32): linalg.yield %x : f32
}
```

---

### F′. Splat

`tt.splat`(scalar → tensor) → **`linalg.fill`**.

**출력**
```mlir
%init = tensor.empty() : tensor<128xf32>
%s = linalg.fill ins(%scalar : f32) outs(%init : tensor<128xf32>) -> tensor<128xf32>
```

**변종**: `convert_splat_float`.

---

### G. Shape 조작 (순수 metadata)

연산 없이 view만 바꾼다.

- `tt.expand_dims` → `tensor.expand_shape`
- `tt.reshape` → `tensor.collapse_shape` / `tensor.expand_shape` (경우에 따라 `linalg.generic` 동반)

**출력**
```mlir
%e = tensor.expand_shape %src [[0, 1]] : tensor<256xf32> into tensor<1x256xf32>
%c = tensor.collapse_shape %src [[0, 1]] : tensor<4x64xf32> into tensor<256xf32>
```

**변종**: `convert_tensor_reshape`, `addptr_reshape_broadcast`.

---

### G′. Concat / slice (순수 레이아웃)

- `tt.cat` → `tensor.concat` + `tensor.extract_slice`
- `tt.join` → `tensor.empty` + `tensor.insert_slice`
- `tt.split` → `tensor.extract_slice`

**출력**
```mlir
%cat = tensor.concat dim(0) %a, %b : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
%s   = tensor.extract_slice %src[0] [64] [1] : tensor<128xf32> to tensor<64xf32>
```

---

### H. Generation (make_range / iota)

`tt.make_range` → `linalg.generic` + **`linalg.index`** (index → value).

**출력**
```mlir
%init = tensor.empty() : tensor<128xi32>
%r = linalg.generic {indexing_maps = [#id], iterator_types = ["parallel"]} outs(%init) {
  ^bb0(%o: i32):
    %i  = linalg.index 0 : index
    %ic = arith.index_cast %i : index to i32
    %v  = arith.addi %ic, %start : i32       // start offset 더함
    linalg.yield %v : i32
}
```

---

### H′. Constant (dense)

`arith.constant`(dense splat) → `DenseConstantConverter` → `tensor.empty` + `linalg.fill`(또는 상수 유지).

---

## 4. `linalg.generic` 분류가 핵심

연산의 대부분이 `linalg.generic`이므로, lowering pass는 이를 **body / indexing_maps / iterator_types**로
세분해야 한다.

| 신호 | 범주 |
|---|---|
| 전부 `parallel` + identity indexing map | **pointwise** (A) — body가 실제 연산 |
| 전부 `parallel` + projection(차원 drop) map | **broadcast** (F) |
| body에 `linalg.index` | **generation/iota** (H) 또는 arg-reduction(C′)의 인덱스 추적 |
| `reduction` iterator 존재 | 보통 `linalg.reduce`로 표현, 수동 generic reduce도 가능 |
| iter_args로 부분합 carry | **scan** (D) |

named op 3종(`matmul`/`reduce`/`transpose`)은 시그니처가 명확해 바로 매핑 가능하고,
`tensor.*`(expand/collapse/slice/concat)는 **데이터 이동 없는 순수 레이아웃**이라 descriptor 조작으로 흡수.

---

## 5. HW-중립 매핑 프레이밍

연산 구조 축이 HW 연산 유닛에 대응한다.

| 연산 범주 | 추상 HW 타깃 |
|---|---|
| pointwise / broadcast (`linalg.generic` parallel) | SIMD/vector ALU 레인 |
| contraction (`linalg.matmul`) | systolic / GEMM 엔진 |
| reduction (`linalg.reduce`) | reduction tree / accumulator |
| scan (`linalg.generic` index-carried) | prefix-scan 유닛 |
| permutation / shape (`linalg.transpose`, `tensor.*`) | descriptor / 주소 재매핑 (데이터 이동 X) |
| generation (`linalg.index`) | 인덱스 / iota 생성기 |

**핵심 요약**: 연산은 `linalg.generic`이 지배(pointwise+broadcast+scan+generation)하고, 구조화 연산
3종 named op(`matmul`/`reduce`/`transpose`), 순수 레이아웃 `tensor.*`로 나뉜다. 연산 lowering pass가
실제로 다뤄야 할 건 (1) `linalg.generic` 분류, (2) named op 3종 직접 매핑, (3) `tensor.*` 레이아웃,
(4) 잔여 scalar `arith`/`math` — 넷이다.

---

## 6. 소스 위치 (triton-shared)

| 대상 | 파일 |
|---|---|
| 연산 패턴 등록 (populate 함수들) | `lib/Conversion/TritonArithToLinalg/TritonArithToLinalg.cpp` |
| 모든 converter 구현 (Matmul/Reduce/Broadcast/Transpose/…) | `include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp` |
| `tt.dot` → `linalg.matmul` | 동 파일 `MatmulConverter` |
| `tt.trans` → `linalg.transpose` | 동 파일 `TransposeConverter` (L141), `CollapseShape.cpp` (L243) |
| libdevice → `math` dialect | `populateExternElementwiseOpToMLIROps` (동 디렉토리) |
| elementwise catch-all | upstream `linalg::populateElementwiseToLinalgConversionPatterns` |
| 테스트 (범주별 concrete IR) | `test/Conversion/TritonToLinalg/` (convert_*, reduce*, dot, cumsum 등) |

---

## 부록 A. 메모리 패턴과의 접점

연산 결과는 대부분 tensor(value semantics)로 흐르다가, 저장 시 메모리 측 store 패턴
(`materialize_in_destination`)으로, 로드 시 메모리 측 load 패턴(`reinterpret_cast`+`copy`+`to_tensor`)에서
tensor로 진입한다. 즉:

```
[메모리 load] reinterpret_cast → alloc → copy → to_tensor
     │  (tensor)
     ▼
[연산] linalg.generic / matmul / reduce / transpose / tensor.*
     │  (tensor)
     ▼
[메모리 store] materialize_in_destination → reinterpret_cast view
```

- 연산 출력 버퍼(`tensor.empty` → alloc)는 레퍼런스 파이프라인에서 최적화되지 않음
  (`--eliminate-empty-tensors` 비활성화). 상세는 메모리 문서 §6 참조.
- `linalg.generic`의 `outs`(init)와 메모리 load의 staging 버퍼가 각각 별도 alloc이므로,
  load+compute fusion으로 중간 버퍼를 줄일 여지가 있다 (메모리 문서 §6.5).
