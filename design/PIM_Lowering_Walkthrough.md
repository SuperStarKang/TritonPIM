# PIM lowering 동작 설명

Triton 커널 하나가 UPMEM DPU에서 도는 코드가 되기까지 각 단계가 무엇을 확정하는지,
데이터가 어떻게 나뉘고 어디에 놓이는지를 설명한다.

설계 계약은 `PIM_IR_Redesign.md`에 있다. 이 문서는 그 계약이 실제 코드에서 어떻게
나타나는지를 보여 준다. §4의 모든 IR 발췌는 `gemm.py`를 `M = N = K = 64`로 실행해
`dump/`에 남은 산출물에서 가져온 것이다.

경로 기준: 상대 경로는 `/home/dlrkdals/TritonPIM/triton_shared/`.

---

# 1. 한눈에 보기

## 1.1 세 계층

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Triton                                                                   │
│   BLOCK_M / BLOCK_N / BLOCK_K 가 logical tile 의 모양을 정한다            │
│   grid 가 그 tile 들을 열거한다                                          │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  --triton-to-linalg-experimental
┌───────────────────────────────▼──────────────────────────────────────────┐
│ TTShared IR                                                              │
│   memref.reinterpret_cast + alloc + copy + to_tensor 로 된 메모리 접근    │
│   linalg.matmul 로 된 계산                                               │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  ① ~ ⑥
┌───────────────────────────────▼──────────────────────────────────────────┐
│ Logical PIM IR            target 를 모른다                               │
│   pim.access / pim.load / pim.store                                      │
│   logical tile 좌표 loop nest (pim.axis)                                 │
│   "어느 tile 이 어느 데이터를 쓰는가" 까지만 안다                        │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  ⑦ ~ ⑫   #pim.target_profile 을 읽는다
┌───────────────────────────────▼──────────────────────────────────────────┐
│ Placed PIM IR             target 파라미터로 배치가 끝났다                │
│   pim.device  ─ PIM Device 하나가 하는 일                                │
│   pim.place   ─ 버퍼 하나가 그 device 에 상주해야 하는 영역              │
│   "어느 device 가 어느 영역을 갖는가" 까지 안다                          │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  ⑭  --convert-pim-to-upmem
┌───────────────────────────────▼──────────────────────────────────────────┐
│ UPMEM IR                  물리 메모리 계층과 전송 단위                   │
│   upmem.mram / host_var / acquire / foreach_unit / stage / to_mram       │
│   upmem.launch / gather / foreach_group / unstage                        │
│   upmem.device_func { mram_read / mram_write / linalg on WRAM }          │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  triton-shared-translate
┌───────────────────────────────▼──────────────────────────────────────────┐
│ host.cpp  (g++ + libdpu)        device.c  (dpu-upmem-dpurte-clang)       │
└──────────────────────────────────────────────────────────────────────────┘
```

각 계층이 답하는 질문이 다르다.

| 계층 | 답하는 것 | 답하지 않는 것 |
|---|---|---|
| Logical PIM IR | tile 의 모양, tile 좌표, tile 이 읽고 쓰는 영역 | 기계가 몇 대인지, 어디에 놓이는지 |
| Placed PIM IR | device 수, device 별 영역·용량·복제 | 메모리 계층, 전송 단위, 주소 |
| UPMEM IR | MRAM/WRAM, DMA 크기, staging 레이아웃 | C 문법 |

## 1.2 pass 파이프라인

각 단계가 **하나씩만** 확정한다.

```
 ①  --pim-normalize-address       주소 재귀 → offset(i) = initial + i·advance
 ②  -cse -licm -canonicalize      잉여 산술을 상수로 확정
 ③  --convert-ttshared-to-pim     메모리 관용구 → pim.access / load / store
 ④  --pim-build-logical-grid      source grid → logical tile loop nest
 ⑤  --pim-recover-access-origin   access 의 축별 시작 위치
 ⑥  --pim-form-compute            누산기를 matmul 의 outs 로 통합
     ─────────────────────────────  여기까지 target 무관
 ⑦  --pim-set-target-profile      #pim.target_profile 부착
 ⑧  --pim-map-cores               logical 차원 ↔ PIM Core, lane 수
 ⑨  --pim-form-execution-waves    core footprint 로 wave 구성
 ⑩  --pim-map-devices             logical tile → PIM Device 균형 분배
 ⑪  --pim-place-data              영역·용량·복제 확정
 ⑫  --pim-verify                  공통 fail-fast 검증
 ⑬  -cse -canonicalize
 ⑭  --convert-pim-to-upmem        UPMEM legality gate + host/device 분리
```

⑦ 이전의 어떤 pass 도 기계 값을 읽지 않는다. ⑧과 ⑨는 UPMEM에서 `pim_core_shape = [1]`
이라 좌표를 만들지 않고, 표현할 수 없는 구성이면 진단을 낸다.

---

# 2. 데이터 모델

## 2.1 tile 에서 하드웨어까지

```
BLOCK_M × BLOCK_N              logical tile 하나 = 출력의 직사각형 조각
        │
        ├─ output_row_tile     ─┐
        ├─ output_column_tile   ├─ logical 좌표. loop nest 의 유도변수다.
        └─ explicit_k_partition ─┘  (split-K 커널에만 있다)
        │
        ▼  --pim-map-devices   balanced block 으로 균형 분배
PIM Device 하나가 tile 의 연속 직사각형 하나를 맡는다
        │
        ▼  --pim-place-data
device 하나가 상주시켜야 하는 A / B / C 영역이 확정된다
        │
        ▼  --convert-pim-to-upmem
MRAM 에 tile-major 로 놓이고, 계산할 때 tile 하나씩 WRAM 으로 DMA 된다
```

## 2.2 왜 A·B·C 를 서로 다르게 나누는가

출력 tile 소유권을 먼저 정하고, 그 tile 을 계산하는 데 필요한 입력을 유도한다.
같은 크기로 나누면 GEMM 의 재사용 관계가 사라진다.

```
        output_column_tile →
        ┌───────┬───────┬───────┬───────┐
    o   │ D0,0  │ D0,1  │ D0,2  │ D0,3  │   각 칸이 PIM Device 하나
    u   ├───────┼───────┼───────┼───────┤
    t   │ D1,0  │ D1,1  │ D1,2  │ D1,3  │
    p   ├───────┼───────┼───────┼───────┤
    u   │ D2,0  │ D2,1  │ D2,2  │ D2,3  │
    t   └───────┴───────┴───────┴───────┘
    _
    r    A: 행 방향으로만 나뉜다 → 같은 행의 device 들은 같은 A 를 받는다
    o       replicated_over = ["output_column_tile"]
    w
    _    B: 열 방향으로만 나뉜다 → 같은 열의 device 들은 같은 B 를 받는다
    t       replicated_over = ["output_row_tile"]
    i
    l    C: 행과 열 양쪽으로 나뉜다 → 복제 없음
    e
    ↓
```

복제 판정이 없으면 host 가 같은 데이터를 device 수만큼 따로 모은다. `upmem.stage` 의
`dedup_over` 가 이 판정을 받아 서로 다른 조합의 개수만큼만 실제로 모은다.

## 2.3 balanced block 분배

tile 개수는 런타임 값이라 컴파일 시점에 나눗셈이 수로 확정되지 않는다. 생성된 host
코드가 launch 시점에 계산한다. 몫과 나머지로 나누어 나머지 partition 이 하나씩 더 맡는다.

```
tile 129 개를 partition 64 개에 나눈다

  minTiles = 129 / 64 = 2      remainder = 129 % 64 = 1

  partition  0 : first = 0·2 + min(0,1) = 0,  trips = 2 + 1 = 3
  partition  1 : first = 1·2 + min(1,1) = 3,  trips = 2 + 0 = 2
  partition  2 : first = 2·2 + min(2,1) = 5,  trips = 2
  ...
  partition 63 : first = 127,                 trips = 2

  capacity = ceildiv(129, 64) = 3    ← 전 device 동일. 버퍼 크기가 여기서 나온다.
```

단순 blocked 분할이면 `ceildiv(129,64) = 3` 씩 나눠 43개만 일하고 21개가 논다.
`mod` 분배는 균형은 맞지만 device 가 맡는 tile 이 등간격 집합이 되어 축마다 연속 구간
하나로 적을 수 없다. balanced block 은 둘 다 만족한다.

`capacity` 와 `trips` 를 나누는 이유는 `dpu_push_xfer` 가 모든 DPU 에 같은 바이트 수를
요구하기 때문이다. 버퍼는 `capacity` 로 잡고 실제로 채우는 것은 `size` 다.

## 2.4 MRAM 안의 배치 — tile-major

원본 버퍼는 행 우선이고 device 가 맡는 영역은 그 안의 직사각형이라 연속이 아니다.
MRAM DMA 는 연속 구간만 옮기므로, host 가 모을 때 **tile 하나가 연속이 되도록** 다시 쓴다.

```
원본 A (행 우선, stride = K)              device 0 이 맡는 MRAM (tile-major)
┌─────────────────────────────┐          ┌──────┬──────┬──────┬──────┐
│ ■■■■□□□□□□□□ … │  ← 행 0    │  stage   │tile 0│tile 1│tile 2│tile 3│
│ ■■■■□□□□□□□□ … │  ← 행 1    │ ───────► │512 B │512 B │512 B │512 B │
│  …                          │          └──────┴──────┴──────┴──────┘
└─────────────────────────────┘            tile 하나가 32×16 = 512 B 로 연속
  ■ = 이 device 의 영역                     → mram_read 한 번이면 된다

  자리 = ((tile 번호[0] × 축1 tile 수 + tile 번호[1]) × …) × tile 원소 수
       + (tile 안의 자리)
```

행 단위로 두면 tile 의 행 수만큼 DMA 가 필요하고, 한 행이 수십 바이트라 준비 비용이
옮기는 양보다 커진다.

## 2.5 경계 처리

tile 크기가 실제 크기를 나누지 못하면 마지막 tile 이 부분 tile 이다.

```
size = 72, tile = 32

  _t0 = 0        _t0 = 1        _t0 = 2
┌───────────┐┌───────────┐┌───────────┐
│  0 .. 31  ││ 32 .. 63  ││ 64 .. 71 │  ← 8 개만 유효
└───────────┘└───────────┘└─────┬─────┘
                                 └── 72 .. 95 는 건너뛴다

  tile loop 상한 = ceildiv(72, 32) = 3     ← 부분 tile 도 돈다
  원소 가드      = _t0·32 + _e0 < 72       ← 범위 밖은 건너뛴다
  건너뛴 자리    = calloc 이 넣은 0        ← pad = 0 이 여기서 공짜로 나온다
```

`size` 는 `--pim-place-data` 가 mask 식에서 원본 크기를 되짚어
`clamp(extent - start, 0, tile 수 × tile)` 로 낸다. 그래서 `start + size` 가 언제나
원본 버퍼 안이다.

---

# 3. 실행 순서

```
host                                              device (DPU 2048 대)
────                                              ─────────────────────
pim_acquire(2048, "kernel.dpu")
   └ 런타임이 쥐고 있던 집합을 그대로 준다

DPU_FOREACH _u = 0 .. 2047                        (아직 안 돈다)
   ├ 좌표 (row_part, col_part) 를 _u 에서 푼다
   ├ balanced block 으로 first/trips/capacity 계산
   ├ stage: 원본 → tile-major 연속 버퍼
   │        dedup_over 축만 다른 device 는 같은 결과 → 한 번만 만든다
   └ 배열에 담아 둔다

dpu_push_xfer(TO_DPU, "…_mram_0", …)              MRAM ← A
dpu_push_xfer(TO_DPU, "…_mram_1", …)              MRAM ← B
dpu_push_xfer(TO_DPU, "…_tiles_*", …)             __host 반복 횟수
dpu_push_xfer(TO_DPU, "…_pitch_*", …)             __host 축 간격
dpu_push_xfer(TO_DPU, "…_host_0",  …)             __host K loop 반복 횟수

dpu_launch(SYNCHRONOUS) ─────────────────────────► for (row tile)
                                                     for (column tile)
                                                       fill(acc, 0)
                                                       for (k)
                                                         mram_read A tile → WRAM
                                                         mram_read B tile → WRAM
                                                         acc += A·B
                                                       mram_write acc → MRAM
◄──────────────────────────────────────────────────  끝
dpu_push_xfer(FROM_DPU, "…_mram_2", …)            MRAM → host

for _g = 0 .. 2047
   └ unstage: tile-major 버퍼 → 원본의 strided 영역
```

`dpu_alloc` 과 `dpu_load` 는 수백 ms 가 드는데 데이터와 무관하므로, 런타임이 같은
device 수·같은 프로그램이면 쥐고 있던 집합을 그대로 준다. 그래서 `upmem.acquire` 에
짝이 되는 반납 op 이 없다.

---

# 4. 예시 — `gemm.py`, M = N = K = 64

```python
BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 16
grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N)) = (2, 2)
pim_target = "upmem",  num_units = 2048 (기본값)
```

logical tile 이 2 × 2 = 4 개뿐인데 device 가 2048 대라 2044 대가 놀지만, 각 단계가
무엇을 하는지는 그대로 드러난다. 크기를 맞춘 경우는 §4.9 끝에 표로 둔다.

## 4.1 출발점

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=..., other=0.0)
    b = tl.load(b_ptrs, mask=..., other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += BLOCK_K * stride_ak      # ← 주소를 loop 로 나른다
    b_ptrs += BLOCK_K * stride_bk
tl.store(c_ptrs, acc, mask=c_mask)
```

## 4.2 TTShared IR — 문제가 시작되는 자리

```mlir
%13:3 = scf.for %arg15 = %c0_i32 to %10 step %c1_i32
        iter_args(%arg16 = %7, %arg17 = %c0, %arg18 = %1) : i32 {
  %rc_A = memref.reinterpret_cast %arg0 to offset: [%arg16], sizes: [32, 16], …
  %rc_B = memref.reinterpret_cast %arg1 to offset: [%34],    sizes: [16, 32], …
  …
  %65 = arith.addi %arg16, %c16   : index      // a_ptrs += BLOCK_K·stride_ak
  %66 = arith.addi %arg17, %12    : index      // b_ptrs += BLOCK_K·stride_bk
  scf.yield %65, %66, %64
}
```

주소가 `iter_args` 에 실려 있다. 이 상태에서는 "이 device 가 A 의 어느 영역을 갖는가"를
유도할 수 없다 — `%arg16` 은 loop 안에서만 뜻이 있는 값이기 때문이다.

## 4.3 ① 주소 정규화

`offset(i) = initial + i·advance` 로 다시 쓴다.

```mlir
scf.for %arg15 = %c0_i32 to %10 step %c1_i32 : i32 {
  %a = arith.subi %arg15, %c0_i32           // iteration = (iv - lb) / step
  %b = arith.divsi %a, %c1_i32
  %c = arith.index_cast %b : i32 to index
  %d = arith.muli %c, %c16                  // i · advance
  %off = arith.addi %7, %d                  // initial + i·advance
  %rc_A = memref.reinterpret_cast %arg0 to offset: [%off], …
```

`iter_arg` 의 모든 사용처를 `%off` 로 바꾼다. `yield` 의 `addi` 도 함께 바뀌어 다음 반복
값이 `initial + (i+1)·advance` 가 되므로 loop 의미가 보존된다. 이어지는 `-canonicalize`
가 `(iv - 0) / 1` 을 상수로 확정해 `iteration = iv` 로 만든다.

이것이 안 되면 조용히 틀린다 — 배치 pass 가 그 버퍼를 보지 못한 채 나머지만 배치한다.
그래서 닫힌 형태로 못 만드는 주소 재귀는 진단을 낸다.

## 4.4 ③ 메모리 관용구 → `pim.access` / `pim.load`

`reinterpret_cast + alloc + fill + subview×2 + copy + to_tensor` 여섯 개가 두 개가 된다.

```mlir
%85 = pim.access %arg0 offset %77 sizes [32, 16] strides[%26, %c1]
      valid[%61, %84] pad %c0_i8 : i8 {id = 0 : i64} : memref<*xi8> -> !pim.access
%86 = pim.load %85 : !pim.access -> tensor<32x16xi8>
```

`valid` 가 mask 를, `pad` 가 `other=` 를 나른다. `id` 는 이 access 를 가리키는 이름으로,
나중에 `pim.place` 가 이 값으로 대응 access 를 가리킨다.

변환 뒤 `memref`/`bufferization` op 이 남아 있으면 진단을 낸다. 남겨 두면 배치 pass 가
그 버퍼를 보지 못한 채 부분 배치를 만든다.

## 4.5 ④ logical tile loop nest

`program_id` 를 loop 유도변수로 되돌린다. source grid 축과 logical tile 좌표는 다르다 —
`split_k.py` 는 축 하나가 둘로 갈라지고, `grouped_gemm.py` 는 `GROUP_M` 순서 사슬을
거쳐야 좌표가 나온다. `gemm.py` 는 항등이다.

```mlir
scf.for %arg14 = %c0 to %9 step %c1 {            // {pim.axis = "output_row_tile"}
  scf.for %arg15 = %c0 to %17 step %c1 {         // {pim.axis = "output_column_tile"}
```

| 커널 | source grid | 복원 방식 | logical 축 |
|---|---|---|---|
| `gemm.py` | `(x, y)` | 항등 | row, column |
| `split_k.py` | `(x, y)` | `pid/d` 와 `pid%d` 쌍 | row, column, explicit_k_partition |
| `grouped_gemm.py` | `(x)` | `GROUP_M` 13-op 사슬 | row, column |

`grouped_gemm.py` 의 사슬은 방문 **순서**만 정하고 배치에는 영향이 없으므로, 좌표를
loop 유도변수로 바꾸고 나면 통째로 죽는다.

## 4.6 ⑤ 축별 시작 위치 복원

`offset` 은 축이 합쳐진 스칼라라 그대로는 device 영역을 유도할 수 없다.
`offset = Σ origin[d]·strides[d]` 로 되돌린다.

각 항을 **좌표 쪽**과 **간격 쪽**으로 가르고, 간격 쪽을 `index_cast` 와 상수 곱을 벗겨 낸
기준 값으로 access 의 `strides` 와 대조한다.

```
B 의 offset 항          간격 쪽                     기준 값   배수   축
─────────────────────  ─────────────────────────   ───────  ────  ────
muli(iteration, %12)   %12 = idx_cast(%arg7 · 16)   %arg7     16    0
%5                     (없음, 상수 32)              없음      32    1   ← stride 1 축
```

SSA 동일성으로 대조하면 여기서 실패한다. `%12` 와 축 0 의 stride `%8 = idx_cast(%arg7)`
은 같은 간격을 다른 식으로 쓴 것이라 서로 다른 값이기 때문이다.

결과:

```mlir
%85 = pim.access %arg0 offset %77 sizes [32, 16] strides[%26, %c1]
      origin[%51, %76] valid[%61, %84] pad %c0_i8 : i8 {id = 0 : i64}
                       ▲       ▲
                       │       └─ k 위치  = iteration · 16
                       └───────── 행 시작 = output_row_tile · 32
```

어느 축과도 맞지 않는 간격이면 크기 1 의 축을 앞에 새로 만든다. `split_k.py` 의 출력이
그 경우로, `stride_ck` 가 rank-2 access 의 stride 목록에 없어 rank 3 이 된다.

```mlir
%104 = pim.access %arg2 offset %102 sizes [1, 32, 32] strides[%66, %67, %c1]
       origin[%70, %75, %103] valid[%c1, %89, %94] {id = 2 : i64}
              ▲
              └─ explicit_k_partition. SPLIT_K 개 출력면이 버퍼 축이 된다.
```

## 4.7 ⑥ 계산 정규화

`matmul(outs = 0으로 채운 tile)` + `generic(add)(누산기, 그 결과)` 를
`matmul(outs = 누산기)` 로 접는다. 임시 tile 하나와 tile 크기만큼의 덧셈이 사라진다.

```mlir
%68 = linalg.fill ins(%c0_i32) outs(%67 : tensor<32x32xi32>)
%69 = scf.for %arg16 = %c0_i32 to %21 step %c1_i32
      iter_args(%arg17 = %68) -> (tensor<32x32xi32>) : i32 {
  …
  %90 = linalg.matmul ins(%86, %89) outs(%arg17 : tensor<32x32xi32>)
  scf.yield %90
}
```

이어서 누산기 초기값을 K loop 바로 앞으로 내린다. 값 의미로는 밖에 있어도 맞지만
버퍼로 내리면 초기화가 한 번만 돌아 두 번째 tile 부터 앞 결과가 남는다.

## 4.8 ⑦⑧⑨ 기계 profile

```mlir
module attributes {
  pim.target_profile = #pim.target_profile<
    target = "upmem", number_of_pim_devices = 2048,
    pim_core_shape = [1], number_of_parallel_lanes_per_pim_core = 1,
    parallel_execution_kind = "multithread",
    requires_uniform_instruction_sequence = false,
    usable_bank_capacity_bytes = 33554432, usable_local_capacity_bytes = 49152>,
  pim.upmem_ext = #pim.upmem_ext<number_of_devices_per_transfer_group = 64>}
```

```mlir
func.func @gemm_kernel(…) attributes {
  pim.core_mapping = #pim.core_mapping<
    core_axis_roles = ["none"], tile_footprint = [1],
    lanes = 1, lane_axis_role = "whole_tile">}
```

UPMEM 은 device 당 PIM Core 가 하나라 core 축에 맡길 logical 차원이 없고
(`core_axis_roles = ["none"]`), logical tile 하나가 core subset 전체를 차지한다
(`tile_footprint = [1]`). 그래서 wave 하나가 tile 하나이고 ⑨는 항등이다.

`lanes` 는 local storage 예산이 정한다. 이 커널의 lane 당 tile 바이트는
A 512 + B 512 + 누산기 4096 = 5120 이므로 48 KiB 안에 9 개까지 들어가고,
요청값 1 이 그보다 작아 1 이 선택된다. **선택된 값은 device 소스 첫머리의
`// pim.lanes = 1` 로 실려 launcher 의 `-DNR_TASKLETS` 가 된다.**

## 4.9 ⑩ PIM Device 분배

explicit K 분할 축이 없으므로 2048 을 정사각에 가깝게 인수분해한다 → `64 × 32`.

```mlir
pim.device <axes = ["output_row_tile", "output_column_tile"], partitions = [64, 32]>
    extents[%0, %1] {
^bb0(%arg12: index, %arg13: index):        // 이 device 의 partition 좌표
  %2  = arith.divsi %0, %c64               // minTiles = row_tiles / 64
  %3  = arith.remsi %0, %c64               // remainder
  %4  = arith.minsi %arg12, %3
  %5  = arith.muli  %arg12, %2
  %6  = arith.addi  %5, %4                 // first = idx·minTiles + min(idx, rem)
  %7  = arith.cmpi  slt, %arg12, %3
  %8  = arith.select %7, %c1, %c0
  %9  = arith.addi  %2, %8                 // trips = minTiles + (idx < rem)
  %18 = arith.ceildivsi %0, %c64           // capacity = ceildiv(row_tiles, 64)
  …
  scf.for %arg14 = %c0 to %9 step %c1 {                    // [0, trips) 로 좁혔다
    %46 = arith.addi %6, %arg14 {pim.device_origin} : index  // 전역 좌표 복원
```

device 안에서 좌표를 0 부터 세는 것이 이 계층의 약속이다. 전역 좌표는 몸체 첫머리에서
`first + local` 로 다시 만들고 `pim.device_origin` 표시를 단다. target 이 그 첫
피연산자를 0 으로 바꾸면 주소가 device 안의 위치가 된다.

`extents` 는 나누기 전의 축별 tile 개수로, 배치 pass 가 여기서 배정량을 다시 계산한다.

**이 실행의 실제 분배** — `extents = [2, 2]`, `partitions = [64, 32]`:

```
  minTiles = 2/64 = 0,  remainder = 2   → partition 0, 1 이 1 개씩, 나머지 62 개가 0 개
  capacity = ceildiv(2, 64) = 1

        col_part 0   col_part 1   col_part 2 …
row 0 │  tile(0,0)    tile(0,1)      —
row 1 │  tile(1,0)    tile(1,1)      —
row 2 │     —            —           —
  …
```

4 대가 tile 하나씩 맡고 2044 대가 빈다. 크기를 맞추면 이렇게 된다.

| | M=N=K=64 (이 실행) | M=N=K=4096 |
|---|---|---|
| logical tile | row 2, column 2 | row 128, column 128 |
| device partition | 64 × 32 | 64 × 32 |
| device 당 tile | 1 (4 대만) | 2 × 4 = 8 (2048 대 전부) |
| A / device | 32 × 64 = 2 KiB | 64 × 4096 = 256 KiB |
| B / device | 64 × 32 = 2 KiB | 4096 × 128 = 512 KiB |
| C / device | 32 × 32 × 4 = 4 KiB | 64 × 128 × 4 = 32 KiB |

## 4.10 ⑪ 배치

```mlir
%25 = …                          // A 행 시작 = first_row_tile · 32
%27 = arith.muli %9,  %c32       // trips · 32
%28 = arith.muli %18, %c32       // capacity · 32
%29 = arith.index_cast %arg3     // M   ← mask 식에서 되짚은 원본 크기
%30 = arith.subi  %29, %25       // M - start
%31 = arith.maxsi %30, %c0
%32 = arith.minsi %31, %27       // size = clamp(M - start, 0, trips·32)

pim.place %arg0 start[%25, %c0] size[%32, %36] capacity[%28, %33] strides[%26, %c1]
    {access = 0 : i64, direction = "in",
     mapping = #pim.place_mapping<axis_sources = ["output_row_tile", ""]>,
     replicated_over = ["output_column_tile"]} : memref<*xi8>

pim.place %arg1 start[%c0, %38] size[%36, %44] capacity[%33, %40] strides[%37, %c1]
    {access = 1 : i64, direction = "in",
     replicated_over = ["output_row_tile"]} : memref<*xi8>

pim.place %arg2 start[%25, %38] size[%32, %44] capacity[%28, %40] strides[%45, %c1]
    {access = 2 : i64, direction = "out"} : memref<*xi32>
```

세 값의 뜻이 다르다.

```
  start ────────────► 원본 버퍼 안의 시작 위치
  size  ────────────► 실제로 맡는 원소 수. 원본 크기로 잘려 있다.
  capacity ─────────► 배정량. device 불변이라 버퍼 크기가 여기서 나온다.

  ┌──────────────── capacity ────────────────┐
  ┌──────── size ────────┐
  │ 실제 데이터           │  0 으로 채운 자리  │
  └──────────────────────┴───────────────────┘
  ▲
  start
```

`replicated_over` 는 그 축만 다른 device 들이 같은 데이터를 받는다는 뜻이다. A 는 열
축에, B 는 행 축에 불변이고 C 는 어느 축에도 불변이 아니다 — §2.2 의 그림 그대로다.

## 4.11 ⑫ 검증

`--pim-verify` 가 공통 계약을 확인한다.

```
  access ↔ place 가 id 로 양방향 일대일
  origin rank == sizes rank
  pad 가 0 이 아니면 진단              (단계 1 은 pad = 0 만 지원)
  device partition 곱 ≤ PIM Device 개수
  tile_footprint 가 pim_core_shape 를 나눈다
  lane 당 tile 바이트 × lane 수 ≤ usable_local_capacity_bytes
  mask 가 있는 축의 size 가 clamp 형태다
  axis_source 가 device mapping 의 축이다
```

버퍼별 용량 검사는 버퍼를 어떻게 나누는지가 target lowering policy 라 여기 두지 않는다.

## 4.12 ⑭ UPMEM 변환

진입 즉시 legality gate 를 지난다 — target 이 `upmem` 이고, lane 이 독립 스레드이며,
bank 용량이 기록돼 있고, 버퍼 하나의 상주량이 bank 예산을 버퍼 수로 나눈 몫 안이어야
한다. 상주량이 런타임 값이면 여기서 증명할 수 없으므로 host 코드가 전송 직전에 검사한다.

### 모듈 수준 선언

```mlir
upmem.mram     @gemm_kernel_mram_0 : memref<11184810xi8,  #upmem<space mram>>
upmem.mram     @gemm_kernel_mram_1 : memref<11184810xi8,  #upmem<space mram>>
upmem.mram     @gemm_kernel_mram_2 : memref<2796202xi32,  #upmem<space mram>>
upmem.host_var @gemm_kernel_pitch_0_1 : index          // MRAM 안의 축 간격
upmem.host_var @gemm_kernel_pitch_1_1 : index
upmem.host_var @gemm_kernel_pitch_2_1 : index
upmem.host_var @gemm_kernel_tiles_output_row_tile    : index   // 이 device 의 반복 횟수
upmem.host_var @gemm_kernel_tiles_output_column_tile : index
upmem.host_var @gemm_kernel_host_0 : i32               // K loop 반복 횟수
```

MRAM 크기 `11184810 = 33554432 / 3` 이다. `capacity` 가 런타임 값인데 device 의
`__mram_noinit` 은 상수를 요구하므로, bank 예산을 버퍼 수로 나눈 상한을 쓴다.

`@gemm_kernel_host_0` 은 `ceildiv(K, 16)` 이다. 커널 인자에서 나오는 수라 device 안에서
다시 만들 수 없으므로 host 변수로 통로를 낸다.

### host 쪽

```mlir
%2 = upmem.acquire 2048, @gemm_kernel_device : !upmem.set
upmem.foreach_unit %2 axes ["output_row_tile", "output_column_tile"] counts [64, 32] {
^bb0(%dpu: !upmem.unit, %row_part: index, %col_part: index):
  …balanced block 계산…
  %43 = upmem.stage %arg0 origin[%31, %c0] sizes[%38, %42] strides[%32, %c1]
        pad_to[%34, %39] tile [32, 16] {dedup_over = ["output_column_tile"]}
  upmem.to_mram %dpu, %43, @gemm_kernel_mram_0
  …
  upmem.to_host_var %dpu, %15, @gemm_kernel_tiles_output_row_tile
  upmem.to_host_var %dpu, %27, @gemm_kernel_host_0
}
upmem.launch %2
%7 = upmem.gather %2, @gemm_kernel_mram_2 capacity[%4, %6] : !upmem.set -> memref<2048x?x?xi32>
upmem.foreach_group %7 axes ["output_row_tile", "output_column_tile"] counts [64, 32] {
^bb0(%part: memref<?x?xi32>, %row_part: index, %col_part: index):
  upmem.unstage %part into %arg2 origin[…] sizes[…] strides[…] capacity[…] tile [32, 32]
}
```

`pim.place` 의 `start`/`size`/`strides` 가 `upmem.stage` 의 `origin`/`sizes`/`strides` 로,
`capacity` 가 `pad_to` 로, `replicated_over` 가 `dedup_over` 로 그대로 간다.
`tile` 은 대응 `pim.access` 의 `sizes` 다.

### device 쪽

```mlir
upmem.device_func @gemm_kernel_device attributes {upmem.lanes = 1 : i64} {
  %0     = upmem.get_host_var @gemm_kernel_host_0 : i32
  %alloc = memref.alloc() : memref<32x32xi32, #upmem<space wram>>   // 누산기
  %alloc_0 = memref.alloc() : memref<32x16xi8, #upmem<space wram>>  // A tile
  %alloc_1 = memref.alloc() : memref<16x32xi8, #upmem<space wram>>  // B tile
  %4 = upmem.get_host_var @gemm_kernel_tiles_output_row_tile : index
  %5 = upmem.get_host_var @gemm_kernel_tiles_output_column_tile : index
  scf.for %arg0 = %c0 to %4 step %c1 {
    scf.for %arg1 = %c0 to %5 step %c1 {
      linalg.fill ins(%c0_i32) outs(%alloc)
      scf.for %arg2 = %c0_i32 to %0 step %c1_i32 : i32 {
        upmem.mram_read  @gemm_kernel_mram_0[%off_a], %subview_a
        upmem.mram_read  @gemm_kernel_mram_1[%off_b], %subview_b
        linalg.matmul ins(%alloc_0, %alloc_1) outs(%alloc)
      }
      upmem.mram_write %subview_c, @gemm_kernel_mram_2[%off_c]
    }
  }
}
```

axis loop 의 상한이 `upmem.get_host_var` 에서 온다. 그래서 device 는 자기 좌표를 몰라도
되고, 덜 맡는 device 도 값 하나로 표현된다. `pim.device_origin` 의 첫 피연산자를 0 으로
바꾸었으므로 `pim.access` 의 `origin` 이 device 안의 위치가 된다.

MRAM 안의 tile 자리는 축별 tile 번호에서 나온다.

```
  번호[d]     = origin[d] / tile[d]
  축별 tile 수 = capacity[d] / tile[d]      ← pitch host 변수에서
  자리         = ((번호[0]·수[1] + 번호[1])·수[2] + …) · tile 원소 수
```

C tile 은 32 × 32 × 4 = 4096 byte 라 DMA 한 번의 상한 2048 을 넘으므로 16 행씩 두 번으로
쪼갠다. A·B tile 은 512 byte 라 한 번이면 된다.

## 4.13 생성된 코드

### `device.c`

```c
// pim.lanes = 1
__mram_noinit int8_t  gemm_kernel_mram_0[11184810];
__mram_noinit int8_t  gemm_kernel_mram_1[11184810];
__mram_noinit int32_t gemm_kernel_mram_2[2796202];
__host int32_t gemm_kernel_host_0;
__host int32_t gemm_kernel_tiles_output_row_tile;
…
__dma_aligned int32_t w0[1024];   // 누산기
__dma_aligned int8_t  w1[512];    // A tile
__dma_aligned int8_t  w2[512];    // B tile

int main() {
  …
  mram_read((__mram_ptr void*)(gemm_kernel_mram_0 + v41), (void*)v44, 512);
  mram_read((__mram_ptr void*)(gemm_kernel_mram_1 + v62), (void*)v65, 512);
  for (int32_t _m = 0; _m < 32; ++_m)
    for (int32_t _n = 0; _n < 32; ++_n) { … }
  mram_write((void*)v86, (__mram_ptr void*)(gemm_kernel_mram_2 + v83), 2048);
}
```

### `host.cpp`

```c
_set = pim_acquire(2048, DPU_BINARY);

int8_t** _s0 = (int8_t**)calloc(64, sizeof(int8_t*));   // A: 행 partition 수만큼만
int8_t** _s1 = (int8_t**)calloc(32, sizeof(int8_t*));   // B: 열 partition 수만큼만

for (int64_t _u = 0; _u < 2048; ++_u) {
  …좌표 풀기, balanced block…
  int8_t* _b = (int8_t*)calloc(v50_bytes, 1);           // 남는 자리는 0
  for (int64_t _t0 = 0; _t0 < ((v45 + 31) / 32); ++_t0) // 부분 tile 도 돈다
    for (int64_t _t1 = …)
      for (int64_t _e0 = 0; _e0 < 32; ++_e0)
        for (int64_t _e1 = 0; _e1 < 16; ++_e1)
          if (_t0 * 32 + _e0 < v45 && _t1 * 16 + _e1 < v49)   // 범위 밖은 건너뛴다
            _b[(_t0 * (v46 / 16) + _t1) * 512 + (_e0 * 16 + _e1)]
              = a0[(v38 + _t0 * 32 + _e0) * v39 + (v10 + _t1 * 16 + _e1) * v9];

  if (v50_bytes > 11184810) { fprintf(stderr, "…"); abort(); }   // 런타임 용량 검사
}
DPU_FOREACH(_set, _dpu, _idx) DPU_ASSERT(dpu_prepare_xfer(_dpu, _p0[_idx]));
DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "gemm_kernel_mram_0", 0, _p0_n[0], …));
…
DPU_ASSERT(dpu_launch(_set, DPU_SYNCHRONOUS));
DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_FROM_DPU, "gemm_kernel_mram_2", 0, v65_bytes, …));
for (int64_t _g = 0; _g < 2048; ++_g) { …unstage… }
```

`calloc(64, …)` 와 `calloc(32, …)` 가 복제 판정의 결과다. A 는 열 축에 불변이라 행
partition 64 개분만, B 는 행 축에 불변이라 열 partition 32 개분만 실제로 모은다.
판정이 없으면 2048 개분을 각각 모으게 된다.

---

# 5. 두 가지 변형

## 5.1 경계 tile — `M = N = K = 72`

`72 = 2·32 + 8` 이라 마지막 tile 이 부분 tile 이다.

```
  device 0 이 3 개의 행 tile 을 맡는다고 하자
  size[0] = clamp(72 - 0, 0, 3·32) = 72          ← capacity 96 보다 작다
  tile loop 상한 = ceildiv(72, 32) = 3
  원소 가드      = _t0·32 + _e0 < 72

  MRAM 버퍼 (capacity 96 × 80)
  ┌────────┬────────┬────────┐
  │ tile 0 │ tile 1 │ tile 2 │  ← tile 2 는 8 행만 유효, 나머지 24 행은 0
  └────────┴────────┴────────┘
```

가드가 없으면 `A[0][72]` 를 선형 위치 `0·72 + 72 = A[1][0]` 에서 읽고, 되돌릴 때는
`C[1][0]` 을 두 번 쓴다. 그 loop 가 `#pragma omp parallel for` 라 경합까지 된다.

## 5.2 split-K — 출력면이 버퍼 축이 된다

`split_k.py` 는 `torch.zeros((SPLIT_K, M, N))` 에 부분합을 쓰고 Python 이 `c.sum(dim=0)`
을 한다. `stride_ck` 가 rank-2 `reinterpret_cast` 의 stride 목록에 없으므로 origin 복원이
크기 1 의 축을 앞에 만든다.

```
  pim.access %C sizes [1, 32, 32] strides[stride_ck, stride_cm, 1]
                origin[explicit_k_partition, row, column]

  device 하나의 MRAM (capacity = [16, 32, 32])
  ┌──────────┬──────────┬─────┬──────────┐
  │ plane 0  │ plane 1  │ … │ plane 15 │   ← 16 개 출력면이 전부 이 device 에 있다
  └──────────┴──────────┴─────┴──────────┘
     ▲
     explicit_k_partition. device 축이 아니라 device 안의 실행 wave 다.
```

K device partition 을 1 로 고정하므로 A·B 의 K 영역이 연속이고, 부분합 C 의 16 개 면이
전부 그 device 에 있어 조밀하다. 수치는 동일하고 커널 IR 에 결합(combine)이 생기지 않는다.

`WRAM` 버퍼는 rank 2 로 남는다. 앞 축의 크기가 1 이라 MRAM 안의 자리 계산에만 나타나기
때문이다.

---

# 6. 지금 막히는 것

| 커널 형태 | 상태 |
|---|---|
| `matmul` + `fill` 만 남는 커널 | 통과 (`gemm.py`, `split_k.py`, `grouped_gemm.py`) |
| dtype 변환 epilogue 가 있는 커널 | `UPMEMToDevice` 에 elementwise `linalg.generic` 의 C 형태가 없다 |
| f16 을 쓰는 커널 | DPU 에 f16 이 없어 WRAM 버퍼를 낼 수 없다 |
| PIM Core 가 여럿인 target | `--pim-map-cores` 가 진단 |
| wave 당 tile 이 둘 이상 | `--pim-form-execution-waves` 가 진단 |
| k device partition > 1 | `--pim-place-data` 가 진단 |
| 0 이 아닌 pad | `--pim-verify` 가 진단 |

아래 넷은 설계상 Phase 2 로 미룬 것이고, 전부 조용히 틀리지 않고 진단으로 막힌다.
