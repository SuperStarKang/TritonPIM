# Silent-PIM (AiMX) Target 설계

Placed PIM Common IR을 Silent-PIM(AiMX) FPGA의 CDMA descriptor chain으로 내리는 계층의
현재 구현과 실행 조건을 정리한다. Common IR의 계약은
[PIM Common IR 설계](PIMCommonIR_Design.md), 모델 session의 상주 정책은
[Model Runtime 설계](ModelRuntime_Design.md)가 다룬다.

| 근거 | 경로 |
| --- | --- |
| 기계 상수·opcode | `~/pim_aimx_sw/pim_sw/src/pim_math_src/pim_math.h` |
| 참조 matmul | `~/pim_aimx_sw/pim_sw/src/pim_math_src/matmul.c` (`_matmul_silent`, `_gemm_silent`) |
| 참조 host 배치 | `~/pim_aimx_sw/ubench/matmul/matmul.cpp` |
| AiMX dialect | `include/triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.{h,td}`, `lib/Dialect/PIMTarget/AiMX/IR/AiMXDialect.cpp` |
| lowering | `lib/Conversion/PIMToAiMX/PIMToAiMXPass.cpp`, `DescriptorBuilder.cpp` |
| codegen | `lib/Target/AiMX/AiMXToDescriptors.cpp`, `AiMXToRuntimeStub.cpp`, `DescriptorReuse.h` |
| runtime | `pim_backend/backend/aimx_runtime.h`, `aimx_resident_runtime.cpp` |
| Python | `pim_backend/backend/{compiler,driver,target}.py`, `aimx_adapter.py`, `aimx_model.py` |
| autotuner | `pim_backend/autotuner/{model,gemm,gemv}.py` |

PIM Common IR은 이 target을 위해 고치지 않는다. AiMX는 Common이 정한 Block, Block Group,
Execution Tile, device mapping, K 누산 순서, host 전송 순서를 그대로 받고 **물리 저장
형태와 descriptor 열만** 정한다.

---

## 1. 하드웨어

### 1.1 구성

| 항목 | 값 |
| --- | --- |
| unit | FPGA 1장 = GDDR6 칩 8개 (칩당 4 GB, 2 channel) |
| channel / bank | 16 channel × 16 bank = 256 bank |
| ALU (lane) | bank당 16 → **4096 lane** |
| 물리 용량 | 32 GB (bank 128 MB) |
| 할당 가능 용량 | runtime heap `0x1FEE00000` (8 GB − 18 MB). 넘으면 `PIM] OOM`으로 종료 |
| column 접근 | 256 bit = bf16 16개 |
| 데이터 형식 | PIM DRAM 안은 언제나 bfloat16 |

`backend/target.py`는 `bank_capacity_limit_bytes = heap / 256 = 33,480,704`로 Common
profile에 싣는다. 물리 bank 용량(128 MB)은 상수로만 기록한다.

### 1.2 기계 상수 (`AiMXDialect.h`)

| 상수 | 값 | 뜻 |
| --- | --- | --- |
| `kNumChannels` | 16 | descriptor group의 record 수 |
| `kTypeSize` | 2 | bf16 |
| `kBurstSize` | 32 B | DRAM burst = bank 하나의 lane 16개 |
| `kRegSize` | 256 원소 | vecA 한 번 적재가 담는 K |
| `kPIMWidth` | 512 B | channel 하나의 operand 폭 |
| `kDeviceWidth` | 8192 B | lane 4096개 × 2 B |
| `kLanes` | 4096 | output block 원소 수 |
| `kStrideShift` | 33 | pitch를 주소에 `addr \| (pitch << 33)`로 싣는다 |
| `kGroupBytes` | 512 B | descriptor group = record 32 B × 16 |
| `kPMMSize` / `kVACCOffset` / `kPMMAlign` | `0x30000` / `0x10000` / 256 KiB | vACC = PMM + `0x10000` |
| `kMaxRuntimeAllocationBytes` | 1023 × 2 MiB | vendor allocator가 크기를 `int`로 받아 2 GiB에서 overflow |

### 1.3 descriptor

`pim_isa_32_t`는 Xilinx CDMA scatter-gather descriptor 그대로다(next, src, dst, len,
status). PIM 동작은 `status`의 flag로 지정한다. memory controller에는 평범한 DMA로
보인다(**silent**).

```text
(src, src_pitch) ──[opcode, batch]──► (dst, dst_pitch),  len byte
```

| bit | flag | 쓰는 곳 |
| --- | --- | --- |
| 1 / 2 / 3 | `a_mask` / `b_mask` / `c_mask` | operand A, B, C (vACC → memory) |
| 4 | `matmul` | MAC |
| 5 / 6 / 7 | `add` / `sub` / `mul` | elementwise |
| 0 | `descr_dst_mask` | elementwise A를 vACC로 |
| 10 / 13 | `bc` / `tp` | A 재배치 (Arel 쓰기) |
| 12 | `pim_mode` | PIM logic 내부 처리 |
| 15–17 | batch 코드 | 1 → 0, 2 → bit15, 4 → bit16, 8 → bit15\|16 |
| 18 | `stride` | src/dst 상위 비트의 pitch 사용 |

- `aimx.desc` 하나 = descriptor group 하나 = 같은 내용의 record 16개.
  **실행되는 것은 첫 record 하나다.** 모든 record의 `next`가 다음 group을 가리켜
  CDMA가 group마다 첫 record만 걷는다(FPGA 실측, §11).
- `len`은 **출력** byte 수다.
- batch 비트는 flag enum에 없다. `aimx.descriptor_program`의 `batch`에서 codegen이
  유도한다.

### 1.4 output block, batch, vecA

```text
batch b : lane 4096개를 b행 × (4096/b)열로 자른다     b ∈ {1, 2, 4, 8}
           K 한 step = 256/b 원소 (vecA 한 번이 b행 × 256/b)

b = 1     b = 8
1 × 4096  8 × 512
```

- A는 vecA에 그대로 들어가지 않는다. `bc|tp` descriptor가 A를 **Arel**(A의 16배 크기)로
  재배치한다. 순열은 FPGA 실측으로 확정됐다:
  `Arel[c*256 + g*16 + t] = A[16c + t]` (`m=1, k=256, b=1`).
- vACC는 lane당 22 bit 한 칸이다. output block 하나를 다 누산하고 drain해야 다음
  block으로 갈 수 있다.

---

## 2. 전체 pipeline

```text
Triton kernel
  │ make_ttir → _ttir_to_ttsharedir                (CPU와 공용)
  ▼ TTSharedIR
  │ PIM_COMMON_PASSES (compiler.py)                (target 무관, profile만 다름)
  │   ... --pim-form-block-groups --pim-form-execution-tiles --pim-map-devices
  │   --pim-place-data --pim-verify --pim-form-host-plan
  │   AiMX는 --pim-reuse-execution-tile-inputs와 마지막 -canonicalize를 뺀다
  ▼ Planned PIM Common IR
  │ --convert-pim-to-aimx=batch=… persistent=… matmul-row-alignment=…
  │                       matmul-useful-n-only=… matmul-k-alignment=…
  │ -cse -canonicalize --mlir-print-op-generic
  ▼ AiMX target IR (host op + descriptor program)
  │ triton-shared-translate --aimx-to-runtime-stub  → stub.cpp
  │ triton-shared-translate --aimx-to-descriptors   → aimx_descriptors.h
  │ (모델 session은 --aimx-to-model-stub)
  ▼ driver.py: g++ -O3 -fopenmp → pim_host.so  (+ libaimx_resident_<hash>.so, libpim.so)
```

AiMX가 `--pim-reuse-execution-tile-inputs`를 빼는 이유: AiMX의 `pim.load`는 상주
panel의 view라 hoist 이득이 없고, vACC 누산은 K Group 안에 A/B load가 직접 있어야 한다.

---

## 3. Common IR에서 받는 것

| 받는 것 | 어디서 | AiMX에서의 용도 |
| --- | --- | --- |
| Block / Block Group / Execution Tile | `#pim.block`, `#pim.block_group`, `#pim.execution_tile` | Group = tensor SSA 값 하나의 크기. Tile = `pim_exec` 한 번이 상주시키는 Group 묶음 |
| device 분할 | `pim.partition`의 `#pim.device_mapping` | `aimx.foreach_device` counts. `p_k > 1`은 FPGA별 partial을 host에서 k 순서로 더한다(§5.8) |
| placement | `pim.place` (`start`, `size`, `allocation_shape`, `strides`, `direction`, `shared_over`) | PIM DRAM 할당, stage/unstage |
| access | `pim.access` (`origin`, `sizes`, `valid`, `pad`) | Group panel 위치, mask |
| 계산 | `linalg.matmul`, `linalg.matvec`, identity-map `linalg.generic`, `linalg.fill`, `arith.add/sub/mulf` | descriptor program |
| K 누산 | `scf.for {pim.temporal_reduction, pim.reduction = "add"}` | vACC에서 K Group을 누산 |
| host 순서 | `pim.broadcast` / `pim.scatter` → `pim.launch` → `pim.gather`, `pim.scheduling_axis` loop | stage → invoke → unstage |

AiMX target이 하지 않는 것: Block/Group/Tile 크기 재선택, K 재분할, 계산 순서 변경,
Common loop 제거.

---

## 4. AiMX target IR

### 4.1 type

| type | 뜻 |
| --- | --- |
| `!aimx.device_set` | 확보한 FPGA 집합 |
| `!aimx.device` | `aimx.foreach_device` 몸체의 FPGA 하나 |
| `!aimx.buffer` | PIM DRAM 할당 하나. byte 단위라 원소 타입이 없다 |

### 4.2 host op

| op | 하는 일 |
| --- | --- |
| `aimx.acquire n` | FPGA n개 확보 |
| `aimx.foreach_device %set axes [m,n,k] counts [...]` | device 좌표 loop. `aimx.persistent`이면 할당·chain을 FPGA별로 상주 |
| `aimx.alloc %dev bytes B align A kind <data\|descriptor\|pmm>` | PIM DRAM 할당. 크기는 compile time 상수 |
| `aimx.stage %src origin sizes strides pad_to pitch [transpose] into %buf {group_layout \| group_rows/cols, shared_over}` | host gather + bf16 변환 + 0 padding + 업로드 |
| `aimx.build_descriptors @prog(args) into %chain` | descriptor program을 chain에 펼친다. `scf.execute_region {no_inline}` 안에서만 |
| `aimx.invoke %dev, %chain` | `pim_exec`. 끝날 때까지 기다린다 |
| `aimx.unstage %buf into %dst origin sizes strides pitch accumulate [transpose] {group_layout}` | download + bf16 → host 타입 + 대입 또는 누산 |
| `aimx.require %cond, "msg"` | 실행 시점 layout 검사 |

`group_layout = [useful_rows, useful_cols, physical_rows, physical_cols, row_groups,
col_groups]`은 **물리 순서**다. Group panel은 group grid에서 row-major로 이어진다.
`transpose`가 붙으면 host 행이 물리 열, host 열이 물리 행이다.

### 4.3 descriptor program

```mlir
aimx.descriptor_program @name batch B {
^bb0(%buf0, %off0, %pitch0, ..., %extra_buf...):   // view = (buffer, byte offset, byte pitch)
  scf.for ... { aimx.desc src(%b, %o) pitch %p dst(...) len L opcode <...> }
}
```

`aimx.desc` verifier: `len`은 32의 배수, `stride` flag와 pitch는 함께 있거나 함께
없고, src·dst 중 하나 이상이 있어야 하며, `copy`(flag 없음)는 src와 dst가 모두 필요하다.
program의 `batch`는 {1, 2, 4, 8}만 허용한다.

---

## 5. `--convert-pim-to-aimx`

### 5.1 device 하나의 구조

`DeviceLowering::run()`이 `pim.partition` 하나를 다음으로 바꾼다.

```text
aimx.acquire (Π partitions)                       ← scheduling loop 바깥
scf.for (Common scheduling rounds)                ← 그대로 유지
  aimx.foreach_device counts [p_m, p_n, p_k] {aimx.persistent}
    aimx.alloc PMM            (가장 먼저)
    aimx.alloc descriptor chain
    aimx.alloc placement buffers, scratch, Arel, 상수
    scf.execute_region no_inline {                ← chain 구성 경계
      Common loop nest 그대로 + aimx.build_descriptors
    }
    aimx.stage   (pim.broadcast / pim.scatter 순서)
    aimx.invoke  (Execution Tile당 한 번)
    aimx.unstage (pim.gather 순서)
```

- PMM을 가장 먼저 잡는 이유: `pim_private_malloc`이 `pim_malloc`이 이미 내준 주소를
  돌려준 사례가 있다. stub은 모든 data buffer가 `[PMM, PMM+0x30000)`과 겹치지 않는지
  실행 시점에 검사한다.
- 용량: 모든 할당의 `Σ ceil(bytes/8192) × 32`를 bank 사용량으로 보고
  `bank_capacity_limit_bytes`를 넘으면 오류다. 결과는 `aimx.allocations`,
  `aimx.bank_bytes`, `aimx.descriptor_bytes` 속성에 남는다.

### 5.2 물리 layout 규칙

tensor SSA 값마다 `Layout {usefulRows, usefulCols, rows, cols, transposed}`를 둔다.
useful은 Common Group의 논리 크기, rows/cols는 물리 크기다. SSA로 연결된 값(loop carry,
같은 access, elementwise 피연산자)은 한 layout을 공유한다.

| 연산 | operand | 물리 rows | 물리 cols | 비고 |
| --- | --- | --- | --- | --- |
| GEMM | A `[m,k]` | `roundup(Gm, b)` | `roundup(Gk, kAlign)` | `kAlign` 기본 `4096/b` |
| GEMM | B `[k,n]` | `roundup(Gk, 256)` | `roundup(Gn, rowAlign)` | `rowAlign`: compiler 기본 4096 |
| GEMM | C `[m,n]` | `roundup(Gm, b)` | `roundup(Gn, rowAlign)` | B와 같은 pitch |
| GEMV | W `[m,k]` → B (전치) | `roundup(Gk, 256)` | `roundup(Gm, 4096)` | Wᵀ로 저장 |
| GEMV | x `[k]` → A (전치) | 1 | `roundup(Gk, 4096)` | b=1의 A pitch 조건 |
| GEMV | y `[m]` → C (전치) | 1 | `roundup(Gm, 4096)` | |
| elementwise·copy | 그 밖의 값 | Group 행 | `roundup(cols, 4096)` | |

모든 값은 `rows × cols`가 4096의 배수여야 한다. copy/elementwise descriptor가 4096
원소씩 선형으로 순회하기 때문이다. 같은 tensor를 matmul과 matvec이 함께 쓰면 오류다.

A pitch 조건 `kAlign`: gather가 256 원소씩 읽으므로 256의 배수이고, batch 행 묶음이
8 KiB 단위로 떨어지도록 `b × kAlign ≡ 0 (mod 4096)`이어야 한다.

### 5.3 placement와 Group panel

`pim.place` 하나가 PIM DRAM 할당 하나다.

```text
row_groups = allocation_rows / usefulRows,  col_groups = allocation_cols / usefulCols
(transposed이면 두 값을 바꾼다)
physical = (row_groups × rows) × (col_groups × cols)

A: [A(m0,k0)][A(m0,k1)] ... [A(m1,k0)] ...   ← panel마다 rows × cols, 행 pitch = cols × 2
panel byte offset = (row_group × col_groups + col_group) × rows × cols × 2
```

`accessView`는 access origin에서 placement start를 뺀 Group 좌표로 panel을 고른다.
두 식을 `상수 + Σ계수·leaf` 선형식으로 펼쳐 빼므로(`localOffset`), round 좌표와
device 좌표가 지워진다. 그래서 descriptor 주소가 round마다 같고 chain을 다시 만들지
않는다(§7.3). access 시작이 panel 경계가 아니거나 placement 밖이면 `aimx.require`가
실행 전에 멈춘다.

### 5.4 GEMM: vACC K-Group 누산

`pim.temporal_reduction` loop이 다음 조건을 모두 만족하면 `lowerMACReduction`이 처리한다.

- accumulator가 positive-zero `linalg.fill` 하나
- loop 몸체 = A load, B load, `linalg.matmul`/`linalg.matvec`, yield (index 연산과 float 상수만 추가 허용)
- MAC 피연산자는 `pim.load` 결과다. bf16 입력을 `.to(tl.float32)`로 넓힌 GEMV는 Common
  `--pim-form-compute`가 `extf`를 벗겨 `linalg.matvec(bf16, bf16, f32 acc)`로 만들고, 넓히기는
  matvec 몸체 안에 남는다(`validateMAC`가 허용)
- MAC 결과가 다음 K Group으로만 전달됨

```text
각 M/N Group:
  [relayout] 모든 K-Group의 A → Arel (Group별 슬롯)
  for row in 0..rows step b:                       ← output block 행
    for col in 0..columns step 4096/b:             ← output block 열
      for K-Group:                                 ← vACC에 누산, 중간 drain 없음
        accumulate: (A Arel 8192B, B 8192/b B) × ceil(Gk/(256/b))
      [padding]  마지막 K-Group 뒤 0 MAC (MAC 길이가 256 배수가 아닐 때만)
      drain: vACC → C (Group panel) 1회
```

- `columns`: `matmul-useful-n-only`이면 `roundup(Gn, 4096/b)`, 아니면 물리 cols 전체.
- Arel 크기: `K-Group 수 × rows × roundup(Gk,256) × 2 × 16`. 다음 M/N Group이 재사용한다.
- 조건을 만족하지 않는 MAC(`lowerMAC`)은 Group마다 partial을 따로 계산해 ADD한다.

### 5.5 GEMV: `yᵀ = xᵀ · Wᵀ`

`linalg.matvec` (`y[m] = Σ W[m,k]·x[k]`)은 lane이 M 방향으로 가도록 operand 역할을 바꾼다.

| 역할 | GEMM | GEMV |
| --- | --- | --- |
| vecA (relayout) | A | x (행 1개) |
| B (MAC 스트림) | B | W (전치 저장 `[K × M]`) |
| C | C | y (행 1개) |
| batch | `batch` 옵션 | **1 고정** |
| output block | `b × 4096/b` | `1 × 4096` (M 방향) |
| 계산 열 | `roundup(Gn, 4096/b)` | `roundup(Gm, 4096)` |
| K step | `256/b` | 256 |

- W와 x, y의 stage/unstage에 `transpose`가 붙는다. stub이 host `(m, k)`를 panel
  `[k × pc + m]`에 쓴다.
- b=1이라 K step이 256이고 Gk가 256의 배수가 아니면 Group 안의 0 행을 읽는다. 끝의
  padding MAC은 생기지 않는다.
- Arel은 `K-Group 수 × 256 × 2 × 16`으로 작다. weight는 재배치되지 않는다.

### 5.6 그 밖의 연산

| Common 연산 | lowering |
| --- | --- |
| `pim.load` | placement panel을 scratch로 copy (snapshot) |
| `pim.store` | scratch → placement panel copy |
| `pim.combine kind="add"` | 서로 다른 M/N 출력에 단독 기여함이 증명되면 copy, 아니면 ADD. 대상 placement는 0으로 초기화 |
| `linalg.fill` (상수) | scalar global을 stage로 채운 상수 buffer |
| identity-map `linalg.generic`, `arith.add/sub/mulf` | elementwise descriptor |
| `arith.extf` (elementwise generic 안) | 입력 view를 그대로 쓴다. 저장 형식이 bf16이라 값이 같다 |
| 일반 `scf.for` (tensor carry) | carry buffer + copy. index carry는 그대로 |

copy = 8192 B씩 `opcode copy`. add/sub: A → vACC(`a_mask|op|descr_dst_mask`),
B(`b_mask|pim_mode|op`), C ← vACC(`c_mask|op`). mul: A(`a_mask|mul|pim_mode`), B, C.

### 5.7 mask

- 정적으로 full이거나, Group 하나에만 걸린 rectangular `valid`는 stage/unstage 범위로
  내린다. 입력 padding은 positive-zero 또는 미지정만 허용한다.
- 여러 Group에 걸친 mask는 `min(max(end - origin, 0), size)` 꼴의 공통 상한이 있을 때만
  placement 전체 범위로 옮긴다.
- Group마다 다른 mask, mask가 걸린 read/write alias는 오류다.

### 5.8 Execution Tile 사이의 K

K가 여러 Execution Tile로 나뉘면(`pim.scheduling_axis = "reduction_k"`) vACC는
`pim_exec`을 건너 살아남지 못한다. 각 Tile의 bf16 결과를 `aimx.unstage accumulate`가
host에서 FP32로 더한다(첫 Tile은 대입).

`p_k > 1`(FPGA 사이 K 분할)도 같은 경로다. FPGA마다 자기 K 구간의 partial을 vACC에서
끝내고, combine 출력의 unstage가 `accumulate = (K round > 0) || (k 좌표 ≠ 0)`이 된다.
FPGA thread가 동시에 같은 host 출력에 쓰지 않도록 lowering이 그 unstage에
`aimx.device_reduction`을 붙이고, stub이 device loop에 `omp ... ordered`를, unstage에
`#pragma omp ordered`를 낸다. stage·invoke는 FPGA마다 병렬이고 unstage만 device 순서(k = 0이
대입, 이후 누산)로 실행되므로 결과는 실행마다 같다.

### 5.9 pass 옵션

| 옵션 | pass 기본 | compiler 기본 (`PIMOptions`) | 뜻 |
| --- | --- | --- | --- |
| `batch` | 8 | `batch = 8` | GEMM output block 행 수. torch 경로는 GEMM마다 `min(batch, M 이상 2의 거듭제곱)`으로 낮춘다(§9). GEMV는 쓰지 않는다 |
| `persistent` | true | `aimx_persistent = True` | 할당·chain을 FPGA별로 상주 |
| `matmul-row-alignment` | 0 (= `4096/b`) | `aimx_matmul_row_alignment = 4096` | GEMM B/C Group panel 행 pitch |
| `matmul-useful-n-only` | false | `aimx_matmul_useful_n_only = True` | 유효 N microtile만 계산 |
| `matmul-k-alignment` | 0 (= `4096/b`) | `aimx_matmul_k_alignment = 0` | GEMM A 행(K) pitch |

---

## 6. descriptor program (`DescriptorBuilder.cpp`)

view 인자는 `(buffer, byte offset, byte pitch)`다. 개수는 생성 위치에서 함께 센다
(`ProgramBuilder::desc`).

| program | 반복 | descriptor | len | opcode |
| --- | --- | --- | --- | --- |
| `copy` | rows·cols / 4096 | src → dst | 8192 | `copy` |
| `elementwise` | rows·cols / 4096 | A, B, C 3개 | 8192 | §5.6 |
| `matmul_relayout` | `m/b` × `k/256` | ① A gather (A view, pitch) | 512 | `pim_mode\|stride` |
| | | ② Arel 쓰기 (`row·k·32 + (col/256)·8192·b`) | `8192·b` | `bc\|tp` |
| `matmul_accumulate` | `K / (256/b)` | ① Arel | 8192 | `a_mask\|matmul\|pim_mode` |
| | | ② B (`k·pitch`) | `8192/b` | `b_mask\|matmul\|pim_mode\|stride` |
| `matmul_padding` | 0 MAC 수 | accumulate와 같음 (0 영역) | | |
| `matmul_drain` | 1 | vACC (pitch `8192/b`) → C (pitch) | `8192/b` | `c_mask\|matmul\|stride` |
| `matmul`/`matvec` | relayout + output block × K + drain | K Group 누산이 아닌 MAC | | |

### 6.1 `--aimx-to-descriptors`

program마다 `static uint32_t aimx_prog_<name>(pim_isa_32_t *code, uint32_t cap,
uint32_t idx, uint64_t *next, int fpga_id, ...)`를 낸다. record 수가 `cap`을 넘으면
쓰기 전에 멈춘다. status word = opcode | batch 비트. 주소 재사용 분류가
`Addresses`인 program은 주소만 고치는 `aimx_patch_<name>`도 낸다.

### 6.2 chain 규약

- chain buffer의 첫 group 다음(`pa + 512`)부터 group을 쌓는다.
- `pim_args.desc_idx = idx - 1`, `desc_last = next - 1024` (마지막 group 시작).
- chain은 자기 물리 주소를 담으므로 `pim_exec`가 끝나기 전에는 어떤 buffer도 놓지 않는다.

---

## 7. runtime stub (`--aimx-to-runtime-stub`)

### 7.1 진입점

```c++
extern "C" void <kernel>(T* a0, T* a1, ...) {
  static std::mutex _host_mutex; ...
  aimx_runtime_init(AIMX_BASE_FPGA, <acquired>);   // init_mpim_drv, FPGA 범위 검사
  for (Common rounds) {
    #pragma omp parallel for collapse(..) num_threads(P)   // FPGA마다 host thread 하나
    for (device 좌표) {
      int _fpga = AIMX_BASE_FPGA + linear(좌표);
      aimx_resident_context *_resident = aimx_resident_acquire(KEY ":site", _fpga, N);
      alloc → construction → stage → invoke → unstage
      aimx_resident_release(_resident);
}}}
```

- `aimx_require`는 `exit()`로 끝낸다. vendor가 hugepage를 돌려주는 경로가 `atexit`뿐이라
  `abort()`하면 hugepage가 샌다.
- host ↔ PIM 전송은 2 MiB씩 나눠 `datacopy_*`를 부른다(큰 요청에서 커널 `kmalloc`
  ENOMEM 실측).

### 7.2 상주 (`aimx.persistent`)

`aimx_resident_acquire(plan, fpga, buffers)`는 FPGA마다 plan 하나를 캐시한다. plan key는
`AIMX_KERNEL_KEY`(stub·descriptor·runtime·base_fpga 해시) + site다. key나 buffer 수가
바뀌면 buffer와 chain을 함께 버리고 다시 잡는다. 같은 kernel의 다음 호출은 할당 없이
같은 주소를 쓴다.

### 7.3 chain 재사용 (`DescriptorReuse.h`)

construction 경계마다 descriptor 인자의 불변성을 분류한다.

| 분류 | 조건 | 동작 |
| --- | --- | --- |
| `Unchanged` | 모든 주소·loop 범위가 상주 buffer·device 좌표·상수에만 의존 | `_resident->ready`이면 build·업로드 생략. 업로드 뒤 host record 페이지 반납 |
| `Addresses` | loop 구조는 같고 주소만 바뀜 | `aimx_patch_*`로 src/dst만 고쳐 다시 업로드 |
| `Rebuild` | 그 밖 | 매번 build·업로드 |

동적 상한 `min(tileGroups, remaining)`은 full tile일 때만 `ready`를 세우고, tail은
같은 저장소를 덮어쓰므로 `ready`를 내린다.

### 7.4 stage / unstage

| 경로 | 조건 | 동작 |
| --- | --- | --- |
| `group_layout` | 모든 placement (현재 lowering) | panel마다 유효 행·열 복사, 행 끝과 panel 끝만 memset |
| `group_layout` + `transpose` | GEMV의 W, x, y | panel `(k, m)` ← host `(m, k)`. host 행 stride가 1이면(이미 물리 순서, torch `x·W`) 물리 행마다 연속 복사, 아니면 32×32 block으로 전치한다. host 32행을 바깥에 두고 K-Group·K를 안쪽에서 끝까지 돌아 행마다 순차로 읽고(prefetch, TLB), panel에는 64 B씩 쓴다. 순회 순서만 다르고 결과는 같다 |
| `shared_over` | broadcast 입력 | 공유 좌표 조합마다 host pool 한 칸. 처음 온 thread만 변환(`std::call_once`), 전송은 FPGA마다 |
| unstage `accumulate` | K Tile 2번째부터 | bf16 → float 덧셈 |

host 변환: f32 → bf16은 round-to-nearest-even, f16 → bf16은 65,536칸 표, bf16은 그대로. 조립 buffer는
thread별로 재사용한다(page fault 제거). stub은 `-O3`로 빌드한다.

### 7.5 timing

`pim_timing_ms[8]`을 누적한다(여러 FPGA면 thread 합계). 순서는
`driver._STAGE_NAMES["aimx"]`와 같다.

| 칸 | 뜻 |
| --- | --- |
| `alloc` | 할당·해제 |
| `build` / `desc_h2d` | descriptor 생성(patch 포함) / chain 업로드 |
| `layout_in` / `h2d` | stage의 host 조립 / 업로드 |
| `invoke` | `pim_exec` (동기) |
| `d2h` / `layout_out` | unstage의 다운로드 / host scatter |

`pim_invoke_count`가 실제 `pim_exec` 횟수다.

---

## 8. 모델 session (`--aimx-to-model-stub`)

`PIMSession` 경로는 같은 target IR을 model stub으로 다시 번역한다. 상세는
[Model Runtime 설계](ModelRuntime_Design.md)에 있다. AiMX 쪽 차이만 적는다.

| 항목 | 내용 |
| --- | --- |
| context | session이 FPGA별로 만든 `aimx_resident_context`를 인자로 받는다. kernel key로 찾지 않는다 |
| 실행 단위 | job = batch slot × device. job마다 FPGA 하나, context 하나 |
| 호출 종류 (`upload`) | 0 실행 (activation stage, invoke, unstage), 1 weight 업로드, 2 pack (host image에만 조립), 3 chain build만 |
| 표 | `pim_model_slot_bytes/input/output/kind`로 slot 크기와 역할을 Python에 알린다 |
| 주소 | memory table: weight → state(chain, PMM) → activation scratch. region 하나 = `pim_malloc` 하나(≤ 1023 × 2 MiB) |
| 정렬 | operand 시작을 256 KiB로 맞춘다. data buffer 8 KiB 정렬은 batch 8에서 결과가 틀렸다(실측) |

resident weight GEMM은 instance 1개, scheduling round 1회로 제한한다(`select_layer_config`).
여러 round가 같은 buffer에 다른 weight tile을 올리면 상주가 성립하지 않기 때문이다.

### 8.1 key/value cache 상주

host cache는 `[위치, kv_heads × head_dim]`(한 행 = 토큰)이고, stage의 cache operand는 그 열
window(연속된 head)이거나 그 전치다(QK의 B = Kᵀ). cache stage는 batch round 없이 instance i가
FPGA `base + i` 하나를 혼자 쓴다(instance 수 ≤ FPGA 수, `devices = 1`). 그래서 window마다 주소가
하나로 고정되고 chain도 바뀌지 않는다.

| 항목 | 내용 |
| --- | --- |
| 주소 | cache operand slot은 memory table의 `("kv", layer, KEY/VALUE)` 항목이다. phase 영역 위 `[kv_base, capacity)`이고 별도 region에 묶인다. phase는 하나만 허용 |
| 실행 | cache 인자는 weight처럼 실행(upload 0) 호출에서 stage하지 않는다 |
| 배치 | model stub이 slot마다 `pim_model_slot_layout` (group_layout 6칸 + transpose)을 내보낸다. B 행 r, 열 c는 panel `(r / ur, c / uc)`(row-major), panel 안 `(r % ur, c % uc)`에 놓인다 |
| 쓰기 | `write_kv`가 위치를 대기열에 넣고, 다음 cache stage 실행 직전에 region host shadow에 bf16으로 조립한 뒤 항목·FPGA마다 바뀐 byte 구간 하나를 `aimx_model_push_ranges`로 보낸다 |
| 대조 | stage의 첫 쓰기에서 cache 전체를 stub pack(upload 2)으로 조립한 결과와 계산한 배치를 비교한다 |
| 비용 | V(B 행 = 위치)는 새 위치의 행만 보낸다. Kᵀ(B 열 = 위치)는 새 위치가 모든 행에 걸쳐 있어 그 열 group의 panel 전체를 보낸다 |

---

## 9. autotuner 연동 (`autotuner/model.py`의 `AiMXModel`)

| 항목 | GEMM (`matmul`) | GEMV (`matvec`) |
| --- | --- | --- |
| Block | `(min(8, b), 16, 16)` | `(8, 1, 16)` |
| Group 상한 (`_compute_unit`) | `(b, 4096, max(256, kAlign))` | `(4096, 1, 256)` |
| profile spec | Group `unit / Block`, f16 → f32 | Group `(512, 1, 16)`, f16 → f32 |
| `scheduled_compute` | `tile_m·tile_n · roundup(Gm,b) · roundup(Gn,4096/b) · roundup(tile_k·macK, 256)` | `tile_m · roundup(Gm,4096) · tile_k · roundup(Gk,256)` |
| operand 대응 | A(m,k), B(k,n), C(m,n) | W(m,k) → B, x(k) → A, y(m) → C |

`aimx_runtime_allocations`는 lowering과 같은 식으로 PMM, A, Arel, B, C, descriptor chain
하한을 계산한다(GEMV는 §5.2의 전치 크기). `ConstraintEngine`은 단일 할당 ≤ 1023 × 2 MiB,
bank 사용량 ≤ `bank_capacity_limit_bytes`를 검사한다. 최종 판정은 lowering의
용량 검사다.

**GEMM batch는 GEMM마다 정한다.** `AiMXModel.stage_target(problem)`이 GEMM의 batch를
`min(spec batch, M 이상인 가장 작은 2의 거듭제곱)`으로 바꾼 spec을 돌려주고,
`block(problem)`이 Block m을 `min(8, batch)`로 둔다. 한 B 행을 batch 행이 함께 읽으므로
M ≥ 8이면 8(B 재읽기 최소), M < 8이면 M에 맞춘 값이 output block 행 padding을 없앤다.
torch 경로의 탐색(`tune_stage_class`), 기록 조회(`select_layer_config`), compile
(`resolve_execution`), packing 비용(`fx_packing._cost`)이 모두 이 spec을 쓰므로 기록 key가
일치한다. spec의 `batch`는 상한이다. GEMV는 batch 1 고정이라 spec을 바꾸지 않는다.

| GEMM M | batch | Block m | output block |
| --- | --- | --- | --- |
| 2 | 2 | 2 | 2 × 2048 |
| 3, 4 | 4 | 4 | 4 × 1024 |
| 5 ~ 8 이상 | 8 | 8 | 8 × 512 |

최종 config는 실측 `total_ms`(stage + invoke + unstage wall) 최소값이다. 정렬 키는 측정
순서만 정한다.

| 진입점 | 용도 |
| --- | --- |
| `examples/silent_pim/autotune_gemm.py` | GEMM 탐색, `--profile-only`로 단일 FPGA 포화점 |
| `examples/silent_pim/autotune_gemv.py` | GEMV 탐색(rel L2 ≤ 2% 판정), `--profile-only` |
| `torch_backend` (`model_compile.tune_stage_class`) | 모델 stage. decode의 M=1/N=1 GEMM은 matvec 문제로 튜닝 |

GEMV 단일 FPGA 프로파일의 정확도 검사는 `x[::256] = 1`, 기대값 `ceil(K/256)`이다.
ones·ones는 부분합이 1024를 넘어 vACC가 작은 곱을 버린다(§11.3).

---

## 10. 실행 조건

### 10.1 환경

| 조건 | 값 / 확인 |
| --- | --- |
| driver | `/dev/pl_dma*`가 보여야 한다 (`aimx_unit_count`) |
| vendor 소스 | `export AIMX_PIM_SW=$HOME/pim_aimx_sw/pim_sw` (`pim_math.h`가 설치되지 않는다) |
| 라이브러리 | `$AIMX_PIM_SW/lib/libpim.so`를 경로로 직접 링크한다. `-lm` 필요 (`pow` 미해결) |
| 빌드 | `g++ -O3 -std=c++17 -fopenmp -Wno-narrowing -Wno-write-strings` (driver가 수행) |
| FPGA 범위 | `[base_fpga, base_fpga + num_units)`가 보이는 FPGA 수 안이어야 한다 (`aimx_runtime_init`가 검사) |
| 프로세스 | 상주 상태는 fork로 물려받을 수 없다 (자식에서 `_exit(1)`) |

### 10.2 launch 옵션

```python
kernel[grid](..., pim_target="aimx",
             num_units=1, base_fpga=0, batch=8,
             group_m=…, group_n=…, group_k=…,      # Block Group = Block 개수
             tile_m=…, tile_n=…, tile_k=…,          # Execution Tile = Group 개수
             p_m=…, p_n=…, p_k=…)                   # device 분할, Π ≤ num_units
```

### 10.3 커널 조건

| 조건 | 어기면 |
| --- | --- |
| host tensor dtype bf16 / f16 / f32 | 오류 (`tensor storage supports BF16/F16/F32`) |
| tile tensor rank 1 또는 2, 정적 shape | 오류 |
| 연산: matmul, matvec, identity-map elementwise add/sub/mul/extf, 상수 fill | 그 밖은 오류 |
| K 누산은 positive-zero accumulator에서 시작 | vACC 경로 불가 → 오류 |
| mask는 §5.7 범위 | 오류 |
| 할당 하나 ≤ 1023 × 2 MiB, bank 합 ≤ 용량 | 오류 |

### 10.4 성능 조건 (padding)

| 조건 | 이유 |
| --- | --- |
| GEMM `Gn`을 4096의 배수로 | B/C panel 행이 4096 원소다. `Gn = 512`면 B/C가 8배 |
| GEMM/GEMV `Gk`를 256의 배수로 | B 행이 256 단위로 채워진다. 원본도 `q_size % 256 != 0`을 거부한다 |
| GEMV `Gm`을 4096으로 | output block이 M 4096이다. `Gm = 32`면 lane 32/4096만 쓴다 |
| N(GEMM) 또는 M(GEMV)이 4096 미만인 행렬 | 4096까지 padding은 피할 수 없다 (원본도 `CHUNK = 4096` padding) |

autotuner의 Group 상한(§9)이 이 크기를 1순위로 고른다.

---

## 11. FPGA 실측으로 확인된 사실

### 11.1 descriptor와 주소

| 사실 | 근거 |
| --- | --- |
| stride shift는 33 | `matmul.c`의 모든 stride 자리가 `<< 33`이고, `kStrideShift = 33`으로 만든 GEMM·GEMV chain(B·C가 pitch ≠ len)이 FPGA에서 맞는다 |
| group 16 record 중 첫 record만 실행 | record마다 다른 src를 넣은 결과와 모두 같게 쓴 결과가 byte 단위로 동일 (`~/kangmlee/test/e11_perch`) |
| A 재배치 순열 | `Arel[c*256 + g*16 + t] = A[16c + t]` (`m=1, k=256, b=1`) |
| 한 chain에 K·N 여러 block | `./matmul 1 8192 28672 1`이 `r` 7회 × `q` 32회를 chain 하나로 실행 |
| B 한 행(4096)을 b=8이면 512 폭 창 8개로 읽음 | `~/kangmlee/test/e12_narrow_n`, `m=8 k=256 n=4096`, `r_size=4096` 전부 일치 |

### 11.2 행 pitch

| 대상 | 결과 |
| --- | --- |
| GEMM B/C Group panel 행 pitch 512 (b=8) | config마다 틀리고 같은 입력 2회 실행 결과도 다름. 4096으로 바꾸면 모델과 일치 → compiler 기본 4096 |
| GEMM A 행 pitch `4096/b` (b=8: 512), 2048 | 4096과 bit 동일, 2회 실행 동일 |
| 모델 경로 operand 정렬 8 KiB (b=8) | 결과가 틀림 → 256 KiB로 정렬 |
| B/C 행 pitch < 4096 (전체 N이 4096 미만) | **미확인.** `~/kangmlee/test/e12_narrow_n`을 `M=8 N=512`로 실행해야 한다 |

### 11.3 vACC 수치 특성

vACC lane = 부호 1 + 그룹 지수 G 5 bit + 가수 16 bit (22 bit), 값 `±가수 × 2^(8G−134)`.

| G | \|acc\| 범위 | LSB | 버려지는 곱 |
| --- | --- | --- | --- |
| 15 | < 4 | 2⁻¹⁴ | \|p\| < 2⁻¹⁵ |
| 16 | 4 ~ 1024 | 1/64 | \|p\| < 2⁻⁷ |
| 17 | 1024 ~ 262144 | 4 | \|p\| < 2 |

- G 차이 −1인 곱은 8 bit 밀어 더하고, −2 이하는 버린다. G는 커지기만 한다.
- drain은 절사다. 가수가 0이면 `±2^(8G−127)`을 낸다(G=16이면 ±2.0).
- 0-상쇄 부호: 가수가 0이면 부호는 그 단계에 들어온 곱의 부호다(vendor 시뮬레이터와
  다른 유일한 점). 이 규칙을 넣은 Python MAC 모델이 FPGA와 bit 단위로 일치한다.
- 입력 0은 누산을 바꾸지 않는다. 그래서 0 padding은 결과에 영향이 없다.

정확도 (M=128, N=512, K=4096, FP64 대비 relative L2):

| 분포 | k_t=256 | 1024 | 4096 |
| --- | --- | --- | --- |
| signed N(0,1)·N(0,1) | 0.80% | 0.78% | 0.78% |
| positive \|N\|·\|N\| | 0.61% | 0.63% | **48.5%** (1024 포화) |
| llm N(0,1)·N(0,0.02) | 0.40% | 0.41% | 0.79% |

무난한 범위는 Execution Tile K `256 ~ 1024`다. compiler는 FP32 누산이나 CPU와의 bit
일치를 약속하지 않는다.

### 11.4 GEMV 실측 (2026-10-25, FPGA 1장)

`examples/silent_pim/autotune_gemv.py --m 4096 --k 4096`, llm 분포, 선택 config는 모두
Block (8,1,16), Group (512,1,16) → extent (4096,1,256), Tile (1,1,4), P (1,1,1)이다.

| 단계 | 입력 | `layout_in` | `h2d` | `invoke` | wall | rel L2 |
| --- | --- | --- | --- | --- | --- | --- |
| 전치 배치 도입 (host 행 → panel 열) | f32 | 33.865 ms | 9.200 ms | 0.658 ms | 44.898 ms | 0.479% |
| block 전치, K 바깥 | f32 | 30.622 ms | 10.095 ms | 0.662 ms | 42.488 ms | 0.479% |
| block 전치, M 32행 바깥 (현재) | f32 | 23.828 ms | 8.968 ms | 0.653 ms | 34.626 ms | 0.479% |
| 같음 + bf16 입력 (변환 없음) | bf16 | **12.949 ms** | 9.737 ms | 0.655 ms | **24.449 ms** | **0.410%** |

- 단일 스레드 하한 참고: 연속 f32 → bf16 변환 1,678만 원소(torch)가 10.6 ms.
- rel L2는 f32 입력이면 입력의 bf16 양자화 오차를 포함한다. bf16 입력의 0.410%는 PIM 누산
  오차만이고 §11.3의 llm 바닥(약 0.40%)과 같다.
- 전치 배치 이전에는 같은 GEMV가 output block `8×512` 중 `8×1`만 유효했다(lane 1/512).

**호출마다 W를 stage하는 경로는 host가 지배한다.** bf16에서도 `layout_in + h2d`가 wall의
93%다. 모델 session처럼 W가 상주하면 호출마다 `invoke`와 x/y 전송만 남는다.

**K Tile 크기에 따라 `layout_in`이 달라진다** (bf16, Group (512,1,16) 고정, autotune 측정):

| Tile K (K-Group 수) | K Tile 수 | W stage buffer | `layout_in` | `invoke` | `d2h` | wall |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 1 | 32 MiB | 24.99 ms | 0.63 ms | 0.08 ms | 35.66 ms |
| 8 | 2 | 16 MiB | 19.73 ms | 0.64 ms | 0.16 ms | 30.25 ms |
| 4 | 4 | 8 MiB | 14.19 ms | 0.65 ms | 0.32 ms | 25.31 ms |

조립 buffer가 작을수록 host cache에 남아 조립이 빠르다(추정). 그래서 autotuner가 Tile K 4를
고른다. W가 상주하는 경로에서는 `layout_in`이 없으므로 이 근거가 사라지고, `d2h`와 host
FP32 합산이 적은 Tile K 16이 유리할 수 있다.

Group M 256(W panel이 4096 폭으로 padding되어 B 저장·전송 2배)은 `h2d`가 18 ms로 두 배가
되어 모두 졌다(§10.4).

---

## 12. 미구현과 알려진 제약

| 항목 | 상태 |
| --- | --- |
| GEMM batch 혼합 (예: M=12 → 8 + 4) | 미구현. 한 Tile에 batch 하나. M은 `roundup(Gm, b)`로 padding. 혼합해도 B 읽기는 행 묶음 수(2)로 같고 descriptor만 12/16로 준다 |
| B/C 행 pitch < 4096 | 하드웨어 미확인 (§11.2) |
| GEMV W stage | §11.4: f32 23.8 ms, bf16 12.9 ms. `[K][M]` 배치(① 연속 복사)의 4096² 측정은 아직 없음. 호출마다 W를 올리는 측정이 autotuner 선택(Tile K)에 영향을 줌 |
| device 사이 K 분할 (`p_k > 1`) | 구현 (§5.8). FPGA 실측 없음 |
| Group마다 다른 mask | 미구현 |
| A 재배치 융합 / host에서 Arel 직접 생성 | 미구현. host Arel로 채널마다 다른 vecA를 주는 것은 실측 확인, 설계 보류 ([AiMX_ChannelUnits.md](AiMX_ChannelUnits.md)) |
| GEMV 커널의 rank-1 masked x/y | 예제 커널은 full-block만 허용한다 (`examples/silent_pim/gemv.py`) |
| stage와 invoke 겹치기 | 없음. `pim_exec`가 동기이고 FPGA마다 host thread 하나 |
