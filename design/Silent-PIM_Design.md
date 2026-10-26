# Silent-PIM (AiMX) Target 설계

Placed PIM Common IR을 Silent-PIM(AiMX) FPGA의 CDMA descriptor chain으로 내리는 계층의
현재 구현과 실행 조건을 정리한다. Common IR의 계약은
[PIM Common IR 설계](PIMCommonIR_Design.md), 모델 session의 상주 정책은
[Model Runtime 설계](ModelRuntime_Design.md)가 다룬다.

| 근거 | 경로 |
| --- | --- |
| 기계 상수·opcode | `~/pim_aimx_sw/pim_sw/src/pim_math_src/pim_math.h` |
| 참조 matmul | `~/pim_aimx_sw/pim_sw/src/pim_math_src/matmul.c` |
| AiMX dialect | `include/triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.{h,td}`, `AiMXFootprint.h`, `lib/Dialect/PIMTarget/AiMX/IR/AiMXDialect.cpp` |
| lowering | `lib/Conversion/PIMToAiMX/PIMToAiMXPass.cpp`, `DescriptorBuilder.cpp`, `include/triton-shared/Conversion/PIMToAiMX/Passes.td` |
| codegen | `lib/Target/AiMX/AiMXToDescriptors.cpp`, `AiMXToRuntimeStub.cpp`, `DescriptorReuse.h` |
| runtime | `pim_backend/backend/aimx_runtime.h`, `aimx_resident_runtime.cpp` |
| Python | `pim_backend/backend/{compiler,driver,target}.py`, `aimx_adapter.py`, `aimx_model.py` |
| autotuner | `pim_backend/autotuner/model.py` (`AiMXModel`, `aimx_runtime_allocations`) |
| 예제 | `pim_backend/examples/silent_pim/`, `pim_backend/torch_backend/examples/aimx/` |

PIM Common IR은 이 target을 위해 고치지 않는다. AiMX는 Common이 정한 Block, Block Group,
Execution Tile, device mapping, K 누산 순서, host 전송 순서를 그대로 받고 **물리 저장
형태와 descriptor 열만** 정한다.

---

## 1. 하드웨어

### 1.1 구성 (`backend/target.py`)

| 항목 | 값 |
| --- | --- |
| unit | FPGA 1장 = GDDR6 칩 8개 (칩당 4 GB, 2 channel) |
| channel / bank | 16 channel × 16 bank = 256 bank (`banks_per_unit`) |
| ALU (lane) | bank당 16 → **4096 lane** |
| 물리 용량 | 32 GB (bank 128 MB) |
| 할당 가능 용량 | FPGA당 runtime heap `0x1FEE00000` (8 GB − 18 MB). 넘으면 vendor runtime이 `PIM] OOM`으로 종료 |
| column 접근 | 256 bit = bf16 16개 |
| 데이터 형식 | PIM DRAM 안은 언제나 bfloat16 |

Common profile의 `bank_capacity_limit_bytes`는 `0x1FEE00000 / 256 = 33,480,704`다
(`AIMX_BANK_CAPACITY_BYTES`). 물리 bank 용량(128 MB)은 상수로만 둔다.

### 1.2 기계 상수 (`AiMXDialect.h`)

| 상수 | 값 | 뜻 |
| --- | --- | --- |
| `kNumChannels` / `kNumBanks` / `kALUSize` | 16 / 16 / 16 | channel, channel당 bank, bank당 lane |
| `kTypeSize` | 2 | bf16 |
| `kBurstSize` | 32 B | DRAM burst = bank 하나의 lane 16개 |
| `kRegSize` | 256 원소 | vecA 한 번 적재가 담는 K |
| `kPIMWidth` | 512 B | channel 하나의 operand 폭 |
| `kDeviceWidth` / `kLanes` | 8192 B / 4096 | lane 4096개 × 2 B |
| `kStrideShift` | 33 | pitch를 주소에 `addr \| (pitch << 33)`로 싣는다 |
| `kRecordBytes` / `kGroupBytes` | 32 B / 512 B | descriptor group = record 32 B × 16 |
| `kPMMSize` / `kVACCOffset` / `kPMMAlign` | `0x30000` / `0x10000` / 256 KiB | vACC = PMM + `0x10000` |
| `kMaxRuntimeAllocationBytes` | 1023 × 2 MiB | vendor allocator가 크기를 `int`로 받아 2 GiB에서 overflow |

### 1.3 descriptor

`pim_isa_32_t`는 Xilinx CDMA scatter-gather descriptor다(next, src, dst, len, status).
PIM 동작은 `status`의 flag로 지정하고, memory controller에는 평범한 DMA로 보인다
(**silent**). descriptor 안의 src/dst는 **절대 물리 주소**(`addr | pitch << 33`)이고 device가
주소를 계산하지 않는다. 그래서 chain은 buffer 물리 주소에 묶인다(§7.3, §8).

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

- `aimx.desc` 하나 = descriptor group 하나 = 같은 내용의 record 16개(`PIM_RD_INSTR_MC`).
- `len`은 **출력** byte 수다.
- batch 비트는 flag enum에 없다. `aimx.descriptor_program`의 `batch`에서 codegen이
  유도한다(`batchBits`).

### 1.4 output block, batch, vecA

```text
batch b : lane 4096개를 b행 × (4096/b)열로 자른다     b ∈ {1, 2, 4, 8}
           K 한 step = 256/b 원소 (vecA 한 번이 b행 × 256/b)

b = 1     b = 8
1 × 4096  8 × 512
```

- A는 vecA에 그대로 들어가지 않는다. `bc|tp` descriptor가 A를 **Arel**(A의 16배 크기,
  `× kNumChannels`)로 재배치한다. 순열은 [AiMX_ChannelUnits.md](AiMX_ChannelUnits.md) §1에 있다.
- vACC는 output block 하나 깊이다. block 하나를 다 누산하고 drain해야 다음 block으로
  갈 수 있고 `pim_exec`을 건너 살아남지 못한다.

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
  │ (모델 session은 같은 target IR을 --aimx-to-model-stub으로)
  ▼ driver.py: g++ -O3 -fopenmp → pim_host.so  (+ libaimx_resident_<hash>.so, libpim.so)
```

- `--pim-reuse-execution-tile-inputs`를 빼는 이유: AiMX의 `pim.load`는 상주 panel의 view라
  hoist 이득이 없고, vACC 누산은 K Group 안에 A/B load가 직접 있어야 한다.
- 마지막 `-canonicalize`를 빼는 이유: 일반 SCF canonicalization이 1회 loop과 그 축 속성을
  지울 수 있다. AiMX는 그 loop/combine 구조를 그대로 읽는다.
- bundle에는 stub, descriptors, `aimx_allocations`, `target_ir`(model stub 재번역용)가 들어간다
  (`_aimx_to_artifact`).

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
| `aimx.foreach_device %set axes [m,n,k] counts [...]` | device 좌표 loop. counts의 곱 = acquire 수. `aimx.persistent`이면 할당·chain을 FPGA별로 상주 |
| `aimx.alloc %dev bytes B align A kind <data\|descriptor\|pmm>` | PIM DRAM 할당. 크기는 compile time 상수. pmm은 `0x30000` B·256 KiB 정렬, descriptor는 512 B 배수 |
| `aimx.stage %src origin sizes strides pad_to pitch [transpose] into %buf {group_layout, shared_over}` | host gather + bf16 변환 + 0 padding + 업로드 |
| `aimx.build_descriptors @prog(args) into %chain` | descriptor program을 chain에 펼친다. `scf.execute_region {no_inline}` 안에서만 |
| `aimx.invoke %dev, %chain` | `pim_exec`. 끝날 때까지 기다린다 |
| `aimx.unstage %buf into %dst origin sizes strides pitch accumulate [transpose] {group_layout}` | download + bf16 → host 타입 + 대입 또는 누산 |
| `aimx.require %cond, "msg"` | 실행 시점 layout 검사 |

`group_layout = [useful_rows, useful_cols, physical_rows, physical_cols, row_groups,
col_groups]`은 **물리 순서**다. Group panel은 group grid에서 row-major로 이어진다.
`transpose`가 붙으면 host 행이 물리 열, host 열이 물리 행이다. dialect에는 주기 배치
`group_rows`/`group_cols`도 있고 stub이 처리하지만, 현재 lowering은 `group_layout`만 낸다.

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

pass 진입 조건: target profile이 `aimx`이고 `banks_per_unit == kNumBanks × kNumChannels`(256),
`bank_capacity_limit_bytes > 0`, `pim-form-host-plan`이 끝난 IR.

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
  돌려줄 수 있다. 비상주 stub은 모든 data buffer가 `[PMM, PMM+0x30000)`과 겹치지 않는지
  실행 시점에 검사하고, 상주 runtime은 모든 buffer 쌍이 겹치지 않는지 검사한다.
- 용량: 할당마다 `ceil(bytes/8192) × 32`를 bank 사용량으로 더해 `bank_capacity_limit_bytes`를
  넘으면 오류다. 할당 하나가 `kMaxRuntimeAllocationBytes`를 넘어도 오류다. 결과는
  `aimx.allocations`, `aimx.bank_bytes`, `aimx.descriptor_bytes` 속성에 남는다.
- chain 크기 = descriptor group 수 × 512 B. group 수는 program이 desc를 내는 자리에서 loop
  반복 수를 곱해 센다(`ProgramBuilder::desc`, `DeviceLowering::append`).

### 5.2 물리 layout 규칙

tensor SSA 값마다 `Layout {usefulRows, usefulCols, rows, cols, transposed}`를 둔다.
useful은 Common Group의 논리 크기, rows/cols는 물리 크기다. SSA로 연결된 값(loop carry,
같은 access, elementwise 피연산자, MAC accumulator와 결과)은 한 layout을 공유한다.

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

`pim.place` 하나가 PIM DRAM 할당 하나다. 같은 memref의 동일한 입력/출력 placement는
저장소를 공유한다(나중의 `pim.load`가 앞의 store를 봐야 한다).

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
- loop 몸체 = A load, B load, `linalg.matmul`/`linalg.matvec`, yield (`pim.access`, index
  연산, float 상수만 추가 허용)
- MAC 피연산자가 같은 loop 몸체의 `pim.load` 결과다. MAC 몸체 안의 `arith.extf`(bf16 입력을
  넓힌 것)는 `validateMAC`이 허용한다
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
- Arel 크기: `K-Group 수 × rows × roundup(Gk,256) × 2 × 16`. 같은 lowering 위치의 모든
  M/N Group이 재사용한다. K-Group 수가 이 용량을 넘으면 `aimx.require`가 멈춘다.
- 조건을 만족하지 않는 MAC(`lowerMAC`)은 partial을 따로 계산해(`buildMatmul`) accumulator에 ADD한다.

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
  `[k × pc + m]`에 쓴다(§7.4).
- K step이 256이므로 MAC 길이가 언제나 256의 배수다. 끝의 padding MAC은 생기지 않고,
  Gk가 256의 배수가 아니면 panel 안의 0 행을 읽는다.
- Arel은 `K-Group 수 × roundup(Gk,256) × 2 × 16`이다. weight는 재배치되지 않는다.

### 5.6 그 밖의 연산

| Common 연산 | lowering |
| --- | --- |
| `pim.load` | placement panel을 scratch로 copy (snapshot) |
| `pim.store` | scratch → placement panel copy |
| `pim.combine kind="add"` | 서로 다른 M/N 출력에 단독 기여함이 증명되면 copy, 아니면 ADD. 대상 placement는 0으로 초기화. 다른 kind는 오류 |
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

### 5.8 Execution Tile·FPGA 사이의 K

K가 여러 Execution Tile로 나뉘면(`pim.scheduling_axis = "reduction_k"`) 각 Tile의 bf16
결과를 `aimx.unstage accumulate`가 host에서 더한다(첫 Tile은 대입). host 타입이 f32면
FP32 누산이고, f16/bf16이면 Tile마다 그 타입으로 반올림된다(`addBF16`).

`p_k > 1`(FPGA 사이 K 분할)도 같은 경로다. FPGA마다 자기 K 구간의 partial을 vACC에서
끝내고, combine 출력의 unstage가 `accumulate = (K round > 0) || (k 좌표 ≠ 0)`이 된다.
lowering이 그 unstage에 `aimx.device_reduction`을 붙이고, stub이 device loop에
`omp ... ordered`를, unstage에 `#pragma omp ordered`를 낸다. stage·invoke는 FPGA마다
병렬이고 unstage만 device 순서(k = 0이 대입, 이후 누산)로 실행되므로 결과는 실행마다 같다.
model stub도 job loop에 같은 절을 낸다.

### 5.9 pass 옵션

| 옵션 | pass 기본 | compiler 기본 (`PIMOptions`) | 뜻 |
| --- | --- | --- | --- |
| `batch` | 8 | `batch = 8` | GEMM output block 행 수. torch 경로는 GEMM마다 낮춘다(§9). GEMV는 쓰지 않는다 |
| `persistent` | true | `aimx_persistent = True` | 할당·chain을 FPGA별로 상주 |
| `matmul-row-alignment` | 0 (= `4096/b`) | `aimx_matmul_row_alignment = 4096` | GEMM B/C Group panel 행 pitch. `4096/b`의 배수 |
| `matmul-useful-n-only` | false | `aimx_matmul_useful_n_only = True` | 유효 N microtile만 계산 |
| `matmul-k-alignment` | 0 (= `4096/b`) | `aimx_matmul_k_alignment = 0` | GEMM A 행(K) pitch |

---

## 6. descriptor program (`DescriptorBuilder.cpp`)

view 인자는 `(buffer, byte offset, byte pitch)`다.

| program | 반복 | descriptor | len | opcode |
| --- | --- | --- | --- | --- |
| `copy` | rows·cols / 4096 | src → dst | 8192 | `copy` |
| `elementwise` | rows·cols / 4096 | A, B, C 3개 | 8192 | §5.6 |
| `matmul_relayout` | `m/b` × `k/256` | ① A gather (A view, pitch) | 512 | `pim_mode\|stride` |
| | | ② Arel 쓰기 (`slot + row·k·32 + (col/256)·8192·b`) | `8192·b` | `bc\|tp` |
| `matmul_accumulate` | `K / (256/b)` | ① Arel | 8192 | `a_mask\|matmul\|pim_mode` |
| | | ② B (`k·pitch`) | `8192/b` | `b_mask\|matmul\|pim_mode\|stride` |
| `matmul_padding` | 0 MAC 수 | accumulate와 같음 (0 영역) | | |
| `matmul_drain` | 1 | vACC (pitch `8192/b`) → C (pitch) | `8192/b` | `c_mask\|matmul\|stride` |
| `matmul`/`matvec` | relayout + output block × K + drain | K Group 누산이 아닌 MAC (`buildMatmul`) | | |

### 6.1 `--aimx-to-descriptors`

program마다 `static uint32_t aimx_prog_<name>(pim_isa_32_t *code, uint32_t cap,
uint32_t idx, uint64_t *next, int fpga_id, ...)`를 낸다. buffer 인자는 물리 주소다.
record 수가 `cap`을 넘으면 쓰기 전에 멈춘다. status word = opcode | batch 비트.
주소 재사용 분류가 `Addresses`인 program은 주소만 고치는 `aimx_patch_<name>`도 낸다
(`aimx_patch_descriptor_addresses`가 record 16개의 src/dst만 바꾼다).

### 6.2 chain 규약

- chain의 `next`는 `pa + 512`에서 시작한다.
- `pim_args.desc_idx = idx - 1`, `desc_last = next - 1024` (마지막 group 시작).
- chain은 buffer 물리 주소를 담으므로 `pim_exec`가 끝나기 전에는 어떤 buffer도 놓지 않는다.

---

## 7. runtime stub (`--aimx-to-runtime-stub`)

### 7.1 진입점

```c++
extern "C" void <kernel>(T* a0, T* a1, ...) {
  static std::mutex _host_mutex; ...
  aimx_runtime_init(AIMX_BASE_FPGA, <acquired>);   // init_mpim_drv, FPGA 범위 검사
  for (Common rounds) {
    #pragma omp parallel for [ordered] collapse(..) num_threads(P)   // FPGA마다 host thread 하나
    for (device 좌표) {
      int _fpga = AIMX_BASE_FPGA + linear(좌표);
      aimx_resident_context *_resident = aimx_resident_acquire(KEY ":site", _fpga, N);
      alloc → construction → stage → invoke → unstage
      aimx_resident_release(_resident);
}}}
```

- `aimx_require`는 `exit()`로 끝낸다. vendor가 hugepage를 돌려주는 경로가 `atexit`뿐이라
  `abort()`하면 hugepage가 샌다.
- host ↔ PIM 전송은 2 MiB씩 나눠 `datacopy_*`를 부른다(`AIMX_DMA_CHUNK_BYTES`; 큰 요청에서
  커널 `kmalloc` ENOMEM 실측).
- 비상주(`persistent=false`) 경로는 매 호출 `pim_malloc`/`pim_free`한다.

### 7.2 상주 (`aimx.persistent`)

`aimx_resident_acquire(plan, fpga, buffers)`는 FPGA마다 plan 하나를 캐시하고 lease(mutex)를
잡는다. plan key는 `AIMX_KERNEL_KEY`(stub·descriptor·include·`aimx_runtime.h`·base_fpga·resident
runtime 해시) + site다. key나 buffer 수가 바뀌면 buffer와 chain을 함께 버리고 다시 잡는다.
같은 kernel의 다음 호출은 할당 없이 같은 주소를 쓴다. `aimx_resident_alloc`은 물리 주소가
33 bit 안이고 할당이 물리적으로 선형인지 검사한다.

### 7.3 chain 재사용 (`DescriptorReuse.h`)

construction 경계마다 descriptor 인자의 불변성을 분류한다.

| 분류 | 조건 | 동작 |
| --- | --- | --- |
| `Unchanged` | 모든 주소·loop 범위가 상주 buffer·device 좌표·상수에만 의존 | `_resident->ready`이면 build·업로드 생략. 업로드 뒤 host record 페이지 반납(`MADV_DONTNEED`) |
| `Addresses` | loop 구조는 같고 주소만 바뀜 | `ready`이면 `aimx_patch_*`로 src/dst만 고쳐 다시 업로드 |
| `Rebuild` | 그 밖 | 매번 build·업로드 |

동적 상한 `min(tileGroups, remaining)`은 full tile일 때만 `ready`를 세우고, tail은
같은 저장소를 덮어쓰므로 `ready`를 내린다.

### 7.4 stage / unstage

| 경로 | 조건 | 동작 |
| --- | --- | --- |
| `group_layout` | 모든 placement (현재 lowering) | panel마다 유효 행·열 복사, 행 끝과 panel 끝만 memset |
| `group_layout` + `transpose` | GEMV의 W, x, y | panel `(k, m)` ← host `(m, k)`. host 행 stride가 1이면(host가 이미 물리 순서) 물리 행마다 연속 복사, 아니면 32×32 block으로 전치한다. host 32행을 바깥에 두고 K-Group·K를 안쪽에서 끝까지 돌며, panel에는 64 B씩 쓴다 |
| `shared_over` | broadcast 입력 | 공유 좌표 조합마다 host pool 한 칸. 처음 온 thread만 변환(`std::call_once`), 전송은 FPGA마다 |
| unstage `accumulate` | K Tile 2번째부터, `p_k`의 k ≠ 0 | bf16 → host 타입 덧셈 |

host 변환: f32 → bf16은 round-to-nearest-even, f16 → bf16은 65,536칸 표, bf16은 그대로
복사. 비상주 경로의 조립 buffer는 thread별 cache(`aimx_host_buffer`, 8칸)로 재사용한다.
stub은 `-O3`로 빌드한다.

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

## 8. 모델 session (`--aimx-to-model-stub`, `aimx_model.py`)

`PIMSession` 경로는 같은 target IR을 model stub으로 다시 번역한다. 공통 정책은
[Model Runtime 설계](ModelRuntime_Design.md)에 있다. AiMX 쪽만 적는다.

### 8.1 stub

| 항목 | 내용 |
| --- | --- |
| 인자 | `aimx_model_call *` + kernel 인자 + grid 3개. context와 FPGA 번호는 session이 넘긴다(`AIMX_BASE_FPGA`, `AIMX_KERNEL_KEY` 없음) |
| 실행 단위 | job = batch slot × device. job마다 FPGA 하나, context 하나. instance를 `batch_size`개씩 돈다 |
| 호출 종류 (`upload`) | 0 실행 (activation·상수 stage, invoke, unstage), 1 weight를 device에 stage, 2 pack (weight를 job의 host image에만 조립, device 접근 없음), 3 chain build·업로드만. Python은 0, 2, 3을 쓴다 |
| 표 | `pim_model_slot_count/argument_count/device_count`, slot마다 `pim_model_slot_bytes/input/output/kind/layout`(layout = `group_layout` 6칸 + transpose) |

### 8.2 주소와 context (`AiMXModelRuntime.place` / `bind`)

| 항목 | 내용 |
| --- | --- |
| phase | 하나만 배치한다. phase가 둘 이상이면 `place`가 `NotImplementedError` |
| memory table | slot들의 weight → slot들의 state(chain, PMM) → activation scratch. key/value cache는 그 위 `[kv_base, capacity)` |
| region | table을 조각 경계에서 잘라 region 하나 ≤ 1023 × 2 MiB(`MAX_REGION_BYTES`). region = FPGA마다 `pim_malloc` 하나 + 같은 크기의 host shadow. cache 항목은 별도 region에 묶는다 |
| 정렬 | operand 크기를 256 KiB(`OPERAND_ALIGNMENT`)로 올려 모든 offset을 정렬한다 |
| context | `aimx_model_create`가 slot마다 buffer를 table 주소(va), 용량, host shadow 위치에 놓는다 |
| slot 공유 | slot을 공유하는 layer들은 그 slot 첫 layer의 context·descriptor chain을 함께 쓴다. `_create_stage`는 `slot_bytes`, table 종류, offset, job 수가 같아야 한다 |
| chain | 물리 주소를 담으므로 slot마다 하나. `bind`에서 slot owner stage마다 `upload=3`으로 한 번 만들어 올린다(`chain_build_ms`) |

resident weight GEMM은 instance 1개, scheduling round 1회로 제한한다(`select_layer_config`).
여러 round가 같은 buffer에 다른 weight tile을 올리면 상주가 성립하지 않기 때문이다.

### 8.3 weight 업로드 (`GroupLoader`)

- group에 들어갈 때(`enter_layer`) 그 group의 weight를 올린다. pack 호출(`upload=2`)이 group의
  weight 범위를 FPGA마다 host image 하나에 조립하고, `aimx_model_push`가 (region, FPGA)마다 한 번
  전송한다. 다음 group의 pack은 background thread에서 실행과 겹친다(pack은 host만 쓴다).
- group이 하나뿐이면(모든 layer 상주) 실행 전에 region 하나씩 pack → push → 해제한다. host는
  region 하나의 image만 쥔다.

### 8.4 key/value cache 상주

host cache는 layer마다 `[kv_heads, positions, head_dim]`이다. cache stage의 B는 token-row view
`[positions, kv_heads × head_dim]`의 열 window(연속된 KV head를 열로 이어 붙인 것, `stage.kv_windows`,
`residency.token_rows`)이거나 그 전치(QK의 B = Kᵀ)다.

| 항목 | 내용 |
| --- | --- |
| 실행 | cache stage는 `devices = 1`, batch round 1회여야 한다(`_create_stage`). instance i가 FPGA `base + i`를 혼자 쓰므로 window 주소가 하나로 고정되고 chain도 바뀌지 않는다 |
| 주소 | cache operand slot은 memory table의 `("kv", layer, KEY/VALUE)` 항목이다. capacity에서 아래로 layer마다 key, value 순서로 놓인다(`plan_memory`). 실행(upload 0) 호출에서 stage하지 않는다 |
| 배치 | B 행 r, 열 c는 panel `(r // ur, c // uc)`(row-major), panel 안 `(r % ur, c % uc)`(`_kv_offsets`, `pim_model_slot_layout`) |
| 쓰기 | `write_kv`는 B operand이고 stage layout이 비전치인 cache만 받는다. 위치를 대기열에 넣고, 다음 cache stage 실행 직전(`_push_kv`)에 region host shadow에 bf16으로 조립한 뒤 (항목, FPGA)마다 바뀐 byte 구간 `[최소 offset, 최대 offset]` 하나를 `aimx_model_push_ranges`로 보낸다 |
| 대조 | stage의 첫 push에서 cache 전체를 stub pack(upload 2)으로 조립한 결과와 계산한 배치를 비교한다(`_check_kv_layout`) |
| 비용 | V(B 행 = 위치)는 새 위치의 행 구간만 보낸다. Kᵀ(B 열 = 위치)는 새 위치가 모든 행에 걸쳐 있어 구간이 그 panel의 거의 전체가 된다 |

cache 항목의 offset은 layer마다 다르다. slot을 공유하는 layer들은 `_create_stage`의 offset
일치 조건을 통과하지 못하므로, cache를 읽는 AiMX decode는 layer마다 slot이 따로인
**group 하나**(모든 layer 상주)에서만 실행된다.

---

## 9. autotuner 연동 (`autotuner/model.py`의 `AiMXModel`)

| 항목 | GEMM (`matmul`) | GEMV (`matvec`) |
| --- | --- | --- |
| Block | `(min(8, b), 16, 16)` | `(8, 1, 16)` (`TRITON_BLOCK`) |
| Group 상한 (`_compute_unit`) | `(b, 4096, max(256, kAlign))` | `(4096, 1, 256)` |
| profile spec | Group `unit / Block`, f16 → f32 | Group `(512, 1, 16)`, f16 → f32 |
| `scheduled_compute` | `tile_m·tile_n · roundup(Gm,b) · roundup(Gn,4096/b) · roundup(tile_k·macK, 256)` | `tile_m · roundup(Gm,4096) · tile_k · roundup(Gk,256)` |
| operand 대응 | A(m,k), B(k,n), C(m,n) | W(m,k) → B, x(k) → A, y(m) → C |

- `aimx_runtime_allocations`는 lowering과 같은 식으로 PMM, A, Arel, B, C, descriptor chain
  하한을 계산한다(GEMV는 §5.2의 전치 크기). `allocation_reason`은 단일 할당 ≤ 1023 × 2 MiB,
  bank 사용량 ≤ `bank_capacity_limit_bytes`를 검사한다. 최종 판정은 lowering의 용량 검사다.
- `mapping_reason`을 두지 않으므로 `p_k > 1`도 후보다(§5.8).
- cache stage: `resident_operand_reason`은 상주 B가 `roundup(K,256) × roundup(N,4096)`보다 크게
  저장되는 config를 거부하고, `resident_operand_config`는 그 조건을 만족하는 config 하나
  (Group m = batch, n = min(4096, N), k = 256, Tile이 문제 전체)를 탐색 초기값으로 낸다.

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
| 5 이상 | 8 | 8 | 8 × 512 |

torch 경로에서 M = 1 또는 N = 1인 GEMM은 matvec 문제로 바뀐다(`execution.gemm_problem`).
최종 config는 실측 `total_ms` 최소값이다.

| 진입점 | 용도 |
| --- | --- |
| `examples/silent_pim/autotune_gemm.py` | GEMM 탐색, `--profile-only`로 단일 FPGA profiling |
| `examples/silent_pim/autotune_gemv.py` | GEMV 탐색(rel L2 ≤ 2% 판정), `--profile-only` |
| `torch_backend` (`model_compile.tune_stage_class`) | 모델 stage |

GEMV 단일 FPGA profiling의 정확도 검사(`autotuner/gemv.py`)는 `x[::256] = 1`, 기대값
`ceil(K/256)`이다. ones·ones는 부분합이 1024를 넘어 vACC가 작은 곱을 버린다(§11.3).

---

## 10. 실행 조건

### 10.1 환경

| 조건 | 값 / 확인 |
| --- | --- |
| driver | `num_units`를 주지 않으면 `/dev/pl_dma*` 개수를 쓴다 (`aimx_unit_count`) |
| vendor 소스 | `export AIMX_PIM_SW=$HOME/pim_aimx_sw/pim_sw` (`pim_math.h`를 소스 트리에서 읽는다) |
| 라이브러리 | `$AIMX_PIM_SW/lib/libpim.so`를 경로로 직접 링크한다. `-lm` 필요 (`pow` 미해결) |
| 빌드 | `g++ -O3 -std=c++17 -fopenmp -Wno-unused-function -Wno-narrowing -Wno-write-strings` (driver가 수행) |
| FPGA 범위 | `[base_fpga, base_fpga + num_units)`가 보이는 FPGA 수 안이어야 한다 (`aimx_runtime_init`, `AiMXAdapter.open`) |
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
| 연산: matmul, matvec, identity-map elementwise add/sub/mul/extf, 상수 fill, `pim.combine` add | 그 밖은 오류 |
| K 누산은 positive-zero accumulator에서 시작 | vACC 경로 불가 → 오류 |
| mask는 §5.7 범위 | 오류 |
| 할당 하나 ≤ 1023 × 2 MiB, bank 합 ≤ 용량 | 오류 |

### 10.4 성능 조건 (padding)

| 조건 | 이유 |
| --- | --- |
| GEMM `Gn`을 4096의 배수로 | B/C panel 행이 4096 원소다. `Gn = 512`면 B/C가 8배 |
| GEMM/GEMV `Gk`를 256의 배수로 | B 행이 256 단위로 채워진다 |
| GEMV `Gm`을 4096으로 | output block이 M 4096이다. `Gm = 32`면 lane 32/4096만 쓴다 |
| N(GEMM) 또는 M(GEMV)이 4096 미만인 행렬 | 4096까지 padding은 피할 수 없다 |

autotuner의 Group 상한(§9)이 이 크기를 1순위로 고른다.

---

## 11. FPGA로 확인된 사실

### 11.1 descriptor와 주소

| 사실 | 근거 |
| --- | --- |
| stride shift는 33 | `matmul.c`의 stride 자리가 `<< 33`. `kStrideShift`, resident runtime이 PA < 2³³을 검사 |
| `len`은 출력 byte 수. 입력 크기를 주면 그만큼만 쓰이고 나머지는 이전 내용이 남는다 | `AiMXDialect.td`의 `aimx.desc` 설명 |
| `pim_private_malloc`이 `pim_malloc`이 내준 주소를 돌려줄 수 있다 | PMM 우선 할당과 겹침 검사 (`AiMXToRuntimeStub.cpp`) |

### 11.2 행 pitch와 정렬

| 대상 | 결과 | 반영 |
| --- | --- | --- |
| GEMM B/C Group panel 행 pitch 512 (b=8) | 실행마다 결과가 다름 | `aimx_matmul_row_alignment = 4096` |
| GEMM A 행 pitch `4096/b` (b=8: 512) | 4096과 같은 결과 | `aimx_matmul_k_alignment = 0` (= `4096/b`). 검증 스크립트 `torch_backend/examples/aimx/verify_k_alignment.py` |
| 모델 경로 data buffer 정렬 8 KiB (b=8) | 결과가 틀림 | `OPERAND_ALIGNMENT = 256 KiB` (PMM 정렬과 같다) |

### 11.3 vACC 수치 특성

vACC lane = 부호 1 + 그룹 지수 G 5 bit + 가수 16 bit (22 bit), 값 `±가수 × 2^(8G−134)`.

| G | \|acc\| 범위 | LSB | 버려지는 곱 |
| --- | --- | --- | --- |
| 15 | < 4 | 2⁻¹⁴ | \|p\| < 2⁻¹⁵ |
| 16 | 4 ~ 1024 | 1/64 | \|p\| < 2⁻⁷ |
| 17 | 1024 ~ 262144 | 4 | \|p\| < 2 |

- G 차이 −1인 곱은 8 bit 밀어 더하고, −2 이하는 버린다. G는 커지기만 한다. 그래서 같은
  부호 부분합이 1024를 넘으면 작은 곱이 버려진다.
- 입력 0은 누산을 바꾸지 않는다. 0 padding은 결과에 영향이 없다.

정확도 (M=128, N=512, K=4096, FP64 대비 relative L2):

| 분포 | k_t=256 | 1024 | 4096 |
| --- | --- | --- | --- |
| signed N(0,1)·N(0,1) | 0.80% | 0.78% | 0.78% |
| positive \|N\|·\|N\| | 0.61% | 0.63% | **48.5%** (1024 포화) |
| llm N(0,1)·N(0,0.02) | 0.40% | 0.41% | 0.79% |

compiler는 FP32 누산이나 CPU와의 bit 일치를 약속하지 않는다.

---

## 12. 미구현과 알려진 제약

| 항목 | 상태 |
| --- | --- |
| GEMM batch 혼합 (예: M=12 → 8 + 4) | 미구현. 한 MAC에 batch 하나, M은 `roundup(Gm, b)`로 padding |
| B/C 행 pitch < 4096 | 하드웨어 미확인. `matmul-row-alignment=0`(= `4096/b`)은 비교용으로만 남아 있다 |
| Group마다 다른 mask, mask가 걸린 read/write alias | 미구현 (lowering 오류) |
| GEMV 커널의 rank-1 masked x/y | 예제 커널은 full-block만 허용한다 (`examples/silent_pim/gemv.py`) |
| A 재배치 융합 / host에서 Arel 직접 생성 | 미구현. 채널마다 다른 vecA는 실측 확인, 설계 보류 ([AiMX_ChannelUnits.md](AiMX_ChannelUnits.md)) |
| 모델 session의 phase 둘 이상 | 미구현 (`AiMXModelRuntime.place`). 예제는 prefill과 decode phase를 따로 compile한다 (`torch_backend/examples/aimx/run_aimx_{prefill,decode}.sh`) |
| cache stage의 slot 공유 | 불가 (§8.4). cache를 읽는 decode는 group 하나 |
| stage와 invoke 겹치기 | 없음. `pim_exec`가 동기이고 FPGA마다 host thread 하나. 모델 경로는 다음 group의 host pack만 실행과 겹친다 |
