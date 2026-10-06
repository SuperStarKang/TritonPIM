# AiMX (silent-PIM) Target 설계

Placed PIM Common IR을 AiMX FPGA의 CDMA descriptor stream으로 옮기는 계층의 설계다.
`UPMEMTarget_Implementation.md`가 UPMEM에 대해 하는 일을 AiMX에 대해 한다.

**이 문서는 하드웨어 구조와 명령어(opcode) 구조에서 출발한다.**
`pim_math_src/matmul.c`의 `_matmul_silent`는 그 명령어 집합으로 쓰인 하나의 프로그램일
뿐이고, 이 문서는 그 함수의 API나 loop 순서를 제약으로 받아들이지 않는다.
`_matmul_silent`는 (a) 명령어 의미를 확인하는 참조 구현이자 (b) 우리가 더 나은 명령열을
낼 수 있는지 비교하는 기준으로만 쓴다. 실제로 §8에 네 가지 개선점이 있다.

근거 파일. 모두 `/home/yonsei_park/kangmlee/pim_aimx_sw` 기준이고, 이 문서의 모든
행 번호도 그 트리 기준이다.

- `pim_sw/src/pim_math_src/pim_math.h` — 기계 상수와 opcode
- `pim_sw/src/pim_math_src/matmul.c` — 의미 확인용 참조
- `pim_sw/include/pim.h` — `PMM_SIZE`, `VACC_OFFSET`, `HIGH_ADDR`
- `ubench/matmul/matmul.cpp` — host 자료 배치 계약 (§1.8)
- `ubench/matmul/pim_mac_simul.cpp` — 수치 형식 (§11.1)

> **소스 트리는 하나다.** `/home/yonsei_park/pim_aimx_sw/pim_sw` 이고, 이 문서의 모든
> 인용은 여기서 온다. 그 트리의 `matmul.c` 는 stride shift 가 **`<< 33`** 이다
> (일곱 자리 전부 — §1.1, §15 E2).
>
> 링크 주의: `libpim.so` 는 `pow` 를 미해결 심볼로 남기므로 링크에 `-lm` 이 필요하다.

현재 구현된 PIM Common IR은 **읽기만 하고 고치지 않는다**.

## 현재 lowering의 Group 배치 계약

아래의 과거 설계에서 whole-tile row-major 배치와 `#aimx.tiling`을 설명하는 부분은
현재 구현과 구분한다. 현재 lowering은 Common의 SSA 계산, Group 반복, K 누적,
mask와 host 전송 순서를 보존하며 AiMX의 물리 배치만 결정한다.
이번 변경은 빌드·실기기 실행 전 상태다.

논리 Group을 `(Gm, Gn, Gk)`, hardware batch를 `b`라 하면, 독립적인 GEMM의
물리 Group 크기는 다음과 같다. 여러 연산 사이에서 tensor가 공유되면 해당 SSA
연결에 필요한 정렬 조건을 함께 만족시킨다. 같은 tensor type이라는 이유만으로
서로 다른 A/B/C의 배치를 합치지는 않는다.

```text
Mp = ceil(Gm / b) * b
Np = ceil(Gn / (4096/b)) * (4096/b)
Kp = ceil(Gk / 256) * 256
Ap = ceil(Gk / 4096) * 4096

A Group: Mp × Ap       (A gather용 4096 정렬 유지)
B Group: Kp × Np       (N pitch의 4096 고정 제거)
C Group: Mp × Np
```

Execution Tile은 각 Group을 독립적인 연속 panel로 저장한다. `tm/tn/tk`는
Common이 정한 Group 개수이며 AiMX가 다시 정하지 않는다.

```text
A: [A(m0,k0)][A(m0,k1)] ... [A(m1,k0)][A(m1,k1)] ...
B: [B(k0,n0)][B(k0,n1)] ... [B(k1,n0)][B(k1,n1)] ...
C: [C(m0,n0)][C(m0,n1)] ... [C(m1,n0)][C(m1,n1)] ...

panel byte offset = (row_group * column_groups + column_group)
                    * physical_rows * physical_cols * 2
```

`aimx.stage/unstage`의 `group_layout`은
`[useful_rows, useful_cols, physical_rows, physical_cols, row_groups, col_groups]`다.
stage는 경계 밖 입력을 0으로 채우고, unstage는 유효한 행/열만 host에 쓴다.
접근 시작점이 panel 경계와 맞지 않거나 상주 범위를 벗어나면 `aimx.require`가
descriptor 실행 전에 오류를 보고한다. Copy와 elementwise는 panel을 4096개 원소씩
순회하므로 B/C의 개별 행을 4096개 원소로 늘릴 필요가 없다.

Group 반복은 기존 `aimx.descriptor_program`을 호출해 **전체 Tile chain에 펼쳐진다**.
`pim_exec`는 Tile마다 한 번이며, 불변인 전체 chain의 기존 재사용을 유지한다.
하나의 Group descriptor 본문을 device에서 반복 실행하는 기능은 구현하지 않았다.

### Tile 내부 K-Group의 vACC 누적

`pim.temporal_reduction`의 positive-zero 초기화와 직접 연결된
`linalg.matmul` accumulator는 AiMX에서 다음 순서로 실행한다.
Common의 Block/Group 크기, Tile 경계, 각 출력에 대한 K 순서는 바꾸지 않는다.

```text
각 M/N Group:
  모든 K-Group의 A → Arel 재배치 (Group별 별도 슬롯)
  각 batch × (4096/batch) 출력 영역:
    모든 K-Group의 A/B MAC (중간 C drain/ADD 없음)
    필요하면 마지막 K-Group 뒤에 zero MAC padding
    vACC → BF16 결과 저장 한 번
```

`Arel`은 이 reduction site에서 `tile_k * Mp * Kp * 2 * 16` bytes를
확보한다. 다음 M/N Group은 이전 출력이 모두 저장된 뒤 이 공간을 재사용한다.
vACC가 전체 Tile 출력을 동시에 담는 것은 아니다. 비어 있는 K 반복은
zero 결과를 유지하고 vACC drain을 생성하지 않는다.

Arel/B의 저장 공간은 여전히 K를 256으로 정렬한다. MAC은 Group마다
`ceil(Gk/(256/batch))`개의 A/B descriptor 쌍을 사용한다. Group K가 이 MAC
단위의 배수이면 Group 사이의 저장용 padding은 실행하지 않는다. Tile의
MAC 길이가 256의 배수가 아니면 마지막 Group의 zero 영역을 반복해서 읽어
끝에서만 맞춘다. Group K가 MAC 단위보다 작거나 정렬되지 않은 경우에는
Group 내부의 MAC padding이 남으므로, compact native matmul과 비트 동등성을
주장하지 않는다. 부분 마지막 Group의 논리 범위를 넘는 입력은 staging의
zero padding을 사용한다.

중간 accumulator 사용, nonzero 초기값, K 반복 안의 추가 계산/부작용은
진단을 내고 중단한다. 서로 다른 M/N 출력 영역에 단독으로 기여한다는 것을
증명한 `pim.combine`은 ADD 대신 copy로 반영한다. 실제로 여러 기여가 같은
출력에 모이면 ADD를 유지한다. K가 여러 Execution Tile로 나뉘면 Tile별 BF16
결과의 기존 host FP32 합산을 유지하며, `pim_exec` 사이에 vACC를 보존하지 않는다.

FPGA 검증은 `examples/compare_aimx_native_gemm.py`와
`examples/aimx_native_grouped_gemm.cpp`에서 수행한다. 원본 libpim으로 Tile의
전체 K를 한 번의 matmul에 넣고 동일한 Tile 간 host 합산을 적용한다.
기존 `aimx_reference.cpp`/`compare_aimx_gemm.py`의 Group별 ADD 모델은 이 경로의
검증 기준이 아니다. 이번 변경은 소스 검토만 했으며 빌드/실행 결과는 아직 없다.

Lowering은 PMM, 상주 버퍼, scratch, 상수, 전체 descriptor chain을 모두 합산해
bank 용량을 검사한다. `aimx.allocations`, `aimx.bank_bytes`,
`aimx.descriptor_bytes`에 컴파일 시점의 할당 계획을 기록한다.
Autotuner의 `constraints.py`는 컴파일 전 GEMM 하한 추정이며, 이 검사 통과가 전체
IR의 용량 적합성을 보장하지 않는다. 최종 판단은 lowering이 하고, 측정 결과의
`physical_allocations`는 해당 컴파일 결과의 할당 계획을 담는다.

---

## 0. 용어

하나의 개념에 하나의 용어만 쓴다. 행렬 축은 언제나 **m**(출력 행), **k**(reduction),
**n**(출력 열)이다. `matmul.c`의 `p_size`/`q_size`/`r_size`는 각각 **m / k / padded_n**
이다. 특히 `r_size`에는 n이 아니라 `padded_n = ceil(n/CHUNK) * CHUNK`를 넣는다
(`matmul.cpp`: `set_info->r_size = padded_r_size`). B와 C의 행 pitch가 이 값이기
때문이다 (§1.8). 아래 어디에서도 p/q/r을 쓰지 않는다.

Compiler option과 target IR의 단위는 다음과 같이 구분한다.

- `tile_m/n/k`: execution tile 하나에 묶는 **block-group 개수**
- `BLOCK * GROUP * TILE`: 저장하지 않고 파생하는 **유효 논리 원소 크기**
- `t_m/t_n/t_k`: AiMX padding을 반영한 `#aimx.tiling`의 **물리 tile 크기**

`#pim.execution_tile`은 Block Group 개수만 기록한다. AiMX target은
`#pim.block`, `#pim.block_group`, `#pim.execution_tile`의 곱에서 논리 원소 크기를
읽고, 그 결정을 바꾸지 않은 채 물리 padding만 `t_*`에 반영한다.

| 용어 | 뜻 | 크기 |
| --- | --- | --- |
| **lane** | multiply-accumulate 유닛 하나. `ALU_SIZE`의 단위 | 출력 원소 1개 |
| **bank** | lane 16개 | 출력 원소 16개 |
| **channel** | bank 16개 | 출력 원소 256개 |
| **PIM Device** | channel 16개 = FPGA 하나 (`fpga_id`) | 출력 원소 4096개 |
| **descriptor record** | `pim_isa_32_t` 하나 | 32 byte |
| **descriptor group** | 16 channel queue에 같은 내용으로 들어가는 record 16개 | 512 byte |
| **descriptor program** | descriptor group을 내는 loop nest. compile time 산물 | op 수십 개 |
| **descriptor chain** | descriptor program을 펼친 record 열. runtime 산물 | 수 MB |
| **lane tile** | lane 하나가 맡는 출력 | `1 × 1` |
| **bank tile** | bank 하나가 맡는 출력 | `1 × 16` |
| **channel tile** | channel 하나가 맡는 출력 | `1 × 256` |
| **output block** | PIM Device의 lane 4096개가 한 번에 맡는 출력 | `batch × (4096/batch)` |
| **reduction step** | vecA 한 번 적재가 지탱하는 reduction 구간 | `256/batch` |
| **execution tile** | `pim_exec` 한 번이 덮는 범위. PIM Device 하나 안의 **시간적 상주 단위**다 (§3.4) | `(t_m, t_n, t_k)` |
| **logical output tile** | Triton program instance 하나가 덮는 **출력** 직사각형 | `BLOCK_M × BLOCK_N` |
| **internal reduction step** | 그 program instance 안의 `k` loop 한 걸음 | `BLOCK_K` |
| **batch** | output block의 행 수. `{1, 2, 4, 8}` | `N_BATCH_SIZE_TYPES = 4`가 정한 상한 |
| **padded_k** | A의 행 pitch. `ceil(k/CHUNK) * CHUNK` | element |
| **padded_n** | B와 C의 행 pitch. `r_size`에 들어가는 값 | element |
| **Arel** | 재배치된 A. §1.6의 순열을 적용한 8192 byte 블록의 나열 | `m * k * 16` element |

> **`BLOCK_K`는 공간 축이 아니다.** `gemm.py`의 grid는
> `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`로 **2차원**이고, `BLOCK_K`는 program instance
> **안**의 `for k in range(cdiv(K, BLOCK_K))` 한 걸음이다. 따라서 logical output tile은
> `BLOCK_M × BLOCK_N`이고 `k`는 grid 축이 아니다. `k`가 분할 축이 되는 것은 커널이
> `split_k.py`처럼 **explicit split-K**를 쓸 때뿐이고, 그때도 그 축은 reduction이 아니라
> 분리다 (§17.4).
| **saturation point** | throughput curve가 평평해지는 execution tile 크기 | 측정으로 정한다 |

`batch`는 machine learning의 batch와 무관하다. lane 4096개를
`batch × (4096/batch)` 직사각형으로 다시 자르는 하드웨어 parameter다.

---

## 1. 하드웨어 구조

### 1.1 기계 상수

`pim_math.h`에서 그대로 온다.

| 상수 | 값 | 정의 |
| --- | --- | --- |
| `NUM_CHANNELS` | 16 | descriptor queue 개수 |
| `NUM_BANKS` | 16 | channel당 bank |
| `ALU_SIZE` = `NUM_ALUS` | 16 | bank당 lane |
| `TYPE_SIZE` | 2 | `sizeof(short)` — bfloat16 |
| `BURST_SIZE` | 32 byte | DRAM burst |
| `VECA_SIZE` | 16 | vecA entry 수 |
| `REG_SIZE` | **256** element | `(BURST_SIZE/TYPE_SIZE) * VECA_SIZE` |
| `PIM_WIDTH` | **512** byte | `ALU_SIZE * NUM_BANKS * TYPE_SIZE` |
| `CHUNK` | **4096** element | `(BURST_SIZE/TYPE_SIZE) * ALU_SIZE * NUM_CHANNELS` |
| `STRIDE_SHIFT` | **33** | pitch를 주소에 실을 때 미는 비트 수 |
| 물리 주소 폭 | **33** bit | `HIGH_ADDR = 0x00000000` (`pim.h`), 관측 PA ≤ `0x1ffe00000` |
| `PMM_SIZE` | `0x30000` | `pim_private_malloc` 영역 크기 (`pim.h`) |
| `VACC_OFFSET` | `0x10000` | vACC = `pmm_pa + VACC_OFFSET` (`pim.h`) |

> **`OPCODE_STRIDE`가 켜진 descriptor는 pitch를 `<< 33` 해서 주소에 OR 한다.**
> `src | (pitch_bytes << 33)` 형태이고 `matmul.c`의 일곱 자리(143, 163, 170, 312, 325,
> 337, 344행)가 전부 그렇다. 물리 주소가 33 bit이므로 33~39는 비어 있고 pitch는 40에서
> 시작한다. `memcpy.c`의 `<< 48`은 `pim_copy` 경로의 다른 관례이므로 한 chain 안에서
> 섞지 않는다.
>
> **정정 이력.** 이 표는 한때 `33`이라고 적었고 근거로
> `/home/yonsei_park/kangmlee/pim_aimx_sw/.../matmul.c`를 들었다. **그 경로는 존재하지
> 않는다.** 디스크의 `matmul.c`는 `/home/yonsei_park/pim_aimx_sw/pim_sw/src/pim_math_src/`
> 하나뿐이고 **전부 `<< 33`**이다. `PIM_RD_INSTR_MC`(`pim_math.h:250`)는 `src`를
> 32 bit 두 조각으로 그대로 저장할 뿐 추가로 밀지 않으므로 33이 최종 shift다.
> `AiMXDialect.h`의 `kStrideShift`는 33으로 맞게 들어가 있다.
>
> **정정 (2026-08-11).** 이 문단은 한때 40이라고 적었고 근거로 '`<< 33`인 소스는
> 트리 어디에도 없다'고 했다. 그것이 사실이 아니었다 — grep 대상 경로를 잘못 잡은
> 것이다. `~/kangmlee/test/common/aimx.h`의 카나리만 40으로 바뀌어 있었고, 그래서
> §15 E0가 M 경로 4096/4096 전부 불일치를 냈다. compiler는 영향 없다.

`TYPE_SIZE = 2`가 bfloat16을 뜻하지만 **누산기는 fp32가 아니다.** 실제 형식과 그 결과인
동적 범위 제약은 §11.1에 있다.

물리 channel은 8개다. `NUM_CHANNELS = 16`과 맞추면 "channel 8개 × pseudo-channel 2개
= descriptor queue 16개"가 가장 일관된 해석이고 lane 수는 `8 × 2 × 16 × 16 = 4096`으로
같다 (§15 E1). 이 문서의 어떤 식도 8이냐 16이냐가 아니라 `NUM_CHANNELS`라는 기호에만
의존한다.

### 1.2 계층과 크기 — 상수들이 서로 맞물린다

```
PIM Device  =  FPGA 1개 (fpga_id)                          출력 4096칸 / step
│
├── channel 0 ──┬── bank 0 ──┬── lane 0  ── vACC[0]        출력 1칸
│               │            ├── lane 1  ── vACC[1]
│               │            │      :
│               │            └── lane 15 ── vACC[15]       bank = 16칸 = 32 byte
│               ├── bank 1        (같은 구조)
│               │      :
│               └── bank 15                                channel = 256칸 = 512 byte
├── channel 1
│      :
└── channel 15                                             device = 4096칸 = 8192 byte
```

이 계층이 상수와 정확히 맞물린다는 것이 이 해석의 근거다.

| 계층 | lane 수 | step 하나에 소비하는 B | 대응 상수 |
| --- | --- | --- | --- |
| lane | 1 | 2 byte | `TYPE_SIZE` |
| bank | 16 | 32 byte | **`BURST_SIZE`** = `ALU_SIZE * TYPE_SIZE` ✔ |
| channel | 256 | 512 byte | **`PIM_WIDTH`** = `ALU_SIZE * NUM_BANKS * TYPE_SIZE` ✔ |
| PIM Device | 4096 | 8192 byte | **`C_len`** = `NUM_BANKS * NUM_CHANNELS * ALU_SIZE * TYPE_SIZE` ✔ |

세 줄이 전부 독립적으로 정의된 상수인데 계층 크기와 맞아떨어진다.
특히 **`BURST_SIZE / TYPE_SIZE = 16 = ALU_SIZE`** 는 우연이 아니다 —
**DRAM burst 하나가 bank 하나의 lane 16개를 정확히 채운다.** 이것이 이 하드웨어
설계의 기본 눈금이다.

`REG_SIZE = 256`은 channel 하나의 vecA 크기이고 `NUM_BANKS(16) × VECA_SIZE(16)`으로
쪼개진다 (§1.6). vACC가 lane당 정확히 한 칸이라는 근거는 `REG_SIZE`가 아니라 C drain의
`C_len = 8192 byte = 4096 element = lane 수`다.

### 1.3 descriptor는 명령이 아니라 주소를 나른다

`pim_isa_32_t`는 Xilinx CDMA scatter-gather descriptor 그대로다 — 필드 주석이
`CDMA_NEXT_DE_L/H`, `CDMA_DESC_SA_L/H`, `CDMA_DESC_DA_L/H`, `CDMA_DESC_LEN`,
`CDMA_DESC_STATUS`이고 크기가 정확히 `BURST_SIZE = 32` byte다. 이 문서는 그 32 byte
하나를 **record**, record 16개 = 512 byte를 **descriptor group**이라고 부른다.

`mPIM_RD_INSTR_MC` (`pim_math.h` 250–271행)는 인자 하나를 받아 record 16개를 **완전히
동일한 내용**(`src`·`dst`·`len`·`status`·`next` 전부)으로 쓰고 `next`를
`0x20 * NUM_CHANNELS` = 512 byte 전진시킨다.

**16개 중 실제로 실행되는 것은 첫 번째 하나뿐이다 — 실측으로 확정됐다 (§15 E11).**

`pim_isa_32_t`는 CDMA scatter-gather descriptor이고 chain은 `next` 포인터로 이어진 연결
리스트다. driver는 head(`desc_base_pa`)와 tail(`desc_last`)만 주고, 엔진은 head에서
시작해 `next`를 따라가다 tail을 처리하면 멈춘다 (`dma_fops_pl.c`의 `MEMCPY_SG`/`DMA_START`).

그런데 `mPIM_RD_INSTR_MC`가 쓰는 16개 record의 `next`가 **전부 같고** 그 값이
`base + 512*(g+1)`이다. 그래서 엔진이 걷는 자리는

```
descriptor group (512 byte)
  base+0x000  record  ← **실행되는 것은 이것 하나뿐이다**
  base+0x020  record  ┐
      :               ├ next 사슬이 건너뛴다. 실행되지 않는다
  base+0x1e0  record  ┘
  base+0x200  다음 group 의 첫 record  ← 모든 record 의 next 가 여기를 가리킨다
```

**E11 실측.** group 하나의 16칸에 서로 다른 `src`(`S_pa + ch*8192`)를 넣고 `pim_copy`로
돌린 결과가, 16칸을 모두 같게 쓴 경우와 **byte 단위로 동일**했다. 옮겨진 8192 byte는
정확히 record 0이 지정한 구간(`S_pa`부터)이었다. 즉 record 1~15는 아무 효과가 없다.

> **descriptor group = 실행되는 descriptor 1개 + 쓰이지 않는 record 15개.**
> 512 byte라는 눈금은 "channel마다 한 칸"이 아니라 chain 주소 산술의 단위일 뿐이다.

| | |
| --- | --- |
| **channel마다 다른 *명령*을 줄 수 있는가** | **아니다.** record 16개 중 첫 번째만 실행된다 (§15 E11) |
| **channel마다 다른 *데이터*를 가게 할 수 있는가** | **그렇다. 이것이 유일한 분배 수단이다** |

> **정정 이력.** 이 절은 한때 "descriptor group은 channel마다 하나씩 놓인 명령 슬롯
> 16칸이고, 소프트웨어가 쓰지 않을 뿐이다"라고 적었다. `ROBACOCH`가 512 byte 구간을
> 16 channel에 32 byte씩 흩는다는 것에서 **descriptor에도 같은 일이 일어날 것이라고
> 유추한 것**인데, `next` 사슬이 그 앞에서 record를 건너뛰므로 성립하지 않는다.
> E11이 그것을 직접 반증했다.

descriptor는 `(src, src_pitch) → (dst, dst_pitch), len, flags`다. 단 `src_pitch`와
`dst_pitch`는 독립된 필드가 아니다 — `pim_isa_32_t`는 `next_l/h`, `src_l/h`, `dst_l/h`,
`len`, `status` 여덟 word뿐이고, pitch는 `src`·`dst`의 상위 비트에 `<< 33`으로 OR 된다
(§1.1). 즉 **주소 범위**를
지정하고, 그 주소 범위를 (channel, bank, lane)에 흩뿌리는 것은 하드웨어의 고정된
address decoding이다. compiler가 하는 일은

> **주소 범위를 하드웨어의 고정된 사상에 정확히 들어맞게 자르는 것**

이지 각 유닛에 일을 배정하는 것이 아니다. 이것이 요청하신
"각 bank, channel에 따로따로 일을 시키는 것이 아니라, 데이터의 주소값만 다르게 보내는
것"이고, 이 문서의 tiling은 전부 이 원리 위에 세워진다.

### 1.4 주소 → lane 사상 — **아직 확정되지 않았다**

서로 배타적인 두 모델이 있고 어느 쪽도 실측되지 않았다. 이 절은 확정될 때까지
설계의 어떤 결정도 지탱하지 않는다.

**모델 A — descriptor 길이에서 유추.** B operand descriptor는 한 reduction 행에 대해
길이 `8192/batch` byte = `4096/batch` element의 **연속 구간**을 읽는다. `batch = 1`이면
연속한 n 원소 4096개가 lane 4096개를 채운다. 이 모델에서 lane 축은 **n**이다.

```
n 방향 연속 주소  ──►  (channel, bank, lane)
```

같은 연장선에서 vACC도 n 순서대로 선형이다. C drain descriptor가 vACC를 pitch
`8192/batch`로 읽어 C의 pitch `padded_n * 2`에 쓰므로, 두 pitch가 서로 다른데 같은
데이터를 옮기는 셈이고 vACC는 `batch × (4096/batch)` 연속 배열이 된다.

**모델 B — §1.6의 실측 순열에서 유추.** A 재배치 결과는 512 byte 덩어리마다 **서로 다른**
k 조각을 담는다. `512 = PIM_WIDTH`, `32 = BURST_SIZE`, `2 = TYPE_SIZE`이므로
`[channel 512 byte][bank 32 byte][lane 2 byte]`로 읽으면 lane 축이 **k**가 된다.

두 모델은 양립하지 않는다. 그리고 모델 B의 "512 byte 덩어리 = channel 하나"는 크기가
맞는다는 것뿐이고 실측이 아니다 — driver가 프로그램하는 주소 사상이 `ROBACOCH`
(`dma_probe.c:419-424`)여서 channel이 최하위 자리이면 512 byte 덩어리는 한 channel에
속하는 것이 아니라 16 channel에 걸쳐 흩어진다.

**descriptor 쪽은 확정됐다.** 같은 `ROBACOCH` 논리를 descriptor chain에 적용하면
512 byte group이 32 byte씩 16 channel에 하나씩 흩어진다는 결론이 나오고(§1.3), 그것이
`mPIM_RD_INSTR_MC`가 record를 16개 쓰는 이유를 그대로 설명한다. **미확정인 것은 데이터
쪽 사상뿐이다.**

데이터 쪽에도 크기 일치가 하나 있다. B operand descriptor의 `len`이 8192 byte이고
`ROBACOCH`가 그것을 16 channel에 나누면 channel당 512 byte인데, 그것이 정확히
`PIM_WIDTH`다 (§1.2). 즉 **"descriptor 하나 = 전 channel이 한 번에 소비하는 폭"**이라는
읽기는 상수와 맞아떨어진다. 이것도 크기 일치이지 실측은 아니다.

> **구현이 필요로 하는 것은 둘 중 어느 쪽도 아니다.** §1.6의 byte 순열과 §2.4의 크기
> 규칙만 있으면 descriptor를 낼 수 있다. `--aimx-lower-matmul`은 주소를 내는 것이지
> 해석을 내는 것이 아니다.

### 1.5 vecA — reduction 방향

vecA는 channel 하나에 `REG_SIZE = 256` 칸이고, 그 256칸은 `NUM_BANKS(16) × VECA_SIZE(16)`
으로 쪼개진다. 256칸은 `batch × (256/batch)`로 쓰인다.

> **정정.** 이 절은 예전에 "모든 channel에 같은 내용이 broadcast된다"고 적었다.
> §1.6의 실측과 어긋난다. 실측된 재배치 결과는 512 byte 덩어리마다 **다른** k 조각을
> 담고 16배 복제는 덩어리 **안**에서 일어난다. 즉 `OPCODE_BC`가 복제하는 방향은
> channel보다 작은 단위다. 그 단위가 bank라는 것은 크기 일치에서 온 유추이지 실측이
> 아니다 (§1.4).

```
batch = 1                       batch = 8
┌──────────────────────────┐    ┌───────────────┐
│ m=0 : k0 k1 … k255       │    │ m=0 : k0 … k31│
└──────────────────────────┘    │ m=1 : k0 … k31│
  → reduction step 256개         │       :       │
                                 │ m=7 : k0 … k31│
                                 └───────────────┘
                                   → reduction step 32개
```

즉 **vecA 한 번 적재가 `256/batch`번의 reduction step을 지탱한다.**
A descriptor의 길이는 `A_len = 512 * NUM_CHANNELS = 8192` byte = 4096 element다.
이것을 `256 element × 16 (16배 복제)`로 볼지 `16 element × 16 × 16`으로 볼지는 §1.4가
확정될 때까지 미정이다. 총량은 어느 쪽이든 같다. 위 그림도 k 256개가 한 곳에 늘어선
것처럼 보이지만 §1.6의 실측 순열에서는 16개씩 나뉘어 실린다.

### 1.6 A는 왜 재배치가 필요한가

vecA는 "행 방향으로 늘어선 A"가 아니라 "reduction 방향으로 늘어선 A"를 원한다.
그래서 `OPCODE_TP`(transpose)와 `OPCODE_BC`(복제)가 있다. `_matmul_silent`는
이 둘을 **별도의 준비 phase**에서 써서 `srcA_buf`(= `m * k * 16` element, 즉 A의 16배)를
만들어 두고, 계산 phase에서는 거기서 읽는다.

**재배치 순열은 실측으로 확정됐다.** `m=1, k=256, batch=1`에서 `A[j] = j`를 넣고
`Arel` 8192 byte를 되읽은 결과다.

```
Arel[c*256 + g*16 + t]  =  A[16*c + t]          (c, g, t = 0 … 15)
       │        │    └ 2 byte  = TYPE_SIZE
       │        └────── 32 byte = BURST_SIZE
       └─────────────── 512 byte = PIM_WIDTH
```

읽는 법: **A의 k 값 256개를 16개씩 잘라, 조각 `c`를 `c`번째 512 byte 덩어리에 놓고,
그 덩어리 안에서 16번 되풀이한다.** `k > 256`이면 재배치 loop이 `REG_SIZE = 256`씩 돌아
`Arel`은 8192 byte 블록의 나열이 되고 블록 하나가 k 256개를 담는다.

> **이 순열은 A만의 함수다.** 하드웨어 비밀이 없으므로 `--aimx-lower-matmul`이 그대로
> 계산해서 낼 수 있고, 나아가 host가 `aimx.stage`에서 곧바로 이 모양으로 써 넣을 수도
> 있다 (§8 I5). 덩어리 `c`가 물리적으로 channel `c`인지는 별개 문제이고 아직 확정되지
> 않았다 (§1.4) — 구현은 그것을 알 필요가 없다.

**준비 phase와 계산 phase의 분리는 하드웨어의 요구가 아니라 그 프로그램의 선택이다.**
순열이 밝혀졌으므로 §8 I2는 "가능한가"가 아니라 "어디서 계산할 것인가"의 문제가 됐다.
그리고 §8 I3(읽기 descriptor와 `BC|TP` 쓰기 descriptor를 하나로 합치기)은 **가능한 것으로
확인됐다** — 합치면 내용이 같고 512 byte 덩어리가 정확히 한 칸 순환한다
(`Arel_merged[c] = Arel_ref[(c+1) mod 16]`).

### 1.7 명령어 구조 (ISA)

`status` 필드(CDMA reserved, `CDMA_DESC_INFO 0x24`)에 실리는 flag word다.
memory controller는 평범한 DMA descriptor로 보고, PIM logic이 이 flag를 보고 동작한다.
이것이 **silent**의 뜻이다.

| bit | 이름 | 뜻 | 확인 상태 |
| --- | --- | --- | --- |
| 0 | `OPCODE_DESCR_DST_MASK` | descriptor 목적지 선택 | 미사용 |
| 1 | `OPCODE_A_MASK` | 이 전송은 operand A | 참조 구현에서 확인 |
| 2 | `OPCODE_B_MASK` | operand B | 확인 |
| 3 | `OPCODE_C_MASK` | operand C (vACC → memory) | 확인 |
| 4 | `OPCODE_MATMUL` | 연산 = matmul | 확인 |
| 5–7 | `OPCODE_ADD/SUB/MUL` | elementwise 연산 | 미사용 |
| 8 | `OPCODE_DECOUPLED` | decoupled 모드 (silent 아님) | 다른 경로 |
| 9 | `OPCODE_FUSION` | — | 미사용 |
| 10 | `OPCODE_BC` | **512 byte 덩어리 안에서 32 byte를 16번 복제** — channel 사이 복제가 아니다 (§1.6 실측) | 확인 |
| 11 | `OPCODE_CL` | — | 미사용 |
| 12 | `OPCODE_PIM_MODE` | PIM logic이 내부에서 처리 (host로 안 돌려보냄) | 확인 |
| 13 | `OPCODE_TP` | transpose | 확인 |
| 14 | `OPCODE_HOLD` | 누산기 유지 (추정) | matmul 경로에서만 안 쓴다. `memcpy.c`는 단독 opcode로 쓴다 — §15 E3 |
| 15–17 | `OPCODE_BATCH_L/M/H` | batch 코드 (1/2/4/8) | 확인 |
| 18 | `OPCODE_STRIDE` | pitch를 `<< 33` 해서 주소에 OR 한다. `src \| (pitch << 33)` | 확인 (§1.1) |
| 19 | `OPCODE_LUT` | lookup table | 미사용 |

descriptor 하나의 의미는 결국 이것이다.

```
(src, src_pitch) ──[flags, batch]──► (dst, dst_pitch),  len byte × rows행
```

**`len`은 출력 byte 수다. 입력 byte 수가 아니다.** 입력 크기를 주면 그만큼만 쓰이고
나머지는 이전 내용이 남는다 — 재배치를 한 descriptor로 합치면서 `len = 512`(입력 크기)를
준 실측에서 512 byte 덩어리 하나만 쓰이고 나머지 15개가 그대로 남았다. 합칠 때는
`len = 512 * batch * NUM_CHANNELS`를 준다 (§9.5).

`rows`가 어디서 오는지는 아직 확정되지 않았다. batch 코드는 1/2/4/8만 표현하는데 B
descriptor 하나는 `len = 8192/batch`로 `256/batch`개의 reduction 행을 덮어야 하므로,
행 수가 batch 코드만으로 결정된다고 볼 수 없다. 참조 구현의 다섯 가지 사용례가 전부
이 한 형태의 특수화다 (§9.4). **이것이 실제 ISA이고, `_matmul_silent`는 이 ISA로 쓴
프로그램 하나다.**

### 1.8 host 자료 배치 계약과 runtime 제약

descriptor 산술만으로는 부족하다. 호출자가 지켜야 하는 계약이 따로 있고, 어기면 오류가
아니라 조용한 오답이 된다. `ubench/matmul/matmul.cpp`가 이것을 하고 있다.

| | 계약 | 근거 |
| --- | --- | --- |
| (a) | `k % 256 == 0`. 어기면 호출자가 거부한다 | `matmul.cpp`: `q_size % 256 != 0`이면 exit |
| (b) | A는 `padded_k = ceil(k/CHUNK) * CHUNK`로 패딩하고 행 pitch가 `padded_k`다 | `matmul.c`의 `padded_q_size` |
| (c) | B의 행 pitch는 `padded_n`이다. `n != CHUNK`이면 다시 깐다 | `matmul.cpp`: `pim_align_chunk()` |
| (d) | `pim_args->r_size`에는 n이 아니라 `padded_n`을 넣는다 | `matmul.cpp`: `set_info->r_size = padded_r_size` |
| (e) | C의 행 pitch도 `padded_n`, 할당은 `m × padded_n` | 같음 |
| (f) | B와 C의 행 pitch는 **서로 같아야 한다** — 같은 `r_size` 필드가 둘 다에 실린다 | `matmul.c`의 B·C descriptor |

runtime 쪽 제약이 둘 더 있다. 둘 다 참조 구현이 그냥 안고 있는 것이라 코드를 읽어서는
드러나지 않는다.

> **(g) PMM을 가장 먼저 잡는다.** vACC는 `pmm_pa + VACC_OFFSET`에 있는데,
> `pim_private_malloc(PMM_SIZE, fpga, 7)`이 `pim_malloc`이 이미 내준 물리 주소를 그대로
> 돌려주는 경우가 관측됐다 (`pmm_pa == A_pa == 0xe00000`). 데이터 buffer가
> `[pmm_pa, pmm_pa + 0x30000)`과 겹치면 계산 도중 덮어써진다. runtime은 PMM을 먼저
> 잡고 겹침을 검사해야 한다.
>
> **(h) `pim_malloc`은 실패해도 NULL을 돌려주지 않는다.** `PIM] OOM`을 찍고 프로세스를
> 죽인다 (exit 255). 게다가 1 MB부터 16 GB까지 요청이 전부 '성공'했다 — 할당은 주소만
> 떼어 준다. 따라서 §6.2의 PIM DRAM 예산은 **compile time에 확정되어야 하고**, 할당
> 실패를 감지해 더 작은 execution tile로 물러서는 설계는 성립하지 않는다.

---

## 2. Tiling — 하드웨어에서 위로 쌓는다

요청하신 대로 "lane이 처리하는 크기 → channel이 처리하는 크기 → PIM Device"의 순서로
아래에서 위로 쌓는다. 각 단계는 하드웨어 상수가 정하고, compiler는 그 배수로만 자른다.

### 2.1 다섯 단계

| 단계 | 출력 tile (m × n) | reduction | 정하는 것 | 크기 근거 |
| --- | --- | --- | --- | --- |
| ① lane tile | `1 × 1` | 1 | 하드웨어 | lane 하나 = vACC 한 칸 |
| ② bank tile | `1 × 16` | 1 | 하드웨어 | `BURST_SIZE / TYPE_SIZE = ALU_SIZE` |
| ③ channel tile | `1 × 256` | 1 | 하드웨어 | `PIM_WIDTH / TYPE_SIZE` |
| ④ **output block** | `batch × (4096/batch)` | 1 | 하드웨어 + `batch` 선택 | lane 4096개 |
| ⑤ reduction step | `batch × (4096/batch)` | `256/batch` | 하드웨어 | vecA `REG_SIZE = 256` |
| ⑥ **execution tile** | `m_t × n_t` | `k_t` | **compiler** | PIM DRAM 예산 |
| ⑦ PIM Device 배정 | `m_dev × n_dev` | `k` | **compiler** | `#pim.device_mapping` |

①②③은 compiler가 고를 것이 없다. 자유도는 **④의 `batch`**, **⑥의 `(m_t, k_t, n_t)`**,
**⑦의 device 분할** 세 개뿐이다.

### 2.2 output block — lane 4096개를 어떻게 자를 것인가

`batch`는 lane 4096개를 `batch × (4096/batch)`로 다시 자른다.

```
batch = 1                                         batch = 8
출력 1행 × 4096열                                 출력 8행 × 512열

 n:  0 ──────────────────────────────── 4095            n: 0 ───────── 511
     ├─ch0─┤├─ch1─┤ …………………… ├─ch15─┤        m=0  ├── 4096 lane의 1/8 ──┤
     0   255 256 511            3840  4095      m=1  ├─────────────────────┤
     └ bank0..15 × lane0..15 ┘                   :
                                                 m=7  ├─────────────────────┤

 눈금:  m 방향 1,  n 방향 4096                    눈금:  m 방향 8,  n 방향 512
```

**하나가 좋아지면 다른 하나가 나빠진다.** `batch`를 키우면 m 눈금이 거칠어지고
n 눈금이 고와진다. lane 수는 언제나 4096으로 같으므로 **`batch`는 연산 시간에 영향이
없다.** 영향을 주는 것은 두 가지뿐이다.

1. **padding 낭비** — m과 n의 눈금이 바뀌므로
2. **descriptor 개수** — `batch`에 정비례해서 늘어난다 (§2.4)

### 2.3 execution tile — 왜 output block보다 커야 하는가

output block 하나는 4096칸이다. 이것만으로는 안 되는 이유가 둘이다.

1. **reduction이 안 끝난다.** vACC는 output block 하나분이므로, 그 4096칸이 확정되려면
   reduction 전체(또는 §15 E3이 성공하면 한 조각)를 다 돌아야 한다.
2. **descriptor 발행 비용.** descriptor group 하나가 512 byte인데 reduction step
   하나가 소비하는 B는 `(256/batch) × (4096/batch) × 2` byte다. `batch = 1`이면
   2 MB, `batch = 8`이면 32 KB다. execution tile을 키워 이 비율을 유지해야 한다.

그래서 execution tile은 **output block을 m·n 두 방향으로 여러 개 묶고, k 방향으로는
전부 도는** 직사각형이다.

```
execution tile (m_t × k_t × n_t)
  = output block  (m_t/batch) × (n_t·batch/4096) 개
  × reduction step (k_t·batch/256) 회
```

### 2.4 크기 규칙 (legality)

> **2026-08-13 — Triton block 조건이 거부에서 padding으로 바뀌었다.** 계층이
> **Triton block → block group → execution tile**로 명시되었다. block group은
> Triton block 정수 개를 하드웨어 실행 단위(output block 높이 `batch`, 폭
> `4096/batch`, reduction step `256/batch`)의 정수 배에 채운 것이고, 딱 맞지 않으면
> 남는 자리를 stage가 0으로 채운다 — 그 몫의 lane과 reduction step이 노는 것이
> 비용이고, `--convert-pim-to-aimx`가 remark로 비율을 알린다. 아래 표의 조건은 이제
> **물리 크기**(`#aimx.tile_group`의 `execution`)에 대한 것이며, 물리 크기는 구성상
> 언제나 만족한다: m은 `lcm(BLOCK_M, batch)`(padding 항상 0, `m_t = group-m * lcm`),
> n·k는 Triton block 묶음을 하드웨어 단위로 올림한 것이다. k tile의 눈금은 block
> group과 vecA 적재 단위 256의 lcm이다. host 좌표(loop 전진 폭, stage/unstage의
> origin·sizes)는 **유효 크기**로 세고, PIM DRAM·chain은 물리 크기만 본다 —
> `aimx.stage`/`aimx.unstage`의 `group_rows`/`group_cols`(`[useful, phys]`)가 그
> 주기 배치다. batch 자동 선택(§2.5)도 "나눠떨어지는 최대"에서 "padding 0인 값 중
> 최대, 없으면 노는 비율 최소"로 바뀌었다.

> **2026-08-14 — 3계층의 손잡이가 공통 IR로 옮겨졌다.** 공통 계층이 데이터 3계층
> (block ⊂ block group ⊂ execution tile)을 직접 갖게 되어(`--pim-form-block-groups`,
> `--pim-form-execution-tiles`/`#pim.execution_tile`), `--convert-pim-to-aimx`의
> `group-m`/`tile-k`/`tile-n` 옵션이 사라졌다. 남는 옵션은 `batch`와
> `bank-capacity-bytes` 뿐이다. 이 pass가 보는 "Triton block"은 이제 공통 block
> group일 수 있고(구분하지 않는다), execution tile 상한은 `#pim.execution_tile`
> (원소 단위, 유효 데이터 기준)에서 읽는다: m 상한은 `lcm(block m, batch)` 경계로
> 올려 `m_t`가 되고(0이면 group 하나), n·k 상한은 담기는 group 수(내림, 최소 1) ×
> 물리 group 크기로 buffer 상한 `n_max`(0이면 4096)·`k_max`(0이면 32768)가 된다.
> k 상한의 예산-최대 모드(`tile-k=0` 이분 탐색)는 소비자가 없어 제거했다. §6.1·§12.4
> 등 본문의 `group-m`/`tile-k`/`tile-n` 서술은 이 정정 이전의 기록이다.

> **2026-08-14 (2차) — 3계층이 단순한 한 규칙으로 통일됐다.** 사용자 방침: 덕지덕지
> 최적화·특례를 걷어내고 "배수가 안 맞으면 padding, 맞으면 없음" 하나로 간다.
> 1. **m의 lcm 특례 제거.** m도 n·k와 같은 채움 규칙이다. block group은 세 축 모두
>    `tiles = max(1, ⌊unit/block⌋)`, `useful = tiles·block`, `phys = ⌈useful/unit⌉·unit`.
>    m padding이 생길 수 있어 `aimx.unstage`에 `group_rows`가 추가됐고 A stage도
>    `group_rows`를 싣는다. batch 자동 선택(padding 비율 최소화)도 제거 — batch는
>    launch 옵션(기본 8)이다.
> 2. **하드웨어 채움이 공통 IR로 올라갔다.** profile이 `execution_unit = [batch,
>    4096/batch, 256/batch]`를 싣고, `--pim-form-block-groups`가 이 눈금에 block을
>    채워 `#pim.block_group<tiles, useful, phys>`를 func에 부착한다.
>    `--convert-pim-to-aimx`는 이것을 소비만 하며 `batch` 옵션이 사라졌다(남는 옵션은
>    `bank-capacity-bytes` 하나). padding remark도 공통 pass가 낸다.
> 3. **상한 0의 정책 숫자 제거.** m: group 하나, n: 한 device 폭, k: **남는 예산
>    전부**(footprint가 k에 단조증가라 이분 탐색으로 최대 kUnit 배수를 찾는다 —
>    execution tile은 PIM DRAM에 맞게 적재된다). 고정값 32768은 사라졌다.
> 4. **profile의 죽은 필드 제거.** `pim_core_shape`, `requires_uniform_instruction_sequence`.
> 5. **map-devices granularity는 partitions > 1인 축에만** 검사·적용된다.

배수 조건을 어기면 **오류가 아니라 조용한 오답**이 난다. 전부 진단으로 막아야 한다.

| 축 | 배수 조건 | 어기면 | 유래 |
| --- | --- | --- | --- |
| `m_t` | `batch` | output block 행이 안 맞아 batch class가 섞인다 (§2.5) | ④ |
| `k_t` | **256** | vecA 적재가 꼬리를 못 덮는다 → **조용한 오답** | ⑤ `REG_SIZE` |
| A의 행 pitch | **4096 element** (`padded_k`) | A 행 간격이 하드웨어 기대와 다름 | `CHUNK` |
| `n_t` | **`4096/batch`의 배수** | 배수가 아니면 `r_loop_var`가 내림되어 **n 꼬리가 조용히 빠진다** | ④ + §15 E9 |
| B와 C의 행 pitch | **서로 같아야 한다** | 같은 `r_size` 필드가 둘 다에 실린다 | §1.8 (f) |
| `batch` | **`{1, 2, 4, 8}`** | 그 밖의 값은 encoding이 없다 | `N_BATCH_SIZE_TYPES = 4` |

절대 하한: `k_t ≥ 256`, **`n_t ≥ 4096/batch`** (§15 E9로 확정, 2026-08-11).
배수 조건은 `n_t % (4096/batch) == 0`이다 — 어기면 `r_loop_var`가 내림되어 n 꼬리가
조용히 빠지고, `0`이 되면 C를 아예 쓰지 않는다. **상한은 legality가 아니라 PIM DRAM
예산이다.**

> **정정.** 이 절은 한때 `n_t = 4096` **고정**, `k_t`도 사실상 한 chain이 덮는 최대치를
> 4096으로 적었다. **참조 구현이 그것을 반증한다.** `_matmul_silent`(`matmul.c:152-154`)는
>
> ```c
> r_loop_var = (r_size * TYPE_SIZE) / (PIM_WIDTH * NUM_CHANNELS / batch_size);  // = n/4096
> for (r = 0; r < r_loop_var; r++)                        // n = 28672 → 7 회
>     for (q = 0; q < q_size; q += REG_SIZE/batch_size)   // k =  8192 → 32 회
> ```
>
> 처럼 `r` loop 7회와 `q` loop 32회를 **하나의 `pim_exec` chain**에 넣는다. 즉 256과
> 4096은 **reduction step과 output block의 크기**이지 chain 하나가 덮을 수 있는 `k`·`n`의
> 상한이 아니다. `./matmul 1 8192 28672 1`이 이 경로를 그대로 돈다.
>
> 반대 방향(`n_t`를 `4096/batch`까지 **줄이는** 것)은 여전히 **실측이 없다.** 참조 host가
> `pim_align_chunk()`로 n을 언제나 `CHUNK = 4096`으로 패딩해 왔으므로 (§1.8 (c)), B의 행
> pitch가 4096 element 미만인 경우는 한 번도 실행된 적이 없다 (§15 E9).

descriptor group 수 (execution tile 하나, `batch` 단일):

```
G = 2*(k_t/256)*(m_t/batch)                                    A 재배치
  + (m_t/batch) * (n_t*batch/4096) * (2*(k_t*batch/256) + 1)   계산
```

`n_t = 4096` 고정에서는 `(n_t*batch/4096) = batch`이므로 두 항이 이렇게 갈린다.

```
재배치 항  = 2*(k_t/256)*(m_t/batch)          ∝ 1/batch
계산   항  ≈ m_t * 2*(k_t*batch/256)          ∝ batch
```

즉 두 항의 비가 정확히 `batch²`이고, 큰 `k_t`에서는 계산 항이 지배하므로
**descriptor 개수는 `batch`에 정비례한다.** `m_t=16, k_t=4096, n_t=4096`에서
`batch=1`이면 1040 group, `batch=8`이면 4176 group으로 4배 차이가 난다.
descriptor 생성이 전체 시간의 52~63%를 차지하므로 (§7) 이 차이는 그대로 성능이다.

§8 I3으로 재배치를 한 descriptor로 합치면 첫 항의 계수 2가 1이 된다.

### 2.5 `batch` 선택

> **2026-08-11 정정 — 기본값이 `batch = 1`에서 "legal한 최대값"으로 바뀌었다.**
> 아래에 있던 "`batch = 1`이 이긴다" 표는 목적 함수를 `G`(descriptor 수) 하나로 잡은
> 것이었고, 그 근거인 "descriptor 생성이 전체 시간의 52~63%"는 **참조 구현의 profile**
> 이지 이 구현의 것이 아니다. 우리 구현에서는 `invoke`(= `pim_exec`, 순수 하드웨어
> 시간)가 wall의 93%이고 `build`는 0.0%다.
>
> 실측하면 `invoke`는 두 항의 합이다.
>
> ```
> T = a * G + bytes / B          a = 0.44 us/group,  B = 56 GB/s
>
>   G     = m_t*n_t*k_t*batch / (4096*128)        ∝ batch
>   bytes = (m_t/batch) * k_t * n_t * 2           ∝ 1/batch      ← B tile 재읽기
> ```
>
> 두 항을 같게 놓으면 `batch = sqrt(2*4096*128 / (a*B)) = 6.5`이고, legal한 값 중 가장
> 가까운 것이 **8**이다. `m_t=8, k=256, n_t=4096`에서 52.5 → 179.4 GFLOPS(3.4배)로
> 실측됐다. `batch = 1`은 전송 항이 발행 항의 20배인 지점이라 최적에서 가장 먼 값이었다.
>
> **arithmetic intensity = `batch`다.** on-card B 총량이 `2*M*N*K/batch`이고 다른 어떤
> 손잡이(`m_t`, `n_t`, `k_t`, `BLOCK_*`)도 이 식에 나타나지 않는다.

**규칙: `batch`는 Triton block이 허락하는 최대값이다.** 값이 아니라 규칙인 이유는
legality가 block에 달려 있기 때문이다 — 출력 block 폭 `4096/batch`가 `BLOCK_N`의,
reduction step `256/batch`가 `BLOCK_K`의 배수여야 한다. 8을 상수로 박으면 `BLOCK_K=64`
같은 흔한 값이 컴파일 오류가 된다. `--convert-pim-to-aimx=batch=0`(기본값)이 이 규칙이고,
명시로 주면 그 값을 쓰되 legality를 어기면 진단이 나간다.

`batch`를 8보다 키우지 못하는 것은 하드웨어 제약이다 — 참조 구현의
`N_BATCH_SIZE_TYPES = 4`라 인코딩이 {1,2,4,8}뿐이다. 균형점이 6.5인 것과 겹쳐,
8이 상한이자 최적이다.

| 기준 | `batch`가 커질 때 |
| --- | --- |
| B 전송량 | `1/batch`. **지배 항이다** |
| descriptor 수 `G` | `batch`에 정비례. `batch=8`에서 발행 항이 전송 항을 넘어선다 |
| n padding | `ceil(n/4096)*4096`으로 `batch`와 무관하다 |
| m padding | `m_t`를 `lcm(batch, BLOCK_M)`으로 올리므로 block 단위로는 0이다 |
| chain buffer | `G`에 비례. `k_max=32768`에서 8.4 MB → 34 MB (예산 안) |

**`batch`는 execution tile 하나 안에서 하나여야 한다.** 참조 구현은 `m`을 이진 분해해
여러 batch class를 한 호출에 섞는데(`batch_size_cnt`), 그러면 `n` 조건이 모든 class에
대해 성립해야 해서 가장 엄격한 `4096`이 된다. `m_t`를 `batch`의 배수로 padding하면
class가 하나만 남고, 그 결과 (a) `n` 조건이 `4096/batch`로 완화되고 (b) descriptor
program이 **단일 loop nest**가 된다 — §7의 재사용이 여기서 나온다.

### 2.6 문제를 전치할 것인가

lane은 **n 방향**에 대응한다 (§1.4). 따라서 `n`이 작고 `m`이 크면 lane이 놀고 padding이
커진다. 그런데 `C = A·B`를 `Cᵀ = Bᵀ·Aᵀ`로 계산하면 lane이 **m 방향**에 대응한다.

| | lane ↔ | vecA에 들어가는 것 | 유리한 모양 |
| --- | --- | --- | --- |
| 정방향 | n | A | `n ≥ m` |
| 전치 | m | B | `m > n` |

전치는 stage 단계에서 index 순서만 바꾸면 되므로 **추가 비용이 없다** — 어차피 host가
PIM DRAM layout으로 다시 모으기 때문이다. `m = 4096, n = 32`인 문제에서 정방향은
`n`을 512로 부풀려 16배를 낭비하지만 전치하면 낭비가 없다.

> **규칙: `m_dev > n_dev`이면 전치해서 실행하고, `aimx.unstage`가 되돌린다.**

---

## 3. 전체 tile 분할과 분배 — 그림

### 3.1 위에서 아래로

```
                                C  (M × N)
    ┌──────────────────────────────────────────────────────────────┐
    │  ① PIM Device 분할   #pim.device_mapping                     │
    │     partitions = factorize(D)      (D = 8 → (4, 2))          │
    │                                                              │
    │     ┌─────────┬─────────┐                                    │
    │     │ dev 0,0 │ dev 0,1 │   각 칸 = pim.place 의 capacity     │
    │     ├─────────┼─────────┤   = (m_dev × n_dev),  reduction 전체 k
    │     │ dev 1,0 │ dev 1,1 │                                    │
    │     ├─────────┼─────────┤   device 간에는 서로 독립           │
    │     │ dev 2,0 │ dev 2,1 │   (fpga_id 가 다르다)               │
    │     ├─────────┼─────────┤                                    │
    │     │ dev 3,0 │ dev 3,1 │                                    │
    │     └─────────┴─────────┘                                    │
    └──────────────────────────────────────────────────────────────┘
                              │
                              ▼  device 하나가 맡은 (m_dev × n_dev), reduction k
    ┌──────────────────────────────────────────────────────────────┐
    │  ② execution tile 분할   PIM DRAM 예산이 정한다                │
    │                                                              │
    │      n_dev ───────────────────────────────►                  │
    │     ┌────────┬────────┬────────┬────────┐  ▲                 │
    │     │  T0    │  T1    │  T2    │  T3    │  │                 │
    │     ├────────┼────────┼────────┼────────┤  m_dev             │
    │     │  T4    │  T5    │  T6    │  T7    │  │                 │
    │     └────────┴────────┴────────┴────────┘  ▼                 │
    │                                                              │
    │     모든 T 가 같은 (m_t × n_t) — 모자라면 0 padding            │
    │     같은 행의 T 들은 A 를 공유한다 (A 재배치를 나눠 쓴다)        │
    │     k 는 나누지 않는다 (§15 E3 성공 시 나눌 수 있게 된다)        │
    └──────────────────────────────────────────────────────────────┘
                              │
                              ▼  execution tile 하나 = descriptor chain 1회 실행
    ┌──────────────────────────────────────────────────────────────┐
    │  ③ output block 분할   하드웨어 고정                           │
    │                                                              │
    │      n_t ─────────────────────────────►                      │
    │     ┌────┬────┬────┬────┬────┬────┬────┬────┐  ▲             │
    │     │ OB │ OB │ OB │ OB │ OB │ OB │ OB │ OB │  │             │
    │     ├────┼────┼────┼────┼────┼────┼────┼────┤  m_t           │
    │     │ OB │ OB │ OB │ OB │ OB │ OB │ OB │ OB │  │             │
    │     └────┴────┴────┴────┴────┴────┴────┴────┘  ▼             │
    │                                                              │
    │     OB 하나 = batch × (4096/batch) = 4096칸 = lane 전체        │
    │     개수  = (m_t/batch) × (n_t·batch/4096)                    │
    │     OB 하나마다 vACC 한 벌을 다 쓰고 C descriptor 로 비운다     │
    └──────────────────────────────────────────────────────────────┘
                              │
                              ▼  output block 하나 = lane 4096개
    ┌──────────────────────────────────────────────────────────────┐
    │  ④ lane 사상   하드웨어 고정, 주소로만 결정된다                 │
    │                                                              │
    │     n 방향 연속 주소 4096/batch 개  ──►  (channel, bank, lane) │
    │                                                              │
    │     compiler 가 아는 것: "연속 구간이 배열 전체를 채운다" 뿐     │
    │     interleave 순서는 몰라도 되고, 알 필요도 없다               │
    └──────────────────────────────────────────────────────────────┘
```

### 3.2 reduction 방향

위 그림은 출력 평면만 보여 준다. reduction은 ③ 안에서 시간축으로 돈다.

```
output block 하나가 확정되기까지

  k:  0 ─────────────────────────────────────────────────────► k_t
      ├──────┼──────┼──────┼── … ──┼──────┤
       step0  step1  step2          step(k_t·batch/256 - 1)

  step 하나 = A descriptor 1개 + B descriptor 1개
            = vecA 적재 (batch × 256/batch)  +  B (256/batch × 4096/batch)
            = 256/batch 사이클

  마지막 step 뒤에 C descriptor 1개  →  vACC 4096칸을 C 로 흘려보낸다
```

**vACC는 lane당 한 칸뿐이므로 output block 하나가 끝나기 전에는 다음 block으로 갈 수
없다.** 그래서 loop 순서는 아래로 고정된다.

```
for  m-group  in  m_t/batch:                # output block 행
  for  r      in  n_t·batch/4096:           # output block 열
    for  step in  k_t·batch/256:            # reduction
        A descriptor          ← vecA 적재
        B descriptor          ← MAC
    C descriptor              ← vACC 비우기
```

B는 m-group마다 다시 읽힌다. 그러나 그 읽기가 곧 연산이므로 낭비가 아니다:
총 사이클 = `(m_t/batch) × (k_t·batch/256) × (256/batch) × (n_t·batch/4096)`
= `m_t·k_t·n_t / 4096` = 이상적인 값과 정확히 같다. **lane 이용률은 `batch`와 무관하게
언제나 100 %다** (padding 부분을 제외하면).

### 3.3 데이터 분배 — 누가 무엇을 읽는가

```
   host DRAM (Triton buffer)
        A (M×K)                B (K×N)                    C (M×N)
          │                       │                          ▲
          │ aimx.stage            │ aimx.stage               │ aimx.unstage
          │  · device 몫만 gather │  · device 몫만 gather     │  · padding 버리기
          │  · f32 → bf16         │  · f32 → bf16            │  · bf16 → f32
          │  · 0 padding          │  · 0 padding             │  · strided scatter
          │  · 행 pitch = padded_k│  · 행 pitch = padded_n    │  · 행 pitch = padded_n
          │                       │    (C와 같아야 한다)      │    (B와 같아야 한다)
          ▼                       ▼                          │
   PIM DRAM  [A block]        [B block]                 [C block]
              m_t × padded_k    k_t × padded_n           m_t × padded_n

             [PMM]  pmm_pa .. +0x30000,  vACC = pmm_pa + 0x10000
             데이터 buffer가 이 구간과 겹치면 안 된다 (§1.8 (g)) — 같은 물리 주소가 관측됐다
                │                   │                        ▲
    ┌───────────┘                   │                        │
    │ BC | TP                       │                        │
    ▼                               │                        │
   [Arel]         ← §1.6 순열       │                        │
    m_t × k_t × 16 element          │                        │
    = m_t * k_t * 32 byte           │                        │
    (§8 I5: host가 직접 만들 수 있다)│                        │
    │                               │                        │
    │ A_MASK|MATMUL|PIM_MODE        │ B_MASK|MATMUL|          │ C_MASK|MATMUL|
    ▼                               ▼ PIM_MODE|STRIDE         │ STRIDE
   vecA ─────────────────►  lane 4096개  ────►  vACC ───────┘
   (덩어리마다 다른 k)       (n → lane, 미확정)  (n → lane, 미확정)
```

**재배치 순열은 실측으로 확정됐다** — `Arel[c*256 + g*16 + t] = A[16*c + t]` (§1.6).
`--aimx-lower-matmul`이 그대로 계산해서 낼 수 있다. `k > 256`이면 8192 byte 블록의
나열이 된다.

정리하면 세 operand의 성격이 다르다.

| operand | 배치 | 어떻게 분배되는가 |
| --- | --- | --- |
| **A** | vecA. **512 byte 덩어리 `c`가 k 조각 `[16c, 16c+16)`을 맡고, 그 조각이 덩어리 안에서 16번 복제된다** (§1.6 실측) | reduction 방향으로 잘려 순차 적재 |
| **B** | bank 상주, 주소로 흩어짐 | n 방향 연속 주소 → lane (§1.4 모델 A, **미확정**) |
| **C** | vACC, B와 같은 사상 | n 방향 연속 주소 → lane (§1.4 모델 A, **미확정**) |

> A행은 실측이고 B·C행은 descriptor 길이에서 온 유추다. 두 줄이 서로 다른 축을 lane에
> 놓으므로 §1.4가 확정되기 전에는 이 표를 근거로 새 결정을 내리지 않는다.

### 3.4 여러 tile을 하나의 PIM Device로 묶는 방법

요청하신 "block을 묶어서 하나의 PIM Device에 준다"가 구체적으로 이렇게 된다.

```
logical tile (BLOCK_M × BLOCK_K × BLOCK_N)  — Triton program instance 하나
      │
      │  ①②  --pim-map-devices 가 balanced block 으로 device 에 나눈다
      │      device 하나의 몫 = pim.place 의 capacity = (m_dev × n_dev)
      ▼
      logical tile 여러 개가 device 하나에 모인다
      예: gemm.py  8 × 128 = 1024 개
      │
      │  ③  --convert-pim-to-aimx 가 그것을 execution tile 로 다시 묶는다
      │      묶는 단위는 BLOCK 이 아니라 §2.4 의 배수 조건과 PIM DRAM 예산
      ▼
      execution tile (m_t × k_t × n_t)  — descriptor chain 1회 실행
      │
      │  ④  하드웨어가 output block 으로 자동으로 자른다 (주소 사상)
      ▼
      output block (batch × 4096/batch)  — lane 4096개
```

**핵심은 ③이다.** `BLOCK_M/N/K`는 눈금일 뿐이고, 묶는 결정은 `--convert-pim-to-aimx`가
`pim.place`의 `capacity`와 PIM DRAM 예산을 보고 한다. logical tile 경계는 execution tile
경계와 일치할 필요조차 없다 — execution tile은 `capacity` 직사각형을 §2.4의 배수로
다시 자른 것이기 때문이다.

**execution tile은 공통 설계의 어느 자리에 있는가.** `PIM_IR_Redesign.md` §1.1의 사슬은
`logical tile → execution wave → PIM Device mapping → PIM Core mapping`이다. execution
tile은 그중 **`PIM Device mapping` 바로 아래**, 즉 *한 PIM Device 안에서 어느 만큼을 한
번에 PIM DRAM에 올려 둘 것인가*를 정하는 **시간적 상주 단위**다. 공간 계층(Device →
Group → Core)과 다른 축이므로 새 하드웨어 계층이 아니다.

`execution wave`가 이 자리를 대신하지 못하는 이유는 하나다. wave는 *한 PIM Group 안의
여러 PIM Core가 서로 다른 logical tile을 lockstep으로 도는 묶음*인데, AiMX는
`pim_core_shape = [1]`이라 Core가 하나뿐이고 (§15 E11로 확정), 그래서 wave가 항상 tile
하나로 퇴화한다. **공간적으로 나눌 것이 없으므로 남는 자유도가 시간적 묶음뿐이다.**

**왜 logical tile 경계에 맞추지 않아도 되는가 — 조건을 명시한다.** 이것은 무조건 참이
아니다. 다음이 성립할 때만 참이다.

> **전제.** `pim.device` 안의 계산이 `pim.place`의 직사각형 위에서 **원소별로 분해되는
> 하나의 `linalg.matmul`**이어야 한다. 즉 (a) `linalg.matmul`이 정확히 하나, (b)
> `pim.store`가 정확히 하나, (c) 누산기 초기값이 0, (d) 출력이 2차원.

이 전제 아래에서 `C[i,j] = Σ_k A[i,k]·B[k,j]`는 **`i`와 `j`를 어떻게 묶든 같은 값**이다.
그러므로 `m_t`가 `BLOCK_M`을 가로질러도 오답이 나지 않는다. 전제가 깨지는 커널(예: block
안에서 정규화를 하는 softmax류)에서는 이 논증이 성립하지 않는다.

**`--convert-pim-to-aimx`가 그 전제를 실제로 검사한다** (`readRoles`). 넷 중 하나라도
어긋나면 진단을 내고 멈춘다 — 출력이 3차원인 `split_k.py`가 지금 그 경로로 걸린다
(§17.4). 즉 block 경계를 무시하는 것은 **가정이 아니라 검사된 조건**이다.

**그 위에서 block 보존이 무엇을 더 주는가 — 아무것도 없다.** GPU에서 block 경계와
순서(`GROUP_M` swizzle 같은 것)가 성능을 바꾸는 이유는 L2가 있어서 직전 block이 읽은
데이터가 우연히 남아 있기 때문이다. **AiMX에는 그런 것이 없다.** host DRAM과 PIM DRAM
사이는 전부 명시적 `datacopy_cpu2pim`이다. 그래서

> AiMX에서 데이터 지역성은 **"어떤 byte를 몇 번 DMA하는가"가 전부**이고, 그것을 정하는
> 것은 (a) PIM DRAM buffer의 크기와 (b) host loop 순서 두 개뿐이다. **둘 다
> `BLOCK_M/N/K`를 알 필요가 없다.**

그리고 배수 조건을 나란히 놓으면 block 조건이 하드웨어 조건보다 항상 느슨하다.

| 축 | block 배수 조건 | 하드웨어 조건 (§2.4) | 실제로 구속하는 쪽 |
| --- | --- | --- | --- |
| `k` | `% BLOCK_K` (예: 16) | `% 256` | **하드웨어** (16이 256을 나눈다) |
| `n` | `% BLOCK_N` (예: 32) | `% 4096` | **하드웨어** (32가 4096을 나눈다) |
| `m` | `% BLOCK_M` (예: 32) | `% batch` (= 1) | block |

`k`·`n`에서는 block 조건이 발동할 일이 없고, `m`에서만 발동하는데 그것이 하는 일은
`m_t`의 하한을 `BLOCK_M`으로 올리는 것뿐이다. §6.1의 전송량 식에서 `m_t`가 커지면 A의
`⌈M/m_t⌉·m_t` 항과 chain 크기가 **둘 다 커진다**. 그래서 block 경계 보존은 AiMX에서
얻는 것이 없고 잃는 것만 있다.

> **남는 것은 이름뿐이다.** "execution tile"이라는 말이 공통 설계의 "execution wave"와
> 나란히 놓이면 같은 계층처럼 읽힌다. 둘은 축이 다르다 — wave는 **공간**(어느 Core가
> 무엇을), execution tile은 **시간**(무엇을 한 번에 상주)이다. 이 문서에서는
> execution tile로 부르고, 그것이 Device 안의 시간적 단위임을 이 절에서 정의한다.

### 3.5 `BLOCK`을 키워서 tile 하나를 크게 만들 수는 없다

이것은 AiMX가 아니라 Triton 쪽 제약이다. gemm.py의 device 직사각형은
`(m, k, n) = (256, 28672, 4096)`인데, 이것을 logical tile 하나로 만들려면

- accumulator `tl.zeros((256, 4096))` = 1,048,576 element — block tensor 한계
- A block `256 × 28672` = 7,340,032 element — 표현 불가

이다. 그래서 묶는 일은 반드시 compiler가 해야 한다.

---

## 4. `pim_core_shape` 상세

### 4.1 공통 실행 계층

`PIMDialect.td` 머리말이 정의한다.

> 공통 실행 계층은 **PIM Device → PIM Group → PIM Core** 이고, target 용어(DPU/tasklet/
> MRAM/channel/bank 등)는 여기 등장하지 않는다.

| 계층 | 뜻 | profile 필드 |
| --- | --- | --- |
| **PIM Device** | 서로 독립적으로 launch 되는 최상위 단위 | `number_of_pim_devices` |
| **PIM Group** | PIM Device 하나 안의 core 묶음 | (암묵적으로 device당 1개) |
| **PIM Core** | **서로 다른 logical tile을 배정받을 수 있는 단위** | `pim_core_shape` |
| (계층 아님) | core 안에서 같은 명령을 함께 수행하는 병렬 lane | `number_of_parallel_lanes_per_pim_core` |

**PIM Core의 정의가 전부다: "compiler가 서로 다른 logical tile을 배정할 수 있는가".**
배정할 수 없으면 core가 아니라 lane이다.

`pim_core_shape`는 그 core들이 이루는 **격자의 모양**이다. rank가 1이면 1차원 배열,
`[4, 8]`이면 4×8 격자를 뜻한다. 격자로 두는 이유는 logical 축(출력 행 / 출력 열)을
core 축에 대응시키기 위해서다.

### 4.2 함께 쓰이는 세 필드

`#pim.core_mapping`이 `pim_core_shape`와 같은 rank로 짝을 이룬다.

| 필드 | 뜻 | 예 |
| --- | --- | --- |
| `core_axis_roles[d]` | core 축 d가 어느 logical 축을 맡는가 | `["output_row_tile", "none"]` |
| `tile_footprint[d]` | logical tile 하나가 core 축 d를 몇 칸 쓰는가 | `[1, 1]` |
| `lanes` | core 하나 안의 병렬 lane 수 | UPMEM: tasklet 수 / AiMX: 4096 |
| `lane_axis_role` | lane이 tile의 무엇을 나눠 갖는가 | `"whole_tile"` / `"output_row"` |

`PIMFormExecutionWavesPass`가 이 둘로 **wave**를 계산한다.

```cpp
tilesPerWave = Π_d ( pim_core_shape[d] / tile_footprint[d] )
```

`tilesPerWave`는 "PIM Group 하나가 **동시에** 처리하는 logical tile 개수"다.
2 이상이면 그 tile들이 lockstep으로 함께 돌아야 하고, 그러려면 core마다 다른 데이터
영역을 잡아 주고 동기화를 넣어야 한다. 그것이 아직 구현되지 않았다.

### 4.3 왜 1이 아니면 막는가

`PIMMapCoresPass.cpp` 42–57행이다.

```cpp
int64_t cores = 1;
for (int64_t c : coreShape) cores *= c;
if (cores != 1)
  return func->emitError("pim-map-cores: PIM Group with ") << cores
         << " PIM Cores is not mapped yet; only one core per group is implemented";

// core 축에 맡길 logical 차원이 없으므로 역할은 전부 "none" 이고 logical tile 하나가
// core subset 전체를 차지한다.
SmallVector<Attribute> roles(coreShape.size(), b.getStringAttr("none"));
SmallVector<int64_t> footprint(coreShape.size(), 1);
```

이 pass가 지금 구현한 것은 **퇴화된 사상 하나뿐**이다: 모든 core 축의 역할이 `"none"`,
`tile_footprint`가 전부 1. 이 사상은 `Π coreShape = 1`일 때만 정확하다.

`coreShape = [8, 16]`이면 core가 128개인데 footprint가 `[1,1]`이므로
`tilesPerWave = 128`이 되어, "PIM Group 하나가 logical tile 128개를 동시에 돌린다"는
뜻이 된다. 그것을 만들 코드가 없다. **그래서 조용히 127개를 놀리는 대신 진단을 내고
멈춘다.** 즉 이 오류는 의미상의 법칙이 아니라 **미구현 방어벽**이다.

`PIMFormExecutionWavesPass.cpp` 59–62행도 같은 이유로 한 번 더 막는다.

### 4.4 AiMX는 왜 `[1]`인가

`_TARGET_PROFILE["aimx"]`가 지금 `"pim_core_shape": "8,16"`인데, 이것을 그대로 읽으면
**"channel 8개 × bank 16개 = PIM Core 128개이고, 각 core에 서로 다른 logical tile을
배정할 수 있다"** 는 주장이 된다.

그런데 §1.3에서 본 것처럼 `mPIM_RD_INSTR_MC`는 descriptor group의 16칸을 **같은
내용**으로 채우고, bank를 가리키는 필드는 descriptor에 아예 없다. 즉

> **지금 소프트웨어 관례에서** channel과 bank는 서로 다른 logical tile을 배정받지
> 않는다. 같은 명령을 받고, 주소 사상에 따라 **같은 tile의 서로 다른 부분**을 맡는다.

이것이 바로 lane의 정의다. 따라서 **지금 실행되는 것을 기술하면**



```
pim_core_shape                        = [1]      ← PIM Core 1개
number_of_parallel_lanes_per_pim_core = 4096     ← 그 안의 lane 4096개
parallel_execution_kind               = "simd"   ← lane들이 같은 명령을 수행
```

이다. **병렬성이 사라지는 것이 아니라 분류가 바뀌는 것이다** — "따로 명령을 줄 수 있는
것"에서 "함께 움직이는 것"으로.

> **§15 E11로 확정됐다 (2026-08-10).** descriptor group의 record 16개 중 첫 번째만
> 실행된다 — `next` 사슬이 나머지 15개를 건너뛴다 (§1.3). 서로 다른 `src`를 넣은
> group과 모두 같게 쓴 group의 결과가 byte 단위로 동일했다. 그러므로 channel에
> 따로 명령을 줄 수단이 **없고**, `[1]`이 최종 모델이다.
>
> `[16]` / lane 256이 상수와 맞아떨어지는 것(`16 × 256 = 4096`, `PIM_WIDTH = 512`)은
> 여전히 사실이지만, 그것은 **데이터가 흩어지는 방식**이지 명령을 받는 단위가 아니다.
> 바로 그 구분이 lane의 정의다.

> **`PIM_IR_Redesign.md` §2.2의 AiMX 열이 낡았다 — 함께 고쳐야 한다.** 그 표는 아직
> 이렇게 적고 있고, 지금 문서와 정면으로 어긋난다.
>
> | 항목 | `PIM_IR_Redesign.md` §2.2 | 여기서 확정된 값 | 근거 |
> | --- | --- | --- | --- |
> | PIM Device 수 | 8 | **1** (FPGA 하나) | `compiler.py`의 `devices: 1` |
> | `pim_core_shape` | `[8, 16]` = channel × bank | **`[1]`** | §15 E11 |
> | lanes / core | 16 | **4096** | §1.2 |
> | logical tile의 core footprint | `[1, 16]` | **`[1]`** | §15 E11 |
> | 권장 `BLOCK (M,N,K)` | `(16, 16, 16)` | 무관 — target이 block 경계를 쓰지 않는다 (§3.4) | — |
> | 누산 | fp32 | **fp32가 아니다** | §11.1 |
>
> `[8, 16]`은 "서로 다른 logical tile을 배정받을 수 있는 단위의 격자"라는
> `pim_core_shape`의 정의(§4.1)를 만족하지 못한다. descriptor에 주소가 하나뿐이고
> group의 record 16개 중 첫 번째만 실행되기 때문이다. **두 문서가 같은 profile을 다르게
> 적고 있는 상태를 남겨 두면 안 된다.**
>
> 개념적으로는 "명령이 같고 주소만 다르면 SIMD lane"이라는 설명이 부족하다는 지적이
> 옳다 — 주소가 서로 다른 logical tile을 가리킨다면 그것은 같은 명령열을 공유하는
> Group 안의 서로 다른 Core다. **AiMX에서는 그 구분이 성립하지 않는다.** descriptor
> 하나가 주소 하나만 나르므로 "서로 다른 주소"를 줄 대상 자체가 없다 (§1.3).

> **주의: 수가 옮겨 적히는 것이 아니다.** 지금 `"8,16"`이 담은 수는 `8 × 16 = 128`이지
> 4096이 아니다. §1.2가 분해한 대로 lane 4096개는 `8 × 2 × 16 × 16`이고,
> pseudo-channel 2와 `ALU_SIZE` 16은 `pim_core_shape`에 애초에 들어 있지 않았다.
> 즉 `"8,16"`은 lane 수를 32배 적게 적고 있었고, 그것을 `[1]` + `lanes = 4096`으로
> **고치는** 것이다.

이 분류는 요청하신 문장과 정확히 같은 말이다 —
*"각 bank, channel에 따로따로 일을 시키는 것이 아니라, 데이터의 주소값만 다르게
보내는거지."* 명령이 같고 주소만 다른 것이 곧 SIMD lane이다.

### 4.5 대조: UPMEM은 왜 `[1]`인가

UPMEM도 `[1]`이다. 이유는 다르다.

| | UPMEM | AiMX |
| --- | --- | --- |
| PIM Device | DPU 하나 | FPGA 하나 |
| PIM Core | DPU 하나 = core 1개 | 4096 lane 배열 = core 1개 |
| lane | tasklet (독립 thread) | multiply-accumulate lane |
| `parallel_execution_kind` | `"multithread"` | `"simd"` |
| lane마다 버퍼 복제 | **필요** (tasklet이 서로 다른 tile을 맡는다) | 불필요 |

이 차이가 `PIMMapCoresPass` 59–72행과 `PIMVerifyPass` 72–87행에서 실제로 갈린다.
`multithread`는 `lanes`만큼 tile 버퍼를 복제하므로 local 메모리 예산으로 lane 수를
깎지만, `simd`는 복제하지 않으므로 곱하지 않는다.

### 4.6 언제 `[1]`이 아니게 되는가

`pim_core_shape ≠ [1]`이 옳으려면 **자기 명령열을 따로 받는 하위 단위**가 있어야 한다.

AiMX에 후보가 하나 있었다. channel이다. 수가 정확히 맞기 때문이다 — channel 하나가
`NUM_BANKS(16) × ALU_SIZE(16) = 256` lane이고 `16 × 256 = 4096`, `PIM_WIDTH = 512` byte가
channel 하나의 operand 폭이다 (§1.2).

```
pim_core_shape                        = [16]     ← 시도했던 모델
number_of_parallel_lanes_per_pim_core = 256
```

**§15 E11이 이것을 반증했다 (2026-08-10).** descriptor group의 16칸에 서로 다른 `src`
(`S_pa + ch*8192`)를 넣고 `pim_copy`로 돌린 결과가, 16칸을 모두 같게 쓴 경우와 **byte
단위로 동일**했다. 옮겨진 8192 byte는 정확히 record 0이 지정한 구간이었다.

원인은 §1.3에 있다 — chain이 `next` 사슬로 이어져 있고 `next`가 group 단위(512 byte)로
뛰므로 **각 group에서 첫 record만 실행된다.** 나머지 15칸에 무엇을 쓰든 하드웨어가 보지
않는다.

> **그러므로 channel에 따로 명령을 줄 수단이 없다. `[1]`이 최종 모델이다.**
> `PIMMapCoresPass`의 `cores != 1` 방어벽과 `PIMFormExecutionWavesPass`의
> `tilesPerWave != 1` 방어벽은 AiMX 경로에서 도달하지 않는다. §4.7의 "PIM Common IR을
> 건드리지 않는다"가 그대로 성립한다.

**남는 유보 하나.** E11의 vehicle은 `pim_copy`(`MEMCPY_SG`)였다. 그 경로에서는 PIU가
켜지지 않으므로(`mm_set_pim_working_signal`을 건드리지 않는다) CDMA만 chain을 걷는다.
`pim_exec`(`DMA_START`)에서 PIU가 group을 다르게 소비할 가능성은 원리적으로 남아 있다.
다만 그것을 재려면 `OPCODE_PIM_MODE`와 MAC 경로를 써야 해서 위험도가 크게 오르고,
**참조 구현의 자료 배치가 이미 "descriptor 하나 = 8192 byte 전송 하나"와만 앞뒤가 맞는다**
(§3.3). 그래서 이 유보를 근거로 설계를 바꾸지 않는다.

`n < 4096`일 때의 padding 낭비는 이제 §15 E9(`n_t < 4096` legality)로만 다룬다.

### 4.7 필요한 변경

현재 구현은 `pim_backend/backend/target.py`의 공유 target builder를 사용한다.

```python
PIMTargetSpec(
  common_profile=PIMTargetProfile(
    target="aimx", num_units=fpga_count,
    banks_per_unit=256, cores_per_bank=1, lanes_per_core=16,
    bank_capacity_limit_bytes=AIMX_BANK_CAPACITY_BYTES,
    local_capacity_limit_bytes=0),
  runtime_config=AiMXRuntimeConfig(batch=batch, base_fpga=base_fpga))
```

`num_units`는 `/dev/pl_dma*`에서 보이는 FPGA 개수이거나 명시적으로 전달한 값이다.
`[p_m,p_n,p_k]`의 곱은 이 수 이하여야 한다.

AiMX capacity는 UPMEM 기본값을 공유하지 않는다. target builder가
runtime heap 사양에서 bank당 `bank_capacity_limit_bytes`를 계산해 Common
profile에 기록한다. §1.8 (h)처럼 `pim_malloc`은 한계를 넘어도 NULL을
반환하지 않으므로, 할당 실험으로 capacity를 유추하지 않는다.

AiMX 물리 실행 눈금 `[batch,4096/batch,256/batch]`는 Common profile에 추가하지
않고 AiMX lowering이 `batch`로 padding한다.

### 4.8 이 값으로 공통 pipeline이 끝까지 도는지

| pass | `coreShape = [1]`, `simd`, `local = 0`에서 | 결과 |
| --- | --- | --- |
| `--pim-map-cores` | `cores = 1` ✔ / `simd`라 lane 예산 검사 건너뜀 | 통과 |
| `--pim-form-execution-waves` | `footprint = [1]` / `coreShape = [1]` → `tilesPerWave = 1` ✔ | 통과 |
| `--pim-map-devices` | target 무관 | 통과 |
| `--pim-place-data` | target 무관 | 통과 |
| `--pim-verify` `verifyCores` | `local.getInt() == 0` → 즉시 advance | 통과 |
| `--pim-verify` `verifyDevice` | `pad`가 0 (`other=0.0`) ✔ | 통과 |

**결론: PIM Common IR을 수정하지 않고 AiMX IR 생성·descriptor 생성·실행이 가능하다.**

---

## 5. PIM Common IR에서 얻는 값

`--convert-pim-to-aimx`가 필요로 하는 값은 PIM Common IR과 공유 target profile에
이미 있다.

| 필요한 것 | 어디서 | §3의 어느 단계 |
| --- | --- | --- |
| PIM Device 개수와 축별 분할 | 최종 `pim.partition`의 `#pim.device_mapping` | ① |
| device 하나가 맡는 직사각형 | `pim.place`의 `allocation_shape` | ①→② |
| 실제 유효 범위 (경계 tile) | `pim.place`의 `size` | padding 결정 |
| host buffer 안의 위치와 간격 | `pim.place`의 `start`, `strides` | stage/unstage |
| A / B / C 구분 | `pim.place.direction` + `pim.access.id` ↔ `linalg.matmul` operand | ③ |
| reduction 축 | `linalg.matmul`의 고정 indexing map (A 축 1, B 축 0) | k |
| logical tile 모양 | `pim.access.sizes` | 진단용 |
| 같은 데이터를 받는 device | `pim.place.shared_over` | stage 중복 제거 |
| padding 값 | `pim.access.pad` (`pim-verify`가 0임을 보장) | §6 |
| 기계 예산 | `#pim.target_profile.bank_capacity_limit_bytes` | ② |

UPMEM target이 쓰는 것과 정확히 같은 집합이다. 즉 **PIM Common IR은 이미 충분히
target-neutral하다.**

profile에 채워 넣어야 하는 값:

| 필드 | 지금 | 되어야 하는 값 |
| --- | --- | --- |
| `num_units` | FPGA 개수 |
| `banks_per_unit` | 256 |
| `cores_per_bank` | 1 |
| `lanes_per_core` | 16 |
| `bank_capacity_limit_bytes` | FPGA heap을 256 bank로 나눈 상한 |
| `local_capacity_limit_bytes` | 0 |

FPGA 전체 unit 상한은 `banks_per_unit * bank_capacity_limit_bytes`로 파생한다.
`pim.place.allocation_shape`는 원소 단위의 논리 직사각형이고 byte capacity와 다른
개념이다.

---

## 6. Execution tile 선택과 padding

### 6.1 알고리즘

**출발점은 전송량 식이다.** host loop를 n 바깥 / m 안쪽으로 두었을 때 launch 한 번의
host↔PIM 전송량은 이렇게 된다.

```
A 전송  =  ⌈N/n_t⌉ × ⌈M/m_t⌉ × m_t × padded_k × 2
B 전송  =  ⌈N/n_t⌉ × k_t × n_t × 2
C 전송  =  M × ⌈N/n_t⌉ × n_t × 2
chain   =  512 × [ 2(k_t/256)·m_t  +  m_t·(n_t/4096)·(2·k_t/256 + 1) ]     ← launch 당 1회
```

**padding을 뺀 근사식을 쓰지 않는다.** 예전 판은 `⌈M/m_t⌉·m_t ≈ M`, `⌈N/n_t⌉·n_t ≈ N`으로
줄여 적었는데, 그러면 마지막 조각의 padding이 사라져 `m_t`·`n_t`의 효과를 잘못 읽는다.
`C`가 `m_t`에 무관한 것은 `aimx.unstage`가 `pad_to`가 아니라 **실제 행 수**만 되읽기 때문이다.

> **2026-08-11 — `m_t`를 고르는 방향이 뒤집혔다.** 예전에는 backend가 `tile-m` 크기를
> 요청하고 pass가 그것을 `lcm(batch, BLOCK_M)` 경계로 올렸다. 그러면 `m_t`가 언제나
> `BLOCK_M`의 배수로 튀어 `BLOCK_M=8`에서 40, 48, 96 같은 값을 만들 수 없었다.
> 지금은 **묶음 수가 입력**이다.
>
> ```
> 옛:  tile-m 요청 → BLOCK_M 경계로 올림 → m_t → group_M = m_t/BLOCK_M  (결과)
> 지금: BLOCK_M 읽기 → group-m 선택 (입력) → m_t = BLOCK_M * group-m
>
>   BLOCK_M=8,  group-m=5  →  [8][8][8][8][8]          = m_t 40
>   BLOCK_M=16, group-m=3  →  [--16--][--16--][--16--] = m_t 48
> ```
>
> `#aimx.tile_group`의 `group[0]`이 곧 `group-m`이 되어 attribute가 계산 결과가 아니라
> **입력의 기록**이 된다. 제약은 하나뿐이다: `(BLOCK_M * group-m) % batch == 0`.
> `batch`가 자동이면 이 조건도 `batch` 선택에 함께 들어가므로(§2.5) 거절되는 조합이 없다.
>
> **n·k 방향은 묶음 수가 입력이 될 수 없다.** 그쪽 눈금은 우리가 고르는 값이 아니라
> 하드웨어 상수(`4096/batch`, `256/batch`)이기 때문이다. 대신 그 상수를 `BLOCK_N`·
> `BLOCK_K`의 정수 배로 **만드는** 것이 `batch` 선택이고, 그 결과가 `group[1]`, `group[2]`다.

여기서 세 가지가 읽힌다.

| | |
| --- | --- |
| **`k_t = k`, `n_t = N`, `m_t \| M`이면 A·B·C가 각각 정확히 한 번씩만 옮겨진다** | 이론적 최소값이고, 참조 구현이 하는 일이 이것이다 |
| **`m_t`는 A 의 `⌈M/m_t⌉·m_t` 항에만 나타난다** | 그 항은 `M`을 `m_t`로 올림한 값이므로 `m_t`가 **작을수록 작아진다**. chain도 `m_t`에 비례한다. 그래서 **두 방향 모두 작은 쪽이 유리하다** |
| **`n_t`는 A 전송량에 반비례한다** | 클수록 좋고, 대가는 B buffer 크기다 |

> `m_t`에 대한 반대 방향의 지적이 있었다 — "마지막 `M` 조각의 padding 때문에 `m_t`가
> 작을수록 좋다는 결론을 완화해야 한다". 식이 그 반대를 말한다. `⌈M/m_t⌉·m_t`는 `M`을
> `m_t`의 배수로 올린 값이므로 `m_t`가 커질수록 padding이 **늘어난다**. `m_t = 16`,
> `M = 1024`이면 padding이 0이고, `m_t = 100`이면 76행이 낭비된다.

**그러나 `k_max`와 `n_max`를 "동시에 최대화"하는 것은 정의되지 않는다.** 둘은 같은 예산을
나눠 쓰는 trade-off이고 Pareto 후보가 여럿이라 단일한 최대값이 없다. 그리고 그 둘을
가를 정보가 **compile time에 없다** — `k`도 `N`도 런타임 값이기 때문이다 (§17.2).

> **그래서 `k_max`와 `n_max`는 선언된 option이다.** 예산은 "이 조합이 들어가는가"만
> 검사한다. `tile-k`가 `0`일 때만 `n_max`를 고정한 뒤 `k_max`를 이분 탐색으로 최대화한다 —
> 그것은 한 축을 고정했으므로 정의되는 문제다.

```
입력
  budget = 이 PIM Device 의 PIM DRAM 예산 (§5 의 이름 주의를 볼 것)
  k, N   = pim.place 에서 오는 런타임 값

0. m_dev > n_dev 이면 전치한다 (§2.6). 이하 전치 후 좌표로 진행한다
1. batch  ← 1 (§2.5)
2. m_t    ← 16 (§12: m 은 16 에서 포화하고, 위 식에서 전송량에 나타나지 않는다)
3. buffer 상한을 정한다 — **컴파일 타임 상수**
     n_max ← tile-n 을 4096 의 배수로 올린 값 (기본 4096)
     k_max ← tile-k 가 양수면 그 값을 256 의 배수로 올린 값 (기본 32768),
             0 이면 n_max 를 고정한 채 예산에 들어가는 최대값을 이분 탐색
     footprint(m_t, k_max, n_max) ≤ budget 을 확인한다. 넘으면 진단을 내고 멈춘다
4. 실제 tile 모양은 **런타임 값**이다 — descriptor program 이 인자로 받는다 (§17.2)
     k_t ← 256  * ceil(k / 256)
     n_t ← 4096 * ceil(N / 4096)      단  n_t ← min(n_t, n_max)
     padded_k ← 4096 * ceil(k_t / 4096)
     padded_n ← n_t
5. k_t > k_max 이면 부분합 합산 경로가 필요하다 — §8 I4 또는 host split-k.
   지금은 생성된 stub 이 실행 시점에 진단을 내고 멈춘다
6. n_t 는 4096 아래로 **절대 줄이지 않는다**. output block 이 덜 차서 lane 이 놀고,
   그 legality 자체가 실측되지 않았다 (§15 E9)
```

**`m_t`를 키우지 않는 이유.** 위 식에서 `m_t`가 나타나는 곳은 chain 하나뿐이고, chain은
§7 때문에 launch당 한 번만 만들어진다. `m_t`를 키우면 chain이 그만큼 커지고 재사용
횟수가 줄어든다 — 즉 **`m_t`를 키우는 것은 §7의 이득을 그대로 반납하는 것**이다.
남는 반대 항은 invocation당 `ioctl` 고정비(`M/m_t × ⌈N/n_t⌉`회)뿐인데 µs 규모다.

**loop 순서 — n이 바깥, m이 안쪽.** §3.1 ②는 한때 그 반대(m 바깥)를 적었고 근거는
"A 재배치 결과를 그 m 대역의 모든 n slice가 공유한다"였다. **그 공유는 §8 I1(chain을
재배치용과 계산용 둘로 분리)이 구현되어야 생긴다.** 지금은 chain 하나에 재배치와 계산이
붙어 있어 `invoke`마다 재배치가 다시 도므로 공유할 것이 없고, 그 순서는 가장 큰 buffer
(B)를 m tile 수만큼 반복해서 올리는 비용만 남긴다. I1이 구현되면 이 절을 다시 판단한다.

### 6.2 PIM DRAM 비용 모델

| buffer | byte |
| --- | --- |
| A block | `m_t * padded_k * 2` |
| Arel | `m_t * k_t * 32` ← 16배 복제 (§1.6). host가 직접 만들 수 있다 (§8 I5) |
| B block | `k_t * padded_n * 2` |
| C block | `m_t * padded_n * 2` |
| descriptor chain | `4*m_t*k_t/batch + m_t*n_t*k_t*batch/1024 + m_t*n_t/8` |
| PMM (vACC 등) | `0x30000` = 196,608 (256 KB 정렬) |

descriptor chain은 §2.4의 `G`에 512를 곱한 것이다.

**B와 C는 `n_t`가 아니라 `padded_n`으로 잡는다** (§1.8 (c)(e)). `n_t = 4096` 고정에서는
둘이 같지만, 식에는 pitch 쪽을 적어 두는 것이 맞다.

**이 예산은 compile time에 확정되어야 한다.** 할당해 보고 실패하면 더 작은 tile로
물러서는 방식은 성립하지 않는다 — `pim_malloc`은 실패해도 NULL을 돌려주지 않고
프로세스를 죽인다 (§1.8 (h)).

### 6.3 Padding — 작은 tile이 실패하지 않게

§2.4의 배수 조건을 어기면 오류가 아니라 조용한 오답이므로, **padding은 성능
최적화가 아니라 정확성 요구사항이다.**

| 위치 | 채우는 값 | 읽히는가 | 이유 |
| --- | --- | --- | --- |
| A의 `[m_dev, m_t)` 행 | 0 | 읽힌다 | 대응 C 행은 버려진다. 결정성 위해 0 |
| A의 `[k, k_t)` 열 | **0 필수** | 읽힌다 | 누산에 더해진다 |
| A의 `[k_t, padded_k)` 열 | 0 | **안 읽힌다** | 행 간격만 필요 |
| B의 `[k, k_t)` 행 | **0 필수** | 읽힌다 | 누산에 더해진다 |
| B의 `[n_dev, n_t)` 열 | 0 | 읽힌다 | 대응 C 열은 버려진다 |
| C의 padding 영역 | — | 쓰인다 | `unstage`가 `size`만 되돌린다 |

padding은 host의 `aimx.stage`에서 이루어진다. 그래서 §13.2의 `M=N=K=32` 같은 극단적
경우도 **실패하지 않고 정확한 답을 낸다** (유효율 0.78 %).

---

## 7. Descriptor program 재사용

§6.1이 모든 execution tile을 같은 모양으로 만들었다. 여기에 하나만 더하면 된다.

> **PIM DRAM block(A, Arel, B, C, descriptor chain, PMM)을 execution tile 하나
> 크기로 한 벌만 잡고, tile마다 그 같은 주소에 다시 stage한다.**

지키지 않으면 조용히 깨지는 규칙이 둘 붙는다.

> **규칙 1 — 주소 겹침 금지.** PMM을 **가장 먼저** 잡고, 나머지 block은
> `[pmm_pa, pmm_pa + 0x30000)`을 피해 배치한다. `pim_private_malloc`이 `pim_malloc`이
> 이미 내준 물리 주소를 그대로 돌려줄 수 있고 (§1.8 (g)), vACC가 `pmm_pa + 0x10000`
> 이므로 겹치면 계산 도중 데이터가 덮어써진다. runtime은 할당 후 겹침을 검사해
> 겹치면 즉시 진단을 낸다.
>
> **규칙 2 — chain은 자기 주소를 싣는다.** descriptor의 `next` 필드는 descriptor buffer
> 자신의 물리 주소다. chain이 byte 단위로 같으려면 descriptor buffer를 launch 내내
> free하지 않고 **같은 물리 주소로 유지**해야 한다. `desc_last`도 마지막 group의 시작
> 주소이므로 함께 고정된다 (§9.6).

descriptor의 주소는 전부 `base + offset(loop index)` 꼴이고 `offset`은
`(m_t, k_t, n_t, batch, padded_k)`만의 함수다. tile 모양이 같고 `base`도 같으므로
**descriptor chain이 모든 tile에서 byte 단위로 동일하다.**

| | naive | 이 설계 |
| --- | --- | --- |
| chain 생성 | invocation마다 | **launch당 한 번** |
| 주소 patching | invocation마다 record 전체 | **없음** |
| descriptor memory 최대 | Σ tile | **tile 1개분** |

**실측 근거.** 참조 구현은 호출 시간의 **52~63%를 hardware 밖**에서 쓴다 — 호출마다
하는 `pim_private_malloc`/`pim_malloc`, descriptor 재생성, `datacopy_cpu2pim`,
`pim_free`다. hardware가 실제로 도는 시간은 37~48%뿐이다 (`k=4096, n=4096`에서 m을
1부터 128까지 바꿔 가며 측정했고 비율이 형상과 무관하게 일정했다). 이 설계가 없애는
것이 바로 그 부분이고 **이론상 상한은 2.4~2.7배**다.

대가는 tile t의 stage와 tile t−1의 실행이 겹치지 않는 것이다.

> **buffer를 두 벌 두는 것만으로는 겹치지 않는다.** `aimx.invoke`는 `pim_exec`가
> 끝날 때까지 기다리는 **동기 op**이고 (`AiMXDialect.td`, `dma_fops_pl.c`의 `DMA_START`가
> PDMA 상태 레지스터를 polling한다), 생성된 stub도 한 thread다. 실제로 겹치려면
> 다음 중 하나가 더 필요하다.
>
> - `aimx.invoke`를 비동기로 나누고 완료를 기다리는 op를 따로 둔다 (driver에 그런
>   경로가 있는지 미확인)
> - stage를 별도 host thread에서 돌린다
> - PIM Device를 둘 이상 써서 서로 겹친다
>
> 지금은 셋 다 없으므로 **double buffering은 아직 설계에 들어 있지 않다.**

---

## 8. 참조 구현보다 나은 명령열

`_matmul_silent`의 API가 아니라 ISA에서 출발했으므로 명령열을 다시 짤 수 있다.
네 가지 개선점이 있고, 앞의 둘은 지금 바로 적용 가능하며 뒤의 둘은 확인이 필요하다.

### I1. A 재배치를 n loop 밖으로 뺀다 — **지금 적용 가능**

참조 구현은 호출마다 A 재배치 phase를 다시 돈다. §3.1 ②의 같은 행 tile들은 A 재배치
결과가 동일하므로 한 번만 하면 된다. descriptor program을 두 개(재배치 / 계산)로
나누고 `aimx.build_descriptors`를 두 번 부른다.

gemm.py 기준: n slice 4개면 `3 × m_t × k_t × 32` = 176 MB의 중복 내부 전송이 사라진다.

### I2. A 재배치 자체를 없앤다 — **확인 필요, 효과 가장 큼**

계산 phase의 A descriptor는 `A_MASK | MATMUL | PIM_MODE`이고, 재배치 phase는
`BC | TP`다. 두 flag 집합이 겹치지 않으므로 **하나로 합칠 수 있는지**가 문제다.

```
A_MASK | MATMUL | PIM_MODE | BC | TP | STRIDE
src = A block + (k_off*2 + m_off*padded_k*2),  src_pitch = padded_k*2,  len = 512
```

성공하면 사라지는 것:

- `Arel` buffer `m_t * k_t * 32` byte — **§6.2에서 가장 큰 항목**
- 재배치 descriptor `4 * m_t * k_t / batch` byte — `batch = 1`이면 chain의 절반

`gemm.py`의 `(m_t, k_t) = (64, 28672)`에서 58.7 MB + 7.3 MB가 없어진다.

### I3. 재배치의 두 descriptor를 하나로 합친다 — **확인됨**

참조 구현은 `(m-group, k-chunk)`마다 두 개를 쓴다: `src = A, dst = 0`(읽기)과
`src = 0, dst = Arel`(쓰기). 하나로 합칠 수 있다.

```
BC | TP | STRIDE | batch
src = A block + …,  src_pitch = padded_k*2
dst = Arel   + …,  len = 512 * batch * NUM_CHANNELS      ← 출력 byte 수 (§1.7)
```

**`len`이 요점이다.** 입력 크기 512를 주면 512 byte 덩어리 하나만 쓰이고 나머지 15개는
이전 내용이 남는다. 출력 크기를 주면 참조판과 **내용이 같다.**

남은 차이는 하나다. 합친 결과는 512 byte 덩어리가 정확히 한 칸 순환한다.

```
Arel_merged[c] = Arel_ref[(c + 1) mod 16]
```

읽는 위치가 밀린 것이 아니라 512 byte 블록 안에서 32 byte만큼 순환한 것이다 — 덩어리
15가 패딩이 아니라 `A[0..15]`로 되돌아온다. 보정은 계산 phase의 B 주소를 256-k 블록
안에서 16행 순환시키는 것이고, 이 보정만 아직 실측하지 않았다.

재배치 descriptor가 절반이 된다 (§2.4의 `G`에서 첫 항의 계수 2가 1이 된다).
I5를 택하면 이 항목 자체가 사라진다.

### I4. reduction 분할 — **확인 필요, 구조가 바뀐다**

ISA에는 **memory → vACC 경로가 있다.** `_gemm_silent`의 bias descriptor가 그 증거다
(`src = Bias_pa`, `dst = mm_vACC_pa`, flags = `STRIDE | batch`). 그 함수가 그 경로를
열 vector broadcast로만 쓰는 것은 소프트웨어의 선택이지 하드웨어의 제약이 아니다.

C drain의 정확한 거울상을 시도한다.

```
restore:  src = C partial | ((padded_n*2) << 33)
          dst = vACC      | ((8192/batch) << 33)
          len = 8192/batch,   flags = STRIDE | batch      (행 수 = batch)
```

성공하면:

- reduction을 execution tile 사이로 나눌 수 있다 → §6.1 단계 6의 막다른 길이 사라진다
- `Arel`과 descriptor chain이 `k_t`에 비례하는데 `k_t`를 줄일 수 있게 된다
- k 방향 pipelining이 가능해져 §12의 tuning 공간이 넓어진다

실패하면 이 문서 그대로다. reduction은 execution tile 안에서 완결되어야 하고,
그 대신 host에서 부분합을 더하는 경로(split_k.py가 Triton 수준에서 하는 것)를
backend가 만든다.

`OPCODE_HOLD`(bit 14)는 matmul 경로에서만 안 쓰일 뿐 `memcpy.c`에서는 단독 opcode로
실제로 쓰인다. 이름으로 보아 "vACC를 비우지 않는다"일 가능성이 있으나, `memcpy`의
용법(고정 dst로 여러 입력을 모았다가 `TP`로 한 번에 내보낸다)은 그 해석과 잘 맞지
않는다. **의미 확인이 먼저다.** 확인되어 누산 유지가 맞으면 restore 없이도 C
descriptor에 `HOLD`를 붙여 여러 chain에 걸쳐 누산을 이어 갈 수 있다.

### I5. A 재배치를 host에서 만든다 — **지금 적용 가능**

§1.6에서 재배치 순열이 실측으로 확정됐다.

```
Arel[c*256 + g*16 + t] = A[16*c + t]
```

**A만의 함수다.** 그러면 `aimx.stage`가 A를 PIM DRAM에 올릴 때 처음부터 이 모양으로
써 넣으면 되고, 재배치 descriptor가 통째로 사라진다 — I1·I2·I3이 다루던 문제 자체가
없어진다.

| | 없어지는 것 | 늘어나는 것 |
| --- | --- | --- |
| descriptor | `2 * m_t * (k_t/256)` group과 그 실행 시간 | 없음 |
| PCIe 전송 | 없음 | A가 16배 (`m_t*k_t*32` byte) |
| PIM DRAM | 없음 | 없음 — `Arel`은 어차피 §6.2에 있다 |

`m_t = 16, k_t = 4096`이면 A 전송이 128 KB에서 2 MB로 는다. 같은 tile의 B가 32 MB이므로
무시할 수준이다. 반대로 없어지는 descriptor는 `G`의 첫 항 전체이고, descriptor 생성이
전체 시간의 52~63%를 차지하므로 (§7) 그대로 성능이 된다.

**전제는 §1.6의 순열이 `k > 256`과 `m > batch`로도 그대로 확장되는가 하나다.** 실측은
`m=1, k=256, batch=1`에서 했다. 참조 구현의 주소식이 `q`를 `REG_SIZE = 256`씩,
`p`를 `batch`씩 도는 단순 반복이므로 확장은 자연스럽지만, 확인 항목으로 남긴다 (§15 E10).

---

## 9. AiMX Target IR

UPMEM dialect와 같은 계약을 따른다.

> codegen은 이 IR을 그대로 옮긴다. op의 파라미터가 나올 코드를 유일하게 정해야 하며,
> codegen이 스스로 정하는 것이 있으면 그 결정은 여기 없는 것이다.

AiMX가 UPMEM과 구조적으로 다른 점: **AiMX에는 device program이 없다.** instruction
fetch를 하는 processor가 없고 descriptor chain 자체가 프로그램이다. 그래서
`upmem.device_func`에 대응하는 것이 없고, 대신 descriptor를 IR 안에서 직접 표현하는
두 번째 층이 생긴다.

### 9.1 두 층

```
Placed PIM Common IR
   │  --convert-pim-to-aimx     결정: 전치 여부, batch, execution tile, padding, 분배
   ▼
AiMX IR  level 1 (placement)    aimx.acquire / alloc / stage / build_descriptors
   │                            / invoke / unstage
   │  --aimx-lower-matmul       결정: descriptor 순서와 주소식
   ▼
AiMX IR  level 2 (descriptor)   aimx.descriptor_program + aimx.desc + scf.for
   │  triton-shared-translate   결정: 없음
   ▼
{"stub": stub.cpp, "descriptors": program.h}
```

두 층으로 나누는 이유는 §7이다. level 1은 tile마다 반복되고 level 2는 launch당 한 번만
펼쳐진다. 이 구분이 IR에 보이지 않으면 재사용이 codegen이 몰래 하는 결정이 되어
계약을 어긴다.

### 9.2 Type

| | 뜻 |
| --- | --- |
| `!aimx.device_set` | 확보한 FPGA 집합 |
| `!aimx.device` | 집합 안의 FPGA 하나. `aimx.foreach_device` 몸체에서만 생긴다 |
| `!aimx.buffer` | PIM DRAM 할당 하나. **원소 타입이 없다** — descriptor 주소가 전부 byte 단위이고, 원소 타입은 그 buffer를 채우는 `aimx.stage`/`aimx.unstage`가 들고 있다 |

descriptor chain도 `!aimx.buffer`다 (`kind = descriptor`). chain이 자기 물리 주소를
record에 싣기 때문에 다른 할당과 같은 것으로 다뤄야 규칙 2(§7)가 IR에 드러난다.

### 9.3 Level 1 — host op

| op | 하는 일 |
| --- | --- |
| `aimx.acquire n : !aimx.device_set` | FPGA 확보 (런타임이 쥐고 있던 것을 준다) |
| `aimx.foreach_device %set axes [...] counts [...]` | device 좌표 loop |
| `aimx.alloc %dev bytes N align A kind <data｜descriptor｜pmm>` | PIM DRAM 할당. `bytes`는 compile time 상수다 (§1.8 (h)) |
| `aimx.stage %src origin[] sizes[] strides[] pad_to[] pitch %p into %buf {element_type}` | host gather + 원소 타입 변환 + 0 채우기 + PIM DRAM 적재. `transpose`는 표시 attribute다 |
| `aimx.build_descriptors @prog(%m_t,%k_t,%padded_k,%padded_n, %a,%arel,%b,%c,%pmm) into %chain` | descriptor program을 chain에 펼친다 |
| `aimx.invoke %dev, %chain` | chain 실행 |
| `aimx.unstage %buf into %dst origin[] sizes[] strides[] pitch %p {element_type}` | `stage`의 역 |

`aimx.alloc`에 붙는 규칙이 둘 있다.

> **(a) `kind = pmm`인 alloc은 program 안에서 가장 먼저 나와야 하고**, codegen은 그것을
> 다른 어떤 `pim_malloc`보다 먼저 emit해야 한다. `pim_private_malloc`이 이미 나눠 준
> 주소를 다시 줄 수 있으므로 (§1.8 (g)) 데이터 buffer가 `[pmm_pa, pmm_pa + 0x30000)`과
> 겹치면 안 된다. codegen은 stub에 겹침 검사도 함께 낸다.
>
> **(b) `pim_malloc`은 실패해도 NULL을 돌려주지 않는다** (§1.8 (h)). 따라서 §6.2의 PIM
> DRAM 예산은 compile time에 확정되어야 하고, 할당 실패를 감지해 더 작은 execution
> tile로 물러서는 설계는 성립하지 않는다.
>
> **(c) `align`은 요구이지 요청이 아니다.** `pim_malloc`은 정렬을 받지 않는다. codegen은
> 참조 구현이 쓰는 호출(`pim_private_malloc(PMM_SIZE, fpga, 7)` / `pim_malloc(bytes, fpga)`)을
> 그대로 내고, 받은 물리 주소가 `align`의 배수인지 **실행 시점에 검사**한다. 짐작하지 않고
> 확인한다.

`upmem.stage`와 다른 점 세 가지.

1. `pitch`를 받는다. AiMX는 descriptor의 stride 필드로 행 간격을 읽으므로 PIM DRAM
   안의 layout이 `pitch`로 규정된 2차원 배열이다 (UPMEM은 tile-major 연속 배열).
   A는 `padded_k`, B와 C는 `padded_n`이고 B와 C는 서로 같아야 한다 (§1.8).
   참조 구현이 `pim_align_chunk()`로 손수 하는 일이 바로 이것이다.
2. 원소 타입을 변환한다. host가 f32이고 PIM이 bf16이다 (§11).
3. `transpose`를 받는다. §2.6의 전치가 여기서 이루어진다.

`aimx.build_descriptors`를 tile loop **밖**에 두는 것이 §7의 재사용을 IR에 드러내는
방식이다.

### 9.4 Level 2 — descriptor op

주소식은 `arith`로, loop는 `scf.for`로 쓴다. 새 op은 **하나뿐**이다.

```
aimx.desc src(%buf, %off) [pitch %sp] dst(%buf, %off) [pitch %dp]
          len %n opcode <flags> batch %b
```

`aimx.desc` 하나 = descriptor group 하나 = record 16개. 1:1 대응이므로 codegen이 정할
것이 남지 않는다. `flags`는 §1.7의 enum이다.

`--convert-pim-to-aimx`가 §3.2의 loop 순서를 그대로 낸다.

`batch`는 block argument가 아니라 **program attribute**다. §9.5의 "program 하나 안에서
`batch`가 하나여야 한다"와 §7의 byte 단위 동일성이 둘 다 이때만 성립한다. 그래서
`aimx.desc`는 `batch`를 따로 들지 않고 감싸는 program에서 읽는다 — 두 곳에 두면 어긋날 수
있다. 같은 이유로 status word의 batch 비트(15/16/17)는 flag enum에 **없다.** codegen이
program attribute 하나에서 유일하게 유도한다.

`len`도 operand가 아니라 attribute다. 세 가지 값(`512`, `8192*batch`, `8192/batch`)이 전부
compile time 상수이고, attribute여야 §9.5의 검사가 언제나 성립한다.

`padded_n`도 `n_t`와 따로 받는다 — descriptor가 실제로 싣는 것은 `padded_n`이다 (§1.8).

`src`와 `dst`는 각각 없을 수 있다. 참조 구현이 쓰지 않는 쪽에 `0x0`을 넣는 자리이고, 그
방향은 PIM logic 내부(vecA / vACC)라 주소가 없다.

주소는 전부 byte이고 참조 구현의 offset 식 그대로다.

```mlir
// 재배치. §8 I5를 택하면 이 program 자체가 없어진다.
aimx.descriptor_program @gemm_relayout attributes {batch = 1 : i32} {
^bb0(%m_t: index, %k_t: index, %padded_k: index,
     %a: !aimx.buffer, %arel: !aimx.buffer):
  scf.for %mg = 0 to %m_t step batch {
    scf.for %s = 0 to %k_t step 256 {
      aimx.desc src(%a, %s*2 + %mg*%padded_k*2) pitch (%padded_k*2) len 512
                opcode <pim_mode, stride> batch batch
      aimx.desc dst(%arel, 8192*batch*(%s/256) + %mg*%k_t*32) len (8192*batch)
                opcode <bc, tp> batch batch
    }
  }
}

// 계산
aimx.descriptor_program @gemm_compute attributes {batch = 1 : i32} {
^bb0(%m_t: index, %k_t: index, %padded_n: index,
     %arel: !aimx.buffer, %b: !aimx.buffer, %c: !aimx.buffer, %pmm: !aimx.buffer):
  scf.for %mg = 0 to %m_t step batch {                        // output block 행
    scf.for %r = 0 to (%padded_n*batch/4096) step 1 {         // output block 열
      scf.for %s = 0 to %k_t step (256/batch) {               // reduction step
        aimx.desc src(%arel, 8192*(%s/(256/batch)) + %mg*%k_t*32) len 8192
                  opcode <a_mask, matmul, pim_mode> batch batch
        aimx.desc src(%b, %r*(8192/batch) + %s*%padded_n*2) pitch (%padded_n*2)
                  len (8192/batch)
                  opcode <b_mask, matmul, pim_mode, stride> batch batch
      }
      aimx.desc src(%pmm, VACC_OFFSET) pitch (8192/batch)
                dst(%c, %r*(8192/batch) + %mg*%padded_n*2) pitch (%padded_n*2)
                len (8192/batch)
                opcode <c_mask, matmul, stride> batch batch
    }
  }
}
```

`pitch`는 IR에서는 별도 operand로 쓰지만 codegen이 `addr | (pitch << 33)`으로 합친다
(§1.1). `32 = TYPE_SIZE * NUM_CHANNELS`이고 `8192 = PIM_WIDTH * NUM_CHANNELS`다.

### 9.5 `aimx.desc` verifier

**허용 조합을 whitelist로 둔다.** 확인된 다섯 가지가 기본이고, §8의 개선안은 확인된
뒤에 항목을 추가한다.

| # | flags | len | pitch | 역할 | 상태 |
| --- | --- | --- | --- | --- | --- |
| 1 | `pim_mode｜stride｜batch` | 512 | src | A gather | 확인 |
| 2 | `bc｜tp｜batch` | `8192*batch` | 없음 | Arel write | 확인 |
| 3 | `a_mask｜matmul｜pim_mode｜batch` | 8192 | 없음 | A operand | 확인 |
| 4 | `b_mask｜matmul｜pim_mode｜stride｜batch` | `8192/batch` | src | B operand | 확인 |
| 5 | `c_mask｜matmul｜stride｜batch` | `8192/batch` | src+dst | C drain | 확인 |
| 6 | `bc｜tp｜stride｜batch` | `8192*batch` | src+dst | 재배치 1회 (§8 I3) | **확인** |
| 7 | `a_mask｜matmul｜pim_mode｜bc｜tp｜stride｜batch` | `8192*batch` | src | 재배치 융합 (§8 I2) | **확인 필요** |
| 8 | `stride｜batch` (dst = vACC) | `8192/batch` | src+dst | vACC restore (§8 I4) | **확인 필요** |

6행 각주: 합친 결과는 512 byte 덩어리가 한 칸 순환한다
(`Arel_merged[c] = Arel_ref[(c+1) mod 16]`, §8 I3). 보정은 계산 phase의 B 주소를
256-k 블록 안에서 16행 순환시키는 것이며 아직 실측하지 않았다.

추가 검사:

- **`len`은 출력 byte 수다.** 입력 크기를 주면 그만큼만 쓰이고 나머지는 이전 내용이
  남는다 (§1.7). 재배치 계열은 `512 * batch * NUM_CHANNELS`, B·C 계열은 `8192/batch`다
- `stride` flag가 있으면 pitch operand가 있어야 하고, 없으면 없어야 한다
- `len`이 `BURST_SIZE` = 32의 배수여야 한다
- `batch ∈ {1,2,4,8}`이고 program attribute이므로 program 하나 안에서 하나뿐이다 (§2.5)
- pitch는 `padded_k`(A) 또는 `padded_n`(B·C)에서 와야 한다. `n_t`가 아니다 (§1.8)
- `src`가 있으면 그 offset이 있어야 하고 없으면 없어야 한다. `dst`도 같다.
  pitch는 같은 쪽 주소가 있을 때만 붙는다
- 주소가 양쪽 다 없는 descriptor는 없다

이 verifier가 있으면 descriptor 생성이 확인된 ISA에서 벗어나는 순간 compile time에
잡힌다. 하드웨어에서 조용히 틀리는 것보다 낫다.

### 9.6 codegen

| translate | 내는 것 |
| --- | --- |
| `--aimx-to-descriptors` | level 2를 C 표로. **chain이 아니라 program이다** — loop nest라 크기가 작다 |
| `--aimx-to-runtime-stub` | `extern "C" void <name>(...)` — acquire / alloc / stage / build / invoke / unstage |

`compiler.py`의 `_aimx_to_artifact`가 stub에서 `extern "C" void (\w+)\(`를 찾아
`metadata["name"]`을 채우는 규약을 그대로 지킨다. descriptor 쪽을 텍스트로 두므로
`_run_translate`의 utf-8 decode도 그대로 쓴다.

**참조 구현의 함수(`_matmul_silent` 등)는 호출하지 않는다.** stub은 하위 런타임 API만
쓰고 descriptor는 level 2에서 직접 낸다. 쓰는 API는 다음이 전부다.

```
init_pim_drv
pim_private_malloc(PMM_SIZE, fpga_id, 7)     ← 가장 먼저 (§9.3 (a))
pim_malloc / pim_free
VA2PA
mmap / munmap                                 ← host 조립 buffer
datacopy_cpu2pim / datacopy_pim2cpu
pim_exec
```

**chain 연결 규약**은 다음으로 고정한다 (참조 구현과 동일).

- record는 `0x20` byte이고 `status`에 §1.7의 flag word가 들어간다
- `aimx.desc` 하나는 **같은 내용의 record `NUM_CHANNELS`개**로 펼쳐진다
- 16개 모두 같은 `next`를 담고, 그 `next`는 **다음** group의 물리 주소다
- chain의 시작은 `desc_base_pa + 0x20 * NUM_CHANNELS`부터 센다
- 다 펼친 뒤 `desc_idx = 총 record 수 - 1`,
  `desc_last = next_final - 0x20 * NUM_CHANNELS * 2` (= 마지막 group의 시작 주소)를
  `pim_args`에 채우고 `pim_exec`을 부른다
- build는 host 메모리에서 하고 `datacopy_cpu2pim`으로 PIM DRAM에 한 번 옮긴다

### 9.7 stage / unstage의 host 비용 (2026-08-11)

`invoke`가 `batch=8`로 3.4배 줄면 그다음 지배 항이 이것이다. host 비중은 `M`에 반비례
한다 — B stage 바이트는 `k*n*2`로 `M`과 무관한데 유효 연산은 `2*M*n*k`이기 때문이다.
`M=1024`에서 7%, `M=80`에서 37%이고, `batch=8` 뒤에는 각각 20%, 67%가 된다.

원소 하나를 옮기는 비용이 전부이므로 셋을 고쳤다.

| 무엇 | 왜 | 실측 |
| --- | --- | --- |
| f16 → bf16을 **65536칸 표**로 | 16 bit → 16 bit 사상이라 가짓수가 유한하다. 산술로 하면 컴파일러가 벡터화해 줄 때만 빨라진다 | 1.4 → 5.7 GB/s |
| host 조립 buffer를 **launch 너머로 재사용** | launch마다 `mmap`/`munmap`하면 언제나 차가워서 첫 접근이 전부 page fault다 (64 MiB = 16384 페이지) | 2.5 → 5.7 GB/s |
| `memset`을 **padding 자리에만** | 남는 자리의 0은 정확성 요구사항이지만 그것뿐이다. buffer 전체를 지우면 padding이 없을 때 전부 헛일이다 | buffer 한 번 훑기 |

합쳐서 4.5배다. 표는 `.so`를 여는 시점에 파일 유효 범위 static의 생성자가 채우므로
OpenMP 스레드가 뜨기 전에 이미 완성돼 있고, buffer 재사용은 스레드마다 따로 둔 칸
(`__thread`)이라 lock이 없다 — PIM Device 하나에 host thread 하나이고 buffer는 그
thread의 loop 몸체 안에서만 산다.

재사용분에는 이전 launch의 값이 남는데, stage가 쓰는 자리와 padding을 0으로 채우는
자리가 합쳐 buffer 전체를 덮으므로 남은 값이 읽히지 않는다. chain은 실제로 쓴 record
까지만 `datacopy`하고, C는 `datacopy_pim2cpu`가 채운 행까지만 되읽는다.

행 주소는 안쪽 loop 밖으로 뺀다. 원소마다 `(o0+i)*s0 + (o1+j)*s1`을 계산하면 gcc가
접근 패턴을 gather로 보고 벡터화를 포기한다. 같은 이유로 stub은 `-O3`로 빌드한다
(`-fversion-loops-for-strides`가 거기서만 켜진다).

---

## 10. 전체 pipeline 안에서의 위치

```
Triton kernel
   │  make_ttir
   ▼  TTIR
   │  _ttir_to_ttsharedir          ← CPU/PIM 공용, target 무관
   ▼  TTSharedIR
   │  PIM_COMMON_PASSES            ← target 무관, profile만 target별
   │  --pim-form-host-plan         ← partition + 전송/launch/gather 명시
   ▼  Planned PIM Common IR        ← 여기까지 완전히 공유
   │
   ├─ --convert-pim-to-upmem  → UPMEM IR → --upmem-to-host / --upmem-to-device
   ├─ --convert-pim-to-aimx   → AiMX IR L1 → --aimx-lower-matmul → L2
   │                                       → --aimx-to-runtime-stub / --aimx-to-descriptors
   └─ (other PIM target)      → …
```

> **PIM Common IR은 "어느 데이터가 어느 PIM Device에 몇 번 가는가"까지,
> AiMX IR은 "그 데이터가 어떤 descriptor 열로 옮겨지고 언제 누산기가 비워지는가"부터.**

Target 진입 시 `pim.partition`의 A/B 입력 전송과 C gather를 확인한다. `pim.launch`의
matmul을 기존 compute template으로 보존한 뒤 descriptor 실행으로 내린다. 분석용
`pim.device`를 자동 변환하거나 지원하지 않는 action 순서를 임의로 바꾸지 않는다.

| 결정 | UPMEM | AiMX |
| --- | --- | --- |
| memory 계층 | host DRAM → MRAM → WRAM 3단 | host DRAM → PIM DRAM 2단 |
| 계층 간 전송 단위 | DMA 8 byte 정렬 / 2048 byte 이하 / 연속 | descriptor group, `len`은 32 byte 배수, 2차원 stride |
| unit 내부 병렬 | tasklet (독립 thread) | 4096 SIMD lane (주소로만 분배) |
| layout 재배치 | tile-major 연속 (`stage`) | pitch 규정 2차원 (`stage`) + 16배 복제 |
| unit이 실행하는 것 | device program | **descriptor chain** — program이 없다 |
| reduction 분할 | 자유 (WRAM 누산기) | §8 I4 확인 전까지 불가 |

---

## 11. Data type

`TYPE_SIZE = sizeof(short) = 2`가 `PIM_WIDTH`, `CHUNK`, `REG_SIZE`, `C_len`을 포함해
**모든** 크기 상수에 들어 있다. 따라서 PIM DRAM 안에서 A·B·C는 모두 2 byte 원소, 즉
bfloat16이다.

| | host buffer | PIM DRAM | 변환 지점 |
| --- | --- | --- | --- |
| A | f32 또는 bf16 | bf16 | `aimx.stage` |
| B | f32 또는 bf16 | bf16 | `aimx.stage` |
| C | f32 또는 bf16 | bf16 | `aimx.unstage` |

int8/int32는 표현할 수 없다 — 2 byte로 int32 출력을 담을 방법이 어떤 누산기 폭으로도
없다. 지금 세 커널은 전부 int8 입력 / int32 출력이라 그대로는 거부된다 (§15 E6).

### 11.1 누산기는 fp32가 아니다

입력이 bf16인 것과 누산이 fp32인 것은 별개다. `ubench/matmul/pim_mac_simul.cpp`가
hardware와 bit 단위로 일치하는 것이 `./matmul`의 정확성 검사이므로, 그 모델이 곧 의미론이다.

> **주의.** `pim_mac_simul.cpp` 첫머리의 `// #define USE_MODIFIED_MAC_UNIT`이 주석
> 처리돼 있고 Makefile에도 `-D`가 없다. 따라서 `#ifdef` 쪽이 아니라 **`#else` 쪽이
> 컴파일된다.** 두 쪽은 지수 처리 규칙이 다르다. 아래는 전부 `#else` 쪽 기준이다.

곱셈기가 지수를 두 조각으로 쪼갠다.

```
mul_exp     = expA + expB - 127
exp_control = mul_exp & 0x7            가수를 왼쪽으로 미는 데 쓴다
exp_compare = (mul_exp >> 3) & 0x1F    ← 누산기가 드는 지수. 한 칸이 8 binade = 256배
```

그리고 누산기는 지수 칸이 **2 이상** 벌어지면 작은 쪽을 **통째로 버린다.**

| `exp_diff` | 하는 일 |
| --- | --- |
| 0 | 가수를 그대로 더한다 |
| ±1 | 작은 쪽을 8 bit 밀고 더한다 |
| **±2 이상** | **작은 쪽을 버린다. 더하지 않는다** |

> **한 내적 안에서 항끼리 `2^16 = 65536`배 넘게 벌어지면 작은 항은 결과에 전혀 기여하지
> 않는다.** 이것이 이 하드웨어의 수치 제약이다.

참조 ubench가 A를 약 `1e-4`(`generate_random()/10`), B를 `[0, 2)`
(`generate_random_255()`)로 좁게 잡는 이유가 이것이다. 곱이 전부 한 지수 칸에 모인다.

**0은 정확히 0이다.** `in_A_exp == 0 || in_B_exp == 0`이면 `check_zero`가 서서
`mul_exp = 0, mul_mant = 0`이 되고, 그 항은 언제나 `smaller_2` 분기로 빠져 누산기를
그대로 통과시킨다. **따라서 §6.3의 zero padding은 값에 영향을 주지 않는다** — 성능
최적화가 아니라 정확성 요구사항이라는 §6.3의 주장이 여기서 증명된다.

예로 `A = 1.0`(`0x3f80`), `B = 3.0`(`0x4040`)이면

```
mul_exp     = 127 + 128 - 127 = 128  →  exp_compare = 16,  exp_control = 0
mul_mant    = (0x80 * 0xC0) >> 4 = 0x600
mant_align  = 0x600 >> 3 = 0xC0
normalize   : bit 7 이 최상위 → out.exp = 16<<3 = 128, out.mant = 0xC0
result      = (128 << 7) | (0xC0 & 0x7F) = 0x4040          ← 정확히 3.0
```

이 계산이 이 문서의 어떤 결정도 바꾸지는 않지만, **compiler가 fp32 누산을 약속해서는
안 된다**는 것과 CPU 참조와의 bit 단위 일치를 요구할 수 없다는 것을 규정한다.

---

## 12. Saturation point 측정과 padding-aware autotuning

`find_sweepspot.png`와 `autotuning.png`가 요구하는 것을 이 설계의 변수로 옮긴다.

### 12.1 무엇을 측정하는가

측정 대상은 **execution tile**이지 `BLOCK_M/N/K`가 아니다 (§3.5). 그래서 측정에
Triton이 필요 없다 — §2.4의 격자 위에서 descriptor chain을 직접 만들어 재면 된다.

```
m ∈ {8, 16, 32, 64, 128, 256, 512}            (batch의 배수)
k ∈ {256, 512, 1024, 2048, 4096, 8192, …}     (256의 배수)
n ∈ {4096, 8192, 16384}                       (4096의 배수 — §2.4)
batch ∈ {1, 2, 4, 8}                          (n에 맞는 것만)
```

결과는 표 `(m, k, n, batch) → GFLOPS`다. **saturation point**는 최고 throughput의
일정 비율(예: 95 %) 안에 드는 가장 작은 tile이다. 같은 sweep에서 §6.2의 비용 모델도
검증된다 — 실패하는 지점이 PIM DRAM 예산을 알려 준다.

**이 표는 하드웨어 성능표이지 compiler가 고를 값이 아니다.** §6.1이 보였듯 `k_t`와
`n_t`는 문제 크기가 정하고(각각 `k`와 `N`), `m_t`는 전송량에 나타나지 않는다. 그래서
compiler 쪽 tuning 변수는 다음이다.

| 변수 | 범위 | 무엇을 바꾸는가 |
| --- | --- | --- |
| `m_t` | 16 ~ 예산 | descriptor chain 크기 ↔ invocation 수. `M`을 나누는 값이 padding 0 |
| `n_max` | 4096의 배수 | A 재stage 횟수 ↔ B buffer 크기. `k_max`와 예산을 나눠 쓴다 |
| `k_max` | 256의 배수 | 덮을 수 있는 `k`의 상한 ↔ B buffer 크기 |
| `batch` | `{1,2,4,8}` | §2.5. §15 E9 전에는 1 고정 |
| PIM Device 수 | **지금은 1 고정** | §12.3. `compiler.py`의 `"devices": 1`이라 tuning 변수가 아직 아니다 |
| ~~double buffering~~ | — | §7. `aimx.invoke`가 동기라 지금은 선택지가 아니다 |

**block/wave 단위로 잡으라는 지적이 있었다** — "wave당 logical tile 수", "device chunk당
wave 수" 같은 것. AiMX에서는 `pim_core_shape = [1]`이라 wave가 tile 하나로 퇴화하므로
(§3.4) 그 두 변수가 항상 1이고 tuning 대상이 되지 않는다. 남는 자유도가 `m_t`·`n_max`
·`k_max`뿐이고, 그것이 곧 "device chunk가 몇 개의 logical output tile을 담는가"이다 —
`(m_t/BLOCK_M) × (n_t/BLOCK_N)`으로 환산하면 같은 값이다.

`k_t`는 `k`가 정하므로 tuning 변수가 아니다.

### 12.2 목적 함수는 GFLOPS가 아니다

`autotuning.png`의 요지는 "GFLOPS가 떨어져도 총 실행 시간이 나아질 수 있다"이다.
목적 함수가 **padding을 포함한 wall clock**이기 때문이다.

```
T_device(m_t, k_t, n_t, batch)
   =  ⌈m_dev/m_t⌉·m_t  ×  ⌈n_dev/n_t⌉·n_t  ×  k_t  ×  2
      ──────────────────────────────────────────────────
                GFLOPS(m_t, k_t, n_t, batch)

T = max over devices of T_device  +  stage/unstage 시간
```

분자가 **padding까지 포함한 실제 수행 연산량**이다. tile을 키우면 분모가 커지지만
분자도 커진다. 균형점이 최적이고 saturation point와 반드시 같지 않다.

### 12.3 PIM Device 개수도 tuning 변수다

grouped_gemm.py (M=N=K=128)가 보여 준다.

| device 수 | device당 `(m_dev, n_dev)` | padding 후 tile | 유효율 | device당 연산량 |
| --- | --- | --- | --- | --- |
| 1 | (128, 128) | (128, 256, 512) | 12.5 % | 16,777,216 |
| 8 | (32, 64) | (32, 256, 512) | 6.25 % | 4,194,304 |

**유효율은 8 device 쪽이 절반이지만 wall clock은 4배 빠르다.** 목적 함수는 언제나
§12.2의 `T`이지 유효율이 아니다. device 수는 `PIMOptions.num_units`로 조절되므로
PIM Common IR을 건드리지 않고 tuning된다.

### 12.4 어떻게 넣는가

AiMX lowering이 tile 크기를 다시 선택하지 않는다. autotuner가 고른
`PIMConfig` 정수 값이 공통 pass에서 다음 속성으로 materialize된다.

```text
BLOCK_M/N/K       -> #pim.block
group_m/n/k       -> #pim.block_group
tile_m/n/k        -> #pim.execution_tile
p_m/n/k           -> #pim.device_mapping
```

AiMX target option에는 `batch`, `base_fpga`만 남는다. lowering은 Common IR의
Execution Tile을 `batch` 물리 눈금에 맞게 padding하고 descriptor chain으로
옮긴다. bank capacity는 `#pim.target_profile` 안의
`bank_capacity_limit_bytes`를 사용하며 별도 tile/capacity policy를 두지 않는다.

---

## 13. 세 커널 walkthrough

> **아래 네 walkthrough의 숫자는 폐기됐다.** `batch`를 자유 선택하고 `n_t`를
> `4096/batch`까지 줄이던 옛 규칙으로 계산한 것인데, §2.4·§2.5가 `batch = 1`,
> `n_t`는 4096의 배수로 바뀌었고 §17.2가 tile 모양을 런타임 값으로 돌렸다.
> **수치를 근거로 쓰지 말 것.** 살아 있는 것은 구조뿐이다 — 작은 tile도 실패하지 않고
> 정확한 답을 내며, padding 때문에 유효율이 낮아진다.
>
> **현재 규칙으로 다시 계산한 것은 아래 표 하나뿐이고, 이것이 실제로 돈 값이다**
> (`tile-m=16`, `tile-k=32768`, `tile-n=4096`, `batch=1`, PIM Device 1개). `batch`
> 기본값이 그 뒤 "legal한 최대값"으로 바뀌었으므로(§2.5) B 전송은 `batch`로 나눠 읽을 것.
>
> | 커널 | `m_t × k_t × n_t` | invoke | A 전송 | B 전송 | C 전송 | 합 |
> | --- | --- | --- | --- | --- | --- | --- |
> | `gemm.py` M=N=K=1024 | 16 × 1024 × 4096 | 64 | 8 MiB | 8 MiB | 8 MiB | **24 MiB** |
> | `gemm.py` M=1024, N=28672, K=8192 | 16 × 8192 × 4096 | 448 | 112 MiB | 448 MiB | 56 MiB | **616 MiB** |
>
> 두 번째 줄의 A 112 MiB는 `n_t`가 4096이라 `⌈N/n_t⌉ = 7`번 다시 모으기 때문이다.
> `tile-n=28672`로 올리면 16 MiB로 줄고 B buffer가 448 MiB가 된다 — §6.1의 trade-off다.

`--pim-map-devices`는 `factorize(D)`로 device를 두 축에 나눈다. `D = 8`이면 `(4, 2)`다.

### 13.1 gemm.py — M=1024, N=8192, K=28672, BLOCK 32/32/16

```
logical grid       32 × 256 tile
device 분할        (4, 2)  →  device당 8 × 128 logical tile
device 직사각형    m_dev = 256,  k = 28672,  n_dev = 4096
전치 여부          m_dev < n_dev → 전치하지 않는다
batch              모든 값에서 padding 면적 동일 → 1 (동점 → 최소)
```

배수 조건: `m_t = 256` ✔ / `k_t = 28672` (= 256×112) ✔ / `padded_k = 28672` (= 4096×7) ✔
/ `n_t = 4096` ✔ → **padding 낭비 0 %**.

device 직사각형 전체를 한 execution tile로 잡으면 §6.2가

| | |
| --- | --- |
| A block | 14.7 MB |
| Arel | **235 MB** |
| B block | **235 MB** |
| C block | 2.1 MB |
| descriptor chain | 58.8 MB |
| 합계 | **≈ 546 MB** |

예산이 그보다 작으면 §6.1이 줄인다. `batch = 4`, `n_t = 1024`, `m_t = 64`로 가면

| | |
| --- | --- |
| A block | 3.7 MB |
| Arel | 58.7 MB (§8 I2 성공 시 **0**) |
| B block | 58.7 MB |
| C block | 0.13 MB |
| descriptor chain | 9.2 MB (§8 I2 성공 시 **7.3 MB**) |
| 합계 | **≈ 130 MB** (I2 성공 시 **≈ 70 MB**) |

device당 execution tile은 `(256/64) × (4096/1024) = 16`개이고 **전부 같은 모양**이므로
descriptor chain은 launch당 한 번만 만든다 (§7). A 재배치는 §8 I1로 4번만 돈다.

### 13.2 gemm.py 작은 경우 — M=N=K=32

현재 `test()`가 실제로 도는 크기다. grid가 `(1,1)`이라 logical tile이 하나뿐이고
`(4,2)` 분할에서 device 하나만 일한다.

```
m_dev = 32, k = 32, n_dev = 32
batch = 8      (n 32 → 512 가 4096 보다 낫다)
padding:  m 32→32,  k 32→256 (8배),  n 32→512 (16배),  padded_k 256→4096
```

유효율 = `32·32·32 / (32·256·512)` = **0.78 %**.

**실패하지 않는다는 것이 중요하다.** §6.3의 padding 덕분에 정확한 답이 나온다.
진단(remark)으로 유효율을 알려 크기를 키우거나 device 수를 줄이도록 유도한다.

### 13.3 split_k.py — M=N=K=256, BLOCK 16/32/16, SPLIT_K=8

```
grid              (16*8, 8) = (128, 8)
logical 축        output_row_tile 16, output_column_tile 8, explicit_k_partition 8
device 분할       explicit_k_partition은 tileAxes에서 빠진다 → (4, 2, 1)
device 직사각형   m_dev = 64, n_dev = 128, plane 8개
plane당 reduction cdiv(256, 16*8) * 16 = 32
```

C의 `pim.place.capacity`가 `[8, 64, 128]`이 되고 device는 plane마다 하나씩
**8번 invoke**한다. 각각 `(m, k, n) = (64, 32, 128)`이다.

`m_dev = 64 < n_dev = 128`이라 전치하지 않는다. `batch = 8`, `m_t = 64`, `k_t = 256`,
`n_t = 512`, `padded_k = 4096`. 유효율 = `64·32·128 / (64·256·512)` = **3.1 %**.

**plane p의 k 범위가 연속이 아니다** — `BLOCK_K = 16`짜리 덩어리를 `16*8 = 128` 간격으로
건너뛴 집합이다. descriptor의 stride 필드는 행 간격 하나만 표현하므로 직접 표현할 수
없다. 그런데 문제가 되지 않는다: A와 B는 어차피 `aimx.stage`가 host software로 모아
PIM DRAM에 새 layout으로 쓰기 때문이다. strided 집합을 모으는 것은 gather의 index
pattern이 달라지는 것뿐이다.

최종 `c.sum(dim=0)`은 PyTorch가 한다. compiler는 plane을 건드리지 않는다.

### 13.4 grouped_gemm.py — M=N=K=128, BLOCK 32/32/32, GROUP_M=2

1차원 grid에 GROUP_M swizzle이 걸려 있고 `PIMBuildLogicalGridPass`의 `matchGroupOrder`가
`output_row_tile = 4`, `output_column_tile = 4`로 되돌린다.

```
device 분할       (4, 2)  →  device당 1 × 2 logical tile
device 직사각형   m_dev = 32,  k = 128,  n_dev = 64
batch = 8
padding:  m 32→32,  k 128→256,  n 64→512,  padded_k 256→4096
```

유효율 = `32·128·64 / (32·256·512)` = **6.25 %**.

이 커널의 문제는 분배가 아니라 **문제 자체가 AiMX의 최소 실행 단위보다 작다**는 것이다.
`(32, 256, 512)` = 4.2 M multiply-accumulate가 최소 단위인데 문제 전체가 2.1 M이다.

### 13.5 요약

| kernel | device당 execution tile | invoke 수 | 유효율 | 판정 |
| --- | --- | --- | --- | --- |
| gemm.py (1024×8192×28672) | (64, 28672, 1024) batch 4 | 16 | **100 %** | 목표 형태 |
| gemm.py (32×32×32) | (32, 256, 512) batch 8 | 1 | 0.78 % | 돌지만 진단 |
| split_k.py | (64, 256, 512) batch 8 | 8 | 3.1 % | 돌지만 진단 |
| grouped_gemm.py | (32, 256, 512) batch 8 | 1 | 6.25 % | 돌지만 진단 |

**네 경우 모두 실행 가능하고, 세 경우는 문제가 작아 비효율적이다.** 그 비효율은
compiler의 문제가 아니라 §12가 존재하는 이유 자체다.

---

## 14. 진단

`--convert-pim-to-aimx`가 진입 즉시 확인한다.

| # | 조건 | 진단 |
| --- | --- | --- |
| 1 | `target == "aimx"` | `the module targets <t>` |
| 2 | `parallel_execution_kind == "simd"` | `AiMX lanes share one descriptor stream, so parallel_execution_kind must be "simd"` |
| 3 | `bank_capacity_limit_bytes` > 0 | `bank_capacity_limit_bytes must be positive` |
| 4 | device 몸체에 `linalg.matmul` 하나 | `the device body has no linalg.matmul` |
| 5 | A·B·C 원소가 2 byte float | `element type <T> cannot be represented; the AiMX descriptor format moves 2 byte per element` |
| 6 | 세 buffer의 마지막 축 stride가 1 | `buffer axis <d> is not unit-stride; A, B and C must be row-major` |
| 7 | §6.1이 예산 안의 tile을 찾음 | `the reduction of length <k> needs <x> byte of PIM DRAM but the device budget is <y>` |
| 8 | (remark) padding 비율 | `the execution tile is padded from <m×k×n> to <m_t×k_t×n_t>; <p>% of the multiply-accumulate operations are on padding` |
| 9 | (remark) 전치 | `the problem is transposed because m (<m>) exceeds n (<n>)` |

8번이 §12로 이어지는 통로다. `BLOCK`이나 `num_units`를 잘못 잡았을 때 compile time에
바로 알 수 있다.

---

## 15. 확정하지 못한 것 — 직접 확인이 필요합니다

빌드·실행·검증은 하지 않았습니다. 아래는 코드만으로 결정할 수 없어 남겨 둔 것들이고,
각각 무엇을 하면 되는지 적었습니다. **E3가 가장 값어치가 큽니다.**

> **실행 절차는 `AiMXTarget_Experiments.md` 에 따로 있습니다.**
> hardware가 막히는 정확한 지점(`dma_fops_pl.c`의 timeout 경로가 정리 코드를 건너뛴다),
> reboot 없이 회복하는 사다리, 실행 직전 guard 코드, 위험도 오름차순 실험 순서가 거기 있습니다.

### E1. channel 8개 vs `NUM_CHANNELS = 16`

물리 channel은 8개인데 `pim_math.h`는 16입니다. "channel 8 × pseudo-channel 2"로
읽으면 lane 4096개가 맞아떨어집니다.

**확인 방법:** 작은 크기(`m=8, k=256, n=4096`)로 한 번 돌리고 `PIM_MATH_LOG`를 켜서
descriptor 개수가 예상과 같은지, 결과가 CPU 참조와 맞는지 보시면 됩니다.

**영향:** 없습니다. 이 문서의 모든 식이 `NUM_CHANNELS`라는 기호에만 의존합니다.

### ~~E2. stride 필드의 shift 폭~~ — **닫힘. 답은 40이다**

실험이 필요 없습니다. 소스가 답을 갖고 있습니다.
`/home/yonsei_park/pim_aimx_sw/pim_sw/src/pim_math_src/matmul.c`의 일곱 자리
(143, 163, 170, 312, 325, 337, 344행)가 전부 `<< 33`입니다.
`memcpy.c`의 `<< 48`은 `pim_copy` 경로의 다른 관례이므로 한 chain 안에서 섞지 않습니다.

**실측으로도 확인됐습니다 (2026-08-10).** `m=1, k=256, n=4096`은 `pitch == len`이라
shift에 둔감한 형상이므로 판정에 쓸 수 없습니다. `n = 8192`로 올리면 B의
`pitch = n*2 = 16384`, `len = 8192`가 되어 strided 읽기가 실제로 걸립니다.
`~/kangmlee/test/e0_diff`에서 `./probe 1 1 256 8192 1 1`을 돌린 결과:

```
[L vs 기대] 전부 일치 (1 x 8192)      ← 라이브러리가 옳다
[L == M]   전부 일치 (1 x 8192)      ← shift 40 을 쓴 우리 chain 도 옳다
```

같은 실행이 **`n = 8192 > 4096`이 한 chain(8 group, `r_loop = 2`)에 들어간다**는 §2.4의
정정도 함께 확인했습니다.

> **이 절은 한때 "답은 33"이라고 적었고 근거로
> `/home/yonsei_park/kangmlee/pim_aimx_sw/.../matmul.c`를 들었습니다. 그 경로는 존재하지
> 않습니다.** 디스크의 `matmul.c`는 하나뿐이고 **전부 `<< 33`**이며, `PIM_RD_INSTR_MC`
> (`pim_math.h:250`)가 `src`를 그대로 저장하므로 33이 최종 shift입니다. `<< 40`인 C 소스는
> 트리 어디에도 없습니다. 구현(`AiMXDialect.h`의 `kStrideShift`)은 40으로 맞습니다.

### E3. reduction을 나눌 수 있는가 (§8 I4)

ISA에 memory → vACC 경로가 있습니다 (`_gemm_silent`의 bias descriptor).
그 함수가 열 vector broadcast로만 쓰는 것은 소프트웨어의 선택입니다.

**확인 방법 (가):** C drain의 거울상을 시도합니다. `_gemm_silent` 325행을

```c
Bias_off = (r * PIM_WIDTH * NUM_CHANNELS / batch) + (p * (r_size * TYPE_SIZE));
PIM_RD_INSTR_MC(&desc_idx, &next_desc,
                Bias_pa | ((r_size * TYPE_SIZE) << 33),
                mm_vACC_pa | ((NUM_CHANNELS * PIM_WIDTH / batch) << 33),
                C_len / batch,
                OPCODE_STRIDE | opcode_batch, fpga_id, pim_code_section_va);
```

로 바꾸고 bias 자리에 임의의 `m × n` 행렬 `D`를 넣어 결과가 `A·B + D`가 되는지 보시면
됩니다. (`num_descr` 계산은 그대로입니다.)

**확인 방법 (나):** 미사용 bit인 `OPCODE_HOLD`(bit 14)를 C descriptor에 붙여 보시고,
vACC가 비워지지 않아 다음 chain이 이어서 누산하는지 보시면 됩니다. 이쪽이 성공하면
restore 없이도 됩니다.

**성공하면:** §6.1 단계 6의 막다른 길이 사라지고, `Arel`과 descriptor chain의
`k` 비례 항이 줄며, k 방향 pipelining이 가능해집니다.

**실패하면:** 이 문서 그대로입니다. reduction은 execution tile 안에서 완결되고,
host에서 부분합을 더하는 경로를 backend가 만듭니다.

### E4. A 재배치를 계산 descriptor에 융합할 수 있는가 (§8 I2)

성공하면 `m_t * k_t * 32` byte짜리 `Arel` buffer가 통째로 사라집니다 —
§6.2에서 가장 큰 항목입니다.

**확인 방법:** 계산 phase의 A descriptor를

```c
PIM_RD_INSTR_MC(&desc_idx, &next_desc,
                A_pa | ((padded_q_size * TYPE_SIZE) << 33), 0, 512,
                OPCODE_A_MASK | OPCODE_MATMUL | OPCODE_PIM_MODE
                | OPCODE_BC | OPCODE_TP | OPCODE_STRIDE | opcode_batch,
                fpga_id, pim_code_section_va);
```

로 바꾸고(원본 A를 직접 가리키고, 재배치 phase는 통째로 제거) 결과가 맞는지 보시면
됩니다.

**함께 확인:** 재배치의 두 descriptor를 하나로 합칠 수 있는지 (§8 I3) —
`src = A_pa | pitch`, `dst = srcA_buf_pa + …`, `flags = BC|TP|STRIDE|batch`.
E4가 성공하면 이 항목은 무의미해집니다.

### E5. device당 PIM DRAM 예산

현재 구현은 AiMX runtime heap 상수에서 `bank_capacity_limit_bytes`를 계산해 공유
profile에 기록한다. 별도 임시 capacity option은 사용하지 않는다.

**크기를 키워 가며 재는 방법은 쓸 수 없습니다.** `pim_malloc`은 실패해도 NULL을
돌려주지 않고 `PIM] OOM`을 찍은 뒤 프로세스를 죽입니다 (exit 255). 게다가 1 MB부터
16 GB까지 요청이 전부 '성공'했습니다 — 할당은 주소만 떼어 줍니다 (§1.8 (h)).

**확인 방법:** driver 적재 로그의 BAR 길이가 가장 믿을 만합니다.

```
sudo dmesg | grep -E "PIM_MEM_[0-9]+\] BAR #4"
```

그 값을 `AIMX_DEVICE_DRAM_BYTES` 상수로 적습니다 (§4.7). 확인하고 싶으면 그 크기를
실제로 **써 보고 읽어서** 검증하십시오 — 할당 성공은 근거가 되지 않습니다.

### ~~E9. `n_t < 4096`이 legal한가~~ — **닫힘. 된다** (2026-08-11)

`~/kangmlee/test/e12_narrow_n`. **검증된 라이브러리(`matmul()`)만 부르고 `pim_args.r_size`
하나만 바꾸었습니다.** `m=4, k=256, n=1024` → 참조 구현의 이진 분해로 `batch = 4` 한
class, output block은 4행 × 1024열로 lane 4096개에 정확히 맞습니다.

| mode | `r_size` | B/C 행 pitch | `r_loop_var` | 결과 |
| --- | --- | --- | --- | --- |
| 0 | `padded_n` = 4096 | 8192 byte | 4 | 전부 일치 (4 × 1024) |
| 1 | **`n` = 1024** | **2048 byte** | 1 | **전부 일치 (4 × 1024)** |

one-hot A(행마다 다른 `j0` = 13/50/87/124) + `B[j][i] = i % 257`이라 기대값이 열마다
다릅니다 — `aimx_check_nontrivial`을 통과한 시험입니다.

**결정적 지점은 C의 행 간격입니다.** `r_size = 1024`로 부르고 host에서 pitch 1024로
되읽었는데 4행이 전부 맞았습니다. 하드웨어가 pitch를 8192 byte로 고정 해석했다면 1~3행이
어긋났을 것입니다. **즉 `r_size`가 2048 byte까지 그대로 지켜집니다.**

**부수적으로 `batch = 4`가 처음으로 확인됐습니다.** 이 트리는 그동안 `batch = 1`만
돌렸고, README §2의 "E3-a에서 `batch = 8`일 때 C의 1~7행이 안 쓰였다"는 관측은 그때
`stride shift`가 33이었기 때문일 가능성이 큽니다 (§15 E2).

> **아직 확인되지 않은 것 하나.** mode 1에서는 B와 C 모두 `pitch == len`(둘 다 2048)이라
> stride가 연속 복사로 퇴화합니다. **`pitch < 8192`이면서 `pitch ≠ len`인 경우**는 아직
> 직접 실행된 적이 없습니다. 다만 mode 0이 `pitch 8192 ≠ len 2048`을 통과했고 mode 1이
> C 행 간격을 지켰으므로, 두 성질이 독립적으로는 확인됐습니다.

**영향 (§2.4, §2.5):**

- `n_t`의 하한이 `4096`에서 **`4096/batch`**로 내려갑니다. 배수 조건은
  `n_t % (4096/batch) == 0`이고 `n_t ≥ 4096/batch`입니다 (`r_loop_var == 0`이면 C를
  아예 쓰지 않습니다)
- **`batch > 1`이 의미를 갖게 됩니다.** `N < 4096`일 때 lane이 노는 문제가 여기서
  풀립니다 — `N = 1024`이면 `batch = 4`, `n_t = 1024`로 이용률이 25% → 100%가 됩니다
- 대가는 descriptor 수입니다. §2.4의 `G`가 큰 `k_t`에서 `batch`에 정비례하므로, `batch`
  선택은 이제 **padding 면적과 `G`를 함께 최소화하는** 목적 함수가 필요합니다 (§2.5)

**구현 상태(2026-08-11 갱신):** `--convert-pim-to-aimx`는 `batch=0`(기본값)일 때 Triton
block이 허락하는 최대값을 고릅니다 (§2.5). `n_max ≥ 4096`은 그대로입니다 — `N < 4096`에서
lane을 채우려면 `n_t`를 `4096/batch`까지 줄이는 규칙이 따로 필요하고, 그것은 아직입니다.

### E10. §1.6의 재배치 순열이 `k > 256`, `m > batch`로 확장되는가 (§8 I5)

순열은 `m=1, k=256, batch=1`에서 실측했습니다. 참조 구현의 주소식이 `q`를 256씩, `p`를
`batch`씩 도는 단순 반복이라 확장은 자연스럽지만 확인되지 않았습니다.

**영향:** host가 `Arel`을 직접 만드는 §8 I5의 전제입니다. 재배치 descriptor를 통째로
없앨 수 있는지가 여기서 갈립니다.

**확인 방법:** `m=8, k=512`로 재배치 phase만 돌리고 `Arel`을 되읽어
`Arel[b][c*256+g*16+t] == A[p][256*b + 16*c + t]`인지 봅니다.

### ~~E11. channel마다 다른 record를 넣을 수 있는가~~ — **닫힘. (c) 안 된다** (2026-08-10)

**결과.** `~/kangmlee/test/e11_perch`, `fpga_id = 1`, `pim_copy`, group 1개, `opcode = 0`.

| mode | record 16칸 | 결과 |
| --- | --- | --- |
| 0 | 전부 `src = S_pa` | `label(d) = d` — harness 정상 |
| 1 | `src[ch] = S_pa + ch*8192` | **`label(d) = d`. mode 0과 byte 단위로 동일** |

record 1~15에 완전히 다른 주소를 넣었는데 옮겨진 8192 byte가 정확히 record 0의
구간이었다. 두 실행 모두 `OK: hardware clean`.

**해석.** chain은 `next` 사슬로 이어진 CDMA scatter-gather 리스트이고 `next`가 group
단위(512 byte)로 뛴다. head = `desc_base_pa`, tail = `desc_last` = `base + 512*(g-1)`
이므로 엔진이 걷는 자리는 `base, base+512, …` — **group마다 첫 record 하나뿐이다.**
나머지 15개는 실행되지 않는다 (§1.3).

**반영.** §1.3 표, §4.4, §4.6, §16 요약 7번. `pim_core_shape = [1]`이 최종이고
`PIMMapCoresPass`·`PIMFormExecutionWavesPass`는 손대지 않는다.

**남는 유보.** vehicle이 `pim_copy`라 PIU가 꺼진 상태였다. `pim_exec`에서 PIU가 group을
다르게 소비할 가능성은 원리적으로 남지만, 재려면 MAC 경로를 써야 해 위험도가 크고
참조 구현의 자료 배치가 이미 "descriptor 하나 = 8192 byte 전송 하나"와만 앞뒤가 맞는다.

<details>
<summary>원래 절차 (참고용)</summary>

**무엇이 확인된 것인가.** descriptor group(512 byte)이 `ROBACOCH` 때문에 32 byte씩
16 channel에 하나씩 흩어진다는 것 — 즉 **channel별 명령 슬롯이 이미 존재한다**는 것은
§1.3에서 구조적으로 확정했습니다. `mPIM_RD_INSTR_MC`가 그 16칸을 같은 값으로 채울
뿐입니다.

**무엇이 미확인인가.** channel `c`의 record에 주소 `X`를 쓰면 channel `c`가 무엇을
읽는지입니다. 세 갈래가 있습니다.

| | channel `c` 가 읽는 것 | 뜻 |
| --- | --- | --- |
| **(a)** | `[X, X+len)` 중 channel 자리가 `c` 인 granule | channel 별 독립 주소가 된다 |
| **(b)** | channel 자리를 무시하고 나머지를 channel-local 주소로 해석 | 역시 독립이지만 주소식이 다르다 |
| **(c)** | record 0 만 유효하고 나머지는 무시 | `[1]` 이 최종 모델이다 |

**vehicle:** `pim_copy`. `OPCODE_PIM_MODE`가 없는 순수 데이터 이동이므로
`AiMXTarget_Experiments.md` §2.3 규칙에 따라 가장 안전하고, MAC pipeline을 건드리지
않습니다. 최악의 경우 잘못된 byte가 복사됩니다 (§1.3 C 유형).

**자리**: `~/kangmlee/test/e11_perch/`. 절차는 `AiMXTarget_Experiments.md` §4 E11에
있습니다. 여기에는 무엇을 왜 하는지만 적습니다.

1. `common/aimx.h`의 `aimx_emit_per_ch()`가 descriptor group의 16칸에 서로 다른 `src`를
   넣습니다. `mPIM_RD_INSTR_MC`와 같고 `src`만 배열로 받습니다.
2. 원본 `S`를 `NUM_CHANNELS * 8192` byte 잡고 **32 byte granule마다 자기 번호를
   표식으로** 채웁니다 — granule `g`의 `short` 16칸을 전부 `g`로 둡니다. MAC을 지나지
   않는 순수 복사이므로 bf16이 아니라 `short`를 그대로 쓸 수 있고, 그래서 표식이
   **0..4095로 전부 구별됩니다.**
3. 목적지 `D`를 8192 byte 잡고 0으로 채웁니다.
4. group 하나만 만듭니다. `src[ch] = S_pa + ch * 8192`, `dst = D_pa`,
   `length = 8192`, `opcode = 0`(플래그 없음 — pitch도 `PIM_MODE`도 쓰지 않으므로
   stride shift 논쟁을 피합니다).
5. `aimx_run(&c, /*use_pim_copy=*/1, /*expect_groups=*/1)`. `aimx_guard_chain`이
   `desc_last = base + GROUP*(g-1)`을 포함한 §2.2 검사를 통과시킨 뒤에만 호출합니다.
6. `D`를 host로 읽어 표식을 봅니다.

**판정.**

| `D` 의 내용 | 결론 |
| --- | --- |
| granule 번호가 `ch * 256` 부근으로 **16 갈래로 갈린다** | **(a)** — channel 별 독립 주소. `[16]` 로 간다 |
| 갈리기는 하는데 번호가 예상과 다르다 | **(b)** — 표식이 실제 주소식을 그대로 알려 준다 |
| 번호가 `0 .. 255` 뿐이다 (record 0 의 것) | **(c)** — `[1]` 이 최종 모델이다 |

표식이 곧 주소이므로 값 비교가 그대로 사상 진단이 됩니다.

**성공하면 (a 또는 b):** §4.6의 profile로 바꾸고, `PIMMapCoresPass`의 `cores != 1`
방어벽과 `PIMFormExecutionWavesPass`의 `tilesPerWave != 1` 방어벽을 확장합니다.
그때 §4.7의 "PIM Common IR을 건드리지 않는다"는 더 이상 성립하지 않습니다.
**얻는 것은 연산량이 아니라 분배 자유도입니다** — lane은 여전히 4096개입니다.

**실패하면 (c):** §4.4의 `[1]`이 최종 모델로 확정되고, `n < 4096` padding 문제는
§15 E9으로만 다룹니다.

</details>

### E6. saturation point sweep

§12.1의 격자 위에서 재 주십시오. Triton 없이 backend 구현 전에 먼저 하실 수 있고,
그 결과가 §12.4의 기본값이 됩니다. 표 `(m, k, n, batch) → GFLOPS`만 주시면 됩니다.

### E7. 커널 3개의 data type

§11 때문에 bf16으로 바꾸셔야 합니다.

| 파일 | 바꿀 곳 |
| --- | --- |
| `gemm.py` | 38행 `tl.int32`→`tl.float32`, 66행 `acc.to(tl.int32)`→`acc.to(tl.bfloat16)`, 77행 `torch.int32`→`torch.bfloat16`, 99–100행 `.to(torch.int8)`→`.to(torch.bfloat16)` |
| `split_k.py` | 34행 `tl.int32`→`tl.float32`, 50행 store 전 `acc.to(tl.bfloat16)`, 57행 `torch.int32`→`torch.bfloat16`, 73행 `.to(torch.int32)` 제거, 80–81행 `.to(torch.int8)`→`.to(torch.bfloat16)` |
| `grouped_gemm.py` | 45행 `tl.int32`→`tl.float32`, 62행 store 전 `acc.to(tl.bfloat16)`, 71행 `torch.int32`→`torch.bfloat16`, 124–125행 `torch.randint(...int8)`→`torch.randn(...).to(torch.bfloat16)` |

### E8. `compiler.py` profile 값 (§4.7)

바꾸기 전에는 `pim-map-cores`에서 멈춥니다.

**확인 방법:** 값만 바꾸고 AiMX pass가 없는 상태로 `pim_target="aimx"`로 돌리시면,
`--convert-pim-to-aimx`가 없다는 이유로 `_pim_common_to_aimx`의 subprocess가 실패해야
합니다. `pim-map-cores` 오류가 아니라 그 오류가 나오면 공통 pipeline이 끝까지
통과했다는 뜻입니다. 구현 시작 전에 확인할 수 있는 가장 좋은 관문입니다.

---

## 16. 요약

1. bank와 channel은 **명령을 따로 받는 단위가 아니라 주소로 데이터가 흩어지는
   단위**다. 그래서 tiling은 "각 유닛에 일을 배정"하는 것이 아니라 **"주소 범위를
   하드웨어의 고정된 사상에 맞게 자르는 것"**이다.
2. 그 사상이 상수와 정확히 맞물린다: `BURST_SIZE/TYPE_SIZE = ALU_SIZE`(bank),
   `PIM_WIDTH`(channel), `C_len`(device). 다만 **주소가 어느 축을 lane에 놓는지는 아직
   확정되지 않았다** (§1.4). 확정된 것은 A 재배치의 byte 순열
   `Arel[c*256+g*16+t] = A[16c+t]` 하나이고, 구현에 필요한 것도 그것뿐이다.
3. tiling은 lane → bank(16) → channel(256) → output block(4096) → execution tile →
   PIM Device 순으로 쌓이고, compiler의 자유도는 **`batch`, execution tile 크기,
   device 분할** 셋뿐이다.
4. `BLOCK_M/N/K`는 execution tile이 될 수 없다 (block tensor 한계). 여러 logical tile을
   묶는 일은 `--convert-pim-to-aimx`가 하고, 재료는 `pim.place`의 `capacity`다.
5. `m`은 `batch`, `k`는 256, **`n_t`는 `4096/batch`의 배수**여야 하고 (2026-08-11까지는
   4096으로 적혀 있었다 — 그보다 거친 눈금은 legality가 아니라 노는 lane이다;
   상한은 legality가 아니라
   PIM DRAM 예산이다 — 참조 구현이 `k=8192`·`n=28672`를 한 chain에 넣는다, §2.4),
   A의 행 pitch는 `padded_k`, B와 C의 행 pitch는 `padded_n`으로 서로 같아야 한다.
   어기면 오류가 아니라 조용한 오답이다. **padding은 정확성 요구사항이고**, 0으로
   채우는 것이 안전하다는 것은 `check_zero`로 증명된다 (§11.1).
6. 모든 execution tile을 같은 모양으로 만들고 PIM DRAM block을 한 벌만 잡으면
   **descriptor chain이 launch당 한 번만 만들어지고 주소 patching조차 필요 없다.**
7. `pim_core_shape`는 **"서로 다른 logical tile을 배정받을 수 있는 단위"의 격자**다.
   AiMX는 `[1]` + `lanes = 4096`이다 (§4.4). 지금 `"8,16"`이 담은 수는 128이지
   4096이 아니므로 이것은 옮겨 적기가 아니라 **고치기**다.
   **`[16]` 가설은 §15 E11로 닫혔다** — descriptor group의 record 16개 중 `next` 사슬이
   첫 번째만 걷는다는 것이 실측됐다 (§1.3). channel에 따로 명령을 줄 수단이 없으므로
   `[1]`이 최종이고, `PIMMapCoresPass`의 `cores != 1` 방어벽은 도달하지 않는다.
8. **지금 모델(`[1]`)에서는 PIM Common IR을 한 줄도 고치지 않는다.** 바꿀 것은 `compiler.py`의
   `_TARGET_PROFILE["aimx"]` 다섯 군데다 — `pim_core_shape`, `lanes`, `devices`,
   `capacity`, 그리고 `_profile_pass`의 두 줄. `devices`와 `capacity`를 빼먹으면
   PIM Device 2048개로 쪼개진 IR과 UPMEM MRAM의 32 MiB 예산을 조용히 받는다 (§4.7).
   **§15 E11이 (c)로 닫혔으므로 이 항목은 확정이다** — 공통 IR을 확장할 이유가 없다.
9. descriptor는 참조 구현의 API가 아니라 ISA에서 직접 낸다. 개선점 다섯 중 셋은 지금
   적용 가능하고(A 재배치 hoisting, 문제 전치, **host가 `Arel`을 직접 만들기**),
   하나는 확인됐으며(재배치 융합), 하나는 확인이 필요하다(reduction 분할).
11. **누산기는 fp32가 아니다.** 지수 칸이 2 이상 벌어지면 작은 항을 통째로 버리므로
    한 내적 안에서 항끼리 `2^16`배를 넘으면 안 된다 (§11.1). compiler가 CPU 참조와의
    bit 단위 일치를 약속해서는 안 된다.
12. **descriptor 생성이 hardware 실행보다 비싸다** — 참조 구현은 시간의 52~63%를
    hardware 밖에서 쓴다. §7의 chain 재사용과 §8 I5의 descriptor 제거가 향하는 곳이
    그 부분이고, 이론상 상한은 2.4~2.7배다.
10. autotuning의 목적 함수는 GFLOPS가 아니라 **padding을 포함한 wall clock**이고,
    device 수도 tuning 변수다.

---

## 17. 구현 상태 (branch `aimx-target-lowering`)

이 절은 실제로 만들어진 것과, 위 설계에서 **바뀐 결정**만 적는다. 아직 하드웨어에서
한 번도 돌려 보지 않았다.

### 17.1 만들어진 것

| | 자리 |
| --- | --- |
| AiMX dialect | `include/triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.td`, `lib/Dialect/PIMTarget/AiMX/IR/AiMXDialect.cpp` |
| `--convert-pim-to-aimx` | `lib/Conversion/PIMToAiMX/PIMToAiMXPass.cpp` |
| `--aimx-lower-matmul` | `lib/Conversion/PIMToAiMX/AiMXLowerMatmulPass.cpp` |
| `--aimx-to-descriptors` | `lib/Target/AiMX/AiMXToDescriptors.cpp` |
| `--aimx-to-runtime-stub` | `lib/Target/AiMX/AiMXToRuntimeStub.cpp` |
| runtime shim | `pim_backend/backend/aimx_runtime.h` |
| profile · pipeline | `pim_backend/backend/compiler.py` (§4.7 다섯 군데) |
| launcher 빌드 | `pim_backend/backend/driver.py` `_build_aimx_so` |
| 시험 | `test/Conversion/PIMToAiMX/convert.mlir`, `test/Target/AiMX/codegen.mlir` |

**PIM Common IR은 한 줄도 고치지 않았다.**

### 17.2 바뀐 결정 — buffer 크기만 compile time이면 되고, tile 모양은 아니다

§1.8 (h) 때문에 **buffer 크기**는 compile time에 확정되어야 한다 — `pim_malloc`은 실패해도
NULL을 돌려주지 않고 프로세스를 죽이므로 할당해 보고 물러서는 길이 없다.

이 절은 한때 거기서 "**그러므로 execution tile 모양도 compile time 상수여야 한다**"를
끌어냈고, `m_t=16, k_t=4096, n_t=4096`을 문제 크기와 무관하게 박았다. **그 추론이
틀렸다.** `--aimx-lower-matmul`이 만드는 descriptor program은 이미 모양에 무관하다 —
`AiMXLowerMatmulPass`가 `#aimx.tiling`에서 compile time 상수로 읽는 것은 `batch`
하나뿐이고, `m_t`·`k_t`·`padded_k`·`padded_n`은 전부 program의 block argument다.

```cpp
int64_t batch = t.getBatch();          // ← compile time 인 것은 이것뿐
Value mT = body->getArgument(0),       // ← 나머지는 생성된 C 함수의 인자다
      kT = body->getArgument(1),
      paddedK = body->getArgument(2);
auto sLoop = p.loop(kT, aimx::kRegSize);
```

그래서 두 값을 분리한다.

| | 무엇 | 정하는 곳 |
| --- | --- | --- |
| `k_max`, `n_max` | `B`·`Arel`·chain **buffer를 잡는** 크기 | compile time. 예산이 정한다 |
| `k_t`, `n_t` | descriptor program이 **실제로 도는** 길이 | 런타임. `pim.place`의 크기에서 온다 |

`Arel`의 행 간격은 `k_t × 2 × 16`이고 relayout program과 compute program이 같은 `k_t`를
받으므로 자동으로 맞는다. `desc_last`도 런타임 `next`에서 계산되므로 chain이 짧아지면
따라온다. 즉 shape specialization(커널 signature 변경, `K`마다 재컴파일)이 필요 없다.

- `batch` ← `pickBatch` (기본값 0 = Triton block이 허락하는 최대값)
- `m_t` = `BLOCK_M × group-m` — Triton block을 묶은 것이다 (§6.1)
- `k_t` ← `min(roundUp(k, 256), k_max)`, `n_t` ← `min(roundUp(N, 4096/batch), n_max)` — 런타임
- host는 `ceil(N/n_t) × ceil(k/k_t) × ceil(M/m_t)` 번 돈다

**`k_t > k_max`이면 k를 여러 tile로 나눈다 (2026-08-12).** 예전에는 이 자리에서 stub이
멈췄다. 지금은 `k_t`를 `k_max`에서 clamp하고 host loop이 `ceil(k/k_t)`번 돈다.

k tile은 **전부 같은 C 영역**에 쓰므로 — n이나 m을 나눌 때와 다른 점이다 — 두 번째
tile부터는 덮는 것이 아니라 이전 부분합에 더해야 한다. 그 신호가 `aimx.unstage`의
`accumulate` operand(런타임 값)이고, host가 f32면 그것이 곧 fp32 누산이다. 첫 tile이
대입인 이유는 host의 C가 0이라는 보장이 없기 때문이다 — Triton kernel의 출력 포인터다.

**이 경로는 손해이고 마지막 수단이다.** k 전체가 한 chain에 들어가면 reduction이 vACC
안에서 끝나고 bf16 반올림이 출력당 한 번뿐이다. 나누면 C 전송과 C drain descriptor가
tile 수만큼 곱해지고 반올림도 그만큼 는다. vACC는 **출력 block 하나 깊이**(주소가
`pmm + VACC_OFFSET` 하나이고 `matmul.c:475`의 drain이 `p`·`r` loop 안에 있다)라 invoke를
건너 들고 있을 수 없으므로, 이것은 피할 수 있는 비용이 아니다.

그래서 `k_max`는 예산이 허락하는 한 크게 둔다. 1 GiB, `batch=8`, `m_t=16`, `n_max=4096`
에서 소요가 `9256 × k + 335872` byte이므로 `k_max`는 **115,712**까지 잡힌다 —
트랜스포머의 `k`(4096, 8192, 28672)는 전부 한 tile에 들어가고 이 경로는 돌지 않는다.

**구현됨 (2026-08-10).** `pickTile`이 `k_max`·`n_max`만 고르고, `buildPlacement`가
`pim.place`에서 런타임 `k_t`·`n_t`·`padded_k`를 만들어 `aimx.build_descriptors`와
`aimx.stage`에 넘긴다. host loop은 **n → k → m**이다 (2026-08-12에 k가 끼어들었다).
가장 큰 buffer인 B가 `(k, n)`에만 의존하므로 그 둘이 바깥이고, 훨씬 작은 A는 안쪽에서
`m`마다 올라간다. k를 가장 안쪽에 두면 B가 m tile마다 다시 올라간다.

| option | 기본값 | 뜻 |
| --- | --- | --- |
| `group-m` | 1 | `m_t = BLOCK_M × group-m`. 전송량 식에서 `m_t`는 A의 `ceil(M/m_t)·m_t` 항에만 나타나므로 작을수록 좋다 |
| `tile-k` | **32768** | `k_max`. 1 GiB 예산의 27%(277 MiB)를 쓰고 현실적인 `k`를 전부 덮는다. `0`이면 예산에 들어가는 최대값을 이분 탐색으로 고른다 |
| `tile-n` | 4096 | `n_max`. 키우면 A를 다시 모으는 횟수가 줄고 B buffer가 커진다 |

세 축이 모두 상한에서 clamp되므로 `aimx.stage`/`aimx.unstage`가 내는
`aimx_require`(`pad_to[0] * pitch * 2 ≤ alloc bytes`)는 이제 도달하지 않는다. 그래도
남겨 둔다 — tile 모양이 런타임 값이라 "이 buffer를 넘었는가"가 유일하게 의미 있는
경계이고, 위 clamp 중 하나라도 깨지면 조용히 잘리는 대신 여기서 멈춘다.

descriptor chain buffer는 검사가 필요 없다. group 수가 `k`와 `padded_n`에 대해 단조
증가하고 `k_t ≤ k_max`, `n_t ≤ n_max`이므로 할당량을 넘을 수 없다.

### 17.3 바뀐 결정 — 예산은 선언 상수다

`compiler.py`의 `AIMX_DEVICE_DRAM_BYTES = 1 GiB`이다. 보수적으로 declared한 값이고,
진짜 값은 driver가 찍는 BAR 길이(`dmesg | grep "PIM_MEM_.*BAR #4"`)에서 읽어 올려야 한다.
실험용으로 `--convert-pim-to-aimx=bank-capacity-bytes=…`도 받는다 (§5).

§6.1의 새 알고리즘에서 이 값은 `k_max`·`n_max`를 정한다. `m_t = 16`, `n_max = 4096`,
`batch = 1`에서 전체 소요는 대략 `8864 × k_max + 336 KB`이므로 1 GiB에서 `k_max`는
118,784까지 잡힌다 — 현실적인 `k`는 전부 덮는다. `k_max = 32768`이면 277 MiB로 예산의
27%만 쓴다.

### 17.4 아직 하지 않은 것

| | 왜 |
| --- | --- |
| §2.6 전치 | 규칙이 `m_dev > n_dev` 비교인데 둘 다 런타임 값이라 compile time에 답이 없다. IR은 이미 표현할 수 있다 (`aimx.stage`의 `transpose`, `#aimx.tiling`의 `transposed`) |
| §8 I1 재배치 hoisting | 지금은 참조 구현과 똑같이 **chain 하나**에 재배치와 계산을 이어 붙이고 tile마다 한 번 `pim_exec`한다. chain을 둘로 나누는 것은 하드웨어에서 확인된 적이 없다. host 쪽 A 모으기만 `n` loop 밖으로 뺐다 |
| §8 I3 재배치 융합 | verifier의 whitelist 6행에 자리는 있다. 512 byte 덩어리가 한 칸 순환하는 보정이 아직 실측되지 않았다 |
| §8 I5 host가 `Arel` 생성 | 순열이 `k > 256`, `m > batch`로 확장되는지가 §15 E10이다 |
| §8 I4 reduction 분할 | §15 E3 |
| host split-K (출력이 3차원) | `split_k.py`가 이 모양이다. C의 `pim.place`가 rank 3이고 `axis_sources`에 `explicit_k_partition`이 붙는다. 그 축은 **reduction이 아니라 분리**다 — slice마다 독립된 matmul이고 host가 `c.sum(dim=0)`으로 더한다. 그래서 §8 I4와 달리 하드웨어 확인이 필요 없고 host loop 한 겹이면 되지만, `k_t`가 덮어야 하는 것이 전체 `k`에서 **partition 하나당 `k`**로 바뀌므로 §6.1을 함께 고쳐야 한다. 지금은 진단을 낸다 |
| `batch > 1` | §15 E9(`n_t < 4096` legality)가 닫히기 전에는 의미가 없다 (§2.5). IR과 pass는 이미 받는다 |

### 17.5 확인해야 하는 것

1. **빌드.** `triton-shared-opt --convert-pim-to-aimx --aimx-lower-matmul`,
   `triton-shared-translate --aimx-to-{descriptors,runtime-stub}`
2. **lit.** `test/Conversion/PIMToAiMX/convert.mlir`, `test/Target/AiMX/codegen.mlir`
3. **생성된 descriptor stream이 `_matmul_silent`와 byte 단위로 같은가.** 같은 `m/k/n`에서
   양쪽 chain을 덤프해 대조하는 것이 가장 확실한 시험이고, 하드웨어 없이도 된다
4. **`AIMX_PIM_SW`를 준 상태에서 stub이 g++로 컴파일되는가** (`pim_math.h`는 설치되지 않는다)
5. 그다음이 하드웨어다. 그 전에 §15의 canary(`./matmul`)가 `Correctness check success`를
   내야 한다 — 지금 `/usr/lib/libpim.so`가 다른 트리 빌드일 수 있다

### 17.6 runtime 헤더를 C++에서 쓸 때 — 실측으로 확인한 세 가지

생성된 stub은 `g++`로 빌드된다(진입점이 `extern "C" void <kernel>(...)`이고 launcher가
`ctypes.CDLL`로 연다). 그런데 `pim_math.h`는 C로 쓰였고 C++에서 읽으면 세 곳이 걸린다.
셋 다 `pim_backend/backend/aimx_runtime.h`가 흡수한다.

| | 증상 | 원인 | 처리 |
| --- | --- | --- | --- |
| (a) | 링크는 되는데 `dlopen`이 `undefined symbol: _Z9_pim_exec9exec_args` | **`pim_math.h`에는 `extern "C"` 가드가 없다.** `pim.h`는 있다. 그래서 `pim_exec`이 부르는 `_pim_exec`만 C++ 이름으로 망가지고, `libpim.so`가 내보내는 C 이름과 만나지 못한다 | `#include "pim_math.h"`를 `extern "C" { }`로 감싼다 |
| (b) | 컴파일이 `narrowing conversion of '3735928481'`로 멈춘다 | `decode_opcode`의 `case (CODE_READ)` 등 네 개가 `0xdeadbea1`~`0xdeadbea4`인데 switch 조건이 `int`다. C에서는 경고, C++에서는 오류다 | 빌드에 `-Wno-narrowing -Wno-write-strings`. 걸리는 곳은 전부 우리가 부르지 않는 진단 문자열이다 |
| (c) | 커널 이름이 `matmul`이면 진입 함수 선언이 통째로 깨진다 | `pim.h`가 연산마다 함수형 매크로를 둔다 (`matmul`, `gemm`, `dot_prod`, `elewise_*`, `bias_*`, `memcpy_*` 등 15개). stub의 진입 함수 이름이 커널 이름 그대로다 | 그 15개를 `#undef`한다. `pim_exec` / `pim_malloc` / `VA2PA` / `PIM_RD_INSTR_MC`처럼 실제로 부르는 매크로는 남긴다 |

host 원소 타입도 여기서 흡수한다. PIM DRAM은 언제나 bfloat16이고 host는 셋 중 하나다.

| host | stage | unstage |
| --- | --- | --- |
| `bf16` | 16 bit 그대로 옮긴다 | 그대로 |
| `f32` | `aimx_to_bf16` | `aimx_from_bf16` |
| `f16` | `aimx_to_bf16(aimx_from_fp16(·))` | `aimx_to_fp16(aimx_from_bf16(·))` |

**f16을 16 bit 그대로 옮기면 안 된다** — 지수 폭이 5 bit와 8 bit로 다르다. `aimx_from_fp16`은
IEEE half 65,536개 전부에 대해 bit 단위로 대조했다.
