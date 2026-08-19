# AiMX 실험 절차 — E1 ~ E8

`AiMXTarget_Design.md` §15 의 확인 항목들을 **hardware 가 막히지 않는 방식으로** 실행하는 절차입니다.
빌드·실행·검증은 전부 직접 하시고, 이 문서는 무엇을 어떻게 하면 되는지만 적습니다.

용어는 `AiMXTarget_Design.md` §0 을 따릅니다. 행렬 축은 m(출력 행) / k(reduction) / n(출력 열) 입니다.
참조 구현의 `p_size` / `q_size` / `r_size` 가 각각 m / k / n 입니다.

---

## 0. 결론부터

| | |
| --- | --- |
| **프로그램은 항상 정상 종료된다** | driver 의 대기 loop 이 `CDMA_MAX_POLLING = 10,000,000` 회로 유한하다. 무한 대기가 아니다. |
| **그런데도 hardware 가 막힌다** | timeout 경로가 **정리 코드를 건너뛰고 반환**하기 때문이다. 프로세스는 죽어도 FPGA 는 실행 중 상태로 남는다. |
| **판정은 dmesg 한 줄로 한다** | `PL_DMA] mm_store_stat_reg_signal polling timeout!!` 이 나오면 더러워진 것이다. |
| **reboot 전에 회복 사다리가 있다** | pim_drv 재적재가 CDMA soft reset 을 다시 실행한다 (§2.1). |

그래서 실험의 안전 기준은 "프로그램이 끝났는가" 가 아니라 **"PDMA 가 idle 로 돌아왔는가"** 입니다.
§1 에서 그 지점을 정확히 짚고, §2 에서 규칙을 세우고, §4 부터 실험별 절차를 적습니다.

---

## 1. 어디서 막히는가 — 코드로 확인한 지점

### 1.1 `DMA_START` 의 실제 경로

`pim_sw/src/drv_src/dma_fops_pl.c:387-461`

```c
case DMA_START:
    /* ── 준비: PIM 을 켠다 ───────────────────────────── */
    mm_clr_intc_signal[0]            = 0;
    mm_clr_piu_reg_signal[0]         = 0;
    mm_set_pim_working_signal[0]     = 0;              // PIM working
    mm_store_piu_code_addr_signal[0] = pim_args.desc_base_pa;
    mm_store_pmm_signal[0]           = va_to_pa(pim_args.pmm_base_addr);

    /* ── 실행: head 를 쓰는 순간 PDMA 가 chain 을 걷기 시작한다 ── */
    mm_store_tail_desc_addr_signal[0] = pim_args.desc_last;
    mm_store_head_desc_addr_signal[0] = pim_args.desc_base_pa;

    /* ── 대기: 유한 loop ─────────────────────────────── */
    for (j = 0; j < CDMA_MAX_POLLING; j++) {
        stat_val = ioread32((void *)mm_store_stat_reg_signal);
        if (stat_val == 0) break;
    }
    if (j >= CDMA_MAX_POLLING) {
        printk(KERN_ERR " PL_DMA] mm_store_stat_reg_signal polling timeout!! ...");
        return -EIOCTL;                    // ★ 여기서 반환하면
    }

    /* ── 정리: timeout 경로는 이 세 줄을 건너뛴다 ────────── */
    mm_clr_piu_reg_signal[0]     = 0x0;    // ★
    mm_set_pim_working_signal[0] = 0x0;    // ★
    mm_clr_intc_signal[0]        = 0x0;    // ★
```

**정상 경로**: PDMA 상태 레지스터가 0 이 되면 loop 을 빠져나오고 세 줄의 정리를 실행한다 → hardware clean.

**timeout 경로**: `-EIOCTL` 을 반환하고 세 줄을 실행하지 않는다 →
FPGA 는 PIM working 이 걸린 채, PDMA 는 chain 중간에서 멈춘 채 남는다.
그 상태에서 다음 프로그램이 QDMA 로 데이터를 넣으려 하면 막힌다. 이것이 겪으신 증상입니다.

`MEMCPY_SG` 경로(`dma_fops_pl.c:522-560`)도 같은 유한 loop 을 쓰지만
`mm_set_pim_working_signal` 을 **건드리지 않습니다**. 그래서 timeout 이 나도 남는 상태가 더 적습니다.
이 차이가 §2.3 의 실험 vehicle 선택 근거입니다.

### 1.2 프로세스는 왜 항상 끝나는가

`CDMA_MAX_POLLING = 10,000,000` 회의 `ioread32` 는 PCIe BAR 위의 non-posted read 이므로
1 회당 대략 1 µs, 전체로 **십 초 ~ 수십 초** 걸립니다. `cond_resched()` 가 없는 busy loop 이라
그동안 해당 CPU core 하나가 kernel 에서 점유되지만, 시스템이 죽지는 않고 ioctl 은 반드시 반환합니다.

그러면 사용자 코드는 이렇게 흘러갑니다.

```c
mpim_exec()  →  ioctl 이 -1        →  "DMA transaction failed" 출력
             →  pthread_mutex_unlock (누수 없음)
             →  _matmul_silent 의 goto free  →  pim_free 들  →  return 0
main()       →  return 0
```

**프로세스는 exit code 0 으로 정상 종료합니다.** 그래서 종료 여부로는 판정할 수 없습니다.
반드시 dmesg 를 봐야 합니다.

> `timeout 180 ./probe ...` 를 걸어도 SIGKILL 은 ioctl 안에서는 즉시 듣지 않습니다.
> kernel 이 loop 을 다 돌고 나온 뒤에야 신호가 처리됩니다. `timeout` 은 최후의 보호막일 뿐,
> 실제 상한은 driver 의 유한 loop 입니다.

### 1.3 무엇이 PDMA 를 멈추게 하는가

세 가지뿐입니다.

| 원인 | 결과 | 위험도 |
| --- | --- | --- |
| **A** `desc_base_pa` / `desc_last` / `next` 가 buffer 밖을 가리킨다 | PDMA 가 쓰레기 descriptor 를 읽고 아무 주소로 AXI 요청을 낸다 | **막힘** |
| **B** operand 조합이 불완전하다 (예: A 만 주고 B 를 안 준 상태로 C 를 요구) | PIM datapath 가 오지 않을 피연산자를 기다린다 | **막힘** |
| **C** 주소·크기·flag 가 규칙을 어긴다 (§2.2) | descriptor 자체는 완결되므로 **결과만 틀린다** | 안전 |

C 는 안전합니다. 값이 틀릴 뿐 hardware 는 깨끗하게 끝납니다.
따라서 **모든 실험을 C 유형으로만 설계하는 것**이 목표이고, §2.2 의 guard 가 A 를,
§2.4 의 실험 순서가 B 를 막습니다.

---

## 2. 안전 규칙

### 2.1 회복 사다리 — reboot 전에 이것부터

`dma_probe.c:379-383` 에서 pim_drv 는 적재될 때마다 CDMA 를 soft reset 하고
(`dma_probe.c:419-424`) address mapping 을 다시 프로그램합니다. 즉 **driver 재적재가 곧 hardware 초기화**입니다.

```bash
# 1단계 — PIM driver 재적재 (CDMA reset + address map 재설정)
cd ~/pim_aimx_sw/pim_sw && sudo ./build.sh 0

# 2단계 — QDMA driver 재적재
cd ~/pim_aimx_sw/xilinx-qdma && sudo ./init.sh

# 3단계 — 둘 다
cd ~/pim_aimx_sw && sudo ./re_init.sh

# 4단계 — 그래도 안 되면 reboot
```

각 단계 뒤에 **canary**(§2.5)를 한 번 돌려서 회복되었는지 확인하십시오.
어느 단계에서 회복되는지 한 번만 기록해 두면, 앞으로 reboot 횟수가 크게 줄어듭니다.

> `build.sh 0` 은 `rmmod src/drv_src/pim_drv.ko` → `insmod` 만 합니다. build 를 다시 하지 않으므로 빠릅니다.
> 반드시 `pim_sw/` 디렉터리에서 실행해야 경로가 맞습니다.

### 2.2 실행 직전 guard — 붙여넣어 쓰십시오

`pim_exec` / `pim_copy` 를 호출하기 **직전에** 이 두 함수를 통과시키고, 하나라도 실패하면 호출하지 않습니다.
§1.3 의 A 유형(진짜 막히는 원인)을 전부 잡습니다.

```c
/* descriptor chain 의 구조 검사 */
static int aimx_guard_chain(pim_args *a, pim_isa_32_t *rec, uint32_t n_rec)
{
    const uint64_t GROUP = 0x20ULL * NUM_CHANNELS;   /* 512 byte = descriptor group 하나 */
    uint64_t base = a->desc_base_pa;
    uint64_t lo   = base + GROUP;                    /* next 가 가질 수 있는 최소값 */
    uint64_t hi;
    uint32_t g, i;

    if (n_rec == 0 || (n_rec % NUM_CHANNELS) != 0) return -1;  /* group 단위가 아니다 */
    g  = n_rec / NUM_CHANNELS;
    hi = base + GROUP * g;                           /* next 가 가질 수 있는 최대값 */

    /* tail 은 마지막 group 의 첫 record 주소여야 한다 */
    if (a->desc_last != base + GROUP * (g - 1)) return -2;

    for (i = 0; i < n_rec; i++) {
        uint64_t nxt = ((uint64_t)rec[i].next_h << 32) | rec[i].next_l;
        if (rec[i].len == 0)                     return -3;
        if (nxt != 0 && (nxt < lo || nxt > hi))  return -4;
    }
    return 0;
}

/* 형상 legality 검사 — 어기면 값이 틀린다 (막히지는 않는다) */
static int aimx_guard_shape(uint32_t m, uint32_t k, uint32_t n, uint32_t batch)
{
    uint32_t block = PIM_WIDTH * NUM_CHANNELS / batch;   /* output block 한 개의 byte */
    if (k % REG_SIZE)                  return -1;        /* k 는 256 의 배수 */
    if (m % batch)                     return -2;
    if ((n * TYPE_SIZE) % block)       return -3;
    if ((n * TYPE_SIZE) / block == 0)  return -4;        /* r_loop_var == 0 → C 를 아예 안 쓴다 */
    return 0;
}
```

`desc_last` 규칙의 근거: `next_desc` 는 `desc_base_pa + GROUP` 에서 시작해 group 마다 `GROUP` 씩 증가하므로,
group 을 `g` 개 쓰고 나면 `next_desc = base + GROUP*(g+1)` 입니다.
`matmul.c` 의 `next_desc - 0x20*NUM_CHANNELS*2` 는 정확히 `base + GROUP*(g-1)`, 즉 마지막 group 의 시작입니다.
(`_memcpy_tp` 계열은 마지막 group 의 `next` 를 0 으로 놓고 `desc_last` 도 0 으로 두는 다른 관례를 씁니다.
두 관례를 **한 chain 안에서 섞지 마십시오**.)

그리고 참조 구현의 이 줄을 절대 지우지 마십시오.

```c
assert(desc_idx == num_descr);      /* matmul.c:179 */
```

descriptor 를 추가하면서 `num_descr` 갱신을 잊으면 이 assert 가 **launch 전에** 프로그램을 죽입니다.
hardware 를 건드리기 전에 죽는 것이 가장 안전한 실패입니다.

### 2.3 `pim_exec` 과 `pim_copy` 는 위험도가 다르다

| | `pim_exec` (DMA_START) | `pim_copy` (MEMCPY_SG) |
| --- | --- | --- |
| PIM working signal | 켠다 → timeout 시 켜진 채 남는다 | 건드리지 않는다 |
| PMM base 주소 | 쓴다 | 쓰지 않는다 |
| `OPCODE_PIM_MODE` descriptor | 필요 | 참조 구현에 전례 없음 |
| timeout 시 남는 상태 | 많다 | 적다 |

참조 구현의 사용 근거:

- `_memcpy_tp` (`memcpy.c:216-228`) — `PIM_MODE` 없음 → `pim_copy`
- `_memcpy_tp_bc` (`memcpy.c:158-173`) — `PIM_MODE` 있음 → `pim_exec`

**규칙: `OPCODE_PIM_MODE` 가 없는 순수 데이터 이동 실험은 `pim_copy` 로 하십시오.**
E2 와 E4-b 의 앞 절반이 여기 해당합니다.

### 2.4 실험 순서 — 위험도 오름차순

이 순서를 지키면 위험한 실험을 할 때는 이미 회복 사다리가 검증되어 있습니다.

| 순서 | 실험 | hardware launch | 위험도 |
| --- | --- | --- | --- |
| 1 | **E7** 커널 bf16 | 없음 (Triton 만) | 없음 |
| 2 | **E8** `compiler.py` profile | 없음 (compile 만) | 없음 |
| 3 | **E5** PIM DRAM 예산 | 없음 (`pim_malloc` 만) | 없음 |
| 4 | **E1** channel 수 | canary 로 해결 | 없음 |
| 5 | **E6** saturation sweep | 검증된 chain 만 | 없음 (§2.2 guard 필수) |
| 6 | ~~**E2** stride shift~~ | — | **닫힘. 답은 `<< 40`** (§4 E2) |
| 7 | **E11** channel 별 record | `pim_copy` | 낮음 |
| 8 | **E3-a** vACC preload | `pim_exec`, 기존 구조 유지 | 낮음 |
| 9 | **E4-b** 재배치 합치기 | `pim_exec`, 재배치만 | 중간 |
| 10 | **E3-b** `OPCODE_HOLD` | `pim_exec`, 미지의 flag | 중간 |
| 11 | **E4-c** 재배치 완전 융합 | `pim_exec`, 미지의 조합 | **높음** |

**E11 절차는 `AiMXTarget_Design.md` §15 E11에 있습니다.** vehicle 이 `pim_copy` 이고
`OPCODE_PIM_MODE` 를 쓰지 않으므로 §2.3 규칙상 이 목록에서 가장 안전한 축에 듭니다.

**한 번에 한 곳만 바꾸십시오.** 그리고 **descriptor 를 삭제하지 마십시오** — §1.3 의 B 유형이
막히는 유일한 논리적 원인입니다. A / B / C 삼중조에서 하나를 빼면 PIM datapath 가 기다립니다.
flag 를 더하거나 주소를 바꾸는 것은 삼중조를 깨지 않습니다.

### 2.5 canary 와 판정 script

**canary** — 가장 작은 합법 형상. batch 분해 규칙상 m=1 이면 batch=1 만 실행됩니다.

```bash
./matmul 1 256 4096 0
```

descriptor group 은 `2·(256/256) + 1·1 + 1·1 + 1 = 5` 개, record 80 개입니다.
CPU 참조 계산도 1M 연산이라 즉시 끝납니다.

batch=8 canary 는 `./matmul 8 256 4096 0` (group 138 개) 입니다.

**판정 script** — `~/bin/safe_run.sh` 로 두고 모든 실행을 이걸로 감싸십시오.

```bash
#!/bin/bash
# 한 번 실행하고 hardware 가 깨끗한지 판정한다
set -u
sudo dmesg -C
timeout 180 "$@"; rc=$?
echo "---- exit code: $rc"

if dmesg | grep -q "polling timeout"; then
    echo "!! PDMA 가 idle 로 돌아오지 않았습니다."
    echo "!! 다음 실행을 하지 마시고 회복 절차(문서 §2.1)를 실행하십시오."
    dmesg | tail -30
    exit 1
fi
if dmesg | grep -qE "ERROR DMA|DMA Timeout|DMA Error|Decoding error|Slave error|Internal error"; then
    echo "!! CDMA error 가 보고되었습니다."
    dmesg | tail -30
    exit 1
fi
echo "OK: hardware clean"
```

사용:

```bash
chmod +x ~/bin/safe_run.sh
~/bin/safe_run.sh ./matmul 1 256 4096 0
```

**모든 실험은 이렇게 감쌉니다.**

```
canary → OK 확인 → 실험 실행 → OK 확인 → canary → OK 확인
```

앞 canary 가 실패하면 hardware 가 이미 더러운 것이므로 실험 결과를 믿을 수 없습니다.
뒤 canary 가 실패하면 실험이 무언가를 남긴 것이므로 그 실험은 되돌립니다.

### 2.6 테스트 데이터 설계 — bit-exact 하게 만들 것

PIM 의 누산기는 표준 bf16 이 아니라 자체 형식(`pim_mac_simul.h`)입니다.
난수를 쓰면 "값이 조금 다르다" 와 "주소가 틀렸다" 를 구분할 수 없습니다. 그래서 이렇게 만듭니다.

**one-hot A 를 쓰십시오.**

```
A[0][j] = (j == j0) ? 1.0 : 0.0
```

이면 `C[0][:] = B[j0][:]` 이고, 곱셈·덧셈이 전부 정확합니다.
결과가 틀리면 **다른 값**이 아니라 **B 의 다른 행**이 나오므로, 그 행 번호가 hardware 가 실제로
읽은 주소를 그대로 알려줍니다. 값 비교가 주소 진단이 됩니다.

**B 는 두 번에 나눠 표식을 넣으십시오.** bf16 은 256 까지 정수를 정확히 표현합니다.

| pass | B 값 | 알려주는 것 |
| --- | --- | --- |
| 1 | `B[j][i] = i % 256` | lane(출력 열 n) 사상 |
| 2 | `B[j][i] = j % 256` | reduction 단계(k) 사상 |

두 pass 다 bit-exact 이므로 `!=` 비교로 바로 판정합니다.

---

## 3. 실험 도구 — `libpim.so` 를 고치지 않는 probe

`PIM_RD_INSTR_MC` 등은 `pim_math.h` 안의 `static inline` 이고,
`pim_exec` / `pim_copy` / `pim_malloc` / `VA2PA` / `datacopy_cpu2pim` 은 `libpim.so` 가 export 합니다.
따라서 **응용 프로그램이 descriptor chain 을 직접 만들 수 있습니다.**
검증된 `_matmul_silent` 를 건드리지 않으므로, 실험이 잘못돼도 canary 는 항상 원래대로 돕니다.

**`~/kangmlee/test/` 에 이미 만들어져 있습니다.** `~/pim_aimx_sw` 는 **읽기만** 합니다 —
include 경로로만 쓰고 그 아래에는 아무것도 만들지 않습니다.

```
~/kangmlee/test/
    common/aimx.h      공용 — bf16, descriptor 문맥, guard, chain 덤프,
                       _matmul_silent 를 옮긴 chain builder, one-hot 데이터
    common/rules.mk    빌드 규칙 (PIMSW 는 include 경로로만 쓴다)
    safe_run.sh        한 번 실행하고 hardware 가 깨끗한지 판정
    recover.sh         reboot 전 회복 사다리
    e0_diff/ e1_canary/ e2_stride/ e3a_vacc/ e3b_hold/
    e4b_merge/ e4c_fuse/ e5_capacity/ e6_sweep/ e7_bf16/ e8_profile/
    e9_memcheck/ e11_perch/
```

```bash
cd ~/kangmlee/test
make syntax     # libpim 없이 문법만 먼저
make            # probe 들을 빌드한다
```

C 로 쓰십시오. `pim_math.h` 는 compound literal(`(struct instr_args){...}`)을 쓰므로 C 가
자연스럽습니다. 링크에 `-lm` 이 필요합니다 — `libpim.so` 가 `pow` 를 미해결로 남깁니다.

새 실험은 디렉터리 하나에 `main.c` + 두 줄짜리 `Makefile` + `run.sh` 면 됩니다.
descriptor 생성부에 `_matmul_silent` (`matmul.c:105-176`) 를 옮겨 놓고 해당 실험이 지정한
**한 줄만** 바꾸십시오. `common/aimx.h` 의 `aimx_matmul_emit` 이 그 옮겨 놓은 판이고
`aimx_knob` 의 필드 하나가 실험 하나에 대응합니다.

> E11 은 MAC 을 지나지 않는 순수 복사라 bf16 이 아니라 `short` 를 그대로 표식으로
> 씁니다. 그래서 표식이 0..4095 로 전부 구별됩니다. 나머지 실험은 `aimx_f2b` /
> `aimx_mark` 를 씁니다 (0~256 정수만 bf16 에서 정확하므로).

---

## 4. 실험별 절차

### E1. channel 8 개 vs `NUM_CHANNELS = 16`

**새로 할 일이 없습니다.** canary 가 이미 답합니다.

`./matmul 1 256 4096 0` 은 n=4096 열을 output block 하나로 계산합니다
(`r_loop_var = 4096·2 / (512·16/1) = 1`). 이게 `Correctness check success` 를 내면
**PIM Device 하나가 lane 4096 개를 동시에 가진다**는 것이 확인된 것입니다.
그 4096 개가 물리 channel 8 개에 담기든 16 개에 담기든, descriptor 산술에 쓰는 상수는
`PIM_WIDTH · NUM_CHANNELS = 8192 byte` 하나이고 그 값은 바뀌지 않습니다.

참고로 보고 싶으시면 driver 적재 로그에 address mapping 이 찍힙니다.

```bash
cd ~/pim_aimx_sw/pim_sw && sudo ./build.sh 0
dmesg | grep -i "Address Mapping"
```

`dma_probe.c:419-424` 가 `CH_NUM`(=16, `pim.h:27`) 에 따라 `ROBACOCH` 를 씁니다.
두 번째 줄이 `ROBACOCH` 면 fabric 이 16-channel interleave 로 설정된 것입니다.

**결정하는 것**: 설계 문서 §1.2 의 "PIM Device = 8192 byte = lane 4096" 을 확정합니다.
canary 가 성공하면 그대로 갑니다.

---

### ~~E2. stride 필드의 shift 폭~~ — **닫힘. `pim_exec` 경로는 `<< 40` 입니다**

| 위치 | shift | 호출 |
| --- | --- | --- |
| `matmul.c:143, 163, 170, 312, 325, 337, 344` | **`<< 40`** | `pim_exec` |
| `memcpy.c:55` (`_memcpy_sg`) | `<< 48` | `pim_copy` |
| `memcpy.c:162` (`_memcpy_tp_bc`) | `<< 48` | `pim_exec` |

`_matmul_silent` 가 `Correctness check success` 를 내는 것이 `pim_exec` + `matmul` opcode
경로에서 `<< 40` 이 맞다는 증거입니다. 구현(`AiMXDialect.h` 의 `kStrideShift`)도 40 입니다.
`memcpy.c` 의 `<< 48` 은 다른 관례이므로 **한 chain 안에서 섞지 마십시오.**

아래 절차는 `pim_copy` 경로의 `<< 48` 을 따로 확인하고 싶을 때만 쓰십시오.

**vehicle**: `pim_copy`. `PIM_MODE` 가 없는 순수 복사이므로 §2.3 규칙에 따라 가장 안전합니다.

**절차**:

1. PIM DRAM 에 `src` 를 잡고 `src[row][col] = f2b(row)` 로 채웁니다. 행 pitch 는 `PITCH = 4096` byte 로 잡습니다.
2. `dst` 를 0 으로 채웁니다.
3. descriptor 하나를 만듭니다. `SHIFT` 만 40 / 48 로 바꿔 두 번 돌립니다.

```c
uint64_t PITCH = 4096;           /* src 행 pitch (byte) */
uint32_t LEN   = 512;            /* 한 행에서 읽을 byte */
PIM_RD_INSTR_MC(&idx, &next,
    src_pa | (PITCH << SHIFT),
    dst_pa | (PITCH << SHIFT),
    LEN, OPCODE_STRIDE | OPCODE_BATCH_M, 0, code);
```

4. guard 통과 → `pim_copy` → `dst` 를 host 로 읽어 확인합니다.

**판정**:

| dst 내용 | 뜻 |
| --- | --- |
| `0,0,…,1,1,…,2,2,…` — 여러 행이 모였다 | 그 `SHIFT` 가 stride 필드를 맞춘 것이다 |
| `0,0,…,0` — 첫 행만 연속으로 | stride 필드가 0 으로 읽혀 연속 복사로 퇴화한 것이다 |

**위험**: 없음. 최악의 경우 잘못된 byte 가 복사됩니다. PIM datapath 를 쓰지 않으므로 멈출 곳이 없습니다.

**결정하는 것**: `aimx.desc` 가 pitch 를 인코딩하는 위치. 설계 문서 §9.4 의 `pitch` 항목이 확정됩니다.

---

### E3. reduction 을 나눌 수 있는가 (§8 I4)

가장 값어치 큰 실험입니다. 성공하면 **"invocation 하나가 reduction 전체를 덮어야 한다"** 는
제약이 사라지고, execution tile 을 k 방향으로 자유롭게 자를 수 있게 됩니다.

두 단계로 나눕니다. **E3-a 만 성공해도 설계 질문은 답이 납니다.**

#### E3-a. vACC 를 DRAM 에서 되살릴 수 있는가 — 낮은 위험

ISA 에 memory → vACC 경로가 이미 있습니다. `_gemm_silent` (`matmul.c:325-327`) 의 bias descriptor 입니다.

```c
Bias_off = (r * PIM_WIDTH * NUM_CHANNELS / batch_size);     /* ← p 항이 없다 */
PIM_RD_INSTR_MC(&desc_idx, &next_desc,
    Bias_pa,
    mm_vACC_pa | ((PIM_WIDTH * NUM_CHANNELS / batch_size) << 40),
    PIM_WIDTH * NUM_CHANNELS / batch_size,
    OPCODE_STRIDE | opcode_batch, fpga_id, pim_code_section_va);
```

`Bias_off` 에 `p` 항이 없으므로 지금은 **한 행을 모든 출력 행에 broadcast** 하는 형태입니다.
질문은 "행마다 다른 값을 넣을 수 있는가" 입니다.

**1 단계 — 코드 수정 없이 현재 형태를 확인한다.**

```bash
~/bin/safe_run.sh ./gemm 1 256 4096 0
```

이미 빌드되어 있는 `ubench/gemm` 입니다. 성공하면 DRAM → vACC 경로가 살아 있는 것입니다.
여기서 실패하면 그 다음은 볼 필요가 없습니다.

**2 단계 — B 를 0 으로 두고 bias 만 통과시킨다.**

`gemm.cpp` 에서 `PL_srcB_buf` 를 전부 0 으로 채우고, bias 를 `f2b(i % 256)` 로 채웁니다.
`C[0][i] == bias[i]` 가 전부 맞으면, vACC 초기값이 그대로 C 로 나오는 것이 확정됩니다.
즉 **vACC 는 임의의 값으로 preload 할 수 있습니다.**

**3 단계 — 행마다 다른 값을 넣는다.** probe 에서 bias descriptor 를 C drain 의 거울상으로 바꿉니다.

```c
/* C drain (matmul.c:170) — vACC → DRAM, 양쪽에 pitch */
PIM_RD_INSTR_MC(..., mm_vACC_pa | ((NUM_CHANNELS*PIM_WIDTH/batch) << 40),
                     C_pa       | ((n * TYPE_SIZE)               << 40),
                     C_len/batch, OPCODE_C_MASK|OPCODE_MATMUL|OPCODE_STRIDE|opcode_batch, ...);

/* 이것을 정확히 뒤집는다 — DRAM → vACC, 양쪽에 pitch */
D_off = (r * PIM_WIDTH * NUM_CHANNELS / batch) + (p * n * TYPE_SIZE);   /* ← p 항 추가 */
PIM_RD_INSTR_MC(..., (D_pa + D_off) | ((n * TYPE_SIZE)               << 40),
                     mm_vACC_pa     | ((NUM_CHANNELS*PIM_WIDTH/batch) << 40),
                     C_len/batch, OPCODE_STRIDE | opcode_batch, ...);
```

m=8, batch=8, k=256, n=4096 으로 두고 B=0, D 를 `D[p][i] = f2b((p*16 + i) % 256)` 로 채웁니다.

**판정**:

| C 결과 | 뜻 | 결론 |
| --- | --- | --- |
| `C == D` 전부 일치 | 행마다 다른 preload 가 된다 | **reduction 분할 가능** — I4 채택 |
| `C[p][:] == D[0][:]` 모든 p 에 대해 | broadcast 만 된다 | 부분합을 host 로 빼서 더해야 한다 |
| 그 외 | pitch 해석이 다르다 | E2 결과와 대조 |

**위험 (낮음)**: descriptor 를 **추가**할 뿐 삼중조를 깨지 않습니다.
순서는 `bias, A, B, A, B, …, C` 로 `_gemm_silent` 와 동일하고, 주소·pitch 만 다릅니다.
최악의 경우 vACC 에 엉뚱한 값이 들어가서 C 가 틀립니다 — §1.3 의 C 유형입니다.

#### E3-b. `OPCODE_HOLD` 가 vACC 를 유지하는가 — 중간 위험

`OPCODE_HOLD` (bit 14 → `pim_math.h:51`) 는 `_memcpy_tp` (`memcpy.c:219-220`) 에서
**입력 descriptor 16 개를 staging 에 누적**하는 데 쓰입니다.

```c
PIM_RD_INSTR_MC(..., src + row0, 0xd0000000, 512, 0,            ...);   /* 첫 행 */
for (k = 1; k < 16; k++)
    PIM_RD_INSTR_MC(..., src + rowk, 0xd0000000, 512, OPCODE_HOLD, ...); /* 15 행 누적 */
PIM_RD_INSTR_MC(..., 0x0, dst, 16*512, OPCODE_TP, ...);                  /* 전치해서 배출 */
```

즉 `HOLD` = "배출하지 말고 유지하라" 입니다. 이것이 C drain 에도 통하는지 봅니다.

**절차**: canary 형상(m=1, k=256, n=4096, batch=1)에서 C descriptor 에 `OPCODE_HOLD` 를 더하고,
**같은 compute 구간을 두 번** 넣습니다.

```c
opcode_C = OPCODE_C_MASK | OPCODE_MATMUL | OPCODE_STRIDE | opcode_batch | OPCODE_HOLD;
```

**판정**: `C == 2·(A·B)` 이면 vACC 가 유지된 것입니다. `C == A·B` 면 무시된 것입니다.

**위험 (중간)**: `HOLD` 가 C descriptor 에서 디코딩되지 않으면 무시되어 안전합니다.
그러나 "배출하지 말고 유지" 를 datapath 가 문자 그대로 받아들여 배출 자체를 하지 않으면
PDMA 가 완료 신호를 못 받을 수 있습니다. **E3-a 가 성공했다면 이 실험은 최적화일 뿐이니 건너뛰어도 됩니다.**
하실 거라면 §2.1 회복 사다리를 한 번 검증한 뒤에 하십시오.

---

### E4. A 재배치를 없앨 수 있는가 (§8 I2, I3)

가장 큰 메모리 항목입니다. `_matmul_silent:101` 이 A 를 16 배로 복제합니다.

```c
srcA_buf_va = (short*)pim_malloc(sizeof(short) * p_size * q_size * 16, fpga_id);
```

`m·k·32` byte 입니다. gemm.py 형상에서는 이것 하나가 execution tile 예산의 절반을 넘습니다.

#### E4-a. hoisting (I1) — hardware 실험이 필요 없습니다

`_matmul_silent` 안에서 재배치 loop 은 이미 compute loop 바깥에 있습니다(`matmul.c:139-150` vs `152-174`).
낭비는 **execution tile 사이**에서 생깁니다 — 같은 A 행 묶음을 n tile 마다 다시 재배치하는 경우입니다.
이건 compiler 가 descriptor program 을 어떻게 짜느냐의 문제이므로 FPGA 로 갈 일이 없습니다.
설계 문서 §7 의 재사용 규칙으로 처리합니다.

#### E4-b. 재배치의 두 descriptor 를 하나로 (I3) — 중간 위험

현재 재배치는 두 개입니다.

```c
/* ① DRAM → staging, stride 로 여러 행을 모은다 */
PIM_RD_INSTR_MC(..., A_pa | ((padded_k*TYPE_SIZE) << 40), 0x0, 512,
                OPCODE_PIM_MODE | opcode_batch | OPCODE_STRIDE, ...);
/* ② staging → DRAM, 전치하고 16 배로 복제한다 */
PIM_RD_INSTR_MC(..., 0x0, srcA_buf_pa + ..., 512*batch*NUM_CHANNELS,
                OPCODE_BC | OPCODE_TP | opcode_batch, ...);
```

하나로 합쳐 봅니다.

```c
PIM_RD_INSTR_MC(..., A_pa | ((padded_k*TYPE_SIZE) << 40), srcA_buf_pa + ...,
                512*batch*NUM_CHANNELS,
                OPCODE_PIM_MODE | OPCODE_STRIDE | OPCODE_BC | OPCODE_TP | opcode_batch, ...);
```

**vehicle**: `pim_exec` 이지만 **chain 에 재배치 descriptor만 넣습니다.** compute(A/B/C)는 넣지 않습니다.
이건 `_memcpy_tp_bc` (`memcpy.c:158-176`) 와 정확히 같은 구조이므로 전례가 있습니다.

**절차**:

1. 원본 두 descriptor 판(reference) 을 돌려 `srcA_buf` 를 host 로 읽어 저장합니다.
2. 합친 한 descriptor 판을 돌려 `srcA_buf` 를 다시 읽습니다.
3. **두 buffer 를 byte 단위로 비교합니다.** 같으면 합쳐도 됩니다.

matmul 안에서 쓰기 전에 buffer 비교로 끝나므로, MAC pipeline 을 한 번도 건드리지 않습니다.

**위험 (중간)**: ② 가 staging 이 채워져 있기를 기대하는 descriptor 라면,
합친 판에서 staging 이 비어 있어 대기할 수 있습니다. 최소 형상(m=1, k=256 → group 1 개)에서 먼저 하십시오.

#### E4-c. 재배치를 계산 descriptor에 융합 (I2) — 가장 위험, 마지막

compute 의 A descriptor 가 재배치본 대신 원본 A 를 직접 읽게 합니다.

```c
/* 현재 (matmul.c:160) */
PIM_RD_INSTR_MC(..., srcA_buf_pa + A_off, 0, A_len,
                OPCODE_A_MASK | OPCODE_MATMUL | OPCODE_PIM_MODE | opcode_batch, ...);

/* 시도 */
PIM_RD_INSTR_MC(..., srcA_pa + raw_A_off | ((padded_k*TYPE_SIZE) << 40), 0, A_len,
                OPCODE_A_MASK | OPCODE_MATMUL | OPCODE_PIM_MODE
                              | OPCODE_BC | OPCODE_TP | OPCODE_STRIDE | opcode_batch, ...);
```

그리고 재배치 phase 를 **통째로** 뺍니다(삼중조를 깨는 것이 아니라 별개의 phase 를 없애는 것이므로 허용됩니다).

**절차**: canary 형상, one-hot A(§2.6), B 는 pass 1 표식. `C[0][:] == B[j0][:]` 이면 성공입니다.

**위험 (높음)**: `A_MASK|MATMUL` 과 `BC|TP|STRIDE` 를 동시에 요구하는 조합은 참조 구현에 전례가 없습니다.
datapath 가 이 조합에서 피연산자를 더 기다릴 수 있습니다 — §1.3 의 B 유형입니다.
**반드시 §2.1 회복 사다리를 검증한 뒤, 마지막에, 최소 형상으로만 하십시오.**

E4-b 가 성공하면 재배치 비용이 이미 절반이 되므로, E4-c 를 포기해도 설계는 성립합니다.

---

### E11. channel 마다 다른 record 를 넣을 수 있는가

**설계 근거와 판정 기준은 `AiMXTarget_Design.md` §15 E11 에 있습니다.** 여기에는 돌리는
절차만 적습니다.

**vehicle**: `pim_copy`. `MEMCPY_SG` 경로는 `mm_set_pim_working_signal` 과
`mm_store_pmm_signal` 을 건드리지 않고 head/tail 만 씁니다 (`dma_fops_pl.c`).
`OPCODE_PIM_MODE` 도 쓰지 않으므로 MAC pipeline 을 한 번도 지나지 않습니다.
§1.3 의 **C 유형** — 최악의 경우 잘못된 byte 가 복사됩니다.

**자리**: `~/kangmlee/test/e11_perch/`

```bash
cd ~/kangmlee/test && make -C e11_perch
```

**절차** — `run.sh` 가 dry run → mode 0 → mode 1 을 순서대로 돕니다.

```bash
cd ~/kangmlee/test/e11_perch && FPGA=1 ./run.sh
```

한 단계씩 직접 하시려면:

```bash
cd ~/kangmlee/test/e11_perch

# 0. descriptor 를 눈으로 먼저 본다 (실행하지 않는다)
AIMX_DUMP=1 AIMX_DRYRUN=1 ./probe 1 1

# 1. mode 0 — record 16 칸을 모두 같게 쓴다. harness 검증이지 실험이 아닙니다.
../safe_run.sh ./probe 0 1

# 2. mode 1 — E11 본체
../safe_run.sh ./probe 1 1
```

두 번째 인자가 `fpga_id` 이고 기본값은 1 입니다.

**1 단계에서 멈춰야 하는 경우**

| 출력 | 뜻 | 다음 |
| --- | --- | --- |
| `dst 가 전부 0` | `opcode 0` 이 이 경로에서 동작하지 않는다 | 아래 대안을 쓰십시오. **E11 결론을 내지 마십시오** |
| `label(d) == d` 가 아니다 | channel 사상 모델이 틀렸다 | 표를 보내 주십시오. mode 1 은 돌리지 마십시오 |
| `label(d) == d` | harness 정상 | 3 단계로 |

**대안 (1 단계가 `dst 가 전부 0` 일 때).** `_memcpy_sg` (`memcpy.c:52`) 가 실제로 쓰는
형태로 바꿉니다 — `probe.c` 의 `PIM_RD_INSTR_PER_CH` 호출에서 `opcode` 를 `0` 대신
`OPCODE_STRIDE | OPCODE_BATCH_M` 로 두고 `src`·`dst` 에 `| ((uint64_t)0x800 << 48)` 을
OR 하고 `length` 를 `0x800` 으로 줄이십시오. 그러면 표식 분석의 예측식이 바뀌므로
`probe.c` 의 판정부도 함께 고쳐야 합니다.

**3 단계 판정** — `probe` 가 직접 찍습니다.

| 출력 | 결론 |
| --- | --- |
| `label(d) == 256*(d&15) + d` | **(a)** channel 별 독립 주소. `pim_core_shape = [16]` 으로 간다 |
| `label(d) == d` | **(c)** record 0 만 유효. `[1]` 이 최종 모델 |
| 그 밖 | **(b)** channel 별 표가 실제 주소식을 알려 준다 |

**돌린 뒤 반드시 보내 주실 것**: `probe 0` 과 `probe 1` 의 전체 출력, 그리고 각각 뒤의
`dmesg` 판정 줄.

### E5. device 당 PIM DRAM 예산

`usable_bank_capacity_bytes` 에 넣을 값입니다. **hardware launch 가 없습니다.**

**1. BAR 크기 확인**

```bash
cd ~/pim_aimx_sw/pim_sw && sudo ./build.sh 0
dmesg | grep -E "PIM_MEM_0\] BAR #4|dma_probe - mem_len"
```

`pim_mem_probe.c:330` 이 `BAR #4 - DRAM : 0x… - 0x… (LEN: 0x…)` 를 찍습니다. 이게 상한입니다.

**2. 실제 할당 가능량 측정** — probe 에 이것만 넣고 돌립니다.

```c
init_pim_drv();
size_t lo = 0, hi = 8ULL<<30;                  /* BAR 크기보다 크게 잡고 시작 */
while (lo + (1<<20) < hi) {
    size_t mid = lo + (hi - lo) / 2;
    void *p = pim_malloc(mid, 0);
    if (p) { pim_free(p, 0); lo = mid; } else { hi = mid; }
}
printf("largest single pim_malloc: %zu byte (%.1f MB)\n", lo, lo/1048576.0);
```

`pim_exec` 을 부르지 않으므로 descriptor 가 실행되지 않습니다. **완전히 안전합니다.**

**넣는 곳** (`compiler.py`, 설계 문서 §4.7):

```python
"capacity": lambda o: (o.unit_memory_bytes, 0),
```

`unit_memory_bytes` 를 측정값의 절반 정도로 잡으십시오.
descriptor chain 과 재배치 buffer 도 같은 pool 을 쓰므로 여유가 필요합니다(§6.2 비용 모델).

---

### E6. saturation point sweep

검증된 `_matmul_silent` 만 돌립니다. **새 ISA 위험이 없습니다.** 형상 legality 만 지키면 됩니다.

**전제 조건** — `aimx_guard_shape` (§2.2) 를 만족해야 합니다. `ubench/matmul` 은
`n` 을 `CHUNK = 4096` 으로 올림하므로 `n` 은 자동으로 안전하고, `k % 256 != 0` 은
`matmul.cpp:131` 이 이미 막습니다. 남는 건 `m` 뿐인데 batch 분해가 알아서 처리합니다.

```bash
#!/bin/bash
# sweep.sh — 설계 문서 §12.1 격자
for m in 1 2 4 8 16 32 64 128 256; do
  for k in 256 512 1024 2048 4096; do
    for n in 4096 8192 16384; do
      echo "=== m=$m k=$k n=$n ==="
      ~/bin/safe_run.sh ./matmul $m $k $n 0 || { echo "STOP: hardware dirty"; exit 1; }
    done
  done
done
```

`safe_run.sh` 가 실패하면 즉시 멈추므로, 한 번 더러워진 상태로 sweep 을 계속 돌지 않습니다.

**수집할 값**: `PIM average matmul latency … ns` 줄과 dmesg 의 `PROF] DMA TX: … ns` 줄.
전자는 host 왕복 포함, 후자는 PDMA 실행 시간만입니다. **후자로 saturation point 를 잡으십시오.**

**보는 것** (설계 문서 §12.2): `m·k·n / DMA_TX` 가 평평해지는 지점이 execution tile 의 하한입니다.
그 아래에서는 descriptor 생성·전송 비용이 지배합니다.

**주의**: `TEST_ITER 1`, `ITER 1` (`matmul.cpp:26,28`) 이므로 한 형상당 한 번만 잽니다.
분산을 보시려면 `ITER` 를 3~5 로 올리십시오 — descriptor chain 은 그대로이므로 안전합니다.

---

### E7. 커널 3 개의 data type

**hardware 를 쓰지 않습니다.** Triton 커널만 고칩니다.

| 파일 | 고칠 곳 |
| --- | --- |
| `workspace/GEMM/gemm.py` | accumulator `tl.int32` → `tl.float32`, 입력 tensor `torch.bfloat16` |
| `workspace/GEMM/split_k.py` | 같음. `c = torch.zeros((SPLIT_K, M, N))` 도 `dtype=torch.float32` |
| `workspace/GEMM/grouped_gemm.py` | 같음. `pim_target="upmem"` → `"aimx"` |

`tl.dot` 의 누산은 fp32 로 두고 저장 시 bf16 으로 내리십시오 — hardware 의 누산기도
bf16 곱 → 넓은 누산 → bf16 저장 구조(§11)라 이쪽이 맞습니다.

**확인**: `python gemm.py` 가 CPU/`torch.matmul` 참조와 허용 오차 안에서 일치하면 됩니다.
아직 aimx backend 가 없으므로 `pim_target="upmem"` 으로 먼저 통과시키십시오.

---

### E8. `compiler.py` profile 값 — 가장 먼저 할 관문

**hardware 를 쓰지 않습니다.** compile 만 합니다. 설계 문서 §4.7 그대로입니다.

`triton_shared/pim_backend/backend/compiler.py` 의 `_TARGET_PROFILE`:

```python
"aimx": {
    "pim_core_shape": "1",                            # "8,16" → "1"
    "parallel_execution_kind": "simd",
    "uniform_instruction_sequence": "true",
    "lanes": 4096,                                    # 새 항목
    "capacity": lambda o: (o.unit_memory_bytes, 0),   # (-1, 0) → E5 측정값, 0
},
```

`_profile_pass` 안:

```python
f"number-of-parallel-lanes-per-pim-core={t.get('lanes', options.local_lanes)} "
```

**확인 방법**:

```bash
cd ~/kangmlee/TritonPIM/workspace/GEMM
# gemm.py 의 pim_target 을 "aimx" 로 바꾼 뒤
python gemm.py 2>&1 | tail -30
```

**판정**:

| 오류 메시지 | 뜻 |
| --- | --- |
| `pim-map-cores: PIM Group with 128 PIM Cores is not mapped yet` | 아직 `"8,16"` 이다. 값이 반영되지 않았다 |
| `pim-form-execution-waves: a PIM Group holds N logical tiles per wave` | `pim_core_shape` 는 고쳤는데 footprint 계산이 안 맞는다 |
| **`--convert-pim-to-aimx` 관련 오류** (unknown pass / no such option) | **통과.** 공통 pipeline 이 끝까지 돌았고 target pass 만 없는 것이다 |

세 번째가 나오면 §4.8 의 표가 검증된 것이고, 그 다음은 dialect 구현입니다.

세 커널 모두에 대해 확인하십시오 — `split_k.py` 는 `explicit_k_partition` 축이,
`grouped_gemm.py` 는 `GROUP_M` swizzle 이 각각 다른 경로를 탑니다.

---

## 5. 요약

| 실험 | vehicle | 위험 | 결정하는 것 |
| --- | --- | --- | --- |
| **E7** 커널 bf16 | Triton | 없음 | 입력 형식 |
| **E8** profile 값 | compile | 없음 | 공통 pipeline 통과 — **가장 먼저** |
| **E5** DRAM 예산 | `pim_malloc` | 없음 | `usable_bank_capacity_bytes` |
| **E1** channel 수 | canary | 없음 | lane 4096 확정 — canary 가 답한다 |
| **E6** sweep | `_matmul_silent` | 없음 | execution tile 하한 |
| ~~**E2** stride shift~~ | — | — | **닫힘. `<< 40`** |
| **E11** channel 별 record | `pim_copy` | 낮음 | **`pim_core_shape` = `[1]` 인가 `[16]` 인가** |
| **E3-a** vACC preload | `pim_exec` | 낮음 | **reduction 분할 가능 여부** |
| **E4-b** 재배치 합치기 | `pim_exec` (재배치만) | 중간 | 재배치 비용 절반 |
| **E3-b** `OPCODE_HOLD` | `pim_exec` | 중간 | E3-a 성공 시 생략 가능 |
| **E4-c** 재배치 융합 | `pim_exec` | **높음** | `m·k·32` byte 제거 — **마지막에** |

**지켜야 할 세 가지**

1. 모든 실행을 `safe_run.sh` 로 감싸고, `polling timeout` 이 보이면 **다음 실행을 하지 않는다.**
2. `pim_exec` / `pim_copy` 직전에 `aimx_guard_chain` 과 `aimx_guard_shape` 를 통과시킨다.
3. A / B / C 삼중조에서 descriptor 를 **빼지 않는다.** flag 를 더하고 주소를 바꾸는 것만 한다.

**가장 값어치 있는 세 개**는 E11 (channel 별 record), E3-a (reduction 분할),
E4-b (재배치 절반) 입니다. 셋 다 위험도가 낮거나 중간이고, 성공하면 각각 설계 문서의
§4.6 (`pim_core_shape`), §8 I4, §8 I3 이 확정됩니다.

**E11 을 먼저 하십시오.** 위험도가 가장 낮고(`pim_copy`, `OPCODE_PIM_MODE` 없음),
결과가 `pim_core_shape` 를 정하므로 그 뒤의 IR 구조 결정이 전부 여기에 매달려 있습니다.
