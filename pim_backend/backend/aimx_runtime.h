// AiMX runtime shim — 생성된 stub(stub.cpp)과 descriptor program(aimx_descriptors.h)이
// 쓰는 것만 모아 둔다. upmem_runtime.h 와 같은 자리이고 같은 역할이다.
//
// 진단, host 조립 buffer, bfloat16 변환, 그리고 AiMX 전용 resident resource ABI.
// 연산 선택은 compiler가 하고, 공유 runtime 구현은 자원 수명만 관리한다.
// 하위 PIM runtime의 기존 API를 사용한다:
//   pim.h       — init_mpim_drv / totalDevNum / pim_malloc / VA2PA / datacopy_*
//   pim_math.h  — pim_isa_32_t / PIM_RD_INSTR_MC / pim_exec
//
// 빌드에는 두 include 경로가 필요하다 (driver.py 의 _build_aimx_so 가 넣는다).
//   -I <pim_sw>/src/pim_math_src  -I <pim_sw>/include
#ifndef AIMX_RUNTIME_H
#define AIMX_RUNTIME_H

// pim_math.h 가 쓰는 C 헤더를 먼저 들여온다. 아래 extern "C" 블록 안에서 처음 열리지
// 않도록 하기 위해서다.
#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <mutex>

// **pim_math.h 에는 extern "C" 가드가 없다.** pim.h 는 있다. 그래서 이 헤더를 C++ 에서
// 그냥 들여오면 pim_exec 이 부르는 _pim_exec 만 C++ 이름으로 망가지고, libpim.so 가
// 내보내는 C 이름 _pim_exec 과 만나지 못해 .so 를 여는 순간 미해결 심볼이 된다.
extern "C" {
#include "pim_math.h"
}

// pim.h 는 연산마다 함수형 매크로를 둔다. 생성된 stub 의 진입 함수 이름이 커널 이름
// 그대로라, 커널을 matmul 이나 gemm 으로 부르면 그 선언이 매크로로 펼쳐져 컴파일이
// 깨진다. 여기서 쓰지 않는 것들이므로 지운다 — pim_exec / pim_malloc / VA2PA /
// PIM_RD_INSTR_MC 처럼 실제로 부르는 매크로는 그대로 둔다.
#undef matmul
#undef matmul_tiled
#undef gemm
#undef matmul_fusion
#undef elewise_add
#undef elewise_sub
#undef elewise_mul
#undef bias_add
#undef bias_sub
#undef bias_mul
#undef dot_prod
#undef unary_op
#undef memcpy_sg
#undef memcpy_input_layout
#undef memcpy_output_layout

// 몇 번 카드부터 쓸 것인가. 생성된 stub 이 `AIMX_BASE_FPGA + 좌표` 로 fpga_id 를 만든다.
// 지금 동작하는 카드가 1 번이라 기본값이 1 이다. 빌드에서 -DAIMX_BASE_FPGA=0 으로 바꾼다.
#ifndef AIMX_BASE_FPGA
#define AIMX_BASE_FPGA 1
#endif

// 실패를 조용히 넘기지 않는다. AiMX 는 어긋난 크기나 겹친 주소를 오류로 알려 주지 않고
// 그냥 다른 값을 낸다.
//
// **abort() 가 아니라 exit() 다.** PIM hugepage 를 driver 에 돌려주는 길은
// __release_mpim_drv 하나뿐이고 그것은 atexit 로 걸려 있는데(mdrv_interface.c:26),
// abort() 는 atexit 핸들러를 건너뛴다. 커널 쪽 fops_release 는 비어 있어
// (pim_mem_probe.c:83) 프로세스가 죽어도 회수하지 않는다 — 즉 abort 하면 그 launch 가
// 잡은 hugepage 가 영구히 샌다. 참조 구현의 OOM 경로도 같은 이유로 exit 를 쓴다.
static inline void aimx_require(int ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "AIMX] %s\n", what);
    exit(1);
  }
}

// host <-> PIM DRAM 전송을 AIMX_DMA_CHUNK_BYTES 단위로 나눠 datacopy_* 를 부른다.
// libpim 은 256 MiB(QDMA_MAX_COPY)보다 작은 요청을 pwrite/pread 한 번으로 보내고, qdma_pf 는
// 그때마다 page 주소 목록(sgl)을 물리 연속 메모리로 kmalloc 한다(page 당 약 62 B). 66 MiB 전송이면
// 1 MiB(order 8)가 필요해 host RAM 이 조각나면 ENOMEM 이 난다(Llama-2-7b BF16 13.5 GB 적재 중 실측).
// 2 MiB 면 목록이 32 KiB(order 3) 이하라 커널이 항상 할당한다. 크기 인자 int 의 2 GiB 한계도 피한다.
// TODO: host RAM 이 늘어 조각화 여유가 생기면(또는 모델 weight 를 host 에 두지 않게 되면)
//       이 분할을 되돌려 datacopy_* 를 한 번에 부른다. pwrite 호출 수가 줄어 전송 고정 비용이 준다.
#define AIMX_DMA_CHUNK_BYTES ((size_t)2 << 20)

static inline int aimx_copy_to_pim(void *pim_dst, const void *host_src, size_t bytes, int fpga) {
  for (size_t done = 0; done < bytes; done += AIMX_DMA_CHUNK_BYTES) {
    size_t size = bytes - done < AIMX_DMA_CHUNK_BYTES ? bytes - done : AIMX_DMA_CHUNK_BYTES;
    if (datacopy_cpu2pim((char *)pim_dst + done, (const char *)host_src + done, (int)size, fpga) < 0)
      return -1;
  }
  return 0;
}

static inline int aimx_copy_from_pim(void *host_dst, const void *pim_src, size_t bytes, int fpga) {
  for (size_t done = 0; done < bytes; done += AIMX_DMA_CHUNK_BYTES) {
    size_t size = bytes - done < AIMX_DMA_CHUNK_BYTES ? bytes - done : AIMX_DMA_CHUNK_BYTES;
    if (datacopy_pim2cpu((char *)host_dst + done, (const char *)pim_src + done, (int)size, fpga) < 0)
      return -1;
  }
  return 0;
}

// Shared across generated kernel DSOs by libaimx_resident. A lease owns one
// physical FPGA through staging, synchronous invocation and result collection.
// Addresses and chain records stay live until plan replacement or explicit reset.
struct aimx_resident_buffer {
  uint64_t va, pa, next;
  void *host;
  int64_t bytes, alignment;
  uint32_t idx;
  int kind;
};
struct aimx_resident_context {
  aimx_resident_buffer *buffers;
  size_t count;
  bool ready;
  void *owner;
};
extern "C" {
int aimx_runtime_init(int base_fpga, int num_fpga);
aimx_resident_context *aimx_resident_acquire(const char *plan, int fpga,
                                           size_t buffers);
aimx_resident_buffer *aimx_resident_alloc(aimx_resident_context *context,
                                         size_t slot, int64_t bytes,
                                         int64_t alignment, int kind);
void aimx_resident_release(aimx_resident_context *context);
// Call only outside a lease; this frees our allocations, not the vendor driver.
void aimx_resident_reset();
}

// Model-session stage call (--aimx-to-model-stub). One entry call runs one stage
// for every instance; job = batch slot * devices + device, one FPGA per job.
struct aimx_model_call {
  aimx_resident_context **contexts; // [batch_size * devices]
  const int *fpgas;                 // [batch_size * devices]
  void *const *pointers;            // [instances * arguments], host tensors per instance
  const int *weights;               // [arguments], 1 when the argument is a resident weight
  uint32_t instances, batch_size, arguments;
  // 0: run (stage activations, invoke, unstage). 1: stage weights to the device.
  // 2: pack, stage weights into pack_hosts instead of the device. 1 and 2 build
  // chains but never invoke or unstage.
  int upload;
  uint32_t batch_base;
  // pack only: per job, the host image of the job's weight range and the device
  // VA that image starts at. A weight buffer at VA v lands at host + (v - va).
  void *const *pack_hosts;          // [batch_size * devices]
  const uint64_t *pack_vas;         // [batch_size * devices]
};

// Stage copy of a model stage: the device, or the job's pack image when upload is 2.
static inline int aimx_model_copy_to_pim(const aimx_model_call *model, uint32_t job,
                                         void *pim_dst, const void *host_src, size_t bytes,
                                         int fpga) {
  if (model->upload != 2)
    return aimx_copy_to_pim(pim_dst, host_src, bytes, fpga);
  memcpy((char *)model->pack_hosts[job] + ((uint64_t)pim_dst - model->pack_vas[job]),
         host_src, bytes);
  return 0;
}

// One memory-table region on one FPGA: a single physically linear pim_malloc and a
// host shadow of the same size. A buffer's host assembly copy sits in the shadow at
// the buffer's table offset, so host copies follow the table: buffers that share
// device memory (the scratch, a shared slot) share their host copy too. The shadow
// is an anonymous mapping, so only pages a stage writes use host memory.
struct aimx_model_region {
  int fpga;
  uint64_t va, pa;
  int64_t bytes;
  void *host;
};
extern "C" {
// Opens every physical FPGA once and returns how many exist.
int aimx_model_open();
// alignment is the region's start alignment in bytes.
aimx_model_region *aimx_model_region_create(int fpga, int64_t bytes, int64_t alignment);
// Destroy the contexts placed in a region first.
void aimx_model_region_destroy(aimx_model_region *region);
// Return the pages of a host shadow range; later writes start from zero pages.
void aimx_model_host_release(void *host, int64_t bytes);
// Copy hosts[i] to the start of regions[i] (bytes[i]); one thread per FPGA.
void aimx_model_push(size_t count, aimx_model_region *const *regions, void *const *hosts,
                     const int64_t *bytes);
// Session-owned contexts. Never replaced implicitly; slots are allocated by the
// stage stub on first use and stay until destroy. Layers that share a residency
// slot share the context, so its buffers and descriptor chain are built once.
// Buffer i (the PMM included) is placed at va[i], its memory-table address in a
// region of this FPGA, may use at most capacity[i] bytes, and assembles in host[i],
// the same offset in the region's host shadow.
aimx_resident_context *aimx_model_create(int fpga, size_t slots, const uint64_t *va,
                                         const int64_t *capacity, void *const *host);
void aimx_model_destroy(aimx_resident_context *context);
int64_t aimx_model_live_bytes(aimx_resident_context *context);
}

// Address relocation for an already-built descriptor group. The compiler emits
// this only when record count/order are unchanged. Match mPIM_RD_INSTR_MC's
// channel replication and high/low packing, leaving next/len/status untouched.
// src/dst include pitch bits when the target instruction uses strided access.
static inline void aimx_patch_descriptor_addresses(pim_isa_32_t *code,
                                                    uint64_t src, uint64_t dst) {
  for (unsigned channel = 0; channel < NUM_CHANNELS; ++channel) {
    code[channel].src_l = (uint32_t)src;
    code[channel].src_h = (uint32_t)(src >> 32);
    code[channel].dst_l = (uint32_t)dst;
    code[channel].dst_h = (uint32_t)(dst >> 32);
  }
}

// host 쪽 조립 buffer. descriptor chain 과 stage 결과가 여기서 만들어진 뒤
// datacopy_cpu2pim 으로 한 번에 건너간다.
//
// **launch 를 넘어 쥐고 있는다.** launch 마다 mmap/munmap 하면 buffer 가 언제나 차가워서
// stage 의 첫 접근이 전부 page fault 다 — 64 MiB 짜리 B 면 16384 페이지이고 실측으로
// 변환 loop 시간의 절반이다(차가운 5.2 GB/s vs 따뜻한 11.9 GB/s). 같은 커널을 여러 번
// 부르는 것이 정상이므로 크기별로 하나씩 남겨 둔다.
//
// 스레드마다 따로 두어 lock 이 없다 — PIM Device 하나에 host thread 하나이고 buffer 는
// 그 thread 의 loop 몸체 안에서만 산다. 한 launch 가 쓰는 buffer 는 다섯 개(A, Arel, B,
// C, chain)라 칸이 남고, 넘치면 그때만 실제로 돌려준다.
//
// memset 도 하지 않는다. MAP_ANONYMOUS 는 0 으로 채운 페이지를 주고, 재사용분은 stage 가
// 쓰는 자리와 padding 을 0 으로 채우는 자리가 합쳐 buffer 전체를 덮으므로 남는 값이 없다
// (chain 은 실제로 쓴 record 까지만 datacopy 한다).
//
// cache 는 libaimx_resident 하나에 있어 모든 커널 DSO 가 함께 쓴다. DSO 마다 따로 두면
// autotuning 처럼 후보마다 새 DSO 를 올릴 때 DSO 수만큼 buffer 가 쌓인다.
// aimx_host_cache_clear 는 모든 thread 의 cache 를 돌려준다 (커널이 돌지 않을 때만 부른다).
// aimx_host_cache_clear_thread 는 부른 thread 의 cache 만 돌려준다 (다른 thread 가 돌아도 된다).
extern "C" {
void *aimx_host_buffer(size_t bytes);
void aimx_host_free(void *p, size_t bytes);
void aimx_host_cache_clear();
void aimx_host_cache_clear_thread();
}

// round-to-nearest-even bfloat16 변환. PIM 의 누산기는 표준 bf16 이 아니지만
// (지수 칸이 8 binade 단위다) 저장 형식은 bf16 그대로다.
static inline uint16_t aimx_to_bf16(float f) {
  union { float f; uint32_t u; } c;
  c.f = f;
  // Rounding a NaN payload must not turn it into infinity or signed zero.
  if ((c.u & 0x7fffffffu) > 0x7f800000u)
    return (uint16_t)((c.u >> 16) | 0x40u);
  uint32_t round = (c.u >> 16) & 1u;
  return (uint16_t)((c.u + 0x7fffu + round) >> 16);
}

static inline float aimx_from_bf16(uint16_t b) {
  union { float f; uint32_t u; } c;
  c.u = ((uint32_t)b) << 16;
  return c.f;
}

// IEEE half 변환. host 가 float16 을 주고 PIM 은 bfloat16 을 쓰므로 float 를 거쳐야 한다.
// 둘은 지수 폭이 다르다 — 원시 16 bit 를 그대로 옮기면 값이 완전히 달라진다.
//
// **분기가 없다.** stage loop 이 원소마다 이것을 부르는데, B 하나가 수억 원소라 분기
// 하나가 곧 벡터화를 막고 그것이 host 시간의 거의 전부였다. 대신 지수·가수를 13 bit
// 올려 놓고 2^112 를 곱한다. 정규수는 bias 차(127-15)가 그 곱셈으로 맞고, subnormal 도
// 같은 곱셈이 정규화해 준다. 곱한 값이 2^16 이상이면 원래 지수가 31 이었다는 뜻이므로
// (fp16 최대 정규수는 65504 다) 그때만 지수 자리를 채워 inf/NaN 으로 되돌린다.
static inline float aimx_from_fp16(uint16_t h) {
  union { float f; uint32_t u; } c, magic;
  c.u = (uint32_t)(h & 0x7fffu) << 13;
  magic.u = 0x77800000u;                                    // 2^112
  c.f *= magic.f;
  c.u |= (c.u >= 0x47800000u ? 0x7f800000u : 0u)            // 2^16 = fp16 exp 31
         | ((uint32_t)(h & 0x8000u) << 16);
  return c.f;
}

// f16 -> bf16 는 16 bit -> 16 bit 사상이라 가짓수가 65536 뿐이다. 그래서 표 하나로
// 끝난다 — stage loop 이 이것을 원소마다 부르고 B 하나가 수억 원소라, 산술로 하면
// 컴파일러가 벡터화해 줄 때만 빨라진다. 표는 최적화 수준과 무관하게 4 배다
// (실측: 분기판 1.4 GB/s, 분기 없는 산술판 -O2 1.2 / -O3 5.4 GB/s, 표 5.1~6.0 GB/s).
//
// 파일 유효 범위의 static 이라 .so 를 여는 시점에 한 번 채워진다. 함수 안의 static 과
// 달리 호출마다 초기화 여부를 확인하지 않고, OpenMP 스레드가 뜨기 전에 이미 끝나 있다.
// 128 KiB 는 L2 에 들어간다.
struct AiMXF16ToBF16 {
  uint16_t t[1 << 16];
  AiMXF16ToBF16() {
    for (uint32_t h = 0; h < (1u << 16); h++)
      t[h] = aimx_to_bf16(aimx_from_fp16((uint16_t)h));
  }
};
static const AiMXF16ToBF16 aimx_f16_bf16;

static inline uint16_t aimx_to_fp16(float f) {
  union { float f; uint32_t u; } c;
  c.f = f;
  uint32_t sign = (c.u >> 16) & 0x8000u;
  int32_t exp = (int32_t)((c.u >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = c.u & 0x7fffffu;
  if (((c.u >> 23) & 0xffu) == 0xffu)  // inf / NaN
    return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
  if (exp >= 0x1f)                     // 넘침
    return (uint16_t)(sign | 0x7c00u);
  if (exp <= 0) {                      // subnormal 또는 0
    if (exp < -10)
      return (uint16_t)sign;
    mant |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t half = mant >> shift;
    uint32_t rest = mant & ((1u << shift) - 1u);
    if (rest > (1u << (shift - 1)) || (rest == (1u << (shift - 1)) && (half & 1u)))
      half++;
    return (uint16_t)(sign | half);
  }
  uint32_t half = ((uint32_t)exp << 10) | (mant >> 13);
  uint32_t rest = mant & 0x1fffu;      // round-to-nearest-even
  if (rest > 0x1000u || (rest == 0x1000u && (half & 1u)))
    half++;
  return (uint16_t)(sign | half);
}

#endif // AIMX_RUNTIME_H
