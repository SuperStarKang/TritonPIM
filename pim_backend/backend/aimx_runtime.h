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
#ifndef AIMX_HOST_BUFFER_SLOTS
#define AIMX_HOST_BUFFER_SLOTS 8
#endif
static __thread void *aimx_hb_ptr[AIMX_HOST_BUFFER_SLOTS];
static __thread size_t aimx_hb_bytes[AIMX_HOST_BUFFER_SLOTS];

// 칸이 비었는지는 포인터로 판단한다. 크기로 판단하면 0 byte 요청이 빈 칸과 같아진다.
static inline void *aimx_host_buffer(size_t bytes) {
  for (int i = 0; i < AIMX_HOST_BUFFER_SLOTS; i++)
    if (aimx_hb_ptr[i] && aimx_hb_bytes[i] == bytes) {
      void *p = aimx_hb_ptr[i];
      aimx_hb_ptr[i] = NULL;
      return p;
    }
  void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  aimx_require(p != MAP_FAILED, "host buffer mmap failed");
  return p;
}

static inline void aimx_host_free(void *p, size_t bytes) {
  for (int i = 0; i < AIMX_HOST_BUFFER_SLOTS; i++)
    if (!aimx_hb_ptr[i]) {
      aimx_hb_ptr[i] = p;
      aimx_hb_bytes[i] = bytes;
      return;
    }
  munmap(p, bytes);
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
