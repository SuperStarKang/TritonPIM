#ifndef UPMEM_RUNTIME_H
#define UPMEM_RUNTIME_H

// config 무관한 공유 UPMEM 런타임. 모든 host.so가 이 단일 libupmem_runtime.so를 링크하므로
// 아래 set 캐시(globals)가 프로세스에서 1개로 공유된다 → DPU 예약(dpu_alloc)을 세션당 1회로.
// (host.so마다 자기 set을 들면 sweep 시 누적 고갈 → 그래서 별도 공유 계층이 필요하다.)
//
// 이름과 노출 타입이 전부 UPMEM 것이다(struct dpu_set_t). PIM 공통 런타임이 아니라
// UPMEM target의 런타임이며, 다른 target은 자기 런타임을 따로 둔다.

#include <stdint.h>

// dpu.h는 C API. C++(host.cpp)에서 include할 때 이름 mangling을 막으려고 C 링키지로 감싼다.
#ifdef __cplusplus
extern "C" {
#endif

#include <dpu.h>

// num_units개 DPU를 예약한 set을 돌려준다.
//   - 같은 num_units 호출이면 이미 예약한 set을 재사용(persistent) → dpu_alloc 반복 제거.
//   - 그 set에 적재된 elf가 binary_path와 다르면 dpu_load로 교체(config 전환 시 reload).
//   - 프로세스 종료 시 atexit으로 일괄 free (per-call free 대신 1회).
struct dpu_set_t upmem_acquire(uint32_t num_units, const char *binary_path);

struct dpu_set_t upmem_runtime_reserve(uint32_t num_units);
int upmem_runtime_load(const char *binary_path);

#include "upmem_resident.h"

#ifdef __cplusplus
}
#endif

#ifdef UPMEM_RESIDENT_HOST
#define dpu_prepare_xfer upmem_resident_prepare_xfer
#define dpu_push_xfer upmem_resident_push_xfer
#define dpu_launch upmem_resident_launch
#endif

#endif  // UPMEM_RUNTIME_H
