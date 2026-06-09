#include "pim_runtime.h"

#include <stdlib.h>
#include <string.h>

// 단일 persistent set. set이 보관하는 건 (1) DPU 예약(dpu_alloc ~280ms 회피) + (2) 적재된 elf
// (dpu_load ~25ms 회피)뿐 — 데이터(A/B/C)는 매 호출 재전송하므로 set에 남겨둘 게 없다.
// 그래서 한 번에 1개만 유지한다:
//   - 같은 num_units: set 재사용(elf가 다르면 그것만 reload — config 전환).
//   - 다른 num_units: 기존 set을 free하고 새로 alloc (예약 크기가 다르면 같은 set을 못 쓴다).
// 한 개만 쥐므로 서로 다른 카운트가 누적돼 DPU가 고갈되는 일이 없다(sweep 안전).
// 프로세스 종료 시 atexit으로 free (per-call free 대신 1회).
// 참고: weight를 MRAM에 상주시키는 weight-stationary로 가면 set이 "재사용할 데이터(weight)"를
//   들게 되어 여러 set 유지 + LRU 교체가 의미를 갖는다(DESIGN 18). 지금은 데이터를 안 들어 불필요.

static int g_have = 0;
static uint32_t g_num_units = 0;
static struct dpu_set_t g_set;
static char g_loaded[1024];  // 현재 적재된 binary 경로 (reload 필요 판단용)
static int g_atexit_done = 0;

static void pim_free_set(void) {
    if (g_have) {
        DPU_ASSERT(dpu_free(g_set));
        g_have = 0;
    }
}

struct dpu_set_t pim_acquire(uint32_t num_units, const char *binary_path) {
    if (!g_atexit_done) {
        atexit(pim_free_set);
        g_atexit_done = 1;
    }

    // 같은 num_units이면 set 재사용. elf가 바뀌었으면(config 전환) 그것만 교체.
    if (g_have && g_num_units == num_units) {
        if (strcmp(g_loaded, binary_path) != 0) {
            DPU_ASSERT(dpu_load(g_set, binary_path, NULL));
            strncpy(g_loaded, binary_path, sizeof(g_loaded) - 1);
            g_loaded[sizeof(g_loaded) - 1] = '\0';
        }
        return g_set;
    }

    // 처음이거나 num_units이 다르면 기존 set을 비우고 새로 예약+적재.
    pim_free_set();
    DPU_ASSERT(dpu_alloc(num_units, NULL, &g_set));
    DPU_ASSERT(dpu_load(g_set, binary_path, NULL));
    g_num_units = num_units;
    strncpy(g_loaded, binary_path, sizeof(g_loaded) - 1);
    g_loaded[sizeof(g_loaded) - 1] = '\0';
    g_have = 1;
    return g_set;
}
