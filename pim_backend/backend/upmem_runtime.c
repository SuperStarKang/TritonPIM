#include "upmem_runtime.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ResidentBlock도 이 집합 전체를 사용하며, 소유 중에는 다른 호출이 교체하지 못한다.

static int g_have = 0;
static uint32_t g_num_units = 0;
static struct dpu_set_t g_set;
static char g_loaded[1024];  // 현재 적재된 binary 경로 (reload 필요 판단용)
static int g_atexit_done = 0;

static void upmem_free_set(void) {
    if (g_have) {
        DPU_ASSERT(dpu_free(g_set));
        g_have = 0;
    }
}

struct dpu_set_t upmem_runtime_reserve(uint32_t num_units) {
    if (!g_atexit_done) {
        atexit(upmem_free_set);
        g_atexit_done = 1;
    }

    if (g_have && g_num_units == num_units) {
        return g_set;
    }

    // 처음이거나 num_units이 다르면 기존 set을 비우고 새로 예약+적재.
    upmem_free_set();
    DPU_ASSERT(dpu_alloc(num_units, NULL, &g_set));
    g_num_units = num_units;
    g_loaded[0] = '\0';
    g_have = 1;
    return g_set;
}

int upmem_runtime_load(const char *binary_path) {
    if (strlen(binary_path) >= sizeof(g_loaded)) {
        fprintf(stderr, "UPMEM binary path exceeds runtime capacity\n");
        abort();
    }
    if (strcmp(g_loaded, binary_path) == 0)
        return 0;
    DPU_ASSERT(dpu_load(g_set, binary_path, NULL));
    strcpy(g_loaded, binary_path);
    return 1;
}

struct dpu_set_t upmem_acquire(uint32_t num_units, const char *binary_path) {
    if (upmem_resident_owns_set())
        return upmem_resident_acquire(num_units);
    struct dpu_set_t set = upmem_runtime_reserve(num_units);
    upmem_runtime_load(binary_path);
    return set;
}
