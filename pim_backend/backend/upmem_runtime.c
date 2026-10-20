#include "upmem_runtime.h"

#include <dpu_management.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ResidentBlock도 이 집합 전체를 사용하며, 소유 중에는 다른 호출이 교체하지 못한다.

static int g_have = 0;
static uint32_t g_num_units = 0;
static struct dpu_set_t g_set;
static char g_loaded[1024];  // 현재 적재된 binary 경로 (reload 필요 판단용)
static int g_atexit_done = 0;
static int g_session_open = 0;

void upmem_bind_devices(struct dpu_set_t set, uint32_t num_axes,
                        const uint32_t *partitions,
                        const uint32_t *device_mapping,
                        uint32_t *logical_for_dpu) {
    uint32_t units = 1, allocated;
    for (uint32_t d = 0; d < num_axes; ++d)
        units *= partitions[d];
    DPU_ASSERT(dpu_get_nr_dpus(set, &allocated));
    if (allocated != units) {
        fprintf(stderr, "UPMEM assignment: expected %u DPUs, got %u\n", units, allocated);
        abort();
    }
    upmem_bind_instances(set, num_axes, partitions, device_mapping, 1, logical_for_dpu);
}

void upmem_bind_instances(struct dpu_set_t set, uint32_t num_axes,
                          const uint32_t *partitions,
                          const uint32_t *device_mapping, uint32_t instances,
                          uint32_t *logical_for_dpu) {
    uint32_t units = 1, group_size = 1, allocated;
    for (uint32_t d = 0; d < num_axes; ++d) {
        units *= partitions[d];
        group_size *= device_mapping[d];
    }
    DPU_ASSERT(dpu_get_nr_dpus(set, &allocated));
    if ((uint64_t)instances * units > allocated) {
        fprintf(stderr, "UPMEM assignment exceeds the allocated DPU set\n");
        abort();
    }

    struct dpu_set_t rank, dpu;
    uint32_t ordinal = 0, index;
    // 완전한 member부터 배정해 비활성 DPU가 다른 묶음의 정렬을 밀지 않게 한다.
    for (uint32_t pass = 0; pass < 2; ++pass) {
        uint32_t rank_base = 0;
        DPU_RANK_FOREACH(set, rank) {
            uint32_t physical[64];
            for (uint32_t slot = 0; slot < 64; ++slot)
                physical[slot] = UINT32_MAX;
            uint32_t active = 0;
            DPU_FOREACH(rank, dpu, index) {
                uint32_t member = dpu_get_member_id(dpu.dpu);
                uint32_t ci = dpu_get_slice_id(dpu.dpu);
                if (member >= 8 || ci >= 8 || physical[member * 8 + ci] != UINT32_MAX) {
                    fprintf(stderr, "UPMEM assignment: invalid rank/member/CI topology\n");
                    abort();
                }
                physical[member * 8 + ci] = rank_base + index;
                ++active;
            }
            for (uint32_t member = 0; member < 8; ++member) {
                uint32_t recipients = 0;
                for (uint32_t ci = 0; ci < 8; ++ci)
                    recipients += physical[member * 8 + ci] != UINT32_MAX;
                if ((recipients == 8) != (pass == 0))
                    continue;
                for (uint32_t ci = 0; ci < 8; ++ci) {
                    if (physical[member * 8 + ci] == UINT32_MAX)
                        continue;
                    if (ordinal >= instances * units) {
                        logical_for_dpu[physical[member * 8 + ci]] = UINT32_MAX;
                        ++ordinal;
                        continue;
                    }
                    uint32_t outer = (ordinal % units) / group_size;
                    uint32_t inner = (ordinal % units) % group_size;
                    uint32_t logical = 0, stride = 1;
                    for (uint32_t d = num_axes; d-- > 0;) {
                        uint32_t blocks = partitions[d] / device_mapping[d];
                        uint32_t coordinate = (outer % blocks) * device_mapping[d] +
                                              inner % device_mapping[d];
                        logical += coordinate * stride;
                        stride *= partitions[d];
                        outer /= blocks;
                        inner /= device_mapping[d];
                    }
                    logical_for_dpu[physical[member * 8 + ci]] =
                        (ordinal / units) * units + logical;
                    ++ordinal;
                }
            }
            rank_base += active;
        }
    }
    if (ordinal != allocated) {
        fprintf(stderr, "UPMEM assignment: physical enumeration size mismatch\n");
        abort();
    }
}

static void upmem_free_set(void) {
    if (g_have) {
        DPU_ASSERT(dpu_free(g_set));
        g_have = 0;
        g_num_units = 0;
        g_loaded[0] = '\0';
    }
}

struct dpu_set_t upmem_runtime_reserve(uint32_t num_units) {
    if (g_session_open && (!g_have || g_num_units != num_units)) {
        fprintf(stderr, "UPMEM session cannot replace its allocated device set\n");
        abort();
    }
    if (!g_atexit_done) {
        atexit(upmem_free_set);
        g_atexit_done = 1;
    }

    if (g_have && g_num_units == num_units) {
        return g_set;
    }

    // 처음이거나 num_units이 다르면 기존 set을 비우고 새로 예약+적재.
    upmem_free_set();
    // layer group의 weight는 DPU마다 여러 host block을 MRAM 연속 영역으로 보낸다(dpu_push_sg_xfer).
    DPU_ASSERT(dpu_alloc(num_units, "sgXferEnable=true,sgXferMaxBlocksPerDpu=1024", &g_set));
    DPU_ASSERT(dpu_get_nr_dpus(g_set, &g_num_units));
    g_loaded[0] = '\0';
    g_have = 1;
    return g_set;
}

void upmem_session_open(uint32_t num_units) {
    if (!num_units || g_session_open || g_have || upmem_resident_owns_set()) {
        fprintf(stderr, "UPMEM session requires exclusive ownership of an unallocated runtime\n");
        abort();
    }
    upmem_runtime_reserve(num_units);
    g_session_open = 1;
}

void upmem_session_close(void) {
    if (!g_session_open || !g_have || upmem_resident_owns_set()) {
        fprintf(stderr, "UPMEM session close requires all resident layers to be released\n");
        abort();
    }
    upmem_free_set();
    g_session_open = 0;
}

uint32_t upmem_session_num_units(void) {
    return g_num_units;
}

struct dpu_set_t upmem_session_devices(void) {
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
    if (g_session_open) {
        fprintf(stderr, "UPMEM session execution requires a bound resident layer\n");
        abort();
    }
    struct dpu_set_t set = upmem_runtime_reserve(num_units);
    upmem_runtime_load(binary_path);
    return set;
}
