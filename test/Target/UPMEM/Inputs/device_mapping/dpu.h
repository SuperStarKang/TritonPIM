#ifndef TEST_DPU_H
#define TEST_DPU_H
#include <assert.h>
#include <stdint.h>

struct dpu_t { uint32_t member, ci; };
struct test_rank { uint32_t count; struct dpu_t dpus[64]; };
struct dpu_set_t {
    struct test_rank *ranks;
    uint32_t rank_count;
    struct dpu_t *dpu;
};

#define DPU_ASSERT(call) assert((call) == 0)
#define DPU_RANK_FOREACH(set, rank) \
    for (uint32_t r = 0; r < (set).rank_count && \
         ((rank = (struct dpu_set_t){.ranks = (set).ranks + r, .rank_count = 1}), 1); ++r)
#define DPU_FOREACH(set, unit, index) \
    for (index = 0; index < (set).ranks->count && \
         ((unit = (struct dpu_set_t){.dpu = &(set).ranks->dpus[index]}), 1); ++index)

int dpu_get_nr_dpus(struct dpu_set_t set, uint32_t *count);
int dpu_alloc(uint32_t count, const char *profile, struct dpu_set_t *set);
int dpu_load(struct dpu_set_t set, const char *path, void *program);
int dpu_free(struct dpu_set_t set);
#endif
