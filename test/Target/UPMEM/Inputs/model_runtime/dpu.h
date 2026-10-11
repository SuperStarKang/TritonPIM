#ifndef MODEL_TEST_DPU_H
#define MODEL_TEST_DPU_H
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

typedef int dpu_error_t;
typedef int dpu_xfer_t;
typedef int dpu_xfer_flags_t;
typedef int dpu_launch_policy_t;
enum { DPU_XFER_TO_DPU, DPU_XFER_FROM_DPU, DPU_XFER_DEFAULT, DPU_SYNCHRONOUS };
struct dpu_symbol_t { uint32_t address, size; };
struct dpu_t {
    uint32_t member, ci, active;
    int32_t mram[6];
    uint32_t metadata[2];
    void *transfer;
};
struct dpu_set_t { struct dpu_t *dpu; uint32_t count; };
#define DPU_ASSERT(call) assert((call) == 0)
#define DPU_RANK_FOREACH(set, rank) \
    for (int once = ((rank = (set)), 1); once; once = 0)
#define DPU_FOREACH(set, unit, index) \
    for (index = 0; index < (set).count && \
         ((unit = (struct dpu_set_t){(set).dpu + index, 1}), 1); ++index)

int dpu_alloc(uint32_t count, const char *profile, struct dpu_set_t *set);
int dpu_free(struct dpu_set_t set);
int dpu_load(struct dpu_set_t set, const char *path, void *program);
int dpu_get_nr_dpus(struct dpu_set_t set, uint32_t *count);
int dpu_prepare_xfer(struct dpu_set_t set, void *buffer);
int dpu_push_xfer_symbol(struct dpu_set_t set, dpu_xfer_t direction,
                         struct dpu_symbol_t symbol, uint32_t offset,
                         size_t bytes, dpu_xfer_flags_t flags);
int dpu_push_xfer(struct dpu_set_t set, dpu_xfer_t direction, const char *symbol,
                  uint32_t offset, size_t bytes, dpu_xfer_flags_t flags);
int dpu_launch(struct dpu_set_t set, dpu_launch_policy_t policy);
#endif
