#include "dpu.h"
static inline uint32_t dpu_get_member_id(struct dpu_t *dpu) { return dpu->member; }
static inline uint32_t dpu_get_slice_id(struct dpu_t *dpu) { return dpu->ci; }
