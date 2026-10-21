#ifndef UPMEM_MODEL_H
#define UPMEM_MODEL_H

#include "upmem_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// upmem_model_arguments의 upload_weights 값. PACK은 weight를 pool에 배치만 하고 보내지 않는다.
#define UPMEM_MODEL_PACK 2

struct upmem_model_kernel {
    struct dpu_set_t set;
    struct dpu_set_t *dpus;
    uint32_t units;
    uint32_t logical_units;
    uint32_t instances;
    uint32_t batch_size;
    uint32_t batch_base;
    uint32_t upload_weights;
    uint32_t *mapping;
    uint32_t *active;
    uint32_t roles[3];
    uint32_t addresses[3];
    uint32_t sizes[3];
    void **pointers[3];
    // host 조립 pool. upmem_model_pools_acquire가 잡고 release가 놓는다 (그 사이에만 stage 호출 가능).
    void **pools;
    uint64_t *pool_bytes;
    uint32_t pool_count;
    void *metadata;
    uint32_t metadata_bytes;
    void **arguments;
    uint32_t argument_count;
    struct upmem_resident_stats stats;
};

// 한 layer group의 weight를 MRAM 연속 영역 [base, base + bytes)로 한 번에 보내는 block 표.
// block은 MRAM 주소 순서이고, DPU마다 stage pool의 배치된 weight 또는 zeros를 가리킨다.
struct upmem_model_group {
    struct dpu_set_t set;
    uint32_t units;
    uint32_t capacity;
    uint32_t blocks;
    uint32_t base;
    uint32_t bytes;
    uint32_t zero_bytes;
    void **addresses;   // [units * capacity]
    uint32_t *lengths;  // [capacity], 모든 DPU에서 같다
    void *zeros;
    struct upmem_resident_stats stats;
};

struct upmem_model_kernel *upmem_model_create(
    uint32_t instances, uint32_t batch_size, const uint32_t *partitions,
    const uint32_t *device_mapping, const uint32_t *roles,
    const uint32_t *addresses, const uint32_t *sizes,
    const uint64_t *pool_bytes, uint32_t pool_count, uint32_t metadata_bytes,
    uint32_t argument_count);
void upmem_model_arguments(struct upmem_model_kernel *kernel, void **arguments,
                           uint32_t upload_weights);
int upmem_model_wants(struct upmem_model_kernel *kernel, uint32_t binding);
void upmem_model_transfer(struct upmem_model_kernel *kernel, uint32_t binding,
                          int output);
void upmem_model_metadata(struct upmem_model_kernel *kernel, const char *symbol);
void upmem_model_launch(struct upmem_model_kernel *kernel);
void upmem_model_stats(struct upmem_model_kernel *kernel,
                       struct upmem_resident_stats *stats);
void upmem_model_mapping(struct upmem_model_kernel *kernel, uint32_t *mapping);
void upmem_model_pools_acquire(struct upmem_model_kernel *kernel);
void upmem_model_pools_release(struct upmem_model_kernel *kernel);
void upmem_model_close(struct upmem_model_kernel *kernel);
struct upmem_model_group *upmem_model_group_create(uint32_t capacity, uint32_t zero_bytes);
void upmem_model_group_add(struct upmem_model_group *group, struct upmem_model_kernel *kernel);
void upmem_model_group_push(struct upmem_model_group *group);
void upmem_model_group_stats(struct upmem_model_group *group,
                             struct upmem_resident_stats *stats);
void upmem_model_group_close(struct upmem_model_group *group);

#ifdef __cplusplus
}
#endif
#endif
