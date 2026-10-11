#ifndef UPMEM_MODEL_H
#define UPMEM_MODEL_H

#include "upmem_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

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
    void **pools;
    uint32_t pool_count;
    void *metadata;
    uint32_t metadata_bytes;
    void **arguments;
    uint32_t argument_count;
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
void upmem_model_close(struct upmem_model_kernel *kernel);

#ifdef __cplusplus
}
#endif
#endif
