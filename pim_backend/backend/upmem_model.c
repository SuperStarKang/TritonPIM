#include "upmem_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *model_allocate(size_t bytes) {
    void *memory = calloc(1, bytes);
    if (memory == NULL) {
        fprintf(stderr, "UPMEM model allocation failed\n");
        abort();
    }
    return memory;
}

struct upmem_model_kernel *upmem_model_create(
    uint32_t instances, uint32_t batch_size, const uint32_t *partitions,
    const uint32_t *device_mapping, const uint32_t *roles,
    const uint32_t *addresses, const uint32_t *sizes,
    const uint64_t *pool_bytes, uint32_t pool_count, uint32_t metadata_bytes,
    uint32_t argument_count) {
    struct upmem_model_kernel *kernel = model_allocate(sizeof(*kernel));
    kernel->set = upmem_session_devices();
    kernel->units = upmem_session_num_units();
    kernel->logical_units = partitions[0] * partitions[1] * partitions[2];
    kernel->instances = instances;
    kernel->batch_size = batch_size;
    kernel->metadata_bytes = metadata_bytes;
    kernel->argument_count = argument_count;
    kernel->dpus = model_allocate(kernel->units * sizeof(*kernel->dpus));
    kernel->mapping = model_allocate(kernel->units * sizeof(*kernel->mapping));
    kernel->active = model_allocate(kernel->units * sizeof(*kernel->active));
    kernel->metadata = model_allocate((size_t)batch_size * kernel->logical_units * metadata_bytes);
    upmem_bind_instances(kernel->set, 3, partitions, device_mapping, batch_size, kernel->mapping);
    struct dpu_set_t dpu;
    uint32_t index;
    DPU_FOREACH(kernel->set, dpu, index)
        kernel->dpus[index] = dpu;
    for (uint32_t binding = 0; binding < 3; ++binding) {
        kernel->roles[binding] = roles[binding];
        kernel->addresses[binding] = addresses[binding];
        kernel->sizes[binding] = sizes[binding];
        kernel->pointers[binding] = model_allocate(
            (size_t)batch_size * kernel->logical_units * sizeof(void *));
    }
    kernel->pool_count = pool_count;
    kernel->pools = model_allocate(pool_count * sizeof(void *));
    for (uint32_t pool = 0; pool < pool_count; ++pool)
        kernel->pools[pool] = model_allocate(batch_size * pool_bytes[pool]);
    return kernel;
}

void upmem_model_arguments(struct upmem_model_kernel *kernel, void **arguments,
                           uint32_t upload_weights) {
    kernel->arguments = arguments;
    kernel->upload_weights = upload_weights;
}

int upmem_model_wants(struct upmem_model_kernel *kernel, uint32_t binding) {
    return kernel->roles[binding] == (kernel->upload_weights
        ? UPMEM_RESIDENT_WEIGHT : UPMEM_RESIDENT_INPUT);
}

static int model_active(struct upmem_model_kernel *kernel, uint32_t physical) {
    uint32_t logical = kernel->mapping[physical];
    return logical != UINT32_MAX
        && kernel->batch_base + logical / kernel->logical_units < kernel->instances;
}

void upmem_model_transfer(struct upmem_model_kernel *kernel, uint32_t binding,
                          int output) {
    if (output ? kernel->upload_weights : !upmem_model_wants(kernel, binding))
        return;
    struct dpu_symbol_t symbol = {kernel->addresses[binding], kernel->sizes[binding]};
    dpu_xfer_t direction = output ? DPU_XFER_FROM_DPU : DPU_XFER_TO_DPU;
    uint32_t active = 0;
    for (uint32_t physical = 0; physical < kernel->units; ++physical) {
        if (!model_active(kernel, physical))
            continue;
        DPU_ASSERT(dpu_prepare_xfer(kernel->dpus[physical],
            kernel->pointers[binding][kernel->mapping[physical]]));
        ++active;
    }
    if (active == kernel->units) {
        DPU_ASSERT(dpu_push_xfer_symbol(kernel->set, direction, symbol, 0,
                                      symbol.size, DPU_XFER_DEFAULT));
    } else {
        for (uint32_t physical = 0; physical < kernel->units; ++physical)
            if (model_active(kernel, physical))
                DPU_ASSERT(dpu_push_xfer_symbol(kernel->dpus[physical], direction,
                    symbol, 0, symbol.size, DPU_XFER_DEFAULT));
    }
    uint64_t bytes = (uint64_t)active * symbol.size;
    if (output)
        kernel->stats.output_bytes += bytes;
    else if (kernel->upload_weights)
        kernel->stats.weight_bytes += bytes;
    else
        kernel->stats.activation_bytes += bytes;
}

void upmem_model_metadata(struct upmem_model_kernel *kernel, const char *symbol) {
    if (kernel->upload_weights)
        return;
    uint32_t active = 0;
    for (uint32_t physical = 0; physical < kernel->units; ++physical) {
        if (!model_active(kernel, physical))
            continue;
        DPU_ASSERT(dpu_prepare_xfer(kernel->dpus[physical],
            (unsigned char *)kernel->metadata
                + (size_t)kernel->mapping[physical] * kernel->metadata_bytes));
        ++active;
    }
    if (active == kernel->units) {
        DPU_ASSERT(dpu_push_xfer(kernel->set, DPU_XFER_TO_DPU, symbol, 0,
                                kernel->metadata_bytes, DPU_XFER_DEFAULT));
    } else {
        for (uint32_t physical = 0; physical < kernel->units; ++physical)
            if (model_active(kernel, physical))
                DPU_ASSERT(dpu_push_xfer(kernel->dpus[physical], DPU_XFER_TO_DPU,
                    symbol, 0, kernel->metadata_bytes, DPU_XFER_DEFAULT));
    }
    kernel->stats.metadata_bytes += (uint64_t)active * kernel->metadata_bytes;
}

void upmem_model_launch(struct upmem_model_kernel *kernel) {
    if (kernel->upload_weights)
        return;
    for (uint32_t physical = 0; physical < kernel->units; ++physical) {
        kernel->active[physical] = model_active(kernel, physical);
        DPU_ASSERT(dpu_prepare_xfer(kernel->dpus[physical], &kernel->active[physical]));
    }
    DPU_ASSERT(dpu_push_xfer(kernel->set, DPU_XFER_TO_DPU, "_pim_model_active", 0,
                            sizeof(uint32_t), DPU_XFER_DEFAULT));
    kernel->stats.metadata_bytes += (uint64_t)kernel->units * sizeof(uint32_t);
    DPU_ASSERT(dpu_launch(kernel->set, DPU_SYNCHRONOUS));
    ++kernel->stats.launches;
}

void upmem_model_stats(struct upmem_model_kernel *kernel,
                       struct upmem_resident_stats *stats) {
    *stats = kernel->stats;
}

void upmem_model_mapping(struct upmem_model_kernel *kernel, uint32_t *mapping) {
    memcpy(mapping, kernel->mapping, kernel->units * sizeof(*mapping));
}

void upmem_model_close(struct upmem_model_kernel *kernel) {
    for (uint32_t pool = 0; pool < kernel->pool_count; ++pool)
        free(kernel->pools[pool]);
    for (uint32_t binding = 0; binding < 3; ++binding)
        free(kernel->pointers[binding]);
    free(kernel->pools);
    free(kernel->metadata);
    free(kernel->mapping);
    free(kernel->active);
    free(kernel->dpus);
    free(kernel);
}
