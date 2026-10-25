#include "upmem_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

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
    kernel->kv_jobs = model_allocate((size_t)batch_size * kernel->logical_units);
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
    // pool은 쓰는 동안에만 잡는다 (upmem_model_pools_acquire).
    kernel->pool_count = pool_count;
    kernel->pools = model_allocate(pool_count * sizeof(void *));
    kernel->pool_bytes = model_allocate(pool_count * sizeof(uint64_t));
    for (uint32_t pool = 0; pool < pool_count; ++pool)
        kernel->pool_bytes[pool] = batch_size * pool_bytes[pool];
    return kernel;
}

void upmem_model_pools_acquire(struct upmem_model_kernel *kernel) {
    // mmap은 0 page를 주고 건드린 page만 host memory를 쓴다. stage가 쓰지 않는 pool은 비용이 없다.
    for (uint32_t pool = 0; pool < kernel->pool_count; ++pool) {
        if (kernel->pools[pool] != NULL || kernel->pool_bytes[pool] == 0)
            continue;
        void *memory = mmap(NULL, kernel->pool_bytes[pool], PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory == MAP_FAILED) {
            fprintf(stderr, "UPMEM model pool allocation failed\n");
            abort();
        }
        kernel->pools[pool] = memory;
    }
}

void upmem_model_pools_release(struct upmem_model_kernel *kernel) {
    for (uint32_t pool = 0; pool < kernel->pool_count; ++pool) {
        if (kernel->pools[pool] == NULL)
            continue;
        munmap(kernel->pools[pool], kernel->pool_bytes[pool]);
        kernel->pools[pool] = NULL;
    }
}

void upmem_model_arguments(struct upmem_model_kernel *kernel, void **arguments,
                           uint32_t upload_weights) {
    kernel->arguments = arguments;
    kernel->upload_weights = upload_weights;
}

static int model_binding_sent(struct upmem_model_kernel *kernel, uint32_t binding) {
    return kernel->roles[binding] == (kernel->upload_weights
        ? UPMEM_RESIDENT_WEIGHT : UPMEM_RESIDENT_INPUT);
}

int upmem_model_wants(struct upmem_model_kernel *kernel, uint32_t binding, uint32_t job) {
    if (kernel->upload_weights == UPMEM_MODEL_KV_PACK)
        return kernel->roles[binding] == UPMEM_RESIDENT_KV && kernel->kv_jobs[job];
    return model_binding_sent(kernel, binding);
}

void upmem_model_kv_select(struct upmem_model_kernel *kernel, const uint32_t *jobs, uint32_t count) {
    memset(kernel->kv_jobs, 0, (size_t)kernel->batch_size * kernel->logical_units);
    for (uint32_t index = 0; index < count; ++index)
        kernel->kv_jobs[jobs[index]] = 1;
}

void upmem_model_kv_gather(struct upmem_model_kernel *kernel, uint32_t binding, uint8_t *images) {
    // KV_PACK이 job마다 채운 cache operand 배치를 job 순서로 복사한다.
    uint32_t jobs = kernel->batch_size * kernel->logical_units;
    for (uint32_t job = 0; job < jobs; ++job)
        memcpy(images + (size_t)job * kernel->sizes[binding], kernel->pointers[binding][job],
               kernel->sizes[binding]);
}

void upmem_model_kv_push_region(struct upmem_model_kernel *kernel, uint32_t address,
                                const uint8_t *images, uint32_t length) {
    // MRAM [address, address + length)를 session의 모든 DPU에 같은 위치로 쓴다. images는 DPU마다 length bytes다.
    struct dpu_symbol_t symbol = {address, length};
    for (uint32_t physical = 0; physical < kernel->units; ++physical)
        DPU_ASSERT(dpu_prepare_xfer(kernel->dpus[physical], (void *)(images + (size_t)physical * length)));
    DPU_ASSERT(dpu_push_xfer_symbol(kernel->set, DPU_XFER_TO_DPU, symbol, 0, length, DPU_XFER_DEFAULT));
    kernel->stats.kv_bytes += (uint64_t)kernel->units * length;
}

static int model_active(struct upmem_model_kernel *kernel, uint32_t physical) {
    uint32_t logical = kernel->mapping[physical];
    return logical != UINT32_MAX
        && kernel->batch_base + logical / kernel->logical_units < kernel->instances;
}

void upmem_model_transfer(struct upmem_model_kernel *kernel, uint32_t binding,
                          int output) {
    // PACK은 prepare 시점에 pool만 채운다. 전송은 layer group 단위로 한다.
    // KV_PACK도 pool만 채운다 (배치식 대조용).
    if (kernel->upload_weights == UPMEM_MODEL_PACK || kernel->upload_weights == UPMEM_MODEL_KV_PACK)
        return;
    if (output ? kernel->upload_weights : !model_binding_sent(kernel, binding))
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
    upmem_model_pools_release(kernel);
    for (uint32_t binding = 0; binding < 3; ++binding)
        free(kernel->pointers[binding]);
    free(kernel->pools);
    free(kernel->pool_bytes);
    free(kernel->metadata);
    free(kernel->kv_jobs);
    free(kernel->mapping);
    free(kernel->active);
    free(kernel->dpus);
    free(kernel);
}

struct upmem_model_group *upmem_model_group_create(uint32_t capacity, uint32_t zero_bytes) {
    struct upmem_model_group *group = model_allocate(sizeof(*group));
    group->set = upmem_session_devices();
    group->units = upmem_session_num_units();
    group->capacity = capacity;
    group->zero_bytes = zero_bytes;
    group->addresses = model_allocate((size_t)group->units * capacity * sizeof(*group->addresses));
    group->lengths = model_allocate(capacity * sizeof(*group->lengths));
    group->zeros = model_allocate(zero_bytes);
    return group;
}

void upmem_model_group_add(struct upmem_model_group *group, struct upmem_model_kernel *kernel) {
    // pool은 batch slot 하나 분량만 weight를 담으므로 weight stage는 batch 하나여야 한다.
    if (kernel->instances > kernel->batch_size) {
        fprintf(stderr, "UPMEM model group: weight stage spans %u instances in batches of %u\n",
                kernel->instances, kernel->batch_size);
        abort();
    }
    for (uint32_t binding = 0; binding < 3; ++binding) {
        if (kernel->roles[binding] != UPMEM_RESIDENT_WEIGHT)
            continue;
        if (group->blocks == 0)
            group->base = kernel->addresses[binding];
        if (group->blocks == group->capacity || kernel->sizes[binding] > group->zero_bytes
                || kernel->addresses[binding] != group->base + group->bytes) {
            fprintf(stderr, "UPMEM model group: weight block %u at 0x%x does not follow 0x%x\n",
                    group->blocks, kernel->addresses[binding], group->base + group->bytes);
            abort();
        }
        for (uint32_t physical = 0; physical < group->units; ++physical) {
            uint32_t logical = kernel->mapping[physical];
            int active = logical != UINT32_MAX && logical / kernel->logical_units < kernel->instances;
            group->addresses[(size_t)physical * group->capacity + group->blocks] =
                active ? kernel->pointers[binding][logical] : group->zeros;
        }
        group->lengths[group->blocks++] = kernel->sizes[binding];
        group->bytes += kernel->sizes[binding];
    }
}

static bool model_group_block(struct sg_block_info *out, uint32_t dpu, uint32_t block, void *args) {
    struct upmem_model_group *group = args;
    if (block >= group->blocks)
        return false;
    out->addr = group->addresses[(size_t)dpu * group->capacity + block];
    out->length = group->lengths[block];
    return true;
}

void upmem_model_group_push(struct upmem_model_group *group) {
    struct dpu_symbol_t symbol = {group->base, group->bytes};
    get_block_t blocks = {model_group_block, group, sizeof(*group)};
    DPU_ASSERT(dpu_push_sg_xfer_symbol(group->set, DPU_XFER_TO_DPU, symbol, 0, group->bytes,
                                       &blocks, DPU_SG_XFER_DEFAULT));
    group->stats.weight_bytes += (uint64_t)group->units * group->bytes;
}

void upmem_model_group_stats(struct upmem_model_group *group,
                             struct upmem_resident_stats *stats) {
    *stats = group->stats;
}

void upmem_model_group_close(struct upmem_model_group *group) {
    free(group->zeros);
    free(group->lengths);
    free(group->addresses);
    free(group);
}
