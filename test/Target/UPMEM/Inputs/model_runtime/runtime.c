#include "upmem_model.h"
#include <stdio.h>
#include <string.h>

static struct dpu_t devices[8];
static unsigned allocations, releases, loads, launches, computed_units;

int dpu_alloc(uint32_t count, const char *profile, struct dpu_set_t *set) {
    (void)profile;
    assert(count == 8);
    ++allocations;
    for (uint32_t i = 0; i < count; ++i)
        devices[i].ci = 7 - i;
    *set = (struct dpu_set_t){devices, count};
    return 0;
}
int dpu_free(struct dpu_set_t set) {
    assert(set.dpu == devices);
    ++releases;
    return 0;
}
int dpu_load(struct dpu_set_t set, const char *path, void *program) {
    (void)path;
    (void)program;
    ++loads;
    for (uint32_t i = 0; i < set.count; ++i) {
        set.dpu[i].active = 1;
        memset(set.dpu[i].metadata, 0, sizeof(set.dpu[i].metadata));
    }
    return 0;
}
int dpu_get_nr_dpus(struct dpu_set_t set, uint32_t *count) {
    *count = set.count;
    return 0;
}
int dpu_prepare_xfer(struct dpu_set_t set, void *buffer) {
    assert(set.count == 1 && buffer != NULL);
    set.dpu->transfer = buffer;
    return 0;
}
int dpu_push_xfer_symbol(struct dpu_set_t set, dpu_xfer_t direction,
                         struct dpu_symbol_t symbol, uint32_t offset,
                         size_t bytes, dpu_xfer_flags_t flags) {
    (void)flags;
    uint32_t address = symbol.address - 0x08000000 + offset;
    assert(address + bytes <= sizeof(devices[0].mram));
    for (uint32_t i = 0; i < set.count; ++i) {
        void *memory = (unsigned char *)set.dpu[i].mram + address;
        if (direction == DPU_XFER_TO_DPU)
            memcpy(memory, set.dpu[i].transfer, bytes);
        else
            memcpy(set.dpu[i].transfer, memory, bytes);
    }
    return 0;
}
int dpu_push_xfer(struct dpu_set_t set, dpu_xfer_t direction, const char *symbol,
                  uint32_t offset, size_t bytes, dpu_xfer_flags_t flags) {
    (void)flags;
    assert(direction == DPU_XFER_TO_DPU && offset == 0);
    for (uint32_t i = 0; i < set.count; ++i) {
        if (strcmp(symbol, "_pim_model_active") == 0) {
            assert(bytes == 4);
            memcpy(&set.dpu[i].active, set.dpu[i].transfer, bytes);
        } else {
            assert(strcmp(symbol, "_pim_metadata") == 0 && bytes == 8);
            memcpy(set.dpu[i].metadata, set.dpu[i].transfer, bytes);
        }
    }
    return 0;
}
int dpu_launch(struct dpu_set_t set, dpu_launch_policy_t policy) {
    assert(policy == DPU_SYNCHRONOUS);
    ++launches;
    for (uint32_t i = 0; i < set.count; ++i) {
        if (!set.dpu[i].active)
            continue;
        ++computed_units;
        assert(set.dpu[i].metadata[0] < 2);
        for (uint32_t j = 0; j < 2; ++j)
            set.dpu[i].mram[4 + j] = set.dpu[i].mram[j] + set.dpu[i].mram[2 + j];
    }
    return 0;
}
int upmem_resident_owns_set(void) { return 0; }
struct dpu_set_t upmem_resident_acquire(uint32_t count) {
    (void)count;
    assert(0);
    return (struct dpu_set_t){0};
}

extern uint64_t pim_model_pool_bytes[];
extern uint32_t pim_model_pool_count, pim_model_metadata_bytes;
extern void model_pack(struct upmem_model_kernel *, int32_t *, int32_t *, int32_t *);

int main(void) {
    const uint32_t partitions[] = {2, 1, 1}, mapping[] = {1, 1, 1};
    const uint32_t addresses[] = {0x08000000, 0x08000008, 0x08000010};
    const uint32_t sizes[] = {8, 8, 8};
    uint32_t roles[] = {UPMEM_RESIDENT_INPUT, UPMEM_RESIDENT_INPUT, UPMEM_RESIDENT_OUTPUT};
    upmem_session_open(8);
    assert(upmem_session_num_units() == 8);
    assert(upmem_runtime_load("batch.elf") == 1);
    assert(upmem_runtime_load("batch.elf") == 0);
    struct upmem_model_kernel *batch = upmem_model_create(5, 3, partitions, mapping,
        roles, addresses, sizes, pim_model_pool_bytes, pim_model_pool_count,
        pim_model_metadata_bytes, 3);
    assert(batch->mapping[0] == UINT32_MAX && batch->mapping[7] == 0);
    int32_t a[5][4], b[5][2], c[5][4];
    void *arguments[15];
    for (unsigned h = 0; h < 5; ++h) {
        arguments[3 * h] = a[h];
        arguments[3 * h + 1] = b[h];
        arguments[3 * h + 2] = c[h];
    }
    upmem_model_arguments(batch, arguments, 0);
    for (unsigned invocation = 0; invocation < 2; ++invocation) {
        if (invocation)
            for (unsigned i = 0; i < 8; ++i)
                if (batch->mapping[i] != UINT32_MAX)
                    batch->mapping[i] = 5 - batch->mapping[i];
        for (unsigned h = 0; h < 5; ++h) {
            for (unsigned j = 0; j < 4; ++j) a[h][j] = 100 * invocation + 10 * h + j;
            for (unsigned j = 0; j < 2; ++j) b[h][j] = 7 * h - (int)j;
        }
        model_pack(batch, NULL, NULL, NULL);
        for (unsigned h = 0; h < 5; ++h)
            for (unsigned j = 0; j < 4; ++j)
                assert(c[h][j] == a[h][j] + b[h][j % 2]);
    }
    assert(batch->stats.launches == 4 && batch->stats.activation_bytes == 320);
    assert(batch->stats.output_bytes == 160 && batch->stats.weight_bytes == 0);
    assert(computed_units == 20);
    upmem_model_close(batch);

    struct upmem_model_kernel *full = upmem_model_create(4, 4, partitions, mapping,
        roles, addresses, sizes, pim_model_pool_bytes, pim_model_pool_count,
        pim_model_metadata_bytes, 3);
    upmem_model_arguments(full, arguments, 0);
    model_pack(full, NULL, NULL, NULL);
    for (unsigned h = 0; h < 4; ++h)
        for (unsigned j = 0; j < 4; ++j)
            assert(c[h][j] == a[h][j] + b[h][j % 2]);
    assert(full->stats.launches == 1 && full->stats.activation_bytes == 128);
    assert(computed_units == 28);
    upmem_model_close(full);

    roles[1] = UPMEM_RESIDENT_WEIGHT;
    struct upmem_model_kernel *weighted = upmem_model_create(1, 1, partitions, mapping,
        roles, addresses, sizes, pim_model_pool_bytes, pim_model_pool_count,
        pim_model_metadata_bytes, 3);
    upmem_model_arguments(weighted, arguments, 1);
    model_pack(weighted, NULL, NULL, NULL);
    assert(weighted->stats.weight_bytes == 16 && weighted->stats.launches == 0);
    assert(upmem_runtime_load("weighted.elf") == 1);
    for (unsigned invocation = 0; invocation < 2; ++invocation) {
        a[0][0] += 100;
        upmem_model_arguments(weighted, arguments, 0);
        model_pack(weighted, NULL, NULL, NULL);
        for (unsigned j = 0; j < 4; ++j)
            assert(c[0][j] == a[0][j] + b[0][j % 2]);
    }
    assert(weighted->stats.weight_bytes == 16 && weighted->stats.activation_bytes == 32);
    upmem_model_close(weighted);
    upmem_session_close();
    assert(allocations == 1 && releases == 1 && loads == 2 && launches == 7);
    puts("model runtime: batch tails, remapping, fresh inputs and stationary weights OK");
    return 0;
}
