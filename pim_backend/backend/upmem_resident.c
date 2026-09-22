#include "upmem_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct resident_window {
    uint64_t hash;
    uint32_t address;
    unsigned char *data;
};

struct resident_binding {
    char *symbol;
    uint32_t bytes;
    uint32_t role;
    uint32_t address;
    uint32_t count;
    struct resident_window *windows;
};

struct resident_round {
    uint32_t window[3];
    char *program;
};

struct resident_scratch {
    uint32_t address;
    uint32_t bytes;
    unsigned char *data;
    int valid;
};

struct upmem_resident_kernel {
    struct upmem_resident_block *block;
    struct resident_binding bindings[3];
    struct resident_round *rounds;
    uint32_t count;
    uint32_t input_mask;
    int captured;
    struct upmem_resident_kernel *next;
};

struct upmem_resident_block {
    struct dpu_set_t set;
    struct dpu_set_t *dpus;
    uint32_t units;
    void **prepared;
    struct resident_scratch scratch[2];
    struct upmem_resident_kernel *first;
    struct upmem_resident_kernel *last;
    struct upmem_resident_stats stats;
    int uploaded;
};

static struct upmem_resident_block *resident_owner;
static struct upmem_resident_kernel *resident_active;
static uint32_t resident_round;
static uint32_t resident_prepared;
static uint32_t resident_seen;
static int resident_open;
static int resident_launched;
static int resident_capture;

static void resident_require(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "UPMEM resident: %s\n", message);
        abort();
    }
}

static void *resident_allocate(size_t bytes) {
    void *memory = calloc(1, bytes);
    resident_require(memory != NULL, "allocation failed");
    return memory;
}

int upmem_resident_owns_set(void) {
    return resident_owner != NULL;
}

struct dpu_set_t upmem_resident_acquire(uint32_t num_units) {
    resident_require(resident_active != NULL, "the DPU set belongs to a ResidentBlock");
    resident_require(num_units == resident_owner->units, "all kernels must use the whole DPU set");
    return resident_owner->set;
}

struct upmem_resident_block *upmem_resident_create(uint32_t num_units) {
    resident_require(resident_owner == NULL, "close the current ResidentBlock before preparing another");
    resident_require(num_units > 0, "DPU count must be positive");
    struct upmem_resident_block *block = resident_allocate(sizeof(*block));
    block->set = upmem_runtime_reserve(num_units);
    block->units = num_units;
    block->dpus = resident_allocate(num_units * sizeof(*block->dpus));
    block->prepared = resident_allocate(num_units * sizeof(*block->prepared));
    uint32_t actual;
    DPU_ASSERT(dpu_get_nr_dpus(block->set, &actual));
    resident_require(actual == num_units, "allocated DPU count differs from the plan");
    struct dpu_set_t dpu;
    uint32_t index;
    DPU_FOREACH(block->set, dpu, index)
        block->dpus[index] = dpu;
    resident_owner = block;
    return block;
}

struct upmem_resident_kernel *upmem_resident_add_kernel(
    struct upmem_resident_block *block, const char **symbols,
    const uint32_t *sizes, const uint32_t *roles) {
    resident_require(block == resident_owner && !block->uploaded, "kernel registration is closed");
    struct upmem_resident_kernel *kernel = resident_allocate(sizeof(*kernel));
    kernel->block = block;
    uint32_t outputs = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        resident_require(sizes[i] > 0 && sizes[i] % 8 == 0, "MRAM slots must have an 8-byte aligned size");
        resident_require(roles[i] >= UPMEM_RESIDENT_WEIGHT && roles[i] <= UPMEM_RESIDENT_OUTPUT,
                         "unknown binding role");
        kernel->bindings[i].symbol = strdup(symbols[i]);
        resident_require(kernel->bindings[i].symbol != NULL, "symbol allocation failed");
        kernel->bindings[i].bytes = sizes[i];
        kernel->bindings[i].role = roles[i];
        if (roles[i] == UPMEM_RESIDENT_OUTPUT)
            ++outputs;
        else
            kernel->input_mask |= 1u << i;
    }
    resident_require(outputs == 1, "a resident GEMM must have two inputs and one output");
    if (block->last != NULL)
        block->last->next = kernel;
    else
        block->first = kernel;
    block->last = kernel;
    return kernel;
}

void upmem_resident_begin(struct upmem_resident_kernel *kernel, int capture) {
    resident_require(resident_active == NULL, "nested or concurrent resident invocation");
    resident_require(kernel->block == resident_owner, "kernel does not own the DPU set");
    resident_require(capture ? !kernel->captured && !kernel->block->uploaded
                             : kernel->captured && kernel->block->uploaded,
                     "invalid capture/run lifecycle");
    resident_active = kernel;
    resident_capture = capture;
    resident_round = 0;
    resident_prepared = 0;
    resident_seen = 0;
    resident_open = 0;
    resident_launched = 0;
}

void upmem_resident_cancel(void) {
    resident_active = NULL;
    resident_prepared = 0;
}

void upmem_resident_end(struct upmem_resident_kernel *kernel) {
    resident_require(resident_active == kernel && resident_prepared == 0 && !resident_open,
                     "unfinished transfer at the end of a host invocation");
    resident_require(resident_round == kernel->count && kernel->count > 0,
                     "host scheduling rounds changed");
    if (resident_capture)
        kernel->captured = 1;
    resident_active = NULL;
}

static void resident_start_round(void) {
    if (resident_open)
        return;
    struct upmem_resident_kernel *kernel = resident_active;
    if (resident_capture) {
        kernel->rounds = realloc(kernel->rounds, (kernel->count + 1) * sizeof(*kernel->rounds));
        resident_require(kernel->rounds != NULL, "round allocation failed");
        struct resident_round *round = &kernel->rounds[kernel->count++];
        for (uint32_t i = 0; i < 3; ++i)
            round->window[i] = UINT32_MAX;
        round->program = NULL;
    } else {
        resident_require(resident_round < kernel->count, "more scheduling rounds than captured");
        resident_require(kernel->rounds[resident_round].program != NULL, "round program is missing");
        kernel->block->stats.program_loads += upmem_runtime_load(kernel->rounds[resident_round].program);
    }
    resident_seen = 0;
    resident_launched = 0;
    resident_open = 1;
}

dpu_error_t upmem_resident_prepare_xfer(struct dpu_set_t dpu, void *buffer) {
    resident_require(resident_active != NULL, "resident transfer outside an invocation");
    struct upmem_resident_block *block = resident_active->block;
    resident_require(resident_prepared < block->units, "too many prepared DPU buffers");
    resident_require(dpu.dpu == block->dpus[resident_prepared].dpu,
                     "DPU order differs from the reserved set");
    block->prepared[resident_prepared++] = buffer;
    return DPU_OK;
}

static int resident_equal(const unsigned char *data, uint32_t bytes) {
    struct upmem_resident_block *block = resident_active->block;
    for (uint32_t dpu = 0; dpu < block->units; ++dpu)
        if (memcmp(data + (size_t)dpu * bytes, block->prepared[dpu], bytes) != 0)
            return 0;
    return 1;
}

static void resident_copy(unsigned char *data, uint32_t bytes) {
    struct upmem_resident_block *block = resident_active->block;
    for (uint32_t dpu = 0; dpu < block->units; ++dpu)
        memcpy(data + (size_t)dpu * bytes, block->prepared[dpu], bytes);
}

static uint32_t resident_capture_weight(struct resident_binding *binding) {
    struct upmem_resident_block *block = resident_active->block;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint32_t dpu = 0; dpu < block->units; ++dpu) {
        const unsigned char *bytes = block->prepared[dpu];
        for (uint32_t i = 0; i < binding->bytes; ++i) {
            hash ^= bytes[i];
            hash *= UINT64_C(1099511628211);
        }
    }
    for (uint32_t i = 0; i < binding->count; ++i)
        if (binding->windows[i].hash == hash && resident_equal(binding->windows[i].data, binding->bytes))
            return i;
    binding->windows = realloc(binding->windows, (binding->count + 1) * sizeof(*binding->windows));
    resident_require(binding->windows != NULL, "weight window allocation failed");
    struct resident_window *window = &binding->windows[binding->count];
    window->hash = hash;
    window->address = 0;
    window->data = resident_allocate((size_t)binding->bytes * block->units);
    resident_copy(window->data, binding->bytes);
    return binding->count++;
}

static void resident_prepare_sdk(void) {
    struct upmem_resident_block *block = resident_active->block;
    for (uint32_t dpu = 0; dpu < block->units; ++dpu)
        DPU_ASSERT(dpu_prepare_xfer(block->dpus[dpu], block->prepared[dpu]));
}

static void resident_send_input(struct resident_binding *binding) {
    struct upmem_resident_block *block = resident_active->block;
    for (uint32_t i = 0; i < 2; ++i) {
        struct resident_scratch *scratch = &block->scratch[i];
        if (scratch->valid && scratch->address == binding->address && scratch->bytes == binding->bytes
            && resident_equal(scratch->data, binding->bytes)) {
            ++block->stats.activation_hits;
            return;
        }
    }
    for (uint32_t i = 0; i < 2; ++i) {
        struct resident_scratch *scratch = &block->scratch[i];
        if (scratch->valid && scratch->address < binding->address + binding->bytes
            && binding->address < scratch->address + scratch->bytes)
            scratch->valid = 0;
    }
    uint32_t slot = 0;
    for (struct resident_binding *previous = resident_active->bindings;
         previous < binding; ++previous)
        if (previous->role == UPMEM_RESIDENT_INPUT)
            ++slot;
    struct resident_scratch *scratch = &block->scratch[slot];
    scratch->data = realloc(scratch->data, (size_t)binding->bytes * block->units);
    resident_require(scratch->data != NULL, "input scratch allocation failed");
    resident_copy(scratch->data, binding->bytes);
    scratch->address = binding->address;
    scratch->bytes = binding->bytes;
    resident_prepare_sdk();
    struct dpu_symbol_t symbol = {binding->address, binding->bytes};
    DPU_ASSERT(dpu_push_xfer_symbol(block->set, DPU_XFER_TO_DPU, symbol, 0,
                                  binding->bytes, DPU_XFER_DEFAULT));
    scratch->valid = 1;
    block->stats.activation_bytes += (uint64_t)binding->bytes * block->units;
}

dpu_error_t upmem_resident_push_xfer(struct dpu_set_t set, dpu_xfer_t direction,
                                    const char *symbol, uint32_t offset,
                                    size_t length, dpu_xfer_flags_t flags) {
    resident_require(resident_active != NULL, "resident push outside an invocation");
    struct upmem_resident_kernel *kernel = resident_active;
    struct upmem_resident_block *block = kernel->block;
    resident_require(resident_prepared == block->units, "incomplete DPU transfer registration");
    resident_require(flags == DPU_XFER_DEFAULT, "resident transfers must be synchronous");
    resident_start_round();
    uint32_t index = 0;
    while (index < 3 && strcmp(symbol, kernel->bindings[index].symbol) != 0)
        ++index;
    if (index == 3) {
        resident_require(direction == DPU_XFER_TO_DPU && !resident_launched,
                         "unexpected metadata transfer");
        if (!resident_capture) {
            resident_prepare_sdk();
            DPU_ASSERT(dpu_push_xfer(set, direction, symbol, offset, length, flags));
            block->stats.metadata_bytes += (uint64_t)length * block->units;
        }
    } else {
        struct resident_binding *binding = &kernel->bindings[index];
        resident_require(offset == 0 && length == binding->bytes, "transfer differs from fixed MRAM slot");
        if (binding->role == UPMEM_RESIDENT_OUTPUT) {
            resident_require(direction == DPU_XFER_FROM_DPU && resident_launched, "output before launch");
            if (resident_capture) {
                for (uint32_t dpu = 0; dpu < block->units; ++dpu)
                    memset(block->prepared[dpu], 0, length);
            } else {
                resident_prepare_sdk();
                struct dpu_symbol_t output = {binding->address, binding->bytes};
                DPU_ASSERT(dpu_push_xfer_symbol(set, direction, output, 0, length, flags));
                block->stats.output_bytes += (uint64_t)length * block->units;
            }
            resident_open = 0;
            ++resident_round;
        } else {
            resident_require(direction == DPU_XFER_TO_DPU && !resident_launched,
                             "input after launch");
            resident_require(!(resident_seen & (1u << index)), "input binding sent twice in a round");
            resident_seen |= 1u << index;
            if (binding->role == UPMEM_RESIDENT_WEIGHT) {
                if (resident_capture)
                    kernel->rounds[resident_round].window[index] = resident_capture_weight(binding);
            } else if (!resident_capture) {
                resident_send_input(binding);
            }
        }
    }
    resident_prepared = 0;
    return DPU_OK;
}

dpu_error_t upmem_resident_launch(struct dpu_set_t set, dpu_launch_policy_t policy) {
    resident_require(resident_active != NULL && resident_open && !resident_launched,
                     "launch outside a prepared round");
    resident_require(resident_seen == resident_active->input_mask && resident_prepared == 0,
                     "launch is missing input transfers");
    resident_require(policy == DPU_SYNCHRONOUS, "resident launch must be synchronous");
    if (!resident_capture) {
        DPU_ASSERT(dpu_launch(set, policy));
        ++resident_active->block->stats.launches;
    }
    resident_launched = 1;
    return DPU_OK;
}

uint32_t upmem_resident_windows(struct upmem_resident_kernel *kernel, uint32_t binding) {
    resident_require(binding < 3 && kernel->captured, "invalid window query");
    return kernel->bindings[binding].count;
}

uint32_t upmem_resident_rounds(struct upmem_resident_kernel *kernel) {
    resident_require(kernel->captured, "kernel has not been captured");
    return kernel->count;
}

uint32_t upmem_resident_window(struct upmem_resident_kernel *kernel,
                             uint32_t round, uint32_t binding) {
    resident_require(round < kernel->count && binding < 3, "invalid round window query");
    return kernel->rounds[round].window[binding];
}

void upmem_resident_address(struct upmem_resident_kernel *kernel,
                            uint32_t binding, uint32_t window, uint32_t address) {
    resident_require(binding < 3 && !kernel->block->uploaded, "invalid address assignment");
    resident_require(address >= 0x08000000 && address % 8 == 0
                     && (uint64_t)address + kernel->bindings[binding].bytes <= 0x0c000000,
                     "MRAM address is out of range or unaligned");
    struct resident_binding *slot = &kernel->bindings[binding];
    if (slot->role == UPMEM_RESIDENT_WEIGHT) {
        resident_require(window < slot->count, "invalid weight window");
        slot->windows[window].address = address;
    } else {
        resident_require(window == 0, "scratch has one shared window");
        slot->address = address;
    }
}

void upmem_resident_program(struct upmem_resident_kernel *kernel,
                            uint32_t round, const char *path) {
    resident_require(round < kernel->count && !kernel->block->uploaded, "invalid program assignment");
    resident_require(kernel->rounds[round].program == NULL, "round program already assigned");
    kernel->rounds[round].program = strdup(path);
    resident_require(kernel->rounds[round].program != NULL, "program path allocation failed");
}

void upmem_resident_upload(struct upmem_resident_block *block) {
    resident_require(block == resident_owner && !block->uploaded && resident_active == NULL,
                     "invalid weight upload lifecycle");
    resident_require(block->first != NULL && block->first->captured, "no captured GEMM");
    resident_require(block->first->rounds[0].program != NULL, "first round program is missing");
    block->stats.program_loads += upmem_runtime_load(block->first->rounds[0].program);
    for (struct upmem_resident_kernel *kernel = block->first; kernel != NULL; kernel = kernel->next) {
        resident_require(kernel->captured, "uncaptured kernel");
        for (uint32_t round = 0; round < kernel->count; ++round)
            resident_require(kernel->rounds[round].program != NULL, "missing round program");
        for (uint32_t binding = 0; binding < 3; ++binding) {
            struct resident_binding *slot = &kernel->bindings[binding];
            if (slot->role != UPMEM_RESIDENT_WEIGHT) {
                resident_require(slot->address != 0, "missing scratch address");
                continue;
            }
            for (uint32_t i = 0; i < slot->count; ++i) {
                struct resident_window *window = &slot->windows[i];
                resident_require(window->address != 0, "missing weight address");
                for (uint32_t dpu = 0; dpu < block->units; ++dpu)
                    DPU_ASSERT(dpu_prepare_xfer(block->dpus[dpu], window->data + (size_t)dpu * slot->bytes));
                struct dpu_symbol_t symbol = {window->address, slot->bytes};
                DPU_ASSERT(dpu_push_xfer_symbol(block->set, DPU_XFER_TO_DPU, symbol, 0,
                                              slot->bytes, DPU_XFER_DEFAULT));
                block->stats.weight_bytes += (uint64_t)slot->bytes * block->units;
                free(window->data);
                window->data = NULL;
            }
        }
    }
    block->uploaded = 1;
}

void upmem_resident_get_stats(struct upmem_resident_block *block,
                              struct upmem_resident_stats *stats) {
    resident_require(block == resident_owner, "closed resident block");
    *stats = block->stats;
}

void upmem_resident_reset_inputs(struct upmem_resident_block *block) {
    resident_require(block == resident_owner && block->uploaded && resident_active == NULL,
                     "cannot start a block invocation in the current state");
    for (uint32_t i = 0; i < 2; ++i)
        block->scratch[i].valid = 0;
}

void upmem_resident_close(struct upmem_resident_block *block) {
    resident_require(block == resident_owner && resident_active == NULL,
                     "cannot close an active or foreign resident block");
    struct upmem_resident_kernel *kernel = block->first;
    while (kernel != NULL) {
        struct upmem_resident_kernel *next = kernel->next;
        for (uint32_t i = 0; i < 3; ++i) {
            for (uint32_t w = 0; w < kernel->bindings[i].count; ++w)
                free(kernel->bindings[i].windows[w].data);
            free(kernel->bindings[i].windows);
            free(kernel->bindings[i].symbol);
        }
        for (uint32_t r = 0; r < kernel->count; ++r)
            free(kernel->rounds[r].program);
        free(kernel->rounds);
        free(kernel);
        kernel = next;
    }
    for (uint32_t i = 0; i < 2; ++i)
        free(block->scratch[i].data);
    free(block->prepared);
    free(block->dpus);
    free(block);
    resident_owner = NULL;
}
