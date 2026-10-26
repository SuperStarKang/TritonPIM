#ifndef UPMEM_RESIDENT_H
#define UPMEM_RESIDENT_H

#include <stddef.h>
#include <stdint.h>
#include <dpu.h>

enum upmem_resident_role {
    UPMEM_RESIDENT_WEIGHT = 1,
    UPMEM_RESIDENT_INPUT = 2,
    UPMEM_RESIDENT_OUTPUT = 3,
    // 매 호출 보내지 않고 MRAM의 key/value cache 영역을 그대로 읽는 입력 (model path)
    UPMEM_RESIDENT_KV = 4
};

struct upmem_resident_block;
struct upmem_resident_kernel;

struct upmem_resident_stats {
    uint64_t weight_bytes;
    uint64_t activation_bytes;
    uint64_t output_bytes;
    uint64_t metadata_bytes;
    uint64_t launches;
    uint64_t program_loads;
    uint64_t activation_hits;  // Retained in the stats ABI; always zero.
    uint64_t kv_bytes;         // key/value cache writes (model path)
};

// Reset for each captured/executed GEMM and for the initial weight upload.
struct upmem_resident_timings {
    uint64_t weight_h2d_ns;
    uint64_t activation_h2d_ns;
    uint64_t metadata_h2d_ns;
    uint64_t program_load_ns;
};

int upmem_resident_owns_set(void);
struct dpu_set_t upmem_resident_acquire(uint32_t num_units);
struct upmem_resident_block *upmem_resident_create(uint32_t num_units);
struct upmem_resident_kernel *upmem_resident_add_kernel(
    struct upmem_resident_block *block, const char **symbols,
    const uint32_t *sizes, const uint32_t *roles);
void upmem_resident_begin(struct upmem_resident_kernel *kernel, int capture);
void upmem_resident_end(struct upmem_resident_kernel *kernel);
void upmem_resident_cancel(void);
uint32_t upmem_resident_windows(struct upmem_resident_kernel *kernel, uint32_t binding);
uint32_t upmem_resident_rounds(struct upmem_resident_kernel *kernel);
uint32_t upmem_resident_window(struct upmem_resident_kernel *kernel,
                             uint32_t round, uint32_t binding);
void upmem_resident_address(struct upmem_resident_kernel *kernel,
                            uint32_t binding, uint32_t window, uint32_t address);
void upmem_resident_program(struct upmem_resident_kernel *kernel,
                            uint32_t round, const char *path);
void upmem_resident_upload(struct upmem_resident_block *block);
// Retained for callers' run-boundary validation; activations are always sent.
void upmem_resident_reset_inputs(struct upmem_resident_block *block);
void upmem_resident_get_stats(struct upmem_resident_block *block,
                              struct upmem_resident_stats *stats);
void upmem_resident_get_timings(struct upmem_resident_block *block,
                                struct upmem_resident_timings *timings);
void upmem_resident_close(struct upmem_resident_block *block);

dpu_error_t upmem_resident_prepare_xfer(struct dpu_set_t dpu, void *buffer);
dpu_error_t upmem_resident_push_xfer(struct dpu_set_t set, dpu_xfer_t direction,
                                    const char *symbol, uint32_t offset,
                                    size_t length, dpu_xfer_flags_t flags);
dpu_error_t upmem_resident_launch(struct dpu_set_t set, dpu_launch_policy_t policy);

#endif
