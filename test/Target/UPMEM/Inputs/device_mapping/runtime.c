#include "upmem_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int dpu_get_nr_dpus(struct dpu_set_t set, uint32_t *count) {
    *count = 0;
    for (uint32_t r = 0; r < set.rank_count; ++r)
        *count += set.ranks[r].count;
    return 0;
}
int dpu_alloc(uint32_t count, const char *profile, struct dpu_set_t *set) {
    (void)count; (void)profile; (void)set;
    return 1;
}
int dpu_load(struct dpu_set_t set, const char *path, void *program) {
    (void)set; (void)path; (void)program;
    return 1;
}
int dpu_free(struct dpu_set_t set) { (void)set; return 0; }

int upmem_resident_owns_set(void) { return 0; }
struct dpu_set_t upmem_resident_acquire(uint32_t num_units) {
    (void)num_units;
    abort();
}

static void make_rank(struct test_rank *rank, int missing_member, int missing_ci,
                      int ci_major) {
    rank->count = 0;
    for (int i = 0; i < 64; ++i) {
        int member = ci_major ? i % 8 : i / 8;
        int ci = ci_major ? i / 8 : i % 8;
        if (member == missing_member && ci == missing_ci)
            continue;
        rank->dpus[rank->count++] = (struct dpu_t){member, ci};
    }
}

static void check_assignment(struct test_rank *ranks, uint32_t rank_count,
                             const uint32_t partitions[3],
                             const uint32_t mapping[3],
                             uint32_t distinct_a, uint32_t distinct_b,
                             uint32_t full_groups) {
    uint32_t units = partitions[0] * partitions[1] * partitions[2];
    uint32_t logical[units], received[units];
    unsigned char seen[units];
    memset(seen, 0, sizeof seen);
    memset(logical, 0xff, sizeof logical);
    struct dpu_set_t set = {.ranks = ranks, .rank_count = rank_count};
    upmem_bind_devices(set, 3, partitions, mapping, logical);
    for (uint32_t i = 0; i < units; ++i) {
        assert(logical[i] < units);
        assert(!seen[logical[i]]);
        seen[logical[i]] = 1;
        // Metadata identifies the logical result, which gather must put back there.
        received[logical[i]] = logical[i] * 17 + 3;
    }
    for (uint32_t i = 0; i < units; ++i)
        assert(received[i] == i * 17 + 3);

    uint32_t base = 0, complete = 0;
    for (uint32_t r = 0; r < rank_count; ++r) {
        for (uint32_t member = 0; member < 8; ++member) {
            uint32_t a[8], b[8], recipients = 0;
            for (uint32_t i = 0; i < ranks[r].count; ++i) {
                if (ranks[r].dpus[i].member != member)
                    continue;
                uint32_t id = logical[base + i];
                uint32_t k = id % partitions[2];
                uint32_t n = id / partitions[2] % partitions[1];
                uint32_t m = id / (partitions[1] * partitions[2]);
                a[recipients] = m * partitions[2] + k;
                b[recipients] = n * partitions[2] + k;
                ++recipients;
            }
            if (recipients != 8)
                continue;
            ++complete;
            uint32_t unique_a = 0, unique_b = 0;
            for (uint32_t i = 0; i < 8; ++i) {
                int new_a = 1, new_b = 1;
                for (uint32_t j = 0; j < i; ++j) {
                    new_a &= a[i] != a[j];
                    new_b &= b[i] != b[j];
                }
                unique_a += new_a;
                unique_b += new_b;
            }
            assert(unique_a == distinct_a);
            assert(unique_b == distinct_b);
        }
        base += ranks[r].count;
    }
    assert(base == units);
    assert(complete == full_groups);
}

int main(void) {
    struct test_rank ranks[33];
    const uint32_t square[] = {4, 4, 4}, mixed[] = {2, 4, 1};
    for (int ci_major = 0; ci_major < 2; ++ci_major) {
        make_rank(&ranks[0], -1, -1, ci_major);
        check_assignment(ranks, 1, square, mixed, 2, 4, 8);
    }

    const uint32_t wide[] = {4, 128, 4};
    const uint32_t share_a[] = {1, 8, 1}, share_b[] = {4, 2, 1};
    for (uint32_t r = 0; r < 32; ++r)
        make_rank(&ranks[r], -1, -1, r % 2);
    check_assignment(ranks, 32, wide, share_a, 1, 8, 256);
    check_assignment(ranks, 32, wide, share_b, 4, 2, 256);

    // A 63-DPU rank plus a one-DPU rank still provides all 2048 logical slots.
    make_rank(&ranks[0], 0, 3, 1);
    ranks[32].count = 1;
    ranks[32].dpus[0] = (struct dpu_t){7, 7};
    check_assignment(ranks, 33, wide, share_a, 1, 8, 255);
    // All complete groups must retain their sharing; the final 8 DPUs are partial.
    uint32_t map[2048];
    unsigned char seen[2048] = {0};
    struct dpu_set_t set = {.ranks = ranks, .rank_count = 33};
    upmem_bind_devices(set, 3, wide, share_a, map);
    uint32_t base = 0;
    for (uint32_t r = 0; r < 33; ++r) {
        for (uint32_t i = 0; i < ranks[r].count; ++i) {
            uint32_t id = map[base + i];
            assert(id < 2048 && !seen[id]);
            seen[id] = 1;
            if ((r == 0 && ranks[r].dpus[i].member == 0) || r == 32)
                assert(id / 512 == 3 && id % 4 == 3 && id / 4 % 128 >= 120);
        }
        base += ranks[r].count;
    }
    assert(base == 2048);
    puts("physical DPU assignment: passed");
    return 0;
}
