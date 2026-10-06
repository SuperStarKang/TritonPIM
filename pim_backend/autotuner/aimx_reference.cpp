// Link with pim_aimx_sw/ubench/matmul/{pim_mac_simul,util}.cpp.
#include "pim_mac_simul.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

static uint16_t aimx_add_words(uint16_t lhs, uint16_t rhs) {
  AccumulatorData acc{};
  acc = addBfloat16(multiplyBfloat16(BF16(lhs), BF16(0x3f80)), acc);
  acc = addBfloat16(multiplyBfloat16(BF16(rhs), BF16(0x3f80)), acc);
  return static_cast<uint16_t>(normalize(acc));
}

// A[sampled_rows,K] and B[K,sampled_cols] contain the staged BF16 bits.
// This models gemm_kernel with one FPGA, including its final pim.combine.
extern "C" void aimx_gemm_reference(
    const uint16_t *a, const uint16_t *b, float *out,
    int64_t sampled_rows, int64_t sampled_cols, int64_t k,
    int64_t group_k, int64_t tile_k) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  static_assert(sizeof(short) == sizeof(uint16_t));
  assert(a && b && out);
  assert(sampled_rows > 0 && sampled_cols > 0 && k > 0);
  assert(group_k > 0 && tile_k > 0);
  const int64_t physical_k = ((group_k + 255) / 256) * 256;
  const int64_t tile_extent = group_k * tile_k;

  for (int64_t i = 0; i < sampled_rows; ++i) {
    for (int64_t j = 0; j < sampled_cols; ++j) {
      for (int64_t tile_begin = 0; tile_begin < k; tile_begin += tile_extent) {
        uint16_t acc = 0;
        for (int64_t group_begin = tile_begin;
             group_begin < std::min(k, tile_begin + tile_extent);
             group_begin += group_k) {
          AccumulatorData mac{};
          for (int64_t offset = 0; offset < physical_k; ++offset) {
            if (offset < group_k && group_begin + offset < k) {
              mac = addBfloat16(
                  multiplyBfloat16(
                      BF16(a[i * k + group_begin + offset]),
                      BF16(b[(group_begin + offset) * sampled_cols + j])),
                  mac);
            } else {
              // Physical K padding and the logical K tail contain zeros.
              mac = addBfloat16(multiplyBfloat16(BF16(0), BF16(0)), mac);
            }
          }
          acc = aimx_add_words(acc, static_cast<uint16_t>(normalize(mac)));
        }

        // pim.combine adds the tile partial into a zero-initialized device C.
        uint32_t bits = static_cast<uint32_t>(aimx_add_words(0, acc)) << 16;
        float tile_output;
        std::memcpy(&tile_output, &bits, sizeof(tile_output));
        if (tile_begin == 0)
          out[i * sampled_cols + j] = tile_output;
        else
          out[i * sampled_cols + j] += tile_output;
      }
    }
  }
}
