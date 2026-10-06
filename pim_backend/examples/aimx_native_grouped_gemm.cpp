// Candidate 1: original libpim matmul across each Tile's complete K range.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

#include <pim.h>

extern "C" {
extern int totalDevNum;
}

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "[native-reference] %s\n", message);
    std::exit(EXIT_FAILURE);
  }
}

static uint16_t *allocate_words(size_t count) {
  auto *buffer = static_cast<uint16_t *>(
      pim_malloc(count * sizeof(uint16_t), 0));
  require(buffer != nullptr && buffer != MAP_FAILED,
          "PIM allocation failed");
  return buffer;
}

static void check_written(const uint16_t *buffer, size_t count) {
  require(std::find(buffer, buffer + count, uint16_t(0x7fc1))
              == buffer + count,
          "operation left an unwritten output");
}

// A[1024,4096] and B[4096,4096] hold BF16 bits; C[1024,4096] holds FP32.
// The caller must first initialize the same libpim through a TritonPIM launch.
extern "C" void aimx_native_grouped_gemm(
    const uint16_t *input_a, const uint16_t *input_b, float *output) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  require(input_a && input_b && output, "null host buffer");
  require(totalDevNum > 0, "run TritonPIM first to initialize the driver");

  constexpr int M = 1024;
  constexpr int N = 4096;
  constexpr int K = 4096;
  constexpr int GM = 512;
  constexpr int GN = 2048;
  constexpr int GK = 128;
  constexpr int TK = 8;
  constexpr int TILE_K = GK * TK;
  constexpr int A_PITCH = 4096;
  constexpr uint32_t GROUP_ELEMENTS = GM * GN;

  uint16_t *group_a = allocate_words(GM * A_PITCH);
  uint16_t *group_b = allocate_words(TILE_K * GN);
  uint16_t *tile_output = allocate_words(GROUP_ELEMENTS);

  pim_args mac{};
  mac.srcA_va = reinterpret_cast<uint64_t>(group_a);
  mac.srcB_va = reinterpret_cast<uint64_t>(group_b);
  mac.dstC_va = reinterpret_cast<uint64_t>(tile_output);
  mac.p_size = GM;
  mac.q_size = TILE_K;
  mac.r_size = GN;

  for (int tile_k = 0; tile_k < K / (GK * TK); ++tile_k) {
    for (int m_begin = 0; m_begin < M; m_begin += GM) {
      for (int n_begin = 0; n_begin < N; n_begin += GN) {
        std::fill_n(group_a, GM * A_PITCH, uint16_t(0));

        for (int group_k = 0; group_k < TK; ++group_k) {
          int k_begin = (tile_k * TK + group_k) * GK;

          for (int row = 0; row < GM; ++row) {
            std::memcpy(
                group_a + row * A_PITCH + group_k * GK,
                input_a + (m_begin + row) * K + k_begin,
                GK * sizeof(uint16_t));
          }
          for (int k = 0; k < GK; ++k) {
            std::memcpy(
                group_b + (group_k * GK + k) * GN,
                input_b + (k_begin + k) * N + n_begin,
                GN * sizeof(uint16_t));
          }
        }
        std::fill_n(tile_output, GROUP_ELEMENTS, uint16_t(0x7fc1));
        int status = matmul(&mac, 0);
        require(status == 0, "matmul failed");
        check_written(tile_output, GROUP_ELEMENTS);

        for (int row = 0; row < GM; ++row) {
          for (int col = 0; col < GN; ++col) {
            uint32_t bits =
                uint32_t(tile_output[row * GN + col]) << 16;
            float value;
            std::memcpy(&value, &bits, sizeof(value));

            size_t index = size_t(m_begin + row) * N + n_begin + col;
            if (tile_k == 0)
              output[index] = value;
            else
              output[index] += value;
          }
        }
      }
    }

    std::printf("[native-reference] K-tile %d/4 complete\n", tile_k + 1);
    std::fflush(stdout);
  }

  pim_free(group_a, 0);
  pim_free(group_b, 0);
  pim_free(tile_output, 0);
}
