// Original libpim matmul with the 4096-element B/C row layout used by ubench.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <vector>

#include <pim.h>

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "[native-gemv] %s\n", message);
    std::exit(EXIT_FAILURE);
  }
}

static uint16_t *allocate_words(size_t count) {
  auto *buffer = static_cast<uint16_t *>(pim_malloc(count * sizeof(uint16_t), 0));
  require(buffer != nullptr && buffer != MAP_FAILED, "PIM allocation failed");
  return buffer;
}

// A[128,256] and x[256] contain BF16 words; output[128] contains FP32.
// Run once in a fresh process, independently of the Triton resident runtime.
extern "C" void aimx_native_gemv_pitch(
    const uint16_t *input_a, const uint16_t *input_x, float *output) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  require(input_a && input_x && output, "null host buffer");
  require(init_mpim_drv() > 0, "FPGA 0 is unavailable");

  constexpr int M = 128;
  constexpr int K = 256;
  constexpr int TILE_M = 32;
  constexpr int ROW_WIDTH = 4096;
  std::vector<uint16_t> staged_a(TILE_M * ROW_WIDTH, 0);
  std::vector<uint16_t> staged_b(K * ROW_WIDTH, 0);
  std::vector<uint16_t> staged_c(TILE_M * ROW_WIDTH);
  uint16_t *device_a = allocate_words(staged_a.size());
  uint16_t *device_b = allocate_words(staged_b.size());
  uint16_t *device_c = allocate_words(staged_c.size());

  for (int k = 0; k < K; ++k)
    staged_b[k * ROW_WIDTH] = input_x[k];

  pim_args mac{};
  mac.srcA_va = reinterpret_cast<uint64_t>(device_a);
  mac.srcB_va = reinterpret_cast<uint64_t>(device_b);
  mac.dstC_va = reinterpret_cast<uint64_t>(device_c);
  mac.p_size = TILE_M;
  mac.q_size = K;
  mac.r_size = ROW_WIDTH;

  for (int m_start = 0; m_start < M; m_start += TILE_M) {
    for (int row = 0; row < TILE_M; ++row)
      std::memcpy(staged_a.data() + row * ROW_WIDTH,
                  input_a + (m_start + row) * K, K * sizeof(uint16_t));
    std::fill(staged_c.begin(), staged_c.end(), uint16_t(0x7fc1));
    require(datacopy_cpu2pim(device_a, staged_a.data(),
                            staged_a.size() * sizeof(uint16_t), 0) >= 0,
            "A upload failed");
    require(datacopy_cpu2pim(device_b, staged_b.data(),
                            staged_b.size() * sizeof(uint16_t), 0) >= 0,
            "B upload failed");
    require(datacopy_cpu2pim(device_c, staged_c.data(),
                            staged_c.size() * sizeof(uint16_t), 0) >= 0,
            "C initialization failed");
    require(matmul(&mac, 0) == 0, "original matmul failed");
    require(datacopy_pim2cpu(staged_c.data(), device_c,
                            staged_c.size() * sizeof(uint16_t), 0) >= 0,
            "C download failed");

    for (int row = 0; row < TILE_M; ++row) {
      uint32_t bits = uint32_t(staged_c[row * ROW_WIDTH]) << 16;
      std::memcpy(output + m_start + row, &bits, sizeof(bits));
    }
  }

  pim_free(device_c, 0);
  pim_free(device_b, 0);
  pim_free(device_a, 0);
}
