// 생성된 host.cpp의 matmul_kernel_host가 실제 DPU에서 정확히 gemm을 수행하는지 검증한다.
// A(M,K) x B(K,N) -> C(M,N), row-major. host.cpp가 B를 내부에서 Bt로 전치한다.
//
// 빌드/실행은 build_and_test.sh 참고.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// 생성된 host.cpp가 제공하는 진입점.
extern "C" void matmul_kernel_host(float *A, float *B, float *C);

int main() {
  // test/Target/UPMEM/matmul.mlir의 local_gemm{m=32,n=64,k=256}과 일치해야 한다.
  const int M = 32, N = 64, K = 256;

  std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f), ref(M * N, 0.0f);
  for (int i = 0; i < M * K; ++i) A[i] = (float)((i * 7 + 3) % 13) * 0.1f;
  for (int i = 0; i < K * N; ++i) B[i] = (float)((i * 5 + 1) % 11) * 0.1f;

  // 기준 CPU gemm
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int l = 0; l < K; ++l) acc += A[i * K + l] * B[l * N + j];
      ref[i * N + j] = acc;
    }

  matmul_kernel_host(A.data(), B.data(), C.data());

  int errors = 0;
  float max_diff = 0.0f;
  for (int i = 0; i < M * N; ++i) {
    float d = std::fabs(C[i] - ref[i]);
    if (d > max_diff) max_diff = d;
    if (d > 1e-3f) {
      if (errors < 5)
        fprintf(stderr, "mismatch at %d: dpu=%f ref=%f\n", i, C[i], ref[i]);
      ++errors;
    }
  }

  if (errors == 0) {
    printf("OK: DPU gemm matches CPU (M=%d N=%d K=%d, max_diff=%g)\n", M, N, K, max_diff);
    return 0;
  }
  fprintf(stderr, "FAIL: %d mismatches (max_diff=%g)\n", errors, max_diff);
  return 1;
}
