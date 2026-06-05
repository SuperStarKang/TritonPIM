// UPMEM dialect 최소 op들의 parse/print roundtrip 확인.
// RUN: triton-shared-opt %s | triton-shared-opt | FileCheck %s

// CHECK-LABEL: upmem.host_func "matmul_host"
upmem.host_func "matmul_host" {
  // CHECK: upmem.alloc_units {num_units = 2048 : i64, split_m = 1 : i64, split_n = 2048 : i64} : !upmem.dpu_set
  %units = upmem.alloc_units {num_units = 2048 : i64, split_m = 1 : i64, split_n = 2048 : i64} : !upmem.dpu_set
  // CHECK: upmem.load_program %{{.*}}, @matmul_device : !upmem.dpu_set
  upmem.load_program %units, @matmul_device : !upmem.dpu_set
}

// CHECK-LABEL: upmem.device_func "matmul_device"
upmem.device_func "matmul_device" {
  // CHECK: upmem.local_gemm {accumulation = "fp32", k = 256 : i64, m = 32 : i64, n = 64 : i64}
  upmem.local_gemm {m = 32 : i64, n = 64 : i64, k = 256 : i64, accumulation = "fp32"}
}
