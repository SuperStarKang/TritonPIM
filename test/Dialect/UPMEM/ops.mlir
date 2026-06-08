// UPMEM dialect 최소 op들의 parse/print roundtrip 확인.
// RUN: triton-shared-opt %s | triton-shared-opt | FileCheck %s

// CHECK-LABEL: upmem.host_func "matmul_host"
upmem.host_func "matmul_host" {
  // CHECK: upmem.alloc_units {num_units = 2048 : i64, split_k = 1 : i64, split_m = 1 : i64, split_n = 2048 : i64} : !upmem.dpu_set
  %units = upmem.alloc_units {num_units = 2048 : i64, split_m = 1 : i64, split_n = 2048 : i64, split_k = 1 : i64} : !upmem.dpu_set
  // CHECK: upmem.load_program %{{.*}}, @matmul_device : !upmem.dpu_set
  upmem.load_program %units, @matmul_device : !upmem.dpu_set
}

// device_func는 faithful-body region: block arg = 유닛-로컬 슬라이스(A/B/C). 본문은 codegen 입력.
// CHECK-LABEL: upmem.device_func "matmul_device"
// CHECK: ^{{.*}}(%{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>):
upmem.device_func "matmul_device" {
^unit(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>):
}
