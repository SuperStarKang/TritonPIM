// PIM dialect 최소 op들의 parse/print roundtrip 확인.
// RUN: triton-shared-opt %s | triton-shared-opt | FileCheck %s

// CHECK-LABEL: func.func @matmul
func.func @matmul(%A: memref<256x512xf32>, %B: memref<512x512xf32>, %C: memref<256x512xf32>,
                  %M: i32, %N: i32, %K: i32) {
  // CHECK: pim.memory_region {{.*}} {layout = "row_major", role = "input_activation", space = "host"}
  %a = pim.memory_region %A {space = "host", layout = "row_major", role = "input_activation"}
      : (memref<256x512xf32>) -> !pim.region
  %b = pim.memory_region %B {space = "device_global", layout = "row_major", role = "weight"}
      : (memref<512x512xf32>) -> !pim.region
  %c = pim.memory_region %C {space = "host", layout = "row_major", role = "output"}
      : (memref<256x512xf32>) -> !pim.region

  // CHECK: pim.tile_plan {{.*}} : !pim.tile_plan
  %tp = pim.tile_plan {block_m = 32 : i64, block_n = 64 : i64, block_k = 256 : i64,
                       split_m = 1 : i64, split_n = 1 : i64, split_k = 1 : i64}
      : !pim.tile_plan

  // CHECK: pim.transfer {{.*}} {dst = "device_global", src = "host"} : !pim.region
  pim.transfer %a {src = "host", dst = "device_global"} : !pim.region
  pim.transfer %b {src = "host", dst = "device_global"} : !pim.region

  // CHECK: pim.compute(%{{.*}}, %{{.*}}, %{{.*}}) using %{{.*}} extent(%{{.*}}, %{{.*}}, %{{.*}}) {op = "gemm"}
  %partial = pim.compute(%a, %b, %c) using %tp extent(%M, %N, %K) {op = "gemm"}
      : (!pim.region, !pim.region, !pim.region, !pim.tile_plan, i32, i32, i32) -> !pim.region

  pim.transfer %c {src = "device_global", dst = "host"} : !pim.region

  // CHECK: pim.launch @matmul {device_id = 0 : i64, vendor = "upmem"}
  pim.launch @matmul {vendor = "upmem", device_id = 0 : i64}
  return
}
