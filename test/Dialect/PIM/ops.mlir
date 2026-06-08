// PIM dialect 최소 op들의 parse/print roundtrip 확인.
// RUN: triton-shared-opt %s | triton-shared-opt | FileCheck %s

// CHECK-LABEL: func.func @matmul
func.func @matmul(%A: memref<256x512xf32>, %B: memref<512x512xf32>, %C: memref<256x512xf32>) {
  // CHECK: pim.memory_region {{.*}} {layout = "row_major", role = "input_activation", space = "host"}
  %a = pim.memory_region %A {space = "host", layout = "row_major", role = "input_activation"}
      : (memref<256x512xf32>) -> !pim.region
  %b = pim.memory_region %B {space = "host", layout = "row_major", role = "weight"}
      : (memref<512x512xf32>) -> !pim.region
  %c = pim.memory_region %C {space = "host", layout = "row_major", role = "output"}
      : (memref<256x512xf32>) -> !pim.region

  // CHECK: pim.tile_plan {{.*}} : !pim.tile_plan
  %tp = pim.tile_plan {block_m = 64 : i64, block_n = 64 : i64, block_k = 32 : i64,
                       split_m = 1 : i64, split_n = 1 : i64, split_k = 1 : i64}
      : !pim.tile_plan

  // CHECK: pim.execution_group {{.*}} : !pim.exec_group
  %eg = pim.execution_group {mapping_policy = "none", num_units = 1 : i64,
                             local_lanes = 1 : i64, stationary_data = "B"}
      : !pim.exec_group

  // CHECK: pim.transfer {{.*}} {dst = "device_global", src = "host"} : !pim.region
  pim.transfer %a {src = "host", dst = "device_global"} : !pim.region
  pim.transfer %b {src = "host", dst = "device_global"} : !pim.region

  // faithful-body: pim.compute가 본문 region을 들고 있다. block arg는 유닛-로컬 슬라이스.
  // CHECK: pim.compute ins(%{{.*}}, %{{.*}}) outs(%{{.*}}) using %{{.*}}, %{{.*}} {primitive = "gemm"}
  // CHECK-SAME: : (!pim.region, !pim.region, !pim.region, !pim.exec_group, !pim.tile_plan)
  // CHECK: ^{{.*}}(%{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>):
  pim.compute ins(%a, %b) outs(%c) using %eg, %tp {primitive = "gemm"}
      : (!pim.region, !pim.region, !pim.region, !pim.exec_group, !pim.tile_plan) {
  ^unit(%la: memref<?x?xf32>, %lb: memref<?x?xf32>, %lc: memref<?x?xf32>):
  }

  pim.transfer %c {src = "device_global", dst = "host"} : !pim.region

  // CHECK: pim.launch @matmul {device_id = 0 : i64, vendor = "upmem"}
  pim.launch @matmul {vendor = "upmem", device_id = 0 : i64}
  return
}
