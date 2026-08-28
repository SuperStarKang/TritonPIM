// device 축을 고르는 순서와 자르는 눈금을 확인한다. 축마다 복제되는 피연산자가 다르고
// 하드웨어가 소비하는 단위도 다르므로, 둘 다 target 이 정하고 이 pass 는 받기만 한다.
// RUN: triton-shared-opt --pim-map-devices %s | FileCheck %s --check-prefix=BALANCED
// RUN: triton-shared-opt --pim-map-devices=device-axis-priority=n,m %s | FileCheck %s --check-prefix=COLUMN
// RUN: triton-shared-opt "--pim-map-devices=device-axis-priority=n,m device-axis-granularity=n:4096" %s | FileCheck %s --check-prefix=GRAN
// RUN: triton-shared-opt "--pim-map-devices=device-axis-partitions=k:2" %s | FileCheck %s --check-prefix=SPLITK

// 기본값은 지금까지의 순서다 — logical 축 순서 그대로 큰 인수가 앞에 간다.
// BALANCED: partitions = [2, 1]
// 눈금이 없으면 tile 단위 균등 분할이라 자를 필요가 없다.
// BALANCED: arith.maxsi

// AiMX 는 열부터 쪼갠다. 행을 쪼개면 B 가 device 수만큼 복제되기 때문이다.
// COLUMN-DAG: partitions = [1, 2]
// COLUMN-DAG: arith.constant 256 : index
// COLUMN-DAG: arith.ceildivsi
// COLUMN-DAG: arith.maxsi

// 눈금을 주면 tile 이 아니라 group 단위로 나눈다. tile 하나가 16 원소이므로
// 4096 원소 = 256 tile 이 단위다. 마지막 group 이 부분일 수 있어 extent 로 자른다.
// GRAN-DAG: partitions = [1, 2]
// GRAN-DAG: arith.constant 256 : index
// GRAN-DAG: arith.ceildivsi
// GRAN-DAG: arith.maxsi
// GRAN-DAG: arith.minsi
// GRAN-DAG: pim.device
// GRAN-DAG: pim.execution_tile = #pim.execution_tile<groups = [1, 256, 0]>

// spatial K loop의 각 좌표가 per-device partial Execution Tile 하나다. device 좌표마다
// 서로 다른 K Tile index를 만들므로 같은 M/N 출력 좌표의 두 partial을 동시에 배치한다.
// SPLITK-LABEL: func.func @split_k_execution_tiles
// SPLITK: partitions = [2]
// SPLITK: arith.muli
// SPLITK: arith.addi
// SPLITK: arith.muli

module attributes {pim.target_profile = #pim.target_profile<
    target = "aimx", number_of_pim_devices = 2,
    number_of_banks_per_pim_device = 256,
    number_of_pim_cores_per_bank = 1,
    number_of_parallel_lanes_per_pim_core = 16,
    parallel_execution_kind = "simd",
    usable_bank_capacity_bytes = 268435456, usable_local_capacity_bytes = 0>} {

// Group 하나의 원소 수는 #pim.block × #pim.block_group에서 파생한다 — 이
// 커널은 16이다. pim.access는 metadata와 실제 access의 정합성을 verifier가
// 확인할 때만 사용한다.
func.func @two_axes(%C: memref<*xf32>, %tx: index, %ty: index)
    attributes {
      pim.block = #pim.block<shape = [16, 16, 0]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
      pim.execution_tile = #pim.execution_tile<groups = [1, 256, 0]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %ldc = arith.constant 4096 : index
  scf.for %x = %c0 to %tx step %c1 {
    %m = arith.muli %x, %c16 : index
    scf.for %y = %c0 to %ty step %c1 {
      %n = arith.muli %y, %c16 : index
      %d = pim.access %C offset %n sizes [16, 16] strides[%ldc, %c1] origin[%m, %n]
           {id = 0 : i64} : memref<*xf32> -> !pim.access
    } {pim.axis = #pim.axis<"n">}
  } {pim.axis = #pim.axis<"m">}
  return
}

func.func @split_k_execution_tiles(%A: memref<*xf32>, %tk: index)
    attributes {
      pim.block = #pim.block<shape = [0, 0, 16]>,
      pim.block_group = #pim.block_group<blocks = [1, 1, 1]>,
      pim.execution_tile = #pim.execution_tile<groups = [0, 0, 0]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %gk = %c0 to %tk step %c1 {
    %k = arith.muli %gk, %c16 : index
    %a = pim.access %A offset %k sizes [16] strides[%c1] origin[%k]
         {id = 0 : i64} : memref<*xf32> -> !pim.access
  } {pim.axis = #pim.axis<"k">}
  return
}
}
