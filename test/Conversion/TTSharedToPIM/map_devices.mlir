// device 축을 고르는 순서와 자르는 눈금을 확인한다. 축마다 복제되는 피연산자가 다르고
// 하드웨어가 소비하는 단위도 다르므로, 둘 다 target 이 정하고 이 pass 는 받기만 한다.
// RUN: triton-shared-opt --pim-map-devices %s | FileCheck %s --check-prefix=BALANCED
// RUN: triton-shared-opt --pim-map-devices=device-axis-priority=output_column_tile,output_row_tile %s | FileCheck %s --check-prefix=COLUMN
// RUN: triton-shared-opt "--pim-map-devices=device-axis-priority=output_column_tile,output_row_tile device-axis-granularity=output_column_tile:4096" %s | FileCheck %s --check-prefix=GRAN

// 기본값은 지금까지의 순서다 — logical 축 순서 그대로 큰 인수가 앞에 간다.
// BALANCED: partitions = [2, 1]
// 눈금이 없으면 tile 단위 균등 분할이라 자를 필요가 없다.
// BALANCED-NOT: arith.maxsi

// AiMX 는 열부터 쪼갠다. 행을 쪼개면 B 가 device 수만큼 복제되기 때문이다.
// COLUMN: partitions = [1, 2]
// COLUMN-NOT: arith.maxsi

// 눈금을 주면 tile 이 아니라 group 단위로 나눈다. tile 하나가 16 원소이므로
// 4096 원소 = 256 tile 이 단위다. 마지막 group 이 부분일 수 있어 extent 로 자른다.
// GRAN: partitions = [1, 2]
// GRAN: arith.constant 256 : index
// GRAN: arith.ceildivsi
// GRAN: arith.maxsi
// GRAN: arith.minsi

module attributes {pim.target_profile = #pim.target_profile<
    target = "aimx", number_of_pim_devices = 2, pim_core_shape = [1],
    number_of_parallel_lanes_per_pim_core = 4096,
    parallel_execution_kind = "simd",
    requires_uniform_instruction_sequence = true,
    usable_bank_capacity_bytes = 1073741824, usable_local_capacity_bytes = 0>} {

// tile 하나가 몇 원소인지는 pim.access 의 sizes 에서 읽는다 — 이 커널은 16 이다.
func.func @two_axes(%C: memref<*xf32>, %tx: index, %ty: index) {
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
    } {pim.axis = "output_column_tile"}
  } {pim.axis = "output_row_tile"}
  return
}
}
