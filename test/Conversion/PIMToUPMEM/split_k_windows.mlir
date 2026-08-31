// Spatial K device plane과 그 plane 내부 Common temporal K Execution Tile이
// 동시에 있을 때도 UPMEM은 Common IR을 그대로 소비한다. target이 별도의
// *_tiles_reduction_k window를 만들거나 place를 다시 자르면 안 된다.
// RUN: triton-shared-opt --convert-pim-to-upmem=host-threads=8 %s | FileCheck %s --check-prefix=IR
// RUN: triton-shared-opt --convert-pim-to-upmem=host-threads=8 %s | triton-shared-translate --upmem-to-host | FileCheck %s --check-prefix=HOST
// RUN: triton-shared-opt --convert-pim-to-upmem=host-threads=8 %s | triton-shared-translate --upmem-to-device | FileCheck %s --check-prefix=DEVICE

// Spatial pim.axis="k"의 host variable만 존재한다.
// IR-DAG: upmem.host_var @matmul_split_k_tiles_k : index
// IR-NOT: tiles_reduction_k
// acquire는 두 Common scheduling loop 밖에 있다. accumulate는 spatial round,
// temporal round, foreach_group의 device_k를 모두 사용한다.
// IR: %[[SET:.*]] = upmem.acquire 2, @matmul_split_k_device
// IR: scf.for %[[SPATIAL:.*]] =
// IR:   scf.for %[[RED:.*]] =
// IR:     upmem.foreach_unit %[[SET]]
// IR:       upmem.stage
// IR-COUNT-1: upmem.launch %[[SET]]
// IR:     %[[ROUND_PARTIAL:.*]] = arith.addi %[[RED]], %[[SPATIAL]] : index
// IR:     upmem.gather %[[SET]], @matmul_split_k_mram_2 {{.*}} -> memref<2x1x8x8xi32>
// IR:       %[[PARTIAL:.*]] = arith.addi %[[ROUND_PARTIAL]], %{{.*}} : index
// IR:       %[[PLANE_ZERO:.*]] = arith.constant 0 : index
// IR:       upmem.unstage {{.*}} origin[%[[PLANE_ZERO]], {{.*}}] {{.*}} accumulate %[[PARTIAL]] tile [1, 8, 8]
// IR-NOT: pim.

// HOST: dpu_push_xfer({{.*}}, "matmul_split_k_tiles_k"
// HOST-NOT: tiles_reduction_k
// HOST: if ({{.*}} == 0) {
// HOST: +=

// DEVICE-DAG: __host int32_t matmul_split_k_tiles_k;
// DEVICE-NOT: tiles_reduction_k

module attributes {pim.target_profile = #pim.target_profile<
    target = "upmem", num_units = 2,
    banks_per_unit = 1, cores_per_bank = 1, lanes_per_core = 1,
    bank_capacity_limit_bytes = 4096,
    local_capacity_limit_bytes = 49152>} {

func.func @matmul_split_k(%A: memref<*xi8>, %B: memref<*xi8>,
                          %C: memref<*xi32>) attributes {
    pim.block = #pim.block<shape = [8, 8, 4]>,
    pim.block_group = #pim.block_group<blocks = [1, 1, 1]>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %zero = arith.constant 0 : i32

  // spatial plane scheduling과 temporal reduction scheduling은 Common IR이
  // 별개의 loop로 이미 확정한다.
  scf.for %spatialRound = %c0 to %c1 step %c1 {
    scf.for %reductionRound = %c0 to %c2 step %c1 {
      pim.device #pim.device_mapping<partitions = [1, 1, 2]> attributes {
          pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
      ^bb0(%pm: index, %pn: index, %pk: index):
        %kStart = arith.muli %reductionRound, %c4 : index

        // 각 pim.place는 이 launch에 resident인 Execution Tile 하나만 기술한다.
        pim.place %A start[%pk, %c0, %kStart]
            size[%c1, %c8, %c4] allocation_shape[%c1, %c8, %c4]
            strides[%c64, %c8, %c1]
            {direction = "in", access = 0 : i64,
             replicated_over = ["n"]} : memref<*xi8>
        pim.place %B start[%pk, %kStart, %c0]
            size[%c1, %c4, %c8] allocation_shape[%c1, %c4, %c8]
            strides[%c64, %c8, %c1]
            {direction = "in", access = 1 : i64,
             replicated_over = ["m"]} : memref<*xi8>
        pim.place %C start[%pk, %c0, %c0]
            size[%c1, %c8, %c8] allocation_shape[%c1, %c8, %c8]
            strides[%c64, %c8, %c1]
            {direction = "out", access = 2 : i64} : memref<*xi32>

        scf.for %mt = %c0 to %c1 step %c1 {
          %gm = arith.addi %c0, %mt {pim.device_origin} : index
          %m = arith.muli %gm, %c8 : index
          scf.for %nt = %c0 to %c1 step %c1 {
            %gn = arith.addi %c0, %nt {pim.device_origin} : index
            %n = arith.muli %gn, %c8 : index
            scf.for %kt = %c0 to %c1 step %c1 {
              %plane = arith.addi %pk, %kt {pim.device_origin} : index
              %empty = tensor.empty() : tensor<8x8xi32>
              %init = linalg.fill ins(%zero : i32)
                  outs(%empty : tensor<8x8xi32>) -> tensor<8x8xi32>
              %result = scf.for %rk = %c0 to %c1 step %c1
                  iter_args(%acc = %init) -> tensor<8x8xi32> {
                %globalRk = arith.addi %reductionRound, %rk
                    {pim.device_origin} : index
                %k = arith.muli %globalRk, %c4 : index
                %aAccess = pim.access %A offset %m sizes [1, 8, 4]
                    strides[%c64, %c8, %c1] origin[%plane, %m, %k]
                    {id = 0 : i64} : memref<*xi8> -> !pim.access
                %a = pim.load %aAccess : !pim.access -> tensor<8x4xi8>
                %bAccess = pim.access %B offset %n sizes [1, 4, 8]
                    strides[%c64, %c8, %c1] origin[%plane, %k, %n]
                    {id = 1 : i64} : memref<*xi8> -> !pim.access
                %b = pim.load %bAccess : !pim.access -> tensor<4x8xi8>
                %next = linalg.matmul
                    ins(%a, %b : tensor<8x4xi8>, tensor<4x8xi8>)
                    outs(%acc : tensor<8x8xi32>) -> tensor<8x8xi32>
                scf.yield %next : tensor<8x8xi32>
              } {pim.temporal_reduction}
              %cAccess = pim.access %C offset %n sizes [1, 8, 8]
                  strides[%c64, %c8, %c1] origin[%plane, %m, %n]
                  {id = 2 : i64} : memref<*xi32> -> !pim.access
              pim.store %result, %cAccess {pim.combine = "add"}
                  : tensor<8x8xi32>, !pim.access
            } {pim.axis = #pim.axis<"k">}
          } {pim.axis = #pim.axis<"n">}
        } {pim.axis = #pim.axis<"m">}
      }
    } {pim.scheduling_axis = "reduction_k"}
  } {pim.scheduling_axis = "k"}
  return
}
}
