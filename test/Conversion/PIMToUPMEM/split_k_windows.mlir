// 하나의 semantic K domain을 reduction_k round와 device_k가 나눈 Common IR을
// UPMEM이 그대로 소비한다. source pim.axis="k"나 별도 spatial K round는 없다.
// RUN: triton-shared-opt --convert-pim-to-upmem=host-threads=8 %s | FileCheck %s --check-prefix=IR
// RUN: triton-shared-opt --convert-pim-to-upmem=host-threads=8 %s | triton-shared-translate --upmem-to-host | FileCheck %s --check-prefix=HOST
// RUN: triton-shared-opt --convert-pim-to-upmem=host-threads=8 %s | triton-shared-translate --upmem-to-device | FileCheck %s --check-prefix=DEVICE

// K는 output axis host variable로 다시 만들지 않는다.
// IR-NOT: tiles_k
// IR-NOT: tiles_reduction_k
// acquire는 Common reduction_k round 밖에 있고, 매 round 같은 p_k device set을
// stage/launch/gather에 사용한다.
// IR: %[[SET:.*]] = upmem.acquire 2, @matmul_split_k_device
// IR: scf.for %[[ROUND:.*]] =
// IR:   upmem.foreach_unit %[[SET]]
// IR-COUNT-1: upmem.launch %[[SET]]
// IR:   upmem.gather %[[SET]], @matmul_split_k_mram_2
// IR:   upmem.foreach_group
// reduction_k round와 foreach_group의 device_k로 첫 partial 여부를 정한다.
// IR:     %[[PARTIAL:.*]] = arith.addi %[[ROUND]], %{{.*}} : index
// IR:     upmem.unstage {{.*}} accumulate %[[PARTIAL]] tile [8, 8]
// IR-NOT: pim.

// HOST-NOT: matmul_split_k_tiles_k
// HOST-NOT: tiles_reduction_k
// HOST: if ({{.*}} == 0) {
// HOST: +=

// DEVICE-NOT: matmul_split_k_tiles_k
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
  %c32 = arith.constant 32 : index
  %zero = arith.constant 0 : i32

  // total_k_groups=4, tile_k=1, p_k=2이므로 reduction_k round는 2회다.
  %kTiles = arith.ceildivsi %c4, %c1 : index
  %rounds = arith.ceildivsi %kTiles, %c2 : index
  scf.for %reductionRound = %c0 to %rounds step %c1 {
    %roundBase = arith.muli %reductionRound, %c2 : index
    pim.device #pim.device_mapping<partitions = [1, 1, 2]> attributes {
        pim.execution_tile = #pim.execution_tile<groups = [1, 1, 1]>} {
    ^bb0(%pm: index, %pn: index, %pk: index):
      %tile = arith.addi %roundBase, %pk : index
      %firstKGroup = arith.muli %tile, %c1 : index
      %left = arith.subi %c4, %firstKGroup : index
      %nonnegative = arith.maxsi %left, %c0 : index
      %localKGroups = arith.minsi %c1, %nonnegative : index
      %kStart = arith.muli %firstKGroup, %c4 : index
      %kSize = arith.muli %localKGroups, %c4 : index

      pim.place %A start[%c0, %kStart] size[%c8, %kSize]
          allocation_shape[%c8, %c4] strides[%c32, %c1]
          {direction = "in", access = 0 : i64,
           replicated_over = ["n"]} : memref<*xi8>
      pim.place %B start[%kStart, %c0] size[%kSize, %c8]
          allocation_shape[%c4, %c8] strides[%c8, %c1]
          {direction = "in", access = 1 : i64,
           replicated_over = ["m"]} : memref<*xi8>
      pim.place %C start[%c0, %c0] size[%c8, %c8]
          allocation_shape[%c8, %c8] strides[%c8, %c1]
          {direction = "out", access = 2 : i64} : memref<*xi32>

      scf.for %mt = %c0 to %c1 step %c1 {
        %gm = arith.addi %c0, %mt {pim.device_origin} : index
        %m = arith.muli %gm, %c8 : index
        scf.for %nt = %c0 to %c1 step %c1 {
          %gn = arith.addi %c0, %nt {pim.device_origin} : index
          %n = arith.muli %gn, %c8 : index
          %empty = tensor.empty() : tensor<8x8xi32>
          %init = linalg.fill ins(%zero : i32)
              outs(%empty : tensor<8x8xi32>) -> tensor<8x8xi32>
          %result = scf.for %rk = %c0 to %localKGroups step %c1
              iter_args(%acc = %init) -> tensor<8x8xi32> {
            %globalKGroup = arith.addi %firstKGroup, %rk
                {pim.device_origin} : index
            %k = arith.muli %globalKGroup, %c4 : index
            %aRow = arith.muli %m, %c32 : index
            %aOffset = arith.addi %aRow, %k : index
            %aAccess = pim.access %A offset %aOffset sizes [8, 4]
                strides[%c32, %c1] origin[%m, %k] {id = 0 : i64}
                : memref<*xi8> -> !pim.access
            %a = pim.load %aAccess : !pim.access -> tensor<8x4xi8>
            %bRow = arith.muli %k, %c8 : index
            %bOffset = arith.addi %bRow, %n : index
            %bAccess = pim.access %B offset %bOffset sizes [4, 8]
                strides[%c8, %c1] origin[%k, %n] {id = 1 : i64}
                : memref<*xi8> -> !pim.access
            %b = pim.load %bAccess : !pim.access -> tensor<4x8xi8>
            %next = linalg.matmul
                ins(%a, %b : tensor<8x4xi8>, tensor<4x8xi8>)
                outs(%acc : tensor<8x8xi32>) -> tensor<8x8xi32>
            scf.yield %next : tensor<8x8xi32>
          } {pim.reduction = "add", pim.temporal_reduction}
          %cRow = arith.muli %m, %c8 : index
          %cOffset = arith.addi %cRow, %n : index
          %cAccess = pim.access %C offset %cOffset sizes [8, 8]
              strides[%c8, %c1] origin[%m, %n] {id = 2 : i64}
              : memref<*xi32> -> !pim.access
          pim.combine %result into %cAccess kind "add"
              : tensor<8x8xi32>, !pim.access
        } {pim.axis = #pim.axis<"n">}
      } {pim.axis = #pim.axis<"m">}
    }
  } {pim.scheduling_axis = "reduction_k"}
  return
}
}
