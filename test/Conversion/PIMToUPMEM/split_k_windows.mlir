// spatial K device partition과 그 plane 내부 temporal K window가 동시에 있어도
// 서로 다른 host variable을 사용해야 한다. 과거에는 둘 다 *_tiles_k를 만들어
// UPMEM target IR의 module symbol이 중복되었다.
// RUN: triton-shared-opt --convert-pim-to-upmem=host-threads=8 %s | FileCheck %s --check-prefix=IR
// RUN: triton-shared-opt --convert-pim-to-upmem=host-threads=8 %s | triton-shared-translate --upmem-to-host | FileCheck %s --check-prefix=HOST
// RUN: triton-shared-opt --convert-pim-to-upmem=host-threads=8 %s | triton-shared-translate --upmem-to-device | FileCheck %s --check-prefix=DEVICE

// IR-DAG: upmem.host_var @matmul_split_k_kernel_tiles_k : index
// IR-DAG: upmem.host_var @matmul_split_k_kernel_tiles_reduction_k : i32
// IR: upmem.get_host_var @matmul_split_k_kernel_tiles_k : index
// IR: upmem.get_host_var @matmul_split_k_kernel_tiles_reduction_k : i32
// IR: upmem.to_host_var {{.*}}, @matmul_split_k_kernel_tiles_k : index
// IR: upmem.to_host_var {{.*}}, @matmul_split_k_kernel_tiles_reduction_k : i32

// HOST: dpu_push_xfer({{.*}}, "matmul_split_k_kernel_tiles_k"
// HOST: dpu_push_xfer({{.*}}, "matmul_split_k_kernel_tiles_reduction_k"

// DEVICE-DAG: __host int32_t matmul_split_k_kernel_tiles_k;
// DEVICE-DAG: __host int32_t matmul_split_k_kernel_tiles_reduction_k;
// DEVICE: matmul_split_k_kernel_tiles_k
// DEVICE: matmul_split_k_kernel_tiles_reduction_k

module attributes {pim.target_profile = #pim.target_profile<target = "upmem", number_of_pim_devices = 2048, number_of_banks_per_pim_device = 1, number_of_pim_cores_per_bank = 1, number_of_parallel_lanes_per_pim_core = 16, parallel_execution_kind = "multithread", usable_bank_capacity_bytes = 62914560 : i64, usable_local_capacity_bytes = 49152 : i64>} {
  func.func @matmul_split_k_kernel(%arg0: memref<*xi8> {tt.divisibility = 16 : i32}, %arg1: memref<*xi8> {tt.divisibility = 16 : i32}, %arg2: memref<*xi32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {pim.block_group = #pim.block_group<tiles = [2, 2, 2], useful = [64, 64, 32]>, pim.execution_tile = #pim.execution_tile<m = 256, n = 512, k = 1792>, pim.intra_device_plan = #pim.intra_device_plan<bank_partitions = [1, 1, 1], core_partitions = [1, 1, 1], lanes = 16, lane_axis = "m">} {
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %c0_i8 = arith.constant 0 : i8
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c15_i32 = arith.constant 15 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %0 = arith.addi %arg4, %c31_i32 : i32
    %1 = arith.divsi %0, %c32_i32 : i32
    %2 = arith.ceildivsi %arg10, %1 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.index_cast %1 : i32 to index
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.ceildivsi %3, %c2 : index
    %7 = arith.ceildivsi %4, %c2 : index
    %8 = arith.ceildivsi %7, %c8 : index
    %9 = arith.ceildivsi %8, %c16 : index
    scf.for %arg13 = %c0 to %9 step %c1 {
      %10 = arith.ceildivsi %5, %c16 : index
      scf.for %arg14 = %c0 to %10 step %c1 {
        %11 = arith.ceildivsi %6, %c4 : index
        %12 = arith.ceildivsi %11, %c4 : index
        scf.for %arg15 = %c0 to %12 step %c1 {
          pim.device <axes = ["m", "n", "k"], partitions = [4, 16, 16]> extents[%6, %7, %5] attributes {pim.execution_tile = #pim.execution_tile<m = 256, n = 512, k = 1792>, pim.inter_device_plan = #pim.inter_device_plan<available_devices = 2048, axes = ["m", "n", "k"], active_grid = [4, 16, 16], execution_tile = [256, 512, 1792], scheduling_order = ["n", "k", "m"]>, pim.reduction_plan = #pim.reduction_plan<spatial_partials = "separate_output_planes", temporal_accumulation = "host", final_accumulation = "external">} {
          ^bb0(%arg16: index, %arg17: index, %arg18: index):
            %13 = arith.muli %arg15, %c4 : index
            %14 = arith.addi %13, %arg16 : index
            %15 = arith.muli %14, %c4 : index
            %16 = arith.subi %6, %15 : index
            %17 = arith.maxsi %16, %c0 : index
            %18 = arith.minsi %17, %c4 : index
            %19 = arith.muli %arg13, %c16 : index
            %20 = arith.addi %19, %arg17 : index
            %21 = arith.muli %20, %c8 : index
            %22 = arith.subi %7, %21 : index
            %23 = arith.maxsi %22, %c0 : index
            %24 = arith.minsi %23, %c8 : index
            %25 = arith.muli %arg14, %c16 : index
            %26 = arith.addi %25, %arg18 : index
            %27 = arith.subi %5, %26 : index
            %28 = arith.maxsi %27, %c0 : index
            %29 = arith.minsi %28, %c1 : index
            %30 = arith.divsi %arg5, %c16_i32 : i32
            %31 = arith.addi %30, %c15_i32 : i32
            %32 = arith.divsi %31, %c16_i32 : i32
            %33 = arith.ceildivsi %32, %c2_i32 : i32
            %34 = arith.index_cast %33 : i32 to index
            %35 = arith.index_cast %26 : index to i32
            %36 = arith.index_cast %30 : i32 to index
            %37 = arith.muli %14, %c8 : index
            %38 = arith.index_cast %37 : index to i32
            %39 = arith.muli %38, %c32_i32 : i32
            %40 = arith.index_cast %39 : i32 to index
            %41 = arith.index_cast %arg6 : i32 to index
            %42 = arith.muli %18, %c64 : index
            %43 = arith.index_cast %arg3 : i32 to index
            %44 = arith.subi %43, %40 : index
            %45 = arith.maxsi %44, %c0 : index
            %46 = arith.minsi %45, %42 : index
            %47 = arith.muli %34, %c32 : index
            %48 = arith.index_cast %arg5 : i32 to index
            %49 = arith.maxsi %48, %c0 : index
            %50 = arith.minsi %49, %47 : index
            pim.place %arg0 start[%26, %40, %c0] size[%29, %46, %50] capacity[%c1, %c256, %47] strides[%36, %41, %c1] {access = 0 : i64, direction = "in", mapping = #pim.place_mapping<axis_sources = ["k", "m", ""]>, replicated_over = ["n"]} : memref<*xi8>
            %51 = arith.muli %35, %30 : i32
            %52 = arith.index_cast %51 : i32 to index
            %53 = arith.index_cast %arg7 : i32 to index
            %54 = arith.muli %29, %34 : index
            %55 = arith.muli %54, %c32 : index
            %56 = arith.muli %34, %c32 : index
            %57 = arith.subi %48, %52 : index
            %58 = arith.maxsi %57, %c0 : index
            %59 = arith.minsi %58, %55 : index
            %60 = arith.muli %20, %c512 : index
            %61 = arith.muli %24, %c64 : index
            %62 = arith.index_cast %arg4 : i32 to index
            %63 = arith.subi %62, %60 : index
            %64 = arith.maxsi %63, %c0 : index
            %65 = arith.minsi %64, %61 : index
            pim.place %arg1 start[%52, %60] size[%59, %65] capacity[%56, %c512] strides[%53, %c1] {access = 1 : i64, direction = "in", mapping = #pim.place_mapping<axis_sources = ["k", "n"]>, replicated_over = ["m"]} : memref<*xi8>
            %66 = arith.index_cast %arg9 : i32 to index
            %67 = arith.index_cast %arg8 : i32 to index
            pim.place %arg2 start[%26, %40, %60] size[%29, %46, %65] capacity[%c1, %c256, %c512] strides[%66, %67, %c1] {access = 2 : i64, direction = "out", mapping = #pim.place_mapping<axis_sources = ["k", "m", "n"]>} : memref<*xi32>
            scf.for %arg19 = %c0 to %18 step %c1 {
              %68 = arith.addi %15, %arg19 {pim.device_origin} : index
              %69 = arith.muli %68, %c2 : index
              scf.for %arg20 = %c0 to %24 step %c1 {
                %70 = arith.addi %21, %arg20 {pim.device_origin} : index
                %71 = arith.muli %70, %c2 : index
                scf.for %arg21 = %c0 to %29 step %c1 {
                  %72 = arith.addi %26, %arg21 {pim.device_origin} : index
                  %73 = arith.index_cast %69 : index to i32
                  %74 = arith.index_cast %71 : index to i32
                  %75 = arith.index_cast %72 : index to i32
                  %76 = arith.muli %73, %c32_i32 : i32
                  %77 = arith.index_cast %76 : i32 to index
                  %78 = arith.muli %74, %c32_i32 : i32
                  %79 = arith.index_cast %78 : i32 to index
                  %80 = arith.muli %75, %30 : i32
                  %81 = arith.index_cast %80 : i32 to index
                  %82 = arith.muli %77, %41 : index
                  %83 = arith.muli %81, %53 : index
                  %84 = arith.muli %arg7, %c16_i32 : i32
                  %85 = arith.index_cast %84 : i32 to index
                  %86 = arith.addi %77, %c32 : index
                  %87 = arith.minsi %86, %43 : index
                  %88 = arith.maxsi %87, %77 : index
                  %89 = arith.subi %88, %77 : index
                  %90 = arith.addi %81, %c16 : index
                  %91 = arith.minsi %89, %c32 : index
                  %92 = arith.addi %79, %c32 : index
                  %93 = arith.minsi %92, %62 : index
                  %94 = arith.maxsi %93, %79 : index
                  %95 = arith.subi %94, %79 : index
                  %96 = arith.minsi %95, %c32 : index
                  %97 = tensor.empty() : tensor<64x64xi32>
                  %98 = linalg.fill ins(%c0_i32 : i32) outs(%97 : tensor<64x64xi32>) -> tensor<64x64xi32>
                  %99 = scf.for %arg22 = %c0_i32 to %33 step %c1_i32 iter_args(%arg23 = %98) -> (tensor<64x64xi32>)  : i32 {
                    %107 = arith.muli %arg22, %c2_i32 : i32
                    %108 = arith.index_cast %107 : i32 to index
                    %109 = arith.muli %108, %85 : index
                    %110 = arith.addi %83, %109 : index
                    %111 = arith.muli %108, %c16 : index
                    %112 = arith.addi %82, %111 : index
                    %113 = arith.addi %110, %79 : index
                    %114 = arith.addi %112, %81 : index
                    %115 = arith.muli %arg22, %c32_i32 : i32
                    %116 = arith.index_cast %115 : i32 to index
                    %117 = arith.addi %81, %116 : index
                    %118 = arith.addi %90, %116 : index
                    %119 = arith.minsi %118, %48 : index
                    %120 = arith.maxsi %119, %117 : index
                    %121 = arith.subi %120, %117 : index
                    %122 = arith.minsi %121, %c16 : index
                    %123 = pim.access %arg0 offset %114 sizes [1, 64, 32] strides[%36, %41, %c1] origin[%72, %77, %111] valid[%c1, %91, %122] pad %c0_i8 : i8 {id = 0 : i64} : memref<*xi8> -> !pim.access
                    %124 = pim.load %123 : !pim.access -> tensor<64x32xi8>
                    %125 = arith.muli %70, %c64 : index
                    %126 = arith.addi %111, %81 : index
                    %127 = pim.access %arg1 offset %113 sizes [32, 64] strides[%53, %c1] origin[%126, %125] valid[%122, %96] pad %c0_i8 : i8 {id = 1 : i64} : memref<*xi8> -> !pim.access
                    %128 = pim.load %127 : !pim.access -> tensor<32x64xi8>
                    %129 = linalg.matmul ins(%124, %128 : tensor<64x32xi8>, tensor<32x64xi8>) outs(%arg23 : tensor<64x64xi32>) -> tensor<64x64xi32>
                    scf.yield %129 : tensor<64x64xi32>
                  }
                  %100 = arith.muli %75, %arg9 : i32
                  %101 = arith.index_cast %100 : i32 to index
                  %102 = arith.muli %77, %67 : index
                  %103 = arith.addi %101, %102 : index
                  %104 = arith.addi %103, %79 : index
                  %105 = arith.muli %70, %c64 : index
                  %106 = pim.access %arg2 offset %104 sizes [1, 64, 64] strides[%66, %67, %c1] origin[%72, %77, %105] valid[%c1, %91, %96] {id = 2 : i64} : memref<*xi32> -> !pim.access
                  pim.store %99, %106 : tensor<64x64xi32>, !pim.access
                } {pim.axis = #pim.axis<"k">}
              } {pim.axis = #pim.axis<"n">}
            } {pim.axis = #pim.axis<"m">}
          }
        } {pim.scheduling_axis = "m"}
      } {pim.scheduling_axis = "k"}
    } {pim.scheduling_axis = "n"}
    return
  }
}



