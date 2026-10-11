// Previously captured TTShared input from pim_backend/examples/gemm.py:
// BLOCK_M/N/K=16, FP16 inputs, FP32 accumulator, dynamic M/N/K masks.
// This fixture is not evidence of a fresh build or hardware run.
// Follow the backend's Common pipeline, without editing any Common pass.
// RUN: triton-shared-opt --pim-normalize-address -cse -loop-invariant-code-motion -canonicalize --pim-form-compute --convert-ttshared-to-pim --pim-build-logical-grid --pim-recover-access-origin --pim-set-target-profile='target=aimx num-units=1 banks-per-unit=256 cores-per-bank=1 lanes-per-core=16 bank-capacity-limit-bytes=268435456 local-capacity-limit-bytes=0' --pim-form-block-groups --pim-form-execution-tiles --pim-map-devices='p-m=1 p-n=1 p-k=1' --pim-place-data --pim-verify -cse %s -o %t.common
// Serialize exactly as the AiMX backend does, retaining no_inline without the
// pinned SCF custom printer's duplicate keyword/attribute representation.
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx -cse -canonicalize --mlir-print-op-generic %t.common -o %t.aimx
// RUN: FileCheck %s --check-prefix=SERIALIZED --input-file=%t.aimx --implicit-check-not='scf.execute_region no_inline'
// Reparse and canonicalize before display to check that the boundary survives.
// RUN: triton-shared-opt -canonicalize %t.aimx | FileCheck %s --check-prefix=IR --implicit-check-not=aimx.compute_template --implicit-check-not=linalg. --implicit-check-not='pim.access ' --implicit-check-not='pim.device '
// RUN: triton-shared-translate --aimx-to-runtime-stub %t.aimx | FileCheck %s --check-prefix=STUB
// RUN: triton-shared-translate --aimx-to-descriptors %t.aimx | FileCheck %s --check-prefix=DESC
// Multiple M/K groups retain the same dynamic masks and Common traversal.
// RUN: triton-shared-opt --pim-normalize-address -cse -loop-invariant-code-motion -canonicalize --pim-form-compute --convert-ttshared-to-pim --pim-build-logical-grid --pim-recover-access-origin --pim-set-target-profile='target=aimx num-units=1 banks-per-unit=256 cores-per-bank=1 lanes-per-core=16 bank-capacity-limit-bytes=268435456 local-capacity-limit-bytes=0' --pim-form-block-groups --pim-form-execution-tiles='tile-m=16 tile-n=1 tile-k=16' --pim-map-devices='p-m=1 p-n=1 p-k=1' --pim-place-data --pim-verify -cse %s -o %t.multi.common
// RUN: triton-shared-opt --pim-form-host-plan --convert-pim-to-aimx -cse -canonicalize --mlir-print-op-generic %t.multi.common -o %t.multi.aimx
// RUN: triton-shared-opt -canonicalize %t.multi.aimx | FileCheck %s --check-prefix=IR --implicit-check-not=linalg. --implicit-check-not='pim.access ' --implicit-check-not='pim.device '
// RUN: triton-shared-translate --aimx-to-runtime-stub %t.multi.aimx | FileCheck %s --check-prefix=STUB
// RUN: triton-shared-translate --aimx-to-descriptors %t.multi.aimx | FileCheck %s --check-prefix=DESC
// SERIALIZED: "scf.execute_region"() <{no_inline}> ({
// SERIALIZED: }) : () -> ()
// IR: aimx.descriptor_program @gemm_kernel_device0_matmul_
// IR: func.func @gemm_kernel
// IR: aimx.foreach_device {{.*}} counts [1, 1, 1]
// IR: scf.execute_region no_inline
// IR: aimx.build_descriptors @gemm_kernel_device0_matmul_
// IR: aimx.stage
// IR: aimx.invoke
// IR: aimx.unstage
// STUB: extern "C" void gemm_kernel
// STUB: aimx_resident_acquire
// STUB: aimx_prog_gemm_kernel_device0_matmul_
// STUB: descriptor upload failed
// STUB: {  // aimx.stage
// STUB: aimx_f16_bf16.t[
// STUB: pim_exec
// STUB: aimx_from_bf16
// DESC: static uint32_t aimx_prog_gemm_kernel_device0_matmul_

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @gemm_kernel(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c15_i32 = arith.constant 15 : i32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = arith.muli %arg12, %c16_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.muli %arg13, %c16_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.index_cast %arg6 : i32 to index
    %7 = arith.muli %3, %6 : index
    %8 = arith.index_cast %arg7 : i32 to index
    %9 = arith.addi %arg5, %c15_i32 : i32
    %10 = arith.divsi %9, %c16_i32 : i32
    %11 = arith.muli %arg7, %c16_i32 : i32
    %12 = arith.index_cast %11 : i32 to index
    %13:3 = scf.for %arg15 = %c0_i32 to %10 step %c1_i32 iter_args(%arg16 = %7, %arg17 = %c0, %arg18 = %1) -> (index, index, tensor<16x16xf32>)  : i32 {
      %29 = arith.addi %arg17, %5 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [16, 16], strides: [%8, 1] : memref<*xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%arg16], sizes: [16, 16], strides: [%6, 1] : memref<*xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
      %30 = arith.muli %arg15, %c16_i32 : i32
      %31 = arith.subi %arg5, %30 : i32
      %32 = arith.addi %3, %c16 : index
      %33 = arith.index_cast %arg3 : i32 to index
      %34 = arith.minsi %32, %33 : index
      %35 = arith.maxsi %34, %3 : index
      %36 = arith.subi %35, %3 : index
      %37 = arith.index_cast %31 : i32 to index
      %38 = arith.minsi %37, %c16 : index
      %39 = arith.maxsi %38, %c0 : index
      %40 = arith.minsi %36, %c16 : index
      %41 = arith.minsi %39, %c16 : index
      %alloc = memref.alloc() : memref<16x16xf16>
      %42 = arith.cmpi slt, %40, %c16 : index
      %43 = arith.cmpi slt, %41, %c16 : index
      %44 = arith.ori %42, %43 : i1
      scf.if %44 {
        linalg.fill ins(%cst_0 : f16) outs(%alloc : memref<16x16xf16>)
      }
      %subview_3 = memref.subview %reinterpret_cast_2[0, 0] [%40, %41] [1, 1] : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_4 = memref.subview %alloc[0, 0] [%40, %41] [1, 1] : memref<16x16xf16> to memref<?x?xf16, strided<[16, 1]>>
      memref.copy %subview_3, %subview_4 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[16, 1]>>
      %45 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf16> to tensor<16x16xf16>
      %46 = arith.addi %5, %c16 : index
      %47 = arith.index_cast %arg4 : i32 to index
      %48 = arith.minsi %46, %47 : index
      %49 = arith.maxsi %48, %5 : index
      %50 = arith.subi %49, %5 : index
      %51 = arith.minsi %50, %c16 : index
      %alloc_5 = memref.alloc() : memref<16x16xf16>
      %52 = arith.cmpi slt, %51, %c16 : index
      %53 = arith.ori %43, %52 : i1
      scf.if %53 {
        linalg.fill ins(%cst_0 : f16) outs(%alloc_5 : memref<16x16xf16>)
      }
      %subview_6 = memref.subview %reinterpret_cast_1[0, 0] [%41, %51] [1, 1] : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %subview_7 = memref.subview %alloc_5[0, 0] [%41, %51] [1, 1] : memref<16x16xf16> to memref<?x?xf16, strided<[16, 1]>>
      memref.copy %subview_6, %subview_7 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[16, 1]>>
      %54 = bufferization.to_tensor %alloc_5 restrict writable : memref<16x16xf16> to tensor<16x16xf16>
      %55 = linalg.matmul ins(%45, %54 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %56 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg18, %55 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%arg18 : tensor<16x16xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %59 = arith.addf %in, %in_8 : f32
        linalg.yield %59 : f32
      } -> tensor<16x16xf32>
      %57 = arith.addi %arg16, %c16 : index
      %58 = arith.addi %arg17, %12 : index
      scf.yield %57, %58, %56 : index, index, tensor<16x16xf32>
    }
    %14 = arith.index_cast %arg8 : i32 to index
    %15 = arith.muli %3, %14 : index
    %16 = arith.addi %15, %5 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [16, 16], strides: [%14, 1] : memref<*xf32> to memref<16x16xf32, strided<[?, 1], offset: ?>>
    %17 = arith.addi %3, %c16 : index
    %18 = arith.index_cast %arg3 : i32 to index
    %19 = arith.minsi %17, %18 : index
    %20 = arith.maxsi %19, %3 : index
    %21 = arith.subi %20, %3 : index
    %22 = arith.addi %5, %c16 : index
    %23 = arith.index_cast %arg4 : i32 to index
    %24 = arith.minsi %22, %23 : index
    %25 = arith.maxsi %24, %5 : index
    %26 = arith.subi %25, %5 : index
    %27 = arith.minsi %21, %c16 : index
    %28 = arith.minsi %26, %c16 : index
    %extracted_slice = tensor.extract_slice %13#2[0, 0] [%27, %28] [1, 1] : tensor<16x16xf32> to tensor<?x?xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%27, %28] [1, 1] : memref<16x16xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}
