// linalg.matmul은 치환하지 않고 그대로 두되 pim.class = "gemm" 태그만 붙는지 확인.
// (이 입력엔 메모리 접근이 없으므로 pim.access/load/store는 생기지 않는다.)
// RUN: triton-shared-opt --convert-ttshared-to-pim %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_kernel
func.func @matmul_kernel(%A: memref<*xf32>, %B: memref<*xf32>, %C: memref<*xf32>,
                         %M: i32, %N: i32, %K: i32) {
  %a = tensor.empty() : tensor<32x256xf32>
  %b = tensor.empty() : tensor<256x64xf32>
  %z = tensor.empty() : tensor<32x64xf32>
  %r = linalg.matmul
       ins(%a, %b : tensor<32x256xf32>, tensor<256x64xf32>)
       outs(%z : tensor<32x64xf32>) -> tensor<32x64xf32>
  return
}

// named op은 유지되고 종류 태그만 단다.
// CHECK:     linalg.matmul
// CHECK-SAME: pim.class = "gemm"
// CHECK-NOT: pim.access
