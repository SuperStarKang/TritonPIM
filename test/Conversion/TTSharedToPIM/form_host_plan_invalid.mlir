// A host placement cannot depend on the result of device computation.
// RUN: triton-shared-opt --pim-form-host-plan -verify-diagnostics %s

func.func @device_dependent_placement(%A: memref<*xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  pim.device #pim.device_mapping<partitions = [1, 1, 1]> {
  ^bb0(%pm: index, %pn: index, %pk: index):
    %tile = tensor.empty() : tensor<8xi8>
    %length = tensor.dim %tile, %c0 : tensor<8xi8>
    // expected-error @below {{pim-form-host-plan: placement depends on a device compute result}}
    pim.place %A start[%c0] size[%length] allocation_shape[%length]
        strides[%c1] {access = 0 : i64, direction = "in"} : memref<*xi8>
  }
  return
}
