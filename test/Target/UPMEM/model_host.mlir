// Test the generated packing entry with the real model runtime and a mock SDK.
// The mock device adds two vectors; this tests transport, not GEMM arithmetic.
// RUN: triton-shared-translate --upmem-to-model-host %s > %t.cpp
// RUN: FileCheck %s --implicit-check-not=malloc --implicit-check-not='free(' < %t.cpp
// RUN: %python %S/Inputs/model_runtime/run.py %S/../../../pim_backend/backend %t.cpp %t

// CHECK: #include "upmem_model.h"
// CHECK: uint32_t pim_model_pool_count = 3;
// CHECK: uint64_t pim_model_pool_bytes[] = {16, 8, 16};
// CHECK: uint32_t pim_model_metadata_bytes = 8;
// CHECK: uint32_t pim_model_device_mapping[] = {1, 1, 1};
// CHECK: extern "C" void model_pack(struct upmem_model_kernel *_model,
// CHECK: for (_model->batch_base = 0;
// CHECK: _set = _model->set;
// CHECK: _model->pools[0]
// CHECK: _model->pools[1]
// CHECK: _model->pools[2]
// CHECK: uint32_t _instance = _model->batch_base + _batch_slot;
// CHECK: if (_instance >= _model->instances) break;
// CHECK: if (upmem_model_wants(_model, 0))
// CHECK: _model->arguments[_instance * _model->argument_count + 0]
// CHECK: if (upmem_model_wants(_model, 1))
// CHECK: upmem_model_transfer(_model, 0, 0);
// CHECK: upmem_model_transfer(_model, 1, 0);
// CHECK: upmem_model_metadata(_model, "_pim_metadata");
// CHECK: upmem_model_launch(_model);
// CHECK: upmem_model_transfer(_model, 2, 1);
// CHECK: if (!_model->upload_weights)
// CHECK: _model->arguments[_instance * _model->argument_count + 2]

upmem.mram @A : memref<2xi32, #upmem<space mram>>
upmem.mram @B : memref<2xi32, #upmem<space mram>>
upmem.mram @C : memref<2xi32, #upmem<space mram>>
upmem.host_var @tile : index

func.func @model_pack(%a: memref<*xi32>, %b: memref<*xi32>, %c: memref<*xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %set = upmem.acquire 2, @kernel : !upmem.set
  %assignment = upmem.bind_devices %set partitions [2, 1, 1] device_mapping [1, 1, 1]
      : !upmem.set -> !upmem.assignment
  %ap = memref.alloc() : memref<2x2xi32>
  %bp = memref.alloc() : memref<1x2xi32>
  %cp = memref.alloc() : memref<2x2xi32>
  upmem.foreach_unit %set assignment %assignment : !upmem.set axes ["m", "n", "k"] counts [2, 1, 1] {
  ^bb0(%dpu: !upmem.unit, %m: index, %n: index, %k: index):
    %offset = arith.muli %m, %c2 : index
    %sa = upmem.stage %a origin[%offset] sizes[%c2] strides[%c1]
        pad_to[%c2] tile [2] into %ap
        : memref<*xi32>, memref<2x2xi32> -> memref<2xi32>
    upmem.to_mram %dpu, %sa, @A : memref<2xi32>
    %sb = upmem.stage %b origin[%c0] sizes[%c2] strides[%c1]
        pad_to[%c2] tile [2] into %bp {shared_over = ["m", "n", "k"]}
        : memref<*xi32>, memref<1x2xi32> -> memref<2xi32>
    upmem.to_mram %dpu, %sb, @B : memref<2xi32>
    upmem.to_host_var %dpu, %m, @tile : index
  }
  upmem.launch %set : !upmem.set
  upmem.gather %set assignment %assignment, @C capacity[%c2] into %cp
      : !upmem.set, memref<2x2xi32>
  upmem.foreach_group %cp : memref<2x2xi32> axes ["m", "n", "k"] counts [2, 1, 1] {
  ^bb0(%part: memref<2xi32>, %m: index, %n: index, %k: index):
    %offset = arith.muli %m, %c2 : index
    upmem.unstage %part into %c origin[%offset] sizes[%c2] strides[%c1]
        capacity[%c2] tile [2] : memref<2xi32>, memref<*xi32>
  }
  memref.dealloc %cp : memref<2x2xi32>
  memref.dealloc %bp : memref<1x2xi32>
  memref.dealloc %ap : memref<2x2xi32>
  return
}
