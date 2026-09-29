// Partial updates must preserve unwritten fields. Repeated writes to one field
// use the last value, and an empty foreach emits no metadata transfer.
// Also exercise a user symbol colliding with the preferred bundle name.
// RUN: triton-shared-translate --upmem-to-host %s | FileCheck %s --check-prefix=HOST
// RUN: triton-shared-translate --upmem-to-device %s | FileCheck %s --check-prefix=DEVICE

// HOST: struct _pim_metadata_t _args0[2] = {};
// HOST: _args0[_u].f0 =
// HOST-NEXT: _args0[_u].f2 =
// HOST-NEXT: _args0[_u].f0 =
// HOST-NEXT: }
// HOST-NEXT: _pim_stop(0);
// HOST-NEXT: _pim_start(1);
// HOST-NEXT: _idx = 0;
// HOST-NEXT: DPU_FOREACH(_set, _dpu, _idx) DPU_ASSERT(dpu_prepare_xfer(_dpu, (uint8_t*)&_args0[{{v[0-9]+}}[_idx]] + 0));
// HOST-NEXT: DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "_pim_metadata_1", 0, 4, DPU_XFER_DEFAULT));
// HOST-NEXT: _idx = 0;
// HOST-NEXT: DPU_FOREACH(_set, _dpu, _idx) DPU_ASSERT(dpu_prepare_xfer(_dpu, (uint8_t*)&_args0[{{v[0-9]+}}[_idx]] + 8));
// HOST-NEXT: DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "_pim_metadata_1", 8, 8, DPU_XFER_DEFAULT));
// HOST-NEXT: _pim_stop(1);
// HOST-NOT: _args1
// HOST: _pim_start(1);
// HOST-NEXT: _pim_stop(1);
// HOST-NOT: dpu_push_xfer
// HOST: dpu_launch
// DEVICE: __host struct _pim_metadata_t _pim_metadata_1;
// DEVICE: int32_t {{.*}} = _pim_metadata_1.f1;

upmem.host_var @first : index
upmem.host_var @_pim_metadata : index
upmem.host_var @last : index

func.func @partial(%value: index) {
  %set = upmem.acquire 2, @partial_device : !upmem.set
  %assignment = upmem.bind_devices %set partitions [2] device_mapping [2]
      : !upmem.set -> !upmem.assignment
  upmem.foreach_unit %set assignment %assignment : !upmem.set axes ["m"] counts [2] {
  ^bb0(%dpu: !upmem.unit, %unit: index):
    upmem.to_host_var %dpu, %unit, @first : index
    upmem.to_host_var %dpu, %unit, @last : index
    upmem.to_host_var %dpu, %value, @first : index
  }
  upmem.foreach_unit %set assignment %assignment : !upmem.set axes ["m"] counts [2] {
  ^bb0(%dpu: !upmem.unit, %unit: index):
  }
  upmem.launch %set : !upmem.set
  return
}

upmem.device_func @partial_device {
  %retained = upmem.get_host_var @_pim_metadata : index
}
