// Mixed scalar types must have identical wire layouts on host and DPU. Every
// field is written, in a different order from the declarations, with per-DPU
// values: emit one metadata transfer, not one transfer per scalar.
// RUN: triton-shared-translate --upmem-to-host %s | FileCheck %s --check-prefixes=ABI,HOST
// RUN: triton-shared-translate --upmem-to-device %s | FileCheck %s --check-prefixes=ABI,DEVICE

// ABI: struct __attribute__((packed, aligned(8))) _pim_metadata_t {
// ABI-NEXT: int32_t f0; // trips
// ABI-NEXT: int8_t f1; // small
// ABI-NEXT: uint8_t _pad0[3];
// ABI-NEXT: int64_t f2; // wide
// ABI-NEXT: float f3; // scale
// ABI-NEXT: int16_t f4; // short_value
// ABI-NEXT: uint8_t f5; // flag
// ABI-NEXT: uint8_t _pad1[1];
// ABI-NEXT: int32_t f6; // count
// ABI-NEXT: uint8_t _pad2[4];
// ABI-NEXT: double f7; // precise
// ABI-NEXT: };
// ABI-NEXT: typedef char _pim_metadata_size_check[sizeof(struct _pim_metadata_t) == 40 ? 1 : -1];
// ABI: __builtin_offsetof(struct _pim_metadata_t, f2) == 8
// ABI: __builtin_offsetof(struct _pim_metadata_t, f7) == 32

// HOST: extern "C" void metadata_kernel(
// HOST: struct _pim_metadata_t _args0[4] = {};
// HOST: for (int64_t _u = 0; _u < 4; ++_u) {
// HOST-NEXT: int64_t [[UNIT:v[0-9]+]] = (_u / 1) % 4;
// HOST-NEXT: _args0[_u].f2 = a1;
// HOST-NEXT: _args0[_u].f0 = [[UNIT]];
// HOST-NEXT: _args0[_u].f7 = a6;
// HOST-NEXT: _args0[_u].f1 = a0;
// HOST-NEXT: _args0[_u].f3 = a2;
// HOST-NEXT: _args0[_u].f4 = a3;
// HOST-NEXT: _args0[_u].f5 = a4;
// HOST-NEXT: _args0[_u].f6 = a5;
// HOST-NEXT: }
// HOST-NEXT: _pim_stop(0);
// HOST-NEXT: _pim_start(1);
// HOST-NEXT: _idx = 0;
// HOST-NEXT: DPU_FOREACH(_set, _dpu, _idx) DPU_ASSERT(dpu_prepare_xfer(_dpu, (uint8_t*)&_args0[_idx] + 0));
// HOST-NEXT: DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "_pim_metadata", 0, 40, DPU_XFER_DEFAULT));
// HOST-NEXT: _pim_stop(1);
// HOST-NOT: dpu_push_xfer
// HOST: dpu_launch

// DEVICE: __host struct _pim_metadata_t _pim_metadata;
// DEVICE: int main()
// DEVICE-NEXT: int32_t {{.*}} = _pim_metadata.f0;
// DEVICE-NEXT: int8_t {{.*}} = _pim_metadata.f1;
// DEVICE-NEXT: int64_t {{.*}} = _pim_metadata.f2;
// DEVICE-NEXT: float {{.*}} = _pim_metadata.f3;
// DEVICE-NEXT: int16_t {{.*}} = _pim_metadata.f4;
// DEVICE-NEXT: uint8_t {{.*}} = _pim_metadata.f5;
// DEVICE-NEXT: int32_t {{.*}} = _pim_metadata.f6;
// DEVICE-NEXT: double {{.*}} = _pim_metadata.f7;

upmem.host_var @trips : index
upmem.host_var @small : i8
upmem.host_var @wide : i64
upmem.host_var @scale : f32
upmem.host_var @short_value : i16
upmem.host_var @flag : i1
upmem.host_var @count : i32
upmem.host_var @precise : f64

func.func @metadata_kernel(%small: i8, %wide: i64, %scale: f32,
                           %short: i16, %flag: i1, %count: i32, %precise: f64) {
  %set = upmem.acquire 4, @metadata_device : !upmem.set
  upmem.foreach_unit %set : !upmem.set axes ["m"] counts [4] {
  ^bb0(%dpu: !upmem.unit, %unit: index):
    upmem.to_host_var %dpu, %wide, @wide : i64
    upmem.to_host_var %dpu, %unit, @trips : index
    upmem.to_host_var %dpu, %precise, @precise : f64
    upmem.to_host_var %dpu, %small, @small : i8
    upmem.to_host_var %dpu, %scale, @scale : f32
    upmem.to_host_var %dpu, %short, @short_value : i16
    upmem.to_host_var %dpu, %flag, @flag : i1
    upmem.to_host_var %dpu, %count, @count : i32
  }
  upmem.launch %set : !upmem.set
  return
}

upmem.device_func @metadata_device {
  %trips = upmem.get_host_var @trips : index
  %small = upmem.get_host_var @small : i8
  %wide = upmem.get_host_var @wide : i64
  %scale = upmem.get_host_var @scale : f32
  %short = upmem.get_host_var @short_value : i16
  %flag = upmem.get_host_var @flag : i1
  %count = upmem.get_host_var @count : i32
  %precise = upmem.get_host_var @precise : f64
}
