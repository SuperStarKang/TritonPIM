// host 의 round loop: scf.for 가 C 의 for 문으로 나오고 launch 가 그 안에서 반복된다.
// RUN: triton-shared-translate --upmem-to-host %s | FileCheck %s

// Each round restarts its interval, but never resets the accumulated time.
// CHECK: static void _pim_start(int slot) {
// CHECK-NEXT: _pim_started_ns[slot] = _pim_now_ns();
// CHECK-NEXT: }
// CHECK: static void _pim_stop(int slot) {
// CHECK: pim_timing_ms[slot] += (double)elapsed_ns / 1e6;
// CHECK: extern "C" void drv()
// CHECK: memset(pim_timing_ms, 0, sizeof pim_timing_ms);
// CHECK-NOT: memset(pim_timing_ms
// CHECK: for (int64_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
// CHECK-NEXT: _pim_start(2);
// CHECK-NEXT: DPU_ASSERT(dpu_launch(_set, DPU_SYNCHRONOUS));
// CHECK-NEXT: _pim_stop(2);
// CHECK-NEXT: }
// CHECK-NEXT: }

func.func @drv() {
  %set = upmem.acquire 4, @kernel_device : !upmem.set
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %r = %c0 to %c2 step %c1 {
    upmem.launch %set : !upmem.set
  }
  return
}
