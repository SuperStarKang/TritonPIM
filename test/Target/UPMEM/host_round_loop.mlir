// host 의 round loop: scf.for 가 C 의 for 문으로 나오고 launch 가 그 안에서 반복된다.
// RUN: triton-shared-translate --upmem-to-host %s | FileCheck %s

// CHECK: extern "C" void drv()
// CHECK: for (int64_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
// CHECK:   DPU_ASSERT(dpu_launch(_set, DPU_SYNCHRONOUS));
// CHECK: }

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
