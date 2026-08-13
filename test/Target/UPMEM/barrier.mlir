// upmem.barrier 가 barrier_wait 로 나가고, 선언(BARRIER_INIT)은 op 이 있을 때만 나온다.
// RUN: triton-shared-translate --upmem-to-device %s | FileCheck %s

// CHECK: #include <barrier.h>
// CHECK: BARRIER_INIT(pim_barrier, NR_TASKLETS);
// CHECK: int main()
// CHECK: barrier_wait(&pim_barrier);

upmem.device_func @kernel attributes {upmem.lanes = 4 : i64} {
  %tid = upmem.tasklet_id : index
  upmem.barrier
}
