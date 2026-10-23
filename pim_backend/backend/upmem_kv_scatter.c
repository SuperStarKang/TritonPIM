// key/value cache의 바뀐 8 byte 단위를 MRAM의 제자리에 쓰는 DPU program.
// host가 launch 전에 단위 수와 MRAM 주소(모든 DPU 공통), DPU마다의 8 byte 값을 WRAM에 쓴다.
// 주소와 값은 host의 cache image에서 나오므로 단위 안의 다른 bytes도 이미 맞는 값이다.
// KV_CHUNKS(최대 단위 수)와 NR_TASKLETS는 compile 때 정한다.
#include <defs.h>
#include <mram.h>
#include <stdint.h>

__host uint32_t kv_count;
__host uint32_t kv_address[KV_CHUNKS];
__host uint64_t kv_data[KV_CHUNKS];

int main(void) {
    for (uint32_t index = me(); index < kv_count; index += NR_TASKLETS)
        mram_write(&kv_data[index], (__mram_ptr void *)kv_address[index], sizeof(uint64_t));
    return 0;
}
