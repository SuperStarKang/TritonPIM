# Target별 PIMHardwareProfile 등록표.
#
# 데코레이터(@pim_autotune)와 torch lowering이 target 이름만으로 profile을
# 얻는 곳이다. 수치는 사용자가 target 실측·사양으로 계산해 여기에 한 번만
# 적는다. C++ 쪽 pim::TargetSpec 레지스트리와 같은 역할의 Python 대응물이다.
#
# UPMEM 값의 근거:
#   num_devices 2048, lanes/core 16   — workspace/GEMM 실험 구성
#   usable local 48KB                 — WRAM에서 runtime/stack을 제외한 커널 몫
#   usable bank 60MB                  — MRAM에서 runtime 영역을 제외한 커널 몫
#   capacity_limit (131072,64,896)    — split_k.py 검증 설정에서 DPU 하나가
#                                       맡던 논리 원소 상한
#   k_split_block_multiple            — split-K 커널의 blocked 분할 전제

from .types import PIMHardwareProfile

# AiMX runtime(mdrv_interface.c)의 FPGA별 heap과 backend/compiler.py가 쓰는
# profile을 같은 값으로 유지한다: 8 GiB에서 runtime 예약분 18 MiB를 뺀 총량을
# 256개 bank로 나눈다.
AIMX_DEVICE_HEAP_BYTES = 0x1FEE00000
AIMX_BANKS_PER_DEVICE = 256
AIMX_BANK_CAPACITY_BYTES = AIMX_DEVICE_HEAP_BYTES // AIMX_BANKS_PER_DEVICE

TARGET_PROFILES = {
    "upmem": PIMHardwareProfile(
        target="upmem",
        num_devices=2048,
        banks_per_device=1,
        cores_per_bank=1,
        lanes_per_core=16,
        usable_bank_capacity_bytes=60 * 1024 * 1024,
        usable_local_capacity_bytes=48 * 1024,
        alignment_bytes=8,
        capacity_limit_m=131072,
        capacity_limit_n=64,
        capacity_limit_k=896,
        local_memory_reserved_bytes=0,
        target_constraints={"k_split_block_multiple": True},
    ),
    "aimx": PIMHardwareProfile(
        target="aimx",
        num_devices=1,
        banks_per_device=AIMX_BANKS_PER_DEVICE,
        cores_per_bank=1,
        lanes_per_core=16,
        usable_bank_capacity_bytes=AIMX_BANK_CAPACITY_BYTES,
        usable_local_capacity_bytes=0,
        alignment_bytes=512,
        capacity_limit_m=8,
        capacity_limit_n=4096,
        capacity_limit_k=256,
        local_memory_reserved_bytes=0,
        target_constraints={"batch_values": (1, 2, 4, 8),
                            "default_batch": 8},
    ),
}


def target_profile(target):
    """등록된 target의 profile. 없는 target은 KeyError로 명시적으로 실패한다."""
    return TARGET_PROFILES[target]
