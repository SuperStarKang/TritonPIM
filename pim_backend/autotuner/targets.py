# Target별 PIMHardwareProfile 등록표.
#
# 데코레이터(@pim_autotune)와 torch lowering이 target 이름만으로 profile을
# 얻는 곳이다. 수치는 사용자가 target 실측·사양으로 계산해 여기에 한 번만
# 적는다. C++ 쪽 pim::TargetSpec 레지스트리와 같은 역할의 Python 대응물이다.
#
# UPMEM 값의 근거:
#   num_units 2048, lanes 16          — workspace/GEMM 실험 구성
#   local 64KB, 예약 24KB             — WRAM 물리 크기와 링커 오류 역산
#                                       (tasklet stack 16×1KB + runtime ≈ 23KB)
#   device 64MB                       — DPU MRAM
#   utilization_tile (131072,64,896)  — split_k.py 검증 설정에서 DPU 하나가
#                                       맡던 작업 묶음 (M/P_M, N/P_N, K/SPLIT_K)
#   k_split_block_multiple            — split-K 커널의 blocked 분할 전제

from .types import PIMHardwareProfile

TARGET_PROFILES = {
    "upmem": PIMHardwareProfile(
        target="upmem",
        num_units=2048,
        local_memory_bytes=64 * 1024,
        device_memory_bytes=64 * 1024 * 1024,
        alignment_bytes=8,
        execution_units_per_device=16,
        utilization_tile_m=131072,
        utilization_tile_n=64,
        utilization_tile_k=896,
        local_memory_reserved_bytes=24576,
        target_constraints={"k_split_block_multiple": True},
    ),
    # AiMX 등 다른 target은 실측값이 정해지면 여기에 추가한다.
}


def target_profile(target):
    """등록된 target의 profile. 없는 target은 KeyError로 명시적으로 실패한다."""
    return TARGET_PROFILES[target]
