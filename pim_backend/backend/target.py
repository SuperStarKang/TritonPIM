"""Shared PIM target description used by the compiler and the autotuner.

The common profile mirrors ``#pim.target_profile`` exactly.  Runtime fields are
kept in target-specific configs and never become PIM Common IR attributes.
"""

from dataclasses import dataclass
import glob


# Silent-PIM (AiMX) 한 unit = GDDR6 칩 8개 (칩당 4 GB, 2 channel).
#   channel 16 (칩당 2) × bank 16 = bank 256, bank당 ALU 16 → lane 4096
#   channel 2 GB, bank 128 MB, 전체 32 GB, column 접근 256 bit (= bf16 16개)
AIMX_CHIPS_PER_UNIT = 8
AIMX_CHANNELS_PER_UNIT = 16
AIMX_BANKS_PER_CHANNEL = 16
AIMX_BANKS_PER_UNIT = AIMX_CHANNELS_PER_UNIT * AIMX_BANKS_PER_CHANNEL
AIMX_ALUS_PER_BANK = 16
AIMX_PHYSICAL_BANK_BYTES = 128 * 1024 * 1024
AIMX_PHYSICAL_UNIT_BYTES = AIMX_BANKS_PER_UNIT * AIMX_PHYSICAL_BANK_BYTES  # 32 GB

# 할당 가능한 용량은 물리 용량이 아니라 pim_sw runtime heap이다
# (mem_src/mdrv_interface.c: heap_size = 0x1FEE00000, 8 GB - 18 MB). 이를 넘으면
# runtime이 "PIM] OOM"으로 종료하므로 planner의 bank 용량은 heap을 bank 수로 나눈 값이다.
AIMX_DEVICE_HEAP_BYTES = 0x1FEE00000
AIMX_BANK_CAPACITY_BYTES = (
    AIMX_DEVICE_HEAP_BYTES // AIMX_BANKS_PER_UNIT)


@dataclass(frozen=True)
class PIMTargetProfile:
    target: str
    num_units: int
    banks_per_unit: int
    cores_per_bank: int
    lanes_per_core: int
    bank_capacity_limit_bytes: int
    local_capacity_limit_bytes: int


@dataclass(frozen=True)
class PIMTargetRuntimeConfig:
    pass


@dataclass(frozen=True)
class UPMEMRuntimeConfig(PIMTargetRuntimeConfig):
    nr_tasklets: int
    host_threads: int


@dataclass(frozen=True)
class AiMXRuntimeConfig(PIMTargetRuntimeConfig):
    batch: int
    base_fpga: int
    # GEMM A의 K 저장 pitch (BF16 원소). lowering의 matmul-k-alignment와 같고 0은 4096/batch.
    matmul_k_alignment: int


@dataclass(frozen=True)
class PIMTargetSpec:
    common_profile: PIMTargetProfile
    runtime_config: PIMTargetRuntimeConfig


@dataclass(frozen=True)
class PIMGemmProfileSpec:
    """Backend-provided starting shape and dtypes for the common GEMM profiler."""

    block_shape: tuple
    input_dtype: str
    output_dtype: str
    group_shape: tuple

    def __post_init__(self):
        if len(self.block_shape) != 3 or any(
                value < 1 or value & (value - 1) for value in self.block_shape):
            raise ValueError(
                f"GEMM profiling Block dimensions must be positive powers of two: {self.block_shape}")
        if len(self.group_shape) != 3 or min(self.group_shape) < 1:
            raise ValueError("GEMM profiling Group dimensions must be positive")

    @property
    def block_candidates(self):
        # Block은 모든 target이 함께 쓰는 Triton Block 하나다. 크기 비교는 Group이 맡는다.
        return (self.block_shape,)


@dataclass(frozen=True)
class PIMGemvProfileSpec:
    """Starting shape and dtypes for single-device GEMV profiling."""

    block_shape: tuple
    input_dtype: str
    output_dtype: str
    group_shape: tuple

    def __post_init__(self):
        if len(self.block_shape) != 3 or any(
                value < 1 or value & (value - 1) for value in self.block_shape):
            raise ValueError(
                f"GEMV profiling Block dimensions must be positive powers of two: {self.block_shape}")
        if len(self.group_shape) != 3 or min(self.group_shape) < 1:
            raise ValueError("GEMV profiling Group dimensions must be positive")
        if self.block_shape[1] != 1 or self.group_shape[1] != 1:
            raise ValueError("GEMV profiling must keep the unused column axis at one")

    @property
    def block_candidates(self):
        return tuple(
            (self.block_shape[0] * rows, 1, self.block_shape[2] * reduction)
            for rows in (1, 2, 4) for reduction in (1, 2))


def aimx_unit_count():
    """Return the number of AiMX units visible to the runtime."""
    count = len(glob.glob("/dev/pl_dma[0-9]*"))
    if count == 0:
        raise RuntimeError(
            "/dev/pl_dma* 가 없습니다. pim_drv 가 올라와 있는지 확인해 "
            "주십시오. 개수를 직접 주려면 num_units 옵션을 쓰십시오.")
    return count


def upmem_target_spec(*, num_units=0, banks_per_unit=0,
                      cores_per_bank=0, lanes_per_core=16,
                      bank_capacity_limit_bytes=0,
                      local_capacity_limit_bytes=0, nr_tasklets=0,
                      host_threads=8):
    return PIMTargetSpec(
        common_profile=PIMTargetProfile(
            target="upmem",
            num_units=num_units or 2048,
            banks_per_unit=banks_per_unit or 1,
            cores_per_bank=cores_per_bank or 1,
            lanes_per_core=lanes_per_core,
            bank_capacity_limit_bytes=(bank_capacity_limit_bytes
                                       or 60 * 1024 * 1024),
            local_capacity_limit_bytes=(local_capacity_limit_bytes
                                        or 46680),
        ),
        runtime_config=UPMEMRuntimeConfig(
            nr_tasklets=nr_tasklets,
            host_threads=host_threads,
        ),
    )


def aimx_target_spec(*, num_units=0, banks_per_unit=0,
                     cores_per_bank=0, lanes_per_core=16,
                     bank_capacity_limit_bytes=0,
                     local_capacity_limit_bytes=0,
                     batch=8, base_fpga=0, matmul_k_alignment=0):
    return PIMTargetSpec(
        common_profile=PIMTargetProfile(
            target="aimx",
            num_units=num_units or aimx_unit_count(),
            banks_per_unit=banks_per_unit or AIMX_BANKS_PER_UNIT,
            cores_per_bank=cores_per_bank or 1,
            lanes_per_core=lanes_per_core,
            bank_capacity_limit_bytes=(bank_capacity_limit_bytes
                                       or AIMX_BANK_CAPACITY_BYTES),
            local_capacity_limit_bytes=local_capacity_limit_bytes,
        ),
        runtime_config=AiMXRuntimeConfig(
            batch=batch,
            base_fpga=base_fpga,
            matmul_k_alignment=matmul_k_alignment,
        ),
    )


def make_target_spec(target, **options):
    builders = {
        "upmem": upmem_target_spec,
        "aimx": aimx_target_spec,
    }
    return builders[target](**options)
