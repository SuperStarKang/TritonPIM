"""Shared PIM target description used by the compiler and the autotuner.

The common profile mirrors ``#pim.target_profile`` exactly.  Runtime fields are
kept in target-specific configs and never become PIM Common IR attributes.
"""

from dataclasses import dataclass
import glob


AIMX_BANKS_PER_UNIT = 256
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


@dataclass(frozen=True)
class PIMTargetSpec:
    common_profile: PIMTargetProfile
    runtime_config: PIMTargetRuntimeConfig


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
                                        or 48 * 1024),
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
                     batch=8, base_fpga=0):
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
        ),
    )


def make_target_spec(target, **options):
    builders = {
        "upmem": upmem_target_spec,
        "aimx": aimx_target_spec,
    }
    return builders[target](**options)
