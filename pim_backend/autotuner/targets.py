"""The autotuner view of the compiler's shared PIM target specification."""

from triton.backends.pim.target import (
    AiMXRuntimeConfig,
    PIMGemmProfileSpec,
    PIMGemvProfileSpec,
    PIMTargetProfile,
    PIMTargetRuntimeConfig,
    PIMTargetSpec,
    UPMEMRuntimeConfig,
    make_target_spec,
)


def target_spec(target, **options):
    return make_target_spec(target, **options)


__all__ = [
    "AiMXRuntimeConfig",
    "PIMGemmProfileSpec",
    "PIMGemvProfileSpec",
    "PIMTargetProfile",
    "PIMTargetRuntimeConfig",
    "PIMTargetSpec",
    "UPMEMRuntimeConfig",
    "target_spec",
]
