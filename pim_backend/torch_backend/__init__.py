# triton_shared.pim_backend.torch_backend
#
# torch.compile된 그래프에서 aten._int_mm만 PIM semantic GEMM template로 실행한다.
# 나머지 연산은 기존 CPU Inductor(C++/OpenMP)가 그대로 맡는다.
#
#     from triton_shared.pim_backend.torch_backend import (
#         PIMInductorBackend, PIMTorchConfig)
#
#     compiled = torch.compile(
#         fn,
#         backend=PIMInductorBackend(PIMTorchConfig(...)),
#         fullgraph=True,
#         dynamic=False,
#     )

from .backend import PIMInductorBackend
from ._autotuner import PIMTargetSpec, target_spec
from .config import PIMTorchConfig, UnsupportedPIMIntMM, validate_pim_options
from .execution import PIMExecution, resolve_execution
from .runtime_compat import set_driver_to_backend
from .scheduling import PIMCombinedScheduling
from .template import pim_gemm_template

__all__ = [
    "PIMInductorBackend",
    "PIMTorchConfig",
    "PIMTargetSpec",
    "PIMExecution",
    "UnsupportedPIMIntMM",
    "PIMCombinedScheduling",
    "pim_gemm_template",
    "resolve_execution",
    "set_driver_to_backend",
    "target_spec",
    "validate_pim_options",
]
