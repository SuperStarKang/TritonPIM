# triton_shared.pim_backend.torch_backend
#
# torch.compile된 그래프에서 aten._int_mm만 PIM split-K template로 실행한다.
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
from .config import PIMTorchConfig, UnsupportedPIMIntMM, validate_pim_options
from .runtime_compat import set_driver_to_backend
from .scheduling import PIMCombinedScheduling
from .template import pim_split_k_template

__all__ = [
    "PIMInductorBackend",
    "PIMTorchConfig",
    "UnsupportedPIMIntMM",
    "PIMCombinedScheduling",
    "pim_split_k_template",
    "set_driver_to_backend",
    "validate_pim_options",
]
