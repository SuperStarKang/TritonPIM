# autotuner 패키지를 torch_backend에서 쓰기 위한 단일 진입점.
#
# [왜 shim이 필요한가]
# pim_backend는 설치된 패키지가 아니라 디렉토리다 — triton/backends/pim은 backend/
# 하위만 가리키는 symlink라서 autotuner는 어떤 import 경로에도 올라와 있지 않다.
# examples가 하는 것과 같은 방식으로 부모 디렉토리를 sys.path에 얹되, 여기 한
# 곳에서만 하고 다른 모듈은 이 모듈에서 이름을 받는다.
#
# Common IR config와 target spec을 torch lowering에 전달하는 단일 import 지점이다.

import sys
from pathlib import Path

_PIM_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _PIM_BACKEND_DIR not in sys.path:
    sys.path.insert(0, _PIM_BACKEND_DIR)

from autotuner import (
    ConstraintEngine,
    PIMBenchmarkResult,
    PIMConfig,
    PIMConfigCache,
    PIMAccess,
    PIMProblem,
    PIMTargetSpec,
    SearchBudget,
    build_config_space,
    heuristic_seeds,
    pim_hierarchy,
    target_spec,
    tune,
)
from autotuner.decorator import (
    DEFAULT_CACHE_PATH,
    pim_launch_options,
)
from autotuner.model import target_model
from autotuner.space import axis_values, legal_partitions

__all__ = [
    "ConstraintEngine", "PIMAccess", "PIMBenchmarkResult", "PIMConfig",
    "PIMConfigCache", "PIMProblem",
    "PIMTargetSpec", "SearchBudget", "build_config_space",
    "heuristic_seeds", "pim_hierarchy", "target_spec", "tune",
    "DEFAULT_CACHE_PATH", "pim_launch_options",
    "target_model", "axis_values", "legal_partitions",
]
