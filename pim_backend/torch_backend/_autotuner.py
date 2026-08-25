# autotuner 패키지를 torch_backend에서 쓰기 위한 단일 진입점.
#
# [왜 shim이 필요한가]
# pim_backend는 설치된 패키지가 아니라 디렉토리다 — triton/backends/pim은 backend/
# 하위만 가리키는 symlink라서 autotuner는 어떤 import 경로에도 올라와 있지 않다.
# examples가 하는 것과 같은 방식으로 부모 디렉토리를 sys.path에 얹되, 여기 한
# 곳에서만 하고 다른 모듈은 이 모듈에서 이름을 받는다.
#
# [autotuner는 읽기만 한다]
# 고치지 않는다. 우리가 쓰는 것은 전부 이미 공개된 이름이다. default_config와
# pim_launch_options 둘만 autotuner/__init__.py가 내보내지 않아 submodule에서
# 직접 가져온다 — 이 둘이 torch_backend가 복제했던 바로 그 코드다.

import sys
from pathlib import Path

_PIM_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _PIM_BACKEND_DIR not in sys.path:
    sys.path.insert(0, _PIM_BACKEND_DIR)

from autotuner import (  # noqa: E402
    ConstraintEngine,
    PIMConfig,
    PIMConfigCache,
    PIMHardwareProfile,
    PIMPlan,
    PIMPlanner,
    PIMProblem,
    build_config_space,
    heuristic_seeds,
    make_config,
    strategy_for_pk,
    target_profile,
)
from autotuner.types import ceil_div  # noqa: E402
from autotuner.decorator import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    default_config,
    pim_launch_options,
)

__all__ = [
    "ConstraintEngine", "PIMConfig", "PIMConfigCache", "PIMHardwareProfile",
    "PIMPlan", "PIMPlanner", "PIMProblem", "build_config_space",
    "heuristic_seeds", "make_config", "strategy_for_pk", "target_profile",
    "DEFAULT_CACHE_PATH", "default_config", "pim_launch_options",
    "ceil_div",
]
