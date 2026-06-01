# PIMDriver — device="pim"일 때 Triton runtime이 활성화하는 driver.
#
# 주의: CPUDriver를 이 모듈의 top-level 이름으로 import하면 DriverBase 서브클래스가
# 2개로 세어져 backend discovery(_find_concrete_subclasses)가 실패한다. 그래서 CPU
# driver는 모듈(cpu_driver)로만 참조한다. CPULauncher/CPUUtils는 DriverBase 서브클래스가
# 아니므로 그 자체를 참조해도 무방하지만, 일관성을 위해 모듈 경유로 접근한다.
#
# 현재 launcher/utils는 CPU 구현을 그대로 재사용한다 (lowering이 CPU 경로라 실행도 CPU에서).
# get_current_target만 "pim"을 반환해 PIMBackend가 dispatch되게 한다.

from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

import triton.backends.triton_shared.driver as cpu_driver


class PIMDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = cpu_driver.CPUUtils()
        self.launcher_cls = cpu_driver.CPULauncher
        self.binary_ext = "pimobj"

    # PIMDriver는 자동 선택되지 않는다. triton.runtime.driver.set_active(PIMDriver())로 명시 활성화.
    @staticmethod
    def is_active():
        return False

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_capability(self):
        return ("cpu", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        # 현재 실행은 CPU 경로를 재사용하므로 CPU를 반환한다.
        return "cpu"

    def set_current_device(self, device):
        assert device == "cpu"
        return

    def get_current_target(self):
        # backend 이름이 "pim"이어야 make_backend가 PIMBackend를 선택한다.
        return GPUTarget("pim", 0, 0)

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def map_python_to_cpp_type(self, ty: str) -> str:
        return cpu_driver._ty_to_cpp(ty)
