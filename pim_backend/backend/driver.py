# PIMDriver — PIMDriver()를 set_active로 활성화하면 Triton runtime이 쓰는 driver.
#
# 주의: CPUDriver를 이 모듈의 top-level 이름으로 import하면 DriverBase 서브클래스가
# 2개로 세어져 backend discovery(_find_concrete_subclasses)가 실패한다. 그래서 CPU
# driver는 모듈(cpu_driver)로만 참조한다.
#
# PIMLauncher는 compiler가 만든 host.cpp/device.c 번들을 받아:
#   1) device.c → device.elf (dpu-upmem-dpurte-clang)
#   2) host.cpp → host.so   (g++ + libdpu, DPU_BINARY를 device.elf 절대경로로)
#   3) host.so를 dlopen해 <kernel>_host(A,B,C)를 ctypes로 호출
# CPULauncher가 launcher .so를 컴파일·로드하는 패턴을 PIM 런타임 glue에 맞게 옮긴 것.

import os
import sys
import json
import shutil
import hashlib
import tempfile
import subprocess
import ctypes
from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

import triton.backends.triton_shared.driver as cpu_driver


# dpu-upmem-dpurte-clang은 기본 v1B용 바이너리를 만든다. 시뮬레이터를 v1B로 맞춘다.
# 실제 하드웨어로 돌리려면 호출 전 UPMEM_PROFILE=backend=hw 로 덮어쓴다.
os.environ.setdefault("UPMEM_PROFILE", "chipId=0x42")


def _dpu_host_flags():
    # host.cpp(C++)가 libdpu를 링크할 때 쓰는 플래그. dpu-pkg-config가 표준 제공원.
    if shutil.which("dpu-pkg-config"):
        out = subprocess.check_output(["dpu-pkg-config", "--cflags", "--libs", "dpu"]).decode()
        return out.split()
    return ["-I/usr/include/dpu", "-ldpu"]


def _build_pim_so(host_src: str, device_src: str) -> str:
    # 소스 번들을 device.elf + host.so로 컴파일하고 host.so 경로를 돌려준다.
    # 소스 해시를 키로 캐시해 같은 커널은 한 번만 컴파일한다(CPULauncher와 동일한 발상).
    key = hashlib.sha256((host_src + device_src).encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    so_name = "pim_host.so"
    so_path = cache.get_file(so_name)
    if so_path is not None:
        return so_path

    with tempfile.TemporaryDirectory() as tmp:
        device_c = os.path.join(tmp, "device.c")
        host_cpp = os.path.join(tmp, "host.cpp")
        Path(device_c).write_text(device_src)
        Path(host_cpp).write_text(host_src)

        # device.c → device.elf (MVP: tasklet 1개)
        elf_tmp = os.path.join(tmp, "device.elf")
        subprocess.check_call(["dpu-upmem-dpurte-clang", "-DNR_TASKLETS=1", device_c, "-o", elf_tmp])
        # host.so는 실행 중 dpu_load(DPU_BINARY)로 elf를 연다 → 캐시에 박아 안정 경로를 확보한다.
        elf_path = cache.put(Path(elf_tmp).read_bytes(), "device.elf", binary=True)

        # host.cpp → host.so : DPU_BINARY 매크로를 캐시의 device.elf 절대경로로 덮어쓴다.
        so_tmp = os.path.join(tmp, so_name)
        subprocess.check_call(
            ["g++", "-O2", "-std=c++17", "-fPIC", "-shared",
             f'-DDPU_BINARY="{elf_path}"', host_cpp, *_dpu_host_flags(), "-o", so_tmp])
        so_path = cache.put(Path(so_tmp).read_bytes(), so_name, binary=True)

    return so_path


class PIMLauncher(object):

    def __init__(self, src, metadata):
        # signature에서 pointer 인자(A,B,C)의 위치를 뽑아둔다. 호출 시 args에서 같은 위치를 꺼낸다.
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        signature = {cst_key(k): v for k, v in src.signature.items()}
        self.pointer_indices = [i for i, ty in sorted(signature.items()) if ty[0] == "*"]

    def __call__(self, gridX, gridY, gridZ, stream, cu_function,
                 kernel_metadata, launch_metadata,
                 launch_enter_hook, launch_exit_hook, *args):
        # cu_function = load_binary가 돌려준 소스 번들 bytes (PIMUtils.load_binary 참고).
        bundle = json.loads(bytes(cu_function).decode("utf-8"))
        so_path = _build_pim_so(bundle["host"], bundle["device"])

        # host 진입 함수 이름 = metadata.name (compiler가 host.cpp에서 파싱해 채운 값). index 6.
        entry = kernel_metadata[6]
        lib = ctypes.CDLL(so_path)
        fn = getattr(lib, entry)
        fn.argtypes = [ctypes.c_void_p] * len(self.pointer_indices)
        fn.restype = None

        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)

        ptrs = [ctypes.c_void_p(args[i].data_ptr()) for i in self.pointer_indices]
        fn(*ptrs)

        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)


class PIMUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(PIMUtils, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": 2 ** 20,
            "multiprocessor_count": None,
            "sm_clock_rate": None,
            "mem_clock_rate": None,
            "mem_bus_width": None,
        }

    # kernel_obj(=compiler가 만든 소스 번들 bytes)를 그대로 function으로 흘려, launcher가 컴파일·실행한다.
    @staticmethod
    def load_binary(name, kernel_obj, shared, device):
        return (
            None,        # module
            kernel_obj,  # function (소스 번들 bytes)
            None,        # n_regs
            None,        # n_spills
            sys.maxsize, # n_max_threads
        )


class PIMDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = PIMUtils()
        self.launcher_cls = PIMLauncher
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
        # host(런타임 glue)는 CPU에서 돌고 DPU가 가속기다. 입력 텐서는 CPU 메모리에 있다.
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
