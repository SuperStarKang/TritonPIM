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
#
# target 분기: 위 1~2(빌드)는 target마다 다르다(upmem=dpu-clang/libdpu, aimx=descriptor runtime).
# 3(dlopen→ctypes 호출)은 stub이 같은 C ABI를 노출한다는 전제로 공유한다. aimx 빌드는 seam만
# 잡아 두었고(_build_aimx_so) 실제 구현은 별도 서버/브랜치에서 채운다.

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


# ── PIM_TIMING (임시): 마지막 launch의 단계별 시간(ms). 지울 때 이 표시를 찾으면 된다.
TIMING_STAGES = ("stage", "h2d", "kernel", "d2h", "combine")
timing = [0.0] * len(TIMING_STAGES)


def _dpu_host_flags():
    # host.cpp(C++)가 libdpu를 링크할 때 쓰는 플래그. dpu-pkg-config가 표준 제공원.
    if shutil.which("dpu-pkg-config"):
        out = subprocess.check_output(["dpu-pkg-config", "--cflags", "--libs", "dpu"]).decode()
        return out.split()
    return ["-I/usr/include/dpu", "-ldpu"]


def _build_pim_runtime() -> str:
    # config 무관한 공유 런타임(pim_runtime.c)을 libpim_runtime.so로 1회 빌드한다.
    # 내용이 고정이라 캐시 해시도 고정 → 모든 host.so가 "같은 경로의" .so를 링크 →
    # 동적 로더가 1개만 적재 → set 캐시 globals가 프로세스에서 공유된다(persistent의 핵심).
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    rt_c = os.path.join(backend_dir, "pim_runtime.c")
    key = hashlib.sha256(("pim_runtime|" + Path(rt_c).read_text()).encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    so_name = "libpim_runtime.so"
    so_path = cache.get_file(so_name)
    if so_path is None:
        with tempfile.TemporaryDirectory() as tmp:
            so_tmp = os.path.join(tmp, so_name)
            subprocess.check_call(
                ["gcc", "-O2", "-fPIC", "-shared", rt_c, *_dpu_host_flags(), "-o", so_tmp])
            so_path = cache.put(Path(so_tmp).read_bytes(), so_name, binary=True)
    return so_path


def _build_pim_so(host_src: str, device_src: str, local_lanes: int) -> str:
    # 소스 번들을 device.elf + host.so로 컴파일하고 host.so 경로를 돌려준다.
    # 소스 해시를 키로 캐시해 같은 커널은 한 번만 컴파일한다(CPULauncher와 동일한 발상).
    # local_lanes는 소스에 안 박히고 -DNR_TASKLETS 플래그로만 들어가므로 키에 따로 포함한다.
    # dpu_opt도 키에 넣어 최적화 레벨을 바꾸면 캐시가 자동 무효화되게 한다(안 그러면 옛 elf 재사용).
    dpu_opt = "-O2"  # device 커널 최적화. -O0(clang 기본)면 MAC 루프 언옵티마이즈드 → ~2× 느림. PGEMM도 -O2(libPGEMM.sh).
    # 공유 런타임 .so 경로(=내용 해시)를 키에 포함 → pim_runtime.c가 바뀌면 host.so도 재빌드되어
    # 모든 host.so가 동일한 최신 런타임을 링크한다(섞이면 set 캐시 globals가 안 공유됨).
    runtime_so = _build_pim_runtime()
    key = hashlib.sha256((host_src + device_src + f"|L={local_lanes}|{dpu_opt}|rt={runtime_so}").encode("utf-8")).hexdigest()
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

        # device.c → device.elf. NR_TASKLETS = local_lanes (DPU 내부 tasklet 수). dpu_opt(-O2)는 위에서 정의.
        elf_tmp = os.path.join(tmp, "device.elf")
        subprocess.check_call(["dpu-upmem-dpurte-clang", dpu_opt, f"-DNR_TASKLETS={local_lanes}",
                               device_c, "-o", elf_tmp])
        # elf를 캐시에 박아 안정 경로를 확보한다. 실행 중 pim_acquire(DPU_BINARY)가 이 경로를 dpu_load한다.
        elf_path = cache.put(Path(elf_tmp).read_bytes(), "device.elf", binary=True)

        # host.cpp → host.so : DPU_BINARY 매크로를 캐시의 device.elf 절대경로로 덮어쓴다.
        # -fopenmp: split_k>1일 때 host reduce(tree)의 `#pragma omp parallel for`용. 없으면 pragma 무시되어 순차 실행.
        # pim_runtime.h(-I backend_dir) + libpim_runtime.so(전체 경로로 링크 + rpath) → 모든 host.so가
        #   같은 런타임 인스턴스를 공유(persistent set 캐시).
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        runtime_dir = os.path.dirname(runtime_so)
        so_tmp = os.path.join(tmp, so_name)
        subprocess.check_call(
            ["g++", "-O2", "-std=c++17", "-fPIC", "-shared", "-fopenmp",
             f'-DDPU_BINARY="{elf_path}"', f"-I{backend_dir}",
             host_cpp, *_dpu_host_flags(),
             runtime_so, "-Wl,-rpath," + runtime_dir, "-o", so_tmp])
        so_path = cache.put(Path(so_tmp).read_bytes(), so_name, binary=True)

    return so_path


def _build_aimx_so(stub_src: str, descriptors: str) -> str:
    # AiMX runtime stub(C++) + descriptor stream을 .so로 빌드해 경로를 돌려준다.
    # _build_pim_so의 캐시 패턴(소스 해시 키)을 그대로 따를 예정이다.
    #
    # 지금은 경로(seam)만 잡아 둔 상태 — AiMX runtime API·descriptor 포맷·링크 라이브러리·컴파일러가
    # 아직 정해지지 않아 실제 빌드 커맨드를 쓸 수 없다. upmem 경로로 fallback하지 않고 여기서
    # 명시적으로 터뜨린다(CLAUDE.md fallback 금지). 실제 구현은 별도 서버/브랜치에서 채운다(B1.4/B1.5).
    raise NotImplementedError(
        "AiMX launcher build 미구현 — descriptor 포맷/runtime API 확정 후 _build_aimx_so를 채울 것 "
        "(DESIGN 13.3/13.5, IMPLEMENT B1.4/B1.5)")


class PIMLauncher(object):

    def __init__(self, src, metadata):
        # 생성된 host 함수의 시그니처는 "커널의 비-constexpr 인자를 원래 순서대로" 다음
        # "num_programs(x,y,z)"다. --pim-build-grid가 program_id 인자를 지우고 num_programs만
        # 남기기 때문이다. 그래서 인자 위치를 특정 커널 모양에 맞춰 박지 않고 시그니처에서 읽는다.
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        signature = {cst_key(k): v for k, v in src.signature.items()}
        self.arg_indices = [i for i, ty in sorted(signature.items()) if ty != "constexpr"]
        self.arg_is_pointer = [signature[i][0] == "*" for i in self.arg_indices]
        # local_lanes(=NR_TASKLETS): device 빌드 파라미터. PIMOptions 필드라 metadata에 들어온다.
        self.local_lanes = metadata.local_lanes
        # pim_target: 런타임에 빌드 툴체인을 고르는 분기 키. local_lanes와 같은 PIMOptions 필드라
        # 같은 경로로 metadata에 들어온다("upmem" | "aimx").
        self.pim_target = metadata.pim_target

    def __call__(self, gridX, gridY, gridZ, stream, cu_function,
                 kernel_metadata, launch_metadata,
                 launch_enter_hook, launch_exit_hook, *args):
        # cu_function = load_binary가 돌려준 소스 번들 bytes (PIMUtils.load_binary 참고).
        bundle = json.loads(bytes(cu_function).decode("utf-8"))
        # target마다 빌드 툴체인이 다르다(upmem=dpu-clang+libdpu, aimx=descriptor runtime).
        # 빌드 단계만 분기하고, 아래 dlopen→ctypes 호출은 stub이 같은 (A,B,C,M,N,K) C ABI를
        # 노출한다는 전제로 공유한다(AiMX가 다른 인자를 요구하면 그때 호출부도 분기).
        if self.pim_target == "upmem":
            so_path = _build_pim_so(bundle["host"], bundle["device"], self.local_lanes)
        elif self.pim_target == "aimx":
            so_path = _build_aimx_so(bundle["stub"], bundle["descriptors"])
        else:
            raise ValueError(f"Unsupported PIM target '{self.pim_target}'.")

        # host 진입 함수 이름 = metadata.name (compiler가 host.cpp에서 파싱해 채운 값). index 6.
        entry = kernel_metadata[6]
        lib = ctypes.CDLL(so_path)
        fn = getattr(lib, entry)
        # 커널 인자를 원래 순서대로, 그 뒤에 grid 크기 셋.
        # grid는 컴파일 캐시 키에 없으므로(jit.py가 캐시를 조회한 뒤 정규화한다) 반드시
        # 런타임 인자로 넘겨야 한다. host 코드가 이 값으로 유닛별 영역을 계산한다.
        fn.argtypes = [ctypes.c_void_p if p else ctypes.c_int32
                       for p in self.arg_is_pointer] + [ctypes.c_int32] * 3
        fn.restype = None

        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)

        call = [ctypes.c_void_p(args[i].data_ptr()) if p else ctypes.c_int32(int(args[i]))
                for i, p in zip(self.arg_indices, self.arg_is_pointer)]
        fn(*call, ctypes.c_int32(gridX), ctypes.c_int32(gridY), ctypes.c_int32(gridZ))

        # ── PIM_TIMING (임시): 생성된 host 코드가 단계별 시간을 전역 배열에 남긴다.
        # 성능 판단이 끝나면 이 블록과 위의 PIM_TIMING 정의, UPMEMToHost.cpp의 같은 표시를 지운다.
        timing[:] = list((ctypes.c_double * len(TIMING_STAGES)).in_dll(lib, "pim_timing_ms"))

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
