# PIMDriver — PIMDriver()를 set_active로 활성화하면 Triton runtime이 쓰는 driver.
#
# 주의: CPUDriver를 이 모듈의 top-level 이름으로 import하면 DriverBase 서브클래스가
# 2개로 세어져 backend discovery(_find_concrete_subclasses)가 실패한다. 그래서 CPU
# driver는 모듈(cpu_driver)로만 참조한다.
#
# PIMLauncher는 compiler가 만든 소스 번들을 받아 (upmem: host.cpp/device.c):
#   1) device.c → device.elf (dpu-upmem-dpurte-clang)
#   2) host.cpp → host.so   (g++ + libdpu, DPU_BINARY를 device.elf 절대경로로)
#   3) host.so를 dlopen해 <kernel>_host(A,B,C)를 ctypes로 호출
# CPULauncher가 launcher .so를 컴파일·로드하는 패턴을 PIM 런타임 glue에 맞게 옮긴 것.
#
# target 분기: 위 1~2(빌드)는 target마다 다르다(upmem=dpu-clang/libdpu, aimx=descriptor runtime).
# 3(dlopen→ctypes 호출)은 stub이 같은 C ABI를 노출한다는 전제로 공유한다. aimx 빌드는 seam만
# 잡아 두었고(_build_aimx_so) 실제 구현은 별도 서버/브랜치에서 채운다.
#
# UPMEM 전용 코드는 전부 _upmem_ 접두사 함수 안에 있다. 그 밖의 코드는 target을 모르며,
# 벤더 어휘(dpu/mram/tasklet 등)를 쓰지 않는다. target을 추가할 때는 _build_<target>_so 하나와
# __call__의 분기 한 줄만 늘리면 된다.

import os
import sys
import time
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


# ── PIM_TIMING (임시): 마지막 launch의 단계별 시간(ms). 지울 때 이 표시를 찾으면 된다.
# 생성된 host 코드가 이 배열을 export하는 target에서만 채워진다(지금은 upmem뿐).
TIMING_STAGES = ("stage", "h2d", "kernel", "d2h", "combine")
timing = [0.0] * len(TIMING_STAGES)


def _upmem_host_flags():
    # host.cpp(C++)가 libdpu를 링크할 때 쓰는 플래그. dpu-pkg-config가 표준 제공원.
    if shutil.which("dpu-pkg-config"):
        out = subprocess.check_output(["dpu-pkg-config", "--cflags", "--libs", "dpu"]).decode()
        return out.split()
    return ["-I/usr/include/dpu", "-ldpu"]


def _build_upmem_runtime() -> str:
    # config 무관한 공유 런타임(upmem_runtime.c)을 libupmem_runtime.so로 1회 빌드한다.
    # 내용이 고정이라 캐시 해시도 고정 → 모든 host.so가 "같은 경로의" .so를 링크 →
    # 동적 로더가 1개만 적재 → set 캐시 globals가 프로세스에서 공유된다(persistent의 핵심).
    # -Wl,-soname: 없으면 링커가 DT_NEEDED에 명령줄 경로(=캐시 절대경로)를 그대로 적어,
    #   캐시 디렉터리가 달라진 런타임 .so가 별개 객체로 적재된다. 그러면 set 캐시 globals가
    #   둘이 되어 DPU 예약이 두 벌 잡히므로, 위의 "1개만 적재"가 성립하지 않는다.
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    rt_c = os.path.join(backend_dir, "upmem_runtime.c")
    key = hashlib.sha256(("upmem_runtime|" + Path(rt_c).read_text()).encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    so_name = "libupmem_runtime.so"
    so_path = cache.get_file(so_name)
    if so_path is None:
        with tempfile.TemporaryDirectory() as tmp:
            so_tmp = os.path.join(tmp, so_name)
            subprocess.check_call(
                ["gcc", "-O2", "-fPIC", "-shared", f"-Wl,-soname,{so_name}",
                 rt_c, *_upmem_host_flags(), "-o", so_tmp])
            so_path = cache.put(Path(so_tmp).read_bytes(), so_name, binary=True)
    return so_path


def _build_upmem_so(host_src: str, device_src: str, local_lanes: int) -> str:
    # 소스 번들을 device.elf + host.so로 컴파일하고 host.so 경로를 돌려준다.
    # 소스 해시를 키로 캐시해 같은 커널은 한 번만 컴파일한다(CPULauncher와 동일한 발상).
    # local_lanes는 소스에 안 박히고 -DNR_TASKLETS 플래그로만 들어가므로 키에 따로 포함한다.
    # dpu_opt도 키에 넣어 최적화 레벨을 바꾸면 캐시가 자동 무효화되게 한다(안 그러면 옛 elf 재사용).

    # dpu-upmem-dpurte-clang은 기본 v1B용 바이너리를 만든다. 시뮬레이터를 v1B로 맞춘다.
    # 실제 하드웨어로 돌리려면 호출 전 UPMEM_PROFILE=backend=hw 로 덮어쓴다.
    # 캐시 조회보다 위에 둔다 — 캐시 적중이면 아래 컴파일을 건너뛰지만, 이 프로세스에서
    # 처음 dpu_alloc이 도는 것은 그 뒤의 launch이므로 여기서 이미 찍혀 있어야 한다.
    os.environ.setdefault("UPMEM_PROFILE", "chipId=0x42")

    dpu_opt = "-O2"  # device 커널 최적화. -O0(clang 기본)면 MAC 루프 언옵티마이즈드 → ~2× 느림. PGEMM도 -O2(libPGEMM.sh).
    # 공유 런타임 .so 경로(=내용 해시)를 키에 포함 → upmem_runtime.c가 바뀌면 host.so도 재빌드되어
    # 모든 host.so가 동일한 최신 런타임을 링크한다(섞이면 set 캐시 globals가 안 공유됨).
    runtime_so = _build_upmem_runtime()
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
        # elf를 캐시에 박아 안정 경로를 확보한다. 실행 중 upmem_acquire(DPU_BINARY)가 이 경로를 dpu_load한다.
        elf_path = cache.put(Path(elf_tmp).read_bytes(), "device.elf", binary=True)

        # host.cpp → host.so : DPU_BINARY 매크로를 캐시의 device.elf 절대경로로 덮어쓴다.
        # -fopenmp: 영역을 모으고 되돌리는 `#pragma omp parallel for`용. 없으면 pragma 무시되어 순차 실행.
        # upmem_runtime.h(-I backend_dir) + libupmem_runtime.so(전체 경로로 링크 + rpath) → 모든 host.so가
        #   같은 런타임 인스턴스를 공유(persistent set 캐시).
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        runtime_dir = os.path.dirname(runtime_so)
        so_tmp = os.path.join(tmp, so_name)
        subprocess.check_call(
            ["g++", "-O2", "-std=c++17", "-fPIC", "-shared", "-fopenmp",
             f'-DDPU_BINARY="{elf_path}"', f"-I{backend_dir}",
             host_cpp, *_upmem_host_flags(),
             runtime_so, "-Wl,-rpath," + runtime_dir, "-o", so_tmp])
        so_path = cache.put(Path(so_tmp).read_bytes(), so_name, binary=True)

    return so_path


def _build_aimx_so(stub_src: str, descriptors: str) -> str:
    # AiMX runtime stub(C++) + descriptor stream을 .so로 빌드해 경로를 돌려준다.
    # _build_upmem_so의 캐시 패턴(소스 해시 키)을 그대로 따를 예정이다.
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
        # "num_programs(x,y,z)"다. --pim-build-logical-grid가 program_id 인자를 지우고 num_programs만
        # 남기기 때문이다. 그래서 인자 위치를 특정 커널 모양에 맞춰 박지 않고 시그니처에서 읽는다.
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        signature = {cst_key(k): v for k, v in src.signature.items()}
        self.arg_indices = [i for i, ty in sorted(signature.items()) if ty != "constexpr"]
        self.arg_is_pointer = [signature[i][0] == "*" for i in self.arg_indices]
        # pim_target: 런타임에 빌드 툴체인을 고르는 분기 키. PIMOptions 필드라 metadata에 들어온다.
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
            # NR_TASKLETS는 컴파일러가 lane 예산 안에서 고른 값이다. 번들에 실려 온다.
            so_path = _build_upmem_so(bundle["host"], bundle["device"], bundle["lanes"])
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
        # 이 전역은 UPMEMToHost.cpp가 내는 것이므로 target 분기 안에 둔다. 밖에 두면 이 배열을
        # 내지 않는 target의 .so에서 in_dll이 ValueError를 던져 launch 자체가 실패한다.
        if self.pim_target == "upmem":
            timing[:] = list((ctypes.c_double * len(TIMING_STAGES)).in_dll(lib, "pim_timing_ms"))
        else:
            # 앞선 다른 커널의 값을 그 target의 것처럼 읽지 않도록 지운다.
            timing[:] = [0.0] * len(TIMING_STAGES)

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


def _pim_do_bench(fn, warmup=1, rep=3, quantiles=None, return_mode="mean", **kwargs):
    # autotune이 config를 고를 때 쓰는 측정 함수. triton.testing.do_bench를 쓸 수 없어서 둔다 —
    # 그쪽은 driver의 get_device_interface / get_empty_cache_for_benchmark / clear_cache를 요구하는데
    # 셋 다 GPU driver에만 있다. PIM launch는 동기 ctypes 호출이라 event도 synchronize도 필요 없다.
    #
    # warmup / rep은 GPU do_bench와 뜻이 다르다 — 그쪽은 밀리초 예산이고 호출 횟수를 estimate_ms로
    # 역산하지만, PIM은 launch 한 번이 수백 ms~수 초라 그 역산이 무의미하다. 여기서는 횟수다.
    #
    # warmup은 선택이 아니다. config마다 첫 호출에 device.elf 컴파일과 (num_units가 바뀌었으면)
    # DPU set 재예약이 섞여 들어온다.
    #
    # 재는 것은 launch 전체(staging + h2d + kernel + d2h + combine)다. tile 크기를 바꾸면 전송량도
    # 함께 변하므로 pim_timing_ms의 kernel 단계만 떼어 재면 최소화 대상이 어긋난다.
    from triton.testing import _summarize_statistics

    for _ in range(warmup):
        fn()

    times = []
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)

    return _summarize_statistics(times, quantiles, return_mode)


class PIMDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = PIMUtils()
        self.launcher_cls = PIMLauncher
        self.binary_ext = "pimobj"

    # 기본은 자동 선택되지 않는다 — triton.runtime.driver.set_active(PIMDriver())로 명시 활성화.
    #
    # 다만 인덕터 경로에서는 그 자리에 끼어들 수 없다. 생성된 커널의 device type이 "cpu"가
    # 아니면 torch가 set_driver_to_gpu()를 부르는데(_inductor/runtime/triton_heuristics.py:855),
    # 그 함수는 backends를 훑어 is_active()가 True인 것을 스스로 고르고 못 찾으면
    # RuntimeError를 낸다(triton_helpers.py:53-68). 그래서 env로 자동 선택 대상이 되게 연다.
    # env가 없으면 종전과 완전히 같다(False).
    @staticmethod
    def is_active():
        return os.environ.get("TRITON_PIM_ACTIVE", "0") == "1"

    def get_benchmarker(self):
        return _pim_do_bench

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
