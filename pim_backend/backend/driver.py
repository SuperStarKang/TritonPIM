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
# target 분기: 위 1~2(빌드)는 target마다 다르다(upmem=dpu-clang/libdpu, aimx=g++/libpim).
# 3(dlopen→ctypes 호출)은 stub이 같은 C ABI를 노출한다는 전제로 공유한다.
# aimx는 device 프로그램이 없다 — descriptor chain 자체가 프로그램이라 stub 하나만 빌드한다.
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
from .runtime import current_invocation, resident_lock


# ── PIM_TIMING (임시): 마지막 launch의 단계별 시간(ms)과 그 단계 이름.
# 지울 때 이 표시를 찾으면 된다. 두 target의 단계 이름이 다르므로 launch마다 다시 놓는다.
UPMEM_TIMING_VERSION = 3  # Generated host ABI; resident reports are adjusted by the runtime.
UPMEM_RESIDENT_TIMING_VERSION = 4
_STAGE_NAMES = {
    # H2D/D2H: prepare+push. kernel: synchronous dpu_launch only.
    # Exclusive action timings; summing them never counts layout twice.
    "upmem": ("prepare", "h2d", "kernel", "d2h", "reduction",
              "buffer_alloc", "buffer_free", "acquire", "buffer_init",
              "layout_a", "layout_b", "layout_other", "layout_c"),
    "aimx": ("alloc", "build", "stage", "invoke", "unstage"),
}
UPMEM_RESIDENT_TIMING_STAGES = _STAGE_NAMES["upmem"] + ("program_load", "resident_host")
TIMING_STAGES = _STAGE_NAMES["upmem"]
TIMING_VERSION = UPMEM_TIMING_VERSION
timing = [0.0] * len(TIMING_STAGES)
# H2D role subtotals, already included in the resident h2d stage.
resident_h2d_timing = None
# 마지막 launch의 pim_exec 호출 수. aimx만 낸다(upmem은 0으로 둔다). execution tile 하나의
# throughput은 invoke 시간을 이 값으로 나눠야 나오는데, host에서 tile 산술을 다시 하면
# device 분할 규칙이 바뀔 때마다 어긋난다.
invokes = 0


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
    runtime_sources = ("upmem_runtime.c", "upmem_runtime.h",
                       "upmem_resident.c", "upmem_resident.h",
                       "upmem_model.c", "upmem_model.h")
    key = hashlib.sha256(("upmem_runtime|" + "".join(
        Path(backend_dir, name).read_text() for name in runtime_sources
    )).encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    so_name = "libupmem_runtime.so"
    so_path = cache.get_file(so_name)
    if so_path is None:
        with tempfile.TemporaryDirectory() as tmp:
            so_tmp = os.path.join(tmp, so_name)
            subprocess.check_call(
                ["gcc", "-O2", "-fPIC", "-shared", f"-Wl,-soname,{so_name}",
                 rt_c, os.path.join(backend_dir, "upmem_resident.c"),
                 os.path.join(backend_dir, "upmem_model.c"),
                 *_upmem_host_flags(), "-o", so_tmp])
            so_path = cache.put(Path(so_tmp).read_bytes(), so_name, binary=True)
    return so_path


def _build_upmem_so(host_src: str, device_src: str, nr_tasklets: int) -> str:
    # 소스 번들을 device.elf + host.so로 컴파일하고 host.so 경로를 돌려준다.
    # 소스 해시를 키로 캐시해 같은 커널은 한 번만 컴파일한다(CPULauncher와 동일한 발상).
    # nr_tasklets는 -DNR_TASKLETS 플래그로만 들어가므로 키에 따로 포함한다.
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
    key = hashlib.sha256((host_src + device_src + f"|T={nr_tasklets}|{dpu_opt}|rt={runtime_so}").encode("utf-8")).hexdigest()
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

        # device.c → device.elf. NR_TASKLETS는 실제 활성 tasklet 수다.
        elf_tmp = os.path.join(tmp, "device.elf")
        subprocess.check_call(["dpu-upmem-dpurte-clang", dpu_opt, f"-DNR_TASKLETS={nr_tasklets}",
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


def _aimx_include_dirs():
    # 생성된 stub이 쓰는 두 헤더의 자리.
    #   pim.h       — libpim이 export하는 것 (설치본은 /usr/include)
    #   pim_math.h  — pim_isa_32_t / PIM_RD_INSTR_MC / pim_exec. 설치되지 않는 소스 트리 헤더라
    #                 경로를 받아야 한다. 짐작하지 않고 없으면 그 자리에서 터뜨린다.
    root = os.getenv("AIMX_PIM_SW", "")
    if not root:
        raise RuntimeError(
            "AIMX_PIM_SW가 설정되지 않았습니다. pim_sw 소스 트리의 경로를 넣어 주십시오 "
            "(예: export AIMX_PIM_SW=$HOME/pim_aimx_sw/pim_sw). "
            "생성된 stub이 pim_math.h를 include하는데 이 헤더는 설치되지 않습니다.")
    math_dir = os.path.join(root, "src", "pim_math_src")
    if not os.path.isfile(os.path.join(math_dir, "pim_math.h")):
        raise RuntimeError(f"{math_dir}/pim_math.h가 없습니다. AIMX_PIM_SW를 확인해 주십시오.")
    return [math_dir, os.path.join(root, "include")]


def _build_aimx_so(stub_src: str, descriptors: str, base_fpga: int) -> str:
    # AiMX runtime stub(C++) + descriptor program(C 헤더)을 .so로 빌드해 경로를 돌려준다.
    # _build_upmem_so의 캐시 패턴(소스 해시 키)을 그대로 따른다.
    #
    # descriptor 쪽은 별도 .so가 아니라 stub이 include하는 헤더다 — program 하나가 함수
    # 하나이고 stub이 launch당 한 번만 부르므로 나눌 이유가 없다.
    includes = _aimx_include_dirs()
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    # aimx_runtime.h 내용을 키에 넣는다. 생성된 소스는 그대로인데 이 헤더만 고치는 경우가
    # 있고(런타임 선언·매크로), 키에 없으면 옛 .so가 그대로 재사용된다.
    runtime_h = Path(os.path.join(backend_dir, "aimx_runtime.h")).read_text()
    # base_fpga는 소스가 아니라 -D로 들어가므로 키에 따로 넣어야 한다.
    key = hashlib.sha256(
        (stub_src + descriptors + "|" + "|".join(includes) + "|" + runtime_h
         + f"|fpga={base_fpga}").encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    so_name = "pim_host.so"
    so_path = cache.get_file(so_name)
    if so_path is not None:
        return so_path

    with tempfile.TemporaryDirectory() as tmp:
        Path(os.path.join(tmp, "aimx_descriptors.h")).write_text(descriptors)
        stub_cpp = os.path.join(tmp, "stub.cpp")
        Path(stub_cpp).write_text(stub_src)
        so_tmp = os.path.join(tmp, so_name)
        # pim_math.h는 C로 쓰였고 C++로 읽으면 걸리는 곳이 셋 있다. 셋 다 우리가 부르지 않는
        # 코드(decode_opcode의 진단 문자열)라 진단만 끈다.
        #   -Wno-unused-function : 쓰지 않는 static inline stub이 여럿이다
        #   -Wno-narrowing       : case (CODE_READ) 같은 0xdeadbea1 라벨이 int로 좁혀진다
        #   -Wno-write-strings   : char* 반환에 문자열 상수를 쓴다
        # -lm: libpim.so가 pow를 미해결 심볼로 남긴다.
        # -fopenmp: 생성된 stub이 PIM Device마다 host thread 하나를 띄운다. 이것 없이는
        # pragma가 무시되어 조용히 순차 실행이 되므로 옵션이 아니라 필수다.
        # -O3: stage/unstage의 변환 loop이 원소 단위라 벡터화 여부가 곧 host 시간이다.
        # -O2는 -ftree-vectorize를 매우 보수적으로만 쓰고, stride가 런타임 값인 loop을
        # stride-1 판으로 복제하는 -fversion-loops-for-strides는 -O3에서만 켜진다.
        subprocess.check_call(
            ["g++", "-O3", "-std=c++17", "-fPIC", "-shared", "-fopenmp",
             f"-DAIMX_BASE_FPGA={int(base_fpga)}",
             "-Wno-unused-function", "-Wno-narrowing", "-Wno-write-strings",
             f"-I{tmp}", f"-I{backend_dir}", *[f"-I{d}" for d in includes],
             stub_cpp, "-lpim", "-lpthread", "-lm", "-o", so_tmp])
        so_path = cache.put(Path(so_tmp).read_bytes(), so_name, binary=True)
        # 생성된 소스를 .so 옆에 남긴다. 하드웨어에서 답이 틀릴 때 볼 곳이 여기뿐이다
        # (descriptor 주소식과 stage/unstage 인덱싱이 전부 이 두 파일에 있다).
        cache.put(stub_src, "stub.cpp", binary=False)
        cache.put(descriptors, "aimx_descriptors.h", binary=False)

    return so_path


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
        with resident_lock:
            return self._launch(gridX, gridY, gridZ, stream, cu_function,
                                kernel_metadata, launch_metadata,
                                launch_enter_hook, launch_exit_hook, *args)

    def _launch(self, gridX, gridY, gridZ, stream, cu_function,
                kernel_metadata, launch_metadata,
                launch_enter_hook, launch_exit_hook, *args):
        # cu_function = load_binary가 돌려준 소스 번들 bytes (PIMUtils.load_binary 참고).
        bundle = json.loads(bytes(cu_function).decode("utf-8"))
        invocation = current_invocation.get()
        if invocation is not None:
            if launch_enter_hook is not None:
                launch_enter_hook(launch_metadata)
            invocation.launch(self, bundle, kernel_metadata[6],
                              (gridX, gridY, gridZ), args)
            if launch_exit_hook is not None:
                launch_exit_hook(launch_metadata)
            return
        # target마다 빌드 툴체인이 다르다(upmem=dpu-clang+libdpu, aimx=descriptor runtime).
        # 빌드 단계만 분기하고, 아래 dlopen→ctypes 호출은 stub이 같은 (A,B,C,M,N,K) C ABI를
        # 노출한다는 전제로 공유한다(AiMX가 다른 인자를 요구하면 그때 호출부도 분기).
        if self.pim_target == "upmem":
            # NR_TASKLETS는 컴파일러가 lane 예산 안에서 고른 값이다. 번들에 실려 온다.
            so_path = _build_upmem_so(
                bundle["host"], bundle["device"], bundle["nr_tasklets"])
        elif self.pim_target == "aimx":
            so_path = _build_aimx_so(bundle["stub"], bundle["descriptors"],
                                     bundle["base_fpga"])
        else:
            raise ValueError(f"Unsupported PIM target '{self.pim_target}'.")

        # host 진입 함수 이름 = metadata.name (compiler가 host.cpp에서 파싱해 채운 값). index 6.
        entry = kernel_metadata[6]
        lib = ctypes.CDLL(so_path)
        if self.pim_target == "upmem":
            timing_version = ctypes.c_uint32.in_dll(lib, "pim_timing_version").value
            if timing_version != UPMEM_TIMING_VERSION:
                raise RuntimeError(
                    f"UPMEM timing version {timing_version!r}; expected {UPMEM_TIMING_VERSION}. "
                    "Rebuild triton-shared-opt and triton-shared-translate, then regenerate the Triton artifacts "
                    "with the current backend compiler version.")
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
        # 성능 판단이 끝나면 이 블록과 위의 PIM_TIMING 정의, UPMEMToHost.cpp / AiMXToRuntimeStub.cpp의
        # 같은 표시를 지운다. 단계 이름은 target마다 다르므로 같이 놓는다.
        global TIMING_STAGES, TIMING_VERSION, resident_h2d_timing, invokes
        TIMING_STAGES = _STAGE_NAMES[self.pim_target]
        TIMING_VERSION = UPMEM_TIMING_VERSION if self.pim_target == "upmem" else None
        resident_h2d_timing = None
        timing[:] = list((ctypes.c_double * len(TIMING_STAGES)).in_dll(lib, "pim_timing_ms"))
        # 이 전역은 AiMXToRuntimeStub.cpp만 낸다. upmem .so에서 in_dll을 부르면
        # ValueError로 launch 자체가 실패하므로 target을 보고 부른다.
        invokes = (ctypes.c_uint64.in_dll(lib, "pim_invoke_count").value
                   if self.pim_target == "aimx" else 0)

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
    # 재는 것은 host 준비와 layout, h2d, kernel, d2h, reduction 전체다. tile 크기를 바꾸면 전송량도
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

    # PIM은 자동 탐지 대상이 아니다. 항상 명시 선택한다.
    #
    #   손으로 띄울 때: triton.runtime.driver.set_active(PIMDriver())
    #   인덕터 경로:    torch_backend.runtime_compat.set_driver_to_backend("pim")
    #                   — compile_backend == "pim"인 template에서만 불린다.
    #
    # True를 돌려주면 안 되는 이유: set_driver_to_gpu()는 backends를 훑어
    # is_active()가 True인 것을 스스로 고른다. 그러면 PIM을 쓰지 않는 다른
    # torch.compile이나 다른 Triton 커널까지 PIM driver로 끌려간다.
    @staticmethod
    def is_active():
        return False

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
