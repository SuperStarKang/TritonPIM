# PIMBackend — Triton에서 device="pim"으로 선택되는 backend.
#
# 이 디렉토리는 nvidia/amd/triton_shared(=CPU)와 동일하게 "backend 1개 = 디렉토리 1개"
# 규약을 따른다. Triton의 backend discovery(_find_concrete_subclasses)는 한 compiler.py에
# BaseBackend 서브클래스가 정확히 하나만 있을 것을 요구한다.
#
# 주의: CPUBackend 같은 다른 BaseBackend 서브클래스를 이 모듈의 top-level 이름으로
# import하면 서브클래스가 2개로 세어져 discovery가 실패한다. 그래서 CPU 쪽 코드는
# 모듈(cpu_compiler)로만 참조하고, 클래스 이름을 이 네임스페이스로 끌어오지 않는다.
#
# 현재 lowering은 CPU lowering 4-stage를 그대로 재사용한다 (device="pim" 라우팅 검증용).
#
# TTShared 생성부(_ttir_to_ttsharedir)를 복사하지 않고 import하는 이유:
# CPU·PIM·hybrid가 동일한 TTSharedIR을 만들어야 하며(특히 hybrid는 같은 IR을 grid 분할해
# 나눠 실행), 복사본은 시간이 지나며 CPU 쪽과 drift날 수 있다. import는 단일 소스를 보장한다.
#

from triton.backends.compiler import BaseBackend, GPUTarget
from dataclasses import dataclass
from typing import Tuple, Dict
from types import ModuleType
from pathlib import Path
import contextlib
import hashlib
import functools
import os
import re
import json
import sys
import tempfile
import subprocess

# CPU backend의 IR 변환 helper / make_ttir을 재사용한다 (클래스가 아닌 모듈로 참조).
import triton.backends.triton_shared.compiler as cpu_compiler


def _translate_path() -> str:
    # triton-shared-translate는 triton-shared-opt와 같은 빌드 트리의 형제 도구다.
    # 별도 env 없이 기존 opt 경로(TRITON_SHARED_OPT_PATH)에서 이름만 바꿔 유도한다.
    opt = cpu_compiler._get_triton_shared_opt_path()
    return opt.replace("triton-shared-opt", "triton-shared-translate")


def _run_opt(mlir_text: str, passes: list) -> str:
    # triton-shared-opt로 PIM/UPMEM 변환 pass를 적용한다.
    opt = cpu_compiler._get_triton_shared_opt_path()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.mlir")
        dst = os.path.join(tmp, "out.mlir")
        Path(src).write_text(mlir_text)
        subprocess.check_call([opt, src, *passes, "-o", dst])
        return Path(dst).read_text()


# ── 경고 감추기 (지우기 쉽게 여기에 모아 둔다) ──────────────────────────────────
# 지우려면: 아래 블록(_KNOWN_WARNING ~ _ttir_to_ttsharedir)을 통째로 지우고, add_stages의
# ttsharedir 줄을 cpu_compiler._ttir_to_ttsharedir(src)로 되돌리고, import contextlib / sys를
# 지운다. 컴파일 결과에는 영향이 없다 — stderr만 걸러 낸다.
#
# TritonToStructured의 prepass는 loop iter-arg인 정수 tensor마다 tts.get_structured_state를
# 넣어 보고, 그것이 주소 계산에 안 쓰이면 canonicalize가 되돌린다(TritonToStructuredPass.cpp 주석).
# 누산기 tensor가 정확히 그 경우라 정수 GEMM마다 이 경고가 뜬다. 되돌리기가 정상 동작이므로
# 이 경고 하나만 감춘다 — 다른 진단이 하나라도 섞이면 통째로 내보낸다.
_KNOWN_WARNING = "Rewriting GetStructuredStateOp failed."


def _noise_only(text: str) -> bool:
    for line in text.splitlines():
        if ("error:" in line or "remark:" in line
                or ("warning:" in line and _KNOWN_WARNING not in line)):
            return False
    return True


@contextlib.contextmanager
def _filtered_stderr():
    # triton-shared-opt은 subprocess라 stderr가 터미널로 곧장 나간다. Python 층의
    # redirect_stderr로는 못 잡으므로 파일 서술자를 바꾼다. finally라 실패해도 되돌아오고,
    # 그때는 감출 수 없는 진단이 섞여 있으므로 그대로 보인다.
    saved = os.dup(2)
    buf = tempfile.TemporaryFile()
    os.dup2(buf.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        buf.seek(0)
        text = buf.read().decode("utf-8", "replace")
        buf.close()
        if text and not _noise_only(text):
            sys.stderr.write(text)


def _ttir_to_ttsharedir(ttir) -> str:
    with _filtered_stderr():
        return cpu_compiler._ttir_to_ttsharedir(ttir)
# ── 경고 감추기 끝 ─────────────────────────────────────────────────────────────


def _run_translate(mlir_text: str, which: str) -> str:
    # triton-shared-translate로 UPMEM IR을 host/device 소스로 번역한다.
    tr = _translate_path()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.mlir")
        Path(src).write_text(mlir_text)
        return subprocess.check_output([tr, which, src]).decode("utf-8")


# target 마다 고정된 기계 값. 나머지는 PIMOptions 에서 온다.
# bank/local 용량이 음수면 미측정이라는 뜻이라 profile 에 필드가 생기지 않고, 그 값을
# 필요로 하는 검사에 도달하는 순간 진단이 된다.
_TARGET_PROFILE = {
    "upmem": {
        "pim_core_shape": "1",
        "parallel_execution_kind": "multithread",
        "uniform_instruction_sequence": "false",
        "devices_per_transfer_group": 64,
    },
    "aimx": {
        "pim_core_shape": "8,16",
        "parallel_execution_kind": "simd",
        "uniform_instruction_sequence": "true",
        "devices_per_transfer_group": 0,
    },
}


# TTSharedIR → Placed PIM IR 파이프라인. 각 pass 를 한 줄씩 두어 실험 시 그 줄을 주석 처리해
# 껐다 켰다 할 수 있다.
PIM_COMMON_PASSES = [
    "--pim-normalize-address",     # 주소 재귀를 닫힌 형태로. 이것 없이는 뒤의 origin 복원이 전부 실패한다
    "-cse",
    "-loop-invariant-code-motion",
    "-canonicalize",               # (iv - 0) / 1 같은 잉여 산술을 상수로 확정한다
    "--convert-ttshared-to-pim",   # 메모리 접근을 pim.access/load/store 로
    "--pim-build-logical-grid",    # source grid 를 logical tile loop nest 로
    "--pim-recover-access-origin", # access 의 축별 origin 을 logical 좌표로 채운다
    "--pim-form-compute",          # 누산기를 matmul 의 outs 로 통합
    "{profile}",                   # 기계 profile 부착. 여기부터 profile 을 읽는다
    "--pim-map-cores",             # logical 차원 ↔ PIM Core topology, lane 수 선택
    "--pim-form-execution-waves",  # core footprint 로 wave 구성
    "--pim-map-devices",           # wave 를 PIM Device 에 균형 분배
    "--pim-place-data",            # 최종 region·replication·capacity materialize
    "--pim-verify",                # 공통 fail-fast 검증
    "-cse",
    "-canonicalize",
]


def _profile_pass(options) -> str:
    t = _TARGET_PROFILE[options.pim_target]
    bank = options.unit_memory_bytes if options.pim_target == "upmem" else -1
    local = options.local_memory_bytes if options.pim_target == "upmem" else 0
    return ("--pim-set-target-profile="
            f"target={options.pim_target} "
            f"number-of-pim-devices={options.num_units} "
            f"pim-core-shape={t['pim_core_shape']} "
            f"number-of-parallel-lanes-per-pim-core={options.local_lanes} "
            f"parallel-execution-kind={t['parallel_execution_kind']} "
            f"requires-uniform-instruction-sequence={t['uniform_instruction_sequence']} "
            f"usable-bank-capacity-bytes={bank} "
            f"usable-local-capacity-bytes={local} "
            f"devices-per-transfer-group={t['devices_per_transfer_group']}")


def _ttsharedir_to_pim_common(ttshared: str, options) -> str:
    profile = _profile_pass(options)
    return _run_opt(ttshared, [p.format(profile=profile) for p in PIM_COMMON_PASSES])


def _pim_common_to_upmem(pim_common: str, options) -> str:
    # 기계 값은 전부 profile 에서 읽는다. host 스레드 수만 lowering policy 라 옵션으로 준다.
    return _run_opt(pim_common,
                    [f"--convert-pim-to-upmem=host-threads={options.host_threads}"])


def _upmem_to_artifact(upmem_ir: str, metadata) -> bytes:
    # UPMEM IR → host.cpp / device.c 소스를 생성해 번들로 묶는다. 컴파일은 launcher가 한다.
    host_src = _run_translate(upmem_ir, "--upmem-to-host")
    device_src = _run_translate(upmem_ir, "--upmem-to-device")
    # launcher가 호출할 host 진입 함수 이름을 metadata.name으로 넘긴다.
    # (Triton은 name을 기본 설정하지 않으므로 backend가 채워야 한다 — CPU의 _llir_to_bin과 동일.)
    match = re.search(r'extern "C" void (\w+)\(', host_src)
    if not match:
        raise RuntimeError("generated host.cpp에서 진입 함수를 찾지 못함")
    metadata["name"] = match.group(1)
    # 실제로 쓰는 tasklet 수. --pim-map-cores가 lane 예산 안에서 고른 값이라 요청값과
    # 다를 수 있고, launcher는 이 값을 -DNR_TASKLETS로 써야 IR과 device 빌드가 맞는다.
    lanes = re.search(r'// pim\.lanes = (\d+)', device_src)
    if not lanes:
        raise RuntimeError("generated device.c에서 tasklet 수를 찾지 못함")
    return json.dumps({"host": host_src, "device": device_src,
                       "lanes": int(lanes.group(1))}).encode("utf-8")


# ── AiMX target 경로 (UPMEM helper와 평행) ───────────────────────────────────
# 아래 두 함수는 _pim_common_to_upmem / _upmem_to_artifact의 AiMX 쌍둥이다. 구조·시그니처는
# 동일하고 target 고유 문자열(pass 이름·translate 이름·번들 키)만 다르다. 호출하는 배관
# (_run_opt / _run_translate)은 target-neutral이라 그대로 재사용한다.
#
# 지금은 경로(seam)만 만들어 둔 상태다. AiMX 변환 pass(--convert-pim-to-aimx)와
# translate(--aimx-to-*)는 아직 triton-shared-opt/-translate에 없으므로, target="aimx"로
# 실행하면 이 함수들이 호출하는 subprocess가 그 자리에서 fail한다(의도된 fail-fast, fallback 없음).
# 실제 C++ 구현(dialect/legalize/codegen)은 별도 서버/브랜치에서 채운다(IMPLEMENT B1.2/B1.4).

def _pim_common_to_aimx(pim_common: str) -> str:
    # 같은 target-neutral PIM IR을 AiMX IR로 변환한다 (Conversion/PIMToAiMX, B1.2).
    return _run_opt(pim_common, ["--convert-pim-to-aimx"])


def _aimx_to_artifact(aimx_ir: str, metadata) -> bytes:
    # AiMX IR → runtime stub(C++) + descriptor stream을 생성해 번들로 묶는다 (Target/AiMX, B1.4).
    # upmem과 동일하게, launcher가 호출할 진입 함수 이름을 stub에서 파싱해 metadata["name"]에 채운다.
    # (runtime stub은 extern "C" void <name>(...) 진입점을 emit해야 이 파싱이 성립 — 평행 구조의 계약.)
    # 주의: descriptor가 binary 포맷으로 확정되면 utf-8 디코드하는 _run_translate로는 못 받으므로,
    #       그때 binary용 translate 헬퍼가 필요하다(포맷 확정 전이라 지금은 텍스트로 둔다, B1.4).
    stub_src = _run_translate(aimx_ir, "--aimx-to-runtime-stub")
    descriptors = _run_translate(aimx_ir, "--aimx-to-descriptors")
    match = re.search(r'extern "C" void (\w+)\(', stub_src)
    if not match:
        raise RuntimeError("generated AiMX stub에서 진입 함수를 찾지 못함")
    metadata["name"] = match.group(1)
    return json.dumps({"stub": stub_src, "descriptors": descriptors}).encode("utf-8")


@dataclass(frozen=True)
class PIMOptions:
    # --- Triton 표준 옵션 (CPUOptions와 동일하게 유지) ---
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None  # 의도적으로 annotation 없음 → dataclass field가 아닌 클래스 속성 (CPUOptions와 동일)
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    sanitize_overflow: bool = True

    # --- PIM 고유 옵션 ---
    # 이름을 target으로 두면 안 된다. Triton이 metadata를 {"target": GPUTarget, **options.__dict__}
    # 로 만들어서 같은 이름의 옵션이 GPUTarget을 덮어쓰고, CompiledKernel이 그것을 다시 읽는다.
    pim_target: str = None       # "upmem" | "aimx" — 누락 시 아래에서 즉시 에러 (fallback 없음)
    device_id: int = 0
    # 기계 — 이름을 target 중립으로 둔다(UPMEM에서는 유닛=DPU, 빠른 메모리=WRAM).
    num_units: int = 2048        # 물리 유닛 개수. split의 곱이 이보다 크면 컴파일 시점에 실패.
    # 유닛당 빠른 메모리. tile 버퍼 합계 × local_lanes가 넘으면 실패한다. UPMEM의 물리 WRAM은
    # 64 KB지만 tasklet 스택과 런타임도 여기 살므로 커널이 써도 되는 몫은 그보다 작다.
    local_memory_bytes: int = 48 * 1024
    host_threads: int = 8        # host가 영역을 모으고 되돌릴 때 쓰는 스레드 수. 1이면 순차.
    # 유닛당 큰 메모리. target이 버퍼 크기를 여기서 나눈다. UPMEM의 물리 MRAM은 64 MB지만
    # DPU 런타임도 MRAM에 자리를 잡으므로 전부를 나눠 주면 링커가 넘친다. 커널이 써도 되는
    # 몫이라는 뜻이라 물리 크기보다 작게 둔다.
    unit_memory_bytes: int = 32 * 1024 * 1024
    local_lanes: int = 1         # DPU 내부 intra-unit 병렬 lane 능력값. 실제 값은 lane 예산이 정한다.

    def __post_init__(self):
        # target 없이 PIM을 쓰는 것은 의도된 사용이 아니므로 기본값으로 때우지 않고 바로 터뜨린다.
        if self.pim_target is None:
            raise ValueError(
                "PIMBackend requires an explicit `pim_target` option. "
                "Supported values: 'upmem', 'aimx'."
            )

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class PIMBackend(BaseBackend):
    binary_ext = 'pimobj'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'pim'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> PIMOptions:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in PIMOptions.__dataclass_fields__.keys() if k in opts})
        return PIMOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # launcher가 쓰는 name 외에는 다른 backend와 형식을 맞추기 위해 그대로 채운다.
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name
        )

    # IR 변환은 triton-shared-opt subprocess가 수행하므로 Python MLIRContext에 dialect를 로드하지 않는다.
    def load_dialects(self, ctx):
        return

    def add_stages(self, stages, options, language):
        # PIM 파이프라인: TTIR → TTSharedIR → PIM Common IR → Target IR → artifact 번들.
        # ttir/ttsharedir/pim_common은 target-neutral이라 공유하고, 뒤 두 stage만 target로 분기한다.
        # 마지막 stage 이름은 binary_ext('pimobj')와 일치해야 CompiledKernel이 kernel을 찾는다.
        stages["ttir"] = lambda src, metadata: cpu_compiler.CPUBackend.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _ttir_to_ttsharedir(src)
        stages["pim_common"] = lambda src, metadata: _ttsharedir_to_pim_common(src, options)
        # target 분기: 변환(target_ir)과 codegen(pimobj)만 target마다 다르다(DESIGN 19).
        # __post_init__은 pim_target is None만 막으므로, 미지원 값은 여기서 fail-fast한다(fallback 금지).
        if options.pim_target == "upmem":
            stages["target_ir"] = lambda src, metadata: _pim_common_to_upmem(src, options)
            stages["pimobj"] = lambda src, metadata: _upmem_to_artifact(src, metadata)
        elif options.pim_target == "aimx":
            stages["target_ir"] = lambda src, metadata: _pim_common_to_aimx(src)
            stages["pimobj"] = lambda src, metadata: _aimx_to_artifact(src, metadata)
        else:
            raise ValueError(
                f"Unsupported PIM target '{options.pim_target}'. Supported: 'upmem', 'aimx'.")

    @functools.lru_cache()
    def hash(self):
        return self.target

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
