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
import hashlib
import functools
import os
import re
import json
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


def _run_translate(mlir_text: str, which: str) -> str:
    # triton-shared-translate로 UPMEM IR을 host/device 소스로 번역한다.
    tr = _translate_path()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.mlir")
        Path(src).write_text(mlir_text)
        return subprocess.check_output([tr, which, src]).decode("utf-8")


# TTSharedIR → PIM Common IR 파이프라인 (Design §20). 각 pass를 한 줄씩 두어, 실험 시 필요 없는
# 단계는 그 줄을 주석(#) 처리해 껐다 켰다 할 수 있다. 순서 의미는 각 줄 주석 참고.
# SPLIT/local_lanes(유닛 분배)와 fusion은 target 관심사라 이 단계에서 하지 않는다.
PIM_COMMON_PASSES = [
    "--convert-ttshared-to-pim",     # 메모리 접근을 pim.access/load/store로 변환 (Stage A)
    "-cse",                          # Pure한 pim.access(와 중복 상수) 병합 → "같은 descriptor"를 SSA 동일성으로 판별(remove-buffers 선행 조건)
    "--pim-remove-buffers",          # passthrough→pim.copy, 반복 load/RAW 재사용 (Stage B)
    "-canonicalize",                 # 죽은 pim.load/연산(결과 미사용) 정리(DCE)
    "-loop-invariant-code-motion",   # 루프 불변 연산(순수 linalg)을 루프 밖으로
    "--pim-analyze-logical-work",    # program ID↔access offset 의존 분석(execution_model/program_dims/role/invariant), classify 앞·구조 변형 뒤
    "--pim-classify-compute",        # 연산 종류 태그(pim.class). 모든 구조 변형 뒤 맨 마지막(late)에 돌려 태그가 안 낡게
]


def _ttsharedir_to_pim_common(ttshared: str) -> str:
    # 파이프라인 구성과 각 단계 의미는 PIM_COMMON_PASSES 정의 참고(줄 주석으로 토글).
    return _run_opt(ttshared, PIM_COMMON_PASSES)


def _pim_common_to_upmem(pim_common: str) -> str:
    return _run_opt(pim_common, ["--legalize-pim-to-upmem"])


def _upmem_to_artifact(upmem_ir: str, metadata) -> bytes:
    # UPMEM IR → host.cpp / device.c 소스를 생성해 번들로 묶는다. 컴파일은 launcher가 한다.
    # local_lanes(=NR_TASKLETS)는 device 빌드 파라미터라 번들에 넣지 않는다 — PIMOptions 필드라
    # 자동으로 metadata에 포함되고, launcher가 metadata.local_lanes를 읽어 -DNR_TASKLETS로 쓴다.
    host_src = _run_translate(upmem_ir, "--upmem-to-host")
    device_src = _run_translate(upmem_ir, "--upmem-to-device")
    # launcher가 호출할 host 진입 함수 이름을 metadata.name으로 넘긴다.
    # (Triton은 name을 기본 설정하지 않으므로 backend가 채워야 한다 — CPU의 _llir_to_bin과 동일.)
    match = re.search(r'extern "C" void (\w+)\(', host_src)
    if not match:
        raise RuntimeError("generated host.cpp에서 진입 함수를 찾지 못함")
    metadata["name"] = match.group(1)
    return json.dumps({"host": host_src, "device": device_src}).encode("utf-8")


# ── AiMX target 경로 (UPMEM helper와 평행) ───────────────────────────────────
# 아래 두 함수는 _pim_common_to_upmem / _upmem_to_artifact의 AiMX 쌍둥이다. 구조·시그니처는
# 동일하고 target 고유 문자열(pass 이름·translate 이름·번들 키)만 다르다. 호출하는 배관
# (_run_opt / _run_translate)은 target-neutral이라 그대로 재사용한다.
#
# 지금은 경로(seam)만 만들어 둔 상태다. AiMX legalize pass(--legalize-pim-to-aimx)와
# translate(--aimx-to-*)는 아직 triton-shared-opt/-translate에 없으므로, target="aimx"로
# 실행하면 이 함수들이 호출하는 subprocess가 그 자리에서 fail한다(의도된 fail-fast, fallback 없음).
# 실제 C++ 구현(dialect/legalize/codegen)은 별도 서버/브랜치에서 채운다(IMPLEMENT B1.2/B1.4).

def _pim_common_to_aimx(pim_common: str) -> str:
    # 같은 target-neutral PIM Common IR을 AiMX IR로 legalize한다 (Conversion/PIMToAiMX, B1.2).
    return _run_opt(pim_common, ["--legalize-pim-to-aimx"])


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
    target: str = None           # "upmem" | "aimx" — 누락 시 아래에서 즉시 에러 (fallback 없음)
    device_id: int = 0
    split_m: int = 1             # M축 inter-unit SPLIT 수. split_m×split_n×split_k = DPU 개수.
    split_n: int = 1             # N축 inter-unit SPLIT 수. 1이면 해당 축 미분배(기존 동작).
    split_k: int = 1             # K축 inter-unit SPLIT 수. >1이면 한 출력 타일을 여러 DPU가 K분담 → host reduce.
    local_lanes: int = 1         # DPU 내부 intra-unit 병렬 수(=NR_TASKLETS). 1이면 단일 tasklet.

    def __post_init__(self):
        # target 없이 PIM을 쓰는 것은 의도된 사용이 아니므로 기본값으로 때우지 않고 바로 터뜨린다.
        if self.target is None:
            raise ValueError(
                "PIMBackend requires an explicit `target` option. "
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
        stages["ttsharedir"] = lambda src, metadata: cpu_compiler._ttir_to_ttsharedir(src)
        stages["pim_common"] = lambda src, metadata: _ttsharedir_to_pim_common(src)
        # target 분기: legalize(target_ir)와 codegen(pimobj)만 target마다 다르다(DESIGN 19).
        # __post_init__은 target is None만 막으므로, 미지원 값은 여기서 fail-fast한다(fallback 금지).
        if options.target == "upmem":
            stages["target_ir"] = lambda src, metadata: _pim_common_to_upmem(src)
            stages["pimobj"] = lambda src, metadata: _upmem_to_artifact(src, metadata)
        elif options.target == "aimx":
            stages["target_ir"] = lambda src, metadata: _pim_common_to_aimx(src)
            stages["pimobj"] = lambda src, metadata: _aimx_to_artifact(src, metadata)
        else:
            raise ValueError(
                f"Unsupported PIM target '{options.target}'. Supported: 'upmem', 'aimx'.")

    @functools.lru_cache()
    def hash(self):
        return self.target

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
