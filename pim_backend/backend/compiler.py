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
import hashlib
import functools

# CPU backend의 IR 변환 helper / make_ttir을 재사용한다 (클래스가 아닌 모듈로 참조).
import triton.backends.triton_shared.compiler as cpu_compiler

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
    vendor: str = None           # "upmem" | "aimx" — 누락 시 아래에서 즉시 에러 (fallback 없음)
    device_id: int = 0

    def __post_init__(self):
        # vendor 없이 PIM을 쓰는 것은 의도된 사용이 아니므로 기본값으로 때우지 않고 바로 터뜨린다.
        if self.vendor is None:
            raise ValueError(
                "PIMBackend requires an explicit `vendor` option. "
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
        # CPU lowering 4-stage를 그대로 차용한다.
        # 마지막 stage 이름은 binary_ext('pimobj')와 일치해야 CompiledKernel이 kernel을 찾는다.
        stages["ttir"] = lambda src, metadata: cpu_compiler.CPUBackend.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: cpu_compiler._optimize_ttsharedir(cpu_compiler._ttir_to_ttsharedir(src))
        stages["llir"] = lambda src, metadata: cpu_compiler._optimize_llir(cpu_compiler._ttsharedir_to_llir(src))
        stages["pimobj"] = lambda src, metadata: cpu_compiler._llir_to_bin(src, metadata)

    @functools.lru_cache()
    def hash(self):
        return self.target

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
