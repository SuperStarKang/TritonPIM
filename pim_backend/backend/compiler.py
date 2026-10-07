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

from .target import (
    PIMTargetProfile,
    PIMTargetSpec,
    aimx_target_spec,
    upmem_target_spec,
)


def _translate_path() -> str:
    # triton-shared-translate는 triton-shared-opt와 같은 빌드 트리의 형제 도구다.
    # 별도 env 없이 기존 opt 경로(TRITON_SHARED_OPT_PATH)에서 이름만 바꿔 유도한다.
    opt = cpu_compiler._get_triton_shared_opt_path()
    return opt.replace("triton-shared-opt", "triton-shared-translate")


def _run_opt(mlir_text: str, passes: list) -> str:
    # triton-shared-opt로 변환 pass를 적용한다. 어떤 pass를 돌릴지는 호출자가 정하므로
    # 이 함수는 target을 모른다.
    opt = cpu_compiler._get_triton_shared_opt_path()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.mlir")
        dst = os.path.join(tmp, "out.mlir")
        Path(src).write_text(mlir_text)
        subprocess.check_call([opt, src, *passes, "-o", dst])
        return Path(dst).read_text()


def _run_translate(mlir_text: str, which: str) -> str:
    # triton-shared-translate로 target IR을 소스로 번역한다. which(--<target>-to-*)는
    # 호출자가 정하므로 이 함수도 target을 모른다.
    tr = _translate_path()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.mlir")
        Path(src).write_text(mlir_text)
        return subprocess.check_output([tr, which, src]).decode("utf-8")


def _upmem_spec(o) -> PIMTargetSpec:
    return upmem_target_spec(
        num_units=o.num_units,
        banks_per_unit=o.banks_per_unit,
        cores_per_bank=o.cores_per_bank,
        lanes_per_core=o.lanes_per_core,
        bank_capacity_limit_bytes=o.bank_capacity_limit_bytes,
        local_capacity_limit_bytes=o.local_capacity_limit_bytes,
        nr_tasklets=o.nr_tasklets,
        host_threads=o.host_threads,
    )


def _aimx_spec(o) -> PIMTargetSpec:
    # PIM Device = FPGA 1장. 16 channel x 16 bank를 256개 bank로 평탄화하고,
    # 각 bank의 16 ALU를 한 core의 lane으로 표현한다. 한 block group이 이 bank
    # 전체에 걸쳐 분할되며 별개의 block group을 bank마다 배정하는 뜻이 아니다.
    # [batch, 4096/batch, 256/batch] 물리 눈금은 profile이 아니라 AiMX target
    # lowering에서 검증하고 padding한다.
    # device 분할 [p_m,p_n,p_k]는 Common IR 옵션에서 그대로 온다. bank 예산은
    # 선언 상수이며(pim_malloc은 실패해도 NULL 대신 프로세스를 종료한다), target
    # lowering도 이 공통 profile 값 하나만 읽는다.
    return aimx_target_spec(
        num_units=o.num_units,
        banks_per_unit=o.banks_per_unit,
        cores_per_bank=o.cores_per_bank,
        lanes_per_core=o.lanes_per_core,
        bank_capacity_limit_bytes=o.bank_capacity_limit_bytes,
        local_capacity_limit_bytes=o.local_capacity_limit_bytes,
        batch=o.batch,
        base_fpga=o.base_fpga,
    )


# target 고유 값은 backend/target.py의 builder에 있다.
_TARGET_SPEC = {"upmem": _upmem_spec, "aimx": _aimx_spec}


def _axis_options(prefix: str, m: int, n: int, k: int) -> str:
    # group-m/n/k, tile-m/n/k 를 축별로 적는다. 0 = 한 단위 = pass 기본값이라 적지 않는다.
    return " ".join(s for s in (
        f"{prefix}-m={m}" if m else "",
        f"{prefix}-n={n}" if n else "",
        f"{prefix}-k={k}" if k else "") if s)


# TTSharedIR → PIM Common IR 파이프라인. 각 pass 를 한 줄씩 두어 실험 시 그 줄을 주석 처리해
# 껐다 켰다 할 수 있다.
PIM_COMMON_PASSES = [
    "--pim-normalize-address",     # 주소 재귀를 닫힌 형태로. 이것 없이는 뒤의 origin 복원이 전부 실패한다
    "-cse",
    "-loop-invariant-code-motion",
    "-canonicalize",               # (iv - 0) / 1 같은 잉여 산술을 상수로 확정한다
    "--pim-form-compute",          # ① linalg와 accumulator에서 compute/reduction 의미 확정
    "--convert-ttshared-to-pim",   # 메모리 접근을 pim.access/load/store 로
    "--pim-build-logical-grid",    # source grid 를 logical tile loop nest 로
    "--pim-recover-access-origin", # access 의 축별 origin 을 logical 좌표로 채운다
    "{profile}",                   # 기계 profile 부착. 여기부터 profile 을 읽는다
    "{form_block_groups}",         # 계층 2: 사용자 묶음(group-m/n/k) → #pim.block_group
    "{form_execution_tiles}",      # 계층 3: 주어진 Block Group 개수 → #pim.execution_tile
    "{map_devices}",               # execution tile grid 를 num_units 이하 active PIM Device grid 에 scheduling
    "--pim-place-data",            # 최종 region·replication·allocation shape materialize
    "--pim-verify",                # 공통 fail-fast 검증
    "--pim-reuse-execution-tile-inputs",  # Group 크기를 유지하며 Tile 내부 입력 load 재사용
    "--pim-verify",                # 이동 뒤에도 hierarchy/placement 계약 유지
    "--pim-form-host-plan",        # 논리 분할·입력 전송·계산·출력 회수를 명시
    "--pim-verify",
    "-cse",
]


def _profile_pass(t: PIMTargetProfile) -> str:
    return ("--pim-set-target-profile="
            f"target={t.target} "
            f"num-units={t.num_units} "
            f"banks-per-unit={t.banks_per_unit} "
            f"cores-per-bank={t.cores_per_bank} "
            f"lanes-per-core={t.lanes_per_core} "
            f"bank-capacity-limit-bytes={t.bank_capacity_limit_bytes} "
            f"local-capacity-limit-bytes={t.local_capacity_limit_bytes}")


def _ttsharedir_to_pim_common(ttshared: str, options) -> str:
    spec = _TARGET_SPEC[options.pim_target](options)
    profile = _profile_pass(spec.common_profile)
    map_devices = ("--pim-map-devices="
                   f"p-m={options.p_m} p-n={options.p_n} p-k={options.p_k}")
    bg = _axis_options("group", options.group_m, options.group_n, options.group_k)
    form_block_groups = "--pim-form-block-groups" + (f"={bg}" if bg else "")
    # Common pass의 tile-m/n/k는 Execution Tile에 묶는 Block Group 개수다.
    # lowering은 이 값을 그대로 materialize하고 비용식이나 용량 최대화를 하지 않는다.
    et = _axis_options("tile", options.tile_m, options.tile_n, options.tile_k)
    form_execution_tiles = "--pim-form-execution-tiles" + (f"={et}" if et else "")
    passes = [p.format(profile=profile,
                       form_block_groups=form_block_groups,
                       map_devices=map_devices,
                       form_execution_tiles=form_execution_tiles)
              for p in PIM_COMMON_PASSES]
    # AiMX consumes the annotated loop/combine structure. General SCF
    # canonicalization can erase single-trip loops and their axis metadata.
    # Its target entry folds scalar arithmetic without changing those loops.
    if options.pim_target != "aimx":
        passes.append("-canonicalize")
    return _run_opt(ttshared, passes)


def _pim_common_to_upmem(pim_common: str, options) -> str:
    # 기계 값은 전부 profile 에서, 시간 창은 공통 IR 의 #pim.execution_tile 에서 읽는다.
    spec = _TARGET_SPEC["upmem"](options)
    runtime = spec.runtime_config
    return _run_opt(
        pim_common,
        [f"--convert-pim-to-upmem=host-threads={runtime.host_threads} "
         f"nr-tasklets={runtime.nr_tasklets}"])


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
    # 실제로 쓰는 tasklet 수. UPMEM lowering이 nr_tasklets/profile에서 확정한 값이며,
    # launcher는 이 값을 -DNR_TASKLETS로 써야 IR과 device 빌드가 맞는다.
    nr_tasklets = re.search(r'// pim\.lanes = (\d+)', device_src)
    if not nr_tasklets:
        raise RuntimeError("generated device.c에서 tasklet 수를 찾지 못함")
    return json.dumps({"host": host_src, "device": device_src,
                       "target_ir": upmem_ir,
                       "nr_tasklets": int(nr_tasklets.group(1))}).encode("utf-8")


# ── AiMX target 경로 (UPMEM helper와 평행) ───────────────────────────────────
# 아래 두 함수는 _pim_common_to_upmem / _upmem_to_artifact의 AiMX 쌍둥이다. 구조·시그니처는
# 동일하고 target 고유 문자열(pass 이름·translate 이름·번들 키)만 다르다. 호출하는 배관
# (_run_opt / _run_translate)은 target-neutral이라 그대로 재사용한다.

# One target pass lowers Common hierarchy and SSA directly to descriptors.
def _aimx_passes(options):
    # 3계층은 공통 pass 가 이미 IR 에 새겼다 — block group 은 #pim.block_group 으로,
    # execution tile 상한은 #pim.execution_tile로 전달된다. batch는 PIM profile에
    # 넣지 않고 AiMX target option으로만 전달한다.
    tile = (f"--convert-pim-to-aimx=batch={options.batch} "
            f"persistent={str(options.aimx_persistent).lower()} "
            f"matvec-row-alignment={options.aimx_matvec_row_alignment} "
            f"matvec-first-n-only={str(options.aimx_matvec_first_n_only).lower()}")
    return [tile, "-cse", "-canonicalize"]


def _pim_common_to_aimx(pim_common: str, options) -> str:
    # Serialize the target IR in generic form for both translators. The pinned
    # SCF custom printer emits no_inline as both a keyword and an attribute,
    # which its parser rejects. Keep the attribute and descriptor boundary;
    # only the textual representation changes.
    return _run_opt(pim_common, _aimx_passes(options) + ["--mlir-print-op-generic"])


def _aimx_to_artifact(aimx_ir: str, metadata, options) -> bytes:
    # AiMX IR → runtime stub(C++) + descriptor program(C 헤더)을 생성해 번들로 묶는다.
    # upmem과 동일하게, launcher가 호출할 진입 함수 이름을 stub에서 파싱해 metadata["name"]에 채운다.
    stub_src = _run_translate(aimx_ir, "--aimx-to-runtime-stub")
    descriptors = _run_translate(aimx_ir, "--aimx-to-descriptors")
    match = re.search(r'extern "C" void (\w+)\(', stub_src)
    if not match:
        raise RuntimeError("generated AiMX stub에서 진입 함수를 찾지 못함")
    metadata["name"] = match.group(1)
    allocations = [
        [int(size.strip()) for size in sizes.split(",")]
        for sizes in re.findall(r'aimx\.allocations\s*=\s*array<i64:\s*([^>]+)>', aimx_ir)
    ]
    if not allocations:
        raise RuntimeError("AiMX target IR에 물리 allocation 계획이 없습니다")
    metadata["aimx_allocations"] = allocations
    # base_fpga는 컴파일이 아니라 빌드 때 -D로 들어간다. launcher가 읽도록 번들에 싣는다.
    return json.dumps({"stub": stub_src, "descriptors": descriptors,
                       "base_fpga": options.base_fpga,
                       "allocations": allocations}).encode("utf-8")


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

    pim_target: str = None       # "upmem" | "aimx"

    # #devices (#dpu | #fpga)
    num_units: int = 0
    banks_per_unit: int = 0
    cores_per_bank: int = 0
    # block group: local 상주 단위로 묶는 tile 수. group_m × group_n 이 출력 묶음이고
    # group_k 는 reduction strip(적재 한 번이 담는 K tile 수 — 누산기는 그대로다).
    # 0 또는 1 = Block 하나인 항등 group. 값을 주면 그대로 쓴다.
    # 두 target 모두 공통 pass(--pim-form-block-groups)가 IR 수준에서 묶는다 —
    # 지나고 나면 target lowering 이 보는 "block" 이 곧 group 이다.
    group_m: int = 0
    group_n: int = 0
    group_k: int = 0
    # ③ execution tile: device 하나가 launch 1회에 상주시키는 Block Group 개수.
    # 0 또는 1 = 존재하는 축에서 Group 하나. 공통 lowering은 값을 그대로
    # #pim.execution_tile에 기록하고 PIM DRAM을 채우기 위한 탐색은 하지 않는다.
    # target lowering 은 읽기만 한다:
    #   upmem  Common lowering이 만든 round/place/combine을 stage/launch/gather와
    #          unstage accumulate로 그대로 옮긴다. target이 K를 다시 자르지 않는다.
    #   aimx   상한이 담는 block group 수 × 물리 group 크기가 m_t·n_max·k_max.
    #          k 는 host 누산 창이라 split-K kernel 이 필요 없다.
    tile_m: int = 0
    tile_n: int = 0
    tile_k: int = 0
    # 공통 hardware profile. capacity 0은 target 기본값을 사용한다.
    lanes_per_core: int = 16
    bank_capacity_limit_bytes: int = 0
    local_capacity_limit_bytes: int = 0
    # Device mapping은 고정 순서 [p_m,p_n,p_k]의 세 정수뿐이다. 없는 축은 1.
    p_m: int = 1
    p_n: int = 1
    p_k: int = 1

    # ── UPMEM 전용 ─────────────────────────────────────────────────────────
    # 0이면 profile의 lanes_per_core를 모두 사용한다.
    nr_tasklets: int = 0
    host_threads: int = 8

    # ── aimx 전용 ───────────────────────────────────────────────────────────
    # 몇 번 카드부터 쓰는가. 런타임은 fpga_id로 pl_dma[id]를 경계 검사 없이 인덱싱하므로
    # [base_fpga, base_fpga + num_units)가 [0, 감지된 개수) 안이어야 한다.
    # aimx_runtime_init()은 전체 장치를 한 번 초기화하고 선택한 실행 범위를 대조한다.
    base_fpga: int = 0
    # 출력 block 을 batch x (4096/batch)로 자르는 값, {1,2,4,8}만 legal — 이것이 곧
    # arithmetic intensity 다. 실행 단위 [batch, 4096/batch, 256/batch]가 여기서
    # 나온다. BLOCK 이 이 눈금의 배수가 아니면 padding 이고, 배수면 padding 이 없다 —
    # 공통 pass 가 remark 로 노는 비율을 알려 준다.
    batch: int = 8
    # AiMX-only policy, not a change to Common IR arithmetic semantics.
    aimx_persistent: bool = True
    # Matvec storage pitch is independent of the 4096/batch-column MAC microtile.
    # Explicit 0/512 alignments retain the narrow-layout diagnostic paths.
    aimx_matvec_row_alignment: int = 4096
    # False retains the padded-N diagnostic; GEMM traversal is unaffected.
    aimx_matvec_first_n_only: bool = True
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

    # 옵션은 전달받은 것만 쓴다. env 통로는 두지 않는다.
    #
    # env를 없앤 이유:
    #   - env는 프로세스 단위라 커널마다 다른 실행 계획을 줄 수 없다. 한 그래프에
    #     _int_mm이 둘 이상 있으면(예: attention의 QK / PV) 둘의 shape가 달라
    #     서로 다른 tile 이 필요한데, env로는 같은 값 하나밖에 못 준다.
    #   - 설정이 어디서 왔는지 추적이 끊긴다. 실패한 실행을 재현하려면 그때의
    #     환경 전체를 알아야 한다.
    #
    # 인덕터 경로의 통로는 이제 template metadata다. torch_backend의
    # runtime_compat이 triton_meta["backend_options"]를 읽어 triton.compile
    # 옵션에 합쳐 준다. 손으로 띄우는 경로는 종전대로 launch kwarg를 쓴다.
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
        stages["pim_common"] = lambda src, metadata: _ttsharedir_to_pim_common(src, options)
        # target 분기: 변환(target_ir)과 codegen(pimobj)만 target마다 다르다(DESIGN 19).
        # __post_init__은 pim_target is None만 막으므로, 미지원 값은 여기서 fail-fast한다(fallback 금지).
        if options.pim_target == "upmem":
            stages["target_ir"] = lambda src, metadata: _pim_common_to_upmem(src, options)
            stages["pimobj"] = lambda src, metadata: _upmem_to_artifact(src, metadata)
        elif options.pim_target == "aimx":
            stages["target_ir"] = lambda src, metadata: _pim_common_to_aimx(src, options)
            stages["pimobj"] = lambda src, metadata: _aimx_to_artifact(src, metadata, options)
        else:
            raise ValueError(
                f"Unsupported PIM target '{options.pim_target}'. Supported: 'upmem', 'aimx'.")

    @functools.lru_cache()
    def hash(self):
        # Regenerate artifacts when a target's execution or allocation plan changes.
        return f"{self.target}-pim-common-ir-v9-device-mapping-host-pools-host-plan-shared-over-input-reuse-upmem-metadata-v2-rhs-row-major-v1-timing-v3-aimx-contiguous-groups-v2-vacc-reduction-v2"

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
