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
import glob
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
    # triton-shared-opt로 변환 pass를 적용한다. 어떤 pass를 돌릴지는 호출자가 정하므로
    # 이 함수는 target을 모른다.
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
    # triton-shared-translate로 target IR을 소스로 번역한다. which(--<target>-to-*)는
    # 호출자가 정하므로 이 함수도 target을 모른다.
    tr = _translate_path()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.mlir")
        Path(src).write_text(mlir_text)
        return subprocess.check_output([tr, which, src]).decode("utf-8")


# AiMX runtime의 FPGA별 PIM DRAM heap과 정확히 같은 예산이다.
# pim_sw/src/mem_src/mdrv_interface.c가 8 GiB에서 runtime 예약분 18 MiB를 뺀
# 0x1FEE00000 byte를 device 하나의 heap으로 고정한다. Common profile은 용량을
# bank 단위로 받으므로 이 총량을 256개 bank에 균등하게 나눈다. 32 MiB를 단순히
# 넣으면 aggregate가 runtime heap보다 18 MiB 커져 경계 config가 다시 OOM 날 수 있다.
AIMX_BANKS_PER_DEVICE = 256
AIMX_DEVICE_HEAP_BYTES = 0x1FEE00000
AIMX_BANK_CAPACITY_BYTES = AIMX_DEVICE_HEAP_BYTES // AIMX_BANKS_PER_DEVICE


def _aimx_device_count() -> int:
    # 런타임(init_mpim_drv)이 /dev/pl_dma<i> 를 i = 0..totalDevNum-1 로 연다. 그러니
    # 그 node 를 세는 것이 런타임이 보는 개수와 같은 값이다. 감지에 실패하면 1 로
    # 때우지 않고 멈춘다 — 조용히 FPGA 한 장만 쓰는 것이 가장 알아채기 어렵다.
    n = len(glob.glob("/dev/pl_dma[0-9]*"))
    if n == 0:
        raise RuntimeError(
            "/dev/pl_dma* 가 없습니다. pim_drv 가 올라와 있는지 확인해 주십시오. "
            "개수를 직접 주려면 num_units 옵션을 쓰십시오.")
    return n

# target 하나의 기계 값 — 전부 단순 숫자/문자열이다. 옵션에 의존하는 값(num_units,
# batch, 예산)은 아래 _upmem_profile/_aimx_profile 이 계산해서 채운다.
# bank/local 바이트가 음수면 미측정이라는 뜻이라 profile 에 필드가 생기지 않고, 그 값을
# 필요로 하는 검사에 도달하는 순간 진단이 된다. 0 은 그 저장소가 존재하지 않는다는 뜻이다.
@dataclass
class TargetProfile:
    parallel_execution_kind: str   # "multithread" | "simd"
    devices: int                   # PIM Device 개수
    banks_per_device: int          # PIM Device 하나의 bank 수
    cores_per_bank: int            # bank 하나의 PIM Core 수
    lanes: int                     # PIM Core 하나의 lane 능력값
    bank_bytes: int                # 큰 메모리 예산 (upmem: MRAM, aimx: PIM DRAM)
    local_bytes: int               # 빠른 메모리 예산 (upmem: WRAM, aimx: 없음 = 0)
    device_axis_priority: str = ""
    device_axis_granularity: str = ""
    device_axis_partitions: str = ""
    bank_partitions: str = "1,1,1"
    core_partitions: str = "1,1,1"
    lane_axis: str = "m"


def _upmem_profile(o) -> TargetProfile:
    # PIM Device = DPU. block group은 WRAM usable 예산에서 공통 pass가 산출하고,
    # tasklet의 m축 협력 방식은 intra-device plan과 UPMEM verifier가 정한다.
    # DPU 는 수천 개라 자동 인수분해는 정사각에 가깝게 나눈다(priority 없음) — 형상이
    # 작으면 device_axis_partitions 로 tile 수에 맞춰 직접 나누는 것이 옳다.
    return TargetProfile(
        parallel_execution_kind="multithread",
        devices=o.num_units or 2048,
        banks_per_device=1,
        cores_per_bank=1,
        lanes=o.local_lanes,
        bank_bytes=o.unit_memory_bytes,     # mram
        local_bytes=o.local_memory_bytes,   # wram
        device_axis_partitions=o.device_axis_partitions,
        lane_axis="m",
    )


def _aimx_profile(o) -> TargetProfile:
    # PIM Device = FPGA 1장. 16 channel x 16 bank를 256개 bank로 평탄화하고,
    # 각 bank의 16 ALU를 한 core의 lane으로 표현한다. 한 block group이 이 bank
    # 전체에 걸쳐 분할되며 별개의 block group을 bank마다 배정하는 뜻이 아니다.
    # [batch, 4096/batch, 256/batch] 물리 눈금은 profile이 아니라 AiMX target
    # lowering에서 검증하고 padding한다.
    # device 분할 순서는 n→m 이다 — 행을 쪼개면 가장 큰 buffer 인 B 가
    # device 수만큼 복제된다(설계 §6.1). 분할 눈금은 target 의 물리 폭 4096을
    # profile 문자열로 다시 넣지 않고, TargetSpec을 통해 공통 pass가 확정한
    # #pim.execution_tile에서 유도한다. bank 예산은 선언 상수(pim_malloc은
    # 실패해도 NULL 대신 프로세스를 죽여 재 보는 방법이 없다)이고,
    # bank_capacity_bytes 옵션은 공통 pass의 k 상한 탐색과
    # --convert-pim-to-aimx 재검증이 같은 값을 보도록 profile에 싣는다.
    batch = o.batch or 8
    return TargetProfile(
        parallel_execution_kind="simd",
        devices=o.num_units or _aimx_device_count(),
        banks_per_device=AIMX_BANKS_PER_DEVICE,
        cores_per_bank=1,
        lanes=16,
        bank_bytes=o.bank_capacity_bytes or AIMX_BANK_CAPACITY_BYTES,
        local_bytes=0,
        bank_partitions=f"{batch},{256 // batch},1",
        core_partitions="1,1,1",
        lane_axis="n",
        device_axis_priority="n,m",
        device_axis_partitions=o.device_axis_partitions,
    )


# target 고유 값은 전부 위 두 함수에 있다 — 아래 코드에는 target 이름 분기가 없다.
_TARGET_PROFILE = {"upmem": _upmem_profile, "aimx": _aimx_profile}


def _axis_options(prefix: str, m: int, n: int, k: int) -> str:
    # group-m/n/k, tile-m/n/k 를 축별로 적는다. 0 = 자동 = pass 기본값이라 적지 않는다.
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
    "--convert-ttshared-to-pim",   # 메모리 접근을 pim.access/load/store 로
    "--pim-build-logical-grid",    # source grid 를 logical tile loop nest 로
    "--pim-recover-access-origin", # access 의 축별 origin 을 logical 좌표로 채운다
    "--pim-form-compute",          # 누산기를 matmul 의 outs 로 통합
    "{profile}",                   # 기계 profile 부착. 여기부터 profile 을 읽는다
    "{form_block_groups}",         # 계층 2: 사용자 묶음(group-m/n/k) → #pim.block_group
    "{plan_intra_device}",         # block group 하나의 bank/core/lane 분할 계획
    "{form_execution_tiles}",      # large-memory 예산으로 execution tile shape 을 먼저 결정
    "{map_devices}",               # execution tile grid 를 num_units 이하 active PIM Device grid 에 scheduling
    "--pim-place-data",            # 최종 region·replication·capacity materialize
    "--pim-verify",                # 공통 fail-fast 검증
    "-cse",
    "-canonicalize",
]


def _profile_pass(target: str, t: TargetProfile) -> str:
    return ("--pim-set-target-profile="
            f"target={target} "
            f"number-of-pim-devices={t.devices} "
            f"number-of-banks-per-pim-device={t.banks_per_device} "
            f"number-of-pim-cores-per-bank={t.cores_per_bank} "
            f"number-of-parallel-lanes-per-pim-core={t.lanes} "
            f"parallel-execution-kind={t.parallel_execution_kind} "
            + f"usable-bank-capacity-bytes={t.bank_bytes} "
            f"usable-local-capacity-bytes={t.local_bytes}")


def _ttsharedir_to_pim_common(ttshared: str, options) -> str:
    t = _TARGET_PROFILE[options.pim_target](options)
    profile = _profile_pass(options.pim_target, t)
    opts = [("device-axis-priority", t.device_axis_priority),
            ("device-axis-granularity", t.device_axis_granularity),
            ("device-axis-partitions", t.device_axis_partitions)]
    given = " ".join(f"{k}={v}" for k, v in opts if v)
    map_devices = "--pim-map-devices" + (f"={given}" if given else "")
    # target profile에서 정한 기본 plan을 공통 pass 옵션으로 전달한다. autotuner가
    # 이 문자열을 바꾸더라도 materialize와 검증은 같은 공통 pass가 담당한다.
    bank_partitions = options.bank_partitions or t.bank_partitions
    core_partitions = options.core_partitions or t.core_partitions
    plan_lanes = options.intra_device_lanes or (
        0 if t.parallel_execution_kind == "multithread" else t.lanes)
    lane_axis = options.lane_axis or t.lane_axis
    plan_intra_device = (
        "--pim-plan-intra-device="
        f"bank-partitions={bank_partitions} "
        f"core-partitions={core_partitions} "
        f"lanes={plan_lanes} lane-axis={lane_axis}")
    bg = _axis_options("group", options.group_m, options.group_n, options.group_k)
    form_block_groups = "--pim-form-block-groups" + (f"={bg}" if bg else "")
    # Common pass의 tile-m/n/k는 논리 원소 상한이다. 공개 option의
    # tile_m/n/k(block-group 개수)와 혼합하지 않고 별도 값을 넘긴다.
    et = _axis_options("tile", options.capacity_limit_m,
                       options.capacity_limit_n, options.capacity_limit_k)
    form_execution_tiles = "--pim-form-execution-tiles" + (f"={et}" if et else "")
    return _run_opt(ttshared, [p.format(profile=profile,
                                        form_block_groups=form_block_groups,
                                        plan_intra_device=plan_intra_device,
                                        map_devices=map_devices,
                                        form_execution_tiles=form_execution_tiles)
                               for p in PIM_COMMON_PASSES])


def _pim_common_to_upmem(pim_common: str, options) -> str:
    # 기계 값은 전부 profile 에서, 시간 창은 공통 IR 의 #pim.execution_tile 에서 읽는다.
    # host 스레드 수만 lowering policy 라 옵션으로 준다.
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
    # 실제로 쓰는 tasklet 수. --pim-plan-intra-device이 lane 예산 안에서 고른 값이라 요청값과
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

# AiMX는 두 단계로 내린다. 첫 단계가 execution tile·padding·배치를 정하고(level 1),
# 두 번째가 그 tile 모양에서 descriptor program을 펼친다(level 2). 두 단계를 나누는 이유는
# level 1이 tile마다 반복되고 level 2가 launch당 한 번만 펼쳐지기 때문이다 — 그 구분이
# IR에 보이지 않으면 chain 재사용이 codegen이 몰래 하는 결정이 된다.
def _aimx_passes(options):
    # 3계층은 공통 pass 가 이미 IR 에 새겼다 — block group 은 #pim.block_group 으로,
    # execution tile 상한은 #pim.execution_tile로 전달된다. batch는 PIM profile에
    # 넣지 않고 AiMX target option으로만 전달한다.
    tile = f"--convert-pim-to-aimx=batch={options.batch}"
    # 0이면 profile의 usable_bank_capacity_bytes를 쓴다. sweep이 실제 예산을 재려고
    # 이것만 흔드는 경우가 있어 따로 둔다.
    if options.bank_capacity_bytes:
        tile += f" bank-capacity-bytes={options.bank_capacity_bytes}"
    return [tile, "--aimx-lower-matmul", "-cse", "-canonicalize"]


def _pim_common_to_aimx(pim_common: str, options) -> str:
    return _run_opt(pim_common, _aimx_passes(options))


def _aimx_to_artifact(aimx_ir: str, metadata, options) -> bytes:
    # AiMX IR → runtime stub(C++) + descriptor program(C 헤더)을 생성해 번들로 묶는다.
    # upmem과 동일하게, launcher가 호출할 진입 함수 이름을 stub에서 파싱해 metadata["name"]에 채운다.
    stub_src = _run_translate(aimx_ir, "--aimx-to-runtime-stub")
    descriptors = _run_translate(aimx_ir, "--aimx-to-descriptors")
    match = re.search(r'extern "C" void (\w+)\(', stub_src)
    if not match:
        raise RuntimeError("generated AiMX stub에서 진입 함수를 찾지 못함")
    metadata["name"] = match.group(1)
    # base_fpga는 컴파일이 아니라 빌드 때 -D로 들어간다. launcher가 읽도록 번들에 싣는다.
    return json.dumps({"stub": stub_src, "descriptors": descriptors,
                       "base_fpga": options.base_fpga}).encode("utf-8")


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
    # block group: local 상주 단위로 묶는 tile 수. group_m × group_n 이 출력 묶음이고
    # group_k 는 reduction strip(적재 한 번이 담는 K tile 수 — 누산기는 그대로다).
    # 0 = 자동: upmem - 공통 pass 가 local(WRAM) 예산에 들어갈
    # 때까지 m → n → k 번갈아 키우고(협력 모드에서만)
    # aimx - 실행 단위 채움이 담당하므로 항등과 같다. 값을 주면 그대로 쓴다(1 = 항등).
    # 두 target 모두 공통 pass(--pim-form-block-groups)가 IR 수준에서 묶는다 —
    # 지나고 나면 target lowering 이 보는 "block" 이 곧 group 이다.
    group_m: int = 0
    group_n: int = 0
    group_k: int = 0
    # ③ execution tile: device 하나가 launch 1회에 상주시키는 몫 (원소 단위).
    # 0 = 자동: 공통 pass(--pim-form-execution-tiles)가 m·n 은 기본 단위(m: block
    # group 1개, n: target 기본 — aimx 는 한 device 폭 4096), k 는 남는 큰 메모리
    # 예산 전부(target 이 등록한 footprint 로 이분 탐색)로 채워 #pim.execution_tile
    # 에 구체값으로 부착한다. 값을 주면 그대로 쓴다(축의 group 원소수 배수여야 함).
    # target lowering 은 읽기만 한다:
    #   upmem  창(round) 반복 — 배정이 상한 안이면 창 1개. K 가 k 상한을 넘으면
    #          k 창마다 부분합이 나와 host 가 누산한다(unstage accumulate).
    #   aimx   상한이 담는 block group 수 × 물리 group 크기가 m_t·n_max·k_max.
    #          k 는 host 누산 창이라 split-K kernel 이 필요 없다.
    tile_m: int = 0
    tile_n: int = 0
    tile_k: int = 0
    # #pim.execution_tile에 넣는 유효 논리 원소 상한. tile_m/n/k는
    # block-group 개수이고 helper가 BLOCK * GROUP * TILE로 이 값을 유도한다.
    capacity_limit_m: int = 0
    capacity_limit_n: int = 0
    capacity_limit_k: int = 0
    local_lanes: int = 16        # PIM Core 내부 병렬 lane 능력값. 실제 값은 공통 pass 가 정한다.
    # 유닛당 빠른 메모리(공통 이름 — upmem 은 WRAM). 물리 64 KB 에서 스택·런타임 몫을 뺀 값.
    local_memory_bytes: int = 48 * 1024
    # 유닛당 큰 메모리(공통 이름 — upmem 은 MRAM). target 이 버퍼 수로 나눈다. 물리 64 MB
    # 에서 DPU 런타임 몫을 뺀 값이라 물리 크기보다 작다.
    unit_memory_bytes: int = 60 * 1024 * 1024
    host_threads: int = 8        # host가 영역을 모으고 되돌릴 때 쓰는 스레드 수. 1이면 순차.
    # 파생 통로: 축별 device 분할 수 "<축>:<개수>,...". tile_* 에서 helper 가 계산해
    # 채우는 값이라 직접 쓰는 것은 디버깅·재현용이다. 빈 값이면 자동 인수분해.
    # k 분할은 kernel 이 K 를 연속 구간으로 가르는 blocked 형태가
    # 전제다 — cyclic 분할은 pim-place-data 가 진단한다.
    device_axis_partitions: str = ""
    # 하나의 block group을 device 내부에서 분할하는 autotuner 입력. 빈 값/0이면
    # target profile이 제공한 seed를 사용하고, 공통 planner가 topology와 정합성을
    # 검증해 #pim.intra_device_plan으로 materialize한다.
    bank_partitions: str = ""
    core_partitions: str = ""
    intra_device_lanes: int = 0
    lane_axis: str = ""

    # ── aimx 전용 ───────────────────────────────────────────────────────────
    # 몇 번 카드부터 쓰는가. 런타임은 fpga_id로 pl_dma[id]를 경계 검사 없이 인덱싱하므로
    # [base_fpga, base_fpga + num_units)가 [0, 감지된 개수) 안이어야 한다. 생성된 stub이
    # init_pim_drv()가 돌려준 개수와 대조해 어기면 즉시 멈춘다.
    base_fpga: int = 0
    # 출력 block 을 batch x (4096/batch)로 자르는 값, {1,2,4,8}만 legal — 이것이 곧
    # arithmetic intensity 다. 실행 단위 [batch, 4096/batch, 256/batch]가 여기서
    # 나온다. BLOCK 이 이 눈금의 배수가 아니면 padding 이고, 배수면 padding 이 없다 —
    # 공통 pass 가 remark 로 노는 비율을 알려 준다.
    batch: int = 8
    # PIM DRAM 예산. 0이면 profile의 usable_bank_capacity_bytes를 쓴다.
    bank_capacity_bytes: int = 0

    def __post_init__(self):
        # target 없이 PIM을 쓰는 것은 의도된 사용이 아니므로 기본값으로 때우지 않고 바로 터뜨린다.
        if self.pim_target is None:
            raise ValueError(
                "PIMBackend requires an explicit `pim_target` option. "
                "Supported values: 'upmem', 'aimx'."
            )
        for axis in "mnk":
            tile = getattr(self, f"tile_{axis}")
            limit = getattr(self, f"capacity_limit_{axis}")
            if tile and not limit:
                raise ValueError(
                    f"tile_{axis} is a block-group count and cannot be "
                    "converted without BLOCK/GROUP context; use "
                    "inter_device_plan/pim_launch_options or also provide "
                    f"capacity_limit_{axis}=BLOCK_{axis.upper()}*"
                    f"group_{axis}*tile_{axis}"
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
        stages["ttsharedir"] = lambda src, metadata: _ttir_to_ttsharedir(src)
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
        return self.target

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
