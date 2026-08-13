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


# AiMX FPGA 하나가 커널에 내주는 PIM DRAM 예산 (byte).
#
# **보수적으로 선언한 값이다.** 진짜 용량은 driver 가 찍는 BAR 길이에서 읽는다:
#   sudo dmesg | grep -E "PIM_MEM_[0-9]+\] BAR #4"
# 할당을 키워 가며 재는 방법은 성립하지 않는다 — pim_malloc 은 1 MB 부터 16 GB 까지
# 전부 '성공' 하고(주소만 떼어 준다) 진짜 실패하면 NULL 대신 프로세스를 죽인다.
# 기본 execution tile(m=16, k=4096, n=4096)이 약 36.6 MB 이므로 이 값으로 tile 이
# 줄어들지 않는다. 더 큰 tile 을 쓰려면 실제 BAR 길이로 올린다.
AIMX_DEVICE_DRAM_BYTES = 1 << 30


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

# target 마다 고정된 기계 값. 나머지는 PIMOptions 에서 온다.
# capacity 는 (usable_bank_capacity_bytes, usable_local_capacity_bytes) 를 옵션에서 뽑는다.
# 음수면 미측정이라는 뜻이라 profile 에 필드가 생기지 않고, 그 값을 필요로 하는 검사에
# 도달하는 순간 진단이 된다. 0 은 그 저장소가 존재하지 않는다는 뜻이다.
# target 고유 값은 전부 여기 한 곳에 있다 — 아래 코드에는 target 이름 분기가 없다.
_TARGET_PROFILE = {
    "upmem": {
        "pim_core_shape": "1",
        "parallel_execution_kind": "multithread",
        "uniform_instruction_sequence": "false",
        # PIM Device = DPU. 옵션이 0 이면 이 값을 물리 개수로 본다.
        "devices": lambda o: o.num_units or 2048,
        # DPU 는 수천 개라 정사각에 가까운 분할이 맞다. 우선순위를 두지 않는다.
        "device_axis_priority": "",
        # DPU 는 tile 하나를 통째로 맡으므로 분할 눈금이 따로 없다.
        "device_axis_granularity": lambda o: "",
        # lane(tasklet) 실행 형태. whole_tile 은 lane 이 tile 을 통째로 맡고 버퍼를
        # lane 수만큼 복제한다. output_row 는 lane 들이 한 tile 을 출력 행으로 나눠
        # 함께 계산한다 — 버퍼가 한 벌이라 tile 을 키워도 lane 수가 줄지 않는다.
        "lane_mode": lambda o: o.lane_mode,
        # block group: WRAM 상주 단위로 묶는 tile 수. 전부 1 이면 항등(기존 동작).
        "block_group": lambda o: (
            f"group-m={o.group_m} group-n={o.group_n} group-k={o.group_k}"
            if o.group_m != 1 or o.group_n != 1 or o.group_k != 1 else ""),
        # 축별 device 분할의 명시 지정. 빈 값이면 자동 인수분해(기존 동작). 형상이
        # 작아 group 수 < DPU 수일 때 explicit_k_partition 을 나눠 점유율을 채운다.
        "device_axis_partitions": lambda o: o.device_axis_partitions,
        # 유닛=DPU, 큰 메모리=MRAM, 빠른 메모리=WRAM 이라 둘 다 옵션 값이 그대로 온다.
        "capacity": lambda o: (o.unit_memory_bytes, o.local_memory_bytes),
    },
    "aimx": {
        # PIM Device = FPGA 1개, 그 안에 PIM Core 1개다. lane 4096 개는 주소 decoding 이
        # 정하는 고정된 갈래라 compiler 가 배정할 대상이 아니므로 topology 는 [1] 이다.
        "pim_core_shape": "1",
        "parallel_execution_kind": "simd",
        "uniform_instruction_sequence": "true",
        "lanes": 4096,      # NUM_BANKS * NUM_CHANNELS * ALU_SIZE
        # PIM Device = FPGA 1 장. 옵션이 0 이면 이 머신에 실제로 있는 개수를 센다.
        "devices": lambda o: o.num_units or _aimx_device_count(),
        # device 축은 열부터 쪼갠다. 행을 쪼개면 B 가 device 수만큼 복제되는데 B 가 가장
        # 큰 buffer 다. 열을 쪼개면 A 가 복제되지만 A 는 이미 n loop 때문에 device 안에서
        # 여러 번 올라가므로 전체 전송량이 늘지 않는다 (설계 §6.1).
        "device_axis_priority": "output_column_tile,output_row_tile",
        # 그리고 그 축은 execution tile 눈금으로만 자른다. device 하나가 n 방향으로
        # 소비하는 단위는 n_t = min(4096*ceil(N/4096), n_max) 라 언제나 4096 의 배수다.
        # 그보다 잘게 자르면 마지막 tile 이 절반만 유효한 채로 온전히 실행된다. batch 와는
        # 무관하다 — 출력 block 폭 4096/batch 는 execution tile **안**의 눈금이다.
        "device_axis_granularity": lambda o: "output_column_tile:4096",
        # simd 는 lane 협력이 하드웨어에 내장돼 있어 pim-plan-group-execution 이 옵션을 읽지
        # 않는다. 빈 값이면 옵션을 붙이지 않는다.
        "lane_mode": lambda o: "",
        # AiMX 의 묶음은 target lowering 의 execution tile 이 담당한다(group_m 은
        # --convert-pim-to-aimx 로 간다). 공통 block group pass 는 항등으로 지난다.
        "block_group": lambda o: "",
        # AiMX 는 priority/granularity 로 나눈다. 명시 분할은 쓰지 않는다.
        "device_axis_partitions": lambda o: "",
        # core 별 local storage 는 없다(0). bank 예산은 선언 상수다 — pim_malloc 은
        # 실패해도 NULL 을 돌려주지 않고 프로세스를 죽이므로 재 보는 방법이 없다.
        "capacity": lambda o: (AIMX_DEVICE_DRAM_BYTES, 0),
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
    "{form_block_groups}",         # 이웃 tile 을 block group(상주 단위)으로 묶는다. 1,1 이면 항등
    "{profile}",                   # 기계 profile 부착. 여기부터 profile 을 읽는다
    "{plan_group_execution}",      # block group 실행 계획 — lane 수, lane 협력 방식
    "{map_devices}",               # execution tile 을 PIM Device 에 균형 분배
    "--pim-place-data",            # 최종 region·replication·capacity materialize
    "--pim-verify",                # 공통 fail-fast 검증
    "-cse",
    "-canonicalize",
]


def _profile_pass(options) -> str:
    t = _TARGET_PROFILE[options.pim_target]
    bank, local = t["capacity"](options)
    return ("--pim-set-target-profile="
            f"target={options.pim_target} "
            f"number-of-pim-devices={t['devices'](options)} "
            f"pim-core-shape={t['pim_core_shape']} "
            f"number-of-parallel-lanes-per-pim-core={t.get('lanes', options.local_lanes)} "
            f"parallel-execution-kind={t['parallel_execution_kind']} "
            f"requires-uniform-instruction-sequence={t['uniform_instruction_sequence']} "
            f"usable-bank-capacity-bytes={bank} "
            f"usable-local-capacity-bytes={local}")


def _ttsharedir_to_pim_common(ttshared: str, options) -> str:
    t = _TARGET_PROFILE[options.pim_target]
    profile = _profile_pass(options)
    opts = [("device-axis-priority", t["device_axis_priority"]),
            ("device-axis-granularity", t["device_axis_granularity"](options)),
            ("device-axis-partitions", t["device_axis_partitions"](options))]
    given = " ".join(f"{k}={v}" for k, v in opts if v)
    map_devices = "--pim-map-devices" + (f"={given}" if given else "")
    lane_mode = t["lane_mode"](options)
    plan_group_execution = ("--pim-plan-group-execution"
                            + (f"=lane-mode={lane_mode}" if lane_mode else ""))
    bg = t["block_group"](options)
    form_block_groups = "--pim-form-block-groups" + (f"={bg}" if bg else "")
    return _run_opt(ttshared, [p.format(profile=profile,
                                        form_block_groups=form_block_groups,
                                        plan_group_execution=plan_group_execution,
                                        map_devices=map_devices)
                               for p in PIM_COMMON_PASSES])


def _pim_common_to_upmem(pim_common: str, options) -> str:
    # 기계 값은 전부 profile 에서 읽는다. host 스레드 수와 round(시간 창)만 lowering
    # policy 라 옵션으로 준다.
    opt = f"--convert-pim-to-upmem=host-threads={options.host_threads}"
    if options.round_tiles:
        opt += f" round-tiles={options.round_tiles}"
    return _run_opt(pim_common, [opt])


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
    # 실제로 쓰는 tasklet 수. --pim-plan-group-execution이 lane 예산 안에서 고른 값이라 요청값과
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
    # execution tile 은 Triton의 @triton.autotune으로 흔들 수 없다(그쪽은 BLOCK_*만 본다).
    # 그래서 backend option으로 두고 여기서 pass에 그대로 넘긴다 — 설계 §12.4.
    tile = (f"--convert-pim-to-aimx=group-m={options.group_m} tile-k={options.tile_k} "
            f"tile-n={options.tile_n} batch={options.batch}")
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

    # --- PIM 고유 옵션 ---
    # 이름을 target으로 두면 안 된다. Triton이 metadata를 {"target": GPUTarget, **options.__dict__}
    # 로 만들어서 같은 이름의 옵션이 GPUTarget을 덮어쓰고, CompiledKernel이 그것을 다시 읽는다.
    pim_target: str = None       # "upmem" | "aimx" — 누락 시 아래에서 즉시 에러 (fallback 없음)
    # 기계 — 이름을 target 중립으로 둔다(UPMEM에서는 유닛=DPU, 빠른 메모리=WRAM).
    # 물리 PIM Device 개수. 0이면 target이 정한다 — upmem은 2048, aimx는 이 머신에
    # 실제로 있는 FPGA 개수(/dev/pl_dma*)를 센다. split의 곱이 이보다 크면 컴파일 시점에 실패.
    num_units: int = 0
    # 몇 번 카드부터 쓰는가. 런타임은 fpga_id로 pl_dma[id]를 **경계 검사 없이** 인덱싱하므로
    # [base_fpga, base_fpga + num_units)가 [0, 감지된 개수) 안에 있어야 한다. 생성된 stub이
    # init_pim_drv()가 돌려준 개수와 대조해 어기면 즉시 멈춘다.
    base_fpga: int = 0
    # execution tile 상한 (원소 단위) — 데이터 3계층의 셋째 층. device 하나가 한 번에
    # 상주시키는 몫의 상한이고, A(tile_m×tile_k) + B(tile_k×tile_n) + C(tile_m×tile_n)
    # 이 PIM DRAM 예산 안이어야 한다. 실제 모양은 문제 크기가 정하는 런타임 값이다.
    # aimx 는 --convert-pim-to-aimx 가 tile_k/tile_n 을 소비하고(m 은 group_m 로 표현),
    # upmem 은 launch helper(tiling.execution_tile)가 분할 유도에 쓴다 — round(시간 창)
    # 가 생기면 upmem 도 상한으로 직접 소비한다. batch 는 aimx 전용: 출력 block 을
    # batch x (4096/batch)로 자르는 값이고 {1,2,4,8}만 legal 하다 — 이것이 곧
    # arithmetic intensity 라 단일 device 성능을 지배한다.
    #
    # m 방향은 크기가 아니라 **Triton block 묶음 수**로 준다. m_t = BLOCK_M * group_m이라
    # BLOCK_M=8이면 group_m=5가 곧 m_t=40이다. 반대로 크기를 주고 BLOCK_M 경계로 올리면
    # m_t가 언제나 BLOCK_M의 배수로 튀어 그런 값을 만들 수 없다.
    group_m: int = 1
    # upmem 은 같은 뜻의 묶음을 공통 pass(--pim-form-block-groups)가 한다:
    # group_m × group_n 개의 tile 이 block group 하나(WRAM 상주 단위)가 되고,
    # group_k 는 reduction 방향 strip — WRAM 적재 한 번이 담는 K tile 수다
    # (누산기는 그대로고 A·B strip 의 K 변만 커진다). group_n/group_k 는 upmem 만
    # 읽는다. 전부 1 이면 항등이다.
    group_n: int = 1
    group_k: int = 1
    tile_m: int = 0   # 0 = 제한 없음(그 축을 자르지 않음). 지금은 upmem helper 만 읽는다.
    tile_k: int = 32768
    tile_n: int = 4096
    # 0이면 pass가 Triton block이 허락하는 최대값을 고른다. BLOCK 16/16/16이면 8이고,
    # batch=1 대비 3.4배로 실측된다. 고정값을 박으면 BLOCK_K=64 같은 흔한 값이 컴파일
    # 오류가 되므로 기본값은 값이 아니라 규칙이다.
    batch: int = 0
    # PIM DRAM 예산. 0이면 profile의 usable_bank_capacity_bytes를 쓴다.
    bank_capacity_bytes: int = 0
    # 유닛당 빠른 메모리. tile 버퍼 합계 × local_lanes가 넘으면 실패한다. UPMEM의 물리 WRAM은
    # 64 KB지만 tasklet 스택과 런타임도 여기 살므로 커널이 써도 되는 몫은 그보다 작다.
    local_memory_bytes: int = 48 * 1024
    host_threads: int = 8        # host가 영역을 모으고 되돌릴 때 쓰는 스레드 수. 1이면 순차.
    # 유닛당 큰 메모리. target이 버퍼 크기를 여기서 나눈다. UPMEM의 물리 MRAM은 64 MB지만
    # DPU 런타임도 MRAM에 자리를 잡으므로 전부를 나눠 주면 링커가 넘친다. 커널이 써도 되는
    # 몫이라는 뜻이라 물리 크기보다 작게 둔다.
    unit_memory_bytes: int = 32 * 1024 * 1024
    local_lanes: int = 1         # DPU 내부 intra-unit 병렬 lane 능력값. 실제 값은 lane 예산이 정한다.
    # 축별 device 분할의 명시 지정: "<축>:<개수>,..." (예:
    # "output_column_tile:128,explicit_k_partition:16"). 빈 값이면 자동 인수분해.
    # explicit_k_partition 을 나누려면 kernel 이 K 를 연속 구간으로 갈라야 한다
    # (split_k.py 의 blocked 형태) — cyclic 분할은 pim-place-data 가 진단한다.
    device_axis_partitions: str = ""
    # round(시간 창, upmem 전용): "<출력축>:<창의 tile 수>". DPU 배정량이 MRAM 몫을
    # 넘을 때 창 단위로 [재전송 → launch → 회수]를 반복한다. 창 축에 매이지 않은
    # 입력(matmul 의 A)은 한 번만 전송된다. 빈 값이면 창 1개(기존 동작).
    round_tiles: str = ""
    # lane 실행 형태 (multithread target 전용, --pim-plan-group-execution lane-mode 로 간다).
    # whole_tile: lane 이 서로 다른 tile 을 통째로 맡는다 — 지금까지의 동작.
    # output_row: lane 들이 한 tile 을 출력 행으로 나눠 함께 계산한다. 버퍼가 한 벌이라
    #             tile 을 키워도 lane 수가 줄지 않는다. 행 수가 lane 수로 나누어떨어져야 한다.
    lane_mode: str = "whole_tile"

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
