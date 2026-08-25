# PIMTorchConfig — "무엇을 고를지"와 Inductor 안전장치.
#
# 실행 계획(block/group/execution tile, 축별 device 분할 수)은 여기 없다.
# autotuner/types.py의 PIMConfig가 갖고, 하드웨어 자원은 targets.py의
# PIMHardwareProfile이 갖는다. 고르는 일은 plan.py가 한다.
#
# 여기 남은 것은 셋뿐이다.
#   1) 어느 target이고, 어떻게 config를 고를 것인가 (seed / autotune cache)
#   2) autotuner가 모르는 template·runtime knob (lane_mode, num_warps, ...)
#   3) Inductor 쪽 안전장치 (compile_threads, fx_graph_cache)

from dataclasses import dataclass
from typing import Optional


class UnsupportedPIMIntMM(RuntimeError):
    """strict 모드에서 PIM이 못 맡는 _int_mm을 만났을 때. CPU fallback을 하지
    않으므로 여기서 멈춘다 — fallback이 있으면 PIM template compile이 실패해도
    테스트가 통과한 것처럼 보인다."""


# PIMOptions로 그대로 넘어가는 key 목록. 여기 없는 이름을 backend_options에
# 넣으면 PIMBackend.parse_options가 조용히 버리므로 넘기기 전에 막는다.
PIM_OPTION_KEYS = (
    "pim_target",
    "num_units",
    "local_lanes",
    "lane_mode",
    "host_threads",
    "group_m", "group_n", "group_k",
    "tile_m", "tile_n", "tile_k",
    "device_axis_partitions",
)


@dataclass(frozen=True)
class PIMTorchConfig:
    # ── 어느 PIM target인가 ────────────────────────────────────────────────
    # 하드웨어 수치(num_units, local_lanes, memory 예산)는 여기 없다 —
    # autotuner/targets.py의 등록표에서 온다.
    target: str = "upmem"
    # strict: 지원 조건을 하나라도 어기면 즉시 에러. False로 두면 stock Inductor
    # lowering(CPU)로 되돌아간다. 경로 검증 중에는 True여야 한다.
    strict: bool = True

    # ── config를 고르는 방법 ──────────────────────────────────────────────
    # 원하는 K 분할 수. 이 P_K를 갖는 heuristic seed를 고른다. 나머지 knob은
    # seed 규칙(local memory 예산, device 활용 단위)이 shape에서 유도한다.
    split_k: int = 4
    # P_M=1로 고정할까. 직접 경로에서 확인된 형태다 (plan.py 주석 참고).
    pin_row_partitions: bool = True
    # 경로를 주면 seed 대신 offline autotune cache의 best를 쓴다 (단계 B).
    # cache miss는 fail-fast다 — 조용히 seed로 돌아가면 무엇으로 실행됐는지
    # 알 수 없다.
    autotune_cache_path: Optional[str] = None

    # ── autotuner가 모르는 knob ──────────────────────────────────────────
    # lane 실행 형태. PIMConfig에도 profile에도 없다.
    lane_mode: str = "output_row"
    # host가 영역을 모으고 되돌릴 때 쓰는 스레드 수.
    host_threads: int = 8
    # PIM은 warp/stage 개념이 없지만 triton.compile 옵션 스키마에는 있어서
    # 채운다. PIMOptions의 기본값(num_warps=0, num_stages=1)과 어긋나지 않게 둔다.
    num_warps: int = 1
    num_stages: int = 1

    # tile이 나누어떨어지지 않는 shape를 허용할까. mask의 extent가 상수면
    # PIMPlaceDataPass의 isExtentCandidate가 extent 복원에 실패한다. template이
    # M/N/K를 런타임 인자로 넘기게 된 뒤로는 풀렸을 수 있지만 확인 전이다.
    allow_masked_tiles: bool = False

    # ── Inductor 쪽 안전장치 ──────────────────────────────────────────────
    # 컴파일을 in-process로 묶는다. worker subprocess는 우리 monkeypatch가 없어
    # PIM 분기를 모른다 (runtime_compat 참고).
    force_single_compile_thread: bool = True
    # FX graph cache는 lowering 함수가 바뀐 것을 key에 넣지 않는다. 캐시 히트가
    # 나면 PIM이 아닌 예전 CPU 코드가 그대로 돌아온다.
    disable_fx_graph_cache: bool = True

    def __post_init__(self):
        if not isinstance(self.split_k, int) or self.split_k < 1:
            raise ValueError(
                f"split_k must be a positive int, got {self.split_k!r}")
        for name in ("host_threads", "num_warps", "num_stages"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"PIMTorchConfig.{name} must be a non-negative int, "
                    f"got {value!r}")
        # target 이름이 등록표에 있는지는 여기서 보지 않는다 — targets.py가
        # KeyError로 명시적으로 실패한다 (plan.resolve_execution_plan).


def validate_pim_options(options: dict) -> dict:
    """PIMOptions가 실제로 받는 key인지 확인한다.

    parse_options는 모르는 key를 조용히 버린다. 오타 하나가 "옵션을 줬는데
    적용이 안 된" 상태로 남는 것을 여기서 막는다."""
    unknown = [k for k in options if k not in PIM_OPTION_KEYS]
    if unknown:
        raise ValueError(
            f"Unknown PIM backend option(s): {unknown}. "
            f"Known: {list(PIM_OPTION_KEYS)}")
    if not options.get("pim_target"):
        raise ValueError("backend_options must carry an explicit 'pim_target'.")
    return dict(options)
