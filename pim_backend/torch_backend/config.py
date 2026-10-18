# PIMTorchConfig — "무엇을 고를지"와 Inductor 안전장치.
#
# 실행 계층(block/group/execution tile, p_m/p_n/p_k)은 여기 없다.
# autotuner의 PIMConfig가 갖고, 하드웨어 자원은 공유 PIMTargetSpec이 갖는다.
#
# 여기 남은 것은 셋뿐이다.
#   1) 어느 target spec과 Common IR config를 쓸 것인가
#   2) Triton template compile knob (num_warps 등)
#   3) Inductor 쪽 안전장치 (compile_threads, fx_graph_cache)

from dataclasses import dataclass
from typing import Optional

from ._autotuner import PIMConfig, PIMTargetSpec


class UnsupportedPIMGemm(RuntimeError):
    """Unsupported PIM GEMM shape, dtype, or configuration."""


# 기존 import 이름을 유지한다.
UnsupportedPIMIntMM = UnsupportedPIMGemm


# PIMOptions로 그대로 넘어가는 key 목록. 여기 없는 이름을 backend_options에
# 넣으면 PIMBackend.parse_options가 조용히 버리므로 넘기기 전에 막는다.
PIM_OPTION_KEYS = (
    "pim_target",
    "num_units",
    "banks_per_unit", "cores_per_bank",
    "lanes_per_core",
    "bank_capacity_limit_bytes", "local_capacity_limit_bytes",
    "p_m", "p_n", "p_k",
    "nr_tasklets",
    "host_threads",
    "group_m", "group_n", "group_k",
    "tile_m", "tile_n", "tile_k",
    "batch", "base_fpga", "aimx_matmul_k_alignment",
)


@dataclass(frozen=True)
class PIMTorchConfig:
    # ── 어느 PIM target인가 ────────────────────────────────────────────────
    # compiler와 autotuner가 사용하는 같은 target spec. None이면
    # backend/target.py의 기본 UPMEM spec을 사용한다.
    spec: Optional[PIMTargetSpec] = None
    # ── config를 고르는 방법 (우선순위 순) ────────────────────────────────
    # 1) config를 직접 준다. 그대로 쓴다.
    #    autotuner가 "이 config를 재라"고 요구할 때 쓰는 통로다. 여기서 값을
    #    바꾸면 잰 것과 기록한 것이 달라진다.
    config: Optional[PIMConfig] = None
    # 2) offline autotune cache의 best (단계 B). cache miss는 fail-fast다 —
    #    조용히 seed로 돌아가면 무엇으로 실행됐는지 알 수 없다.
    autotune_cache_path: Optional[str] = None
    # 3) 둘 다 없으면 Common IR 값으로 만든 기본 seed.
    # ── GEMM packing ──────────────────────────────────────────────────────
    # False면 fx_packing이 이 GEMM을 다른 GEMM과 묶지 않는다. model compile이
    # 따로 실행하는 쪽이 더 빠르다고 측정한 GEMM에 쓴다.
    pack: bool = True
    # ── Triton compile config ─────────────────────────────────────────────
    # PIM은 warp/stage 개념이 없지만 triton.compile 옵션 스키마에는 있어서
    # 채운다. PIMOptions의 기본값(num_warps=0, num_stages=1)과 어긋나지 않게 둔다.
    num_warps: int = 1
    num_stages: int = 1

    # ── Inductor 쪽 안전장치 ──────────────────────────────────────────────
    # 컴파일을 in-process로 묶는다. worker subprocess는 우리 monkeypatch가 없어
    # PIM 분기를 모른다 (runtime_compat 참고).
    force_single_compile_thread: bool = True
    # FX graph cache는 lowering 함수가 바뀐 것을 key에 넣지 않는다. 캐시 히트가
    # 나면 PIM이 아닌 예전 CPU 코드가 그대로 돌아온다.
    disable_fx_graph_cache: bool = True

    def __post_init__(self):
        for name in ("num_warps", "num_stages"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"PIMTorchConfig.{name} must be a non-negative int, "
                    f"got {value!r}")


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
