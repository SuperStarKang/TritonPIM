# PIMTorchConfig — torch.compile → PIM 경로가 쓰는 설정 전부.
#
# 원칙 두 가지다.
#
#   1) env 통로를 쓰지 않는다. 값은 이 dataclass 하나에서 나와서 template metadata
#      (triton_meta["backend_options"])로 실려 PIMBackend.parse_options까지 간다.
#      env는 프로세스 단위라 커널마다 다른 값을 줄 수 없고, 설정이 어디서 왔는지
#      추적이 끊긴다.
#   2) 첫 성공까지는 고정값을 쓴다. autotuner 연결은 단일 MM이 PIM에서 정답을
#      낸 다음에 붙인다 (resolve_pim_config의 cache 경로).

from dataclasses import dataclass, replace
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
    target: str = "upmem"          # "upmem" | "aimx"
    # strict: 지원 조건을 하나라도 어기면 즉시 에러. False로 두면 stock Inductor
    # lowering(CPU)로 되돌아간다. 경로 검증 중에는 True여야 한다.
    strict: bool = True

    # ── split-K kernel의 tile 모양 (Triton constexpr) ──────────────────────
    block_m: int = 32
    block_n: int = 32
    block_k: int = 16
    split_k: int = 4

    # ── PIM 자원 ──────────────────────────────────────────────────────────
    num_units: int = 256           # 쓸 device 수 (upmem=DPU, aimx=FPGA)
    local_lanes: int = 8           # device 하나의 실행 lane 수
    lane_mode: str = "output_row"  # lane이 한 tile을 출력 행으로 나눠 협력
    host_threads: int = 8          # host가 영역을 모으고 되돌릴 때 쓰는 스레드 수

    # ── block group: local 메모리에 함께 상주하는 tile 수 ──────────────────
    group_m: int = 1
    group_n: int = 1
    group_k: int = 1

    # ── execution tile: device 하나가 launch 1회에 상주시키는 몫(원소 단위) ─
    tile_m: int = 32
    tile_n: int = 32
    tile_k: int = 256

    # 축별 device 분할 수 "<축>:<개수>,...".
    #
    # 자동 인수분해(빈 값)를 쓸 수 없다. PIMMapDevicesPass는 explicit_k_partition을
    # device 축에서 제외하고 output_row_tile / output_column_tile 에만
    # factorize(num_units)를 나눠 준다. 그러면 split-K가 device 병렬로 전혀
    # 쓰이지 않는다.
    #
    # 문자열을 직접 주면 그대로 쓰고, 비워 두면 row/column_partitions로 만든다.
    device_axis_partitions: str = ""
    # p_n. device_axis_partitions를 비워 둘 때 반드시 지정해야 한다.
    column_partitions: Optional[int] = None
    # p_m. 기본 1은 직접 경로(examples/split_k.py)에서 확인된 형태다. 출력 access가
    # rank-3(plane, row, col)로 승격되므로 축마다 loop가 하나씩이고, 1보다 커도
    # 구조상 문제가 없다 — 다만 아직 검증하지 않았다.
    row_partitions: int = 1

    # ── Triton launch metadata ────────────────────────────────────────────
    # PIM은 warp/stage 개념이 없지만 triton.compile 옵션 스키마에는 있어서 채운다.
    # PIMOptions의 기본값(num_warps=0, num_stages=1)과 어긋나지 않게 둔다.
    num_warps: int = 1
    num_stages: int = 1

    # tile이 나누어떨어지지 않는 shape(M % BLOCK_M != 0 또는 N % BLOCK_N != 0)를
    # 허용할까. 기본은 False다.
    #
    # 나누어떨어지지 않으면 template이 mask를 만들어야 한다. mask의 extent가
    # 상수면 PIMPlaceDataPass의 isExtentCandidate가 상수를 extent 후보에서
    # 배제하므로 extent 복원이 실패한다. template이 M/N/K를 런타임 인자로 넘기게
    # 된 뒤로는 extent가 SSA 값이라 이 제약이 풀렸을 수 있지만, 아직 확인하지
    # 않았다. 확인할 때 True로 둔다.
    allow_masked_tiles: bool = False

    # ── offline autotune cache (단계 B). None이면 위 고정값을 그대로 쓴다. ──
    # 경로를 주면 lowering이 (M,N,K)로 cache를 조회한다. miss면 fail-fast다 —
    # 조용히 고정값으로 돌아가면 무엇으로 실행됐는지 알 수 없다.
    autotune_cache_path: Optional[str] = None

    # ── Inductor 쪽 안전장치 ──────────────────────────────────────────────
    # 컴파일을 in-process로 묶는다. worker subprocess는 우리 monkeypatch가 없어
    # PIM 분기를 모른다 (runtime_compat 참고).
    force_single_compile_thread: bool = True
    # FX graph cache는 lowering 함수가 바뀐 것을 key에 넣지 않는다. 캐시 히트가
    # 나면 PIM이 아닌 예전 CPU 코드가 그대로 돌아온다.
    disable_fx_graph_cache: bool = True

    def __post_init__(self):
        if self.target not in ("upmem", "aimx"):
            raise ValueError(
                f"Unsupported PIM target {self.target!r}. Supported: 'upmem', 'aimx'.")
        for name in ("block_m", "block_n", "block_k", "split_k",
                     "num_units", "local_lanes", "host_threads",
                     "group_m", "group_n", "group_k",
                     "tile_m", "tile_n", "tile_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"PIMTorchConfig.{name} must be a positive int, got {value!r}")
        # execution tile은 block group의 배수여야 tile 주소가 성립한다.
        for tile, block, group, axis in (
            (self.tile_m, self.block_m, self.group_m, "m"),
            (self.tile_n, self.block_n, self.group_n, "n"),
            (self.tile_k, self.block_k, self.group_k, "k"),
        ):
            unit = block * group
            if tile % unit:
                raise ValueError(
                    f"tile_{axis}={tile} must be a multiple of "
                    f"block_{axis}*group_{axis}={unit}")
        # 문자열을 만들어 보면서 검증한다 (여기서 못 잡으면 MLIR pass에서 터진다).
        self.axis_partitions()

    def axis_partitions(self) -> str:
        """확정된 device_axis_partitions 문자열. 구조 제약을 함께 검사한다."""
        text = self.device_axis_partitions
        if not text:
            if self.column_partitions is None:
                raise ValueError(
                    "device_axis_partitions 또는 column_partitions 중 하나를 "
                    "지정해야 한다. 자동 인수분해는 이 template에서 성립하지 "
                    "않는다 (필드 주석 참고).\n"
                    f"  device_axis_partitions="
                    f"\"output_row_tile:<p_m>,output_column_tile:<p_n>,"
                    f"explicit_k_partition:{self.split_k}\"\n"
                    f"  제약: p_n은 output tile 수 ceil(N/BLOCK_N) 이하가 의미 있고,\n"
                    f"        p_m * p_n * {self.split_k} <= num_units="
                    f"{self.num_units} 이어야 한다.")
            text = (f"output_row_tile:{self.row_partitions},"
                    f"output_column_tile:{self.column_partitions},"
                    f"explicit_k_partition:{self.split_k}")

        counts = {"output_row_tile": 1, "output_column_tile": 1,
                  "explicit_k_partition": 1}
        for entry in text.split(","):
            entry = entry.strip()
            if not entry:
                continue
            name, _, value = entry.partition(":")
            name = name.strip()
            if name not in counts:
                raise ValueError(
                    f"unknown device axis {name!r} in device_axis_partitions="
                    f"{text!r}. Known: {sorted(counts)}")
            try:
                counts[name] = int(value)
            except ValueError:
                raise ValueError(
                    f"device_axis_partitions entry {entry!r} is not "
                    f"<axis>:<count>") from None
            if counts[name] < 1:
                raise ValueError(f"device_axis_partitions entry {entry!r} "
                                 f"must be >= 1")

        p_m, p_n, p_k = (counts["output_row_tile"],
                         counts["output_column_tile"],
                         counts["explicit_k_partition"])
        if self.split_k % p_k:
            raise ValueError(
                f"explicit_k_partition={p_k} must divide split_k="
                f"{self.split_k} — kernel이 K를 SPLIT_K개의 연속 구간으로 "
                f"가르므로, device 하나가 맡는 k 평면 묶음도 연속이어야 한다.")
        if p_m * p_n * p_k > self.num_units:
            raise ValueError(
                f"the partitions need {p_m * p_n * p_k} PIM Devices but "
                f"num_units={self.num_units}")
        return text

    def backend_options(self) -> dict:
        """triton_meta에 실어 PIMBackend.parse_options까지 보낼 option dict."""
        options = {
            "pim_target": self.target,
            "num_units": self.num_units,
            "local_lanes": self.local_lanes,
            "lane_mode": self.lane_mode,
            "host_threads": self.host_threads,
            "group_m": self.group_m,
            "group_n": self.group_n,
            "group_k": self.group_k,
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "device_axis_partitions": self.axis_partitions(),
        }
        return validate_pim_options(options)

    def template_meta(self, even_tiles: bool) -> dict:
        """template의 constexpr / jinja 변수."""
        return {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k,
            "SPLIT_K": self.split_k,
            # True면 mask 없는 본문을 낸다 (allow_masked_tiles 주석 참고).
            "EVEN_TILES": even_tiles,
        }


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


def resolve_pim_config(config: PIMTorchConfig, problem) -> PIMTorchConfig:
    """실행할 (M,N,K)에 맞는 config를 고른다.

    단계 A (autotune_cache_path=None): 주어진 고정 config를 그대로 쓴다.
    단계 B (경로를 준 경우): offline autotuner가 저장해 둔 best config를 읽는다.
    lowering은 fake tensor를 다루므로 여기서 실측 autotuning을 돌리지 않는다 —
    실제 입력이 없다. cache miss는 fail-fast다.
    """
    if config.autotune_cache_path is None:
        return config
    return _config_from_cache(config, problem)


def _config_from_cache(config: PIMTorchConfig, problem) -> PIMTorchConfig:
    # import를 함수 안에 두는 이유가 둘이다.
    #   - 단계 A만 쓰는 사용자가 autotuner 모듈에 의존하지 않게 한다.
    #   - pim_backend는 설치된 패키지가 아니라 디렉토리다(triton/backends/pim은
    #     backend/ 하위만 가리키는 symlink다). examples가 하는 것과 같은 방식으로
    #     여기서 부모 디렉토리를 sys.path에 얹는다.
    import sys
    from pathlib import Path

    pim_backend_dir = str(Path(__file__).resolve().parent.parent)
    if pim_backend_dir not in sys.path:
        sys.path.insert(0, pim_backend_dir)

    from autotuner.cache import PIMConfigCache
    from autotuner.planner import PIMPlanner
    from autotuner.targets import target_profile
    from autotuner.types import PIMProblem

    profile = target_profile(config.target)
    tuning_problem = PIMProblem(M=problem.M, N=problem.N, K=problem.K,
                                input_dtype="int8", output_dtype="int32")
    cache = PIMConfigCache(config.autotune_cache_path)
    found = cache.lookup_result(tuning_problem, profile)
    if found is None:
        raise UnsupportedPIMIntMM(
            f"autotune cache miss for (M,N,K)=({problem.M},{problem.N},{problem.K}) "
            f"at {config.autotune_cache_path}. "
            f"Tune this shape offline first (examples/autotune_gemm.py), "
            f"or set autotune_cache_path=None to use the fixed config.")
    tuned, _ = found
    plan = PIMPlanner(tuning_problem, profile).build(tuned)
    p_m, p_n, p_k = plan.active_device_grid
    return replace(
        config,
        block_m=tuned.block_m, block_n=tuned.block_n, block_k=tuned.block_k,
        # SPLIT_K는 kernel이 K를 가르는 수 = K축 device 분할 수여야 한다.
        split_k=p_k,
        group_m=tuned.group_m, group_n=tuned.group_n, group_k=tuned.group_k,
        tile_m=tuned.tile_m, tile_n=tuned.tile_n, tile_k=tuned.tile_k,
        num_units=profile.num_units,
        local_lanes=profile.execution_units_per_device,
        device_axis_partitions=(
            f"output_row_tile:{p_m},output_column_tile:{p_n},"
            f"explicit_k_partition:{p_k}"),
    )
