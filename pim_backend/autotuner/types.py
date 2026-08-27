# PIM AutoTuner 핵심 객체 (design/AutoTuner_Design.md §2, §51)
#
#   PIMProblem          실행해야 하는 GEMM
#   PIMHardwareProfile  사용 가능한 PIM 자원과 제약
#   PIMConfig           AutoTuner가 변경하는 선택 변수
#   PIMPlan             Config를 실제 hardware에 배치한 결과
#   PIMPlanFeatures     Plan에서 계산한 PIM-specific feature
#   PIMBenchmarkResult  실제 실행 결과
#
# Config와 실제 실행 배치는 동일한 개념이 아니므로(§2) 네 객체를 분리하고,
# Plan = Planner(Problem, Hardware, Config) 관계를 그대로 따른다.

from dataclasses import dataclass, field

DTYPE_BYTES = {
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
}


def ceil_div(value, divisor):
    return -(-value // divisor)


@dataclass(frozen=True)
class PIMProblem:
    """실행해야 하는 GEMM: A[M,K] @ B[K,N] (§3, §51)."""

    M: int
    N: int
    K: int
    input_dtype: str = "int8"
    output_dtype: str = "int32"
    a_layout: str = "row_major"
    b_layout: str = "row_major"
    # residency는 Phase 2에서 search space에 들어간다 (§53). 여기서는 보존만 한다.
    a_resident: bool = False
    b_resident: bool = False

    @property
    def input_bytes(self):
        return DTYPE_BYTES[self.input_dtype]

    @property
    def output_bytes(self):
        return DTYPE_BYTES[self.output_dtype]


@dataclass(frozen=True)
class PIMHardwareProfile:
    """사용 가능한 PIM 자원과 제약 (§3, §51)."""

    target: str
    num_devices: int
    banks_per_device: int
    cores_per_bank: int
    lanes_per_core: int
    usable_bank_capacity_bytes: int
    usable_local_capacity_bytes: int
    alignment_bytes: int
    # execution tile이 담을 유효 논리 원소 수의 target별 seed 상한.
    # tile_m/n/k(block-group 개수)와 단위가 다르다. device memory는
    # 이 상한을 만족하는지 별도로 검사한다.
    capacity_limit_m: int
    capacity_limit_n: int
    capacity_limit_k: int
    # local memory에서 tile 버퍼가 쓸 수 없는 runtime 상주분 (§10의 S_runtime).
    # 예: UPMEM WRAM의 tasklet stack(lane 수 × stack 크기), sw_cache, heap.
    local_memory_reserved_bytes: int = 0
    # (device, round) 작업 하나의 descriptor 크기 (§23의 metadata 항).
    # 0이면 descriptor 전송량을 계산하지 않는다.
    descriptor_bytes: int = 0
    target_constraints: dict = field(default_factory=dict)

    @property
    def num_units(self):
        """Legacy launch spelling; one unit is one PIM Device."""
        return self.num_devices

    @property
    def device_memory_bytes(self):
        return self.banks_per_device * self.usable_bank_capacity_bytes

    @property
    def local_memory_bytes(self):
        return self.usable_local_capacity_bytes


@dataclass(frozen=True)
class PIMConfig:
    """AutoTuner가 변경하는 선택 변수 (§4, §51).

    tile_m/n/k는 execution tile 하나에 묶는 block-group 개수다:
        capacity_limit_M = B_M * G_M * tile_m
    p_k와 reduction_policy는 KSplitFragment로 함께 결정되고 (§6),
    p_m/p_n은 distribution_policy로부터 tile grid를 보고 계산된다 (§6.1).
    """

    block_m: int
    block_n: int
    block_k: int
    group_m: int
    group_n: int
    group_k: int
    tile_m: int
    tile_n: int
    tile_k: int
    distribution_policy: str
    p_m: int
    p_n: int
    p_k: int
    reduction_policy: str
    # 하나의 block group 내부 분할. 0/빈 값은 target seed를 사용한다. 현재 search
    # fragment는 이 값을 흔들지 않지만 cache/config 경로는 처음부터 전달 가능하다.
    bank_p_m: int = 0
    bank_p_n: int = 0
    bank_p_k: int = 0
    core_p_m: int = 0
    core_p_n: int = 0
    core_p_k: int = 0
    intra_device_lanes: int = 0
    lane_axis: str = ""
    # Phase 2 (§53) — residency / replication / streaming.
    # resident operand는 device memory에 상주해 steady-state H2D에서 빠지고,
    # replication_policy는 상주 사본의 범위(slice=필요한 조각만, full=전체),
    # streaming_policy는 stream operand의 적재 단위(per_round / all_rounds)다.
    a_residency: str = "stream"
    b_residency: str = "stream"
    replication_policy: str = "slice"
    streaming_policy: str = "per_round"
    # Phase 3 (§53) — persistent execution. PersistenceFragment가 legal 조합만
    # 만든다: descriptor_reuse ⇒ device_set_reuse, command_reuse ⇒
    # descriptor_reuse, persistent_execution ⇒ command_reuse.
    device_set_reuse: bool = True
    descriptor_reuse: bool = False
    command_reuse: bool = False
    persistent_execution: bool = False

    @property
    def capacity_limit_m(self):
        return self.block_m * self.group_m * self.tile_m

    @property
    def capacity_limit_n(self):
        return self.block_n * self.group_n * self.tile_n

    @property
    def capacity_limit_k(self):
        return self.block_k * self.group_k * self.tile_k

    def key(self):
        """중복 측정 방지용 identity (§43 loop에서 사용)."""
        return (self.block_m, self.block_n, self.block_k,
                self.group_m, self.group_n, self.group_k,
                self.tile_m, self.tile_n, self.tile_k,
                self.distribution_policy,
                self.p_m, self.p_n, self.p_k,
                self.reduction_policy,
                self.bank_p_m, self.bank_p_n, self.bank_p_k,
                self.core_p_m, self.core_p_n, self.core_p_k,
                self.intra_device_lanes, self.lane_axis,
                self.a_residency, self.b_residency,
                self.replication_policy, self.streaming_policy,
                self.persistence_strategy())

    def knobs(self):
        """diversity 비교용 tuning knob 목록 (§42)."""
        return (self.block_m, self.block_n, self.block_k,
                self.group_m, self.group_n, self.group_k,
                self.tile_m, self.tile_n, self.tile_k,
                self.distribution_policy, self.p_k, self.reduction_policy,
                self.bank_p_m, self.bank_p_n, self.bank_p_k,
                self.core_p_m, self.core_p_n, self.core_p_k,
                self.intra_device_lanes, self.lane_axis,
                self.a_residency, self.b_residency,
                self.replication_policy, self.streaming_policy,
                self.persistence_strategy())

    def persistence_strategy(self):
        """PersistenceFragment 표현 (Phase 3, §53)."""
        return (self.device_set_reuse, self.descriptor_reuse,
                self.command_reuse, self.persistent_execution)


@dataclass
class TransferPlan:
    """device 전체에 실제로 전달되는 데이터 (§23).

    a_bytes/b_bytes는 호출마다 반복되는 steady-state 전송이고,
    a_resident_bytes/b_resident_bytes는 resident operand의 1회 placement다
    (Phase 2 — transfer minimization의 대상).
    """

    a_bytes: int
    b_bytes: int
    a_replication_bytes: int
    b_replication_bytes: int
    a_resident_bytes: int
    b_resident_bytes: int
    h2d_bytes: int
    d2h_bytes: int


@dataclass
class ReductionPlan:
    """partial result를 최종 결과로 합치는 방법 (§18, §19)."""

    policy: str
    owner_plane: int
    ordering: str
    partial_outputs: int
    reduction_bytes: int
    reduction_rounds: int


@dataclass
class DescriptorPlan:
    """Phase 3 (§53) — descriptor/command/launch 재사용 계획.

    descriptor는 (device, round) 작업 하나의 전송·실행 metadata다.
    """

    descriptor_build_count: int      # 새로 구성하는 descriptor 수
    descriptor_transfer_bytes: int   # descriptor 전송량 (§23의 metadata 항)
    descriptor_reuse_factor: float   # 전체 작업 수 / build 수
    command_build_count: int
    kernel_launch_count: int
    device_alloc_count: int


@dataclass
class PIMPlan:
    """Config를 실제 hardware에 배치한 결과 (§14, §51)."""

    execution_tile_shape: tuple      # (T_M, T_N, T_K)                (§15)
    execution_grid: tuple            # (E_M, E_N, E_K) logical grid   (§15)
    active_device_grid: tuple        # (P_M, P_N, P_K)                (§16)
    num_rounds: int                  # device-set 재사용 round 수      (§17)
    k_windows_per_plane: int         # spatial K plane 안 temporal window 수
    device_assignments: list         # round별 [(device, e_m, e_n, plane), ...] (§17)
    transfer_plan: TransferPlan
    reduction_plan: ReductionPlan
    descriptor_plan: DescriptorPlan


@dataclass
class PIMPlanFeatures:
    """Plan에서 계산한 PIM-specific feature (§20~§27, §51)."""

    effective_utilization: float
    load_imbalance: float
    h2d_bytes: int
    d2h_bytes: int
    a_replication_bytes: int
    b_replication_bytes: int
    reduction_bytes: int
    reduction_rounds: int
    padding_ratio: float
    scheduling_rounds: int
    local_memory_pressure: float
    device_memory_pressure: float
    # §27 feature vector가 추가로 요구하는 값
    active_units: int
    execution_tile_count: int
    # Phase 2 (§53) — resident operand의 1회 placement 총량
    a_resident_bytes: int = 0
    b_resident_bytes: int = 0
    # Phase 3 (§53) — descriptor/launch 재사용
    descriptor_build_count: int = 0
    descriptor_transfer_bytes: int = 0
    descriptor_reuse_factor: float = 1.0
    command_build_count: int = 0
    kernel_launch_count: int = 0
    device_alloc_count: int = 0


@dataclass
class PIMBenchmarkResult:
    """실제 실행 결과 (§30, §32, §51).

    objective는 항상 실제 wall-clock total_ms이다 (§30). 각 breakdown은
    objective가 아니라 설명과 학습을 위한 auxiliary information이다.
    """

    total_ms: float
    stage_ms: float = 0.0
    h2d_ms: float = 0.0
    compute_ms: float = 0.0
    reduction_ms: float = 0.0
    d2h_ms: float = 0.0
    launch_ms: float = 0.0
    compile_status: str = "ok"
    correctness: bool = True
    # tune()이 측정 후 채운다.
    config: PIMConfig = None
    plan: PIMPlan = None
    plan_features: PIMPlanFeatures = None

    @property
    def ok(self):
        """compile/runtime 실패는 BAD sample로만 쓰인다 (§35)."""
        return self.compile_status == "ok" and self.correctness
