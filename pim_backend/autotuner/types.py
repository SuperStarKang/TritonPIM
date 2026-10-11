"""Values shared by PIM Common IR analysis and autotuning."""

from dataclasses import dataclass
from math import prod


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


AXES = ("m", "n", "k")


@dataclass(frozen=True)
class PIMAccess:
    """One buffer placement visible in PIM Common IR.

    ``axes`` is the set of logical axes used by the access.  Axes absent from
    an input access become ``pim.place.shared_over`` after scheduling.
    Each buffer is listed once even when several fused compute operations use
    it inside the same Execution Tile residency.
    """

    direction: str
    axes: tuple
    dtype: str
    layout: str = "row_major"
    irregular: bool = False

    @property
    def element_bytes(self):
        return DTYPE_BYTES[self.dtype]

    def elements(self, extents):
        by_axis = dict(zip(AXES, extents))
        result = 1
        for axis in self.axes:
            result *= by_axis[axis]
        return result

    def key(self):
        return (
            self.direction, self.axes, self.dtype, self.layout,
            self.irregular,
        )


@dataclass(frozen=True)
class PIMProblem:
    """Operation-independent logical axes and buffer access dependencies.

    The current PIM hierarchy uses ``m`` and ``n`` as output-parallel axes and
    ``k`` as the optional reduction axis.  An operation with fewer axes sets
    the unused extent to one.  Empty ``accesses`` preserves the original GEMM
    construction API and is normalized to A(m,k), B(k,n), C(m,n).
    """

    M: int
    N: int
    K: int
    input_dtype: str = "int8"
    output_dtype: str = "int32"
    compute_kind: str = "matmul"
    accesses: tuple = ()

    def __post_init__(self):
        if not self.accesses:
            object.__setattr__(self, "accesses", (
                PIMAccess("in", ("m", "k"), self.input_dtype),
                PIMAccess("in", ("k", "n"), self.input_dtype),
                PIMAccess("out", ("m", "n"), self.output_dtype),
            ))

    @property
    def axis_extents(self):
        return (self.M, self.N, self.K)

    @property
    def input_accesses(self):
        return tuple(
            access for access in self.accesses
            if access.direction in ("in", "inout"))

    @property
    def output_accesses(self):
        return tuple(
            access for access in self.accesses
            if access.direction in ("out", "inout"))

    @property
    def reduction_axes(self):
        inputs = {
            axis for access in self.input_accesses for axis in access.axes}
        return tuple(
            axis for axis in AXES
            if axis in inputs
            and any(axis not in access.axes
                    for access in self.output_accesses)
        )

    @property
    def has_reduction(self):
        return "k" in self.reduction_axes

    @property
    def uses_aimx_matmul_primitive(self):
        return self.compute_kind in ("matmul", "matvec")

    def shared_over(self, access):
        return tuple(
            axis for axis in AXES
            if self.axis_extents[AXES.index(axis)] > 1
            and axis not in access.axes
        )

    def access_key(self):
        return tuple(access.key() for access in self.accesses)

    @classmethod
    def elementwise(cls, M, N=1, *, input_dtypes=("float32",),
                    output_dtype="float32"):
        axes = tuple(
            axis for axis, extent in zip(("m", "n"), (M, N))
            if extent > 1)
        accesses = tuple(
            PIMAccess("in", axes, dtype) for dtype in input_dtypes)
        accesses += (PIMAccess("out", axes, output_dtype),)
        return cls(
            M=M, N=N, K=1,
            input_dtype=input_dtypes[0], output_dtype=output_dtype,
            compute_kind="elementwise",
            accesses=accesses,
        )

    @classmethod
    def matvec(cls, M, K, *, input_dtype="int8",
               output_dtype="int32"):
        return cls(
            M=M, N=1, K=K,
            input_dtype=input_dtype, output_dtype=output_dtype,
            compute_kind="matvec",
            accesses=(
                PIMAccess("in", ("m", "k"), input_dtype),
                PIMAccess("in", ("k",), input_dtype),
                PIMAccess("out", ("m",), output_dtype),
            ),
        )

    @classmethod
    def reduction(cls, M, K, N=1, *, input_dtype="float32",
                  output_dtype="float32"):
        output_axes = tuple(
            axis for axis, extent in zip(("m", "n"), (M, N))
            if extent > 1)
        return cls(
            M=M, N=N, K=K,
            input_dtype=input_dtype, output_dtype=output_dtype,
            compute_kind="reduction",
            accesses=(
                PIMAccess("in", output_axes + ("k",), input_dtype),
                PIMAccess("out", output_axes, output_dtype),
            ),
        )

    @classmethod
    def convolution(cls, spatial, output_channels, reduction_extent, *,
                    input_dtype="int8", output_dtype="int32"):
        return cls(
            M=spatial, N=output_channels, K=reduction_extent,
            input_dtype=input_dtype, output_dtype=output_dtype,
            compute_kind="convolution",
            accesses=(
                PIMAccess("in", ("m", "k"), input_dtype),
                PIMAccess("in", ("k", "n"), input_dtype),
                PIMAccess("out", ("m", "n"), output_dtype),
            ),
        )

    @property
    def input_bytes(self):
        return DTYPE_BYTES[self.input_dtype]

    @property
    def output_bytes(self):
        return DTYPE_BYTES[self.output_dtype]


@dataclass(frozen=True)
class PIMConfig:
    """Values materialized as the PIM Common IR hierarchy and mapping.

    ``block_*`` is inferred from the Triton program, ``group_*`` and ``tile_*``
    become the two hierarchy attributes, and ``p_*`` becomes
    ``#pim.device_mapping<partitions=[p_m,p_n,p_k]>``.
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
    p_m: int
    p_n: int
    p_k: int

    @property
    def execution_extent(self):
        return (
            self.block_m * self.group_m * self.tile_m,
            self.block_n * self.group_n * self.tile_n,
            self.block_k * self.group_k * self.tile_k,
        )

    def key(self):
        return (
            self.block_m, self.block_n, self.block_k,
            self.group_m, self.group_n, self.group_k,
            self.tile_m, self.tile_n, self.tile_k,
            self.p_m, self.p_n, self.p_k,
        )


@dataclass(frozen=True)
class PIMHierarchy:
    """The three Common IR hierarchy levels and their scheduled grid."""

    block_group_extent: tuple
    total_groups: tuple
    execution_tiles: tuple
    rounds: tuple

    @property
    def useful_execution_tiles(self):
        return prod(self.execution_tiles)

    @property
    def scheduling_rounds(self):
        return prod(self.rounds)


def pim_hierarchy(problem, config):
    """Materialize the same group/tile/partition arithmetic as Common IR."""
    blocks = (config.block_m, config.block_n, config.block_k)
    groups = (config.group_m, config.group_n, config.group_k)
    tiles = (config.tile_m, config.tile_n, config.tile_k)
    partitions = (config.p_m, config.p_n, config.p_k)
    block_group_extent = tuple(
        blocks[axis] * groups[axis] for axis in range(3))
    total_groups = tuple(
        ceil_div(problem.axis_extents[axis], block_group_extent[axis])
        for axis in range(3))
    execution_tiles = tuple(
        ceil_div(total_groups[axis], tiles[axis]) for axis in range(3))
    rounds = tuple(
        ceil_div(execution_tiles[axis], partitions[axis])
        for axis in range(3))
    return PIMHierarchy(
        block_group_extent=block_group_extent,
        total_groups=total_groups,
        execution_tiles=execution_tiles,
        rounds=rounds,
    )


@dataclass
class PIMBenchmarkResult:
    """Measured end-to-end result for one Common IR configuration."""

    total_ms: float
    correctness: bool | None = True
    timings: tuple = ()
    physical_allocations: tuple = ()

    @property
    def ok(self):
        # 검증을 생략한 결과도 성능 탐색 후보로 사용한다.
        if self.correctness is None:
            return True
        return self.correctness
