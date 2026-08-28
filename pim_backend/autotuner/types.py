"""Minimal values used by the PIM Common IR autotuner."""

from dataclasses import dataclass


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
    """GEMM problem currently recognized by the autotuner."""

    M: int
    N: int
    K: int
    input_dtype: str = "int8"
    output_dtype: str = "int32"
    a_layout: str = "row_major"
    b_layout: str = "row_major"

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

@dataclass
class PIMBenchmarkResult:
    """Measured end-to-end result for one Common IR configuration."""

    total_ms: float
    correctness: bool = True

    @property
    def ok(self):
        return self.correctness
