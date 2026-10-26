"""Inductor TritonTemplates for one semantic GEMM or GEMV reduction.

The source expresses only the output grid and the complete semantic K
reduction. ``p_k`` and ``tile_k`` are PIM Common IR scheduling values passed
as backend options; they are absent from the source grid and output ABI.
A GEMM with one output row or column is a GEMV; its template widens the
matrix and the vector before multiplying, which lowers to linalg.matvec.
"""

import sympy

from torch._inductor.select_algorithm import (
    SymbolicGridFn,
    TritonTemplate,
    TritonTemplateKernel,
)
from torch._inductor.utils import sympy_str


TORCH_GEMM_TEMPLATE_VERSION = "torch-gemm-semantic-k-v2"
TORCH_INT_MM_TEMPLATE_VERSION = TORCH_GEMM_TEMPLATE_VERSION


@SymbolicGridFn
def pim_gemm_grid(m, n, meta, *, cdiv):
    """The source grid contains only independent M/N output blocks."""
    return (
        cdiv(m, meta["BLOCK_M"]),
        cdiv(n, meta["BLOCK_N"]),
        1,
    )


class PIMTemplateKernel(TritonTemplateKernel):
    """Pass static input shapes and strides as runtime scalar arguments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_extra_template_env_fns(self.rt_size, self.rt_stride)

    def _runtime_scalar(self, expr) -> str:
        expr = sympy.expand(expr)
        if expr.free_symbols:
            return sympy_str(self.rename_indexing(expr))
        return self.args._lookup("ks", self.args.sizevars, expr)

    def rt_size(self, name: str, index: int) -> str:
        return self._runtime_scalar(self.named_input_nodes[name].get_size()[index])

    def rt_stride(self, name: str, index: int) -> str:
        strides = self.get_stride_and_maybe_freeze_layout(
            self.named_input_nodes[name])
        return self._runtime_scalar(strides[index])


class PIMTritonTemplate(TritonTemplate):
    kernel_type = PIMTemplateKernel


PIM_GEMM_SOURCE = r"""
{{def_kernel("A", "B")}}
    M = {{rt_size("A", 0)}}
    N = {{rt_size("B", 1)}}
    K = {{rt_size("A", 1)}}

    stride_am = {{rt_stride("A", 0)}}
    stride_ak = {{rt_stride("A", 1)}}
    stride_bk = {{rt_stride("B", 0)}}
    stride_bn = {{rt_stride("B", 1)}}

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        {% if ACC_TYPE == "tl.int32" %}
        acc += tl.dot(a, b)
        {% else %}
        acc += tl.dot(a, b, input_precision="ieee", out_dtype=ACC_TYPE)
        {% endif %}
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    idx_m = offs_m[:, None]
    idx_n = offs_n[None, :]
    mask = (idx_m < M) & (idx_n < N)
    {{store_output(("idx_m", "idx_n"), "acc", "mask",
                   val_shape=("BLOCK_M", "BLOCK_N"))}}
"""


pim_gemm_template = PIMTritonTemplate(
    name="pim_semantic_gemm",
    grid=pim_gemm_grid,
    source=PIM_GEMM_SOURCE,
)


@SymbolicGridFn
def pim_gemv_grid(rows, meta, *, cdiv):
    """The source grid contains only independent blocks of output rows."""
    return (cdiv(rows, meta["BLOCK_M"]), 1, 1)


# A (M × K)·B (K × N)에서 ROW_VECTOR이면 M = 1이다: 행렬은 Bᵀ (N × K)를 B의 stride로 읽고
# vector는 A의 한 행이다. 아니면 N = 1이다: 행렬은 A, vector는 B의 한 열이다.
# 출력은 행 수만큼의 1차원 buffer이고 lowering이 원래 (1, N) 또는 (M, 1)로 본다.
PIM_GEMV_SOURCE = r"""
{{def_kernel("A", "B")}}
    {% if ROW_VECTOR %}
    ROWS = {{rt_size("B", 1)}}
    K = {{rt_size("B", 0)}}
    stride_row = {{rt_stride("B", 1)}}
    stride_k = {{rt_stride("B", 0)}}
    stride_vector = {{rt_stride("A", 1)}}
    matrix_ptr = B
    vector_ptr = A
    {% else %}
    ROWS = {{rt_size("A", 0)}}
    K = {{rt_size("A", 1)}}
    stride_row = {{rt_stride("A", 0)}}
    stride_k = {{rt_stride("A", 1)}}
    stride_vector = {{rt_stride("B", 0)}}
    matrix_ptr = A
    vector_ptr = B
    {% endif %}

    offs_r = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M,), dtype=ACC_TYPE)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k = k_block * BLOCK_K + offs_k
        matrix = tl.load(
            matrix_ptr + offs_r[:, None] * stride_row + k[None, :] * stride_k,
            mask=(offs_r[:, None] < ROWS) & (k[None, :] < K),
            other=0,
        )
        vector = tl.load(vector_ptr + k * stride_vector, mask=k < K, other=0)
        # 곱하기 전에 누산 type으로 넓힌다. INT8은 lowering이 INT8 matvec로 만든다.
        acc += tl.sum(matrix.to(ACC_TYPE) * vector.to(ACC_TYPE)[None, :], axis=1)

    idx_r = offs_r
    mask = idx_r < ROWS
    {{store_output(("idx_r",), "acc", "mask", val_shape=("BLOCK_M",))}}
"""


pim_gemv_template = PIMTritonTemplate(
    name="pim_semantic_gemv",
    grid=pim_gemv_grid,
    source=PIM_GEMV_SOURCE,
)
