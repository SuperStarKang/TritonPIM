"""CPU FP32/BF16 mm and INT8 _int_mm lowering to a PIM GEMM or GEMV template."""

import logging

import sympy
import torch

from torch._inductor import ir
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor.lowering import _register_lowering, lowerings

from ._autotuner import PIMProblem
from .config import PIMTorchConfig, UnsupportedPIMGemm
from .execution import gemm_problem, resolve_execution
from .template import pim_gemm_template, pim_gemv_template

aten = torch.ops.aten
log = logging.getLogger(__name__)


def _static_int(value, what):
    expression = sympy.expand(value)
    if (
        expression.free_symbols
        or expression.is_integer is not True
        or expression <= 0
    ):
        raise UnsupportedPIMGemm(
            f"{what} must be a positive static integer, got {expression}"
        )
    return int(expression)


def validate_pim_gemm(
    mat1, mat2, m, n, k, *, input_dtype, output_dtype,
) -> PIMProblem:
    for name, operand in (("mat1", mat1), ("mat2", mat2)):
        if len(operand.get_size()) != 2:
            raise UnsupportedPIMGemm(
                f"{name} must have rank 2, got {list(operand.get_size())}"
            )
        if operand.get_dtype() != input_dtype:
            raise UnsupportedPIMGemm(
                f"{name} must have dtype {input_dtype}, "
                f"got {operand.get_dtype()}"
            )
        device = operand.get_device()
        if device is None or device.type != "cpu":
            raise UnsupportedPIMGemm(
                f"{name} must be a CPU tensor, got {device}"
            )

    return gemm_problem(
        _static_int(m, "M"), _static_int(n, "N"), _static_int(k, "K"),
        input_dtype=str(input_dtype).removeprefix("torch."),
        output_dtype=str(output_dtype).removeprefix("torch."),
    )


def _unwrap_template_buffer(node):
    while isinstance(node, (ir.TensorBox, ir.StorageBox)):
        node = node.data
    if not isinstance(node, ir.TritonTemplateBuffer):
        raise AssertionError("PIM template buffer not found")
    return node


def is_pim_template_buffer(node) -> bool:
    """scheduling이 PIM template node를 알아보는 유일한 표식."""
    return hasattr(node, "_pim_template") and node._pim_template is True


def generate_pim_gemm_template(mat1, mat2, layout, execution):
    problem = execution.problem
    caller = pim_gemm_template.generate(
        input_nodes=(mat1, mat2),
        layout=layout,
        num_stages=execution.torch_config.num_stages,
        num_warps=execution.torch_config.num_warps,
        call_sizes=[sympy.Integer(problem.M), sympy.Integer(problem.N)],
        triton_meta={
            "compile_backend": "pim",
            "backend_options": execution.backend_options(),
        },
        **execution.template_meta(),
    )
    if caller is None:
        raise UnsupportedPIMGemm(
            f"PIM GEMM template failed to generate for {problem}"
        )

    node = caller.output_node()
    buffer = _unwrap_template_buffer(node)
    buffer._pim_template = True
    # Fusion은 Common IR이 fused compute/access 의미를 받는 별도 단계에서 연다.
    buffer.allow_epilogue_fusion = False
    buffer.allow_prologue_fusion = False
    return node


def generate_pim_gemv_template(mat1, mat2, layout, execution, *, row_vector):
    """GEMV template into a buffer of the output rows, viewed back as the GEMM output."""
    problem = execution.problem
    rows = ir.FixedLayout(layout.device, layout.dtype, [sympy.Integer(problem.M)], [sympy.Integer(1)])
    caller = pim_gemv_template.generate(
        input_nodes=(mat1, mat2),
        layout=rows,
        num_stages=execution.torch_config.num_stages,
        num_warps=execution.torch_config.num_warps,
        call_sizes=[sympy.Integer(problem.M)],
        triton_meta={
            "compile_backend": "pim",
            "backend_options": execution.backend_options(),
        },
        ROW_VECTOR=row_vector,
        **execution.template_meta(),
    )
    if caller is None:
        raise UnsupportedPIMGemm(
            f"PIM GEMV template failed to generate for {problem}"
        )

    node = caller.output_node()
    buffer = _unwrap_template_buffer(node)
    buffer._pim_template = True
    buffer.allow_epilogue_fusion = False
    buffer.allow_prologue_fusion = False
    return lowerings[aten.view.default](node, list(layout.size))


def make_pim_gemm_lowering(config: PIMTorchConfig, output_dtypes):
    """output_dtypes maps each accepted input dtype to the GEMM output dtype."""

    def pim_gemm(mat1, mat2, *, layout=None):
        input_dtype = mat1.get_dtype()
        if input_dtype not in output_dtypes:
            raise UnsupportedPIMGemm(
                f"mat1 must have one of the dtypes {tuple(output_dtypes)}, got {input_dtype}"
            )
        output_dtype = output_dtypes[input_dtype]
        m, n, k, out_layout, mat1_r, mat2_r = mm_args(
            mat1,
            mat2,
            layout=layout,
            out_dtype=output_dtype if layout is None else None,
        )
        problem = validate_pim_gemm(
            mat1_r,
            mat2_r,
            m,
            n,
            k,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )

        if out_layout.dtype != output_dtype:
            raise UnsupportedPIMGemm(
                f"output layout must have dtype {output_dtype}, "
                f"got {out_layout.dtype}"
            )
        if out_layout.device is None or out_layout.device.type != "cpu":
            raise UnsupportedPIMGemm("output layout must be on CPU")

        execution = resolve_execution(config, problem)
        log.info(
            "PIM %s -> %s (M,N,K)=(%d,%d,%d): %s",
            problem.input_dtype,
            problem.output_dtype,
            problem.M,
            problem.N,
            problem.K,
            execution.describe(),
        )

        if problem.compute_kind == "matvec":
            # 출력 행이 하나면(M = 1) 행렬이 B 쪽이다. GEMV template은 행렬을 실행 시 stride로
            # 읽고 host 배치(AiMX stage, UPMEM pack)도 stride를 따르므로, checkpoint weight의 전치
            # view를 복사하지 않고 받는다. 연속 사본을 끼우면 model stage가 넘기는 weight와 kernel
            # 인자가 달라진다.
            return generate_pim_gemv_template(
                mat1_r, mat2_r, out_layout, execution,
                row_vector=_static_int(m, "M") == 1,
            )
        # GEMM의 PIM 데이터 배치는 row-major 연속을 전제로 한다. Inductor가 고른 layout이
        # 그렇지 않으면 복사를 하나 끼워 맞춘다 (연산 자체는 여전히 PIM에서 돈다).
        mat1_r = ir.ExternKernel.require_contiguous(mat1_r)
        mat2_r = ir.ExternKernel.require_contiguous(mat2_r)
        return generate_pim_gemm_template(
            mat1_r, mat2_r, out_layout, execution,
        )

    return pim_gemm


class PIMGemmLowering:
    """Install PIM GEMM lowerings for the surrounding compile_fx call."""

    def __init__(self, config: PIMTorchConfig):
        self.config = config
        self._saved = None

    def __enter__(self):
        import torch._inductor.kernel.mm

        replacements = {}
        for operation, output_dtypes in (
            (aten.mm, {torch.float32: torch.float32, torch.bfloat16: torch.bfloat16}),
            (aten._int_mm, {torch.int8: torch.int32}),
        ):
            wrapped = _register_lowering(
                operation.default,
                make_pim_gemm_lowering(self.config, output_dtypes),
                broadcast=False,
                type_promotion_kind=None,
                convert_input_to_bool=False,
                lowering_dict={},
            )
            replacements[operation] = wrapped
            replacements[operation.default] = wrapped

        # 등록 준비가 끝난 후 전역 표를 변경한다.
        self._saved = {key: lowerings[key] for key in replacements}
        lowerings.update(replacements)
        return self

    def __exit__(self, exc_type, exc, tb):
        lowerings.update(self._saved)
        self._saved = None
        return False
