# aten._int_mm → one semantic GEMM template.
#
# 그래프 모양:
#
#     PIM semantic GEMM template ── PIMBackend compile → PIM Common IR
#              │                    → semantic K scheduling/combine
#              ▼
#          C[M,N] int32
#
# [stock tuned_int_mm을 전역 수정하지 않는 이유]
# torch/_inductor/kernel/mm.py의 tuned_int_mm에서 mm_template를 PIM template로
# 바꾸면 PIMInductorBackend를 쓰지 않는 torch.compile까지 전부 PIM으로 간다.
# 여기서는 PIMInductorBackend가 켜져 있는 동안에만 lowerings[aten._int_mm]을
# 갈아 끼우고, 끝나면 되돌린다.
#
# 지원 조건을 어기면 CPU fallback 없이 바로 멈춘다.
#
# [실행 계획은 여기서 정하지 않는다]
# 이 파일이 보는 것은 "그래프가 이 template에 들어올 모양인가"뿐이다. block/
# group/execution tile과 축별 device 분할 수는 execution.resolve_execution이
# autotuner에게서 받아 오고, 그 config가 이 shape에서 성립하는지는 autotuner의
# ConstraintEngine이 판정한다.

import logging

import sympy
import torch

from torch._inductor import ir
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor.lowering import _register_lowering, lowerings

from ._autotuner import PIMProblem
from .config import PIMTorchConfig, UnsupportedPIMIntMM
from .execution import resolve_execution
from .template import pim_gemm_template

aten = torch.ops.aten
log = logging.getLogger(__name__)


def _static_int(value, what):
    expr = sympy.expand(value)
    if expr.free_symbols:
        raise UnsupportedPIMIntMM(
            f"{what} must be static, got {expr}. "
            f"Compile with dynamic=False.")
    return int(expr)


def validate_pim_int_mm(mat1, mat2, m, n, k) -> PIMProblem:
    """PIM이 맡을 수 있는 _int_mm인지 본다 (설계 §4).

    여기서 보는 것은 "그래프가 이 template에 들어올 모양인가"뿐이다. 실행
    계획이 이 shape에서 성립하는지(memory 예산, 축별 분할 수, alignment)는
    autotuner의 ConstraintEngine이 본다 — execution.resolve_execution."""
    if len(mat1.get_size()) != 2 or len(mat2.get_size()) != 2:
        raise UnsupportedPIMIntMM(
            f"PIM _int_mm supports 2-D inputs only, got "
            f"{list(mat1.get_size())} @ {list(mat2.get_size())}")
    for name, node in (("mat1", mat1), ("mat2", mat2)):
        if node.get_dtype() != torch.int8:
            raise UnsupportedPIMIntMM(
                f"{name} dtype must be torch.int8, got {node.get_dtype()}")
        device = node.get_device()
        if device is None or device.type != "cpu":
            raise UnsupportedPIMIntMM(
                f"{name} must live on CPU (PIM is a compile/exec backend, "
                f"not a tensor device), got {device}")

    # dtype과 layout을 problem이 들고 다닌다 — ConstraintEngine의 alignment
    # 검사가 problem.input_bytes / output_bytes를 쓴다.
    return PIMProblem(
        M=_static_int(m, "M"), N=_static_int(n, "N"), K=_static_int(k, "K"),
        input_dtype="int8", output_dtype="int32")


def _unwrap_template_buffer(node):
    """TensorBox.create(...)가 감싼 TritonTemplateBuffer를 꺼낸다."""
    current = node
    while not isinstance(current, ir.TritonTemplateBuffer):
        current = getattr(current, "data", None)
        if current is None:
            raise AssertionError("PIM template buffer not found in output node")
    return current


def is_pim_template_buffer(node) -> bool:
    """scheduling이 PIM template node를 알아보는 유일한 표식."""
    return getattr(node, "_pim_template", False) is True


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
        raise UnsupportedPIMIntMM(
            f"PIM semantic GEMM template failed to generate for {problem}")

    node = caller.output_node()
    buffer = _unwrap_template_buffer(node)
    buffer._pim_template = True
    # Fusion은 Common IR이 fused compute/access 의미를 받는 별도 단계에서 연다.
    buffer.allow_epilogue_fusion = False
    buffer.allow_prologue_fusion = False
    return node


def make_pim_int_mm_lowering(config: PIMTorchConfig):
    def pim_int_mm(mat1, mat2, *, layout=None):
        m, n, k, out_layout, mat1_r, mat2_r = mm_args(
            mat1, mat2, layout=layout, out_dtype=torch.int32)
        problem = validate_pim_int_mm(mat1_r, mat2_r, m, n, k)
        execution = resolve_execution(config, problem)

        log.info("PIM _int_mm (M,N,K)=(%d,%d,%d): %s",
                 problem.M, problem.N, problem.K, execution.describe())

        # PIM 데이터 배치는 row-major 연속을 전제로 한다. Inductor가 고른 layout이
        # 그렇지 않으면 복사를 하나 끼워 맞춘다 (연산 자체는 여전히 PIM에서 돈다).
        mat1_r = ir.ExternKernel.require_contiguous(mat1_r)
        mat2_r = ir.ExternKernel.require_contiguous(mat2_r)

        return generate_pim_gemm_template(
            mat1_r, mat2_r, out_layout, execution)

    return pim_int_mm


def _int_mm_lowering_keys():
    """lowerings에서 aten._int_mm을 가리키는 key 전부.

    register_lowering을 그대로 쓸 수 없는 이유: 그 안의 get_overloads는 이미
    등록된 overload를 건너뛴다. stock tuned_int_mm이 먼저 등록해 두었으므로
    aten._int_mm.default가 그대로 남아, 정작 GraphLowering이 조회하는 key가
    바뀌지 않는다. 그래서 key 목록을 직접 만든다."""
    keys = [aten._int_mm]
    for name in aten._int_mm.overloads():
        keys.append(getattr(aten._int_mm, name))
    return keys


class PIMIntMMLowering:
    """lowerings[aten._int_mm]을 PIM lowering으로 갈아 끼우는 context manager.

    전역 dict를 건드리지만 범위는 compile_fx 호출 하나다. 나가면서 되돌리므로
    PIMInductorBackend를 쓰지 않는 torch.compile에는 영향이 없다."""

    def __init__(self, config: PIMTorchConfig):
        self.config = config
        self._saved = None

    def __enter__(self):
        # stock lowering이 아직 등록 전이면 import로 강제한다.
        import torch._inductor.kernel.mm  # noqa: F401

        keys = _int_mm_lowering_keys()
        self._saved = {key: lowerings.get(key) for key in keys}
        wrapped = _register_lowering(
            aten._int_mm,
            make_pim_int_mm_lowering(self.config),
            broadcast=False,
            type_promotion_kind=None,
            convert_input_to_bool=False,
            # 전역 dict를 여기서 직접 갱신하지 않고 아래에서 key마다 넣는다.
            lowering_dict={},
        )
        for key in keys:
            lowerings[key] = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, value in self._saved.items():
            if value is None:
                lowerings.pop(key, None)
            else:
                lowerings[key] = value
        self._saved = None
        return False
