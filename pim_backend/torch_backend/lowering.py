# aten._int_mm → PIM split-K template + CPU partial reduction.
#
# 그래프 모양:
#
#     zeros[SPLIT_K,M,N] (CPU Cpp)
#              │ (mutated input)
#     PIM split-K template  ── PIMBackend compile → PIM 실행
#              │
#     sum(partial, dim=0)  (CPU Cpp)
#              │
#          C[M,N] int32
#
# [stock tuned_int_mm을 전역 수정하지 않는 이유]
# torch/_inductor/kernel/mm.py의 tuned_int_mm에서 mm_template를 PIM template로
# 바꾸면 PIMInductorBackend를 쓰지 않는 torch.compile까지 전부 PIM으로 간다.
# 여기서는 PIMInductorBackend가 켜져 있는 동안에만 lowerings[aten._int_mm]을
# 갈아 끼우고, 끝나면 되돌린다.
#
# [CPU fallback을 두지 않는 이유]
# strict 모드에서 조건을 어기면 바로 멈춘다. fallback이 있으면 PIM template
# compile이 실패했는데도 결과가 맞아 테스트가 통과한 것처럼 보인다.
#
# [실행 계획은 여기서 정하지 않는다]
# 이 파일이 보는 것은 "그래프가 이 template에 들어올 모양인가"뿐이다. block/
# group/execution tile과 축별 device 분할 수는 plan.resolve_execution_plan이
# autotuner에게서 받아 오고, 그 config가 이 shape에서 성립하는지는 autotuner의
# ConstraintEngine이 판정한다.

import logging

import sympy
import torch

from torch._inductor import ir
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor.lowering import _full, _register_lowering, lowerings

from ._autotuner import PIMProblem
from .config import PIMTorchConfig, UnsupportedPIMIntMM
from .plan import resolve_execution_plan
from .template import pim_split_k_template

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
    autotuner의 ConstraintEngine이 본다 — plan.resolve_execution_plan."""
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
        input_dtype="int8", output_dtype="int32",
        a_layout="row_major", b_layout="row_major")


def _zeroed_partial(device, split_k, m, n):
    """zero로 채운 partial[SPLIT_K,M,N] int32 buffer를 만든다.

    lowering.empty_strided를 쓰지 않는 이유: 그쪽은 _full로 만든 뒤 ranges를
    0으로 덮어 NopKernel(할당만)로 만든다. 여기서는 실제로 0을 채우는 CPU
    커널이 필요하다 — examples/split_k.py의 torch.zeros와 같은 의미다."""
    size = [sympy.Integer(split_k), sympy.expand(m), sympy.expand(n)]
    partial = _full(0, device, torch.int32, size)
    partial.realize()
    buffer = partial.data.data
    assert isinstance(buffer, ir.ComputedBuffer), type(buffer)
    # template이 {{stride("PARTIAL", i)}}로 읽을 stride를 여기서 못박는다.
    buffer.layout = ir.FixedLayout(
        device=device,
        dtype=torch.int32,
        size=size,
        stride=ir.FlexibleLayout.contiguous_strides(size),
    )
    return partial


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


def generate_pim_split_k_template(mat1, mat2, partial, execution):
    problem = execution.problem
    caller = pim_split_k_template.generate(
        input_nodes=(mat1, mat2, partial),
        # 선언상 출력은 쓰이지 않는다. 실제 출력은 mutated input인 partial이다
        # (template.py의 [PARTIAL을 mutated input으로 두는 이유] 참고).
        layout=ir.FixedLayout(
            device=mat1.get_device(),
            dtype=torch.int32,
            size=[sympy.Integer(1)],
            stride=[sympy.Integer(1)],
        ),
        num_stages=execution.torch_config.num_stages,
        num_warps=execution.torch_config.num_warps,
        mutated_inputs=[partial],
        # grid_fn(m, n, meta)가 쓰는 값. 선언상 출력([1])이 아니라 실제 문제
        # 크기를 넘겨야 grid가 맞는다.
        call_sizes=[sympy.Integer(problem.M), sympy.Integer(problem.N)],
        # 이 커널만 PIM으로 컴파일하라는 표식과 그때 쓸 PIM option.
        # triton_heuristics의 patch(_precompile_config)가 이것을 읽는다.
        triton_meta={
            "compile_backend": "pim",
            "backend_options": execution.backend_options(),
        },
        **execution.template_meta(),
    )
    if caller is None:
        raise UnsupportedPIMIntMM(
            f"PIM split-K template failed to generate for {problem}")

    node = caller.output_node()
    buffer = _unwrap_template_buffer(node)
    buffer._pim_template = True
    # 초기에는 template 주변 fusion을 전부 막는다. epilogue/prologue가 붙으면
    # 생성되는 Triton 소스가 split-K kernel과 달라져서, PIM 파이프라인이
    # 실패했을 때 원인이 갈라지지 않는다.
    buffer.allow_epilogue_fusion = False
    buffer.allow_prologue_fusion = False
    return node


def make_pim_int_mm_lowering(config: PIMTorchConfig, fallback):
    def pim_int_mm(mat1, mat2, *, layout=None):
        try:
            m, n, k, _out_layout, mat1_r, mat2_r = mm_args(
                mat1, mat2, layout=layout, out_dtype=torch.int32)
            problem = validate_pim_int_mm(mat1_r, mat2_r, m, n, k)
            execution = resolve_execution_plan(config, problem)
        except UnsupportedPIMIntMM:
            if config.strict:
                raise
            return fallback(mat1, mat2, layout=layout)

        log.info("PIM _int_mm (M,N,K)=(%d,%d,%d): %s",
                 problem.M, problem.N, problem.K, execution.describe())

        # PIM 데이터 배치는 row-major 연속을 전제로 한다. Inductor가 고른 layout이
        # 그렇지 않으면 복사를 하나 끼워 맞춘다 (연산 자체는 여전히 PIM에서 돈다).
        mat1_r = ir.ExternKernel.require_contiguous(mat1_r)
        mat2_r = ir.ExternKernel.require_contiguous(mat2_r)

        partial = _zeroed_partial(
            mat1_r.get_device(), execution.config.p_k, problem.M, problem.N)

        generate_pim_split_k_template(mat1_r, mat2_r, partial, execution)

        # partial을 읽는 쪽은 mutation 이후의 이름을 보게 된다 (MutationOutput).
        # dtype을 명시하지 않으면 torch.sum 규약대로 int64로 승격된다.
        #
        # reduction 위치는 target이 정한다 (autotuner의 TARGET_REDUCTION_POLICY).
        # "host"가 아니면 device가 이미 합친 결과라 여기서 또 합치면 안 된다.
        if execution.config.reduction_policy not in ("host", "none"):
            raise UnsupportedPIMIntMM(
                f"reduction_policy={execution.config.reduction_policy!r} is not "
                f"supported yet — this lowering always adds a host reduction. "
                f"target={config.target}")
        return lowerings[aten.sum](partial, axis=0, dtype=torch.int32)

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
        # stock lowering이 아직 등록 전이면 import로 강제한다 (fallback 확보).
        import torch._inductor.kernel.mm  # noqa: F401

        keys = _int_mm_lowering_keys()
        self._saved = {key: lowerings.get(key) for key in keys}
        fallback = self._saved.get(aten._int_mm.default)

        wrapped = _register_lowering(
            aten._int_mm,
            make_pim_int_mm_lowering(self.config, fallback),
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
