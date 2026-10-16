from dataclasses import dataclass
import math

import torch
from torch import fx
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx.operator_schemas import normalize_function


aten = torch.ops.aten

_SUPPORTED_GEMMS = frozenset((
    aten.linear.default,
    aten.mm.default,
    aten.addmm.default,
    aten._int_mm.default,
))

# 실수 GEMM의 입력 dtype -> (출력, accumulator). BF16은 Triton이 BF16 누산을 받지 않으므로
# FP32로 누산하고 출력에서 BF16으로 바꾼다. _int_mm은 언제나 int8 -> int32다.
FLOAT_GEMM_DTYPES = {
    torch.float32: (torch.float32, torch.float32),
    torch.bfloat16: (torch.bfloat16, torch.float32),
}

_GEMM_SCHEMAS = frozenset((
    "aten::linear",
    "aten::mm",
    "aten::addmm",
    "aten::_int_mm",
    "aten::matmul",
    "aten::bmm",
    "aten::baddbmm",
))

_WEIGHT_TRANSFORMS = frozenset((
    aten.t.default,
    aten.transpose.int,
    aten.permute.default,
    aten.view.default,
    aten.reshape.default,
    aten.contiguous.default,
    aten.clone.default,
    aten.detach.default,
    aten._to_copy.default,
))


@dataclass(frozen=True)
class OperandInfo:
    node: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    weight_target: str | None
    weight_kind: str | None
    weight_path: tuple[str, ...]


@dataclass(frozen=True)
class GemmInfo:
    node: str
    op: str
    module_paths: tuple[str, ...]
    left: OperandInfo
    right: OperandInfo
    additive_input: OperandInfo | None
    mnk: tuple[int, int, int]
    output_shape: tuple[int, ...]
    output_dtype: torch.dtype
    accumulator_dtype: torch.dtype
    reshape_left: bool
    transpose_right: bool
    alpha: int | float
    beta: int | float
    reads_additive_input: bool

    @property
    def instance_count(self):
        return self.output_shape[0] if self.op == "aten.bmm.default" else 1


def _tensor(node):
    if not isinstance(node, fx.Node):
        raise TypeError("a tensor operand must be an FX node")

    tensor = node.meta["val"]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{node.name}: tensor metadata is required")
    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        raise ValueError(f"{node.name}: a CPU strided tensor is required")
    if any(type(size) is not int or size < 1 for size in tensor.shape):
        raise ValueError(f"{node.name}: positive static dimensions are required")
    if any(type(stride) is not int for stride in tensor.stride()):
        raise ValueError(f"{node.name}: static strides are required")
    return tensor


def _operand(node, weights):
    tensor = _tensor(node)

    if node in weights:
        weight_target, weight_kind, weight_path = weights[node]
    else:
        weight_target = None
        weight_kind = None
        weight_path = ()

    return OperandInfo(
        node=node.name,
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        weight_target=weight_target,
        weight_kind=weight_kind,
        weight_path=weight_path,
    )


def _analyze_gemm(node, weights):
    normalized = normalize_function(
        node.target,
        node.args,
        dict(node.kwargs),
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        raise ValueError(f"{node.name}: cannot resolve operator arguments")
    arguments = normalized.kwargs

    is_linear = node.target == aten.linear.default
    is_integer = node.target == aten._int_mm.default
    is_bmm = node.target == aten.bmm.default

    if is_linear:
        left = _operand(arguments["input"], weights)
        right = _operand(arguments["weight"], weights)
        additive = (
            None
            if arguments["bias"] is None
            else _operand(arguments["bias"], weights)
        )
        alpha = 1
        beta = 0 if additive is None else 1
    elif node.target == aten.addmm.default:
        left = _operand(arguments["mat1"], weights)
        right = _operand(arguments["mat2"], weights)
        additive = _operand(arguments["input"], weights)
        alpha = arguments["alpha"]
        beta = arguments["beta"]
    else:
        left = _operand(arguments["input"], weights)
        right = _operand(arguments["mat2"], weights)
        additive = None
        alpha = 1
        beta = 0

    if type(alpha) not in (int, float) or type(beta) not in (int, float):
        raise ValueError(f"{node.name}: alpha and beta must be static real scalars")

    if is_integer:
        input_dtype = torch.int8
        output_dtype = accumulator_dtype = torch.int32
    else:
        input_dtype = left.dtype
        if input_dtype not in FLOAT_GEMM_DTYPES:
            raise ValueError(
                f"{node.name}: inputs must be one of {tuple(FLOAT_GEMM_DTYPES)}, got {left.dtype}"
            )
        output_dtype, accumulator_dtype = FLOAT_GEMM_DTYPES[input_dtype]

    if left.dtype != input_dtype or right.dtype != input_dtype:
        raise ValueError(
            f"{node.name}: inputs must be {input_dtype}, "
            f"got {left.dtype} and {right.dtype}"
        )
    if len(right.shape) != (3 if is_bmm else 2):
        raise ValueError(f"{node.name}: the right operand has an incompatible rank")

    if is_bmm:
        if len(left.shape) != 3 or left.shape[0] != right.shape[0]:
            raise ValueError(f"{node.name}: bmm requires matching rank-3 batches")
        instances, m, k = left.shape
        n = right.shape[2]
        if right.shape[1] != k:
            raise ValueError(f"{node.name}: bmm reduction dimensions disagree")
        output_shape = (instances, m, n)
    elif is_linear:
        if len(left.shape) < 1:
            raise ValueError(f"{node.name}: linear input must have rank >= 1")
        m = math.prod(left.shape[:-1])
        n, k = right.shape
        if left.shape[-1] != k:
            raise ValueError(f"{node.name}: linear reduction dimensions disagree")
        output_shape = left.shape[:-1] + (n,)
    else:
        if len(left.shape) != 2:
            raise ValueError(f"{node.name}: the left operand must have rank 2")
        m, k = left.shape
        n = right.shape[1]
        if right.shape[0] != k:
            raise ValueError(f"{node.name}: GEMM reduction dimensions disagree")
        output_shape = (m, n)

    if additive is not None:
        if additive.dtype != input_dtype:
            raise ValueError(f"{node.name}: additive input has an incompatible dtype")
        if is_linear:
            if additive.shape != (n,):
                raise ValueError(f"{node.name}: linear bias must have shape [{n}]")
        elif tuple(torch.broadcast_shapes(additive.shape, output_shape)) != output_shape:
            raise ValueError(f"{node.name}: addmm input cannot broadcast to {output_shape}")

    output = _tensor(node)
    if tuple(output.shape) != output_shape or output.dtype != output_dtype:
        raise ValueError(f"{node.name}: output metadata disagrees with GEMM semantics")

    return GemmInfo(
        node=node.name,
        op=str(node.target),
        module_paths=tuple(
            path
            for path, _ in node.meta["nn_module_stack"].values()
            if path
        ),
        left=left,
        right=right,
        additive_input=additive,
        mnk=(m, n, k),
        output_shape=output_shape,
        output_dtype=output_dtype,
        accumulator_dtype=accumulator_dtype,
        reshape_left=is_linear and len(left.shape) != 2,
        transpose_right=is_linear,
        alpha=alpha,
        beta=beta,
        reads_additive_input=additive is not None and beta != 0,
    )


def analyze_fx_gemms(exported_program, *, allow_bmm=False):
    if not isinstance(exported_program, torch.export.ExportedProgram):
        raise TypeError("analyze_fx_gemms requires a torch.export.ExportedProgram")

    if any(
        spec.kind != OutputKind.USER_OUTPUT
        for spec in exported_program.graph_signature.output_specs
    ):
        raise ValueError("an inference graph without input/buffer mutation is required")

    input_specs = {
        spec.arg.name: spec
        for spec in exported_program.graph_signature.input_specs
    }
    weights = {}
    gemms = []

    for node in exported_program.graph_module.graph.nodes:
        if node.op == "placeholder":
            spec = input_specs[node.name]

            if spec.kind in (
                InputKind.PARAMETER,
                InputKind.BUFFER,
                InputKind.CONSTANT_TENSOR,
            ):
                if not isinstance(spec.target, str):
                    raise ValueError(f"{node.name}: missing state target")
                weights[node] = (spec.target, spec.kind.name, (node.name,))
            elif spec.kind != InputKind.USER_INPUT:
                raise ValueError(f"{node.name}: unsupported input kind {spec.kind}")
            continue

        if node.op == "output":
            continue
        if node.op != "call_function":
            raise ValueError(f"{node.name}: a lifted ATen export graph is required")

        if isinstance(node.target, torch._ops.OpOverload):
            if node.target._schema.is_mutable:
                raise ValueError(f"{node.name}: mutable operation {node.target}")

        if node.target in _WEIGHT_TRANSFORMS:
            dependencies = tuple(node.all_input_nodes)
            if len(dependencies) == 1 and dependencies[0] in weights:
                target, kind, path = weights[dependencies[0]]
                weights[node] = (target, kind, path + (node.name,))

        if not isinstance(node.target, torch._ops.OpOverload):
            continue
        if node.target._schema.name not in _GEMM_SCHEMAS:
            continue
        if node.target not in _SUPPORTED_GEMMS and not (allow_bmm and node.target == aten.bmm.default):
            raise ValueError(
                f"{node.name}: unsupported GEMM {node.target}; "
                "decompose it into supported operations first"
            )

        gemms.append(_analyze_gemm(node, weights))

    if not gemms:
        raise ValueError("the graph contains no supported GEMM")

    return tuple(gemms)
