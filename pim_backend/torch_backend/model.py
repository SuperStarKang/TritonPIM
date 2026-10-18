"""FX layer planning and synchronous multi-instance GEMM execution."""

from dataclasses import dataclass, replace
import operator

import torch
from torch import fx
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx.passes.split_module import split_module
from torch.utils import _pytree

from ._autotuner import PIMProblem
from .backend import PIMInductorBackend
from .execution import select_layer_config
from .fx_normalization import normalize_fx_gemms
from .fx_packing import PackedGemm, gemm_operands, pack_gemms


# weight에서 출발해 weight만 읽는 view/copy는 freeze 시 미리 계산해 상수로 둔다.
_WEIGHT_VIEWS = frozenset((
    torch.ops.aten.t.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.detach.default,
    torch.ops.aten._to_copy.default,
))


def _freeze_graph(exported_program):
    """Replace parameters, buffers and their weight-only views with graph constants.

    Returns the frozen GraphModule and, per constant node, the frozen tensor.
    """
    if any(spec.kind != OutputKind.USER_OUTPUT
           for spec in exported_program.graph_signature.output_specs):
        raise ValueError("model layers require an inference graph without input or buffer mutation")
    graph = fx.Graph()
    root = torch.nn.Module()
    copied = {}
    values = {}
    frozen = {}
    inputs = {spec.arg.name: spec for spec in exported_program.graph_signature.input_specs}
    for node in exported_program.graph_module.graph.nodes:
        value = None
        if node.op == "placeholder":
            spec = inputs[node.name]
            if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                table = (exported_program.constants
                         if spec.kind == InputKind.BUFFER and spec.persistent is False
                         else exported_program.state_dict)
                value = table[spec.target]
            elif spec.kind == InputKind.CONSTANT_TENSOR:
                value = exported_program.constants[spec.target]
            elif spec.kind != InputKind.USER_INPUT:
                raise ValueError(f"unsupported exported input kind: {spec.kind}")
            if value is not None:
                if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                    raise ValueError(f"frozen input {spec.target!r} must be a CPU tensor")
                value = value.detach()
        elif node.op == "call_function" and node.target in _WEIGHT_VIEWS:
            dependencies = list(node.all_input_nodes)
            if dependencies and all(dependency in values for dependency in dependencies):
                value = node.target(*fx.map_arg(node.args, values.__getitem__),
                                    **fx.map_arg(node.kwargs, values.__getitem__))
        elif node.op == "get_attr":
            raise ValueError("model layers expect the lifted graph of a torch.export.ExportedProgram")

        if value is not None:
            root.register_buffer("resident_" + node.name, value)
            replacement = graph.get_attr("resident_" + node.name)
            replacement.name = node.name
            replacement.meta = node.meta.copy()
            copied[node] = replacement
            values[node] = value
            frozen[replacement] = value
        else:
            copied[node] = graph.node_copy(node, copied.__getitem__)
    return fx.GraphModule(root, graph), frozen


@dataclass
class LayerPlan:
    """One Torch block split into CPU partitions and PIM stages."""

    index: int
    graph: fx.GraphModule
    output_spec: object
    stages: tuple
    example_inputs: tuple
    exported_graph: fx.GraphModule
    spec: object = None
    backend_plan: object = None
    weight_bytes_per_unit: int = 0

    def plan_model(self, spec, runtime):
        self.spec = spec
        for stage in self.stages:
            stage.prepare(spec, runtime)

    def compile_cpu(self):
        leaves, _ = _pytree.tree_flatten((self.example_inputs, {}))
        outputs = CompileModelCPU(self.graph).run(*leaves)
        return _pytree.tree_unflatten(list(outputs), self.output_spec)

    def run_model(self, *inputs):
        leaves, _ = _pytree.tree_flatten((inputs, {}))
        with torch.no_grad():
            outputs = self.graph(*leaves)
        return _pytree.tree_unflatten(list(outputs), self.output_spec)


class _LayerGemm(torch.nn.Module):
    def __init__(self, name, gemms, weights, torch_config, sources, origin):
        super().__init__()
        self.name = name
        # 이 stage가 계산하는 원래 GEMM 이름 (packing으로 여러 개일 수 있다)
        self.sources = tuple(sources)
        # packing으로 이어 붙인 GEMM이면 그 축과 member shape (fx_packing.PackedGemm.origin)
        self.origin = origin
        self.gemms = tuple(gemms)
        self.weights = tuple(weights)
        self.weight_mask = tuple(weight is not None for weight in weights)
        self.torch_config = torch_config
        self.batched_operand = gemms[0].op == "aten.bmm.default"
        self.instance_count = gemms[0].instance_count if self.batched_operand else len(gemms)
        self.shape = gemms[0].mnk
        self.input_dtype = gemms[0].left.dtype
        self.output_dtype = gemms[0].output_dtype
        self.operation = torch.ops.aten._int_mm.default if self.input_dtype == torch.int8 else torch.ops.aten.mm.default

    def prepare(self, spec, runtime):
        self.runtime = runtime
        if self.batched_operand and any(self.weight_mask):
            raise ValueError("resident batched weights require expert placement support")
        self.config, self.p_b, self.batch_rounds, self.mnk_rounds = select_layer_config(
            PIMProblem(*self.shape, input_dtype=str(self.input_dtype).removeprefix("torch."),
                       output_dtype=str(self.output_dtype).removeprefix("torch.")),
            spec, self.torch_config, self.instance_count, has_weights=any(self.weight_mask),
        )
        self.weights = tuple(weight.contiguous() if weight is not None else None
                             for weight in self.weights)
        m, n, k = self.shape
        physical = tuple(weight if weight is not None else torch.empty(shape, dtype=self.input_dtype)
                         for weight, shape in zip(self.weights, ((m, k), (k, n))))
        graph = fx.Graph()
        left = graph.placeholder("left")
        right = graph.placeholder("right")
        graph.output(graph.call_function(self.operation, (left, right)))
        compiled = torch.compile(
            fx.GraphModule(torch.nn.Module(), graph),
            backend=PIMInductorBackend(replace(self.torch_config, spec=spec, config=self.config,
                                               autotune_cache_path=None)),
            fullgraph=True, dynamic=False,
        )
        with runtime.compile_invocation(self, physical):
            compiled(*physical)
        # Weight upload uses only weight arguments; scratch pointers are never read.
        self.preparation_matrices = (tuple(
            weight if weight is not None else torch.empty((1, 1), dtype=self.input_dtype)
            for weight in self.weights),)
        self.preparation_output = torch.empty((1, 1), dtype=self.output_dtype)

    def reference(self, *inputs):
        if self.batched_operand:
            return torch.bmm(*inputs)
        outputs = tuple(self.operation(inputs[index], inputs[index + 1])
                        for index in range(0, len(inputs), 2))
        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, *inputs):
        if self.batched_operand:
            matrices = tuple((inputs[0][instance].contiguous(), inputs[1][instance].contiguous())
                             for instance in range(self.instance_count))
        else:
            matrices = tuple(tuple(self.weights[operand] if self.weight_mask[operand]
                                   else inputs[2 * instance + operand].contiguous()
                                   for operand in range(2))
                             for instance in range(self.instance_count))
        output = torch.empty((self.instance_count, self.shape[0], self.shape[1]),
                             dtype=self.output_dtype)
        self.runtime.run(self, matrices, output)
        if self.batched_operand:
            return output
        return output[0] if self.instance_count == 1 else tuple(output.unbind(0))


class _ReferenceLayer(fx.Interpreter):
    def call_module(self, target, args, kwargs):
        module = self.module.get_submodule(target)
        if isinstance(module, _LayerGemm):
            return module.reference(*args)
        if isinstance(module, fx.GraphModule):
            return _ReferenceLayer(module).run(*args)
        return super().call_module(target, args, kwargs)


class CompileModelCPU(fx.Interpreter):
    def call_module(self, target, args, kwargs):
        module = self.module.get_submodule(target)
        if any(isinstance(child, _LayerGemm) for child in module.modules()):
            return _ReferenceLayer(module).run(*args)
        compiled = torch.compile(module, backend="inductor", fullgraph=True, dynamic=False)
        self.module.set_submodule(target, compiled)
        return compiled(*args, **kwargs)


def build_layer_plan(exported_program, example_inputs, *, configs, index, spec):
    """Analyze one exported block without compiling it or acquiring hardware.

    configs maps original GEMM names to PIMTorchConfig. GEMMs are first packed on
    the frozen graph (fx_packing, decided by spec's target model) unless their
    config sets pack=False; GEMMs packed together must share one config. Independent activation GEMMs of the same
    shape/config are then grouped; weight GEMMs remain stationary single-instance
    stages. aten.bmm uses its explicit operand batch dimension.
    """
    _, specification = _pytree.tree_flatten((example_inputs, {}))
    if specification != exported_program.call_spec.in_spec:
        raise ValueError("example_inputs must match the exported positional inputs")
    normalized, mappings = normalize_fx_gemms(exported_program, allow_bmm=True)
    if set(configs) != {mapping.original.node for mapping in mappings}:
        raise ValueError("configs must contain every original GEMM exactly once")
    frozen_graph, frozen = _freeze_graph(normalized)
    gemms = pack_gemms(frozen_graph, frozen, {
        mapping.normalized.node: PackedGemm(mapping, (mapping.original.node,), None) for mapping in mappings
    }, spec, frozenset(name for name, config in configs.items() if not config.pack))
    for entry in gemms.values():
        if len({configs[source] for source in entry.sources}) != 1:
            raise ValueError(f"packed GEMMs {entry.sources} must share one config")
    depths = {}
    groups = {}
    group_for_node = {}
    for node in frozen_graph.graph.nodes:
        depth = max((depths[argument] for argument in node.all_input_nodes), default=0)
        if node.name in gemms:
            mapping = gemms[node.name].mapping
            gemm = mapping.normalized
            depth += 1
            operands = gemm_operands(node)
            if any(operand in frozen for operand in operands) or gemm.op == "aten.bmm.default":
                key = (node.name,)
            else:
                key = (depth, gemm.op, gemm.mnk, gemm.left.dtype, configs[mapping.original.node])
            if key not in groups:
                groups[key] = []
            groups[key].append(node)
            group_for_node[node] = key
        depths[node] = depth

    graph = fx.Graph()
    copied = {}
    stages = []

    def emit(node):
        if node in copied:
            return copied[node]
        if node in group_for_node:
            nodes = groups[group_for_node[node]]
            arguments = tuple(emit(operand) for member in nodes for operand in gemm_operands(member))
            mapping = gemms[nodes[0].name].mapping
            operands = gemm_operands(nodes[0])
            stage = _LayerGemm(
                mapping.original.node,
                tuple(gemms[member.name].mapping.normalized for member in nodes),
                tuple(frozen[operand] if operand in frozen else None for operand in operands),
                configs[mapping.original.node],
                tuple(source for member in nodes for source in gemms[member.name].sources),
                gemms[nodes[0].name].origin,
            )
            name = "pim_stage_" + str(len(stages))
            frozen_graph.add_module(name, stage)
            stages.append(stage)
            call = graph.call_module(name, arguments)
            if len(nodes) == 1:
                call.meta = nodes[0].meta.copy()
                copied[nodes[0]] = call
            else:
                call.meta["val"] = tuple(member.meta["val"] for member in nodes)
                for instance, member in enumerate(nodes):
                    value = graph.call_function(operator.getitem, (call, instance))
                    value.meta = member.meta.copy()
                    copied[member] = value
            return copied[node]
        copied[node] = graph.node_copy(node, emit)
        return copied[node]

    for node in frozen_graph.graph.nodes:
        emit(node)
    grouped = fx.GraphModule(frozen_graph, graph)
    partitions = {}
    partition = 0
    for node in grouped.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        if node.op == "call_module" and isinstance(grouped.get_submodule(node.target), _LayerGemm):
            partition += 1
            partitions[node] = partition
            partition += 1
        else:
            partitions[node] = partition
    split = split_module(grouped, grouped, partitions.__getitem__,
                         keep_original_order=True, keep_original_node_name=True)
    return LayerPlan(
        index, split, exported_program.call_spec.out_spec, tuple(stages),
        tuple(example_inputs), exported_program.graph_module,
    )
