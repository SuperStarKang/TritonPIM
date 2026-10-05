"""FX layer planning and synchronous multi-instance GEMM execution."""

from dataclasses import replace
import operator

import torch
from torch import fx
from torch.fx.passes.split_module import split_module
from torch.utils import _pytree

from ._autotuner import PIMProblem
from .backend import PIMInductorBackend
from .execution import select_layer_config
from .fx_normalization import normalize_fx_gemms
from .layer import LayerPlan
from .resident import _freeze_graph, _gemm_operands, _tensor_signature


class _LayerGemm(torch.nn.Module):
    def __init__(self, name, gemms, weights, torch_config):
        super().__init__()
        self.name = name
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


def build_layer_plan(exported_program, example_inputs, *, configs, index):
    """Analyze one exported block without compiling it or acquiring hardware.

    configs maps original GEMM names to PIMTorchConfig. Independent activation
    GEMMs of the same shape/config are grouped; weight GEMMs remain stationary
    single-instance stages. aten.bmm uses its explicit operand batch dimension.
    """
    leaves, specification = _pytree.tree_flatten((example_inputs, {}))
    if specification != exported_program.call_spec.in_spec:
        raise ValueError("example_inputs must match the exported positional inputs")
    normalized, mappings = normalize_fx_gemms(exported_program, allow_bmm=True)
    if set(configs) != {mapping.original.node for mapping in mappings}:
        raise ValueError("configs must contain every original GEMM exactly once")
    frozen_graph, frozen, states = _freeze_graph(normalized)
    mappings_by_name = {mapping.normalized.node: mapping for mapping in mappings}
    depths = {}
    groups = {}
    group_for_node = {}
    for node in frozen_graph.graph.nodes:
        depth = max((depths[argument] for argument in node.all_input_nodes), default=0)
        if node.name in mappings_by_name:
            mapping = mappings_by_name[node.name]
            gemm = mapping.normalized
            depth += 1
            operands = _gemm_operands(node)
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
            arguments = tuple(emit(operand) for member in nodes for operand in _gemm_operands(member))
            mapping = mappings_by_name[nodes[0].name]
            operands = _gemm_operands(nodes[0])
            stage = _LayerGemm(
                mapping.original.node,
                tuple(mappings_by_name[member.name].normalized for member in nodes),
                tuple(frozen[operand] if operand in frozen else None for operand in operands),
                configs[mapping.original.node],
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
        index, None, split, None, states, specification, exported_program.call_spec.out_spec,
        tuple(_tensor_signature(value) if isinstance(value, torch.Tensor) else value for value in leaves),
        tuple(isinstance(value, torch.Tensor) for value in leaves),
        stages=tuple(stages), example_inputs=tuple(example_inputs),
    )
