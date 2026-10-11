"""FX block plans for model preparation and the legacy compiled-layer API."""

from dataclasses import dataclass
import time

import torch
from torch import fx
from torch.fx.passes.split_module import split_module
from torch.utils import _pytree

from triton.backends.pim.runtime import (
    create_runtime, current_layer, current_session, resident_lock,
)

from .fx_normalization import normalize_fx_gemms
from .resident import _ResidentGemm, _check_config, _freeze_graph, _gemm_operands, _tensor_signature


class LayerRuntime:
    def __init__(self, compiler):
        self.compiler = compiler
        self.kernels = []
        self.compiled = False

    @property
    def ready(self):
        active = current_layer.get()
        return self.compiled and active is not None and active[1].runtime is self

    def invocation(self, kernel, physical_inputs, *, capture):
        if capture:
            if self.compiled:
                raise RuntimeError("a compiled layer cannot be compiled again")
            return self.compiler.compile_invocation(kernel, physical_inputs)
        if not self.ready:
            raise RuntimeError("layer kernels require an active PIMSession execution")
        session, _, ready = current_layer.get()
        return session.invocation(kernel, physical_inputs, ready)


@dataclass
class LayerPlan:
    index: int
    spec: object
    graph: fx.GraphModule
    runtime: LayerRuntime
    states: tuple
    input_spec: object
    output_spec: object
    inputs: tuple
    tensor_inputs: tuple
    backend_plan: object = None
    compile_ms: float = 0.0
    stages: tuple = ()
    example_inputs: tuple = ()
    weight_bytes_per_unit: int = 0
    exported_graph: fx.GraphModule | None = None

    def plan_model(self, spec, runtime):
        self.spec = spec
        for stage in self.stages:
            stage.prepare(spec, runtime)

    def compile_cpu(self):
        from .model import CompileModelCPU

        leaves, _ = _pytree.tree_flatten((self.example_inputs, {}))
        outputs = CompileModelCPU(self.graph).run(*leaves)
        return _pytree.tree_unflatten(list(outputs), self.output_spec)

    def run_model(self, *inputs):
        leaves, _ = _pytree.tree_flatten((inputs, {}))
        with torch.no_grad():
            outputs = self.graph(*leaves)
        return _pytree.tree_unflatten(list(outputs), self.output_spec)

    def check_state(self):
        if not self.runtime.compiled:
            raise RuntimeError("layer compilation has not finished")
        for state in self.states:
            state.check()

    def run(self, *inputs):
        if not self.runtime.ready:
            raise RuntimeError("use PIMSession.run_layer to execute a layer plan")
        self.check_state()
        leaves, specification = _pytree.tree_flatten((inputs, {}))
        if specification != self.input_spec:
            raise ValueError("layer input structure differs from the exported graph")
        for value, tensor, expected in zip(leaves, self.tensor_inputs, self.inputs, strict=True):
            if tensor:
                if not isinstance(value, torch.Tensor) or _tensor_signature(value) != expected:
                    raise ValueError("layer input shape/stride/dtype/device changed")
            elif type(value) is not type(expected) or value != expected:
                raise ValueError("a specialized non-tensor layer input changed")
        with torch.no_grad():
            outputs = self.graph(*leaves)
        return _pytree.tree_unflatten(list(outputs), self.output_spec)


class CompileCPUPartitions(fx.Interpreter):
    def call_module(self, target, args, kwargs):
        module = self.module.get_submodule(target)
        if isinstance(module, _ResidentGemm):
            matrices = tuple(weight if weight is not None else args[index]
                             for weight, index in zip(module.weights, module.input_indices))
            # Warm CPU partitions using real tensors, never discarded PIM compile outputs.
            return module.operation(*matrices).contiguous()
        compiled = torch.compile(module, backend="inductor", fullgraph=True, dynamic=False)
        self.module.set_submodule(target, compiled)
        return compiled(*args, **kwargs)


def compile_layer(exported_program, example_inputs, *, configs, index):
    """Compile PIM and CPU partitions without allocating units or transferring weights.

    configs maps original FX GEMM names to ResidentGemmConfig. The returned plan
    holds immutable weights and vendor artifacts; per-request resident state is
    created by the selected adapter when LoadLayerWeights executes.
    """
    if current_session.get() is not None:
        raise RuntimeError("compile layers before opening a request session")
    with resident_lock, torch.no_grad():
        start = time.perf_counter_ns()
        leaves, specification = _pytree.tree_flatten((example_inputs, {}))
        if specification != exported_program.call_spec.in_spec:
            raise ValueError("example_inputs must match the exported positional input structure")
        normalized, mappings = normalize_fx_gemms(exported_program)
        if not mappings or set(configs) != {mapping.original.node for mapping in mappings}:
            raise ValueError("configs must contain exactly one entry for every FX GEMM")
        spec = configs[mappings[0].original.node].torch_config.spec
        if any(config.torch_config.spec != spec for config in configs.values()):
            raise ValueError("all layer kernels must have the same target specification")
        compiler = create_runtime(spec)
        graph, frozen, states = _freeze_graph(normalized)
        nodes = {node.name: node for node in graph.graph.nodes}
        gemms = {nodes[mapping.normalized.node]: mapping for mapping in mappings}
        weights = {}
        for node, mapping in gemms.items():
            config = configs[mapping.original.node]
            _check_config(mapping.original.node, mapping.normalized, config,
                          spec.common_profile.num_units)
            compiler.validate_config(config)
            weights[node] = tuple(frozen[argument] if argument in frozen else None
                                  for argument in _gemm_operands(node))
            if tuple(weight is not None for weight in weights[node]) != tuple(
                operand.weight_target is not None
                for operand in (mapping.normalized.left, mapping.normalized.right)
            ):
                raise ValueError(f"{node.name}: frozen operands disagree with FX weight provenance")

        partitions = {}
        gemm_partitions = {}
        partition = 0
        for node in graph.graph.nodes:
            if node.op in ("placeholder", "get_attr", "output"):
                continue
            if node in gemms:
                partition += 1
                partitions[node] = partition
                gemm_partitions[partition] = node
                partition += 1
            else:
                partitions[node] = partition
        split = split_module(graph, graph, partitions.__getitem__,
                             keep_original_order=True, keep_original_node_name=True)
        runtime = LayerRuntime(compiler)
        for name, module in list(split.named_children()):
            partition_id = int(name.removeprefix("submod_"))
            if partition_id in gemm_partitions:
                node = gemm_partitions[partition_id]
                mapping = gemms[node]
                kernel = _ResidentGemm(mapping.original.node, module, mapping.normalized,
                                       weights[node], configs[mapping.original.node], runtime)
                split.add_module(name, kernel)
                runtime.kernels.append(kernel)
                kernel.prepare()
        CompileCPUPartitions(split).run(*leaves)
        plan = LayerPlan(
            index, spec, split, runtime, states, specification, exported_program.call_spec.out_spec,
            tuple(_tensor_signature(value) if isinstance(value, torch.Tensor) else value
                  for value in leaves),
            tuple(isinstance(value, torch.Tensor) for value in leaves),
        )
        plan.backend_plan = compiler.finalize_plan(plan)
        runtime.compiled = True
        plan.check_state()
        plan.compile_ms = (time.perf_counter_ns() - start) / 1e6
        return plan


class LoadLayerWeights(torch.nn.Module):
    _is_impure = True

    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def forward(self, session, previous):
        return session.load_weights(self.plan, previous)


class RunLayer(torch.nn.Module):
    _is_impure = True

    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def forward(self, session, activation, ready):
        return session.run_layer(self.plan, (activation,), ready)
