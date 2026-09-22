"""Explicit FX plans for block-wide UPMEM weight residency."""

from contextlib import ExitStack
from dataclasses import dataclass
import statistics
import time

import torch
from torch import fx
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx.passes.split_module import split_module
from torch.utils import _pytree

from triton.backends.pim.resident import (
    ResidentInvocation, ResidentRuntime, resident_lock,
)

from ._autotuner import ConstraintEngine, PIMProblem
from .backend import PIMInductorBackend
from .config import PIMTorchConfig


@dataclass(frozen=True)
class ResidentGemmConfig:
    """padded_shape is logical (M, N, K), before an optional operand swap."""

    torch_config: PIMTorchConfig
    orientation: str
    padded_shape: tuple[int, int, int]

    def __post_init__(self):
        if self.orientation not in ("normal", "dual"):
            raise ValueError("orientation must be 'normal' or 'dual'")
        if len(self.padded_shape) != 3 or any(type(size) is not int or size < 1
                                             for size in self.padded_shape):
            raise ValueError("padded_shape must contain three positive static integers")
        if self.torch_config.spec is None or self.torch_config.config is None:
            raise ValueError("resident GEMMs require an explicit target spec and PIMConfig")
        if self.torch_config.autotune_cache_path is not None:
            raise ValueError("resident GEMMs take explicit configs, not an autotune cache path")
        if self.torch_config.spec.common_profile.target != "upmem":
            raise ValueError("weight residency currently requires UPMEM")
        if self.torch_config.spec.runtime_config.nr_tasklets < 1:
            raise ValueError("resident GEMMs require an explicit positive nr_tasklets")


def _tensor_signature(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


@dataclass(frozen=True)
class FrozenState:
    table: dict
    name: str
    tensor: torch.Tensor
    version: int
    pointer: int
    signature: tuple

    def check(self):
        current = self.table[self.name]
        if (current is not self.tensor or current._version != self.version
                or current.data_ptr() != self.pointer
                or _tensor_signature(current) != self.signature):
            raise RuntimeError(f"frozen state {self.name!r} changed; close and prepare a new ResidentBlock")


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
    if any(spec.kind != OutputKind.USER_OUTPUT
           for spec in exported_program.graph_signature.output_specs):
        raise ValueError("ResidentBlock requires an inference graph without input or buffer mutation")
    graph = fx.Graph()
    root = torch.nn.Module()
    copied = {}
    values = {}
    frozen = {}
    states = []
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
                table = exported_program.constants
                value = table[spec.target]
            elif spec.kind != InputKind.USER_INPUT:
                raise ValueError(f"unsupported exported input kind: {spec.kind}")
            if value is not None:
                if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                    raise ValueError(f"frozen input {spec.target!r} must be a CPU tensor")
                states.append(FrozenState(table, spec.target, value, value._version,
                                          value.data_ptr(), _tensor_signature(value)))
                value = value.detach()
        elif node.op == "call_function" and node.target in _WEIGHT_VIEWS:
            dependencies = list(node.all_input_nodes)
            if dependencies and all(dependency in values for dependency in dependencies):
                value = node.target(*fx.map_arg(node.args, values.__getitem__),
                                    **fx.map_arg(node.kwargs, values.__getitem__))
        elif node.op == "get_attr":
            raise ValueError("prepare_block expects the lifted graph of a torch.export.ExportedProgram")

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
    return fx.GraphModule(root, graph), frozen, tuple(states)


def _logical_shape(node):
    left, right = (argument.meta["val"] for argument in node.args)
    for tensor in (left, right):
        if tensor.ndim != 2 or tensor.dtype != torch.int8 or tensor.device.type != "cpu":
            raise ValueError(f"{node.name}: resident GEMM requires two CPU int8 matrices")
        if any(type(size) is not int or size < 1 for size in tensor.shape):
            raise ValueError(f"{node.name}: resident GEMM requires positive static dimensions")
    if left.shape[1] != right.shape[0]:
        raise ValueError(f"{node.name}: GEMM reduction dimensions disagree")
    return (left.shape[0], right.shape[1], left.shape[1])


def _check_config(name, logical_shape, config, dpu_count):
    if any(padded < logical for padded, logical in zip(config.padded_shape, logical_shape)):
        raise ValueError(f"{name}: padded_shape cannot shrink the logical GEMM")
    target = config.torch_config.spec
    hierarchy = config.torch_config.config
    if (target.common_profile.num_units != dpu_count
            or hierarchy.p_m * hierarchy.p_n * hierarchy.p_k != dpu_count):
        raise ValueError(f"{name}: every GEMM must use the same full set of {dpu_count} DPUs")
    m, n, k = config.padded_shape
    if config.orientation == "dual":
        m, n = n, m
    reason = ConstraintEngine(PIMProblem(m, n, k, "int8", "int32"), target).check(hierarchy)
    if reason is not None:
        raise ValueError(f"{name}: {reason}")
    lanes = target.runtime_config.nr_tasklets
    if lanes > target.common_profile.lanes_per_core or hierarchy.block_m * hierarchy.group_m % lanes:
        raise ValueError(f"{name}: nr_tasklets must divide the physical Block Group M extent")


class _ResidentGemm(torch.nn.Module):
    def __init__(self, name, partition, logical_shape, weights, config, runtime):
        super().__init__()
        self.name = name
        self.logical_shape = logical_shape
        self.config = config
        self.runtime = runtime
        self.artifact = None
        self.weights = weights
        placeholders = [node for node in partition.graph.nodes if node.op == "placeholder"]
        operation = next(node for node in partition.graph.nodes
                         if node.op == "call_function" and node.target == torch.ops.aten._int_mm.default)
        output = next(node for node in partition.graph.nodes if node.op == "output")
        if output.args[0] is not operation:
            raise ValueError(f"{name}: PIM partition must return its GEMM tensor directly")
        self.input_indices = tuple(
            placeholders.index(argument) if weight is None else None
            for argument, weight in zip(operation.args, weights))
        start = time.perf_counter_ns()
        self.physical_weights = tuple(
            self._layout(index, weight) if weight is not None else None
            for index, weight in enumerate(weights))
        self.weight_layout_ms = (time.perf_counter_ns() - start) / 1e6
        self.weight_mask = tuple(weight is not None for weight in weights)
        if config.orientation == "dual":
            self.weight_mask = self.weight_mask[::-1]
        graph = fx.Graph()
        left = graph.placeholder("left")
        right = graph.placeholder("right")
        graph.output(graph.call_function(torch.ops.aten._int_mm.default, (left, right)))
        self.compiled = torch.compile(
            fx.GraphModule(torch.nn.Module(), graph),
            backend=PIMInductorBackend(config.torch_config), fullgraph=True, dynamic=False)

    def _layout(self, index, tensor):
        m, n, k = self.config.padded_shape
        rows, columns = ((m, k), (k, n))[index]
        if tensor.shape != (rows, columns):
            tensor = torch.nn.functional.pad(
                tensor, (0, columns - tensor.shape[1], 0, rows - tensor.shape[0]))
        if self.config.orientation == "dual":
            tensor = tensor.transpose(0, 1)
        return tensor.contiguous()

    def _physical_inputs(self, inputs):
        physical = tuple(
            self.physical_weights[index] if self.weights[index] is not None
            else self._layout(index, tensor)
            for index, tensor in enumerate(inputs))
        return physical[::-1] if self.config.orientation == "dual" else physical

    def prepare(self):
        start = time.perf_counter_ns()
        m, n, k = self.logical_shape
        inputs = tuple(weight if weight is not None else torch.zeros(shape, dtype=torch.int8)
                       for weight, shape in zip(self.weights, ((m, k), (k, n))))
        physical = self._physical_inputs(inputs)
        self.weight_layout_ms += (time.perf_counter_ns() - start) / 1e6
        with ResidentInvocation(self, physical, capture=True):
            self.compiled(*physical)

    def forward(self, *inputs):
        if not self.runtime.ready:
            raise RuntimeError("ResidentBlock is not ready")
        matrices = tuple(weight if weight is not None else inputs[index]
                         for weight, index in zip(self.weights, self.input_indices))
        m, n, k = self.logical_shape
        for matrix, shape in zip(matrices, ((m, k), (k, n))):
            if matrix.shape != shape or matrix.dtype != torch.int8 or matrix.device.type != "cpu":
                raise ValueError(f"{self.name}: GEMM input differs from its prepared shape/dtype/device")
        physical = self._physical_inputs(matrices)
        with ResidentInvocation(self, physical, capture=False):
            output = self.compiled(*physical)
        if self.config.orientation == "dual":
            output = output.transpose(0, 1)
        return output[:m, :n].contiguous()


class ResidentBlock:
    def __init__(self, graph, runtime, states, input_spec, output_spec, examples,
                 prepare_ms, weight_layout_ms):
        self.graph = graph
        self.runtime = runtime
        self.states = states
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.inputs = tuple(_tensor_signature(value) if isinstance(value, torch.Tensor) else value
                            for value in examples)
        self.tensor_inputs = tuple(isinstance(value, torch.Tensor) for value in examples)
        self.prepare_ms = prepare_ms
        self.weight_prepare_ms = weight_layout_ms + runtime.capture_ms + runtime.upload_ms
        self.last_run_ms = None
        self.last_transfers = None

    @property
    def layout(self):
        return tuple(dict(region) for region in self.runtime.layout)

    def stats(self):
        with resident_lock:
            return self.runtime.stats()

    def run(self, *inputs):
        with resident_lock, torch.no_grad():
            if not self.runtime.ready:
                raise RuntimeError("ResidentBlock is closed or has not finished preparation")
            leaves, specification = _pytree.tree_flatten((inputs, {}))
            if specification != self.input_spec:
                raise ValueError("input structure differs from the exported graph")
            for value, tensor, expected in zip(leaves, self.tensor_inputs, self.inputs):
                if tensor:
                    if not isinstance(value, torch.Tensor) or _tensor_signature(value) != expected:
                        raise ValueError("input shape/stride/dtype/device differs from preparation")
                elif type(value) is not type(expected) or value != expected:
                    raise ValueError("a specialized non-tensor input changed")
            for state in self.states:
                state.check()
            before = self.runtime.stats()
            start = time.perf_counter_ns()
            self.runtime.library.upmem_resident_reset_inputs(self.runtime.handle)
            outputs = self.graph(*leaves)
            self.last_run_ms = (time.perf_counter_ns() - start) / 1e6
            after = self.runtime.stats()
            self.last_transfers = {name: after[name] - before[name] for name in before}
            return _pytree.tree_unflatten(list(outputs), self.output_spec)

    def close(self):
        with resident_lock:
            self.runtime.close()

    def __enter__(self):
        if not self.runtime.ready:
            raise RuntimeError("ResidentBlock is not ready")
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False


def prepare_block(exported_program, example_inputs, *, configs, dpu_count):
    """Prepare immutable int8 weights for explicit plans keyed by FX GEMM name.

    Compilation, host staging capture, ELF linking and weight H2D happen here.
    DPU GEMM execution starts only when the returned block's run() is called.
    """
    with resident_lock, torch.no_grad(), ExitStack() as cleanup:
        start = time.perf_counter_ns()
        leaves, specification = _pytree.tree_flatten((example_inputs, {}))
        if specification != exported_program.call_spec.in_spec:
            raise ValueError("example_inputs must match the exported positional input structure")
        layout_start = time.perf_counter_ns()
        graph, frozen, states = _freeze_graph(exported_program)
        weight_layout_ms = (time.perf_counter_ns() - layout_start) / 1e6
        gemms = [node for node in graph.graph.nodes
                 if node.op == "call_function" and node.target == torch.ops.aten._int_mm.default]
        if not gemms or set(configs) != {node.name for node in gemms}:
            raise ValueError("configs must contain exactly one entry for every exported _int_mm node")
        profile = configs[gemms[0].name].torch_config.spec.common_profile
        if any(config.torch_config.spec.common_profile != profile for config in configs.values()):
            raise ValueError("all resident GEMMs must use the same UPMEM hardware profile")
        shapes = {node: _logical_shape(node) for node in gemms}
        for node in gemms:
            _check_config(node.name, shapes[node], configs[node.name], dpu_count)

        partitions = {}
        gemm_partitions = {}
        partition = 0
        for node in graph.graph.nodes:
            if node.op in ("placeholder", "get_attr", "output"):
                continue
            if node in shapes:
                partition += 1
                partitions[node] = partition
                gemm_partitions[partition] = node
                partition += 1
            else:
                partitions[node] = partition
        split = split_module(graph, graph, partitions.__getitem__,
                             keep_original_order=True, keep_original_node_name=True)
        runtime = ResidentRuntime(dpu_count, profile.bank_capacity_limit_bytes)
        cleanup.callback(runtime.close)
        for name, module in list(split.named_children()):
            partition_id = int(name.removeprefix("submod_"))
            if partition_id in gemm_partitions:
                node = gemm_partitions[partition_id]
                weights = tuple(frozen[argument] if argument in frozen else None for argument in node.args)
                resident = _ResidentGemm(node.name, module, shapes[node], weights,
                                         configs[node.name], runtime)
                split.add_module(name, resident)
                resident.prepare()
                weight_layout_ms += resident.weight_layout_ms
            else:
                split.add_module(name, torch.compile(module, backend="inductor",
                                                       fullgraph=True, dynamic=False))
        for state in states:
            state.check()
        runtime.finalize()
        block = ResidentBlock(split, runtime, states, specification, exported_program.call_spec.out_spec,
                              leaves, (time.perf_counter_ns() - start) / 1e6, weight_layout_ms)
        cleanup.pop_all()
        return block


def benchmark_resident_plans(exported_program, example_inputs, *, plans,
                             dpu_count, repetitions, reuse_count):
    """Measure explicitly supplied complete block plans; any failure propagates."""
    if not plans or repetitions < 1 or reuse_count < 1:
        raise ValueError("plans must be nonempty and repetitions/reuse_count must be positive")
    measurements = []
    with torch.no_grad():
        reference = exported_program.module()(*example_inputs)
    for name, configs in plans.items():
        with prepare_block(exported_program, example_inputs,
                           configs=configs, dpu_count=dpu_count) as block:
            torch.testing.assert_close(block.run(*example_inputs), reference)
            samples = []
            transfers = []
            for _ in range(repetitions):
                block.run(*example_inputs)
                samples.append(block.last_run_ms)
                transfers.append(dict(block.last_transfers))
            run_ms = statistics.median(samples)
            measurements.append({
                "plan": name,
                "prepare_ms": block.prepare_ms,
                "weight_prepare_ms": block.weight_prepare_ms,
                "run_ms": run_ms,
                "score_ms": block.weight_prepare_ms + reuse_count * run_ms,
                "transfers": transfers,
                "layout": block.layout,
            })
    return sorted(measurements, key=lambda measurement: measurement["score_ms"])
