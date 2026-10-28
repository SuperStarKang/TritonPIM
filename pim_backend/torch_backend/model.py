"""FX layer planning and synchronous multi-instance GEMM execution."""

from dataclasses import dataclass, replace
import operator

import torch
from torch import fx
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx.passes.split_module import split_module
from torch.utils import _pytree
from triton.backends.pim.residency import token_rows

from .backend import PIMInductorBackend
from .execution import gemm_problem, select_layer_config
from .fx_normalization import normalize_fx_gemms
from .fx_packing import ConcatenatedWeight, PackedGemm, cache_window, gemm_operands, pack_gemms


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
    # 이 layer가 속한 요청 단계 (prefill, decode). PIMSession이 정한다.
    phase: str = None
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
    def __init__(self, name, gemms, weights, torch_config, sources, origin, kv, kv_windows):
        super().__init__()
        self.name = name
        # 이 stage가 계산하는 원래 GEMM 이름 (packing으로 여러 개일 수 있다)
        self.sources = tuple(sources)
        # packing으로 이어 붙인 GEMM이면 그 축과 member shape (fx_packing.PackedGemm.origin)
        self.origin = origin
        self.gemms = tuple(gemms)
        # operand마다 frozen weight(tensor 또는 view), ConcatenatedWeight(규칙 ②), 또는 None.
        # host에서 복사하지 않고 그대로 둔다.
        self.weights = tuple(weights)
        self.weight_mask = tuple(weight is not None for weight in weights)
        # (operand, KEY/VALUE, transposed): 이 operand는 매 호출 보내지 않고 장치의 key/value cache
        # 영역을 그대로 읽는다. 그 자리의 입력은 host cache [kv_heads, positions, head_dim] 자체다.
        # transposed면 operand가 cache window의 전치다. kv_windows[instance]는 그 instance가 읽는
        # token-row view의 열 범위 (start, stop)다 (residency.token_rows). 아니면 None.
        self.kv = kv
        self.kv_windows = kv_windows
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
        problem = gemm_problem(*self.shape, input_dtype=str(self.input_dtype).removeprefix("torch."),
                               output_dtype=str(self.output_dtype).removeprefix("torch."))
        self.config, self.p_b, self.batch_rounds, self.mnk_rounds = select_layer_config(
            problem, spec, self.torch_config, self.instance_count, has_weights=any(self.weight_mask),
            resident_kv=self.kv is not None,
        )
        m, n, k = self.shape
        if problem.compute_kind == "matmul":
            # GEMM lowering은 연속 operand만 받는다(lowering.make_pim_gemm_lowering). weight view는
            # 연속 사본으로 둔다. GEMV는 view를 그 stride대로 compile하고 업로드 때 그대로 읽는다.
            self.weights = tuple(weight.contiguous() if isinstance(weight, torch.Tensor) else weight
                                 for weight in self.weights)
        # ConcatenatedWeight는 업로드 때 만드는 연속 행렬과 같은 배치로 compile한다.
        physical = tuple(weight if isinstance(weight, torch.Tensor) else torch.empty(shape, dtype=self.input_dtype)
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
        # 실행 중 weight operand 자리의 pointer는 읽히지 않는다 (weight는 장치에 있다).
        self.weight_placeholder = torch.empty((1, 1), dtype=self.input_dtype)
        # weight 업로드와 key/value cache 쓰기는 출력을 쓰지 않는다. instance마다 하나씩 둔다.
        self.preparation_output = torch.empty((self.instance_count, 1, 1), dtype=self.output_dtype)

    def _weight(self, operand):
        weight = self.weights[operand]
        return weight.tensor() if isinstance(weight, ConcatenatedWeight) else weight

    @property
    def preparation_matrices(self):
        """Arguments of a weight upload: the weights, a ConcatenatedWeight built for this upload
        only. Scratch pointers are never read."""
        return (tuple(self._weight(operand) if self.weight_mask[operand]
                      else torch.empty((1, 1), dtype=self.input_dtype) for operand in range(2)),)

    def _reference_operand(self, inputs, instance, operand):
        if self.weight_mask[operand]:
            return self._weight(operand)
        tensor = inputs[2 * instance + operand]
        if self.kv is None or self.kv[0] != operand:
            return tensor.contiguous()
        window = token_rows(tensor, slice(None), *self.kv_windows[instance])
        return (window.t() if self.kv[2] else window).contiguous()

    def reference(self, *inputs):
        if self.batched_operand:
            return torch.bmm(*inputs)
        outputs = tuple(self.operation(self._reference_operand(inputs, instance, 0),
                                       self._reference_operand(inputs, instance, 1))
                        for instance in range(self.instance_count))
        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, *inputs):
        if self.batched_operand:
            matrices = tuple((inputs[0][instance].contiguous(), inputs[1][instance].contiguous())
                             for instance in range(self.instance_count))
        else:
            # cache operand는 장치 영역을 읽으므로 host cache를 그대로 넘긴다 (pointer만 쓰인다).
            matrices = tuple(tuple(self.weight_placeholder if self.weight_mask[operand]
                                   else inputs[2 * instance + operand]
                                   if self.kv is not None and self.kv[0] == operand
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


def build_layer_plan(exported_program, example_inputs, *, configs, index, spec, kv=None, packing=True):
    """Analyze one exported block without compiling it or acquiring hardware.

    configs maps original GEMM names to PIMTorchConfig. GEMMs are first packed on
    the frozen graph (fx_packing, decided by spec's target model) unless their
    config sets pack=False; GEMMs packed together must share one config. Independent activation GEMMs of the same
    shape/config are then grouped; weight GEMMs remain stationary single-instance
    stages. aten.bmm uses its explicit operand batch dimension.
    kv maps KEY/VALUE to the tensors of example_inputs that are this block's host
    key/value cache, [kv_heads, positions, head_dim]; a GEMM operand that is one KV head of
    it, cache[h], or its transpose then reads the device cache region (stage.kv,
    stage.kv_windows); the stage takes the cache input itself in that operand's place.
    packing False skips packing rules (1)-(3) and instance grouping: every GEMM is its own
    stage, except the GEMMs that read the device cache, which stay grouped (rules (4)/(5)).
    """
    leaves, specification = _pytree.tree_flatten((example_inputs, {}))
    if specification != exported_program.call_spec.in_spec:
        raise ValueError("example_inputs must match the exported positional inputs")
    normalized, mappings = normalize_fx_gemms(exported_program, allow_bmm=True)
    if set(configs) != {mapping.original.node for mapping in mappings}:
        raise ValueError("configs must contain every original GEMM exactly once")
    # placeholder 이름 -> KEY/VALUE
    kv_inputs = {}
    if kv is not None:
        user_inputs = [input_spec.arg.name for input_spec in normalized.graph_signature.input_specs
                       if input_spec.kind == InputKind.USER_INPUT]
        if len(user_inputs) != len(leaves):
            raise ValueError(f"{len(leaves)} example input leaves export to {len(user_inputs)} user inputs")
        for kind, tensor in kv.items():
            matches = [name for name, leaf in zip(user_inputs, leaves) if leaf is tensor]
            if len(matches) != 1:
                raise ValueError(f"the {kind} cache must be exactly one example input leaf; found {len(matches)}")
            kv_inputs[matches[0]] = kind
    frozen_graph, frozen = _freeze_graph(normalized)
    gemms = pack_gemms(frozen_graph, frozen, {
        mapping.normalized.node: PackedGemm(mapping, (mapping.original.node,), None) for mapping in mappings
    }, spec, frozenset(name for name, config in configs.items() if not config.pack), frozenset(kv_inputs),
                       packing)
    for entry in gemms.values():
        if len({configs[source] for source in entry.sources}) != 1:
            raise ValueError(f"packed GEMMs {entry.sources} must share one config")

    def kv_source(operand):
        """(KEY/VALUE, transposed, (start, stop)) if the operand reads KV heads of a cache input
        (fx_packing.cache_window), else None."""
        window = cache_window(operand, kv_inputs)
        if window is None:
            return None
        cache, transposed, start, stop = window
        return kv_inputs[cache.name], transposed, (start, stop)
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
            sources = [kv_source(operand) for operand in operands]
            if (any(operand in frozen for operand in operands) or gemm.op == "aten.bmm.default"
                    or not (packing or any(sources))):
                key = (node.name,)
            else:
                # 같은 cache(KEY/VALUE, 전치 여부)를 같은 operand로 읽는 GEMM끼리만 한 stage가 된다.
                # 모양이 같은 QK와 PV(MHA)가 한 stage로 묶이지 않게 한다.
                key = (depth, gemm.op, gemm.mnk, gemm.left.dtype, configs[mapping.original.node],
                       tuple(None if source is None else source[:2] for source in sources))
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
            mapping = gemms[nodes[0].name].mapping
            operands = gemm_operands(nodes[0])
            # operand마다 instance별 (KEY/VALUE, transposed, (start, stop)) 또는 None
            cache_sources = [[kv_source(gemm_operands(member)[operand]) for member in nodes] for operand in range(2)]
            kv_operands = [operand for operand in range(2) if any(cache_sources[operand])]
            if len(kv_operands) > 1 or any(None in cache_sources[operand] for operand in kv_operands) \
                    or any(len({source[:2] for source in cache_sources[operand]}) != 1 for operand in kv_operands):
                raise ValueError(f"{nodes[0].name}: every instance must read one cache kind, transposed or not, "
                                 f"with the same operand")
            if kv_operands and gemms[nodes[0].name].origin is not None:
                raise ValueError(f"{nodes[0].name}: a packed GEMM cannot read the key/value cache")
            # weight operand는 stage가 가지므로 넘기지 않는다(None). cache operand 자리에는 cache 입력을
            # 넘긴다. 규칙 ②의 weight cat과 cache window(head cat 포함)는 host에서 만들지 않는다.
            arguments = tuple(None if operand in frozen
                              else emit(cache_window(operand, kv_inputs)[0]) if index in kv_operands
                              else emit(operand)
                              for member in nodes for index, operand in enumerate(gemm_operands(member)))
            stage = _LayerGemm(
                mapping.original.node,
                tuple(gemms[member.name].mapping.normalized for member in nodes),
                tuple(frozen[operand] if operand in frozen else None for operand in operands),
                configs[mapping.original.node],
                tuple(source for member in nodes for source in gemms[member.name].sources),
                gemms[nodes[0].name].origin,
                (kv_operands[0], *cache_sources[kv_operands[0]][0][:2]) if kv_operands else None,
                tuple(window for _, _, window in cache_sources[kv_operands[0]]) if kv_operands else None,
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
