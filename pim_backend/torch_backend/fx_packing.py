"""Packing of independent GEMMs of one layer into larger GEMMs on its frozen FX graph.

Two GEMMs C1 = A1 B1 and C2 = A2 B2 become one GEMM along the axis their shared
operand allows:

  (2) shared A, weight B   x @ [W1 | W2]               -> [C1 | C2]   n axis, no zeros
  (1) shared B             [A1; A2] @ B                -> [C1; C2]    m axis, no zeros
  (3) nothing shared       [A1 | A2] @ blockdiag(B1, B2) -> [C1 | C2]  zeros off the diagonal

(1) and (2) never add computation and always apply. (2) does not build [W1 | W2] on the
host: the packed B is a ConcatenatedWeight of the frozen weights, built only for an upload. (3) packs p GEMMs only when
the target model schedules no more computation than p separate GEMMs, which
holds when the zero blocks fill padding the target computes anyway.

GEMMs whose B reads a key/value cache input take none of these: the cache must stay in
place. A host cache is [kv_heads, positions, head_dim]; B is one KV head of it, cache[h]
([positions, head_dim]), or its transpose. Its window is the columns [h·head_dim,
(h+1)·head_dim) of the token-row view [positions, kv_heads·head_dim]
(residency.token_rows). A stage reading the cache runs all its instances in one round,
one unit or more each, so when a group has more GEMMs than the target has units,
consecutive windows are packed until it fits, the B becoming their union window, the
heads side by side (cat of cache[h] along the columns; the stage reads the device region,
so this B is never built on the host):
  (4) transposed windows (B = window^T, window on B's k): rows of A stacked, each
      row zero outside its own window's columns  -> [A1 0; 0 A2] @ [W1 | W2]^T
  (5) windows on B's n: rows of A stacked, each output row keeps its own window's
      columns (the other columns are cross terms)  -> [A1; A2] @ [W1 | W2]
A target with as many units as GEMMs (UPMEM) packs nothing and keeps one GEMM per
head without zeros; one with fewer (AiMX FPGAs) packs, its zeros and cross terms
being the price of one instance per unit. Query heads sharing a KV head (GQA) read
the same window and simply become more rows of the same packed GEMM.

gemms maps a GEMM node name to a PackedGemm. Packed nodes take a member's mapping
and the members' sources, and record their origin so that tuning can start from
the members' configs. GEMMs whose original name is in separate are not packed.
"""

from dataclasses import dataclass, replace
from functools import cache
import heapq

import torch
from torch import fx
from torch.fx.operator_schemas import normalize_function

from ._autotuner import (
    ConstraintEngine, best_partitions, build_config_space, legal_partitions,
    make_config, pim_hierarchy, rank_block_groups, target_model, tile_ladder,
)
from .execution import gemm_problem
from .fx_normalization import GemmNormalization


aten = torch.ops.aten


@dataclass(frozen=True)
class ConcatenatedWeight:
    """Weights [k, n_i] side by side along the columns, the B of rule (2).

    The parts stay the frozen weights they are; the [k, sum n_i] matrix is built only by
    tensor(), for the time of one weight upload or reference computation.
    """

    parts: tuple

    @property
    def shape(self):
        return torch.Size((self.parts[0].shape[0], sum(part.shape[1] for part in self.parts)))

    @property
    def dtype(self):
        return self.parts[0].dtype

    def tensor(self):
        return torch.cat(self.parts, dim=1)


@dataclass(frozen=True)
class PackedGemm:
    mapping: GemmNormalization
    # 이 GEMM이 계산하는 원래 GEMM 이름
    sources: tuple
    # 이어 붙여 만든 GEMM이면 (축 "m" 또는 "n", member (m, n, k)들, member stage의 instance 수), 아니면 None
    origin: tuple | None
_LEAVES = (fx.Node, int, float, bool, str, type(None), torch.dtype, torch.device, torch.layout,
           torch.memory_format)


def gemm_operands(node):
    arguments = normalize_function(
        node.target, node.args, dict(node.kwargs),
        normalize_to_only_use_kwargs=True,
    )
    if arguments is None:
        raise ValueError(f"{node.name}: cannot resolve GEMM arguments")
    return arguments.kwargs["input"], arguments.kwargs["mat2"]


def _cache_head(node, caches):
    """(cache placeholder, h) if node is cache[h] of a cache input; else None."""
    if (node.op == "call_function" and node.target == aten.select.int
            and node.args[0].op == "placeholder" and node.args[0].name in caches):
        if node.args[1] != 0:
            raise ValueError(f"{node.name}: a cache operand must be one KV head cache[h]")
        return node.args[0], node.args[2]
    return None


def cache_window(node, caches):
    """(cache placeholder, transposed, start, stop) if node reads KV heads [h0, h1) of a cache
    input, as one head cache[h] or consecutive heads side by side (cat along the columns), or
    its transpose; start and stop are the token-row view columns h0·head_dim, h1·head_dim.
    Else None."""
    # 정규화(run_decompositions)는 2차원 전치를 permute [1, 0]으로 바꾼다.
    transposed = (node.op == "call_function"
                  and (node.target == aten.t.default
                       or (node.target == aten.permute.default and list(node.args[1]) == [1, 0])))
    if transposed:
        node = node.args[0]
    if (node.op == "call_function" and node.target == aten.cat.default
            and len(node.args) > 1 and node.args[1] == 1):
        heads = [_cache_head(part, caches) for part in node.args[0]]
    else:
        heads = [_cache_head(node, caches)]
    if None in heads or len({cache for cache, _ in heads}) != 1:
        return None
    cache = heads[0][0]
    indices = [index for _, index in heads]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError(f"{node.name}: packed cache heads {indices} must be consecutive")
    head_dim = cache.meta["val"].shape[2]
    return cache, transposed, indices[0] * head_dim, (indices[-1] + 1) * head_dim


def pack_gemms(graph_module, frozen, gemms, spec, separate, caches=frozenset(), packing=True):
    """caches: names of the placeholders that are key/value cache inputs. packing False applies
    only rules (4)/(5), which the cache stages need."""
    graph = graph_module.graph
    _eliminate_common_subexpressions(graph, gemms)
    if packing:
        _concatenate_weights(graph_module, frozen, gemms, separate)
        _reorder(graph)
        for transform in (_stack_shared_rights, _block_diagonal):
            for members in _instance_groups(graph, frozen, gemms, separate, caches, False):
                transform(graph, gemms, members, spec)
            _reorder(graph)
    for members in _instance_groups(graph, frozen, gemms, separate, caches, True):
        _pack_cache_windows(graph, gemms, members, spec, caches)
    _reorder(graph)
    graph.lint()
    graph_module.recompile()
    return gemms


def _hashable(value):
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    if isinstance(value, _LEAVES):
        return value
    # 비교할 수 없는 인자는 다른 어떤 node와도 같지 않게 둔다.
    return object()


def _eliminate_common_subexpressions(graph, gemms):
    """Merge pure ATen nodes with the same target and inputs; GEMMs stay as they are."""
    seen = {}
    for node in list(graph.nodes):
        if (node.op != "call_function" or node.name in gemms
                or not isinstance(node.target, torch._ops.OpOverload)
                or torch.Tag.nondeterministic_seeded in node.target.tags):
            continue
        key = (node.target, _hashable(node.args), _hashable(node.kwargs))
        if key in seen:
            node.replace_all_uses_with(seen[key])
            graph.erase_node(node)
        else:
            seen[key] = node


def _reorder(graph):
    """Move nodes so every node follows its inputs; otherwise keep the current order."""
    nodes = list(graph.nodes)
    position = {node: index for index, node in enumerate(nodes)}
    output = nodes[-1]
    waiting = {node: len(node.all_input_nodes) for node in nodes}
    ready = [position[node] for node in nodes if not waiting[node] and node is not output]
    heapq.heapify(ready)
    order = []
    while ready:
        node = nodes[heapq.heappop(ready)]
        order.append(node)
        for user in node.users:
            waiting[user] -= 1
            if not waiting[user] and user is not output:
                heapq.heappush(ready, position[user])
    if len(order) != len(nodes) - 1:
        raise RuntimeError("GEMM packing left a cycle in the layer graph")
    for previous, node in zip(order, order[1:] + [output]):
        previous.append(node)


def _replace_outputs(graph, packed, members, dimension, sizes):
    """Slice the packed output along dimension back into each member's output."""
    start = 0
    with graph.inserting_after(packed):
        for member, size in zip(members, sizes):
            piece = graph.call_function(aten.slice.Tensor, (packed, dimension, start, start + size))
            copy = graph.call_function(aten.clone.default, (piece,),
                                       {"memory_format": torch.contiguous_format})
            member.replace_all_uses_with(copy)
            start += size
    for member in members:
        graph.erase_node(member)


def _packed_entry(gemms, members, node, mnk, left, right, weight_target, origin):
    """Entry of a packed GEMM: the first member's mapping with new operands and shape."""
    mapping = gemms[members[0].name].mapping
    info = mapping.normalized
    m, n, k = mnk
    normalized = replace(
        info, node=node.name, mnk=mnk, output_shape=(m, n),
        left=replace(info.left, node=left.name, shape=(m, k), stride=(k, 1)),
        right=replace(info.right, node=right.name, shape=(k, n), stride=(n, 1),
                      weight_target=weight_target),
        reshape_left=False, transpose_right=False,
    )
    sources = tuple(source for member in members for source in gemms[member.name].sources)
    # build_layer_plan은 묶은 stage의 출력 metadata를 읽는다. member와 같은 fake mode로 만든다.
    template = members[0].meta["val"]
    with template.fake_mode:
        node.meta["val"] = torch.empty((m, n), dtype=template.dtype)
    for member in members:
        del gemms[member.name]
    gemms[node.name] = PackedGemm(GemmNormalization(mapping.original, normalized), sources, origin)


def _concatenate_weights(graph_module, frozen, gemms, separate):
    """(2) Weight GEMMs that read the same activation become one GEMM over [W1 | W2 | ...]."""
    graph = graph_module.graph
    groups = {}
    for node in graph.nodes:
        if (node.name not in gemms or node.target == aten.bmm.default
                or separate.intersection(gemms[node.name].sources)):
            continue
        left, right = gemm_operands(node)
        if left in frozen or right not in frozen:
            continue
        info = gemms[node.name].mapping.normalized
        groups.setdefault((left, node.target, info.mnk[2], frozen[right].dtype), []).append(node)
    for (left, target, k, _), members in groups.items():
        if len(members) < 2:
            continue
        rights = [gemm_operands(member)[1] for member in members]
        shapes = tuple(gemms[member.name].mapping.normalized.mnk for member in members)
        sizes = [n for _, n, _ in shapes]
        targets = "+".join(gemms[member.name].mapping.normalized.right.weight_target for member in members)
        # host에 [W1 | W2 | ...]를 만들지 않는다. cat node는 graph 위의 표시이고 실행되지 않는다
        # (model.build_layer_plan은 weight operand를 stage 인자로 넘기지 않는다).
        with graph.inserting_before(members[-1]):
            weight = graph.call_function(aten.cat.default, (rights, 1))
            packed = graph.call_function(target, (left, weight))
        frozen[weight] = ConcatenatedWeight(tuple(frozen[right] for right in rights))
        _packed_entry(gemms, members, packed, (shapes[0][0], sum(sizes), k), left, weight, targets,
                      ("n", shapes, 1))
        _replace_outputs(graph, packed, members, 1, sizes)


def _instance_groups(graph, frozen, gemms, separate, caches, reading):
    """Independent activation GEMMs of the same depth, shape and dtype (build_layer_plan groups),
    those that read a key/value cache if reading, the others if not."""
    depths = {}
    groups = {}
    for node in graph.nodes:
        depth = max((depths[argument] for argument in node.all_input_nodes), default=0)
        if node.name in gemms:
            depth += 1
            left, right = gemm_operands(node)
            reads = cache_window(left, caches) is not None or cache_window(right, caches) is not None
            if (node.target != aten.bmm.default and left not in frozen and right not in frozen
                    and reads == reading and not separate.intersection(gemms[node.name].sources)):
                info = gemms[node.name].mapping.normalized
                groups.setdefault((depth, node.target, info.mnk, info.left.dtype, info.right.dtype),
                                  []).append(node)
        depths[node] = depth
    return [members for members in groups.values() if len(members) > 1]


def _stack_shared_rights(graph, gemms, members, spec):
    """(1) Members that read the same B become one GEMM over the stacked A rows."""
    shared = {}
    for member in members:
        shared.setdefault(gemm_operands(member)[1], []).append(member)
    for right, group in shared.items():
        if len(group) < 2:
            continue
        m, n, k = gemms[group[0].name].mapping.normalized.mnk
        with graph.inserting_before(group[-1]):
            left = graph.call_function(aten.cat.default, ([gemm_operands(member)[0] for member in group], 0))
            packed = graph.call_function(group[0].target, (left, right))
        # 쌓기 전에는 묶음 전체(members)가 instance 하나씩인 stage 하나였다.
        _packed_entry(gemms, group, packed, (len(group) * m, n, k), left, right, None,
                      ("m", ((m, n, k),), len(members)))
        _replace_outputs(graph, packed, group, 0, [m] * len(group))


# 모든 layer가 같은 판정을 하므로 (spec, problem)마다 한 번만 계산한다.
@cache
def _cost(spec, problem):
    """Static packing cost at the most parallel frontier point; no execution seed is selected."""
    # 묶은 GEMM이 실제로 compile될 target(AiMX GEMM batch)으로 잰다.
    spec = target_model(spec).stage_target(problem)
    engine = ConstraintEngine(problem, spec)
    model = target_model(spec)
    space = build_config_space(problem, spec)
    partitions = legal_partitions(problem, spec, space)
    for blocks, groups in rank_block_groups(problem, spec, space)[:space.block_group_top_k]:
        ladder = tile_ladder(problem, spec, blocks, groups, space)
        if not ladder:
            continue
        for partition in best_partitions(problem, spec, blocks, groups, ladder[-1], partitions, engine):
            config = make_config(blocks, groups, ladder[-1], partition)
            tiles = pim_hierarchy(problem, config).useful_execution_tiles
            return (tiles * model.scheduled_compute(problem, config),
                    tiles * sum(model.access_allocations(problem, config)))
    return None


def _block_diagonal(graph, gemms, members, spec):
    """(3) Pack p members as [A1 | ... | Ap] @ blockdiag(B1 ... Bp) when the zeros cost nothing."""
    info = gemms[members[0].name].mapping.normalized
    m, n, k = info.mnk
    costs = {}
    for count in range(1, len(members) + 1):
        if len(members) % count:
            continue
        cost = _cost(spec, gemm_problem(m, count * n, count * k,
                                        input_dtype=str(info.left.dtype).removeprefix("torch."),
                                        output_dtype=str(info.output_dtype).removeprefix("torch.")))
        if cost is not None:
            # 묶음 하나의 비용 × 묶음 수
            costs[count] = (cost[0] * len(members) // count, cost[1] * len(members) // count)
    if 1 not in costs:
        return
    free = [count for count in costs if costs[count][0] <= costs[1][0]]
    count = min(free, key=lambda count: (costs[count][1], -count))
    if count == 1:
        return
    for first in range(0, len(members), count):
        group = members[first:first + count]
        operands = [gemm_operands(member) for member in group]
        with graph.inserting_before(group[-1]):
            left = graph.call_function(aten.cat.default, ([left for left, _ in operands], 1))
            # B_i를 열 i·n 자리에 두고 나머지 열을 0으로 채운 뒤 행 방향으로 쌓는다.
            padded = [graph.call_function(aten.constant_pad_nd.default,
                                          (right, [index * n, (count - 1 - index) * n], 0.0))
                      for index, (_, right) in enumerate(operands)]
            right = graph.call_function(aten.cat.default, (padded, 0))
            packed = graph.call_function(group[0].target, (left, right))
        _packed_entry(gemms, group, packed, (m, count * n, count * k), left, right, None, None)
        _replace_outputs(graph, packed, group, 1, [n] * count)


def _pack_cache_windows(graph, gemms, members, spec, caches):
    """(4)/(5): pack consecutive cache windows until the group has no more GEMMs than units."""
    windows = [cache_window(gemm_operands(member)[1], caches) for member in members]
    # B가 cache window인 GEMM만 묶는다. 왼쪽 operand가 cache면 그대로 둔다.
    if None in windows or len({window[:2] for window in windows}) != 1:
        return
    cache, transposed = windows[0][:2]
    head_dim = cache.meta["val"].shape[2]
    # instance 하나가 unit 하나 이상을 쓰고 한 round에 모두 돈다: unit 수 이하의 가장 큰 약수
    instances = max(count for count in range(1, min(len(members), spec.common_profile.num_units) + 1)
                    if len(members) % count == 0)
    if instances == len(members):
        return
    size = len(members) // instances
    m, n, k = gemms[members[0].name].mapping.normalized.mnk
    order = sorted(range(len(members)), key=lambda index: windows[index][2])
    position = {node: index for index, node in enumerate(graph.nodes)}
    for first in range(0, len(members), size):
        group = [members[index] for index in order[first:first + size]]
        group_windows = [windows[index] for index in order[first:first + size]]
        start = min(window[2] for window in group_windows)
        stop = max(window[3] for window in group_windows)
        lefts = [gemm_operands(member)[0] for member in group]
        with graph.inserting_before(max(group, key=position.get)):
            union = graph.call_function(aten.cat.default, ([
                graph.call_function(aten.select.int, (cache, 0, head))
                for head in range(start // head_dim, stop // head_dim)], 1))
            if transposed:
                # (4) B = union^T: 행마다 자기 window 열 자리에만 A를 두고 나머지는 0이다.
                right = graph.call_function(aten.permute.default, (union, [1, 0]))
                left = graph.call_function(aten.cat.default, ([
                    graph.call_function(aten.constant_pad_nd.default,
                                        (operand, [window[2] - start, stop - window[3]], 0.0))
                    for operand, window in zip(lefts, group_windows)], 0))
                mnk = (size * m, n, stop - start)
            else:
                # (5) B = union: A 행을 쌓고, 결과 행마다 자기 window 열만 쓴다.
                right = union
                left = graph.call_function(aten.cat.default, (lefts, 0))
                mnk = (size * m, stop - start, k)
            packed = graph.call_function(group[0].target, (left, right))
        _packed_entry(gemms, group, packed, mnk, left, right, None, None)
        with graph.inserting_after(packed):
            for index, (member, window) in enumerate(zip(group, group_windows)):
                piece = graph.call_function(aten.slice.Tensor, (packed, 0, index * m, (index + 1) * m))
                if not transposed:
                    piece = graph.call_function(aten.slice.Tensor,
                                                (piece, 1, window[2] - start, window[2] - start + n))
                member.replace_all_uses_with(graph.call_function(
                    aten.clone.default, (piece,), {"memory_format": torch.contiguous_format}))
        for member in group:
            graph.erase_node(member)
