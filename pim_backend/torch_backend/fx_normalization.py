from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import fx

from .fx_analysis import GemmInfo, analyze_fx_gemms


aten = torch.ops.aten


@dataclass(frozen=True)
class GemmNormalization:
    original: GemmInfo
    normalized: GemmInfo


def _decompose_linear(input, weight, bias=None):
    output = torch.mm(
        input.reshape(math.prod(input.shape[:-1]), input.shape[-1]),
        weight.t(),
    )
    if bias is not None:
        output = output + bias
    return output.reshape(tuple(input.shape[:-1]) + (weight.shape[0],))


def _decompose_addmm(self, mat1, mat2, *, beta=1, alpha=1):
    output = alpha * torch.mm(mat1, mat2)
    if beta == 0:
        return output

    # mm 결과를 첫 operand로 두어 contiguous 출력 배치를 유지한다.
    return output + beta * self


def normalize_fx_gemms(exported_program, *, allow_bmm=False):
    originals = {
        gemm.node: gemm
        for gemm in analyze_fx_gemms(exported_program, allow_bmm=allow_bmm)
    }

    graph = fx.Graph()
    copied = {}

    for node in exported_program.graph_module.graph.nodes:
        copied[node] = graph.node_copy(node, copied.__getitem__)

        if node.name not in originals:
            continue

        # custom 사전도 복사해 원본 node의 metadata를 변경하지 않는다.
        metadata = copied[node].meta
        if "custom" in metadata:
            metadata["custom"] = dict(metadata["custom"])
        else:
            metadata["custom"] = {}
        metadata["custom"]["pim_gemm_source"] = node.name

    graph_module = fx.GraphModule(exported_program.graph_module, graph)
    graph_module.meta = exported_program.graph_module.meta.copy()

    tagged_program = exported_program._update(
        graph_module,
        deepcopy(exported_program.graph_signature),
    )
    normalized_program = tagged_program.run_decompositions({
        aten.linear.default: _decompose_linear,
        aten.addmm.default: _decompose_addmm,
    })

    normalized_nodes = {
        node.name: node
        for node in normalized_program.graph_module.graph.nodes
    }
    mappings = []
    matched_sources = set()

    for gemm in analyze_fx_gemms(normalized_program, allow_bmm=allow_bmm):
        node = normalized_nodes[gemm.node]
        if node.target not in (aten.mm.default, aten._int_mm.default, aten.bmm.default):
            raise ValueError(f"{node.name}: GEMM normalization was incomplete")

        if (
            "custom" not in node.meta
            or "pim_gemm_source" not in node.meta["custom"]
        ):
            raise ValueError(f"{node.name}: original GEMM metadata was lost")

        source_name = node.meta["custom"]["pim_gemm_source"]
        original = originals[source_name]

        if source_name in matched_sources:
            raise ValueError(
                f"{source_name}: expected exactly one normalized GEMM"
            )

        if (
            gemm.mnk != original.mnk
            or gemm.left.dtype != original.left.dtype
            or gemm.right.dtype != original.right.dtype
            or gemm.output_dtype != original.output_dtype
            or gemm.accumulator_dtype != original.accumulator_dtype
            or gemm.instance_count != original.instance_count
        ):
            raise ValueError(
                f"{source_name}: GEMM dimensions or dtype changed"
            )

        if (
            gemm.left.weight_target != original.left.weight_target
            or gemm.right.weight_target != original.right.weight_target
        ):
            raise ValueError(
                f"{source_name}: weight provenance changed"
            )

        mappings.append(GemmNormalization(
            original=original,
            normalized=gemm,
        ))
        matched_sources.add(source_name)

    if matched_sources != set(originals):
        raise ValueError(
            "GEMMs disappeared during normalization: "
            f"{sorted(set(originals) - matched_sources)}"
        )

    return normalized_program, tuple(mappings)
