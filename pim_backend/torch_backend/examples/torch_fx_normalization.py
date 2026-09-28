from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from torch_backend.fx_normalization import normalize_fx_gemms


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(16, 16)
        self.k = torch.nn.Linear(16, 16)
        self.v = torch.nn.Linear(16, 16)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.mm(q, k.t()) / 4.0
        return torch.mm(torch.softmax(scores, dim=-1), v)


class EdgeCases(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(16, 8)
        self.register_buffer("tail", torch.randn(8, 4))
        self.register_buffer(
            "integer_weight",
            torch.randint(-3, 4, (32, 32), dtype=torch.int8),
        )

    def forward(self, x, c, c_vector, c_nan, integer_input):
        projected = self.projection(x)
        vector = self.projection(x[0, 0])
        matrix = projected.reshape(6, 8)

        scaled = torch.addmm(
            c, matrix, self.tail, alpha=0.5, beta=2.0,
        )
        broadcast = torch.addmm(
            c_vector, scaled, self.tail.t(), alpha=1.5, beta=0.25,
        )
        ignored_c = torch.addmm(
            c_nan, matrix, self.tail, alpha=0.5, beta=0,
        )
        scores = torch.mm(matrix, matrix.t())
        integers = torch._int_mm(integer_input, self.integer_weight)

        return (
            projected,
            vector,
            scaled,
            broadcast,
            ignored_c,
            scores,
            integers,
        )


def verify(model, inputs):
    with torch.no_grad():
        exported = torch.export.export(model, inputs)
        graph_before = str(exported.graph_module.graph)

        normalized, mappings = normalize_fx_gemms(exported)
        actual = normalized.module()(*inputs)
        expected = model(*inputs)

        torch.testing.assert_close(
            actual, expected, rtol=1e-4, atol=1e-5,
        )

    assert str(exported.graph_module.graph) == graph_before
    assert all(
        node.target not in (
            torch.ops.aten.linear.default,
            torch.ops.aten.addmm.default,
        )
        for node in normalized.graph_module.graph.nodes
        if node.op == "call_function"
    )

    return actual, mappings


def main():
    torch.manual_seed(0)

    attention_output, attention_mappings = verify(
        Attention().eval(),
        (torch.randn(8, 16),),
    )
    assert attention_output.shape == (8, 16)
    assert len(attention_mappings) == 5
    assert [
        mapping.normalized.right.weight_target
        for mapping in attention_mappings
    ] == ["q.weight", "k.weight", "v.weight", None, None]

    for mapping in attention_mappings:
        print(
            f"{mapping.original.op} -> {mapping.normalized.op} "
            f"mnk={mapping.normalized.mnk} "
            f"B_weight={mapping.normalized.right.weight_target}"
        )
    print("attention correctness=PASS")

    edge_model = EdgeCases().eval()
    edge_inputs = (
        torch.randn(3, 2, 16).transpose(0, 1),
        torch.randn(4, 6).t(),
        torch.randn(8),
        torch.full((6, 4), float("nan")),
        torch.randint(-3, 4, (32, 32), dtype=torch.int8),
    )
    edge_output, edge_mappings = verify(edge_model, edge_inputs)

    assert len(edge_mappings) == 7
    assert edge_output[0].shape == (2, 3, 8)
    assert edge_output[1].shape == (8,)
    assert all(output.is_contiguous() for output in edge_output[2:5])
    assert torch.isfinite(edge_output[4]).all()
    torch.testing.assert_close(
        edge_output[-1],
        torch._int_mm(edge_inputs[-1], edge_model.integer_weight),
        rtol=0,
        atol=0,
    )
    print("edge_cases correctness=PASS")


if __name__ == "__main__":
    main()
