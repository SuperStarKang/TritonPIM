from dataclasses import asdict, replace
import json

import pytest
import torch
from torch.utils import _pytree

from torch_backend import PIMTorchConfig, build_layer_plan, target_spec
from torch_backend._autotuner import PIMConfig, PIMConfigCache, PIMProblem
from torch_backend.execution import select_layer_config
from torch_backend.fx_analysis import analyze_fx_gemms
from torch_backend.model import _ReferenceLayer
from torch_backend.template import TORCH_GEMM_TEMPLATE_VERSION


SMALL = PIMConfig(16, 16, 32, 1, 1, 1, 8, 8, 4, 8, 8, 1)
LARGE = PIMConfig(16, 16, 32, 1, 1, 1, 16, 16, 4, 4, 4, 1)
QK = PIMProblem(1024, 1024, 128, input_dtype="int8", output_dtype="int32")


@pytest.mark.parametrize("units,config,heads,p_b,rounds", [
    (2048, SMALL, 32, 32, 1),
    (512, SMALL, 32, 8, 4),
    (512, LARGE, 32, 32, 1),
    (2048, SMALL, 33, 32, 2),
    (513, SMALL, 32, 8, 4),
])
def test_resource_assignment(units, config, heads, p_b, rounds):
    selected = select_layer_config(
        QK, target_spec("upmem", num_units=units, nr_tasklets=16),
        PIMTorchConfig(config=config), heads, has_weights=False,
    )
    assert selected == (config, p_b, rounds, (1, 1, 1))


def test_mnk_rounds_remain_separate_from_batch_rounds():
    config = replace(SMALL, p_m=4, p_n=4, tile_k=2)
    selected = select_layer_config(
        QK, target_spec("upmem", num_units=32, nr_tasklets=16),
        PIMTorchConfig(config=config), 5, has_weights=False,
    )
    assert selected[1:] == (2, 3, (2, 2, 2))


def test_config_larger_than_allocated_resources_fails():
    with pytest.raises(ValueError, match="no model config"):
        select_layer_config(QK, target_spec("upmem", num_units=32, nr_tasklets=16),
                            PIMTorchConfig(config=SMALL), 32, has_weights=False)


def test_cached_cost_selects_tile_and_batch_together(tmp_path):
    spec = target_spec("upmem", num_units=512, nr_tasklets=16)
    path = tmp_path / "configs.json"
    cache = PIMConfigCache(str(path), template_version=TORCH_GEMM_TEMPLATE_VERSION)
    path.write_text(json.dumps({cache._key(QK, spec): {"records": [
        {"config": asdict(SMALL), "total_ms": 2.0, "correctness": True, "timings": []},
        {"config": asdict(LARGE), "total_ms": 5.0, "correctness": True, "timings": []},
    ]}}))
    selected = select_layer_config(QK, spec, PIMTorchConfig(autotune_cache_path=str(path)),
                                   32, has_weights=False)
    assert selected[:3] == (LARGE, 32, 1)


class Heads(torch.nn.Module):
    def forward(self, query, key):
        return torch.stack([torch._int_mm(query[h], key[h].t().contiguous())
                            for h in range(3)])


class DependentHeads(torch.nn.Module):
    def forward(self, query, key, value):
        outputs = []
        for head in range(3):
            scores = torch._int_mm(query[head], key[head].t().contiguous())
            activation = scores.clamp(-8, 8).to(torch.int8)
            outputs.append(torch._int_mm(activation, value[head]))
        return torch.stack(outputs)


class FloatBMM(torch.nn.Module):
    def forward(self, left, right):
        return torch.bmm(left, right).relu()


class WeightedHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.randint(-2, 3, (32, 32), dtype=torch.int8))

    def forward(self, hidden):
        projected = torch._int_mm(hidden.reshape(48, 32), self.weight)
        heads = projected.clamp(-3, 3).to(torch.int8).reshape(3, 16, 32)
        return torch.stack([torch._int_mm(heads[h], heads[h].t().contiguous())
                            for h in range(3)])


def make_plan(module, inputs):
    exported = torch.export.export(module, inputs)
    configs = {gemm.node: PIMTorchConfig(config=SMALL)
               for gemm in analyze_fx_gemms(exported, allow_bmm=True)}
    return build_layer_plan(exported, inputs, configs=configs, index=0)


def reference_plan(plan, inputs):
    leaves, _ = _pytree.tree_flatten((inputs, {}))
    outputs = _ReferenceLayer(plan.graph).run(*leaves)
    return _pytree.tree_unflatten(list(outputs), plan.output_spec)


def test_independent_int8_heads_form_one_stage_without_compilation():
    inputs = tuple(torch.randint(-3, 4, (3, 16, 32), dtype=torch.int8) for _ in range(2))
    plan = make_plan(Heads(), inputs)
    assert len(plan.stages) == 1
    assert plan.stages[0].instance_count == 3
    torch.testing.assert_close(reference_plan(plan, inputs), Heads()(*inputs), rtol=0, atol=0)


def test_qk_and_pv_are_separate_batches_with_dependencies_preserved():
    inputs = (torch.randint(-3, 4, (3, 16, 32), dtype=torch.int8),
              torch.randint(-3, 4, (3, 16, 32), dtype=torch.int8),
              torch.randint(-3, 4, (3, 16, 32), dtype=torch.int8))
    plan = make_plan(DependentHeads(), inputs)
    assert [stage.instance_count for stage in plan.stages] == [3, 3]
    assert [stage.shape for stage in plan.stages] == [(16, 16, 32), (16, 32, 16)]
    torch.testing.assert_close(reference_plan(plan, inputs), DependentHeads()(*inputs), rtol=0, atol=0)


def test_explicit_bmm_keeps_transposed_operand_semantics():
    inputs = (torch.randn(3, 16, 32), torch.randn(3, 16, 32).transpose(1, 2))
    plan = make_plan(FloatBMM(), inputs)
    assert plan.stages[0].instance_count == 3
    torch.testing.assert_close(reference_plan(plan, inputs), FloatBMM()(*inputs))


def test_resident_weight_stages_require_one_scheduling_round():
    with pytest.raises(ValueError, match="resident weights"):
        select_layer_config(QK, target_spec("upmem", num_units=512, nr_tasklets=16),
                            PIMTorchConfig(config=replace(SMALL, p_m=4)), 1,
                            has_weights=True)


def test_weight_projection_precedes_independent_activation_heads():
    block = WeightedHeads()
    inputs = (torch.randint(-2, 3, (3, 16, 32), dtype=torch.int8),)
    plan = make_plan(block, inputs)
    assert [stage.instance_count for stage in plan.stages] == [1, 3]
    assert [stage.weight_mask for stage in plan.stages] == [(False, True), (False, False)]
    torch.testing.assert_close(reference_plan(plan, inputs), block(*inputs), rtol=0, atol=0)
