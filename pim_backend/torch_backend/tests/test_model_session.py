"""Model session integration tests; require the configured SDK and eight UPMEM DPUs."""

import pytest
import torch
from triton.backends.pim.residency import LAYER

from torch_backend import PIMSession, PIMTorchConfig, target_spec
from torch_backend._autotuner import PIMConfig
from torch_backend.examples.torch_model_session import HeadBlock


def test_model_owns_resources_and_kv_across_layer_replacements():
    torch.manual_seed(42)
    blocks = [HeadBlock().eval(), HeadBlock().eval()]
    hidden = torch.randint(-3, 4, (5, 32, 32), dtype=torch.int8)
    configs = {
        (160, 32, 32): PIMConfig(16, 16, 32, 1, 1, 1, 5, 1, 1, 2, 2, 1),
        (32, 32, 32): PIMConfig(16, 16, 32, 1, 1, 1, 1, 1, 1, 2, 2, 1),
    }
    with torch.no_grad(), PIMSession(target_spec("upmem", num_units=8, nr_tasklets=16)) as session:
        def configure(gemms, spec):
            assert spec is session.spec
            assert spec.common_profile.num_units == session.backend.num_units
            assert session.stats()["resource_opens"] == 1
            return {gemm.node: PIMTorchConfig(spec=spec, config=configs[gemm.mnk])
                    for gemm in gemms}

        plans = session.prepare(blocks, (hidden,), configure=configure)
        assert plans is session.layers
        assert [plan.index for plan in plans] == [0, 1]
        for plan in plans:
            assert plan.spec is session.spec
            assert [(stage.instance_count, stage.p_b, stage.batch_rounds)
                    for stage in plan.stages] == [(1, 1, 1), (5, 2, 3)]
        torch.testing.assert_close(plans[1].example_inputs[0], blocks[0](hidden), rtol=0, atol=0)
        assert len(session.weight_table) == 2
        prepared = session.stats()
        assert prepared["launches"] == prepared["weight_bytes"] == 0

        session.reserve_kv(0, (1, 2, 8, 4), torch.float32)
        key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        session.append_kv(0, key, key + 10)
        for invocation in range(2):
            if invocation:
                hidden.copy_(torch.randint(-3, 4, hidden.shape, dtype=torch.int8))
            before = session.stats()
            torch.testing.assert_close(session.run_model(hidden), blocks[1](blocks[0](hidden)),
                                       rtol=0, atol=0)
            after = session.stats()
            assert after["launches"] - before["launches"] == 8
            uploads = invocation == 0 or session.residency.policy == LAYER
            assert after["weight_bytes"] - before["weight_bytes"] == (4096 if uploads else 0)
            assert after["device_compiles"] == prepared["device_compiles"]
            stored_key, stored_value = session.kv_view(0)
            torch.testing.assert_close(stored_key, key)
            torch.testing.assert_close(stored_value, key + 10)
            assert session.loaded_layer == 1
    assert session.stats()["resource_opens"] == session.stats()["resource_closes"] == 1


class SplitInputs(torch.nn.Module):
    def forward(self, hidden):
        projected = torch._int_mm(hidden, hidden)
        activation = projected.clamp(-3, 3).to(torch.int8)
        return activation + 1, activation - 1


class MatmulInputs(torch.nn.Module):
    def forward(self, left, right):
        return torch._int_mm(left, right)


def test_preparation_passes_tuple_outputs_to_the_next_layer():
    hidden = torch.arange(1024, dtype=torch.int32).reshape(32, 32)
    hidden = (hidden.remainder(7) - 3).to(torch.int8)
    with torch.no_grad(), PIMSession(target_spec("upmem", num_units=8, nr_tasklets=16)) as session:
        def configure(gemms, spec):
            return {
                gemm.node: PIMTorchConfig(
                    spec=spec, config=PIMConfig(16, 16, 32, 1, 1, 1, 1, 1, 1, 2, 2, 1),
                )
                for gemm in gemms
            }

        plans = session.prepare([SplitInputs(), MatmulInputs()], (hidden,), configure=configure)
        expected_inputs = SplitInputs()(hidden)
        torch.testing.assert_close(plans[1].example_inputs, expected_inputs, rtol=0, atol=0)
        assert session.stats()["launches"] == 0
        torch.testing.assert_close(session.run_model(hidden), torch._int_mm(*expected_inputs),
                                   rtol=0, atol=0)
        assert session.stats()["launches"] == 2
    assert session.stats()["resource_opens"] == session.stats()["resource_closes"] == 1


def test_kv_capacity_overflow_does_not_advance_length():
    with PIMSession(target_spec("upmem", num_units=8, nr_tasklets=16)) as session:
        session.reserve_kv(0, (2, 4), torch.int8)
        with pytest.raises(ValueError, match="capacity"):
            session.append_kv(0, torch.ones(3, 4, dtype=torch.int8),
                               torch.ones(3, 4, dtype=torch.int8))
        assert session.kv_table[0]["length"] == 0
