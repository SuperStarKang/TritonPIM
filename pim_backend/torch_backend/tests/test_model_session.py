from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from torch_backend import PIMSession, target_spec


class ModelBackend:
    def __init__(self):
        self.num_units = 8
        self.events = []
        self.model = SimpleNamespace(load_weights=self.load_weights, loaded_program=None)

    def open(self):
        self.events.append("alloc")

    def prepare_model(self, plans, session):
        self.events.append(("prepare", session.spec.common_profile.num_units))

    def load_weights(self, plan):
        self.events.append(("weights", plan.index))

    def close(self):
        self.events.append("free")

    def stats(self):
        return {"resource_opens": self.events.count("alloc"),
                "resource_closes": self.events.count("free")}


def test_model_owns_resources_and_kv_across_layer_replacements():
    backend = ModelBackend()
    plans = [SimpleNamespace(index=index, stages=(), run_model=lambda x: x + 1)
             for index in range(2)]
    with patch("triton.backends.pim.runtime.create_runtime", return_value=backend):
        with PIMSession(target_spec("upmem", num_units=16)) as session:
            session.prepare(plans)
            session.reserve_kv(0, (1, 2, 8, 4), torch.float32)
            key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
            session.append_kv(0, key, key + 10)
            assert session.run_model(torch.tensor(1)).item() == 3
            assert session.run_model(torch.tensor(4)).item() == 6
            stored_key, stored_value = session.kv_view(0)
            torch.testing.assert_close(stored_key, key)
            torch.testing.assert_close(stored_value, key + 10)
            assert session.loaded_layer == 1
    assert backend.events == ["alloc", ("prepare", 8),
                              ("weights", 0), ("weights", 1),
                              ("weights", 0), ("weights", 1), "free"]


def test_kv_capacity_overflow_does_not_advance_length():
    backend = ModelBackend()
    with patch("triton.backends.pim.runtime.create_runtime", return_value=backend):
        with PIMSession(target_spec("upmem", num_units=8)) as session:
            session.reserve_kv(0, (2, 4), torch.int8)
            with pytest.raises(ValueError, match="capacity"):
                session.append_kv(0, torch.ones(3, 4, dtype=torch.int8),
                                   torch.ones(3, 4, dtype=torch.int8))
            assert session.kv_table[0]["length"] == 0
