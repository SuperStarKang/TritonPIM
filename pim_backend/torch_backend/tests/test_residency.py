"""Common memory plan; pure arithmetic, no device."""

import pytest

from triton.backends.pim.residency import StageRequirement, UnitResources, plan_memory

GIB = 1 << 30


def _layers(count, weight, state):
    return [StageRequirement(layer, "gemm", weight, state, 0, 0) for layer in range(count)]


def test_small_model_keeps_every_layer():
    plan = plan_memory(UnitResources(2, 8 * GIB), _layers(4, GIB // 4, GIB // 8))

    assert (plan.slots, plan.groups) == (4, 1)
    assert plan.slot_of_layer == {0: 0, 1: 1, 2: 2, 3: 3}
    assert plan.bytes_per_unit == 4 * (GIB // 4 + GIB // 8)


def test_large_model_cycles_layers_through_slots():
    layer = 420 * (1 << 20) + 900 * (1 << 20)
    plan = plan_memory(UnitResources(1, 8 * GIB), _layers(32, 420 * (1 << 20), 900 * (1 << 20)))

    assert plan.slots == 8 * GIB // layer
    assert plan.groups == -(-32 // plan.slots)
    assert [plan.slot_of_layer[index] for index in range(plan.slots + 1)] == (
        list(range(plan.slots)) + [0])
    assert plan.group_of_layer[31] == 31 // plan.slots


def test_shared_scratch_counts_once():
    requirements = [StageRequirement(0, "a", 10, 0, 100, 50),
                    StageRequirement(0, "b", 10, 0, 80, 70)]

    plan = plan_memory(UnitResources(1, 1000), requirements)

    assert plan.bytes_per_unit == 20 + 100 + 70


def test_nothing_fits():
    with pytest.raises(ValueError, match="one layer does not fit one unit"):
        plan_memory(UnitResources(1, 100), _layers(2, 80, 40))


def test_shared_slot_needs_equal_layers():
    requirements = _layers(2, 60, 0) + [StageRequirement(1, "extra", 10, 0, 0, 0)]

    with pytest.raises(ValueError, match="layers share a slot"):
        plan_memory(UnitResources(1, 100), requirements)
