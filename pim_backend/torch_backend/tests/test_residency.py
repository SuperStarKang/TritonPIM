"""Common residency policy; pure arithmetic, no device."""

import pytest

from triton.backends.pim.residency import (
    LAYER, MODEL, StageRequirement, UnitResources, plan_residency,
)

GIB = 1 << 30


def _layers(count, weight, state):
    return [StageRequirement(layer, "gemm", weight, state, 0, 0) for layer in range(count)]


def test_small_model_keeps_everything():
    plan = plan_residency(UnitResources(2, 8 * GIB), _layers(4, GIB // 4, GIB // 8))

    assert plan.policy == MODEL
    assert plan.bytes_per_unit == 4 * (GIB // 4 + GIB // 8)


def test_large_weights_replace_layers():
    plan = plan_residency(UnitResources(1, 8 * GIB), _layers(32, 420 * (1 << 20), 900 * (1 << 20)))

    assert plan.policy == LAYER


def test_shared_scratch_counts_once():
    requirements = [StageRequirement(0, "a", 10, 0, 100, 50),
                    StageRequirement(0, "b", 10, 0, 80, 70)]

    plan = plan_residency(UnitResources(1, 1000), requirements)

    assert plan.bytes_per_unit == 20 + 100 + 70


def test_nothing_fits():
    with pytest.raises(ValueError, match="no residency policy fits one unit"):
        plan_residency(UnitResources(1, 100), _layers(2, 80, 40))
