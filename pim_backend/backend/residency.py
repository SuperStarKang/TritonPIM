"""Target-neutral choice of what stays on the device across layers and requests.

Adapters report only numbers: their actual units, the capacity of one unit and
the per-unit bytes each stage needs. The planner never sees vendor handles.
"""

from dataclasses import dataclass


MODEL = "model"  # P-1: every layer's weights and state stay resident
LAYER = "layer"  # P-2: one layer's weights and state; replaced at every layer


@dataclass(frozen=True)
class UnitResources:
    units: int
    capacity_bytes: int


@dataclass(frozen=True)
class StageRequirement:
    """Per-unit bytes of one stage.

    state_bytes is device-resident memory that is not a weight (descriptor
    chains, private activation buffers). Scratch shared by every stage is
    reported as input/output bytes and counted once at its maximum.
    """

    layer: int
    stage: str
    weight_bytes: int
    state_bytes: int
    input_bytes: int
    output_bytes: int


@dataclass(frozen=True)
class ResidencyPlan:
    policy: str
    bytes_per_unit: int
    capacity_bytes: int
    units: int


def plan_residency(resources, requirements):
    """Pick P-1 (all layers) or P-2 (one layer), whichever fits one unit first.

    Every stage is assumed to occupy the first unit, so stage sums bound the
    most loaded unit.
    """
    layers = sorted({requirement.layer for requirement in requirements})
    weights = {layer: 0 for layer in layers}
    states = {layer: 0 for layer in layers}
    for requirement in requirements:
        weights[requirement.layer] += requirement.weight_bytes
        states[requirement.layer] += requirement.state_bytes
    scratch = (max(requirement.input_bytes for requirement in requirements)
               + max(requirement.output_bytes for requirement in requirements))
    candidates = (
        (MODEL, sum(weights.values()) + sum(states.values()) + scratch),
        (LAYER, max(weights[layer] + states[layer] for layer in layers) + scratch),
    )
    for policy, needed in candidates:
        if needed <= resources.capacity_bytes:
            return ResidencyPlan(policy, needed, resources.capacity_bytes, resources.units)
    raise ValueError(
        f"no residency policy fits one unit: per-layer replacement needs "
        f"{candidates[-1][1]} bytes, capacity is {resources.capacity_bytes} "
        f"(largest layer weight {max(weights.values())}, state {max(states.values())}, "
        f"scratch {scratch})")
