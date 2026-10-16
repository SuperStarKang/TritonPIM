"""Target-neutral choice of what stays on the device across layers and requests.

Adapters report only numbers: their actual units, the capacity of one unit and
the per-unit bytes each stage needs. The planner never sees vendor handles.

Both policies place layers in device slots (one slot = the buffers and state of
one layer's stages). P-1 gives every layer its own slot; P-2 lets every layer
share the slot of the first layer, so a layer change replaces only the weights.
"""

from dataclasses import dataclass


MODEL = "model"  # P-1: every layer's weights and state stay resident
LAYER = "layer"  # P-2: one layer's slot; every layer writes its weights into it


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
            if policy == LAYER:
                # P-2는 모든 layer가 첫 layer의 slot을 쓰므로 stage별 크기가 같아야 한다.
                layouts = {layer: [(requirement.weight_bytes, requirement.state_bytes,
                                    requirement.input_bytes, requirement.output_bytes)
                                   for requirement in requirements if requirement.layer == layer]
                           for layer in layers}
                for layer in layers[1:]:
                    if layouts[layer] != layouts[layers[0]]:
                        raise ValueError(
                            f"P-2 shares the slot of layer {layers[0]}; layer {layer} stages need "
                            f"{layouts[layer]} bytes, layer {layers[0]} stages {layouts[layers[0]]}")
            return ResidencyPlan(policy, needed, resources.capacity_bytes, resources.units)
    raise ValueError(
        f"no residency policy fits one unit: per-layer replacement needs "
        f"{candidates[-1][1]} bytes, capacity is {resources.capacity_bytes} "
        f"(largest layer weight {max(weights.values())}, state {max(states.values())}, "
        f"scratch {scratch})")


class LayerSlots:
    """Which layer's weights each device slot holds.

    P-1 maps every layer to its own slot, so each layer uploads once. P-2 maps
    every layer to the first layer's slot: buffers and state stay allocated and
    entering another layer uploads only its weights.
    """

    def __init__(self, policy, layers):
        if policy not in (MODEL, LAYER):
            raise ValueError(f"unknown residency policy {policy}")
        first = min(layers)
        self.slot = {layer: layer if policy == MODEL else first for layer in layers}
        self.owner = {}

    def enter(self, layer):
        """True when the layer's weights must be uploaded into its slot."""
        slot = self.slot[layer]
        if self.owner.get(slot) == layer:
            return False
        self.owner[slot] = layer
        return True
