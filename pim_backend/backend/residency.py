"""Target-neutral memory management table of a compiled model.

Adapters report only numbers: their actual units, the capacity of one unit and
the per-unit bytes each stage needs. The planner never sees vendor handles.

One slot holds the buffers and state of one layer's stages. The plan fits as
many slots as one unit holds: layer i takes slot i mod slots and belongs to
group i // slots. If every layer fits there is one group; otherwise the layers
of one group fill every slot and are uploaded together, overwriting only the
weights of the previous group while buffers and instructions stay in place.
"""

from dataclasses import dataclass


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
class MemoryPlan:
    units: int
    capacity_bytes: int
    # slot 수와 layer group 수. layer i는 slot i mod slots, group i // slots에 놓인다.
    slots: int
    groups: int
    # layer index -> slot
    slot_of_layer: dict
    # layer index -> group. 한 group의 layer들은 함께 올라간다.
    group_of_layer: dict
    # slot마다 weight + state bytes
    slot_bytes: tuple
    scratch_bytes: int
    bytes_per_unit: int


def plan_memory(resources, requirements):
    """Slots and the layer -> slot/group tables that fit one unit.

    Every stage is assumed to occupy the first unit, so stage sums bound the
    most loaded unit.
    """
    layers = sorted({requirement.layer for requirement in requirements})
    layouts = {layer: [(requirement.weight_bytes, requirement.state_bytes,
                        requirement.input_bytes, requirement.output_bytes)
                       for requirement in requirements if requirement.layer == layer]
               for layer in layers}
    layer_bytes = {layer: sum(weight + state for weight, state, _, _ in layouts[layer])
                   for layer in layers}
    scratch = (max(requirement.input_bytes for requirement in requirements)
               + max(requirement.output_bytes for requirement in requirements))
    if sum(layer_bytes.values()) + scratch <= resources.capacity_bytes:
        slot_bytes = tuple(layer_bytes[layer] for layer in layers)
    else:
        # 한 slot에 여러 layer가 차례로 들어가므로 stage별 크기가 모두 같아야 한다.
        for layer in layers[1:]:
            if layouts[layer] != layouts[layers[0]]:
                raise ValueError(
                    f"layers share a slot; layer {layer} stages need {layouts[layer]} bytes, "
                    f"layer {layers[0]} stages {layouts[layers[0]]}")
        slots = (resources.capacity_bytes - scratch) // layer_bytes[layers[0]]
        if slots < 1:
            raise ValueError(
                f"one layer does not fit one unit: it needs {layer_bytes[layers[0]]} bytes "
                f"and scratch {scratch}, capacity is {resources.capacity_bytes}")
        slot_bytes = (layer_bytes[layers[0]],) * slots
    slots = len(slot_bytes)
    slot_of_layer = {layer: position % slots for position, layer in enumerate(layers)}
    group_of_layer = {layer: position // slots for position, layer in enumerate(layers)}
    return MemoryPlan(resources.units, resources.capacity_bytes, slots,
                      max(group_of_layer.values()) + 1, slot_of_layer, group_of_layer,
                      slot_bytes, scratch, sum(slot_bytes) + scratch)


class SlotOwners:
    """Which layer's weights each slot holds now."""

    def __init__(self, plan):
        self.slot_of_layer = plan.slot_of_layer
        self.owner = {}

    def enter(self, layer):
        """True when the layer's weights must be uploaded into its slot."""
        slot = self.slot_of_layer[layer]
        if self.owner.get(slot) == layer:
            return False
        self.owner[slot] = layer
        return True
