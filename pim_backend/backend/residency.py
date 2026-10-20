"""Target-neutral memory management table of a compiled model.

Adapters report only numbers: their actual units, the capacity of one unit and
the per-unit bytes of every stage operand. The planner never sees vendor handles.

Every unit has the same layout, so one table of offsets from the unit's memory
base serves all units. Weight slots come first: one slot holds the weights and
state of one layer's stages, layer i takes slot i mod slots and belongs to
group i // slots. If every layer fits there is one group; otherwise the layers
of one group fill every slot and are uploaded together, overwriting only the
weights of the previous group while buffers and instructions stay in place.
The input and output scratch follow the slots; every stage uses them from
their start, so each is as large as its largest stage.
"""

from dataclasses import dataclass


WEIGHT = "weight"
# weight가 아닌 장치 상주 memory (descriptor chain, stage 전용 activation buffer)
STATE = "state"
INPUT = "input"
OUTPUT = "output"


@dataclass(frozen=True)
class UnitResources:
    units: int
    capacity_bytes: int


@dataclass(frozen=True)
class OperandRequirement:
    """Per-unit bytes of one stage operand; operand is the adapter's binding index."""

    operand: int
    kind: str
    bytes: int


@dataclass(frozen=True)
class StageRequirement:
    layer: int
    stage: str
    operands: tuple

    def _bytes(self, kind):
        return sum(operand.bytes for operand in self.operands if operand.kind == kind)

    @property
    def weight_bytes(self):
        return self._bytes(WEIGHT)

    @property
    def state_bytes(self):
        return self._bytes(STATE)

    @property
    def input_bytes(self):
        return self._bytes(INPUT)

    @property
    def output_bytes(self):
        return self._bytes(OUTPUT)


@dataclass(frozen=True)
class MemoryEntry:
    """One operand region; offset is from the memory base of every unit."""

    kind: str
    layer: int
    stage: str
    operand: int
    # weight/state는 slot과 group에 속한다. scratch(input/output)는 None이다.
    slot: int | None
    group: int | None
    offset: int
    bytes: int


@dataclass(frozen=True)
class MemoryTable:
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
    # (layer, stage, operand) -> MemoryEntry
    entries: dict


def plan_memory(resources, requirements):
    """The memory table that fits one unit.

    Every stage is assumed to occupy the first unit, so stage sums bound the
    most loaded unit.
    """
    layers = sorted({requirement.layer for requirement in requirements})
    stages = {layer: [requirement for requirement in requirements if requirement.layer == layer]
              for layer in layers}
    layer_bytes = {layer: sum(requirement.weight_bytes + requirement.state_bytes
                              for requirement in stages[layer])
                   for layer in layers}
    input_bytes = max(requirement.input_bytes for requirement in requirements)
    output_bytes = max(requirement.output_bytes for requirement in requirements)
    scratch = input_bytes + output_bytes
    if sum(layer_bytes.values()) + scratch <= resources.capacity_bytes:
        slot_bytes = tuple(layer_bytes[layer] for layer in layers)
    else:
        # 한 slot에 여러 layer가 차례로 들어가므로 operand별 크기가 모두 같아야 한다.
        layouts = {layer: [(operand.kind, operand.bytes) for requirement in stages[layer]
                           for operand in requirement.operands]
                   for layer in layers}
        for layer in layers[1:]:
            if layouts[layer] != layouts[layers[0]]:
                raise ValueError(
                    f"layers share a slot; layer {layer} operands need {layouts[layer]} bytes, "
                    f"layer {layers[0]} operands {layouts[layers[0]]}")
        slots = (resources.capacity_bytes - scratch) // layer_bytes[layers[0]]
        if slots < 1:
            raise ValueError(
                f"one layer does not fit one unit: it needs {layer_bytes[layers[0]]} bytes "
                f"and scratch {scratch}, capacity is {resources.capacity_bytes}")
        slot_bytes = (layer_bytes[layers[0]],) * slots
    slots = len(slot_bytes)
    slot_of_layer = {layer: position % slots for position, layer in enumerate(layers)}
    group_of_layer = {layer: position // slots for position, layer in enumerate(layers)}

    weight_end = sum(slot_bytes)
    entries = {}
    for layer in layers:
        slot = slot_of_layer[layer]
        # slot은 앞 slot들의 영역 바로 뒤에서 시작한다. 같은 slot의 layer는 같은 offset을 받는다.
        offset = sum(slot_bytes[:slot])
        for requirement in stages[layer]:
            input_offset = weight_end
            output_offset = weight_end + input_bytes
            for operand in requirement.operands:
                key = (layer, requirement.stage, operand.operand)
                if operand.kind in (WEIGHT, STATE):
                    entries[key] = MemoryEntry(operand.kind, layer, requirement.stage,
                                               operand.operand, slot, group_of_layer[layer],
                                               offset, operand.bytes)
                    offset += operand.bytes
                elif operand.kind == INPUT:
                    entries[key] = MemoryEntry(INPUT, layer, requirement.stage, operand.operand,
                                               None, None, input_offset, operand.bytes)
                    input_offset += operand.bytes
                else:
                    entries[key] = MemoryEntry(OUTPUT, layer, requirement.stage, operand.operand,
                                               None, None, output_offset, operand.bytes)
                    output_offset += operand.bytes
    return MemoryTable(resources.units, resources.capacity_bytes, slots,
                       max(group_of_layer.values()) + 1, slot_of_layer, group_of_layer,
                       slot_bytes, scratch, weight_end + scratch, entries)
